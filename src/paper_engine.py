import logging
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime

from config.settings import DB_PATH, INITIAL_CAPITAL, MAX_LOSS_PER_DAY, SLIPPAGE_PCT
from src.fees import calculate_fees

log = logging.getLogger("autotheta.paper")


@dataclass
class Position:
    trade_id: str
    symbol: str
    strike: float
    option_type: str  # CE or PE
    side: str  # BUY or SELL
    quantity: int
    entry_price: float
    entry_fees: float


class TradeJournal:
    """SQLite-backed trade journal for paper and live trades."""

    def __init__(self, db_path=None):
        self.db_path = str(db_path or DB_PATH)
        self.conn = sqlite3.connect(self.db_path)
        self._init_db()

    def _init_db(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE NOT NULL,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                strike REAL NOT NULL,
                expiry TEXT NOT NULL,
                option_type TEXT CHECK(option_type IN ('CE','PE')),
                side TEXT CHECK(side IN ('BUY','SELL')),
                quantity INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                exit_timestamp TEXT,
                status TEXT DEFAULT 'OPEN',
                pnl REAL,
                fees_entry REAL DEFAULT 0,
                fees_exit REAL DEFAULT 0,
                strategy_name TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades(timestamp);
            CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
        """)
        self.conn.commit()

    def record_entry(self, trade_id, symbol, strike, expiry, option_type,
                     side, quantity, price, fees, strategy):
        self.conn.execute(
            "INSERT INTO trades (trade_id, timestamp, symbol, strike, expiry, "
            "option_type, side, quantity, entry_price, fees_entry, strategy_name) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (trade_id, datetime.now().isoformat(), symbol, strike, str(expiry),
             option_type, side, quantity, price, fees, strategy),
        )
        self.conn.commit()

    def record_exit(self, trade_id, exit_price, fees) -> float | None:
        row = self.conn.execute(
            "SELECT entry_price, quantity, side FROM trades WHERE trade_id=?",
            (trade_id,),
        ).fetchone()
        if not row:
            return None
        entry_price, qty, side = row
        if side == "SELL":
            pnl = (entry_price - exit_price) * qty
        else:
            pnl = (exit_price - entry_price) * qty
        self.conn.execute(
            "UPDATE trades SET exit_price=?, exit_timestamp=?, fees_exit=?, pnl=?, status='CLOSED' "
            "WHERE trade_id=?",
            (exit_price, datetime.now().isoformat(), fees, pnl, trade_id),
        )
        self.conn.commit()
        return pnl

    def get_open_trades(self) -> list[dict]:
        cur = self.conn.execute(
            "SELECT trade_id, symbol, strike, option_type, side, quantity, entry_price "
            "FROM trades WHERE status='OPEN'"
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_summary(self, days: int = 30) -> dict:
        """Get P&L summary for the last N days."""
        cur = self.conn.execute(
            "SELECT COUNT(*) as total, "
            "SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins, "
            "SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losses, "
            "SUM(pnl) as total_pnl, "
            "SUM(fees_entry + fees_exit) as total_fees, "
            "MIN(pnl) as worst_trade, "
            "MAX(pnl) as best_trade "
            "FROM trades WHERE status='CLOSED' "
            "AND timestamp >= datetime('now', ?)",
            (f"-{days} days",),
        )
        row = cur.fetchone()
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))

    def close(self):
        self.conn.close()


class PaperTradingEngine:
    """Simulates trade execution with realistic slippage and fees."""

    def __init__(self, capital=None, slippage_pct=None, max_daily_loss=None):
        self.capital = capital or INITIAL_CAPITAL
        self.slippage_pct = slippage_pct or SLIPPAGE_PCT
        self.max_daily_loss = max_daily_loss or MAX_LOSS_PER_DAY
        self.positions: dict[str, Position] = {}
        self.daily_pnl = 0.0
        self.journal = TradeJournal()

    def _slipped_price(self, price: float, side: str) -> float:
        """Simulate slippage: worse fill for the trader."""
        if side == "BUY":
            return round(price * (1 + self.slippage_pct / 100), 2)
        else:
            return round(price * (1 - self.slippage_pct / 100), 2)

    def open_position(
        self, symbol: str, strike: float, expiry: str,
        option_type: str, side: str, quantity: int, market_price: float,
    ) -> str | None:
        """Open a paper position. Returns trade_id or None if blocked by risk."""
        if self.daily_pnl <= -self.max_daily_loss:
            log.warning("Daily loss cap hit (₹%.0f). Cannot open new positions.", self.daily_pnl)
            return None

        fill = self._slipped_price(market_price, side)
        fees = calculate_fees(fill, quantity, side)
        trade_id = f"PT-{uuid.uuid4().hex[:8].upper()}"

        self.positions[trade_id] = Position(
            trade_id=trade_id, symbol=symbol, strike=strike,
            option_type=option_type, side=side, quantity=quantity,
            entry_price=fill, entry_fees=fees["total"],
        )
        self.capital -= fees["total"]
        self.journal.record_entry(
            trade_id, symbol, strike, expiry, option_type,
            side, quantity, fill, fees["total"], "otm_skew_iron_condor",
        )
        log.info("PAPER %s %s %s @ ₹%.2f (slipped from ₹%.2f) | fees: ₹%.2f | id: %s",
                 side, option_type, symbol, fill, market_price, fees["total"], trade_id)
        return trade_id

    def close_position(self, trade_id: str, market_price: float) -> float | None:
        """Close a paper position. Returns net P&L or None."""
        pos = self.positions.pop(trade_id, None)
        if not pos:
            log.warning("Position %s not found", trade_id)
            return None

        exit_side = "BUY" if pos.side == "SELL" else "SELL"
        fill = self._slipped_price(market_price, exit_side)
        fees = calculate_fees(fill, pos.quantity, exit_side)

        if pos.side == "SELL":
            gross = (pos.entry_price - fill) * pos.quantity
        else:
            gross = (fill - pos.entry_price) * pos.quantity

        net = gross - pos.entry_fees - fees["total"]
        self.capital += net
        self.daily_pnl += net
        self.journal.record_exit(trade_id, fill, fees["total"])

        log.info("PAPER CLOSE %s @ ₹%.2f | gross: ₹%.2f | net: ₹%.2f | id: %s",
                 pos.symbol, fill, gross, net, trade_id)
        return net

    def close_all(self, price_getter) -> float:
        """Close all open positions. price_getter(symbol) -> current price.

        Returns total P&L from closing all positions.
        """
        total = 0.0
        for trade_id in list(self.positions.keys()):
            pos = self.positions[trade_id]
            price = price_getter(pos.symbol)
            pnl = self.close_position(trade_id, price)
            if pnl is not None:
                total += pnl
        return total
