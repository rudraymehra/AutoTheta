"""SQLite trade journal with partial exits and daily summary support.

Tables:
- trades: All trade entries across strategies
- trade_exits: Partial and full exits (supports RSI 40/50 split exits)
- daily_summary: End-of-day aggregated stats per strategy
"""

import logging
import sqlite3
import uuid
from datetime import date, datetime
from pathlib import Path

log = logging.getLogger("autotheta.journal")


class TradeJournal:
    """Unified trade journal supporting both strategies."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                strategy_name TEXT NOT NULL,
                symbol TEXT NOT NULL,
                token TEXT,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                entry_time TEXT NOT NULL,
                status TEXT DEFAULT 'open',
                stop_loss REAL,
                realized_pnl REAL DEFAULT 0.0,
                total_fees REAL DEFAULT 0.0,
                entry_indicators TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS trade_exits (
                exit_id TEXT PRIMARY KEY,
                trade_id TEXT NOT NULL REFERENCES trades(trade_id),
                exit_price REAL NOT NULL,
                exit_quantity INTEGER NOT NULL,
                exit_time TEXT NOT NULL,
                exit_reason TEXT,
                realized_pnl REAL NOT NULL,
                fees REAL DEFAULT 0.0,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS daily_summary (
                date TEXT NOT NULL,
                strategy_name TEXT NOT NULL,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                losing_trades INTEGER DEFAULT 0,
                net_pnl REAL DEFAULT 0.0,
                total_fees REAL DEFAULT 0.0,
                max_drawdown REAL DEFAULT 0.0,
                UNIQUE(date, strategy_name)
            );

            CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy_name);
            CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
            CREATE INDEX IF NOT EXISTS idx_exits_trade ON trade_exits(trade_id);
            CREATE INDEX IF NOT EXISTS idx_summary_date ON daily_summary(date);
        """)
        self.conn.commit()

    @staticmethod
    def generate_trade_id(prefix: str = "T") -> str:
        return f"{prefix}-{uuid.uuid4().hex[:8].upper()}"

    def record_entry(self, trade_id: str, strategy_name: str, symbol: str,
                     token: str, side: str, quantity: int, price: float,
                     stop_loss: float = 0.0, fees: float = 0.0,
                     indicators: str = "{}") -> str:
        self.conn.execute(
            "INSERT INTO trades (trade_id, strategy_name, symbol, token, side, "
            "quantity, entry_price, entry_time, stop_loss, total_fees, entry_indicators) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (trade_id, strategy_name, symbol, token, side, quantity, price,
             datetime.now().isoformat(), stop_loss, fees, indicators),
        )
        self.conn.commit()
        log.info("ENTRY | %s | %s %s %s x%d @ ₹%.2f | SL=₹%.2f",
                 strategy_name, side, symbol, token, quantity, price, stop_loss)
        return trade_id

    def record_exit(self, trade_id: str, exit_price: float, exit_quantity: int,
                    exit_reason: str, fees: float = 0.0) -> float:
        """Record a partial or full exit. Returns realized P&L for this exit."""
        row = self.conn.execute(
            "SELECT entry_price, quantity, side, status FROM trades WHERE trade_id=?",
            (trade_id,),
        ).fetchone()
        if not row:
            log.warning("Trade %s not found for exit", trade_id)
            return 0.0

        entry_price, total_qty, side = row["entry_price"], row["quantity"], row["side"]

        if side == "SELL":
            pnl = (entry_price - exit_price) * exit_quantity
        else:
            pnl = (exit_price - entry_price) * exit_quantity
        pnl -= fees

        exit_id = f"EX-{uuid.uuid4().hex[:8].upper()}"
        self.conn.execute(
            "INSERT INTO trade_exits (exit_id, trade_id, exit_price, exit_quantity, "
            "exit_time, exit_reason, realized_pnl, fees) VALUES (?,?,?,?,?,?,?,?)",
            (exit_id, trade_id, exit_price, exit_quantity,
             datetime.now().isoformat(), exit_reason, pnl, fees),
        )

        # Update trade status
        total_exited = self.conn.execute(
            "SELECT COALESCE(SUM(exit_quantity), 0) FROM trade_exits WHERE trade_id=?",
            (trade_id,),
        ).fetchone()[0]

        new_status = "closed" if total_exited >= total_qty else "partial_exit"
        total_pnl = self.conn.execute(
            "SELECT COALESCE(SUM(realized_pnl), 0) FROM trade_exits WHERE trade_id=?",
            (trade_id,),
        ).fetchone()[0]
        total_exit_fees = self.conn.execute(
            "SELECT COALESCE(SUM(fees), 0) FROM trade_exits WHERE trade_id=?",
            (trade_id,),
        ).fetchone()[0]

        self.conn.execute(
            "UPDATE trades SET status=?, realized_pnl=?, total_fees=total_fees+? WHERE trade_id=?",
            (new_status, total_pnl, fees, trade_id),
        )
        self.conn.commit()

        log.info("EXIT | %s | %s x%d @ ₹%.2f | reason=%s | P&L=₹%.2f",
                 trade_id, row["symbol"] if hasattr(row, "keys") else "", exit_quantity,
                 exit_price, exit_reason, pnl)
        return pnl

    def update_daily_summary(self, strategy_name: str):
        """Aggregate today's trades into the daily summary."""
        today = date.today().isoformat()
        stats = self.conn.execute(
            "SELECT COUNT(*) as total, "
            "SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins, "
            "SUM(CASE WHEN realized_pnl <= 0 THEN 1 ELSE 0 END) as losses, "
            "COALESCE(SUM(realized_pnl), 0) as net_pnl, "
            "COALESCE(SUM(total_fees), 0) as total_fees "
            "FROM trades WHERE strategy_name=? AND status='closed' "
            "AND DATE(entry_time)=?",
            (strategy_name, today),
        ).fetchone()

        self.conn.execute(
            "INSERT OR REPLACE INTO daily_summary "
            "(date, strategy_name, total_trades, winning_trades, losing_trades, net_pnl, total_fees) "
            "VALUES (?,?,?,?,?,?,?)",
            (today, strategy_name, stats["total"], stats["wins"] or 0,
             stats["losses"] or 0, stats["net_pnl"], stats["total_fees"]),
        )
        self.conn.commit()

    def get_performance(self, strategy_name: str, days: int = 30) -> dict:
        """Get performance metrics for a strategy over the last N days."""
        row = self.conn.execute(
            "SELECT COUNT(*) as total_trades, "
            "COALESCE(SUM(winning_trades), 0) as wins, "
            "COALESCE(SUM(losing_trades), 0) as losses, "
            "COALESCE(SUM(net_pnl), 0) as net_pnl, "
            "COALESCE(SUM(total_fees), 0) as fees "
            "FROM daily_summary WHERE strategy_name=? "
            "AND date >= DATE('now', ?)",
            (strategy_name, f"-{days} days"),
        ).fetchone()
        total = (row["wins"] or 0) + (row["losses"] or 0)
        return {
            "total_trades": row["total_trades"] or 0,
            "wins": row["wins"] or 0,
            "losses": row["losses"] or 0,
            "win_rate": (row["wins"] or 0) / total * 100 if total else 0,
            "net_pnl": row["net_pnl"] or 0,
            "total_fees": row["fees"] or 0,
        }

    def close(self):
        self.conn.close()
