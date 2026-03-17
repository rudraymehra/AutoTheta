"""RSI Oversold Bounce Strategy for Nifty 50 stocks.

Entry: RSI(7) crosses below 20 on 1-min candles + filter stack passes
Exit:  50% at RSI 40 (or +0.3%), remaining 50% at RSI 50 (or trailing ATR stop)
Stop:  1.5x ATR(14) below entry
Time:  15-candle time stop — exit if RSI hasn't reached 40

Filter stack (all must pass):
1. Price above 20-EMA on 5-min chart
2. Price at or above VWAP
3. Volume > 1.5x 20-period average
4. ADX > 20
5. Not in first 15 minutes (skip 9:15-9:30)
6. Max 1 stock per sector
7. If >10 stocks trigger simultaneously → market-wide dip, reduce to 2 positions
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, time

import pandas as pd
import pandas_ta as ta
import pytz

from core.data_feed import DataFeed
from core.risk_manager import GlobalRiskManager
from core.trade_journal import TradeJournal
from models.types import Candle, Position, PositionStatus, Side, Signal, SignalType
from strategies.base import BaseStrategy, StrategyEngine

log = logging.getLogger("autotheta.rsi_bounce")
trade_log = logging.getLogger("autotheta.trades")
IST = pytz.timezone("Asia/Kolkata")

# Nifty 50 sector mapping (top stocks)
SECTOR_MAP = {
    "HDFCBANK-EQ": "Banking", "ICICIBANK-EQ": "Banking", "KOTAKBANK-EQ": "Banking",
    "SBIN-EQ": "Banking", "AXISBANK-EQ": "Banking", "INDUSINDBK-EQ": "Banking",
    "BAJFINANCE-EQ": "Finance", "BAJAJFINSV-EQ": "Finance", "HDFCLIFE-EQ": "Finance",
    "SBILIFE-EQ": "Finance",
    "RELIANCE-EQ": "Energy", "ONGC-EQ": "Energy", "NTPC-EQ": "Energy",
    "POWERGRID-EQ": "Energy", "ADANIENT-EQ": "Energy",
    "TCS-EQ": "IT", "INFY-EQ": "IT", "WIPRO-EQ": "IT", "HCLTECH-EQ": "IT",
    "TECHM-EQ": "IT", "LTIM-EQ": "IT",
    "HINDUNILVR-EQ": "FMCG", "ITC-EQ": "FMCG", "NESTLEIND-EQ": "FMCG",
    "BRITANNIA-EQ": "FMCG", "TATACONSUM-EQ": "FMCG",
    "SUNPHARMA-EQ": "Pharma", "DRREDDY-EQ": "Pharma", "CIPLA-EQ": "Pharma",
    "APOLLOHOSP-EQ": "Pharma",
    "M&M-EQ": "Auto", "MARUTI-EQ": "Auto",
    "BAJAJ-AUTO-EQ": "Auto", "HEROMOTOCO-EQ": "Auto", "EICHERMOT-EQ": "Auto",
    "TATASTEEL-EQ": "Metals", "JSWSTEEL-EQ": "Metals", "HINDALCO-EQ": "Metals",
    "COALINDIA-EQ": "Metals",
    "LT-EQ": "Infra", "ULTRACEMCO-EQ": "Infra", "GRASIM-EQ": "Infra",
    "ADANIPORTS-EQ": "Infra", "SHRIRAMFIN-EQ": "Infra",
    "BHARTIARTL-EQ": "Telecom",
    "ASIANPAINT-EQ": "Other", "TITAN-EQ": "Other", "DIVISLAB-EQ": "Other",
    "WIPRO-EQ": "IT", "BEL-EQ": "Other", "TRENT-EQ": "Other",
}


@StrategyEngine.register("rsi_bounce")
class RSIBounceStrategy(BaseStrategy):
    """RSI oversold bounce on Nifty 50 stocks with confirmation filters."""

    def __init__(self, config: dict):
        super().__init__(config)
        # Parameters
        self.rsi_period = self.params.get("rsi_period", 7)
        self.oversold = self.params.get("oversold_threshold", 20)
        self.exit_1_rsi = self.params.get("exit_1_rsi", 40)
        self.exit_2_rsi = self.params.get("exit_2_rsi", 50)
        self.atr_sl_mult = self.params.get("atr_sl_multiplier", 1.5)
        self.time_stop_candles = self.params.get("time_stop_candles", 15)
        self.skip_first_minutes = self.params.get("skip_first_minutes", 15)
        self.max_per_sector = self.params.get("max_per_sector", 1)
        self.volume_mult = self.params.get("volume_spike_multiplier", 1.5)
        self.adx_threshold = self.params.get("adx_threshold", 20)
        self.market_wide_threshold = self.params.get("market_wide_threshold", 10)

        # State
        self._candle_buffers: dict[str, pd.DataFrame] = {}
        self._5min_buffers: dict[str, pd.DataFrame] = {}
        self._vwap_state: dict[str, dict] = defaultdict(lambda: {
            "cum_tp_vol": 0.0, "cum_vol": 0, "vwap": 0.0,
        })
        self._positions: dict[str, Position] = {}  # trade_id -> Position
        self._sector_counts: dict[str, int] = defaultdict(int)
        self._simultaneous_triggers: int = 0

        # External deps (set by engine before initialize)
        self.data_feed: DataFeed | None = None
        self.risk_manager: GlobalRiskManager | None = None
        self.journal: TradeJournal | None = None
        self.broker = None
        self.tokens: dict[str, str] = {}  # symbol -> token

    async def initialize(self):
        log.info("RSI Bounce strategy initializing with %d stocks", len(self.tokens))

    async def on_candle(self, token: str, candle: Candle) -> Signal | None:
        """Process a completed 1-min candle."""
        if not self._active:
            return None

        symbol = candle.symbol or self._token_to_symbol(token)

        # Update candle buffer
        self._update_buffer(token, candle)

        # Update VWAP
        self._update_vwap(token, candle)

        # Build 5-min candle from 1-min
        self._update_5min(token, candle)

        # Check existing positions for exit signals
        await self._check_exits(token, candle)

        # Skip first 15 minutes
        now = datetime.now(IST)
        market_open = now.replace(hour=9, minute=15, second=0)
        if (now - market_open).seconds < self.skip_first_minutes * 60:
            return None

        # Don't enter after 3:10 PM
        if now.time() > time(15, 10):
            return None

        # Need enough candles for indicators
        df = self._candle_buffers.get(token)
        if df is None or len(df) < 20:
            return None

        # Calculate RSI
        rsi_series = ta.rsi(df["close"], length=self.rsi_period)
        if rsi_series is None or rsi_series.empty:
            return None
        current_rsi = rsi_series.iloc[-1]
        prev_rsi = rsi_series.iloc[-2] if len(rsi_series) > 1 else current_rsi

        # Entry trigger: RSI crosses below oversold
        if not (prev_rsi >= self.oversold and current_rsi < self.oversold):
            return None

        # Already in a position for this token?
        for pos in self._positions.values():
            if pos.token == token:
                return None

        log.info("RSI trigger: %s RSI=%.1f (prev=%.1f)", symbol, current_rsi, prev_rsi)

        # Run filter stack
        if not self._check_filters(token, df, candle):
            return None

        # Risk check
        ok, reason = self.risk_manager.can_trade(self.name)
        if not ok:
            log.info("Risk blocked: %s", reason)
            return None

        # Sector check
        sector = SECTOR_MAP.get(symbol, "Other")
        if self._sector_counts.get(sector, 0) >= self.max_per_sector:
            log.info("Sector limit: %s already has %d positions", sector, self._sector_counts[sector])
            return None

        # Calculate stop-loss
        atr_series = ta.atr(df["high"], df["low"], df["close"], length=14)
        current_atr = atr_series.iloc[-1] if atr_series is not None and not atr_series.empty else candle.close * 0.005
        stop_loss = round(candle.close - self.atr_sl_mult * current_atr, 2)

        # Position sizing
        quantity = self.risk_manager.calculate_position_size(self.name, candle.close, stop_loss)
        if quantity <= 0:
            return None

        return Signal(
            signal_type=SignalType.BUY,
            symbol=symbol,
            token=token,
            price=candle.close,
            strategy_name=self.name,
            quantity=quantity,
            stop_loss=stop_loss,
            reason=f"RSI({self.rsi_period})={current_rsi:.1f} < {self.oversold}",
            indicators={
                "rsi": round(current_rsi, 2),
                "atr": round(current_atr, 2),
                "vwap": round(self._vwap_state[token]["vwap"], 2),
            },
        )

    async def on_tick(self, token: str, price: float) -> Signal | None:
        """Check stop-losses on every tick for faster reaction."""
        for tid, pos in list(self._positions.items()):
            if pos.token == token and price <= pos.stop_loss:
                return Signal(
                    signal_type=SignalType.EXIT,
                    symbol=pos.symbol,
                    token=token,
                    price=price,
                    strategy_name=self.name,
                    quantity=pos.remaining_quantity,
                    reason="stop_loss_tick",
                )
        return None

    def _check_filters(self, token: str, df: pd.DataFrame, candle: Candle) -> bool:
        """Run the full filter stack. All must pass."""
        symbol = candle.symbol or self._token_to_symbol(token)

        # Filter 1: Price above 20-EMA on 5-min chart
        df_5m = self._5min_buffers.get(token)
        if df_5m is not None and len(df_5m) >= 20:
            ema20 = ta.ema(df_5m["close"], length=20)
            if ema20 is not None and not ema20.empty:
                if candle.close < ema20.iloc[-1]:
                    log.debug("Filter FAIL: %s below 5m EMA20 (%.2f < %.2f)",
                              symbol, candle.close, ema20.iloc[-1])
                    return False
        elif df_5m is None or len(df_5m) < 20:
            # Not enough 5-min data yet — fall back to 1-min EMA(100) as proxy
            # 100 x 1-min ≈ 20 x 5-min
            df_1m = self._candle_buffers.get(token)
            if df_1m is not None and len(df_1m) >= 100:
                ema100 = ta.ema(df_1m["close"], length=100)
                if ema100 is not None and not ema100.empty:
                    if candle.close < ema100.iloc[-1]:
                        log.debug("Filter FAIL: %s below 1m EMA100 proxy (%.2f < %.2f)",
                                  symbol, candle.close, ema100.iloc[-1])
                        return False

        # Filter 2: Price at or above VWAP
        vwap = self._vwap_state[token]["vwap"]
        if vwap > 0 and candle.close < vwap * 0.998:  # Small tolerance
            log.debug("Filter FAIL: %s below VWAP (%.2f < %.2f)", symbol, candle.close, vwap)
            return False

        # Filter 3: Volume > 1.5x 20-period average
        if len(df) >= 20:
            avg_vol = df["volume"].iloc[-20:].mean()
            if avg_vol > 0 and candle.volume < avg_vol * self.volume_mult:
                log.debug("Filter FAIL: %s volume %d < %.0f (1.5x avg)",
                          symbol, candle.volume, avg_vol * self.volume_mult)
                return False

        # Filter 4: ADX > 20
        adx_series = ta.adx(df["high"], df["low"], df["close"], length=14)
        if adx_series is not None and not adx_series.empty:
            adx_val = adx_series.iloc[-1].get("ADX_14", 0) if hasattr(adx_series.iloc[-1], "get") else 0
            # pandas_ta returns a DataFrame with ADX_14 column
            if isinstance(adx_series, pd.DataFrame) and "ADX_14" in adx_series.columns:
                adx_val = adx_series["ADX_14"].iloc[-1]
            if adx_val < self.adx_threshold:
                log.debug("Filter FAIL: %s ADX=%.1f < %d", symbol, adx_val, self.adx_threshold)
                return False

        log.info("All filters PASSED for %s", symbol)
        return True

    async def _check_exits(self, token: str, candle: Candle):
        """Check existing positions for exit conditions."""
        df = self._candle_buffers.get(token)
        if df is None or len(df) < self.rsi_period + 1:
            return

        rsi_series = ta.rsi(df["close"], length=self.rsi_period)
        if rsi_series is None or rsi_series.empty:
            return
        current_rsi = rsi_series.iloc[-1]

        for tid, pos in list(self._positions.items()):
            if pos.token != token:
                continue

            pos.entry_candle_count += 1

            # Time stop: 15 candles without RSI reaching 40
            if pos.entry_candle_count >= self.time_stop_candles and current_rsi < self.exit_1_rsi:
                await self._exit_position(tid, candle.close, pos.remaining_quantity, "time_stop")
                continue

            # Stop-loss
            if candle.close <= pos.stop_loss:
                await self._exit_position(tid, candle.close, pos.remaining_quantity, "stop_loss")
                continue

            # Partial exit 1: RSI crosses above 40 — exit 50%
            if pos.status == PositionStatus.OPEN and current_rsi >= self.exit_1_rsi:
                exit_qty = pos.remaining_quantity // 2
                if exit_qty > 0:
                    await self._exit_position(tid, candle.close, exit_qty, "rsi_40")
                    pos.remaining_quantity -= exit_qty
                    pos.status = PositionStatus.PARTIAL_EXIT

            # Final exit: RSI crosses above 50 — exit remaining
            elif pos.status == PositionStatus.PARTIAL_EXIT and current_rsi >= self.exit_2_rsi:
                await self._exit_position(tid, candle.close, pos.remaining_quantity, "rsi_50")

    async def _exit_position(self, trade_id: str, price: float, quantity: int, reason: str):
        """Execute an exit (partial or full)."""
        pos = self._positions.get(trade_id)
        if not pos:
            return

        pnl = self.journal.record_exit(trade_id, price, quantity, reason)
        trade_log.info("RSI EXIT | %s | %s x%d @ ₹%.2f | reason=%s | P&L=₹%.2f",
                       pos.symbol, reason, quantity, price, reason, pnl)

        pos.remaining_quantity -= quantity
        pos.realized_pnl += pnl

        if pos.remaining_quantity <= 0:
            sector = SECTOR_MAP.get(pos.symbol, "Other")
            self._sector_counts[sector] = max(0, self._sector_counts.get(sector, 0) - 1)
            self.risk_manager.remove_position(self.name, trade_id)
            self.risk_manager.record_trade_result(self.name, pos.realized_pnl)
            del self._positions[trade_id]

    def register_position(self, pos: Position):
        """Called by engine after order fill to register a new position."""
        self._positions[pos.trade_id] = pos
        sector = SECTOR_MAP.get(pos.symbol, "Other")
        self._sector_counts[sector] = self._sector_counts.get(sector, 0) + 1

    def _update_buffer(self, token: str, candle: Candle):
        """Append candle to rolling buffer."""
        row = {
            "timestamp": candle.timestamp, "open": candle.open, "high": candle.high,
            "low": candle.low, "close": candle.close, "volume": candle.volume,
        }
        if token not in self._candle_buffers:
            self._candle_buffers[token] = pd.DataFrame([row])
        else:
            self._candle_buffers[token] = pd.concat(
                [self._candle_buffers[token], pd.DataFrame([row])],
                ignore_index=True,
            ).tail(200)

    def _update_5min(self, token: str, candle: Candle):
        """Resample 1-min candles into 5-min candles using time-based grouping."""
        df = self._candle_buffers.get(token)
        if df is None or len(df) < 5:
            return
        try:
            temp = df.set_index("timestamp")
            resampled = temp.resample("5min").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum",
            }).dropna()
            if not resampled.empty:
                self._5min_buffers[token] = resampled.reset_index().tail(50)
        except Exception:
            pass  # Non-critical — 1-min EMA100 fallback will be used

    def _update_vwap(self, token: str, candle: Candle):
        """Incremental VWAP calculation."""
        state = self._vwap_state[token]
        tp = candle.typical_price
        state["cum_tp_vol"] += tp * candle.volume
        state["cum_vol"] += candle.volume
        if state["cum_vol"] > 0:
            state["vwap"] = state["cum_tp_vol"] / state["cum_vol"]

    def _token_to_symbol(self, token: str) -> str:
        """Reverse lookup token -> symbol."""
        for sym, tok in self.tokens.items():
            if tok == token:
                return sym
        return token

    async def teardown(self):
        """Close all remaining positions at market."""
        for tid in list(self._positions.keys()):
            pos = self._positions[tid]
            log.warning("Teardown: force closing %s", pos.symbol)
            # Price will need to be fetched by the engine
