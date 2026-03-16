"""Strategy engine — runs all strategies concurrently via asyncio.

Responsibilities:
- Load and initialize strategies from config
- Run each strategy as an independent asyncio task with crash isolation
- Route candles from DataFeed to subscribed strategies
- Handle order submission through the broker
- Coordinate graceful shutdown
"""

import asyncio
import json
import logging
from datetime import datetime, time

import pytz

from core.data_feed import DataFeed
from core.rate_limiter import RateLimiter
from core.risk_manager import GlobalRiskManager
from core.trade_journal import TradeJournal
from models.types import Position, Side, Signal, SignalType
from src.auth import AngelSession
from src.instruments import InstrumentMaster
from strategies.base import BaseStrategy, StrategyEngine

# Import strategies to trigger @register decorators
import strategies.rsi_bounce  # noqa: F401
import strategies.expiry_skew  # noqa: F401

log = logging.getLogger("autotheta.engine")
IST = pytz.timezone("Asia/Kolkata")


class TradingEngine:
    """Main engine that orchestrates strategies, data, and execution."""

    def __init__(self, config: dict):
        self.config = config
        self._shutdown = asyncio.Event()

        # Core components
        self.session = AngelSession()
        self.rate_limiter = RateLimiter()
        self.risk_manager = GlobalRiskManager(config.get("risk", {}))
        self.journal = TradeJournal(config["bot"].get("db_path", "data/trades.db"))
        self.instruments = InstrumentMaster()
        self.data_feed: DataFeed | None = None

        # Strategies
        self.strategies: dict[str, BaseStrategy] = {}

        # Nifty 50 tokens (loaded at startup)
        self._nifty50_tokens: dict[str, str] = {}  # symbol -> token

    async def start(self):
        """Main entry point — authenticate, load strategies, run event loop."""
        log.info("=" * 60)
        log.info("AutoTheta Trading Engine starting")
        log.info("Mode: %s", "PAPER" if self.config["bot"].get("paper_mode", True) else "LIVE")
        log.info("=" * 60)

        # 1. Authenticate
        if not self.session.login():
            log.error("Authentication failed — cannot start")
            return

        # 2. Load instruments
        if not self.instruments.load():
            log.error("Instrument master failed — cannot start")
            return

        # 3. Load Nifty 50 token map
        self._load_nifty50_tokens()

        # 4. Initialize data feed
        self.data_feed = DataFeed(
            self.session.api, self.session.auth_token,
            self.config["broker"]["api_key"], self.config["broker"]["client_code"],
            self.session.feed_token, self.rate_limiter,
        )
        self.data_feed.set_token_map({v: k for k, v in self._nifty50_tokens.items()})

        # 5. Load strategies
        strategies_config = self.config.get("strategies", [])
        self.strategies = StrategyEngine.load_all(strategies_config)
        log.info("Loaded %d strategies: %s", len(self.strategies), list(self.strategies.keys()))

        # 6. Wire up dependencies and register risk configs
        for name, strategy in self.strategies.items():
            strategy.data_feed = self.data_feed
            strategy.risk_manager = self.risk_manager
            strategy.journal = self.journal
            strategy.api = self.session.api

            risk_cfg = strategy.risk_config
            self.risk_manager.register_strategy(name, risk_cfg)

            # RSI bounce needs token map
            if hasattr(strategy, "tokens"):
                strategy.tokens = self._nifty50_tokens

            # Expiry skew needs instruments
            if hasattr(strategy, "instruments"):
                strategy.instruments = self.instruments

        # 7. Initialize all strategies
        for name, strategy in self.strategies.items():
            await strategy.initialize()
            log.info("Strategy %s initialized", name)

        # 8. Start WebSocket for RSI bounce (if enabled)
        rsi_tokens = list(self._nifty50_tokens.values())
        if rsi_tokens:
            loop = asyncio.get_event_loop()
            self.data_feed.start_websocket(rsi_tokens, loop)

        # 9. Run strategy tasks concurrently
        tasks = []
        for name, strategy in self.strategies.items():
            if strategy.config["strategy"]["type"] == "rsi_bounce":
                tasks.append(asyncio.create_task(
                    self._run_candle_strategy(strategy), name=f"task_{name}",
                ))
            elif strategy.config["strategy"]["type"] == "expiry_skew":
                tasks.append(asyncio.create_task(
                    self._run_scheduled_strategy(strategy), name=f"task_{name}",
                ))

        log.info("All strategy tasks started (%d)", len(tasks))

        try:
            await self._shutdown.wait()
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self._teardown()

    async def _run_candle_strategy(self, strategy: BaseStrategy):
        """Run a candle-driven strategy (RSI bounce). Crash-isolated."""
        log.info("Starting candle loop for %s", strategy.name)
        while not self._shutdown.is_set():
            try:
                # Check trading window: 9:15 AM - 3:15 PM
                now = datetime.now(IST)
                if now.time() < time(9, 15) or now.time() > time(15, 15):
                    await asyncio.sleep(60)
                    continue

                # Process candles from all subscribed tokens
                for token in self._nifty50_tokens.values():
                    candles = self.data_feed.get_candles(token)
                    if not candles:
                        continue

                    latest = candles[-1]
                    signal = await strategy.on_candle(token, latest)
                    if signal and signal.signal_type != SignalType.HOLD:
                        await self._process_signal(signal)

                await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("Error in %s candle loop", strategy.name)
                await asyncio.sleep(5)

    async def _run_scheduled_strategy(self, strategy):
        """Run a time-triggered strategy (expiry skew). Waits until trigger time."""
        log.info("Scheduled strategy %s waiting for trigger", strategy.name)
        entry_hour = strategy.params.get("entry_hour", 14)
        entry_minute = strategy.params.get("entry_minute", 0)

        while not self._shutdown.is_set():
            try:
                now = datetime.now(IST)
                # Check if it's trigger time (within 5 minute window)
                if (now.hour == entry_hour and
                        entry_minute <= now.minute < entry_minute + 5):
                    await strategy.execute()

                await asyncio.sleep(30)
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("Error in %s scheduled loop", strategy.name)
                await asyncio.sleep(60)

    async def _process_signal(self, signal: Signal):
        """Process a trading signal — place order and register position."""
        log.info("Signal: %s %s %s @ ₹%.2f qty=%d [%s]",
                 signal.signal_type.value, signal.symbol, signal.token,
                 signal.price, signal.quantity, signal.reason)

        if signal.signal_type == SignalType.BUY:
            trade_id = TradeJournal.generate_trade_id("RSI")
            self.journal.record_entry(
                trade_id, signal.strategy_name, signal.symbol, signal.token,
                "BUY", signal.quantity, signal.price, signal.stop_loss,
                indicators=json.dumps(signal.indicators),
            )
            pos = Position(
                trade_id=trade_id, symbol=signal.symbol, token=signal.token,
                side=Side.BUY, quantity=signal.quantity,
                entry_price=signal.price, entry_time=datetime.now(IST),
                strategy_name=signal.strategy_name,
                stop_loss=signal.stop_loss, indicators=signal.indicators,
            )
            self.risk_manager.add_position(signal.strategy_name, pos)

            # Register with the strategy
            strategy = self.strategies.get(signal.strategy_name)
            if strategy and hasattr(strategy, "register_position"):
                strategy.register_position(pos)

        elif signal.signal_type == SignalType.EXIT:
            pnl = self.journal.record_exit(
                signal.reason,  # trade_id passed via reason for exits
                signal.price, signal.quantity, "exit_signal",
            )
            self.risk_manager.record_trade_result(signal.strategy_name, pnl)

    async def _teardown(self):
        """Graceful shutdown — close positions, stop feeds, logout."""
        log.info("Shutting down engine...")
        for name, strategy in self.strategies.items():
            await strategy.teardown()
            self.journal.update_daily_summary(name)

        if self.data_feed:
            self.data_feed.stop_websocket()

        self.journal.close()
        self.session.logout()
        log.info("Engine shutdown complete")

    def shutdown(self):
        """Signal the engine to shut down gracefully."""
        self._shutdown.set()

    def _load_nifty50_tokens(self):
        """Load Nifty 50 equity token map from instrument master."""
        nifty50_symbols = list(strategies.rsi_bounce.SECTOR_MAP.keys())
        for sym in nifty50_symbols:
            matches = self.instruments.df[
                (self.instruments.df["symbol"] == sym)
                & (self.instruments.df["exch_seg"] == "NSE")
            ]
            if not matches.empty:
                self._nifty50_tokens[sym] = matches.iloc[0]["token"]
        log.info("Loaded %d Nifty 50 tokens", len(self._nifty50_tokens))
