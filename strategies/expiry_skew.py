"""OTM Premium Skew Strategy (Iron Condor) — Plugin version.

Runs on Nifty expiry days (Tuesday) at 2 PM IST.
Compares OTM Put vs Call premiums, enters iron condor if skew >= 2:1.
Monitors with SL at 2x premium, hard exit at 3:15 PM.
"""

import logging
import time
from dataclasses import dataclass
from datetime import date, datetime

import pytz

from core.data_feed import DataFeed
from core.risk_manager import GlobalRiskManager
from core.trade_journal import TradeJournal
from models.types import Candle, Signal, SignalType
from src.expiry import is_nifty_expiry_day
from src.fees import calculate_fees
from src.instruments import InstrumentMaster
from strategies.base import BaseStrategy, StrategyEngine

log = logging.getLogger("autotheta.expiry_skew")
trade_log = logging.getLogger("autotheta.trades")
IST = pytz.timezone("Asia/Kolkata")

NIFTY_SPOT_TOKEN = "99926000"
INDIA_VIX_TOKEN = "99926004"


@dataclass
class IronCondorState:
    """Tracks the 4 legs of an active iron condor."""
    sell_put: dict  # {'symbol', 'token', 'strike', 'premium', 'trade_id'}
    sell_call: dict
    buy_put: dict
    buy_call: dict
    sl_put: float
    sl_call: float
    net_credit: float
    put_closed: bool = False
    call_closed: bool = False


@StrategyEngine.register("expiry_skew")
class ExpirySkewStrategy(BaseStrategy):
    """OTM premium skew iron condor on Nifty expiry days."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.otm_offset = self.params.get("otm_offset", 50)
        self.wing_width = self.params.get("wing_width", 100)
        self.min_skew_ratio = self.params.get("min_skew_ratio", 2.0)
        self.sl_multiplier = self.params.get("sl_multiplier", 2.0)
        self.entry_hour = self.params.get("entry_hour", 14)
        self.entry_minute = self.params.get("entry_minute", 0)
        self.exit_hour = self.params.get("exit_hour", 15)
        self.exit_minute = self.params.get("exit_minute", 15)
        self.monitor_interval = self.params.get("monitor_interval", 30)
        self.lot_size = self.params.get("lot_size", 65)
        self.vix_min = self.params.get("vix_min", 12)
        self.vix_max = self.params.get("vix_max", 18)

        # External deps (set by engine)
        self.instruments: InstrumentMaster | None = None
        self.risk_manager: GlobalRiskManager | None = None
        self.journal: TradeJournal | None = None
        self.api = None  # SmartConnect instance for LTP
        self.broker = None

        # State
        self._condor: IronCondorState | None = None
        self._executed_today: bool = False

    async def initialize(self):
        log.info("Expiry Skew strategy initialized")

    async def on_candle(self, token: str, candle: Candle) -> Signal | None:
        """Not candle-driven — this strategy is time-triggered."""
        return None

    async def on_tick(self, token: str, price: float) -> Signal | None:
        """Not tick-driven."""
        return None

    async def execute(self) -> bool:
        """Main execution — called by the scheduler at 2 PM on expiry days.

        Returns True if a trade was taken.
        """
        if not is_nifty_expiry_day():
            log.info("Not an expiry day — skipping")
            return False

        if self._executed_today:
            log.info("Already executed today")
            return False

        log.info("=" * 50)
        log.info("Expiry Skew — executing iron condor strategy")

        # 1. VIX check
        try:
            vix = self._fetch_ltp("NSE", "India VIX", INDIA_VIX_TOKEN)
            if vix < self.vix_min or vix > self.vix_max:
                log.warning("VIX %.1f outside [%d, %d] range — skipping", vix, self.vix_min, self.vix_max)
                return False
        except Exception:
            log.warning("VIX fetch failed — proceeding cautiously")

        # 2. Risk check
        ok, reason = self.risk_manager.can_trade(self.name)
        if not ok:
            log.info("Risk blocked: %s", reason)
            return False

        # 3. Spot price and strikes
        spot = self._fetch_ltp("NSE", "NIFTY", NIFTY_SPOT_TOKEN)
        atm = round(spot / 50) * 50
        sell_put_strike = atm - self.otm_offset
        sell_call_strike = atm + self.otm_offset
        buy_put_strike = sell_put_strike - self.wing_width
        buy_call_strike = sell_call_strike + self.wing_width

        log.info("Spot: %.2f | ATM: %d | Sells: %dPE/%dCE | Buys: %dPE/%dCE",
                 spot, atm, sell_put_strike, sell_call_strike, buy_put_strike, buy_call_strike)

        # 4. Look up instruments
        expiry = self.instruments.get_nearest_expiry()
        if not expiry:
            log.error("No expiry found")
            return False

        legs = {}
        for label, strike, opt_type in [
            ("sell_put", sell_put_strike, "PE"),
            ("sell_call", sell_call_strike, "CE"),
            ("buy_put", buy_put_strike, "PE"),
            ("buy_call", buy_call_strike, "CE"),
        ]:
            info = self.instruments.lookup(strike, opt_type, expiry)
            if not info:
                log.error("Instrument not found: %s %d%s", label, strike, opt_type)
                return False
            premium = self._fetch_ltp("NFO", info["symbol"], info["token"])
            legs[label] = {**info, "strike": strike, "premium": premium}

        # 5. Skew check
        sp = legs["sell_put"]["premium"]
        sc = legs["sell_call"]["premium"]
        denom = max(min(sp, sc), 0.05)
        ratio = max(sp, sc) / denom
        if ratio < self.min_skew_ratio:
            log.info("Skew ratio %.1f < %.1f threshold — no trade", ratio, self.min_skew_ratio)
            return False

        net_credit = (sp + sc) - (legs["buy_put"]["premium"] + legs["buy_call"]["premium"])
        log.info("Premiums: SellPut=%.2f SellCall=%.2f BuyPut=%.2f BuyCall=%.2f Net=%.2f",
                 sp, sc, legs["buy_put"]["premium"], legs["buy_call"]["premium"], net_credit)

        # 6. Place all 4 legs
        for label, side in [("sell_put", "SELL"), ("sell_call", "SELL"),
                            ("buy_put", "BUY"), ("buy_call", "BUY")]:
            leg = legs[label]
            tid = TradeJournal.generate_trade_id("IC")
            self.journal.record_entry(
                tid, self.name, leg["symbol"], leg["token"], side,
                self.lot_size, leg["premium"],
                indicators=f'{{"strike": {leg["strike"]}, "leg": "{label}"}}',
            )
            leg["trade_id"] = tid
            log.info("Placed %s %s %s @ ₹%.2f", side, label, leg["symbol"], leg["premium"])

        # 7. Set up monitoring state
        self._condor = IronCondorState(
            sell_put=legs["sell_put"], sell_call=legs["sell_call"],
            buy_put=legs["buy_put"], buy_call=legs["buy_call"],
            sl_put=sp * self.sl_multiplier,
            sl_call=sc * self.sl_multiplier,
            net_credit=net_credit,
        )

        trade_log.info("IC ENTRY | Spot=%.2f ATM=%d | SP=%d@%.2f SC=%d@%.2f | Net=%.2f/unit",
                       spot, atm, sell_put_strike, sp, sell_call_strike, sc, net_credit)

        # 8. Monitor until exit time
        self._monitor_condor()

        self._executed_today = True
        return True

    def _monitor_condor(self):
        """Blocking monitor loop — checks SL and hard exit time."""
        if not self._condor:
            return

        log.info("Monitoring: SL Put=%.2f | SL Call=%.2f | Exit at %02d:%02d",
                 self._condor.sl_put, self._condor.sl_call, self.exit_hour, self.exit_minute)

        while True:
            now = datetime.now(IST)
            if now.hour > self.exit_hour or (now.hour == self.exit_hour and now.minute >= self.exit_minute):
                log.info("Hard exit time reached")
                break

            time.sleep(self.monitor_interval)

            # Check sell put SL
            if not self._condor.put_closed:
                try:
                    curr = self._fetch_ltp("NFO", self._condor.sell_put["symbol"],
                                           self._condor.sell_put["token"])
                    if curr >= self._condor.sl_put:
                        pnl = self.journal.record_exit(
                            self._condor.sell_put["trade_id"], curr,
                            self.lot_size, "stop_loss",
                        )
                        log.warning("SL HIT PUT @ ₹%.2f P&L=₹%.2f", curr, pnl)
                        self._condor.put_closed = True
                except Exception:
                    log.exception("Error checking put SL")

            # Check sell call SL
            if not self._condor.call_closed:
                try:
                    curr = self._fetch_ltp("NFO", self._condor.sell_call["symbol"],
                                           self._condor.sell_call["token"])
                    if curr >= self._condor.sl_call:
                        pnl = self.journal.record_exit(
                            self._condor.sell_call["trade_id"], curr,
                            self.lot_size, "stop_loss",
                        )
                        log.warning("SL HIT CALL @ ₹%.2f P&L=₹%.2f", curr, pnl)
                        self._condor.call_closed = True
                except Exception:
                    log.exception("Error checking call SL")

            if self._condor.put_closed and self._condor.call_closed:
                break

        # Hard exit: close remaining legs
        self._close_remaining()

    def _close_remaining(self):
        """Close all remaining open legs."""
        if not self._condor:
            return

        legs_to_close = []
        if not self._condor.put_closed:
            legs_to_close.append(("sell_put", self._condor.sell_put))
        if not self._condor.call_closed:
            legs_to_close.append(("sell_call", self._condor.sell_call))
        legs_to_close.append(("buy_put", self._condor.buy_put))
        legs_to_close.append(("buy_call", self._condor.buy_call))

        total_pnl = 0.0
        for label, leg in legs_to_close:
            try:
                curr = self._fetch_ltp("NFO", leg["symbol"], leg["token"])
                pnl = self.journal.record_exit(
                    leg["trade_id"], curr, self.lot_size, "hard_exit",
                )
                total_pnl += pnl
                log.info("EXIT %s %s @ ₹%.2f P&L=₹%.2f", label, leg["symbol"], curr, pnl)
            except Exception:
                log.exception("Failed to close %s", label)

        self.risk_manager.record_trade_result(self.name, total_pnl)
        trade_log.info("IC EXIT | Total P&L=₹%.2f", total_pnl)
        self._condor = None

    def _fetch_ltp(self, exchange: str, symbol: str, token: str) -> float:
        """Fetch LTP via REST API."""
        result = self.api.ltpData(exchange, symbol, token)
        if not result or not result.get("status"):
            raise RuntimeError(f"LTP failed: {symbol}")
        return float(result["data"]["ltp"])

    async def teardown(self):
        if self._condor:
            log.warning("Teardown: closing remaining condor legs")
            self._close_remaining()
