"""OTM Premium Skew Strategy with Iron Condor construction.

Flow:
1. At 2 PM on expiry day, fetch Nifty spot
2. Calculate ATM, then 50-point OTM Put and Call strikes
3. Compare premiums — need >= 2:1 skew ratio to enter
4. Build iron condor (sell OTM + buy further OTM on both sides)
5. Monitor with 30s interval, SL at 2x premium
6. Hard exit at 3:15 PM
"""

import logging
import time
from dataclasses import dataclass
from datetime import date

import pytz

from config.settings import (
    MONITOR_INTERVAL,
    NIFTY_LOT_SIZE,
    NIFTY_STRIKE_GAP,
    OTM_OFFSET,
    WING_WIDTH,
)
from src.broker import BaseBroker
from src.data import MarketData
from src.instruments import InstrumentMaster
from src.risk import RiskManager

log = logging.getLogger("autotheta.strategy")
trade_log = logging.getLogger("autotheta.trades")
IST = pytz.timezone("Asia/Kolkata")


@dataclass
class IronCondorLegs:
    """Represents all 4 legs of an iron condor."""
    # Short legs (sell)
    sell_put_strike: float
    sell_call_strike: float
    # Long legs (buy, protection)
    buy_put_strike: float
    buy_call_strike: float
    # Instrument info
    sell_put: dict  # {'symbol': ..., 'token': ...}
    sell_call: dict
    buy_put: dict
    buy_call: dict
    # Premiums
    sell_put_premium: float = 0.0
    sell_call_premium: float = 0.0
    buy_put_premium: float = 0.0
    buy_call_premium: float = 0.0

    @property
    def net_credit(self) -> float:
        return (self.sell_put_premium + self.sell_call_premium
                - self.buy_put_premium - self.buy_call_premium)

    @property
    def max_loss_per_unit(self) -> float:
        """Max loss = wing width - net credit."""
        return WING_WIDTH - self.net_credit

    @property
    def max_loss_per_lot(self) -> float:
        return self.max_loss_per_unit * NIFTY_LOT_SIZE


class OTMSkewStrategy:
    """Main strategy class — orchestrates the full trade lifecycle."""

    def __init__(self, market_data: MarketData, instruments: InstrumentMaster,
                 broker: BaseBroker, risk: RiskManager):
        self.md = market_data
        self.instruments = instruments
        self.broker = broker
        self.risk = risk
        # Track open order IDs for monitoring
        self._sell_put_id: str | None = None
        self._sell_call_id: str | None = None
        self._buy_put_id: str | None = None
        self._buy_call_id: str | None = None

    def run(self) -> bool:
        """Execute the full strategy. Returns True if trade was taken."""
        log.info("=" * 50)
        log.info("Strategy execution started")

        # 1. Get expiry and load chain
        expiry = self.instruments.get_nearest_expiry()
        if not expiry:
            log.error("No expiry found")
            return False
        log.info("Target expiry: %s", expiry)

        # 2. Fetch VIX
        try:
            vix = self.md.get_india_vix()
        except Exception:
            log.warning("VIX fetch failed — proceeding with caution")
            vix = 15.0  # Assume mid-range

        # 3. Fetch spot and compute strikes
        spot = self.md.get_nifty_spot()
        atm = round(spot / NIFTY_STRIKE_GAP) * NIFTY_STRIKE_GAP
        otm_put_strike = atm - OTM_OFFSET
        otm_call_strike = atm + OTM_OFFSET
        log.info("Spot: %.2f | ATM: %d | OTM Put: %d | OTM Call: %d",
                 spot, atm, otm_put_strike, otm_call_strike)

        # 4. Look up instruments for all 4 iron condor legs
        legs = self._build_iron_condor_legs(expiry, otm_put_strike, otm_call_strike)
        if not legs:
            return False

        # 5. Fetch all premiums
        legs = self._fetch_premiums(legs)
        if not legs:
            return False

        log.info("Iron Condor: Sell %dPE@%.2f + Sell %dCE@%.2f | "
                 "Buy %dPE@%.2f + Buy %dCE@%.2f | Net credit: %.2f/unit",
                 legs.sell_put_strike, legs.sell_put_premium,
                 legs.sell_call_strike, legs.sell_call_premium,
                 legs.buy_put_strike, legs.buy_put_premium,
                 legs.buy_call_strike, legs.buy_call_premium,
                 legs.net_credit)

        # 6. Run pre-trade risk checks
        ok, reason = self.risk.pre_trade_checks(vix, legs.sell_put_premium, legs.sell_call_premium)
        if not ok:
            log.info("Trade skipped: %s", reason)
            return False

        # 7. Check position sizing (1% rule)
        ok, reason = self.risk.check_position_size(legs.max_loss_per_lot, 1)
        if not ok:
            log.warning("Position size check failed: %s", reason)
            return False

        # 8. Place all 4 legs
        self._place_iron_condor(legs, expiry)

        trade_log.info("ENTRY | Spot=%.2f ATM=%d | Put=%d@%.2f Call=%d@%.2f | "
                       "Net=%.2f/unit MaxLoss=%.0f/lot",
                       spot, atm, otm_put_strike, legs.sell_put_premium,
                       otm_call_strike, legs.sell_call_premium,
                       legs.net_credit, legs.max_loss_per_lot)

        # 9. Monitor until hard exit time
        self._monitor_positions(legs)

        log.info("Strategy execution complete")
        return True

    def _build_iron_condor_legs(self, expiry: date, put_strike: float,
                                call_strike: float) -> IronCondorLegs | None:
        """Look up all 4 instruments for the iron condor."""
        buy_put_strike = put_strike - WING_WIDTH
        buy_call_strike = call_strike + WING_WIDTH

        sell_put = self.instruments.lookup(put_strike, "PE", expiry)
        sell_call = self.instruments.lookup(call_strike, "CE", expiry)
        buy_put = self.instruments.lookup(buy_put_strike, "PE", expiry)
        buy_call = self.instruments.lookup(buy_call_strike, "CE", expiry)

        if not all([sell_put, sell_call, buy_put, buy_call]):
            log.error("Could not find all iron condor instruments")
            return None

        return IronCondorLegs(
            sell_put_strike=put_strike, sell_call_strike=call_strike,
            buy_put_strike=buy_put_strike, buy_call_strike=buy_call_strike,
            sell_put=sell_put, sell_call=sell_call,
            buy_put=buy_put, buy_call=buy_call,
        )

    def _fetch_premiums(self, legs: IronCondorLegs) -> IronCondorLegs | None:
        """Fetch live premiums for all 4 legs."""
        try:
            legs.sell_put_premium = self.md.get_option_ltp(legs.sell_put["symbol"], legs.sell_put["token"])
            legs.sell_call_premium = self.md.get_option_ltp(legs.sell_call["symbol"], legs.sell_call["token"])
            legs.buy_put_premium = self.md.get_option_ltp(legs.buy_put["symbol"], legs.buy_put["token"])
            legs.buy_call_premium = self.md.get_option_ltp(legs.buy_call["symbol"], legs.buy_call["token"])
            return legs
        except Exception:
            log.exception("Failed to fetch premiums")
            return None

    def _place_iron_condor(self, legs: IronCondorLegs, expiry: date):
        """Place all 4 legs of the iron condor."""
        expiry_str = str(expiry)

        # Sell OTM Put
        self._sell_put_id = self.broker.sell_option(
            legs.sell_put["symbol"], legs.sell_put["token"],
            legs.sell_put_strike, expiry_str, "PE", legs.sell_put_premium,
        )
        # Sell OTM Call
        self._sell_call_id = self.broker.sell_option(
            legs.sell_call["symbol"], legs.sell_call["token"],
            legs.sell_call_strike, expiry_str, "CE", legs.sell_call_premium,
        )
        # Buy far OTM Put (hedge)
        self._buy_put_id = self.broker.buy_option(
            legs.buy_put["symbol"], legs.buy_put["token"],
            legs.buy_put_strike, expiry_str, "PE", legs.buy_put_premium,
        )
        # Buy far OTM Call (hedge)
        self._buy_call_id = self.broker.buy_option(
            legs.buy_call["symbol"], legs.buy_call["token"],
            legs.buy_call_strike, expiry_str, "CE", legs.buy_call_premium,
        )

        log.info("Iron condor placed: sell_put=%s sell_call=%s buy_put=%s buy_call=%s",
                 self._sell_put_id, self._sell_call_id, self._buy_put_id, self._buy_call_id)

    def _monitor_positions(self, legs: IronCondorLegs):
        """Monitor open positions until hard exit, checking SL every MONITOR_INTERVAL seconds."""
        sl_put = self.risk.calculate_sl(legs.sell_put_premium)
        sl_call = self.risk.calculate_sl(legs.sell_call_premium)
        log.info("Stop-losses set: Put SL @ ₹%.2f | Call SL @ ₹%.2f", sl_put, sl_call)

        put_closed = False
        call_closed = False

        while True:
            # Check hard exit time
            ok, _ = self.risk.check_exit_time()
            if not ok:
                log.info("Hard exit time reached — closing all positions")
                break

            # Check daily loss cap
            ok, _ = self.risk.check_daily_loss_cap()
            if not ok:
                log.warning("Daily loss cap breached — closing all positions")
                break

            time.sleep(MONITOR_INTERVAL)

            # Check sell put SL
            if not put_closed and self._sell_put_id:
                try:
                    curr_put = self.md.get_option_ltp(legs.sell_put["symbol"], legs.sell_put["token"])
                    if curr_put >= sl_put:
                        pnl = self.broker.close_position(
                            self._sell_put_id, legs.sell_put["symbol"],
                            legs.sell_put["token"], curr_put,
                        )
                        log.warning("SL HIT on PUT @ ₹%.2f (SL=₹%.2f) P&L: %s", curr_put, sl_put, pnl)
                        trade_log.info("SL HIT | PUT %s @ %.2f | P&L=%s", legs.sell_put["symbol"], curr_put, pnl)
                        if pnl is not None:
                            self.risk.update_pnl(pnl)
                        put_closed = True
                except Exception:
                    log.exception("Error checking put SL")

            # Check sell call SL
            if not call_closed and self._sell_call_id:
                try:
                    curr_call = self.md.get_option_ltp(legs.sell_call["symbol"], legs.sell_call["token"])
                    if curr_call >= sl_call:
                        pnl = self.broker.close_position(
                            self._sell_call_id, legs.sell_call["symbol"],
                            legs.sell_call["token"], curr_call,
                        )
                        log.warning("SL HIT on CALL @ ₹%.2f (SL=₹%.2f) P&L: %s", curr_call, sl_call, pnl)
                        trade_log.info("SL HIT | CALL %s @ %.2f | P&L=%s", legs.sell_call["symbol"], curr_call, pnl)
                        if pnl is not None:
                            self.risk.update_pnl(pnl)
                        call_closed = True
                except Exception:
                    log.exception("Error checking call SL")

            if put_closed and call_closed:
                log.info("Both short legs closed via SL")
                break

        # Hard exit: close any remaining positions
        self._close_remaining(legs, put_closed, call_closed)

    def _close_remaining(self, legs: IronCondorLegs, put_closed: bool, call_closed: bool):
        """Close all remaining open positions at market."""
        remaining = []
        if not put_closed and self._sell_put_id:
            remaining.append(("SELL PUT", self._sell_put_id, legs.sell_put))
        if not call_closed and self._sell_call_id:
            remaining.append(("SELL CALL", self._sell_call_id, legs.sell_call))
        # Always close hedge legs
        if self._buy_put_id:
            remaining.append(("BUY PUT", self._buy_put_id, legs.buy_put))
        if self._buy_call_id:
            remaining.append(("BUY CALL", self._buy_call_id, legs.buy_call))

        for label, tid, info in remaining:
            try:
                curr = self.md.get_option_ltp(info["symbol"], info["token"])
                pnl = self.broker.close_position(tid, info["symbol"], info["token"], curr)
                log.info("EXIT %s %s @ ₹%.2f | P&L: %s", label, info["symbol"], curr, pnl)
                trade_log.info("EXIT | %s %s @ %.2f | P&L=%s", label, info["symbol"], curr, pnl)
                if pnl is not None:
                    self.risk.update_pnl(pnl)
            except Exception:
                log.exception("Failed to close %s %s", label, info["symbol"])
