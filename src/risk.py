"""Risk management engine implementing the 12 survival rules.

Rules:
1. Never trade naked — always use iron condor
2. Risk max 1% of capital per trade
3. VIX filter: only trade when India VIX is 12-18
4. SL at 2x premium on each leg
5. Hard exit by 3:15 PM
6. Skip event days (Budget, RBI, elections, FOMC)
7. Daily loss cap at 3% of capital
8. Keep 20-30% margin buffer
9. Start with 1 lot only
10. Only enter when premium skew ratio >= 2:1
11. Use LIMIT orders only (SEBI mandate)
12. Paper trade 8+ weeks before going live
"""

import logging
from datetime import date, datetime

import pytz

from config.settings import (
    EXIT_HOUR,
    EXIT_MINUTE,
    MAX_LOSS_PER_DAY,
    MIN_SKEW_RATIO,
    SL_MULTIPLIER,
    VIX_MAX,
    VIX_MIN,
)

log = logging.getLogger("autotheta.risk")
IST = pytz.timezone("Asia/Kolkata")

# Known event dates for 2026 — update as announced
EVENT_DATES: set[date] = {
    # Union Budget
    date(2026, 2, 1),
    # RBI MPC dates (tentative — update when announced)
    date(2026, 2, 6),
    date(2026, 4, 9),
    date(2026, 6, 5),
    date(2026, 8, 7),
    date(2026, 10, 2),
    date(2026, 12, 4),
}


class RiskManager:
    """Evaluates risk rules before and during trades."""

    def __init__(self, capital: float, daily_pnl: float = 0.0):
        self.capital = capital
        self.daily_pnl = daily_pnl

    def check_vix(self, vix: float) -> tuple[bool, str]:
        """Rule 3: VIX must be within 12-18 range."""
        if vix < VIX_MIN:
            return False, f"VIX {vix:.1f} below {VIX_MIN} — complacency risk"
        if vix > VIX_MAX:
            return False, f"VIX {vix:.1f} above {VIX_MAX} — too much uncertainty"
        return True, f"VIX {vix:.1f} OK"

    def check_skew_ratio(self, put_premium: float, call_premium: float) -> tuple[bool, str]:
        """Rule 10: Premium skew ratio must be >= MIN_SKEW_RATIO."""
        denom = max(min(put_premium, call_premium), 0.05)
        ratio = max(put_premium, call_premium) / denom
        if ratio < MIN_SKEW_RATIO:
            return False, f"Skew ratio {ratio:.1f} below {MIN_SKEW_RATIO} threshold"
        return True, f"Skew ratio {ratio:.1f} OK"

    def check_daily_loss_cap(self) -> tuple[bool, str]:
        """Rule 7: Daily loss cap at 3% of capital (or MAX_LOSS_PER_DAY)."""
        cap = min(self.capital * 0.03, MAX_LOSS_PER_DAY)
        if self.daily_pnl <= -cap:
            return False, f"Daily loss cap hit: ₹{self.daily_pnl:.0f} (cap: ₹{-cap:.0f})"
        return True, f"Daily P&L ₹{self.daily_pnl:.0f} within cap"

    def check_event_day(self, check_date: date | None = None) -> tuple[bool, str]:
        """Rule 6: Skip known event days."""
        check_date = check_date or date.today()
        if check_date in EVENT_DATES:
            return False, f"{check_date} is a known event day — skipping"
        return True, "Not an event day"

    def check_exit_time(self) -> tuple[bool, str]:
        """Rule 5: Must exit by EXIT_HOUR:EXIT_MINUTE IST."""
        now = datetime.now(IST)
        cutoff_minutes = EXIT_HOUR * 60 + EXIT_MINUTE
        current_minutes = now.hour * 60 + now.minute
        if current_minutes >= cutoff_minutes:
            return False, f"Past hard exit time {EXIT_HOUR}:{EXIT_MINUTE:02d} IST"
        return True, f"Within trading window (exit at {EXIT_HOUR}:{EXIT_MINUTE:02d})"

    def calculate_sl(self, entry_premium: float) -> float:
        """Rule 4: Stop-loss at SL_MULTIPLIER × entry premium."""
        return round(entry_premium * SL_MULTIPLIER, 2)

    def check_position_size(self, max_loss_per_lot: float, lots: int) -> tuple[bool, str]:
        """Rule 2: Max risk per trade is 1% of capital."""
        max_risk = self.capital * 0.01
        total_risk = max_loss_per_lot * lots
        if total_risk > max_risk:
            max_lots = int(max_risk / max_loss_per_lot)
            return False, f"Risk ₹{total_risk:.0f} exceeds 1% cap ₹{max_risk:.0f}. Max lots: {max_lots}"
        return True, f"Risk ₹{total_risk:.0f} within 1% cap ₹{max_risk:.0f}"

    def pre_trade_checks(self, vix: float, put_premium: float, call_premium: float) -> tuple[bool, str]:
        """Run all pre-trade risk checks. Returns (can_trade, reason)."""
        checks = [
            self.check_vix(vix),
            self.check_skew_ratio(put_premium, call_premium),
            self.check_daily_loss_cap(),
            self.check_event_day(),
            self.check_exit_time(),
        ]
        for ok, msg in checks:
            if not ok:
                log.warning("Pre-trade check failed: %s", msg)
                return False, msg
            log.info("Pre-trade check passed: %s", msg)
        return True, "All checks passed"

    def update_pnl(self, pnl: float):
        """Update running daily P&L after a trade closes."""
        self.daily_pnl += pnl
