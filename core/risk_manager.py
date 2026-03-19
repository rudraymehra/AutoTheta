"""Global risk manager shared across all strategies.

Enforces:
- Global daily loss cap across all strategies
- Per-strategy daily loss cap
- Per-strategy capital allocation
- Total exposure limit
- Cooldown after consecutive losses
- Market-wide selloff detection
"""

import logging
from datetime import date, datetime, timedelta
from collections import defaultdict

import pytz

from models.types import Position, Side

log = logging.getLogger("autotheta.risk")
IST = pytz.timezone("Asia/Kolkata")

# Known event dates — update as announced
EVENT_DATES: set[date] = {
    date(2026, 2, 1),   # Union Budget
    date(2026, 2, 6),   # RBI MPC
    date(2026, 4, 9),   # RBI MPC
    date(2026, 6, 5),   # RBI MPC
    date(2026, 8, 7),   # RBI MPC
    date(2026, 10, 2),  # RBI MPC
    date(2026, 12, 4),  # RBI MPC
}


class GlobalRiskManager:
    """Tracks risk across all strategies from a single point."""

    def __init__(self, config: dict):
        self.max_daily_loss = config.get("max_daily_loss", 15000)
        self.max_total_exposure = config.get("max_total_exposure", 500000)

        # Per-strategy state
        self._strategy_config: dict[str, dict] = {}
        self._strategy_pnl: dict[str, float] = defaultdict(float)
        self._strategy_positions: dict[str, list[Position]] = defaultdict(list)
        self._consecutive_losses: dict[str, int] = defaultdict(int)
        self._cooldown_until: dict[str, datetime | None] = defaultdict(lambda: None)

        # Global state
        self._global_pnl = 0.0
        self._today: date = date.today()

    def register_strategy(self, name: str, risk_config: dict):
        """Register a strategy's risk parameters."""
        self._strategy_config[name] = {
            "capital_allocation": risk_config.get("capital_allocation", 250000),
            "risk_per_trade_pct": risk_config.get("risk_per_trade_pct", 1.0),
            "daily_loss_cap": risk_config.get("daily_loss_cap", 7500),
            "max_positions": risk_config.get("max_positions", 4),
            "max_consecutive_losses": risk_config.get("max_consecutive_losses", 3),
            "cooldown_minutes": risk_config.get("cooldown_minutes", 30),
        }
        log.info("Registered risk config for %s: %s", name, self._strategy_config[name])

    def _reset_if_new_day(self):
        today = date.today()
        if today != self._today:
            log.info("New day detected — resetting daily risk counters")
            self._global_pnl = 0.0
            self._strategy_pnl.clear()
            self._consecutive_losses.clear()
            self._cooldown_until.clear()
            self._today = today

    def can_trade(self, strategy_name: str) -> tuple[bool, str]:
        """Check if a strategy is allowed to take a new trade."""
        self._reset_if_new_day()

        # Global daily loss cap
        if self._global_pnl <= -self.max_daily_loss:
            return False, f"Global daily loss cap hit: ₹{self._global_pnl:.0f}"

        cfg = self._strategy_config.get(strategy_name, {})
        if not cfg:
            return False, f"Strategy {strategy_name} not registered"

        # Per-strategy daily loss cap
        strat_pnl = self._strategy_pnl[strategy_name]
        strat_cap = cfg["daily_loss_cap"]
        if strat_pnl <= -strat_cap:
            return False, f"{strategy_name} daily loss cap: ₹{strat_pnl:.0f} (cap: ₹{strat_cap})"

        # Cooldown check
        cooldown = self._cooldown_until[strategy_name]
        if cooldown and datetime.now(IST) < cooldown:
            remaining = (cooldown - datetime.now(IST)).seconds // 60
            return False, f"{strategy_name} in cooldown for {remaining}m more"

        # Max positions check
        open_positions = len(self._strategy_positions[strategy_name])
        max_pos = cfg["max_positions"]
        if open_positions >= max_pos:
            return False, f"{strategy_name} at max positions: {open_positions}/{max_pos}"

        # Event day check
        if date.today() in EVENT_DATES:
            return False, f"Event day: {date.today()}"

        # Total exposure check
        total_exposure = self._calculate_total_exposure()
        if total_exposure >= self.max_total_exposure:
            return False, f"Total exposure ₹{total_exposure:.0f} >= limit ₹{self.max_total_exposure:.0f}"

        return True, "OK"

    def calculate_position_size(self, strategy_name: str, entry_price: float,
                                stop_loss: float) -> int:
        """Calculate shares based on ATR risk sizing. Returns quantity."""
        cfg = self._strategy_config.get(strategy_name, {})
        if not cfg:
            return 0

        capital = cfg["capital_allocation"]
        risk_pct = cfg["risk_per_trade_pct"] / 100
        risk_amount = capital * risk_pct  # e.g., ₹2,500

        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            return 0

        shares = int(risk_amount / risk_per_share)

        # Cap position value at 1/3 of allocation
        max_value = capital / 3
        max_shares_by_value = int(max_value / entry_price)
        shares = min(shares, max_shares_by_value)

        return max(shares, 0)

    def record_trade_result(self, strategy_name: str, pnl: float):
        """Record a completed trade's P&L and update counters."""
        self._reset_if_new_day()
        self._strategy_pnl[strategy_name] += pnl
        self._global_pnl += pnl

        if pnl < 0:
            self._consecutive_losses[strategy_name] += 1
            cfg = self._strategy_config.get(strategy_name, {})
            max_losses = cfg.get("max_consecutive_losses", 3)
            cooldown_mins = cfg.get("cooldown_minutes", 30)
            if self._consecutive_losses[strategy_name] >= max_losses:
                until = datetime.now(IST) + timedelta(minutes=cooldown_mins)
                self._cooldown_until[strategy_name] = until
                log.warning("%s: %d consecutive losses — cooldown until %s",
                            strategy_name, self._consecutive_losses[strategy_name],
                            until.strftime("%H:%M"))
        else:
            self._consecutive_losses[strategy_name] = 0

        log.info("P&L recorded: %s ₹%.2f | Strategy total: ₹%.2f | Global: ₹%.2f",
                 strategy_name, pnl, self._strategy_pnl[strategy_name], self._global_pnl)

    def add_position(self, strategy_name: str, position: Position):
        self._strategy_positions[strategy_name].append(position)

    def remove_position(self, strategy_name: str, trade_id: str):
        self._strategy_positions[strategy_name] = [
            p for p in self._strategy_positions[strategy_name]
            if p.trade_id != trade_id
        ]

    def get_open_positions(self, strategy_name: str) -> list[Position]:
        return list(self._strategy_positions[strategy_name])

    def _calculate_total_exposure(self) -> float:
        total = 0.0
        for positions in self._strategy_positions.values():
            for pos in positions:
                total += pos.entry_price * pos.remaining_quantity
        return total

    def get_summary(self) -> dict:
        return {
            "global_pnl": self._global_pnl,
            "strategy_pnl": dict(self._strategy_pnl),
            "open_positions": {k: len(v) for k, v in self._strategy_positions.items()},
            "total_exposure": self._calculate_total_exposure(),
        }
