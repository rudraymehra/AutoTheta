import logging
from datetime import date, timedelta

import pandas_market_calendars as mcal

log = logging.getLogger("autotheta.expiry")

# NSE calendar from pandas_market_calendars
_nse_cal = mcal.get_calendar("NSE")


def get_trading_days(year: int) -> set[date]:
    """Get all NSE trading days for a given year."""
    schedule = _nse_cal.schedule(
        start_date=f"{year}-01-01", end_date=f"{year}-12-31"
    )
    return set(schedule.index.date)


def is_nifty_expiry_day(check_date: date | None = None) -> bool:
    """Check if a date is a Nifty weekly expiry day.

    Post Sep 2025: Weekly expiry is Tuesday.
    If Tuesday is a market holiday, expiry shifts to the previous trading day (typically Monday).
    """
    check_date = check_date or date.today()
    trading_days = get_trading_days(check_date.year)

    weekday = check_date.weekday()  # 0=Mon, 1=Tue

    # Case 1: It's Tuesday and it's a trading day → expiry day
    if weekday == 1 and check_date in trading_days:
        return True

    # Case 2: It's Monday (a trading day) and Tuesday is a holiday → shifted expiry
    if weekday == 0 and check_date in trading_days:
        tuesday = check_date + timedelta(days=1)
        if tuesday not in trading_days:
            log.info("Holiday-shifted expiry: Tuesday %s is a holiday, expiry moved to Monday %s",
                     tuesday, check_date)
            return True

    return False


def next_expiry_date(from_date: date | None = None) -> date:
    """Find the next Nifty expiry date from a given date."""
    current = from_date or date.today()
    trading_days = get_trading_days(current.year)

    # Look ahead up to 14 days
    for i in range(14):
        candidate = current + timedelta(days=i)
        if is_nifty_expiry_day(candidate):
            return candidate

    # If crossing year boundary, check next year too
    next_year_days = get_trading_days(current.year + 1)
    trading_days = trading_days | next_year_days
    for i in range(14, 21):
        candidate = current + timedelta(days=i)
        if is_nifty_expiry_day(candidate):
            return candidate

    log.error("Could not find next expiry within 21 days of %s", current)
    raise RuntimeError(f"No expiry found within 21 days of {current}")
