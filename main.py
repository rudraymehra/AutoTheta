"""AutoTheta — Nifty Expiry Day OTM Premium Skew Trading Bot.

Entry point. Runs the strategy at 2:00 PM IST on Nifty expiry days (Tuesday).
"""

import logging
import logging.handlers
import sys

import pytz
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from config.settings import (
    ENTRY_HOUR,
    ENTRY_MINUTE,
    INITIAL_CAPITAL,
    LOGS_DIR,
    TRADING_MODE,
)
from src.auth import AngelSession
from src.broker import create_broker
from src.data import MarketData
from src.expiry import is_nifty_expiry_day
from src.instruments import InstrumentMaster
from src.paper_engine import PaperTradingEngine
from src.risk import RiskManager
from src.strategy import OTMSkewStrategy

IST = pytz.timezone("Asia/Kolkata")


def setup_logging() -> logging.Logger:
    """Configure console + rotating file + trade-specific loggers."""
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("autotheta")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Rotating system log (10MB, keep 5)
    fh = logging.handlers.RotatingFileHandler(
        LOGS_DIR / "system.log", maxBytes=10 * 1024 * 1024, backupCount=5,
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Trade-specific daily log
    trade_logger = logging.getLogger("autotheta.trades")
    th = logging.handlers.TimedRotatingFileHandler(
        LOGS_DIR / "trades.log", when="midnight", backupCount=90,
    )
    th.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    trade_logger.addHandler(th)

    return logger


log = setup_logging()


def execute_strategy():
    """Called by the scheduler at 2 PM IST on Mon-Tue."""
    # 1. Check if today is actually an expiry day
    if not is_nifty_expiry_day():
        log.info("Not an expiry day — skipping")
        return

    log.info("=" * 60)
    log.info("EXPIRY DAY — Starting AutoTheta strategy [mode=%s]", TRADING_MODE)
    log.info("=" * 60)

    # 2. Authenticate
    session = AngelSession()
    if not session.login():
        log.error("Authentication failed — aborting")
        return

    try:
        # 3. Load instruments
        instruments = InstrumentMaster()
        if not instruments.load():
            log.error("Instrument master load failed — aborting")
            return

        # 4. Initialize components
        market_data = MarketData(session.api)
        risk = RiskManager(capital=INITIAL_CAPITAL)

        if TRADING_MODE == "live":
            broker = create_broker(api=session.api)
        else:
            engine = PaperTradingEngine()
            broker = create_broker(paper_engine=engine)

        # 5. Run strategy
        strategy = OTMSkewStrategy(market_data, instruments, broker, risk)
        traded = strategy.run()

        if traded:
            log.info("Trade executed. Daily P&L: ₹%.2f", risk.daily_pnl)
        else:
            log.info("No trade taken today")

    except Exception:
        log.exception("Unhandled error in strategy execution")
    finally:
        session.logout()


def main():
    log.info("=" * 60)
    log.info("AutoTheta Bot v1.0 — Nifty OTM Premium Skew")
    log.info("Mode: %s | Capital: ₹%.0f", TRADING_MODE, INITIAL_CAPITAL)
    log.info("Schedule: %02d:%02d IST on Nifty expiry days (Tue)", ENTRY_HOUR, ENTRY_MINUTE)
    log.info("=" * 60)

    scheduler = BlockingScheduler(timezone=IST)
    # Trigger Mon-Tue to catch both regular Tuesday expiry and holiday-shifted Monday
    scheduler.add_job(
        execute_strategy,
        CronTrigger(day_of_week="mon-tue", hour=ENTRY_HOUR, minute=ENTRY_MINUTE, timezone=IST),
        id="autotheta_main",
        misfire_grace_time=300,
    )

    try:
        log.info("Scheduler started. Waiting for next expiry day...")
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Shutting down...")
        scheduler.shutdown()


if __name__ == "__main__":
    main()
