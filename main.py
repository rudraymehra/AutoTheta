"""AutoTheta — Multi-Strategy Nifty Trading Bot.

Strategies:
1. RSI Oversold Bounce on Nifty 50 stocks (intraday, every trading day)
2. OTM Premium Skew Iron Condor on Nifty expiry (Tuesday)

Both run concurrently via asyncio through a single Angel One SmartAPI session.
"""

import asyncio
import logging
import logging.handlers
import os
import signal
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Load env vars before anything else
load_dotenv()

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure console + rotating file + trade-specific loggers."""
    logger = logging.getLogger("autotheta")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

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


def load_config(path: str = "config/config.yaml") -> dict:
    """Load YAML config with env var interpolation."""
    with open(path) as f:
        raw = f.read()

    # Replace ${VAR} with env values
    for key, value in os.environ.items():
        raw = raw.replace(f"${{{key}}}", value)

    return yaml.safe_load(raw)


async def main():
    config = load_config()
    log = setup_logging(config["bot"].get("log_level", "INFO"))

    log.info("=" * 60)
    log.info("AutoTheta v2.0 — Multi-Strategy Trading Bot")
    log.info("Mode: %s", "PAPER" if config["bot"].get("paper_mode", True) else "LIVE")
    enabled = [s["strategy"]["name"] for s in config["strategies"]
               if s["strategy"].get("enabled", True)]
    log.info("Strategies: %s", ", ".join(enabled))
    log.info("=" * 60)

    # Import here to avoid circular imports at module level
    from core.engine import TradingEngine

    engine = TradingEngine(config)

    # Graceful shutdown on SIGINT/SIGTERM
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, engine.shutdown)

    await engine.start()


if __name__ == "__main__":
    asyncio.run(main())
