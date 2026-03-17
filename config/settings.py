import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _require(key: str) -> str:
    val = os.getenv(key)
    if not val:
        sys.exit(f"FATAL: Missing required env var: {key}. See .env.example")
    return val


# ── Angel One Credentials ──
ANGEL_API_KEY = _require("ANGEL_API_KEY")
ANGEL_CLIENT_ID = _require("ANGEL_CLIENT_ID")
ANGEL_PASSWORD = _require("ANGEL_PASSWORD")
ANGEL_TOTP_SECRET = _require("ANGEL_TOTP_SECRET")

# ── Trading Mode ──
TRADING_MODE = os.getenv("TRADING_MODE", "paper").lower()  # "paper" or "live"

# ── Capital & Risk ──
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "500000"))
MAX_LOSS_PER_DAY = float(os.getenv("MAX_LOSS_PER_DAY", "15000"))
SLIPPAGE_PCT = float(os.getenv("SLIPPAGE_PCT", "0.05"))

# ── Strategy Parameters ──
MIN_SKEW_RATIO = float(os.getenv("MIN_SKEW_RATIO", "2.0"))
OTM_OFFSET = int(os.getenv("OTM_OFFSET", "50"))
WING_WIDTH = int(os.getenv("WING_WIDTH", "100"))
SL_MULTIPLIER = float(os.getenv("SL_MULTIPLIER", "2.0"))

# ── VIX Filter ──
VIX_MIN = float(os.getenv("VIX_MIN", "12"))
VIX_MAX = float(os.getenv("VIX_MAX", "18"))

# ── Schedule (IST) ──
ENTRY_HOUR = int(os.getenv("ENTRY_HOUR", "14"))
ENTRY_MINUTE = int(os.getenv("ENTRY_MINUTE", "0"))
EXIT_HOUR = int(os.getenv("EXIT_HOUR", "15"))
EXIT_MINUTE = int(os.getenv("EXIT_MINUTE", "15"))

# ── Monitoring ──
MONITOR_INTERVAL = int(os.getenv("MONITOR_INTERVAL", "30"))

# ── Constants ──
NIFTY_LOT_SIZE = 65  # Changed from 75 in Jan 2026
NIFTY_SPOT_TOKEN = "99926000"
INDIA_VIX_TOKEN = "99926017"  # 99926004 is Nifty 500, NOT VIX
NIFTY_STRIKE_GAP = 50  # Nifty strikes are 50 points apart

# ── Instrument Master URL ──
INSTRUMENT_MASTER_URL = (
    "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
)

# ── Paths ──
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
DB_PATH = DATA_DIR / "trades.db"
INSTRUMENT_CACHE_PATH = DATA_DIR / "instruments.json"
