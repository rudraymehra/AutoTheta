"""Paper trading with LIVE WebSocket data — v2.1 Filter Stack Redesign.

Uses SmartWebSocketV2 for real-time ticks → builds 1-min candles → runs strategies.
No historical API needed = no rate limit issues.

S1: RSI(5) Mean Reversion on 5-min candles
  - Entry: RSI(5) < 20 with uptick confirmation (prev < 20, curr > prev)
  - Filters: Daily 2-of-3 regime check, KER(10)<0.30, below VWAP, MFI(8)<30
  - Exit: 60% at VWAP touch, 40% at RSI>50, 75-min time stop, 3x ATR disaster stop
  - Time: 10:15 AM - 2:00 PM entries, hard exit 3:00 PM

Generates two log files daily in logs/YYYY-MM-DD/:
  1. thoughts.csv — Every signal the bot considered + why it acted or didn't
  2. trades.csv   — Only actual paper trades (entries and exits)
"""

import csv
import json
import os
import signal
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, date, timedelta
from pathlib import Path

import pandas as pd
import pyotp
from SmartApi import SmartConnect
from SmartApi.smartWebSocketV2 import SmartWebSocketV2
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

from strategies.rsi_15min import scan_15min_rsi, reset_state as reset_s3_state, get_positions as get_s3_positions, get_daily_pnl as get_s3_pnl

API_KEY = os.getenv("ANGEL_API_KEY")
CLIENT_ID = os.getenv("ANGEL_CLIENT_ID")
PASSWORD = os.getenv("ANGEL_PASSWORD")
TOTP_SECRET = os.getenv("ANGEL_TOTP_SECRET")


# ── Indicators ──
def rsi(series, period=5):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def rsi_calc(series, period=14):
    """RSI with configurable period — used for daily regime check."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def atr_calc(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def vwap_calc(df):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (tp * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum().replace(0, 1)
    return cum_tp_vol / cum_vol


def adx_calc(high, low, close, period=14):
    """Average Directional Index — measures trend strength."""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_smooth = tr.ewm(alpha=1 / period, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr_smooth.replace(0, 1e-10)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr_smooth.replace(0, 1e-10)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10) * 100
    return dx.ewm(alpha=1 / period, min_periods=period).mean()


def kaufman_er(series, period=10):
    """Kaufman Efficiency Ratio. 0=pure chop, 1=perfect trend."""
    direction = abs(series - series.shift(period))
    volatility = series.diff().abs().rolling(period).sum()
    return direction / volatility.replace(0, 1e-10)


def mfi(high, low, close, volume, period=8):
    """Money Flow Index — RSI with volume."""
    tp = (high + low + close) / 3
    mf = tp * volume
    pos_mf = mf.where(tp > tp.shift(1), 0.0).rolling(period).sum()
    neg_mf = mf.where(tp < tp.shift(1), 0.0).rolling(period).sum()
    mr = pos_mf / neg_mf.replace(0, 1e-10)
    return 100 - (100 / (1 + mr))


# ── Config ──
STOCKS = [
    "HDFCBANK-EQ", "ICICIBANK-EQ", "KOTAKBANK-EQ", "AXISBANK-EQ", "SBIN-EQ",
    "INFY-EQ", "TCS-EQ", "HCLTECH-EQ", "WIPRO-EQ",
    "ITC-EQ", "HINDUNILVR-EQ",
    "SUNPHARMA-EQ",
    "BAJFINANCE-EQ",
    "LT-EQ", "TITAN-EQ", "BHARTIARTL-EQ", "MARUTI-EQ",
]

SECTOR_MAP = {
    "HDFCBANK-EQ": "Banking", "ICICIBANK-EQ": "Banking", "KOTAKBANK-EQ": "Banking",
    "SBIN-EQ": "Banking", "AXISBANK-EQ": "Banking",
    "BAJFINANCE-EQ": "Finance",
    "TCS-EQ": "IT", "INFY-EQ": "IT", "WIPRO-EQ": "IT", "HCLTECH-EQ": "IT",
    "ITC-EQ": "FMCG", "HINDUNILVR-EQ": "FMCG",
    "SUNPHARMA-EQ": "Pharma",
    "LT-EQ": "Infra", "BHARTIARTL-EQ": "Telecom",
    "TITAN-EQ": "Other", "MARUTI-EQ": "Auto",
}

# S1 v2.1: RSI(5) on 5-min candles with uptick confirmation
RSI_PERIOD = 5
RSI_OVERSOLD = 20              # RSI must be below 20
RSI_EXIT_PARTIAL = 50          # 40% position exits at RSI > 50
ATR_SL_MULT = 3.0              # 3x ATR on 15-min = disaster stop only
TIME_STOP_MINUTES = 75         # 75-minute time stop
RISK_PER_TRADE = 2500          # 1% of Rs 2.5L
MAX_POSITIONS = 3
MAX_PER_SECTOR = 1
CAPITAL = 250000

# VIX regime sizing (checked at startup)
VIX_FULL_THRESHOLD = 18        # VIX < 18: full size
VIX_HALF_THRESHOLD = 22        # VIX 18-22: half size
# VIX > 22: no trades

# Time windows
ENTRY_START_HOUR, ENTRY_START_MIN = 10, 15   # 10:15 AM
ENTRY_END_HOUR, ENTRY_END_MIN = 14, 0        # 2:00 PM
HARD_EXIT_HOUR, HARD_EXIT_MIN = 15, 0        # 3:00 PM

# KER filter on 15-min (replaces ADX)
KER_PERIOD = 10
KER_MAX = 0.30                 # Only trade when KER < 0.30 (choppy/mean-reverting)

# MFI confirmation
MFI_PERIOD = 8
MFI_THRESHOLD = 30             # MFI < 30 as volume confirmation


def fetch_vix(api):
    """Fetch India VIX at startup via REST API. Returns VIX value or None."""
    try:
        # India VIX token on NSE
        data = api.ltpData("NSE", "India VIX", "26017")
        if data and data.get("data"):
            ltp = data["data"].get("ltp", 0)
            if isinstance(ltp, (int, float)):
                return ltp / 100 if ltp > 100 else ltp
    except Exception:
        pass
    return None


# ══════════════════════════════════════════
# DAILY LOGGER — two CSV files
# ══════════════════════════════════════════
class DailyLogger:
    """Writes two CSV logs per day:
    - thoughts.csv: Every signal considered (what the bot saw and why it acted/didn't)
    - trades.csv:   Only actual entries and exits
    """

    def __init__(self):
        self._today = None
        self._thoughts_writer = None
        self._trades_writer = None
        self._thoughts_file = None
        self._trades_file = None
        self._ensure_files()

    def _ensure_files(self):
        today = date.today().isoformat()
        if today == self._today:
            return

        # Close previous files
        if self._thoughts_file:
            self._thoughts_file.close()
        if self._trades_file:
            self._trades_file.close()

        self._today = today
        log_dir = PROJECT_ROOT / "logs" / today
        log_dir.mkdir(parents=True, exist_ok=True)

        # Thoughts log — what the bot considered
        thoughts_path = log_dir / "thoughts.csv"
        is_new = not thoughts_path.exists()
        self._thoughts_file = open(thoughts_path, "a", newline="")
        self._thoughts_writer = csv.writer(self._thoughts_file)
        if is_new:
            self._thoughts_writer.writerow([
                "Time", "Stock", "Price", "RSI(5)_5m", "Signal",
                "DailyRegime", "VWAP", "VWAP_dist",
                "KER10_15m", "MFI8",
                "Decision", "Reason",
            ])

        # Trades log — what the bot actually did
        trades_path = log_dir / "trades.csv"
        is_new = not trades_path.exists()
        self._trades_file = open(trades_path, "a", newline="")
        self._trades_writer = csv.writer(self._trades_file)
        if is_new:
            self._trades_writer.writerow([
                "Time", "Action", "Stock", "Qty", "Price",
                "RSI", "Stop_Loss", "Reason", "P&L",
            ])

    def log_thought(self, stock, price, rsi_val, signal_type,
                    regime_ok, vwap_val, vwap_dist, ker_val,
                    decision, reason, mfi_val=None):
        """Log what the bot saw and why it decided what it did."""
        self._ensure_files()
        now = datetime.now().strftime("%H:%M:%S")

        regime_str = "PASS" if regime_ok else ("FAIL" if regime_ok is not None and not regime_ok else "N/A")

        self._thoughts_writer.writerow([
            now, stock, f"{price:.2f}", f"{rsi_val:.1f}", signal_type,
            regime_str,
            f"{vwap_val:.2f}" if vwap_val else "N/A",
            f"{vwap_dist:.4f}" if vwap_dist is not None else "N/A",
            f"{ker_val:.3f}" if ker_val is not None else "N/A",
            f"{mfi_val:.1f}" if mfi_val is not None else "N/A",
            decision, reason,
        ])
        self._thoughts_file.flush()

    def log_trade(self, action, stock, qty, price, rsi_val=0,
                  stop_loss=0, reason="", pnl=0):
        """Log an actual trade action."""
        self._ensure_files()
        now = datetime.now().strftime("%H:%M:%S")
        self._trades_writer.writerow([
            now, action, stock, qty, f"{price:.2f}",
            f"{rsi_val:.1f}" if rsi_val else "",
            f"{stop_loss:.2f}" if stop_loss else "",
            reason,
            f"{pnl:+.2f}" if pnl else "",
        ])
        self._trades_file.flush()

    def close(self):
        if self._thoughts_file:
            self._thoughts_file.close()
        if self._trades_file:
            self._trades_file.close()


# ══════════════════════════════════════════
# CANDLE BUILDER — ticks → 1-min OHLCV
# ══════════════════════════════════════════
class CandleBuilder:
    def __init__(self):
        self._building = defaultdict(dict)
        self._candles = defaultdict(list)
        self._lock = threading.Lock()

    def on_tick(self, token: str, ltp: float, volume: int):
        now = datetime.now()
        minute_key = now.strftime("%Y-%m-%d %H:%M")

        with self._lock:
            buckets = self._building[token]
            for mk in list(buckets.keys()):
                if mk != minute_key:
                    candle = buckets.pop(mk)
                    candle["timestamp"] = pd.Timestamp(mk)
                    self._candles[token].append(candle)
                    if len(self._candles[token]) > 400:
                        self._candles[token] = self._candles[token][-400:]

            if minute_key not in buckets:
                buckets[minute_key] = {
                    "open": ltp, "high": ltp, "low": ltp, "close": ltp, "volume": volume,
                }
            else:
                c = buckets[minute_key]
                c["high"] = max(c["high"], ltp)
                c["low"] = min(c["low"], ltp)
                c["close"] = ltp
                c["volume"] = volume

    def get_df(self, token: str) -> pd.DataFrame | None:
        with self._lock:
            candles = self._candles.get(token, [])
            if len(candles) < 10:
                return None
            return pd.DataFrame(candles)

    def candle_count(self, token: str) -> int:
        with self._lock:
            return len(self._candles.get(token, []))


# ══════════════════════════════════════════
# PAPER PORTFOLIO
# ══════════════════════════════════════════
class PaperPortfolio:
    def __init__(self, capital, logger: DailyLogger):
        self.capital = capital
        self.positions = {}
        self.closed_trades = []
        self.daily_pnl = 0.0
        self.sector_count = defaultdict(int)
        self.log = logger

    def open_position(self, trade_id, symbol, price, quantity, stop_loss, rsi_val):
        sector = SECTOR_MAP.get(symbol, "Other")
        self.positions[trade_id] = {
            "symbol": symbol, "entry_price": price, "quantity": quantity,
            "initial_qty": quantity, "remaining": quantity,
            "stop_loss": stop_loss, "status": "open",
            "entry_time": datetime.now(), "entry_timestamp": datetime.now(),
            "vwap_exit_done": False,
            "entry_rsi": rsi_val,
        }
        self.sector_count[sector] += 1
        self.log.log_trade("BUY", symbol, quantity, price, rsi_val, stop_loss, "RSI5_UPTICK")
        print(f"\n  >> PAPER BUY {symbol} x{quantity} @ Rs{price:.2f} | "
              f"SL=Rs{stop_loss:.2f} | RSI(5)={rsi_val:.1f}")

    def close_position(self, trade_id, price, quantity, reason):
        pos = self.positions.get(trade_id)
        if not pos:
            return
        pnl = (price - pos["entry_price"]) * quantity
        pos["realized_pnl"] = pos.get("realized_pnl", 0.0) + pnl
        pos["remaining"] -= quantity
        self.daily_pnl += pnl

        self.log.log_trade("SELL", pos["symbol"], quantity, price, reason=reason, pnl=pnl)

        emoji = "+" if pnl >= 0 else ""
        print(f"\n  << PAPER SELL {pos['symbol']} x{quantity} @ Rs{price:.2f} | "
              f"reason={reason} | P&L=Rs{emoji}{pnl:.2f}")

        if pos["remaining"] <= 0:
            sector = SECTOR_MAP.get(pos["symbol"], "Other")
            self.sector_count[sector] = max(0, self.sector_count[sector] - 1)
            self.closed_trades.append({
                **pos, "exit_price": price, "exit_time": datetime.now(),
                "exit_reason": reason, "total_pnl": pos.get("realized_pnl", 0.0),
            })
            del self.positions[trade_id]

    def summary(self):
        pnl_pct = (self.daily_pnl / self.capital) * 100

        print(f"\n  {'='*60}")
        print(f"  PAPER TRADING SUMMARY — {date.today().isoformat()}")
        print(f"  {'='*60}")
        print(f"  Starting Capital:  Rs{self.capital:,.2f}")
        print(f"  Daily P&L:         Rs{self.daily_pnl:+,.2f} ({pnl_pct:+.2f}%)")
        print(f"  Closing Capital:   Rs{self.capital + self.daily_pnl:,.2f}")
        print(f"  {'─'*60}")
        print(f"  Closed trades:     {len(self.closed_trades)}")
        if self.closed_trades:
            wins = [t for t in self.closed_trades if t["total_pnl"] > 0]
            losses = [t for t in self.closed_trades if t["total_pnl"] <= 0]
            win_rate = len(wins) / len(self.closed_trades) * 100
            print(f"  Wins/Losses:       {len(wins)}/{len(losses)} ({win_rate:.0f}% win rate)")
            if wins:
                print(f"  Avg Win:           Rs{sum(t['total_pnl'] for t in wins)/len(wins):+,.2f}")
            if losses:
                print(f"  Avg Loss:          Rs{sum(t['total_pnl'] for t in losses)/len(losses):+,.2f}")
            # Per-trade breakdown
            print(f"\n  {'─'*60}")
            print(f"  {'':2s}{'Stock':15s} {'Entry':>10s} {'Exit':>10s} {'Qty':>6s} {'Reason':>12s} {'P&L':>12s} {'%':>8s}")
            print(f"  {'─'*60}")
            for t in self.closed_trades:
                t_pnl_pct = t["total_pnl"] / (t["entry_price"] * t["initial_qty"]) * 100
                marker = "W" if t["total_pnl"] > 0 else "L"
                print(f"  [{marker}] {t['symbol']:15s} {t['entry_price']:>10.2f} {t['exit_price']:>10.2f} "
                      f"{t['initial_qty']:>5d}  {t['exit_reason']:>12s} {t['total_pnl']:>+11,.2f} {t_pnl_pct:>+7.2f}%")
        else:
            print(f"  (no trades today)")

        if self.positions:
            print(f"\n  Open positions:")
            for tid, pos in self.positions.items():
                elapsed = (datetime.now() - pos["entry_timestamp"]).total_seconds() / 60
                print(f"    [OPEN] {pos['symbol']:15s} Rs{pos['entry_price']:.2f} "
                      f"x{pos['remaining']} | {elapsed:.0f}min elapsed")

        print(f"\n  Logs: logs/{date.today().isoformat()}/")
        print(f"    thoughts.csv  — what the bot considered")
        print(f"    trades.csv    — actual paper trades")
        print(f"  {'='*60}")


# ══════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════
def main():
    logger = DailyLogger()
    reset_s3_state()  # Clear Strategy 3 state for new day

    print("=" * 70)
    print(f"  AutoTheta PAPER TRADING v2.1 — Filter Stack Redesign")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} IST")
    print(f"  Capital: Rs{CAPITAL:,}")
    print(f"  S1: RSI(5) on 5-min | Entry: <20 uptick | Daily regime + KER + MFI")
    print(f"  S3: RSI(9) on 15-min setup + 5-min trigger + KER + MFI")
    print("=" * 70)

    # Authenticate
    api = SmartConnect(API_KEY)
    totp = pyotp.TOTP(TOTP_SECRET).now()
    session = api.generateSession(CLIENT_ID, PASSWORD, totp)
    if not session.get("status"):
        print(f"AUTH FAILED: {session}")
        return
    auth_token = session["data"]["jwtToken"]
    feed_token = api.getfeedToken()
    print("[OK] Authenticated")

    # ── VIX regime check ──
    vix = fetch_vix(api)
    if vix is not None:
        print(f"[OK] India VIX: {vix:.2f}")
        if vix > VIX_HALF_THRESHOLD:
            print(f"[!!] VIX > {VIX_HALF_THRESHOLD} — NO TRADES TODAY (high volatility regime)")
            vix_size_mult = 0.0
        elif vix > VIX_FULL_THRESHOLD:
            print(f"[..] VIX {VIX_FULL_THRESHOLD}-{VIX_HALF_THRESHOLD} — HALF SIZE (Rs 1,250 risk per trade)")
            vix_size_mult = 0.5
        else:
            print(f"[OK] VIX < {VIX_FULL_THRESHOLD} — FULL SIZE")
            vix_size_mult = 1.0
    else:
        print("[..] Could not fetch VIX — defaulting to full size")
        vix_size_mult = 1.0

    if vix_size_mult == 0.0:
        print("\n  VIX too high. Bot will monitor but not trade.")

    # ── Load token map (needed for daily regime check) ──
    instruments_path = PROJECT_ROOT / "data" / "instruments.json"
    if not instruments_path.exists():
        print("[..] Downloading instrument master...")
        import requests
        url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        resp = requests.get(url, timeout=120)
        instruments_path.parent.mkdir(parents=True, exist_ok=True)
        with open(instruments_path, "w") as f:
            f.write(resp.text)
        master_data = resp.json()
    else:
        with open(instruments_path) as f:
            master_data = json.load(f)

    master_df = pd.DataFrame(master_data)

    token_map = {}
    token_to_sym = {}
    for sym in STOCKS:
        matches = master_df[(master_df["symbol"] == sym) & (master_df["exch_seg"] == "NSE")]
        if not matches.empty:
            tok = matches.iloc[0]["token"]
            token_map[sym] = tok
            token_to_sym[tok] = sym

    print(f"[OK] {len(token_map)} stocks mapped")

    # ── Daily Regime Check (2-of-3: EMA proximity, RSI range, ADX) ──
    daily_regime = {}
    print("[..] Running daily regime check (REST API)...")
    for sym, tok in token_map.items():
        time.sleep(0.5)
        try:
            r = api.getCandleData({
                "exchange": "NSE", "symboltoken": tok, "interval": "ONE_DAY",
                "fromdate": "2025-06-01 09:15",
                "todate": datetime.now().strftime("%Y-%m-%d %H:%M"),
            })
            if r and r.get("data") and len(r["data"]) > 50:
                ddf = pd.DataFrame(r["data"], columns=["timestamp", "open", "high", "low", "close", "volume"])
                ddf["close"] = pd.to_numeric(ddf["close"], errors="coerce")
                ddf["high"] = pd.to_numeric(ddf["high"], errors="coerce")
                ddf["low"] = pd.to_numeric(ddf["low"], errors="coerce")
                ddf["ema200"] = ema(ddf["close"], 200)
                ddf["rsi14"] = rsi_calc(ddf["close"], 14)
                ddf["adx14"] = adx_calc(ddf["high"], ddf["low"], ddf["close"], 14)
                last = ddf.iloc[-1]

                checks_passed = 0
                # Check 1: within 8% of daily EMA(200)
                if pd.notna(last["ema200"]):
                    if abs(last["close"] - last["ema200"]) / last["ema200"] < 0.08:
                        checks_passed += 1
                # Check 2: daily RSI(14) between 30 and 65
                if pd.notna(last["rsi14"]):
                    if 30 <= last["rsi14"] <= 65:
                        checks_passed += 1
                # Check 3: daily ADX(14) < 25
                if pd.notna(last["adx14"]):
                    if last["adx14"] < 25:
                        checks_passed += 1

                daily_regime[sym] = checks_passed >= 2
                status = "PASS" if daily_regime[sym] else "FAIL"
                print(f"    {sym:18s} {status} ({checks_passed}/3 conditions met)")
            else:
                daily_regime[sym] = True  # Default allow if no data
                print(f"    {sym:18s} PASS (insufficient data, default allow)")
        except Exception as e:
            daily_regime[sym] = True  # Default allow on error
            print(f"    {sym:18s} PASS (error: {e}, default allow)")

    regime_pass = sum(1 for v in daily_regime.values() if v)
    print(f"[OK] Daily regime: {regime_pass}/{len(daily_regime)} stocks passed")

    # Initialize
    candle_builder = CandleBuilder()
    portfolio = PaperPortfolio(CAPITAL, logger)
    trade_counter = [0]
    running = [True]

    # WebSocket callbacks
    def on_data(wsapp, msg):
        try:
            token = str(msg.get("token", ""))
            ltp = msg.get("last_traded_price", 0)
            if isinstance(ltp, (int, float)):
                ltp = ltp / 100
            volume = msg.get("volume_trade_for_the_day", 0)
            if token and ltp > 0:
                candle_builder.on_tick(token, ltp, volume)
        except Exception:
            pass

    def on_open(wsapp):
        tokens = list(token_map.values())
        token_list = [{"exchangeType": 1, "tokens": tokens}]
        # Use ws_state["sws"] instead of closed-over variable
        ws_state["sws"].subscribe("autotheta_paper", 2, token_list)
        print(f"\n[OK] WebSocket connected — {len(tokens)} stocks streaming")
        print(f"[OK] Scanning starts after ~10 candles (~10 min)")
        print(f"     Logs: logs/{date.today().isoformat()}/thoughts.csv & trades.csv\n")

    def on_error(wsapp, error):
        print(f"\n  [WS ERROR] {error}")

    def on_close(wsapp):
        print("\n  [WS] Connection closed — will auto-reconnect")

    print("[OK] Pure WebSocket mode — no API rate limits")

    # ── WebSocket with auto-reconnect ──
    ws_state = {"sws": None, "last_candle_time": time.time(), "reconnecting": False}

    def start_websocket():
        """Create and start a new WebSocket connection."""
        # Re-auth to get fresh tokens (old ones may have expired)
        try:
            totp_now = pyotp.TOTP(TOTP_SECRET).now()
            sess = api.generateSession(CLIENT_ID, PASSWORD, totp_now)
            if sess.get("status"):
                fresh_auth = sess["data"]["jwtToken"]
                fresh_feed = api.getfeedToken()
            else:
                fresh_auth = auth_token
                fresh_feed = feed_token
        except Exception:
            fresh_auth = auth_token
            fresh_feed = feed_token

        sws = SmartWebSocketV2(
            fresh_auth, API_KEY, CLIENT_ID, fresh_feed,
            max_retry_attempt=50, retry_strategy=0, retry_delay=5,
        )
        sws.on_data = on_data
        sws.on_open = on_open
        sws.on_error = on_error
        sws.on_close = on_close
        ws_state["sws"] = sws

        t = threading.Thread(target=sws.connect, daemon=True)
        t.start()
        return t

    ws_thread = start_websocket()

    # Graceful shutdown
    def shutdown(sig, frame):
        running[0] = False
        print("\n\n  Shutting down...")

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # ── Strategy loop ──
    last_scan = time.time() - 55
    tick_count = 0
    last_candle_count = 0

    while running[0]:
        time.sleep(1)
        tick_count += 1

        # Status bar every 10s
        if tick_count % 10 == 0:
            candle_counts = {token_to_sym.get(t, t): candle_builder.candle_count(t)
                            for t in token_map.values()}
            max_candles = max(candle_counts.values()) if candle_counts else 0
            active = sum(1 for c in candle_counts.values() if c > 0)
            now = datetime.now()
            s3_positions = get_s3_positions()
            total_positions = len(portfolio.positions) + len(s3_positions)
            combined_pnl = portfolio.daily_pnl + get_s3_pnl()
            pnl_pct = (combined_pnl / portfolio.capital) * 100
            vix_tag = f" VIX={vix:.1f}" if vix else ""
            print(f"\r  [{now.strftime('%H:%M:%S')}] "
                  f"Candles: {max_candles} | Feeds: {active}/{len(token_map)} | "
                  f"Pos: S1={len(portfolio.positions)} S3={len(s3_positions)} | "
                  f"P&L: Rs{combined_pnl:+,.2f} ({pnl_pct:+.2f}%){vix_tag}",
                  end="", flush=True)

            # Auto-reconnect: if candles haven't grown in 3 minutes, WS is dead
            if max_candles > 0 and max_candles == last_candle_count:
                if time.time() - ws_state["last_candle_time"] > 180:
                    if not ws_state["reconnecting"]:
                        ws_state["reconnecting"] = True
                        print(f"\n  [RECONNECT] No new candles in 3 min — reconnecting WebSocket...")
                        try:
                            ws_state["sws"].close_connection()
                        except Exception:
                            pass
                        time.sleep(3)
                        ws_thread = start_websocket()
                        ws_state["last_candle_time"] = time.time()
                        ws_state["reconnecting"] = False
            else:
                last_candle_count = max_candles
                ws_state["last_candle_time"] = time.time()

        # Strategy scan every 60s
        if time.time() - last_scan < 60:
            continue
        last_scan = time.time()

        now = datetime.now()

        # Market hours: 9:30 AM - 3:10 PM
        if now.hour < 9 or (now.hour == 9 and now.minute < 30):
            continue

        # After 3:10 PM — close all positions (S1 and S3)
        if now.hour >= 15 and now.minute > 10:
            for tid in list(portfolio.positions.keys()):
                pos = portfolio.positions[tid]
                tok = token_map.get(pos["symbol"])
                df = candle_builder.get_df(tok) if tok else None
                if df is not None and len(df) > 0:
                    portfolio.close_position(tid, df["close"].iloc[-1], pos["remaining"], "EOD_EXIT")

            # Close S3 positions
            s3_stock_data = {}
            for sym, tok in token_map.items():
                df = candle_builder.get_df(tok)
                if df is not None:
                    s3_stock_data[tok] = df
            if s3_stock_data:
                scan_15min_rsi(s3_stock_data, token_to_sym, portfolio, logger, now)

            # Auto-stop at 3:30 PM — market is closed
            if now.hour >= 15 and now.minute >= 30:
                print(f"\n\n  [3:30 PM] Market closed — auto-stopping bot")
                running[0] = False
            continue

        # ── S1: Check exits for open positions ──
        for tid in list(portfolio.positions.keys()):
            pos = portfolio.positions.get(tid)
            if not pos:
                continue
            sym = pos["symbol"]
            tok = token_map.get(sym)
            df = candle_builder.get_df(tok) if tok else None
            if df is None or len(df) < 5:
                continue

            curr_price = df["close"].iloc[-1]
            elapsed_min = (now - pos["entry_timestamp"]).total_seconds() / 60

            # Disaster stop: 3x ATR on 15-min
            if curr_price <= pos["stop_loss"]:
                portfolio.close_position(tid, curr_price, pos["remaining"], "DISASTER_STOP")
                continue

            # Hard exit at 3:00 PM
            if now.hour >= HARD_EXIT_HOUR and now.minute >= HARD_EXIT_MIN:
                portfolio.close_position(tid, curr_price, pos["remaining"], "HARD_EXIT_3PM")
                continue

            # Time stop: 75 minutes
            if elapsed_min >= TIME_STOP_MINUTES:
                portfolio.close_position(tid, curr_price, pos["remaining"], "TIME_STOP_75M")
                continue

            # Exit 1: 60% at VWAP touch
            if not pos["vwap_exit_done"]:
                df["vwap_val"] = vwap_calc(df)
                curr_vwap = df["vwap_val"].iloc[-1]
                if not pd.isna(curr_vwap) and curr_vwap > 0 and curr_price >= curr_vwap:
                    exit_qty = int(pos["initial_qty"] * 0.6)
                    exit_qty = min(exit_qty, pos["remaining"])
                    if exit_qty > 0:
                        portfolio.close_position(tid, curr_price, exit_qty, "VWAP_TOUCH_60pct")
                        pos["vwap_exit_done"] = True
                        continue

            # Exit 2: remaining 40% at RSI(5) > 50 on 5-min
            df_5m = df.set_index("timestamp").resample("5min").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum",
            }).dropna().reset_index()
            if len(df_5m) >= RSI_PERIOD + 1:
                df_5m["rsi5"] = rsi(df_5m["close"], RSI_PERIOD)
                curr_rsi5 = df_5m["rsi5"].iloc[-1]
                if not pd.isna(curr_rsi5) and curr_rsi5 >= RSI_EXIT_PARTIAL:
                    portfolio.close_position(tid, curr_price, pos["remaining"], "RSI5_GT50")
                    continue

        # ── S1: Check entries (only 10:15-14:00, VIX permitting) ──
        now_min = now.hour * 60 + now.minute
        entry_start = ENTRY_START_HOUR * 60 + ENTRY_START_MIN
        entry_end = ENTRY_END_HOUR * 60 + ENTRY_END_MIN
        in_entry_window = entry_start <= now_min < entry_end

        if in_entry_window and vix_size_mult > 0:
            for sym, tok in token_map.items():
                df = candle_builder.get_df(tok)
                if df is None or len(df) < 20:
                    continue

                # Resample 1-min to 5-min for RSI(5)
                df_5m = df.set_index("timestamp").resample("5min").agg({
                    "open": "first", "high": "max", "low": "min",
                    "close": "last", "volume": "sum",
                }).dropna().reset_index()

                if len(df_5m) < RSI_PERIOD + 2:
                    continue

                df_5m["rsi5"] = rsi(df_5m["close"], RSI_PERIOD)
                prev_rsi5 = df_5m["rsi5"].iloc[-2]
                curr_rsi5 = df_5m["rsi5"].iloc[-1]
                curr_price = df["close"].iloc[-1]

                if pd.isna(prev_rsi5) or pd.isna(curr_rsi5):
                    continue

                # Only log thoughts when RSI is interesting (< 30 on 5-min)
                if curr_rsi5 >= 30 and prev_rsi5 >= 30:
                    continue

                # Compute indicators for logging
                df["vwap_val"] = vwap_calc(df)
                curr_vwap = df["vwap_val"].iloc[-1]
                vwap_distance = (curr_price - curr_vwap) / curr_vwap if curr_vwap > 0 else 0

                # Daily regime check result
                regime_ok = daily_regime.get(sym, False)

                # 15-min KER (Kaufman Efficiency Ratio)
                ker_val = None
                df_15m = df.set_index("timestamp").resample("15min").agg({
                    "open": "first", "high": "max", "low": "min",
                    "close": "last", "volume": "sum",
                }).dropna().reset_index()
                if len(df_15m) >= KER_PERIOD + 2:
                    df_15m["ker10"] = kaufman_er(df_15m["close"], KER_PERIOD)
                    ker_val = df_15m["ker10"].iloc[-1]

                # MFI(8) on 5-min
                mfi_val = None
                if len(df_5m) >= MFI_PERIOD + 2:
                    df_5m["mfi8"] = mfi(df_5m["high"], df_5m["low"], df_5m["close"], df_5m["volume"], MFI_PERIOD)
                    mfi_val = df_5m["mfi8"].iloc[-1]

                # RSI(5) uptick: prev < 20 AND curr > prev (RSI rising from oversold)
                is_uptick = prev_rsi5 < RSI_OVERSOLD and curr_rsi5 > prev_rsi5

                if not is_uptick:
                    # RSI is low but no uptick yet
                    if curr_rsi5 < RSI_OVERSOLD:
                        logger.log_thought(
                            sym, curr_price, curr_rsi5, "RSI5_LOW",
                            regime_ok, curr_vwap, vwap_distance, ker_val,
                            "WATCHING", f"RSI(5)={curr_rsi5:.1f} below 20 — waiting for uptick",
                            mfi_val=mfi_val,
                        )
                    continue

                # ── RSI(5) uptick from below 20 — run filter stack ──

                # Already holding in S1?
                if any(p["symbol"] == sym for p in portfolio.positions.values()):
                    logger.log_thought(
                        sym, curr_price, curr_rsi5, "RSI5_UPTICK",
                        regime_ok, curr_vwap, vwap_distance, ker_val,
                        "SKIP", "Already holding this stock in S1",
                        mfi_val=mfi_val,
                    )
                    continue

                # Already holding in S3?
                s3_syms = {p["symbol"] for p in get_s3_positions().values()}
                if sym in s3_syms:
                    logger.log_thought(
                        sym, curr_price, curr_rsi5, "RSI5_UPTICK",
                        regime_ok, curr_vwap, vwap_distance, ker_val,
                        "SKIP", "Already holding in S3",
                        mfi_val=mfi_val,
                    )
                    continue

                if len(portfolio.positions) >= MAX_POSITIONS:
                    logger.log_thought(
                        sym, curr_price, curr_rsi5, "RSI5_UPTICK",
                        regime_ok, curr_vwap, vwap_distance, ker_val,
                        "SKIP", f"Max positions ({MAX_POSITIONS}) reached",
                        mfi_val=mfi_val,
                    )
                    continue

                sector = SECTOR_MAP.get(sym, "Other")
                if portfolio.sector_count.get(sector, 0) >= MAX_PER_SECTOR:
                    logger.log_thought(
                        sym, curr_price, curr_rsi5, "RSI5_UPTICK",
                        regime_ok, curr_vwap, vwap_distance, ker_val,
                        "SKIP", f"Sector limit: {sector} already has a position",
                        mfi_val=mfi_val,
                    )
                    continue

                # Filter 1: Daily regime check (2 of 3 conditions)
                if not regime_ok:
                    logger.log_thought(
                        sym, curr_price, curr_rsi5, "RSI5_UPTICK",
                        regime_ok, curr_vwap, vwap_distance, ker_val,
                        "FILTERED", "Daily regime check failed (2/3 conditions not met)",
                        mfi_val=mfi_val,
                    )
                    print(f"\n  [{now.strftime('%H:%M')}] {sym} RSI(5)={curr_rsi5:.1f} "
                          f"— FILTERED (daily regime)")
                    continue

                # Filter 2: KER(10) < 0.30 on 15-min (choppy/mean-reverting)
                if ker_val is not None and not pd.isna(ker_val) and ker_val >= KER_MAX:
                    logger.log_thought(
                        sym, curr_price, curr_rsi5, "RSI5_UPTICK",
                        regime_ok, curr_vwap, vwap_distance, ker_val,
                        "FILTERED", f"KER(10)={ker_val:.3f} >= {KER_MAX} (trending, not mean-reverting)",
                        mfi_val=mfi_val,
                    )
                    print(f"\n  [{now.strftime('%H:%M')}] {sym} RSI(5)={curr_rsi5:.1f} "
                          f"— FILTERED (KER too high: {ker_val:.3f})")
                    continue

                # Filter 3: Price must be below VWAP
                if curr_vwap > 0 and curr_price >= curr_vwap:
                    logger.log_thought(
                        sym, curr_price, curr_rsi5, "RSI5_UPTICK",
                        regime_ok, curr_vwap, vwap_distance, ker_val,
                        "FILTERED", f"Price above VWAP ({curr_vwap:.2f})",
                        mfi_val=mfi_val,
                    )
                    print(f"\n  [{now.strftime('%H:%M')}] {sym} RSI(5)={curr_rsi5:.1f} "
                          f"— FILTERED (above VWAP)")
                    continue

                # Filter 4: MFI(8) < 30 confirmation (optional — skip if unavailable)
                if mfi_val is not None and not pd.isna(mfi_val) and mfi_val >= MFI_THRESHOLD:
                    logger.log_thought(
                        sym, curr_price, curr_rsi5, "RSI5_UPTICK",
                        regime_ok, curr_vwap, vwap_distance, ker_val,
                        "FILTERED", f"MFI(8)={mfi_val:.1f} >= {MFI_THRESHOLD} (volume not confirming)",
                        mfi_val=mfi_val,
                    )
                    print(f"\n  [{now.strftime('%H:%M')}] {sym} RSI(5)={curr_rsi5:.1f} "
                          f"— FILTERED (MFI too high: {mfi_val:.1f})")
                    continue

                # All filters passed — calculate position using 15-min ATR
                if df_15m is not None and len(df_15m) >= 15:
                    atr_15m = atr_calc(df_15m["high"], df_15m["low"], df_15m["close"], 14)
                    curr_atr = atr_15m.iloc[-1]
                else:
                    # Fallback to 1-min ATR * sqrt(15)
                    atr_1m = atr_calc(df["high"], df["low"], df["close"], 14)
                    curr_atr = atr_1m.iloc[-1] * (15 ** 0.5) if not pd.isna(atr_1m.iloc[-1]) else curr_price * 0.01

                if pd.isna(curr_atr) or curr_atr <= 0:
                    curr_atr = curr_price * 0.01

                stop_loss = round(curr_price - ATR_SL_MULT * curr_atr, 2)
                risk_per_share = curr_price - stop_loss
                if risk_per_share <= 0:
                    continue

                effective_risk = RISK_PER_TRADE * vix_size_mult
                qty = int(effective_risk / risk_per_share)
                qty = min(qty, int(83000 / curr_price))
                if qty <= 0:
                    continue

                # ── ENTRY ──
                logger.log_thought(
                    sym, curr_price, curr_rsi5, "RSI5_UPTICK",
                    regime_ok, curr_vwap, vwap_distance, ker_val,
                    "BUY", f"All filters passed | Qty={qty} SL={stop_loss:.2f} ATR={curr_atr:.2f} VIX_mult={vix_size_mult}",
                    mfi_val=mfi_val,
                )

                trade_counter[0] += 1
                tid = f"PAPER-{trade_counter[0]:04d}"
                portfolio.open_position(tid, sym, curr_price, qty, stop_loss, curr_rsi5)

        # ── Strategy 3: Multi-Timeframe RSI Mean Reversion ──
        try:
            s3_stock_data = {}
            for sym, tok in token_map.items():
                df = candle_builder.get_df(tok)
                if df is not None:
                    s3_stock_data[tok] = df
            if s3_stock_data:
                scan_15min_rsi(s3_stock_data, token_to_sym, portfolio, logger, now)
        except Exception as e:
            print(f"\n  [S3 ERROR] {e}")

    # ── Shutdown ──
    try:
        if ws_state["sws"]:
            ws_state["sws"].close_connection()
    except Exception:
        pass

    portfolio.summary()
    logger.close()

    # Generate daily diary report
    try:
        from daily_report import generate_report
        print("\n  Generating daily diary...")
        generate_report()
    except Exception as e:
        print(f"  Report generation failed: {e}")


if __name__ == "__main__":
    main()
