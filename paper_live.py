"""Paper trading with LIVE WebSocket data.

Uses SmartWebSocketV2 for real-time ticks → builds 1-min candles → runs RSI bounce strategy.
No historical API needed = no rate limit issues.

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
def rsi(series, period=7):
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


# ── Config ──
STOCKS = [
    "SBIN-EQ", "HDFCBANK-EQ", "RELIANCE-EQ", "ICICIBANK-EQ", "INFY-EQ",
    "TCS-EQ", "KOTAKBANK-EQ", "LT-EQ", "ITC-EQ", "AXISBANK-EQ",
    "BHARTIARTL-EQ", "BAJFINANCE-EQ", "SUNPHARMA-EQ", "HCLTECH-EQ",
    "WIPRO-EQ", "TATASTEEL-EQ", "TITAN-EQ", "MARUTI-EQ",
]

SECTOR_MAP = {
    "HDFCBANK-EQ": "Banking", "ICICIBANK-EQ": "Banking", "KOTAKBANK-EQ": "Banking",
    "SBIN-EQ": "Banking", "AXISBANK-EQ": "Banking",
    "BAJFINANCE-EQ": "Finance", "RELIANCE-EQ": "Energy",
    "TCS-EQ": "IT", "INFY-EQ": "IT", "WIPRO-EQ": "IT", "HCLTECH-EQ": "IT",
    "ITC-EQ": "FMCG", "SUNPHARMA-EQ": "Pharma",
    "TATASTEEL-EQ": "Metals", "LT-EQ": "Infra", "BHARTIARTL-EQ": "Telecom",
    "TITAN-EQ": "Other", "MARUTI-EQ": "Auto",
}

RSI_PERIOD = 7
OVERSOLD = 20
EXIT_1_RSI = 40
EXIT_2_RSI = 50
ATR_SL_MULT = 1.5
TIME_STOP = 15
RISK_PER_TRADE = 2500
MAX_POSITIONS = 4
MAX_PER_SECTOR = 1
CAPITAL = 250000


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
                "Time", "Stock", "Price", "RSI(7)", "Signal",
                "EMA20_5m", "vs_EMA", "VWAP", "vs_VWAP",
                "Volume", "Vol_Avg", "vs_Vol",
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
                    ema20_5m, vwap_val, volume, vol_avg,
                    decision, reason):
        """Log what the bot saw and why it decided what it did."""
        self._ensure_files()
        now = datetime.now().strftime("%H:%M:%S")

        vs_ema = ""
        if ema20_5m and ema20_5m > 0:
            vs_ema = "ABOVE" if price >= ema20_5m else "BELOW"
        vs_vwap = "ABOVE" if vwap_val and price >= vwap_val * 0.998 else "BELOW"
        vs_vol = ""
        if vol_avg and vol_avg > 0:
            vs_vol = "HIGH" if volume >= vol_avg * 1.5 else "LOW"

        self._thoughts_writer.writerow([
            now, stock, f"{price:.2f}", f"{rsi_val:.1f}", signal_type,
            f"{ema20_5m:.2f}" if ema20_5m else "N/A", vs_ema,
            f"{vwap_val:.2f}" if vwap_val else "N/A", vs_vwap,
            f"{volume:.0f}", f"{vol_avg:.0f}" if vol_avg else "N/A", vs_vol,
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
                    if len(self._candles[token]) > 200:
                        self._candles[token] = self._candles[token][-200:]

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
            "remaining": quantity, "stop_loss": stop_loss, "status": "open",
            "entry_time": datetime.now(), "candles_held": 0, "realized_pnl": 0.0,
            "entry_rsi": rsi_val,
        }
        self.sector_count[sector] += 1
        self.log.log_trade("BUY", symbol, quantity, price, rsi_val, stop_loss, "RSI_OVERSOLD")
        print(f"\n  >> PAPER BUY {symbol} x{quantity} @ Rs{price:.2f} | "
              f"SL=Rs{stop_loss:.2f} | RSI={rsi_val:.1f}")

    def close_position(self, trade_id, price, quantity, reason):
        pos = self.positions.get(trade_id)
        if not pos:
            return
        pnl = (price - pos["entry_price"]) * quantity
        pos["realized_pnl"] += pnl
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
                "exit_reason": reason, "total_pnl": pos["realized_pnl"],
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
                t_pnl_pct = t["total_pnl"] / (t["entry_price"] * t["quantity"]) * 100
                marker = "W" if t["total_pnl"] > 0 else "L"
                print(f"  [{marker}] {t['symbol']:15s} {t['entry_price']:>10.2f} {t['exit_price']:>10.2f} "
                      f"{t['quantity']:>5d}  {t['exit_reason']:>12s} {t['total_pnl']:>+11,.2f} {t_pnl_pct:>+7.2f}%")
        else:
            print(f"  (no trades today)")

        if self.positions:
            print(f"\n  Open positions:")
            for tid, pos in self.positions.items():
                print(f"    [OPEN] {pos['symbol']:15s} Rs{pos['entry_price']:.2f} "
                      f"x{pos['remaining']} | candles={pos['candles_held']}")

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
    print(f"  AutoTheta PAPER TRADING — WebSocket Mode")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} IST")
    print(f"  Capital: Rs{CAPITAL:,}")
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

    # Load token map
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
            print(f"\r  [{now.strftime('%H:%M:%S')}] "
                  f"Candles: {max_candles} | Feeds: {active}/{len(token_map)} | "
                  f"Pos: S1={len(portfolio.positions)} S3={len(s3_positions)} | "
                  f"P&L: Rs{combined_pnl:+,.2f} ({pnl_pct:+.2f}%)",
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
                # Force exit by calling scan with a time past 14:30 (the hard exit triggers)
                scan_15min_rsi(s3_stock_data, token_to_sym, portfolio, logger, now)

            # Auto-stop at 3:30 PM — market is closed
            if now.hour >= 15 and now.minute >= 30:
                print(f"\n\n  [3:30 PM] Market closed — auto-stopping bot")
                running[0] = False
            continue

        # ── Scan all stocks ──
        for sym, tok in token_map.items():
            df = candle_builder.get_df(tok)
            if df is None or len(df) < 20:
                continue

            df["rsi"] = rsi(df["close"], RSI_PERIOD)
            df["atr"] = atr_calc(df["high"], df["low"], df["close"], 14)
            df["vwap_val"] = vwap_calc(df)

            current_rsi = df["rsi"].iloc[-1]
            prev_rsi = df["rsi"].iloc[-2]
            curr_price = df["close"].iloc[-1]
            curr_vwap = df["vwap_val"].iloc[-1]
            curr_volume = df["volume"].iloc[-1]
            vol_avg = df["volume"].iloc[-20:].mean()

            # 5-min EMA20
            ema20_5m = None
            df_5m = df.set_index("timestamp").resample("5min").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum",
            }).dropna().reset_index()
            if len(df_5m) >= 20:
                ema20_5m = ema(df_5m["close"], 20).iloc[-1]

            # ── Check exits for open positions ──
            for tid in list(portfolio.positions.keys()):
                pos = portfolio.positions.get(tid)
                if not pos or pos["symbol"] != sym:
                    continue
                pos["candles_held"] += 1

                if curr_price <= pos["stop_loss"]:
                    portfolio.close_position(tid, curr_price, pos["remaining"], "STOP_LOSS")
                    continue
                if pos["candles_held"] >= TIME_STOP and current_rsi < EXIT_1_RSI:
                    portfolio.close_position(tid, curr_price, pos["remaining"], "TIME_STOP")
                    continue
                if pos["status"] == "open" and current_rsi >= EXIT_1_RSI:
                    exit_qty = pos["remaining"] // 2
                    if exit_qty > 0:
                        portfolio.close_position(tid, curr_price, exit_qty, "RSI_40")
                        pos["status"] = "partial"
                elif pos["status"] == "partial" and current_rsi >= EXIT_2_RSI:
                    portfolio.close_position(tid, curr_price, pos["remaining"], "RSI_50")

            # ── Check entry ──
            # Only log thoughts when RSI is interesting (< 30)
            if current_rsi >= 30:
                continue

            # RSI crossover below 20?
            is_trigger = prev_rsi >= OVERSOLD and current_rsi < OVERSOLD

            if not is_trigger:
                # RSI is low but no crossover yet — log as WATCHING
                if current_rsi < OVERSOLD:
                    logger.log_thought(
                        sym, curr_price, current_rsi, "OVERSOLD",
                        ema20_5m, curr_vwap, curr_volume, vol_avg,
                        "WATCHING", "RSI below 20 but no crossover yet",
                    )
                continue

            # ── RSI crossed below 20 — run filter stack ──
            # Already holding in S1?
            if any(p["symbol"] == sym for p in portfolio.positions.values()):
                logger.log_thought(
                    sym, curr_price, current_rsi, "RSI<20",
                    ema20_5m, curr_vwap, curr_volume, vol_avg,
                    "SKIP", "Already holding this stock",
                )
                continue

            # Already holding in S3?
            s3_syms = {p["symbol"] for p in get_s3_positions().values()}
            if sym in s3_syms:
                logger.log_thought(
                    sym, curr_price, current_rsi, "RSI<20",
                    ema20_5m, curr_vwap, curr_volume, vol_avg,
                    "SKIP", "Already holding in S3",
                )
                continue

            if len(portfolio.positions) >= MAX_POSITIONS:
                logger.log_thought(
                    sym, curr_price, current_rsi, "RSI<20",
                    ema20_5m, curr_vwap, curr_volume, vol_avg,
                    "SKIP", f"Max positions ({MAX_POSITIONS}) reached",
                )
                continue

            sector = SECTOR_MAP.get(sym, "Other")
            if portfolio.sector_count.get(sector, 0) >= MAX_PER_SECTOR:
                logger.log_thought(
                    sym, curr_price, current_rsi, "RSI<20",
                    ema20_5m, curr_vwap, curr_volume, vol_avg,
                    "SKIP", f"Sector limit: {sector} already has a position",
                )
                continue

            # Filter 1: 5-min EMA20
            if ema20_5m and curr_price < ema20_5m:
                logger.log_thought(
                    sym, curr_price, current_rsi, "RSI<20",
                    ema20_5m, curr_vwap, curr_volume, vol_avg,
                    "FILTERED", f"Price below 5m EMA20 ({ema20_5m:.2f})",
                )
                print(f"\n  [{now.strftime('%H:%M')}] {sym} RSI={current_rsi:.1f} "
                      f"— FILTERED (below 5m EMA20)")
                continue

            # Filter 2: VWAP
            if curr_price < curr_vwap * 0.998:
                logger.log_thought(
                    sym, curr_price, current_rsi, "RSI<20",
                    ema20_5m, curr_vwap, curr_volume, vol_avg,
                    "FILTERED", f"Price below VWAP ({curr_vwap:.2f})",
                )
                print(f"\n  [{now.strftime('%H:%M')}] {sym} RSI={current_rsi:.1f} "
                      f"— FILTERED (below VWAP)")
                continue

            # Filter 3: Volume
            if vol_avg > 0 and curr_volume < vol_avg * 1.5:
                logger.log_thought(
                    sym, curr_price, current_rsi, "RSI<20",
                    ema20_5m, curr_vwap, curr_volume, vol_avg,
                    "FILTERED", f"Volume {curr_volume:.0f} < 1.5x avg ({vol_avg*1.5:.0f})",
                )
                print(f"\n  [{now.strftime('%H:%M')}] {sym} RSI={current_rsi:.1f} "
                      f"— FILTERED (low volume)")
                continue

            # All filters passed — calculate position
            curr_atr = df["atr"].iloc[-1]
            if pd.isna(curr_atr) or curr_atr <= 0:
                curr_atr = curr_price * 0.005
            stop_loss = round(curr_price - ATR_SL_MULT * curr_atr, 2)
            risk_per_share = curr_price - stop_loss
            if risk_per_share <= 0:
                continue
            qty = int(RISK_PER_TRADE / risk_per_share)
            qty = min(qty, int(83000 / curr_price))
            if qty <= 0:
                continue

            # ── ENTRY ──
            logger.log_thought(
                sym, curr_price, current_rsi, "RSI<20",
                ema20_5m, curr_vwap, curr_volume, vol_avg,
                "BUY", f"All filters passed | Qty={qty} SL={stop_loss:.2f} ATR={curr_atr:.2f}",
            )

            trade_counter[0] += 1
            tid = f"PAPER-{trade_counter[0]:04d}"
            portfolio.open_position(tid, sym, curr_price, qty, stop_loss, current_rsi)

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
