"""Paper trading with LIVE WebSocket data.

Uses SmartWebSocketV2 for real-time ticks → builds 1-min candles → runs both strategies.
No historical API needed = no rate limit issues.
"""

import json
import os
import signal
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, date, timedelta

import pandas as pd
import pyotp
from SmartApi import SmartConnect
from SmartApi.smartWebSocketV2 import SmartWebSocketV2
from dotenv import load_dotenv

load_dotenv("/Users/rudraym/Trader/.env")

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
# CANDLE BUILDER — ticks → 1-min OHLCV
# ══════════════════════════════════════════
class CandleBuilder:
    def __init__(self):
        # token -> {minute_key -> {o, h, l, c, v}}
        self._building = defaultdict(dict)
        # token -> list of completed candle dicts
        self._candles = defaultdict(list)
        self._lock = threading.Lock()

    def on_tick(self, token: str, ltp: float, volume: int):
        now = datetime.now()
        minute_key = now.strftime("%Y-%m-%d %H:%M")

        with self._lock:
            buckets = self._building[token]

            # Finalize old candles
            for mk in list(buckets.keys()):
                if mk != minute_key:
                    candle = buckets.pop(mk)
                    candle["timestamp"] = pd.Timestamp(mk)
                    self._candles[token].append(candle)
                    # Trim to 200
                    if len(self._candles[token]) > 200:
                        self._candles[token] = self._candles[token][-200:]

            # Update current candle
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
    def __init__(self, capital):
        self.capital = capital
        self.positions = {}  # trade_id -> dict
        self.closed_trades = []
        self.daily_pnl = 0.0
        self.sector_count = defaultdict(int)

    def open_position(self, trade_id, symbol, price, quantity, stop_loss, rsi_val):
        sector = SECTOR_MAP.get(symbol, "Other")
        self.positions[trade_id] = {
            "symbol": symbol, "entry_price": price, "quantity": quantity,
            "remaining": quantity, "stop_loss": stop_loss, "status": "open",
            "entry_time": datetime.now(), "candles_held": 0, "realized_pnl": 0.0,
            "entry_rsi": rsi_val,
        }
        self.sector_count[sector] += 1
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

        emoji = "+" if pnl >= 0 else ""
        print(f"  << PAPER SELL {pos['symbol']} x{quantity} @ Rs{price:.2f} | "
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
        print(f"\n  {'='*60}")
        print(f"  PAPER TRADING SUMMARY")
        print(f"  {'='*60}")
        print(f"  Open positions:  {len(self.positions)}")
        print(f"  Closed trades:   {len(self.closed_trades)}")
        print(f"  Daily P&L:       Rs{self.daily_pnl:+,.2f}")
        if self.closed_trades:
            wins = [t for t in self.closed_trades if t["total_pnl"] > 0]
            losses = [t for t in self.closed_trades if t["total_pnl"] <= 0]
            print(f"  Wins/Losses:     {len(wins)}/{len(losses)}")
            print(f"\n  Trade Log:")
            for t in self.closed_trades:
                e = "W" if t["total_pnl"] > 0 else "L"
                print(f"    [{e}] {t['symbol']:15s} Rs{t['entry_price']:.2f} -> Rs{t['exit_price']:.2f} "
                      f"| {t['exit_reason']:12s} | P&L: Rs{t['total_pnl']:+,.2f}")
        for tid, pos in self.positions.items():
            print(f"    [OPEN] {pos['symbol']:15s} Rs{pos['entry_price']:.2f} "
                  f"x{pos['remaining']} | candles={pos['candles_held']}")


# ══════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════
def main():
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
    with open("/Users/rudraym/Trader/data/instruments.json") as f:
        master_data = json.load(f)
    master_df = pd.DataFrame(master_data)

    token_map = {}  # symbol -> token
    token_to_sym = {}  # token -> symbol
    for sym in STOCKS:
        matches = master_df[(master_df["symbol"] == sym) & (master_df["exch_seg"] == "NSE")]
        if not matches.empty:
            tok = matches.iloc[0]["token"]
            token_map[sym] = tok
            token_to_sym[tok] = sym

    print(f"[OK] {len(token_map)} stocks mapped")

    # Initialize
    candle_builder = CandleBuilder()
    portfolio = PaperPortfolio(CAPITAL)
    trade_counter = [0]
    running = [True]

    # ── WebSocket callbacks ──
    def on_data(wsapp, msg):
        try:
            token = str(msg.get("token", ""))
            ltp = msg.get("last_traded_price", 0)
            if isinstance(ltp, (int, float)):
                ltp = ltp / 100  # Paise -> Rupees
            volume = msg.get("volume_trade_for_the_day", 0)
            if token and ltp > 0:
                candle_builder.on_tick(token, ltp, volume)
        except Exception as e:
            pass

    def on_open(wsapp):
        tokens = list(token_map.values())
        token_list = [{"exchangeType": 1, "tokens": tokens}]
        sws.subscribe("autotheta_paper", 2, token_list)  # Mode 2 = Quote
        print(f"[OK] WebSocket connected — subscribed to {len(tokens)} tokens")
        print(f"[OK] Building candles from live ticks...")
        print(f"     (Need ~10 candles before RSI can be calculated)")
        print(f"\n  Ctrl+C to stop and see summary\n")

    def on_error(wsapp, error):
        print(f"  [WS ERROR] {error}")

    def on_close(wsapp):
        print("  [WS] Connection closed")

    # ── Pure WebSocket mode — no historical API dependency ──
    # RSI(7) needs ~10 candles to produce meaningful values.
    # The bot will start scanning after enough candles are built from live ticks.
    # For instant readiness, start before 9:25 AM — by 9:35 you'll have 10+ candles.
    print("[OK] Pure WebSocket mode — no historical API needed")
    print("     RSI scanning starts after ~10 candles are built from live ticks\n")

    # Start WebSocket
    sws = SmartWebSocketV2(auth_token, API_KEY, CLIENT_ID, feed_token)
    sws.on_data = on_data
    sws.on_open = on_open
    sws.on_error = on_error
    sws.on_close = on_close

    ws_thread = threading.Thread(target=sws.connect, daemon=True)
    ws_thread.start()

    # Graceful shutdown
    def shutdown(sig, frame):
        running[0] = False
        print("\n\n  Shutting down...")

    signal.signal(signal.SIGINT, shutdown)

    # ── Strategy loop — runs every 60s ──
    last_scan = time.time() - 55  # First scan after 5s warmup
    tick_count = 0

    while running[0]:
        time.sleep(1)
        tick_count += 1

        # Print status every 10s
        if tick_count % 10 == 0:
            candle_counts = {token_to_sym.get(t, t): candle_builder.candle_count(t)
                            for t in token_map.values()}
            max_candles = max(candle_counts.values()) if candle_counts else 0
            active = sum(1 for c in candle_counts.values() if c > 0)
            now = datetime.now()
            print(f"\r  [{now.strftime('%H:%M:%S')}] "
                  f"Candles: {max_candles} | Active feeds: {active}/{len(token_map)} | "
                  f"Positions: {len(portfolio.positions)} | "
                  f"P&L: Rs{portfolio.daily_pnl:+,.2f}",
                  end="", flush=True)

        # Run strategy scan every 60 seconds
        if time.time() - last_scan < 60:
            continue
        last_scan = time.time()

        now = datetime.now()
        # Skip before 9:30 and after 3:10
        if now.hour < 9 or (now.hour == 9 and now.minute < 30):
            continue
        if now.hour >= 15 and now.minute > 10:
            # Close all open positions at EOD
            for tid in list(portfolio.positions.keys()):
                pos = portfolio.positions[tid]
                tok = token_map.get(pos["symbol"])
                df = candle_builder.get_df(tok) if tok else None
                if df is not None and len(df) > 0:
                    portfolio.close_position(tid, df["close"].iloc[-1], pos["remaining"], "EOD_EXIT")
            continue

        # ── RSI Bounce scan ──
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

            # ── Check exits ──
            for tid in list(portfolio.positions.keys()):
                pos = portfolio.positions.get(tid)
                if not pos or pos["symbol"] != sym:
                    continue
                pos["candles_held"] += 1

                # Stop-loss
                if curr_price <= pos["stop_loss"]:
                    portfolio.close_position(tid, curr_price, pos["remaining"], "STOP_LOSS")
                    continue

                # Time stop
                if pos["candles_held"] >= TIME_STOP and current_rsi < EXIT_1_RSI:
                    portfolio.close_position(tid, curr_price, pos["remaining"], "TIME_STOP")
                    continue

                # Partial exit at RSI 40
                if pos["status"] == "open" and current_rsi >= EXIT_1_RSI:
                    exit_qty = pos["remaining"] // 2
                    if exit_qty > 0:
                        portfolio.close_position(tid, curr_price, exit_qty, "RSI_40")
                        pos["status"] = "partial"

                # Final exit at RSI 50
                elif pos["status"] == "partial" and current_rsi >= EXIT_2_RSI:
                    portfolio.close_position(tid, curr_price, pos["remaining"], "RSI_50")

            # ── Check entry ──
            if not (prev_rsi >= OVERSOLD and current_rsi < OVERSOLD):
                continue

            # Already holding?
            if any(p["symbol"] == sym for p in portfolio.positions.values()):
                continue
            if len(portfolio.positions) >= MAX_POSITIONS:
                continue

            sector = SECTOR_MAP.get(sym, "Other")
            if portfolio.sector_count.get(sector, 0) >= MAX_PER_SECTOR:
                continue

            # 5-min EMA20 filter
            df_5m = df.set_index("timestamp").resample("5min").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum",
            }).dropna().reset_index()

            if len(df_5m) >= 20:
                ema20_5m = ema(df_5m["close"], 20).iloc[-1]
                if curr_price < ema20_5m:
                    print(f"\n  [{now.strftime('%H:%M')}] {sym} RSI={current_rsi:.1f} "
                          f"— FILTERED (below 5m EMA20: {ema20_5m:.2f})")
                    continue

            # VWAP filter
            curr_vwap = df["vwap_val"].iloc[-1]
            if curr_price < curr_vwap * 0.998:
                print(f"\n  [{now.strftime('%H:%M')}] {sym} RSI={current_rsi:.1f} "
                      f"— FILTERED (below VWAP: {curr_vwap:.2f})")
                continue

            # Volume filter
            vol_avg = df["volume"].iloc[-20:].mean()
            if vol_avg > 0 and df["volume"].iloc[-1] < vol_avg * 1.5:
                print(f"\n  [{now.strftime('%H:%M')}] {sym} RSI={current_rsi:.1f} "
                      f"— FILTERED (low volume)")
                continue

            # Position sizing
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

            # ENTRY
            trade_counter[0] += 1
            tid = f"PAPER-{trade_counter[0]:04d}"
            portfolio.open_position(tid, sym, curr_price, qty, stop_loss, current_rsi)

    # ── Shutdown ──
    try:
        sws.close_connection()
    except:
        pass

    portfolio.summary()


if __name__ == "__main__":
    main()
