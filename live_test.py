"""Live test — Run both strategies against real-time market data.

March 17, 2026 (Tuesday) = Nifty expiry day
- RSI Bounce: Scans Nifty 50 stocks for oversold bounces
- Expiry Skew: Checks iron condor setup at current prices
"""

import time
import os
from datetime import datetime, timedelta
from collections import defaultdict

import pandas as pd
import pyotp
from SmartApi import SmartConnect
from dotenv import load_dotenv

load_dotenv("/Users/rudraym/Trader/.env")

API_KEY = os.getenv("ANGEL_API_KEY")
CLIENT_ID = os.getenv("ANGEL_CLIENT_ID")
PASSWORD = os.getenv("ANGEL_PASSWORD")
TOTP_SECRET = os.getenv("ANGEL_TOTP_SECRET")


def rsi(series, period=7):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def vwap(df):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (tp * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum().replace(0, 1)
    return cum_tp_vol / cum_vol


# ── Stock list (reduced to avoid rate limits) ──
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

import json

print("=" * 70)
print(f"  AutoTheta LIVE TEST — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} IST")
print("  Nifty Expiry Day (Tuesday)")
print("=" * 70)

# ── Authenticate ──
api = SmartConnect(API_KEY)
totp = pyotp.TOTP(TOTP_SECRET).now()
session = api.generateSession(CLIENT_ID, PASSWORD, totp)
if not session.get("status"):
    print(f"AUTH FAILED: {session}")
    exit(1)
print("[OK] Authenticated\n")

# ── Load token map from cached instruments ──
with open("/Users/rudraym/Trader/data/instruments.json") as f:
    master_data = json.load(f)
master_df = pd.DataFrame(master_data)

token_map = {}
for sym in STOCKS:
    matches = master_df[(master_df["symbol"] == sym) & (master_df["exch_seg"] == "NSE")]
    if not matches.empty:
        token_map[sym] = matches.iloc[0]["token"]

print(f"[OK] {len(token_map)} stocks mapped\n")

# ══════════════════════════════════════════
# PART 1: EXPIRY SKEW (it's after 2 PM!)
# ══════════════════════════════════════════
print("=" * 70)
print("  STRATEGY 1: EXPIRY SKEW IRON CONDOR (LIVE)")
print("=" * 70)

# Nifty spot
spot_data = api.ltpData("NSE", "NIFTY", "99926000")
nifty_spot = float(spot_data["data"]["ltp"])
atm = round(nifty_spot / 50) * 50

# VIX
time.sleep(0.5)
vix_data = api.ltpData("NSE", "India VIX", "99926017")
vix = float(vix_data["data"]["ltp"])

print(f"\n  Nifty Spot:    {nifty_spot:,.2f}")
print(f"  ATM Strike:    {atm}")
print(f"  India VIX:     {vix:.2f}", end="")
if 12 <= vix <= 18:
    print("  [PASS]")
else:
    print(f"  [FAIL — outside 12-18]")

# Option chain
master_df["expiry_dt"] = pd.to_datetime(master_df["expiry"], format="mixed", dayfirst=True).dt.date
master_df["actual_strike"] = master_df["strike"].astype(float) / 100
nifty_opts = master_df[
    (master_df["name"] == "NIFTY")
    & (master_df["instrumenttype"] == "OPTIDX")
    & (master_df["exch_seg"] == "NFO")
]
from datetime import date
nearest_expiry = nifty_opts[nifty_opts["expiry_dt"] >= date.today()]["expiry_dt"].min()
chain = nifty_opts[nifty_opts["expiry_dt"] == nearest_expiry]

print(f"  Expiry:        {nearest_expiry}")
print()

# Try multiple OTM offsets to find tradeable setups
for offset in [50, 100, 150, 200]:
    sell_put_strike = atm - offset
    sell_call_strike = atm + offset
    buy_put_strike = sell_put_strike - 100
    buy_call_strike = sell_call_strike + 100

    legs = {}
    for label, strike, opt_type in [
        ("SP", sell_put_strike, "PE"), ("SC", sell_call_strike, "CE"),
        ("BP", buy_put_strike, "PE"), ("BC", buy_call_strike, "CE"),
    ]:
        matches = chain[
            (chain["actual_strike"] == strike) & (chain["symbol"].str.endswith(opt_type))
        ]
        if not matches.empty:
            row = matches.iloc[0]
            time.sleep(0.5)
            try:
                r = api.ltpData("NFO", row["symbol"], row["token"])
                premium = float(r["data"]["ltp"])
                legs[label] = premium
            except:
                legs[label] = 0
        else:
            legs[label] = 0

    if all(legs.values()):
        net = (legs["SP"] + legs["SC"]) - (legs["BP"] + legs["BC"])
        max_profit = net * 65
        max_loss = (100 - net) * 65
        skew = max(legs["SP"], legs["SC"]) / max(min(legs["SP"], legs["SC"]), 0.05)

        print(f"  OTM {offset}pt | Sell {sell_put_strike}PE@{legs['SP']:.2f} + "
              f"{sell_call_strike}CE@{legs['SC']:.2f} | "
              f"Buy {buy_put_strike}PE@{legs['BP']:.2f} + {buy_call_strike}CE@{legs['BC']:.2f}")
        print(f"           Net: {net:.2f}/unit | Profit/lot: {max_profit:,.0f} | "
              f"Loss/lot: {max_loss:,.0f} | Skew: {skew:.1f}x", end="")

        trade_ok = True
        if skew < 2.0:
            print(" [SKIP: skew<2]", end="")
            trade_ok = False
        if vix < 12 or vix > 18:
            print(" [SKIP: VIX]", end="")
            trade_ok = False
        if trade_ok:
            print(" [TRADE SIGNAL]", end="")
        print()
    print()


# ══════════════════════════════════════════
# PART 2: RSI BOUNCE LIVE SCAN
# ══════════════════════════════════════════
print("=" * 70)
print("  STRATEGY 2: RSI BOUNCE — LIVE SCAN")
print("=" * 70)

now = datetime.now()
from_time = (now - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M")
to_time = now.strftime("%Y-%m-%d %H:%M")

print(f"\n  Fetching 1-min candles ({from_time} to {to_time})...")
print(f"  Rate limit: 1 request per 1s (historical API)\n")

stock_data = {}
for sym, token in token_map.items():
    time.sleep(1.0)  # Respect 3 req/sec limit with margin
    try:
        params = {
            "exchange": "NSE",
            "symboltoken": token,
            "interval": "ONE_MINUTE",
            "fromdate": from_time,
            "todate": to_time,
        }
        result = api.getCandleData(params)
        if result and result.get("data"):
            df = pd.DataFrame(result["data"],
                              columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            stock_data[sym] = df
        else:
            print(f"  {sym:18s} — No data")
    except Exception as e:
        print(f"  {sym:18s} — Error: {e}")

print(f"\n  Got data for {len(stock_data)} stocks. Running RSI scan...\n")

# ── Scan each stock ──
print(f"  {'STOCK':18s} {'PRICE':>10s} {'RSI(7)':>8s} {'vs 5m EMA':>10s} {'vs VWAP':>10s} {'SIGNAL':>10s}")
print("  " + "-" * 68)

signals = []
for sym, df in sorted(stock_data.items()):
    if len(df) < 20:
        continue

    df["rsi"] = rsi(df["close"], 7)
    df["vwap_val"] = vwap(df)

    # 5-min EMA20
    df_5m = df.set_index("timestamp").resample("5min").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna().reset_index()
    df_5m["ema20"] = ema(df_5m["close"], 20)
    ema20_5m = df_5m["ema20"].iloc[-1] if len(df_5m) >= 20 else None

    last = df.iloc[-1]
    curr_rsi = last["rsi"]
    curr_price = last["close"]
    curr_vwap = last["vwap_val"]

    # Status indicators
    ema_status = ""
    if ema20_5m is not None:
        if curr_price >= ema20_5m:
            ema_status = f"ABOVE ({ema20_5m:.1f})"
        else:
            ema_status = f"BELOW ({ema20_5m:.1f})"
    else:
        ema_status = "N/A"

    vwap_status = "ABOVE" if curr_price >= curr_vwap else "BELOW"

    signal = ""
    if curr_rsi < 20:
        if ema20_5m and curr_price >= ema20_5m and curr_price >= curr_vwap * 0.998:
            signal = "** BUY **"
            signals.append(sym)
        else:
            signal = "OVERSOLD"
    elif curr_rsi < 30:
        signal = "WATCH"
    elif curr_rsi > 80:
        signal = "OVERBOUGHT"

    rsi_color = f"{curr_rsi:.1f}"
    print(f"  {sym:18s} {curr_price:>10.2f} {rsi_color:>8s} {ema_status:>10s} {vwap_status:>10s} {signal:>10s}")

print()
if signals:
    print(f"  ** ACTIVE BUY SIGNALS: {', '.join(signals)} **")
else:
    print("  No active buy signals right now.")

# ── Recent RSI dips (last 30 min) ──
print(f"\n  Recent RSI < 20 events (last 30 min):")
print(f"  {'STOCK':18s} {'TIME':>12s} {'RSI':>8s} {'PRICE':>10s} {'CURRENT':>10s} {'IF BOUGHT':>12s}")
print("  " + "-" * 72)

any_dips = False
for sym, df in stock_data.items():
    if len(df) < 20:
        continue
    df["rsi_val"] = rsi(df["close"], 7)
    cutoff = datetime.now() - timedelta(minutes=30)

    for i in range(20, len(df)):
        row = df.iloc[i]
        if row["timestamp"].replace(tzinfo=None) < cutoff:
            continue
        if row["rsi_val"] < 20:
            current_price = df["close"].iloc[-1]
            dip_price = row["close"]
            pnl_pct = (current_price - dip_price) / dip_price * 100
            print(f"  {sym:18s} {str(row['timestamp'].time()):>12s} "
                  f"{row['rsi_val']:>8.1f} {dip_price:>10.2f} {current_price:>10.2f} "
                  f"{pnl_pct:>+11.2f}%")
            any_dips = True

if not any_dips:
    print("  (none in last 30 min)")

print("\n" + "=" * 70)
print(f"  Scan complete at {datetime.now().strftime('%H:%M:%S')} IST")
print("=" * 70)
