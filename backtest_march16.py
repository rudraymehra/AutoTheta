"""Backtest — Simulate both strategies on March 16, 2026 (Monday) data.

March 16 = Monday (RSI Bounce runs, Expiry Skew does NOT — expiry is Tuesday March 17)
March 17 = Tuesday (Expiry day — Expiry Skew would run at 2 PM)

This script:
1. Authenticates with Angel One
2. Fetches real 1-min historical candles for March 16
3. Runs RSI(7) oversold bounce logic across Nifty 50 stocks
4. Fetches March 17 option chain data for expiry skew simulation
5. Reports all signals, entries, exits, and P&L
"""

import time
import json
from datetime import datetime, date
from collections import defaultdict

import pandas as pd
import pyotp
from SmartApi import SmartConnect
from dotenv import load_dotenv
import os

load_dotenv("/Users/rudraym/Trader/.env")

API_KEY = os.getenv("ANGEL_API_KEY")
CLIENT_ID = os.getenv("ANGEL_CLIENT_ID")
PASSWORD = os.getenv("ANGEL_PASSWORD")
TOTP_SECRET = os.getenv("ANGEL_TOTP_SECRET")

# ── Top Nifty 50 stocks to test (symbol -> expected token, we'll look up actual) ──
NIFTY50_TEST = [
    "SBIN-EQ", "HDFCBANK-EQ", "RELIANCE-EQ", "ICICIBANK-EQ", "INFY-EQ",
    "TCS-EQ", "KOTAKBANK-EQ", "LT-EQ", "ITC-EQ", "AXISBANK-EQ",
    "BHARTIARTL-EQ", "HINDUNILVR-EQ", "BAJFINANCE-EQ", "MARUTI-EQ",
    "SUNPHARMA-EQ", "TITAN-EQ", "HCLTECH-EQ", "M&M-EQ",
    "WIPRO-EQ", "TATASTEEL-EQ",
]

SECTOR_MAP = {
    "HDFCBANK-EQ": "Banking", "ICICIBANK-EQ": "Banking", "KOTAKBANK-EQ": "Banking",
    "SBIN-EQ": "Banking", "AXISBANK-EQ": "Banking",
    "BAJFINANCE-EQ": "Finance",
    "RELIANCE-EQ": "Energy",
    "TCS-EQ": "IT", "INFY-EQ": "IT", "WIPRO-EQ": "IT", "HCLTECH-EQ": "IT",
    "HINDUNILVR-EQ": "FMCG", "ITC-EQ": "FMCG",
    "SUNPHARMA-EQ": "Pharma",
    "TATAMOTORS-EQ": "Auto", "MARUTI-EQ": "Auto",
    "TATASTEEL-EQ": "Metals",
    "LT-EQ": "Infra",
    "BHARTIARTL-EQ": "Telecom",
    "TITAN-EQ": "Other",
}


def rsi(series, period=7):
    """Calculate RSI using Wilder's smoothing."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def atr(high, low, close, period=14):
    """Calculate ATR."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def vwap(df):
    """Cumulative VWAP."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (tp * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum().replace(0, 1)
    return cum_tp_vol / cum_vol


# ═══════════════════════════════════════
# AUTHENTICATE
# ═══════════════════════════════════════
print("=" * 60)
print("AutoTheta Backtest — March 16, 2026")
print("=" * 60)

api = SmartConnect(API_KEY)
totp = pyotp.TOTP(TOTP_SECRET).now()
session = api.generateSession(CLIENT_ID, PASSWORD, totp)
if not session.get("status"):
    print(f"AUTH FAILED: {session}")
    exit(1)
print("✓ Authenticated\n")


# ═══════════════════════════════════════
# LOAD INSTRUMENT MASTER (for token lookup)
# ═══════════════════════════════════════
print("Loading instrument master...")
import requests
master_url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"

# Check if cached
cache_path = "/Users/rudraym/Trader/data/instruments.json"
try:
    cache_mtime = datetime.fromtimestamp(os.path.getmtime(cache_path)).date()
    if cache_mtime == date.today():
        with open(cache_path) as f:
            master_data = json.load(f)
        print(f"✓ Loaded from cache ({len(master_data)} instruments)")
    else:
        raise FileNotFoundError
except (FileNotFoundError, OSError):
    print("Downloading fresh instrument master (~80MB)...")
    resp = requests.get(master_url, timeout=120)
    master_data = resp.json()
    with open(cache_path, "w") as f:
        json.dump(master_data, f)
    print(f"✓ Downloaded ({len(master_data)} instruments)")

master_df = pd.DataFrame(master_data)

# Look up tokens for our test stocks
token_map = {}
for sym in NIFTY50_TEST:
    matches = master_df[(master_df["symbol"] == sym) & (master_df["exch_seg"] == "NSE")]
    if not matches.empty:
        token_map[sym] = matches.iloc[0]["token"]
    else:
        print(f"  ⚠ Token not found for {sym}")

print(f"✓ Mapped {len(token_map)} stock tokens\n")


# ═══════════════════════════════════════
# FETCH 1-MIN CANDLES FOR MARCH 16
# ═══════════════════════════════════════
print("Fetching 1-min candles for March 16, 2026...")
print("-" * 60)

stock_data = {}
for sym, token in token_map.items():
    time.sleep(0.4)  # Rate limit: 3 req/sec for historical
    try:
        params = {
            "exchange": "NSE",
            "symboltoken": token,
            "interval": "ONE_MINUTE",
            "fromdate": "2026-03-16 09:15",
            "todate": "2026-03-16 15:30",
        }
        result = api.getCandleData(params)
        if result and result.get("data"):
            candles = result["data"]
            df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            stock_data[sym] = df
            print(f"  {sym:20s} → {len(df)} candles | "
                  f"Open: ₹{df['open'].iloc[0]:.2f} | Close: ₹{df['close'].iloc[-1]:.2f}")
        else:
            print(f"  {sym:20s} → No data")
    except Exception as e:
        print(f"  {sym:20s} → Error: {e}")

print(f"\n✓ Fetched data for {len(stock_data)} stocks\n")


# ═══════════════════════════════════════
# RSI BOUNCE STRATEGY BACKTEST
# ═══════════════════════════════════════
print("=" * 60)
print("STRATEGY 1: RSI(7) Oversold Bounce")
print("=" * 60)

RSI_PERIOD = 7
OVERSOLD = 20
EXIT_1_RSI = 40
EXIT_2_RSI = 50
ATR_SL_MULT = 1.5
TIME_STOP = 15  # candles
SKIP_FIRST = 15  # minutes from 9:15
RISK_PER_TRADE = 2500  # 1% of 2.5L
MAX_POSITIONS = 4
MAX_PER_SECTOR = 1

trades = []
active_positions = {}
sector_count = defaultdict(int)
signals_found = 0

for sym, df in stock_data.items():
    if len(df) < 20:
        continue

    # Calculate indicators
    df["rsi"] = rsi(df["close"], RSI_PERIOD)
    df["atr"] = atr(df["high"], df["low"], df["close"], 14)
    df["vwap"] = vwap(df)
    df["vol_avg20"] = df["volume"].rolling(20).mean()

    # 5-min EMA20 (resample 1-min → 5-min, then EMA20)
    df_5m = df.set_index("timestamp").resample("5min").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna().reset_index()
    df_5m["ema20"] = ema(df_5m["close"], 20)

    # Map 5-min EMA20 back to 1-min candles (forward-fill)
    df["ema20_5m"] = None
    for _, bar in df_5m.iterrows():
        mask = (df["timestamp"] >= bar["timestamp"]) & (df["timestamp"] < bar["timestamp"] + pd.Timedelta(minutes=5))
        df.loc[mask, "ema20_5m"] = bar["ema20"]
    df["ema20_5m"] = df["ema20_5m"].ffill()

    for i in range(20, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        ts = row["timestamp"]

        # Skip first 15 minutes (9:15 - 9:30)
        if ts.hour == 9 and ts.minute < 30:
            continue
        # Don't enter after 3:10 PM
        if ts.hour == 15 and ts.minute > 10:
            continue

        # ── Check exits for active positions ──
        for tid in list(active_positions.keys()):
            pos = active_positions[tid]
            if pos["symbol"] != sym:
                continue
            pos["candles_held"] += 1

            current_rsi = row["rsi"]
            exit_reason = None
            exit_qty = 0
            exit_price = row["close"]

            # Stop-loss
            if row["close"] <= pos["stop_loss"]:
                exit_reason = "STOP_LOSS"
                exit_qty = pos["remaining"]

            # Time stop
            elif pos["candles_held"] >= TIME_STOP and current_rsi < EXIT_1_RSI:
                exit_reason = "TIME_STOP"
                exit_qty = pos["remaining"]

            # Partial exit at RSI 40
            elif pos["status"] == "open" and current_rsi >= EXIT_1_RSI:
                exit_reason = "RSI_40"
                exit_qty = pos["remaining"] // 2
                if exit_qty > 0:
                    pos["remaining"] -= exit_qty
                    pos["status"] = "partial"

            # Final exit at RSI 50
            elif pos["status"] == "partial" and current_rsi >= EXIT_2_RSI:
                exit_reason = "RSI_50"
                exit_qty = pos["remaining"]

            if exit_reason and exit_qty > 0:
                pnl = (exit_price - pos["entry_price"]) * exit_qty
                pos["realized_pnl"] += pnl
                pos["remaining"] -= exit_qty if exit_reason != "RSI_40" else 0

                trades.append({
                    "symbol": sym, "entry_time": pos["entry_time"],
                    "entry_price": pos["entry_price"], "exit_time": str(ts),
                    "exit_price": exit_price, "quantity": exit_qty,
                    "reason": exit_reason, "pnl": pnl,
                    "entry_rsi": pos["entry_rsi"],
                })

                if pos["remaining"] <= 0:
                    sector = SECTOR_MAP.get(sym, "Other")
                    sector_count[sector] = max(0, sector_count.get(sector, 0) - 1)
                    del active_positions[tid]

        # ── Check entry signals ──
        if pd.isna(row["rsi"]) or pd.isna(prev["rsi"]):
            continue

        # RSI crosses below 20
        if not (prev["rsi"] >= OVERSOLD and row["rsi"] < OVERSOLD):
            continue

        signals_found += 1

        # Already in position?
        if any(p["symbol"] == sym for p in active_positions.values()):
            continue

        # Max positions
        if len(active_positions) >= MAX_POSITIONS:
            continue

        # Sector limit
        sector = SECTOR_MAP.get(sym, "Other")
        if sector_count.get(sector, 0) >= MAX_PER_SECTOR:
            continue

        # Filter 1: Price above 5-min EMA(20)
        if pd.notna(row["ema20_5m"]) and row["close"] < row["ema20_5m"]:
            print(f"  [{ts}] {sym} RSI={row['rsi']:.1f} — FILTERED (below 5m EMA20)")
            continue

        # Filter 2: Price above VWAP
        if row["close"] < row["vwap"] * 0.998:
            print(f"  [{ts}] {sym} RSI={row['rsi']:.1f} — FILTERED (below VWAP)")
            continue

        # Filter 3: Volume > 1.5x average
        if pd.notna(row["vol_avg20"]) and row["vol_avg20"] > 0:
            if row["volume"] < row["vol_avg20"] * 1.5:
                print(f"  [{ts}] {sym} RSI={row['rsi']:.1f} — FILTERED (low volume)")
                continue

        # Calculate position size
        current_atr = row["atr"] if pd.notna(row["atr"]) else row["close"] * 0.005
        stop_loss = round(row["close"] - ATR_SL_MULT * current_atr, 2)
        risk_per_share = row["close"] - stop_loss
        if risk_per_share <= 0:
            continue
        qty = int(RISK_PER_TRADE / risk_per_share)
        qty = min(qty, int(83000 / row["close"]))  # Cap position value
        if qty <= 0:
            continue

        # ENTRY!
        tid = f"BT-{sym}-{i}"
        active_positions[tid] = {
            "symbol": sym, "entry_price": row["close"], "entry_time": str(ts),
            "stop_loss": stop_loss, "quantity": qty, "remaining": qty,
            "candles_held": 0, "status": "open", "realized_pnl": 0.0,
            "entry_rsi": round(row["rsi"], 1),
        }
        sector_count[sector] = sector_count.get(sector, 0) + 1

        print(f"  ✦ [{ts}] BUY {sym} x{qty} @ ₹{row['close']:.2f} | "
              f"RSI={row['rsi']:.1f} | SL=₹{stop_loss:.2f} | ATR=₹{current_atr:.2f}")

# Force close any remaining positions at 3:25 PM price
for tid, pos in list(active_positions.items()):
    sym = pos["symbol"]
    df = stock_data.get(sym)
    if df is not None and len(df) > 0:
        last_price = df["close"].iloc[-1]
        pnl = (last_price - pos["entry_price"]) * pos["remaining"]
        trades.append({
            "symbol": sym, "entry_time": pos["entry_time"],
            "entry_price": pos["entry_price"], "exit_time": "15:30 (EOD)",
            "exit_price": last_price, "quantity": pos["remaining"],
            "reason": "EOD_CLOSE", "pnl": pnl,
            "entry_rsi": pos["entry_rsi"],
        })

# ── Results ──
print("\n" + "=" * 60)
print("RSI BOUNCE — RESULTS")
print("=" * 60)
print(f"Signals found (RSI < 20):  {signals_found}")
print(f"Trades taken:              {len(trades)}")

if trades:
    total_pnl = sum(t["pnl"] for t in trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]

    print(f"Winning trades:            {len(wins)}")
    print(f"Losing trades:             {len(losses)}")
    print(f"Win rate:                  {len(wins)/len(trades)*100:.1f}%")
    print(f"Total P&L:                 ₹{total_pnl:,.2f}")
    if wins:
        print(f"Avg win:                   ₹{sum(t['pnl'] for t in wins)/len(wins):,.2f}")
    if losses:
        print(f"Avg loss:                  ₹{sum(t['pnl'] for t in losses)/len(losses):,.2f}")

    print("\n── Trade Log ──")
    for t in trades:
        emoji = "✓" if t["pnl"] > 0 else "✗"
        print(f"  {emoji} {t['symbol']:15s} | Entry RSI={t['entry_rsi']} | "
              f"₹{t['entry_price']:.2f} → ₹{t['exit_price']:.2f} | "
              f"{t['reason']:12s} | P&L: ₹{t['pnl']:+,.2f}")
else:
    print("  No trades triggered — all signals were filtered out")
    print("  (This is normal — the filter stack is strict by design)")


# ═══════════════════════════════════════
# EXPIRY SKEW PREVIEW (March 17 data)
# ═══════════════════════════════════════
print("\n" + "=" * 60)
print("STRATEGY 2: Expiry Skew Iron Condor (Preview for March 17)")
print("=" * 60)
print("March 16 is Monday — NOT an expiry day.")
print("March 17 (Tuesday) IS the expiry day.")
print("\nFetching Nifty spot to preview potential strikes...\n")

time.sleep(0.5)
try:
    spot_data = api.ltpData("NSE", "NIFTY", "99926000")
    nifty_spot = float(spot_data["data"]["ltp"])
    atm = round(nifty_spot / 50) * 50
    otm_put = atm - 50
    otm_call = atm + 50
    buy_put = otm_put - 100
    buy_call = otm_call + 100

    print(f"Nifty spot (current):  ₹{nifty_spot:,.2f}")
    print(f"ATM strike:            {atm}")
    print(f"Iron Condor legs:")
    print(f"  Sell {otm_put}PE + Buy {buy_put}PE")
    print(f"  Sell {otm_call}CE + Buy {buy_call}CE")

    # Try to fetch current option premiums for the nearest expiry
    master_df["expiry_dt"] = pd.to_datetime(master_df["expiry"], format="mixed", dayfirst=True).dt.date
    master_df["actual_strike"] = master_df["strike"].astype(float) / 100

    nifty_opts = master_df[
        (master_df["name"] == "NIFTY")
        & (master_df["instrumenttype"] == "OPTIDX")
        & (master_df["exch_seg"] == "NFO")
    ]

    nearest_expiry = nifty_opts[nifty_opts["expiry_dt"] >= date.today()]["expiry_dt"].min()
    print(f"  Nearest expiry:      {nearest_expiry}")

    expiry_chain = nifty_opts[nifty_opts["expiry_dt"] == nearest_expiry]

    # Fetch premiums for all 4 legs
    legs = {}
    for label, strike, opt_type in [
        ("Sell PUT", otm_put, "PE"), ("Sell CALL", otm_call, "CE"),
        ("Buy PUT", buy_put, "PE"), ("Buy CALL", buy_call, "CE"),
    ]:
        matches = expiry_chain[
            (expiry_chain["actual_strike"] == strike)
            & (expiry_chain["symbol"].str.endswith(opt_type))
        ]
        if not matches.empty:
            row = matches.iloc[0]
            time.sleep(0.35)
            try:
                ltp_data = api.ltpData("NFO", row["symbol"], row["token"])
                premium = float(ltp_data["data"]["ltp"])
                legs[label] = {"symbol": row["symbol"], "premium": premium, "strike": strike}
                print(f"  {label:12s} {strike}{opt_type}: ₹{premium:.2f} ({row['symbol']})")
            except Exception as e:
                print(f"  {label:12s} {strike}{opt_type}: Error fetching LTP — {e}")
        else:
            print(f"  {label:12s} {strike}{opt_type}: Not found in chain")

    if len(legs) == 4:
        net_credit = (legs["Sell PUT"]["premium"] + legs["Sell CALL"]["premium"]
                      - legs["Buy PUT"]["premium"] - legs["Buy CALL"]["premium"])
        max_profit = net_credit * 65  # Per lot
        max_loss = (100 - net_credit) * 65  # Wing width - net credit

        sp = legs["Sell PUT"]["premium"]
        sc = legs["Sell CALL"]["premium"]
        skew = max(sp, sc) / max(min(sp, sc), 0.05)

        print(f"\n  Net credit/unit:     ₹{net_credit:.2f}")
        print(f"  Max profit/lot:      ₹{max_profit:,.2f}")
        print(f"  Max loss/lot:        ₹{max_loss:,.2f}")
        print(f"  Risk:Reward:         1:{max_profit/max_loss:.2f}")
        print(f"  Premium skew ratio:  {skew:.1f}x")
        if skew >= 2.0:
            print(f"  Signal:              ✓ TRADE (skew >= 2.0)")
        else:
            print(f"  Signal:              ✗ NO TRADE (skew < 2.0)")

except Exception as e:
    print(f"Error fetching Nifty data: {e}")

# ── VIX Check ──
time.sleep(0.35)
try:
    vix_data = api.ltpData("NSE", "India VIX", "99926017")
    vix = float(vix_data["data"]["ltp"])
    print(f"\n  India VIX:           {vix:.2f}", end="")
    if 12 <= vix <= 18:
        print(" ✓ (within 12-18 range)")
    else:
        print(f" ✗ (outside 12-18 range — would SKIP trade)")
except Exception:
    print("\n  India VIX:           Could not fetch")

print("\n" + "=" * 60)
print("Backtest complete.")
print("=" * 60)
