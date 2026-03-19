"""Simulate a range of trading days and produce a summary table.

v2.0 — Research-backed parameters:
  S1: RSI(4) on 5-min, hook above 15, VWAP below, 3x ATR, 75-min time stop
  S3: RSI(9)<40 on 15-min, RSI(9) cross above 25 on 5-min, ADX(14)<30, 3x ATR

Usage: python simulate_range.py 2026-02-03 2026-02-13
"""

import os
import sys
import json
import time
import pickle
from datetime import datetime, date, timedelta
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyotp
from SmartApi import SmartConnect
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# ── Indicators ──
def rsi(series, period=4):
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
    return (tp * df["volume"]).cumsum() / df["volume"].cumsum().replace(0, 1)

def adx_calc(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr.replace(0, 1e-10))
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr.replace(0, 1e-10))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10))
    return dx.ewm(alpha=1/period, min_periods=period).mean()


# v2.0 stock universe — dropped metals/energy
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

STOCKS = list(SECTOR_MAP.keys())
CAPITAL = 250000
RISK_PER_TRADE = 2500


def get_trading_days(start_date, end_date):
    """Return weekdays between start and end (inclusive)."""
    days = []
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:  # Mon-Fri
            days.append(current)
        current += timedelta(days=1)
    return days


def fetch_day(api, token_map, target_date):
    """Fetch 1-min candles for one day."""
    cache = PROJECT_ROOT / "data" / f"cache_{target_date}.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    data = {}
    ds = str(target_date)
    for sym, tok in token_map.items():
        time.sleep(1.2)
        try:
            r = api.getCandleData({
                "exchange": "NSE", "symboltoken": tok, "interval": "ONE_MINUTE",
                "fromdate": f"{ds} 09:15", "todate": f"{ds} 15:30",
            })
            if r and r.get("data") and len(r["data"]) > 50:
                df = pd.DataFrame(r["data"], columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                data[sym] = df
        except Exception:
            pass

    if data:
        cache.parent.mkdir(exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump(data, f)
    return data


def simulate_one_day(data):
    """Run both strategies on one day's data. Returns dict of results."""
    if not data:
        return None

    # Precompute indicators
    for sym, df in data.items():
        df["atr14"] = atr_calc(df["high"], df["low"], df["close"], 14)
        df["ema200"] = ema(df["close"], 200)
        df["vwap"] = vwap_calc(df)

        # 5-min resampled
        df_5m = df.set_index("timestamp").resample("5min").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
        }).dropna().reset_index()
        if len(df_5m) >= 5:
            df_5m["rsi4"] = rsi(df_5m["close"], 4)
            df["rsi4_5m"] = None
            for _, bar in df_5m.iterrows():
                mask = (df["timestamp"] >= bar["timestamp"]) & (df["timestamp"] < bar["timestamp"] + pd.Timedelta(minutes=5))
                df.loc[mask, "rsi4_5m"] = bar.get("rsi4")
            df["rsi4_5m"] = df["rsi4_5m"].ffill()

        # 15-min resampled
        df_15m = df.set_index("timestamp").resample("15min").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
        }).dropna().reset_index()
        if len(df_15m) >= 5:
            df_15m["adx14"] = adx_calc(df_15m["high"], df_15m["low"], df_15m["close"], 14)
            df_15m["atr14"] = atr_calc(df_15m["high"], df_15m["low"], df_15m["close"], 14)
            df_15m["rsi9"] = rsi(df_15m["close"], 9)
            for col in ["adx14", "atr14", "rsi9"]:
                df[f"{col}_15m"] = None
                for _, bar in df_15m.iterrows():
                    mask = (df["timestamp"] >= bar["timestamp"]) & (df["timestamp"] < bar["timestamp"] + pd.Timedelta(minutes=15))
                    df.loc[mask, f"{col}_15m"] = bar.get(col)
                df[f"{col}_15m"] = df[f"{col}_15m"].ffill()

        # 5-min RSI(9) for S3
        if len(df_5m) >= 10:
            df_5m["rsi9"] = rsi(df_5m["close"], 9)
            df["rsi9_5m"] = None
            for _, bar in df_5m.iterrows():
                mask = (df["timestamp"] >= bar["timestamp"]) & (df["timestamp"] < bar["timestamp"] + pd.Timedelta(minutes=5))
                df.loc[mask, "rsi9_5m"] = bar.get("rsi9")
            df["rsi9_5m"] = df["rsi9_5m"].ffill()

    # State
    positions = {}
    closed = []
    sector_count = defaultdict(int)
    trade_count = 0
    s1_signals = 0
    s3_setups = 0

    max_candles = max(len(df) for df in data.values())
    sample_sym = list(data.keys())[0]

    for i in range(20, max_candles):
        if i >= len(data[sample_sym]):
            break
        ts = data[sample_sym]["timestamp"].iloc[i]
        hour, minute = ts.hour, ts.minute

        if hour < 9 or (hour == 9 and minute < 30):
            continue
        if hour >= 15 and minute > 10:
            for tid in list(positions.keys()):
                pos = positions[tid]
                sym = pos["symbol"]
                if sym in data and i < len(data[sym]):
                    pnl = (data[sym]["close"].iloc[i] - pos["entry_price"]) * pos["remaining"]
                    pos["realized_pnl"] += pnl
                    closed.append({**pos, "exit_price": data[sym]["close"].iloc[i], "exit_reason": "EOD"})
                    sector_count[pos["sector"]] = max(0, sector_count[pos["sector"]] - 1)
                    del positions[tid]
            break

        # Exits
        for tid in list(positions.keys()):
            pos = positions.get(tid)
            if not pos:
                continue
            sym = pos["symbol"]
            if sym not in data or i >= len(data[sym]):
                continue
            row = data[sym].iloc[i]
            pos["candles_held"] += 1

            if pos["strategy"] == "S1":
                r4 = row.get("rsi4_5m")
                if row["close"] <= pos["stop_loss"]:
                    pos["realized_pnl"] += (row["close"] - pos["entry_price"]) * pos["remaining"]
                    closed.append({**pos, "exit_price": row["close"], "exit_reason": "DISASTER_SL"})
                    sector_count[pos["sector"]] = max(0, sector_count[pos["sector"]] - 1)
                    del positions[tid]
                    continue
                if hour >= 15:
                    pos["realized_pnl"] += (row["close"] - pos["entry_price"]) * pos["remaining"]
                    closed.append({**pos, "exit_price": row["close"], "exit_reason": "HARD_3PM"})
                    sector_count[pos["sector"]] = max(0, sector_count[pos["sector"]] - 1)
                    del positions[tid]
                    continue
                if pos["candles_held"] >= 75:
                    pos["realized_pnl"] += (row["close"] - pos["entry_price"]) * pos["remaining"]
                    closed.append({**pos, "exit_price": row["close"], "exit_reason": "TIME75"})
                    sector_count[pos["sector"]] = max(0, sector_count[pos["sector"]] - 1)
                    del positions[tid]
                    continue
                # VWAP touch: 60% exit
                if not pos.get("vwap_exit_done") and row["close"] >= row["vwap"]:
                    eq = int(pos["initial_qty"] * 0.6)
                    eq = min(eq, pos["remaining"])
                    if eq > 0:
                        pos["realized_pnl"] += (row["close"] - pos["entry_price"]) * eq
                        pos["remaining"] -= eq
                        pos["vwap_exit_done"] = True
                        if pos["remaining"] <= 0:
                            closed.append({**pos, "exit_price": row["close"], "exit_reason": "VWAP60"})
                            sector_count[pos["sector"]] = max(0, sector_count[pos["sector"]] - 1)
                            del positions[tid]
                            continue
                # RSI(4) > 50: remaining exit
                if not pd.isna(r4) and r4 >= 50:
                    pos["realized_pnl"] += (row["close"] - pos["entry_price"]) * pos["remaining"]
                    closed.append({**pos, "exit_price": row["close"], "exit_reason": "RSI4_50"})
                    sector_count[pos["sector"]] = max(0, sector_count[pos["sector"]] - 1)
                    del positions[tid]

            elif pos["strategy"] == "S3":
                r9 = row.get("rsi9_5m")
                if row["close"] <= pos["stop_loss"]:
                    pos["realized_pnl"] += (row["close"] - pos["entry_price"]) * pos["remaining"]
                    closed.append({**pos, "exit_price": row["close"], "exit_reason": "S3_SL"})
                    sector_count[pos["sector"]] = max(0, sector_count[pos["sector"]] - 1)
                    del positions[tid]
                    continue
                if hour > 14 or (hour == 14 and minute >= 30):
                    pos["realized_pnl"] += (row["close"] - pos["entry_price"]) * pos["remaining"]
                    closed.append({**pos, "exit_price": row["close"], "exit_reason": "S3_EXIT"})
                    sector_count[pos["sector"]] = max(0, sector_count[pos["sector"]] - 1)
                    del positions[tid]
                    continue
                if pos["candles_held"] >= 75:
                    pos["realized_pnl"] += (row["close"] - pos["entry_price"]) * pos["remaining"]
                    closed.append({**pos, "exit_price": row["close"], "exit_reason": "S3_TIME75"})
                    sector_count[pos["sector"]] = max(0, sector_count[pos["sector"]] - 1)
                    del positions[tid]
                    continue
                if not pd.isna(r9) and r9 >= 50:
                    pos["realized_pnl"] += (row["close"] - pos["entry_price"]) * pos["remaining"]
                    closed.append({**pos, "exit_price": row["close"], "exit_reason": "S3_RSI50"})
                    sector_count[pos["sector"]] = max(0, sector_count[pos["sector"]] - 1)
                    del positions[tid]
                    continue
                if row["close"] >= row["vwap"] and pos.get("entered_below_vwap"):
                    pos["realized_pnl"] += (row["close"] - pos["entry_price"]) * pos["remaining"]
                    closed.append({**pos, "exit_price": row["close"], "exit_reason": "S3_VWAP"})
                    sector_count[pos["sector"]] = max(0, sector_count[pos["sector"]] - 1)
                    del positions[tid]

        # S1 entries — 10:15-14:00 only
        now_min = hour * 60 + minute
        s1_entry_ok = (10 * 60 + 15) <= now_min < (14 * 60)

        if s1_entry_ok:
            for sym, df in data.items():
                if i >= len(df) or i < 1:
                    continue
                row = df.iloc[i]
                prev = df.iloc[i - 1]
                curr_r4 = row.get("rsi4_5m")
                prev_r4 = prev.get("rsi4_5m")
                if curr_r4 is None or prev_r4 is None:
                    continue
                if pd.isna(curr_r4) or pd.isna(prev_r4):
                    continue
                # RSI(4) hook: prev < 15 AND current >= 15
                if not (prev_r4 < 15 and curr_r4 >= 15):
                    continue
                s1_signals += 1
                if any(p["symbol"] == sym for p in positions.values()):
                    continue
                s1_pos = sum(1 for p in positions.values() if p["strategy"] == "S1")
                if s1_pos >= 3:
                    continue
                sector = SECTOR_MAP.get(sym, "Other")
                if sector_count.get(sector, 0) >= 1:
                    continue
                # EMA(200) filter
                e200 = row.get("ema200")
                if pd.notna(e200) and row["close"] < e200:
                    continue
                # ADX filter — range-bound only
                adx15 = row.get("adx14_15m")
                if pd.notna(adx15) and adx15 >= 30:
                    continue
                # VWAP below: 0.1%-2.0% (wider window)
                vd = (row["close"] - row["vwap"]) / max(row["vwap"], 1) if row["vwap"] > 0 else 0
                if not (-0.020 <= vd <= -0.001):
                    continue
                # 3x ATR on 15-min
                a15 = row.get("atr14_15m")
                a = a15 if pd.notna(a15) else row["close"] * 0.01
                sl = round(row["close"] - 3.0 * a, 2)
                risk = row["close"] - sl
                if risk <= 0:
                    continue
                qty = min(int(RISK_PER_TRADE / risk), int(83000 / row["close"]))
                if qty <= 0:
                    continue
                trade_count += 1
                positions[f"S1-{trade_count}"] = {
                    "strategy": "S1", "symbol": sym, "entry_price": row["close"],
                    "quantity": qty, "initial_qty": qty, "remaining": qty,
                    "stop_loss": sl, "status": "open",
                    "entry_time": ts.strftime("%H:%M"), "candles_held": 0,
                    "realized_pnl": 0.0, "sector": sector,
                    "vwap_exit_done": False,
                }
                sector_count[sector] += 1

        # S3 entries
        in_prime = (hour == 10 and minute >= 15) or hour == 11
        in_secondary = (hour == 13 and minute >= 30) or (hour == 14 and minute < 30)
        if in_prime or in_secondary:
            for sym, df in data.items():
                if i >= len(df):
                    continue
                row = df.iloc[i]
                r9_15 = row.get("rsi9_15m")
                adx_15 = row.get("adx14_15m")
                r9_5 = row.get("rsi9_5m")
                atr_15 = row.get("atr14_15m")
                if any(v is None or (isinstance(v, float) and pd.isna(v)) for v in [r9_15, adx_15, r9_5]):
                    continue
                # RSI(9) < 40, ADX(14) < 30
                if r9_15 >= 40 or adx_15 >= 30:
                    continue
                s3_setups += 1
                prev_r9 = df.iloc[i-1].get("rsi9_5m") if i > 0 else None
                if prev_r9 is None or pd.isna(prev_r9) or not (prev_r9 < 25 and r9_5 >= 25):
                    continue
                if row["close"] <= row["open"]:
                    continue
                # Below VWAP
                if row["close"] >= row["vwap"]:
                    continue
                if any(p["symbol"] == sym for p in positions.values()):
                    continue
                s3_pos = sum(1 for p in positions.values() if p["strategy"] == "S3")
                if s3_pos >= 3:
                    continue
                a15 = atr_15 if pd.notna(atr_15) else row["close"] * 0.008
                sl = round(row["close"] - 3.0 * a15, 2)
                risk = row["close"] - sl
                if risk <= 0:
                    continue
                mult = 0.5 if in_secondary else 1.0
                qty = min(int(RISK_PER_TRADE * mult / risk), int(83000 / row["close"]))
                if qty <= 0:
                    continue
                trade_count += 1
                sector = SECTOR_MAP.get(sym, "Other")
                positions[f"S3-{trade_count}"] = {
                    "strategy": "S3", "symbol": sym, "entry_price": row["close"],
                    "quantity": qty, "initial_qty": qty, "remaining": qty,
                    "stop_loss": sl, "status": "open",
                    "entry_time": ts.strftime("%H:%M"), "candles_held": 0,
                    "realized_pnl": 0.0, "sector": sector,
                    "entered_below_vwap": row["close"] < row["vwap"],
                }
                sector_count[sector] += 1

    s1_trades = [t for t in closed if t["strategy"] == "S1"]
    s3_trades = [t for t in closed if t["strategy"] == "S3"]
    s1_pnl = sum(t["realized_pnl"] for t in s1_trades)
    s3_pnl = sum(t["realized_pnl"] for t in s3_trades)
    total_pnl = s1_pnl + s3_pnl
    s1_wins = len([t for t in s1_trades if t["realized_pnl"] > 0])
    s3_wins = len([t for t in s3_trades if t["realized_pnl"] > 0])

    return {
        "s1_signals": s1_signals, "s1_trades": len(s1_trades), "s1_wins": s1_wins,
        "s1_pnl": s1_pnl, "s1_details": s1_trades,
        "s3_setups": s3_setups, "s3_trades": len(s3_trades), "s3_wins": s3_wins,
        "s3_pnl": s3_pnl, "s3_details": s3_trades,
        "total_pnl": total_pnl,
    }


def main():
    if len(sys.argv) < 3:
        print("Usage: python simulate_range.py 2026-02-03 2026-02-13")
        return

    start = date.fromisoformat(sys.argv[1])
    end = date.fromisoformat(sys.argv[2])
    days = get_trading_days(start, end)

    print(f"{'='*80}")
    print(f"  AutoTheta Range Simulation v2.0: {start} to {end} ({len(days)} trading days)")
    print(f"{'='*80}")

    # Auth
    api = SmartConnect(os.getenv("ANGEL_API_KEY"))
    totp = pyotp.TOTP(os.getenv("ANGEL_TOTP_SECRET")).now()
    api.generateSession(os.getenv("ANGEL_CLIENT_ID"), os.getenv("ANGEL_PASSWORD"), totp)

    with open(PROJECT_ROOT / "data" / "instruments.json") as f:
        master = json.load(f)
    mdf = pd.DataFrame(master)
    token_map = {}
    for sym in STOCKS:
        m = mdf[(mdf["symbol"] == sym) & (mdf["exch_seg"] == "NSE")]
        if not m.empty:
            token_map[sym] = m.iloc[0]["token"]

    print(f"  {len(token_map)} stocks | Fetching data...\n")

    results = {}
    for d in days:
        ds = d.isoformat()
        sys.stdout.write(f"  {ds} ({d.strftime('%A')[:3]})... ")
        sys.stdout.flush()
        data = fetch_day(api, token_map, d)
        if not data:
            print("HOLIDAY/NO DATA")
            continue
        r = simulate_one_day(data)
        if r:
            results[ds] = r
            total = r["s1_trades"] + r["s3_trades"]
            print(f"S1:{r['s1_trades']}t/{r['s1_signals']}s S3:{r['s3_trades']}t/{r['s3_setups']}s | P&L: Rs{r['total_pnl']:+,.2f}")
        else:
            print("NO DATA")

    # ── Summary Table ──
    print(f"\n{'='*80}")
    print(f"  RESULTS v2.0: {start} to {end}")
    print(f"{'='*80}")
    print()
    print(f"  {'Date':12s} {'Day':4s} {'S1 Sig':>7s} {'S1 Trd':>7s} {'S1 W/L':>7s} {'S1 P&L':>10s} {'S3 Set':>7s} {'S3 Trd':>7s} {'S3 W/L':>7s} {'S3 P&L':>10s} {'TOTAL':>10s}")
    print(f"  {'─'*76}")

    cum_pnl = 0
    total_s1_trades = 0
    total_s1_wins = 0
    total_s3_trades = 0
    total_s3_wins = 0
    total_s1_pnl = 0
    total_s3_pnl = 0

    for ds, r in results.items():
        d = date.fromisoformat(ds)
        day_name = d.strftime("%a")
        s1_wl = f"{r['s1_wins']}/{r['s1_trades']-r['s1_wins']}" if r["s1_trades"] else "—"
        s3_wl = f"{r['s3_wins']}/{r['s3_trades']-r['s3_wins']}" if r["s3_trades"] else "—"
        cum_pnl += r["total_pnl"]

        total_s1_trades += r["s1_trades"]
        total_s1_wins += r["s1_wins"]
        total_s3_trades += r["s3_trades"]
        total_s3_wins += r["s3_wins"]
        total_s1_pnl += r["s1_pnl"]
        total_s3_pnl += r["s3_pnl"]

        print(f"  {ds:12s} {day_name:4s} {r['s1_signals']:>7d} {r['s1_trades']:>7d} {s1_wl:>7s} {r['s1_pnl']:>+10,.2f} "
              f"{r['s3_setups']:>7d} {r['s3_trades']:>7d} {s3_wl:>7s} {r['s3_pnl']:>+10,.2f} {r['total_pnl']:>+10,.2f}")

    print(f"  {'─'*76}")
    s1_wr = f"{total_s1_wins}/{total_s1_trades-total_s1_wins}" if total_s1_trades else "—"
    s3_wr = f"{total_s3_wins}/{total_s3_trades-total_s3_wins}" if total_s3_trades else "—"
    print(f"  {'TOTAL':12s} {'':4s} {'':>7s} {total_s1_trades:>7d} {s1_wr:>7s} {total_s1_pnl:>+10,.2f} "
          f"{'':>7s} {total_s3_trades:>7d} {s3_wr:>7s} {total_s3_pnl:>+10,.2f} {cum_pnl:>+10,.2f}")

    pnl_pct = (cum_pnl / CAPITAL) * 100
    print(f"\n  Capital: Rs{CAPITAL:,} → Rs{CAPITAL + cum_pnl:,.2f} ({pnl_pct:+.2f}%)")
    total_trades = total_s1_trades + total_s3_trades
    total_wins = total_s1_wins + total_s3_wins
    if total_trades:
        print(f"  Total trades: {total_trades} | Win rate: {total_wins}/{total_trades} ({total_wins/total_trades*100:.0f}%)")
        avg_win = cum_pnl / total_trades
        print(f"  Avg P&L per trade: Rs{avg_win:+,.2f}")
    print(f"\n  Trade Details:")
    for ds, r in results.items():
        for t in r["s1_details"] + r["s3_details"]:
            m = "W" if t["realized_pnl"] > 0 else "L"
            print(f"    [{m}] {ds} {t['entry_time']:>5s} {t['strategy']:3s} {t['symbol']:15s} "
                  f"Rs{t['entry_price']:.2f} → {t['exit_price']:.2f} | {t['exit_reason']:8s} | Rs{t['realized_pnl']:+,.2f}")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
