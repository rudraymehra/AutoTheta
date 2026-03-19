"""Simulate a full trading day — replay 1-min candles as if it's live.

Tests all strategies: S1 (RSI(4) Mean Reversion on 5-min), S3 (15-min Mean Reversion)
Also checks S2 (Expiry Skew) conditions if it's a Tuesday.

v2.0 — Research-backed parameters:
  S1: RSI(4) on 5-min, hook above 15, VWAP below entry, 3x ATR stop, 75-min time stop
  S3: RSI(9) setup<40 on 15-min, entry<25 on 5-min, ADX(14)<30, 3x ATR stop

Usage: python simulate_day.py [YYYY-MM-DD]
Default: today
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
    cum_tp_vol = (tp * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum().replace(0, 1)
    return cum_tp_vol / cum_vol

def adx_calc(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr.replace(0, 1e-10))
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr.replace(0, 1e-10))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10))
    return dx.ewm(alpha=1/period, min_periods=period).mean()


# v2.0 stock universe — dropped metals/energy, focus banking/IT/FMCG
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

CAPITAL = 250000
RISK_PER_TRADE = 2500

# ══════════════════════════════════════════
# PORTFOLIO TRACKER
# ══════════════════════════════════════════
class SimPortfolio:
    def __init__(self):
        self.positions = {}  # trade_id -> dict
        self.closed = []
        self.daily_pnl = 0.0
        self.trade_count = 0
        self.sector_count = defaultdict(int)

    def open(self, strategy, symbol, price, qty, stop_loss, indicators, time_str):
        self.trade_count += 1
        tid = f"{strategy}-{self.trade_count:04d}"
        sector = SECTOR_MAP.get(symbol, "Other")
        self.positions[tid] = {
            "strategy": strategy, "symbol": symbol, "entry_price": price,
            "quantity": qty, "initial_qty": qty, "remaining": qty,
            "stop_loss": stop_loss, "status": "open",
            "entry_time": time_str, "candles_held": 0,
            "realized_pnl": 0.0, "indicators": indicators, "sector": sector,
            "vwap_exit_done": False,
        }
        self.sector_count[sector] += 1
        print(f"  >> [{time_str}] {strategy} BUY {symbol} x{qty} @ Rs{price:.2f} | SL={stop_loss:.2f} | {indicators}")
        return tid

    def close(self, tid, price, qty, reason, time_str):
        pos = self.positions.get(tid)
        if not pos:
            return 0
        pnl = (price - pos["entry_price"]) * qty
        pos["realized_pnl"] += pnl
        pos["remaining"] -= qty
        self.daily_pnl += pnl
        marker = "WIN" if pnl > 0 else "LOSS"
        print(f"  << [{time_str}] {pos['strategy']} SELL {pos['symbol']} x{qty} @ Rs{price:.2f} | {reason} | Rs{pnl:+,.2f} [{marker}]")
        if pos["remaining"] <= 0:
            self.sector_count[pos["sector"]] = max(0, self.sector_count[pos["sector"]] - 1)
            self.closed.append({**pos, "exit_price": price, "exit_time": time_str, "exit_reason": reason})
            del self.positions[tid]
        return pnl

    def has_symbol(self, symbol):
        return any(p["symbol"] == symbol for p in self.positions.values())

    def strategy_positions(self, strategy):
        return {k: v for k, v in self.positions.items() if v["strategy"] == strategy}


def fetch_data(target_date):
    """Fetch 1-min candles for all stocks."""
    api = SmartConnect(os.getenv("ANGEL_API_KEY"))
    totp = pyotp.TOTP(os.getenv("ANGEL_TOTP_SECRET")).now()
    api.generateSession(os.getenv("ANGEL_CLIENT_ID"), os.getenv("ANGEL_PASSWORD"), totp)

    with open(PROJECT_ROOT / "data" / "instruments.json") as f:
        master = json.load(f)
    mdf = pd.DataFrame(master)

    stocks = list(SECTOR_MAP.keys())
    token_map = {}
    for sym in stocks:
        m = mdf[(mdf["symbol"] == sym) & (mdf["exch_seg"] == "NSE")]
        if not m.empty:
            token_map[sym] = m.iloc[0]["token"]

    print(f"  Fetching {len(token_map)} stocks for {target_date}...")
    data = {}
    for sym, tok in token_map.items():
        time.sleep(1.5)
        try:
            r = api.getCandleData({
                "exchange": "NSE", "symboltoken": tok, "interval": "ONE_MINUTE",
                "fromdate": f"{target_date} 09:15", "todate": f"{target_date} 15:30",
            })
            if r and r.get("data"):
                df = pd.DataFrame(r["data"], columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                data[sym] = df
                print(f"    {sym:18s} {len(df)} candles")
            else:
                print(f"    {sym:18s} No data")
        except Exception as e:
            print(f"    {sym:18s} Error: {e}")
    return data


def simulate(data, target_date):
    """Replay candles minute by minute, running all strategies."""
    portfolio = SimPortfolio()

    print(f"\n{'='*70}")
    print(f"  FULL DAY SIMULATION v2.0 — {target_date}")
    print(f"  Strategies: S1 (RSI(4) 5-min Mean Reversion) + S3 (15-min Mean Reversion)")
    is_tuesday = datetime.strptime(str(target_date), "%Y-%m-%d").weekday() == 1
    if is_tuesday:
        print(f"  Tuesday = Expiry Day — S2 (Expiry Skew) also checked")
    print(f"  Capital: Rs{CAPITAL:,}")
    print(f"{'='*70}\n")

    # Precompute all indicators for each stock
    for sym, df in data.items():
        df["atr14"] = atr_calc(df["high"], df["low"], df["close"], 14)
        df["ema200"] = ema(df["close"], 200)
        df["vwap"] = vwap_calc(df)

        # 5-min resampled for S1: RSI(4) on 5-min
        df_5m = df.set_index("timestamp").resample("5min").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna().reset_index()
        if len(df_5m) >= 5:
            df_5m["rsi4"] = rsi(df_5m["close"], 4)
            df["rsi4_5m"] = None
            for _, bar in df_5m.iterrows():
                mask = (df["timestamp"] >= bar["timestamp"]) & (df["timestamp"] < bar["timestamp"] + pd.Timedelta(minutes=5))
                df.loc[mask, "rsi4_5m"] = bar.get("rsi4")
            df["rsi4_5m"] = df["rsi4_5m"].ffill()

        # 15-min resampled for filters
        df_15m = df.set_index("timestamp").resample("15min").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna().reset_index()
        if len(df_15m) >= 5:
            df_15m["adx14"] = adx_calc(df_15m["high"], df_15m["low"], df_15m["close"], 14)
            df_15m["atr14"] = atr_calc(df_15m["high"], df_15m["low"], df_15m["close"], 14)
            df_15m["rsi9"] = rsi(df_15m["close"], 9)

            # Map 15-min indicators back
            for col in ["adx14", "atr14", "rsi9"]:
                df[f"{col}_15m"] = None
                for _, bar in df_15m.iterrows():
                    mask = (df["timestamp"] >= bar["timestamp"]) & (df["timestamp"] < bar["timestamp"] + pd.Timedelta(minutes=15))
                    df.loc[mask, f"{col}_15m"] = bar.get(col)
                df[f"{col}_15m"] = df[f"{col}_15m"].ffill()

        # 5-min RSI(9) for S3 entry trigger
        if len(df_5m) >= 10:
            df_5m["rsi9"] = rsi(df_5m["close"], 9)
            df["rsi9_5m"] = None
            for _, bar in df_5m.iterrows():
                mask = (df["timestamp"] >= bar["timestamp"]) & (df["timestamp"] < bar["timestamp"] + pd.Timedelta(minutes=5))
                df.loc[mask, "rsi9_5m"] = bar.get("rsi9")
            df["rsi9_5m"] = df["rsi9_5m"].ffill()

    # ── Replay minute by minute ──
    max_candles = max(len(df) for df in data.values())
    s1_signals = 0
    s1_filtered = 0
    s3_setups = 0
    s3_entries = 0

    for i in range(20, max_candles):  # Start at 20 to have indicator history
        # Get timestamp from first stock
        sample_sym = list(data.keys())[0]
        if i >= len(data[sample_sym]):
            break
        ts = data[sample_sym]["timestamp"].iloc[i]
        ts_str = ts.strftime("%H:%M")
        hour, minute = ts.hour, ts.minute

        # Skip before 9:30
        if hour < 9 or (hour == 9 and minute < 30):
            continue

        # After 3:10 — close all
        if hour >= 15 and minute > 10:
            for tid in list(portfolio.positions.keys()):
                pos = portfolio.positions[tid]
                sym = pos["symbol"]
                if sym in data and i < len(data[sym]):
                    portfolio.close(tid, data[sym]["close"].iloc[i], pos["remaining"], "EOD_EXIT", ts_str)
            break

        # ── Check exits for all positions ──
        for tid in list(portfolio.positions.keys()):
            pos = portfolio.positions.get(tid)
            if not pos:
                continue
            sym = pos["symbol"]
            if sym not in data or i >= len(data[sym]):
                continue
            row = data[sym].iloc[i]
            pos["candles_held"] += 1

            if pos["strategy"] == "S1":
                curr_rsi4 = row.get("rsi4_5m")

                # Disaster stop: 3x ATR on 15-min
                if row["close"] <= pos["stop_loss"]:
                    portfolio.close(tid, row["close"], pos["remaining"], "DISASTER_STOP", ts_str)
                    continue

                # Hard exit at 3:00 PM
                if hour >= 15:
                    portfolio.close(tid, row["close"], pos["remaining"], "HARD_EXIT_3PM", ts_str)
                    continue

                # Time stop: 75 minutes
                if pos["candles_held"] >= 75:
                    portfolio.close(tid, row["close"], pos["remaining"], "TIME_STOP_75M", ts_str)
                    continue

                # Exit 1: 60% at VWAP touch
                if not pos.get("vwap_exit_done") and row["close"] >= row["vwap"]:
                    exit_qty = int(pos["initial_qty"] * 0.6)
                    exit_qty = min(exit_qty, pos["remaining"])
                    if exit_qty > 0:
                        portfolio.close(tid, row["close"], exit_qty, "VWAP_TOUCH_60pct", ts_str)
                        pos["vwap_exit_done"] = True
                        continue

                # Exit 2: remaining 40% at RSI(4) > 50
                if not pd.isna(curr_rsi4) and curr_rsi4 >= 50:
                    portfolio.close(tid, row["close"], pos["remaining"], "RSI4_GT50", ts_str)

            elif pos["strategy"] == "S3":
                rsi9_5 = row.get("rsi9_5m")

                # Disaster stop: 3x ATR on 15-min
                if row["close"] <= pos["stop_loss"]:
                    portfolio.close(tid, row["close"], pos["remaining"], "S3_DISASTER_STOP", ts_str)
                    continue
                # Hard exit at 2:30 PM
                if hour > 14 or (hour == 14 and minute >= 30):
                    portfolio.close(tid, row["close"], pos["remaining"], "S3_HARD_EXIT", ts_str)
                    continue
                # Time stop (75 min)
                if pos["candles_held"] >= 75:
                    portfolio.close(tid, row["close"], pos["remaining"], "S3_TIME_STOP_75M", ts_str)
                    continue
                # RSI exit at 50
                if not pd.isna(rsi9_5) and rsi9_5 >= 50:
                    portfolio.close(tid, row["close"], pos["remaining"], "S3_RSI_50", ts_str)
                    continue
                # VWAP touch exit
                if row["close"] >= row["vwap"] and pos.get("entered_below_vwap"):
                    portfolio.close(tid, row["close"], pos["remaining"], "S3_VWAP_TOUCH", ts_str)

        # ── S1: RSI(4) on 5-min — entries only 10:15-14:00 ──
        now_min = hour * 60 + minute
        s1_entry_ok = (10 * 60 + 15) <= now_min < (14 * 60)

        if s1_entry_ok:
            for sym, df in data.items():
                if i >= len(df) or i < 1:
                    continue
                row = df.iloc[i]
                prev = df.iloc[i - 1]

                curr_rsi4 = row.get("rsi4_5m")
                prev_rsi4 = prev.get("rsi4_5m")

                if curr_rsi4 is None or prev_rsi4 is None:
                    continue
                if pd.isna(curr_rsi4) or pd.isna(prev_rsi4):
                    continue

                # RSI(4) hook: prev < 15 AND current >= 15
                if not (prev_rsi4 < 15 and curr_rsi4 >= 15):
                    continue

                s1_signals += 1

                if portfolio.has_symbol(sym):
                    continue
                if len(portfolio.strategy_positions("S1")) >= 3:
                    continue
                sector = SECTOR_MAP.get(sym, "Other")
                if portfolio.sector_count.get(sector, 0) >= 1:
                    continue

                # Filter 1: EMA(200) proxy — price above 1-min EMA(200)
                ema200 = row.get("ema200")
                if pd.notna(ema200) and row["close"] < ema200:
                    s1_filtered += 1
                    continue

                # Filter 2: 15-min ADX(14) < 25 (range-bound)
                adx14_15 = row.get("adx14_15m")
                if pd.notna(adx14_15) and adx14_15 >= 25:
                    s1_filtered += 1
                    continue

                # Filter 3: VWAP positioning — 0.3% to 1.2% below VWAP
                vwap_distance = (row["close"] - row["vwap"]) / row["vwap"] if row["vwap"] > 0 else 0
                if not (-0.012 <= vwap_distance <= -0.003):
                    s1_filtered += 1
                    continue

                # Position sizing using 15-min ATR, 3x disaster stop
                atr_15m = row.get("atr14_15m")
                curr_atr = atr_15m if pd.notna(atr_15m) else row["close"] * 0.01
                stop_loss = round(row["close"] - 3.0 * curr_atr, 2)
                risk = row["close"] - stop_loss
                if risk <= 0:
                    continue
                qty = min(int(RISK_PER_TRADE / risk), int(83000 / row["close"]))
                if qty <= 0:
                    continue

                portfolio.open("S1", sym, row["close"], qty, stop_loss,
                              f"RSI(4)={curr_rsi4:.1f} VWAP_dist={vwap_distance*100:.2f}%", ts_str)

        # ── S3: 15-min Mean Reversion entries ──
        # Only in prime window (10:15-12:00) and secondary (1:30-2:30)
        in_prime = (hour == 10 and minute >= 15) or (hour == 11)
        in_secondary = (hour == 13 and minute >= 30) or (hour == 14 and minute < 30)

        if in_prime or in_secondary:
            for sym, df in data.items():
                if i >= len(df):
                    continue
                row = df.iloc[i]

                rsi9_15 = row.get("rsi9_15m")
                adx14_15 = row.get("adx14_15m")
                rsi9_5 = row.get("rsi9_5m")
                atr14_15 = row.get("atr14_15m")

                # Need all indicators
                if any(v is None or (isinstance(v, float) and pd.isna(v)) for v in [rsi9_15, adx14_15, rsi9_5]):
                    continue
                if rsi9_15 is None or adx14_15 is None or rsi9_5 is None:
                    continue

                # Screen 2: 15-min setup — RSI(9) < 40, ADX(14) < 30
                if rsi9_15 >= 40:
                    continue
                if adx14_15 >= 30:
                    continue

                s3_setups += 1

                # Screen 3: 5-min entry trigger — RSI(9) crosses above 25
                prev_rsi9 = df.iloc[i-1].get("rsi9_5m") if i > 0 else None
                if prev_rsi9 is None or pd.isna(prev_rsi9):
                    continue
                if not (prev_rsi9 < 25 and rsi9_5 >= 25):
                    continue

                # Bullish candle
                if row["close"] <= row["open"]:
                    continue

                # VWAP: price must be BELOW VWAP (entry below the mean)
                if row["close"] >= row["vwap"]:
                    continue

                # Position limits
                if portfolio.has_symbol(sym):
                    continue
                if len(portfolio.strategy_positions("S3")) >= 3:
                    continue

                # Position sizing: 3x ATR disaster stop
                sl_atr = atr14_15 if pd.notna(atr14_15) else row["close"] * 0.008
                stop_loss = round(row["close"] - 3.0 * sl_atr, 2)
                risk = row["close"] - stop_loss
                if risk <= 0:
                    continue
                size_mult = 0.5 if in_secondary else 1.0
                qty = min(int(RISK_PER_TRADE * size_mult / risk), int(83000 / row["close"]))
                if qty <= 0:
                    continue

                s3_entries += 1
                window = "PRIME" if in_prime else "SECONDARY"
                tid = portfolio.open("S3", sym, row["close"], qty, stop_loss,
                              f"15mRSI(9)={rsi9_15:.1f} 5mRSI(9)={rsi9_5:.1f} ADX(14)={adx14_15:.1f} [{window}]",
                              ts_str)
                # Track VWAP entry for exit logic
                if tid and tid in portfolio.positions:
                    portfolio.positions[tid]["entered_below_vwap"] = row["close"] < row["vwap"]

    # ── Summary ──
    pnl_pct = (portfolio.daily_pnl / CAPITAL) * 100

    print(f"\n{'='*70}")
    print(f"  SIMULATION RESULTS v2.0 — {target_date}")
    print(f"{'='*70}")

    # S1 summary
    s1_trades = [t for t in portfolio.closed if t["strategy"] == "S1"]
    s1_pnl = sum(t["realized_pnl"] for t in s1_trades)
    s1_wins = len([t for t in s1_trades if t["realized_pnl"] > 0])
    print(f"\n  S1: RSI(4) 5-min Mean Reversion")
    print(f"    Signals: {s1_signals} | Filtered: {s1_filtered} | Traded: {len(s1_trades)}")
    if s1_trades:
        print(f"    Win/Loss: {s1_wins}/{len(s1_trades)-s1_wins} ({s1_wins/len(s1_trades)*100:.0f}%)")
        print(f"    P&L: Rs{s1_pnl:+,.2f}")
        for t in s1_trades:
            m = "W" if t["realized_pnl"] > 0 else "L"
            print(f"      [{m}] {t['symbol']:15s} {t['entry_time']}→{t['exit_time']} "
                  f"Rs{t['entry_price']:.2f}→{t['exit_price']:.2f} | {t['exit_reason']} | Rs{t['realized_pnl']:+,.2f}")
    else:
        print(f"    No trades (market conditions didn't match)")

    # S3 summary
    s3_trades = [t for t in portfolio.closed if t["strategy"] == "S3"]
    s3_pnl = sum(t["realized_pnl"] for t in s3_trades)
    s3_wins = len([t for t in s3_trades if t["realized_pnl"] > 0])
    print(f"\n  S3: 15-min Mean Reversion (RSI(9)<40, ADX(14)<30)")
    print(f"    Setups: {s3_setups} | Entries: {s3_entries} | Traded: {len(s3_trades)}")
    if s3_trades:
        print(f"    Win/Loss: {s3_wins}/{len(s3_trades)-s3_wins} ({s3_wins/len(s3_trades)*100:.0f}%)")
        print(f"    P&L: Rs{s3_pnl:+,.2f}")
        for t in s3_trades:
            m = "W" if t["realized_pnl"] > 0 else "L"
            print(f"      [{m}] {t['symbol']:15s} {t['entry_time']}→{t['exit_time']} "
                  f"Rs{t['entry_price']:.2f}→{t['exit_price']:.2f} | {t['exit_reason']} | Rs{t['realized_pnl']:+,.2f}")
    else:
        print(f"    No trades (strict multi-timeframe filters)")

    # Combined
    print(f"\n  {'─'*66}")
    print(f"  COMBINED P&L:     Rs{portfolio.daily_pnl:+,.2f} ({pnl_pct:+.2f}%)")
    print(f"  Capital:          Rs{CAPITAL:,} → Rs{CAPITAL + portfolio.daily_pnl:,.2f}")
    total_trades = len(portfolio.closed)
    if total_trades:
        total_wins = s1_wins + s3_wins
        print(f"  Total trades:     {total_trades} | Win rate: {total_wins}/{total_trades} ({total_wins/total_trades*100:.0f}%)")
    print(f"{'='*70}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
    print(f"{'='*70}")
    print(f"  AutoTheta Day Simulator v2.0")
    print(f"{'='*70}")

    # Check for cached data
    cache = Path(f"/tmp/march{target[-2:]}_data.pkl")
    if cache.exists():
        print(f"  Loading cached data from {cache}")
        with open(cache, "rb") as f:
            data = pickle.load(f)
    else:
        data = fetch_data(target)
        with open(cache, "wb") as f:
            pickle.dump(data, f)

    if not data:
        print("  No data available!")
        sys.exit(1)

    simulate(data, target)
