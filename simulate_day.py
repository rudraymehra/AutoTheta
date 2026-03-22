"""Simulate a full trading day — replay 1-min candles as if it's live.

Tests all strategies: S1 (RSI(5) Mean Reversion on 5-min), S3 (15-min Mean Reversion)
Also checks S2 (Expiry Skew) conditions if it's a Tuesday.

v3.0 — Regime-adaptive system:
  Market regime classified at startup: BULL / BEAR / CRASH
  BULL: S1 RSI(5) buy dips + S3 15-min mean reversion
  BEAR: Sell overbought rallies (short to VWAP)
  CRASH: No new entries

  S1: RSI(5) on 5-min, uptick from <20, daily regime 2/3, KER<0.30, below VWAP, MFI(8)<30
  BEAR: RSI(5) > 80 downtick, above VWAP, KER<0.30, short to VWAP
  S3: RSI(9) setup<40 on 15-min, entry<25 on 5-min, KER(10)<0.30, 3x ATR stop

  Circuit breakers: 3 consecutive losses = stop, Rs 7,500 daily hard cap
  IBS filter: prior day IBS > 0.25 reduces position size 50%

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

from core.regime import classify_regime_from_data, MarketRegime

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


# v2.1 stock universe — dropped metals/energy, focus banking/IT/FMCG
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

    def open(self, strategy, symbol, price, qty, stop_loss, indicators, time_str, side="LONG"):
        self.trade_count += 1
        tid = f"{strategy}-{self.trade_count:04d}"
        sector = SECTOR_MAP.get(symbol, "Other")
        self.positions[tid] = {
            "strategy": strategy, "symbol": symbol, "entry_price": price,
            "quantity": qty, "initial_qty": qty, "remaining": qty,
            "stop_loss": stop_loss, "status": "open",
            "entry_time": time_str, "candles_held": 0,
            "realized_pnl": 0.0, "indicators": indicators, "sector": sector,
            "vwap_exit_done": False, "side": side,
        }
        self.sector_count[sector] += 1
        action = "SHORT" if side == "SHORT" else "BUY"
        print(f"  >> [{time_str}] {strategy} {action} {symbol} x{qty} @ Rs{price:.2f} | SL={stop_loss:.2f} | {indicators}")
        return tid

    def close(self, tid, price, qty, reason, time_str):
        pos = self.positions.get(tid)
        if not pos:
            return 0
        side = pos.get("side", "LONG")
        if side == "LONG":
            pnl = (price - pos["entry_price"]) * qty
        else:  # SHORT
            pnl = (pos["entry_price"] - price) * qty
        pos["realized_pnl"] += pnl
        pos["remaining"] -= qty
        self.daily_pnl += pnl
        marker = "WIN" if pnl > 0 else "LOSS"
        action = "COVER" if side == "SHORT" else "SELL"
        print(f"  << [{time_str}] {pos['strategy']} {action} {pos['symbol']} x{qty} @ Rs{price:.2f} | {reason} | Rs{pnl:+,.2f} [{marker}]")
        if pos["remaining"] <= 0:
            self.sector_count[pos["sector"]] = max(0, self.sector_count[pos["sector"]] - 1)
            self.closed.append({**pos, "exit_price": price, "exit_time": time_str, "exit_reason": reason})
            del self.positions[tid]
        return pnl

    def has_symbol(self, symbol):
        return any(p["symbol"] == symbol for p in self.positions.values())

    def strategy_positions(self, strategy):
        return {k: v for k, v in self.positions.items() if v["strategy"] == strategy}


def fetch_daily_regime(api, token_map, target_date):
    """Fetch daily candles and compute 2-of-3 regime check for each stock."""
    daily_regime = {}
    print(f"  Running daily regime check...")
    for sym, tok in token_map.items():
        time.sleep(0.5)
        try:
            r = api.getCandleData({
                "exchange": "NSE", "symboltoken": tok, "interval": "ONE_DAY",
                "fromdate": "2025-06-01 09:15",
                "todate": f"{target_date} 15:30",
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
                if pd.notna(last["ema200"]):
                    if abs(last["close"] - last["ema200"]) / last["ema200"] < 0.08:
                        checks_passed += 1
                if pd.notna(last["rsi14"]):
                    if 30 <= last["rsi14"] <= 65:
                        checks_passed += 1
                if pd.notna(last["adx14"]):
                    if last["adx14"] < 25:
                        checks_passed += 1

                daily_regime[sym] = checks_passed >= 2
                status = "PASS" if daily_regime[sym] else "FAIL"
                print(f"    {sym:18s} regime={status} ({checks_passed}/3)")
            else:
                daily_regime[sym] = True
        except Exception:
            daily_regime[sym] = True
    return daily_regime


def fetch_nifty_daily(api, target_date):
    """Fetch Nifty daily candles for regime detection."""
    try:
        time.sleep(0.5)
        r = api.getCandleData({
            "exchange": "NSE", "symboltoken": "99926000",
            "interval": "ONE_DAY",
            "fromdate": "2025-04-01 09:15",
            "todate": f"{target_date} 15:30",
        })
        if r and r.get("data") and len(r["data"]) > 50:
            df = pd.DataFrame(r["data"], columns=["timestamp", "open", "high", "low", "close", "volume"])
            return df
    except Exception as e:
        print(f"    Could not fetch Nifty daily data: {e}")
    return None


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

    # Fetch daily regime data
    daily_regime = fetch_daily_regime(api, token_map, target_date)

    # Fetch Nifty daily data for market regime classification
    nifty_daily = fetch_nifty_daily(api, target_date)
    market_regime = MarketRegime.BULL
    regime_details = {}
    if nifty_daily is not None:
        market_regime, regime_details = classify_regime_from_data(nifty_daily)
        print(f"  Market regime: {market_regime.value} | "
              f"DMA dist: {regime_details.get('dma_dist_pct', 'N/A')}% | "
              f"VIX: {regime_details.get('india_vix', 'N/A')} | "
              f"ADX: {regime_details.get('adx14', 'N/A')}")

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
    return data, daily_regime, market_regime, regime_details


def simulate(data, target_date, daily_regime=None, market_regime=None, regime_details=None):
    """Replay candles minute by minute, running all strategies."""
    portfolio = SimPortfolio()
    if daily_regime is None:
        daily_regime = {sym: True for sym in data}
    if market_regime is None:
        market_regime = MarketRegime.BULL
    if regime_details is None:
        regime_details = {}

    prior_ibs = regime_details.get("prior_day_ibs", 0.5)
    ibs_size_mult = 0.5 if prior_ibs > 0.25 else 1.0

    print(f"\n{'='*70}")
    print(f"  FULL DAY SIMULATION v3.0 — {target_date}")
    print(f"  Market Regime: {market_regime.value}")
    if market_regime == MarketRegime.BULL:
        print(f"  Strategies: S1 (RSI(5) buy dips) + S3 (15-min Mean Reversion)")
    elif market_regime == MarketRegime.BEAR:
        print(f"  Strategies: BEAR (sell overbought rallies) + S3 (15-min Mean Reversion)")
    else:
        print(f"  Strategies: CRASH — no new entries, manage existing only")
    print(f"  DMA dist: {regime_details.get('dma_dist_pct', 'N/A')}% | "
          f"VIX: {regime_details.get('india_vix', 'N/A')} | "
          f"ADX: {regime_details.get('adx14', 'N/A')} | "
          f"IBS: {prior_ibs:.3f} (size mult: {ibs_size_mult})")
    is_tuesday = datetime.strptime(str(target_date), "%Y-%m-%d").weekday() == 1
    if is_tuesday:
        print(f"  Tuesday = Expiry Day — S2 (Expiry Skew) also checked")
    print(f"  Capital: Rs{CAPITAL:,}")
    regime_pass = sum(1 for v in daily_regime.values() if v)
    print(f"  Daily regime: {regime_pass}/{len(daily_regime)} stocks passed")
    print(f"{'='*70}\n")

    # Circuit breaker state
    daily_losses_consecutive = 0
    circuit_breaker_active = False

    # Precompute all indicators for each stock
    for sym, df in data.items():
        df["atr14"] = atr_calc(df["high"], df["low"], df["close"], 14)
        df["vwap"] = vwap_calc(df)

        # 5-min resampled for S1: RSI(5) on 5-min
        df_5m = df.set_index("timestamp").resample("5min").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna().reset_index()
        if len(df_5m) >= 6:
            df_5m["rsi5"] = rsi(df_5m["close"], 5)
            df["rsi5_5m"] = None
            for _, bar in df_5m.iterrows():
                mask = (df["timestamp"] >= bar["timestamp"]) & (df["timestamp"] < bar["timestamp"] + pd.Timedelta(minutes=5))
                df.loc[mask, "rsi5_5m"] = bar.get("rsi5")
            df["rsi5_5m"] = df["rsi5_5m"].ffill()

            # MFI(8) on 5-min
            if len(df_5m) >= 10:
                df_5m["mfi8"] = mfi(df_5m["high"], df_5m["low"], df_5m["close"], df_5m["volume"], 8)
                df["mfi8_5m"] = None
                for _, bar in df_5m.iterrows():
                    mask = (df["timestamp"] >= bar["timestamp"]) & (df["timestamp"] < bar["timestamp"] + pd.Timedelta(minutes=5))
                    df.loc[mask, "mfi8_5m"] = bar.get("mfi8")
                df["mfi8_5m"] = df["mfi8_5m"].ffill()

        # 15-min resampled for filters
        df_15m = df.set_index("timestamp").resample("15min").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna().reset_index()
        if len(df_15m) >= 5:
            df_15m["ker10"] = kaufman_er(df_15m["close"], 10)
            df_15m["atr14"] = atr_calc(df_15m["high"], df_15m["low"], df_15m["close"], 14)
            df_15m["rsi9"] = rsi(df_15m["close"], 9)

            # Map 15-min indicators back
            for col in ["ker10", "atr14", "rsi9"]:
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
            side = pos.get("side", "LONG")

            if pos["strategy"] == "S1":
                curr_rsi5 = row.get("rsi5_5m")

                # Disaster stop: 3x ATR on 15-min
                if row["close"] <= pos["stop_loss"]:
                    pnl = portfolio.close(tid, row["close"], pos["remaining"], "DISASTER_STOP", ts_str)
                    daily_losses_consecutive += 1
                    continue

                # Hard exit at 3:00 PM
                if hour >= 15:
                    pnl = portfolio.close(tid, row["close"], pos["remaining"], "HARD_EXIT_3PM", ts_str)
                    if pnl < 0:
                        daily_losses_consecutive += 1
                    else:
                        daily_losses_consecutive = 0
                    continue

                # Time stop: 75 minutes
                if pos["candles_held"] >= 75:
                    pnl = portfolio.close(tid, row["close"], pos["remaining"], "TIME_STOP_75M", ts_str)
                    if pnl < 0:
                        daily_losses_consecutive += 1
                    else:
                        daily_losses_consecutive = 0
                    continue

                # Exit 1: 60% at VWAP touch
                if not pos.get("vwap_exit_done") and row["close"] >= row["vwap"]:
                    exit_qty = int(pos["initial_qty"] * 0.6)
                    exit_qty = min(exit_qty, pos["remaining"])
                    if exit_qty > 0:
                        portfolio.close(tid, row["close"], exit_qty, "VWAP_TOUCH_60pct", ts_str)
                        pos["vwap_exit_done"] = True
                        continue

                # Exit 2: remaining 40% at RSI(5) > 50
                if not pd.isna(curr_rsi5) and curr_rsi5 >= 50:
                    pnl = portfolio.close(tid, row["close"], pos["remaining"], "RSI5_GT50", ts_str)
                    if pnl < 0:
                        daily_losses_consecutive += 1
                    else:
                        daily_losses_consecutive = 0

            elif pos["strategy"] == "BEAR":
                curr_rsi5 = row.get("rsi5_5m")

                # Disaster stop: price ABOVE stop (short)
                if row["close"] >= pos["stop_loss"]:
                    pnl = portfolio.close(tid, row["close"], pos["remaining"], "BEAR_DISASTER_STOP", ts_str)
                    daily_losses_consecutive += 1
                    continue

                # Hard exit at 3:00 PM
                if hour >= 15:
                    pnl = portfolio.close(tid, row["close"], pos["remaining"], "BEAR_HARD_EXIT_3PM", ts_str)
                    if pnl < 0:
                        daily_losses_consecutive += 1
                    else:
                        daily_losses_consecutive = 0
                    continue

                # Time stop: 75 minutes
                if pos["candles_held"] >= 75:
                    pnl = portfolio.close(tid, row["close"], pos["remaining"], "BEAR_TIME_STOP_75M", ts_str)
                    if pnl < 0:
                        daily_losses_consecutive += 1
                    else:
                        daily_losses_consecutive = 0
                    continue

                # Exit 1: 60% when price drops to VWAP (short — price falls to VWAP)
                if not pos.get("vwap_exit_done") and row["close"] <= row["vwap"]:
                    exit_qty = int(pos["initial_qty"] * 0.6)
                    exit_qty = min(exit_qty, pos["remaining"])
                    if exit_qty > 0:
                        portfolio.close(tid, row["close"], exit_qty, "BEAR_VWAP_DROP_60pct", ts_str)
                        pos["vwap_exit_done"] = True
                        continue

                # Exit 2: remaining 40% when RSI(5) < 50
                if curr_rsi5 is not None and not pd.isna(curr_rsi5) and curr_rsi5 < 50:
                    pnl = portfolio.close(tid, row["close"], pos["remaining"], "BEAR_RSI5_LT50", ts_str)
                    if pnl < 0:
                        daily_losses_consecutive += 1
                    else:
                        daily_losses_consecutive = 0

            elif pos["strategy"] == "S3":
                rsi9_5 = row.get("rsi9_5m")

                # Disaster stop: 3x ATR on 15-min
                if row["close"] <= pos["stop_loss"]:
                    pnl = portfolio.close(tid, row["close"], pos["remaining"], "S3_DISASTER_STOP", ts_str)
                    daily_losses_consecutive += 1
                    continue
                # Hard exit at 2:30 PM
                if hour > 14 or (hour == 14 and minute >= 30):
                    pnl = portfolio.close(tid, row["close"], pos["remaining"], "S3_HARD_EXIT", ts_str)
                    if pnl < 0:
                        daily_losses_consecutive += 1
                    else:
                        daily_losses_consecutive = 0
                    continue
                # Time stop (75 min)
                if pos["candles_held"] >= 75:
                    pnl = portfolio.close(tid, row["close"], pos["remaining"], "S3_TIME_STOP_75M", ts_str)
                    if pnl < 0:
                        daily_losses_consecutive += 1
                    else:
                        daily_losses_consecutive = 0
                    continue
                # RSI exit at 50
                if not pd.isna(rsi9_5) and rsi9_5 >= 50:
                    pnl = portfolio.close(tid, row["close"], pos["remaining"], "S3_RSI_50", ts_str)
                    if pnl < 0:
                        daily_losses_consecutive += 1
                    else:
                        daily_losses_consecutive = 0
                    continue
                # VWAP touch exit
                if row["close"] >= row["vwap"] and pos.get("entered_below_vwap"):
                    portfolio.close(tid, row["close"], pos["remaining"], "S3_VWAP_TOUCH", ts_str)

        # ── Circuit breaker check ──
        if daily_losses_consecutive >= 3 or portfolio.daily_pnl <= -7500:
            circuit_breaker_active = True

        # ── Entries — regime-aware, only 10:15-14:00, circuit breaker off ──
        now_min = hour * 60 + minute
        s1_entry_ok = (10 * 60 + 15) <= now_min < (14 * 60)

        # CRASH regime: no new entries
        if market_regime == MarketRegime.CRASH:
            s1_entry_ok = False

        if s1_entry_ok and not circuit_breaker_active:

            # ── BULL: S1 RSI(5) buy dips ──
            if market_regime == MarketRegime.BULL:
                for sym, df in data.items():
                    if i >= len(df) or i < 1:
                        continue
                    row = df.iloc[i]
                    prev = df.iloc[i - 1]

                    curr_rsi5 = row.get("rsi5_5m")
                    prev_rsi5 = prev.get("rsi5_5m")

                    if curr_rsi5 is None or prev_rsi5 is None:
                        continue
                    if pd.isna(curr_rsi5) or pd.isna(prev_rsi5):
                        continue

                    # RSI(5) uptick: prev < 20 AND curr > prev (rising from oversold)
                    if not (prev_rsi5 < 20 and curr_rsi5 > prev_rsi5):
                        continue

                    s1_signals += 1

                    if portfolio.has_symbol(sym):
                        continue
                    if len(portfolio.strategy_positions("S1")) >= 3:
                        continue
                    sector = SECTOR_MAP.get(sym, "Other")
                    if portfolio.sector_count.get(sector, 0) >= 1:
                        continue

                    # Filter 1: Daily regime check (2 of 3 conditions)
                    if not daily_regime.get(sym, True):
                        s1_filtered += 1
                        continue

                    # Filter 2: KER(10) < 0.30 on 15-min (choppy/mean-reverting)
                    ker_15 = row.get("ker10_15m")
                    if pd.notna(ker_15) and ker_15 >= 0.30:
                        s1_filtered += 1
                        continue

                    # Filter 3: Price must be below VWAP
                    if row["vwap"] > 0 and row["close"] >= row["vwap"]:
                        s1_filtered += 1
                        continue

                    # Filter 4: MFI(8) < 30 confirmation (optional)
                    mfi_val = row.get("mfi8_5m")
                    if pd.notna(mfi_val) and mfi_val >= 30:
                        s1_filtered += 1
                        continue

                    # Position sizing using 15-min ATR, 3x disaster stop
                    atr_15m = row.get("atr14_15m")
                    curr_atr = atr_15m if pd.notna(atr_15m) else row["close"] * 0.01
                    stop_loss = round(row["close"] - 3.0 * curr_atr, 2)
                    risk = row["close"] - stop_loss
                    if risk <= 0:
                        continue
                    # IBS filter
                    effective_risk = RISK_PER_TRADE * ibs_size_mult
                    qty = min(int(effective_risk / risk), int(83000 / row["close"]))
                    if qty <= 0:
                        continue

                    vwap_distance = (row["close"] - row["vwap"]) / row["vwap"] if row["vwap"] > 0 else 0
                    portfolio.open("S1", sym, row["close"], qty, stop_loss,
                                  f"RSI(5)={curr_rsi5:.1f} VWAP_dist={vwap_distance*100:.2f}%", ts_str)

            # ── BEAR: sell overbought rallies (short to VWAP) ──
            elif market_regime == MarketRegime.BEAR:
                for sym, df in data.items():
                    if i >= len(df) or i < 1:
                        continue
                    row = df.iloc[i]
                    prev = df.iloc[i - 1]

                    curr_rsi5 = row.get("rsi5_5m")
                    prev_rsi5 = prev.get("rsi5_5m")

                    if curr_rsi5 is None or prev_rsi5 is None:
                        continue
                    if pd.isna(curr_rsi5) or pd.isna(prev_rsi5):
                        continue

                    # BEAR: RSI(5) was ABOVE 80 on prev candle AND downtick
                    if not (prev_rsi5 > 80 and curr_rsi5 < prev_rsi5):
                        continue

                    s1_signals += 1

                    if portfolio.has_symbol(sym):
                        continue
                    if len(portfolio.strategy_positions("BEAR")) >= 3:
                        continue
                    sector = SECTOR_MAP.get(sym, "Other")
                    if portfolio.sector_count.get(sector, 0) >= 1:
                        continue

                    # Filter: Price must be ABOVE VWAP (selling the rally)
                    if row["vwap"] > 0 and row["close"] <= row["vwap"]:
                        s1_filtered += 1
                        continue

                    # Filter: KER(10) < 0.30 on 15-min
                    ker_15 = row.get("ker10_15m")
                    if pd.notna(ker_15) and ker_15 >= 0.30:
                        s1_filtered += 1
                        continue

                    # Position sizing: 3x ATR ABOVE entry (short stop)
                    atr_15m = row.get("atr14_15m")
                    curr_atr = atr_15m if pd.notna(atr_15m) else row["close"] * 0.01
                    stop_loss = round(row["close"] + 3.0 * curr_atr, 2)  # ABOVE for short
                    risk = stop_loss - row["close"]
                    if risk <= 0:
                        continue
                    effective_risk = RISK_PER_TRADE * ibs_size_mult
                    qty = min(int(effective_risk / risk), int(83000 / row["close"]))
                    if qty <= 0:
                        continue

                    vwap_distance = (row["close"] - row["vwap"]) / row["vwap"] if row["vwap"] > 0 else 0
                    portfolio.open("BEAR", sym, row["close"], qty, stop_loss,
                                  f"RSI(5)={curr_rsi5:.1f} BEAR_SHORT VWAP_dist={vwap_distance*100:+.2f}%",
                                  ts_str, side="SHORT")

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
                ker_15 = row.get("ker10_15m")
                rsi9_5 = row.get("rsi9_5m")
                atr14_15 = row.get("atr14_15m")

                # Need all indicators
                if any(v is None or (isinstance(v, float) and pd.isna(v)) for v in [rsi9_15, rsi9_5]):
                    continue
                if rsi9_15 is None or rsi9_5 is None:
                    continue

                # Screen 2: 15-min setup — RSI(9) < 40, KER(10) < 0.30
                if rsi9_15 >= 40:
                    continue
                if pd.notna(ker_15) and ker_15 >= 0.30:
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
                ker_str = f"{ker_15:.3f}" if pd.notna(ker_15) else "N/A"
                tid = portfolio.open("S3", sym, row["close"], qty, stop_loss,
                              f"15mRSI(9)={rsi9_15:.1f} 5mRSI(9)={rsi9_5:.1f} KER(10)={ker_str} [{window}]",
                              ts_str)
                # Track VWAP entry for exit logic
                if tid and tid in portfolio.positions:
                    portfolio.positions[tid]["entered_below_vwap"] = row["close"] < row["vwap"]

    # ── Summary ──
    pnl_pct = (portfolio.daily_pnl / CAPITAL) * 100

    print(f"\n{'='*70}")
    print(f"  SIMULATION RESULTS v3.0 — {target_date}")
    print(f"  Market Regime: {market_regime.value}")
    print(f"{'='*70}")

    # S1 / BEAR summary (depending on regime)
    s1_trades = [t for t in portfolio.closed if t["strategy"] == "S1"]
    bear_trades = [t for t in portfolio.closed if t["strategy"] == "BEAR"]
    primary_trades = s1_trades if market_regime == MarketRegime.BULL else bear_trades
    primary_pnl = sum(t["realized_pnl"] for t in primary_trades)
    primary_wins = len([t for t in primary_trades if t["realized_pnl"] > 0])

    if market_regime == MarketRegime.BULL:
        print(f"\n  S1: RSI(5) 5-min Mean Reversion (BULL)")
    elif market_regime == MarketRegime.BEAR:
        print(f"\n  BEAR: Sell Overbought Rallies (SHORT)")
    else:
        print(f"\n  CRASH: No entries (manage existing only)")
    print(f"    Signals: {s1_signals} | Filtered: {s1_filtered} | Traded: {len(primary_trades)}")
    if primary_trades:
        print(f"    Win/Loss: {primary_wins}/{len(primary_trades)-primary_wins} ({primary_wins/len(primary_trades)*100:.0f}%)")
        print(f"    P&L: Rs{primary_pnl:+,.2f}")
        for t in primary_trades:
            m = "W" if t["realized_pnl"] > 0 else "L"
            side_tag = f" [{t.get('side', 'LONG')}]" if t.get("side") == "SHORT" else ""
            print(f"      [{m}] {t['symbol']:15s} {t['entry_time']}→{t['exit_time']} "
                  f"Rs{t['entry_price']:.2f}→{t['exit_price']:.2f} | {t['exit_reason']} | Rs{t['realized_pnl']:+,.2f}{side_tag}")
    else:
        print(f"    No trades (market conditions didn't match)")

    if circuit_breaker_active:
        print(f"\n  [CIRCUIT BREAKER] Triggered during session")

    # S3 summary
    s3_trades = [t for t in portfolio.closed if t["strategy"] == "S3"]
    s3_pnl = sum(t["realized_pnl"] for t in s3_trades)
    s3_wins = len([t for t in s3_trades if t["realized_pnl"] > 0])
    print(f"\n  S3: 15-min Mean Reversion (RSI(9)<40, KER(10)<0.30)")
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
        total_wins = primary_wins + s3_wins
        print(f"  Total trades:     {total_trades} | Win rate: {total_wins}/{total_trades} ({total_wins/total_trades*100:.0f}%)")
    print(f"{'='*70}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
    print(f"{'='*70}")
    print(f"  AutoTheta Day Simulator v3.0 — Regime-Adaptive")
    print(f"{'='*70}")

    # Check for cached data
    cache = Path(f"/tmp/march{target[-2:]}_data_v30.pkl")
    if cache.exists():
        print(f"  Loading cached data from {cache}")
        with open(cache, "rb") as f:
            cached = pickle.load(f)
        if isinstance(cached, tuple) and len(cached) == 4:
            data, daily_regime, market_regime, regime_details = cached
        elif isinstance(cached, tuple) and len(cached) == 2:
            data, daily_regime = cached
            market_regime = MarketRegime.BULL
            regime_details = {}
        else:
            data = cached
            daily_regime = {sym: True for sym in data}
            market_regime = MarketRegime.BULL
            regime_details = {}
    else:
        data, daily_regime, market_regime, regime_details = fetch_data(target)
        with open(cache, "wb") as f:
            pickle.dump((data, daily_regime, market_regime, regime_details), f)

    if not data:
        print("  No data available!")
        sys.exit(1)

    simulate(data, target, daily_regime, market_regime, regime_details)
