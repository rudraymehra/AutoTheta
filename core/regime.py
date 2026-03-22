"""Market Regime Detection Engine.

Classifies market into BULL/RANGE, BEAR/CORRECTION, or CRASH/PANIC at 9:15 AM daily.
Uses: Nifty vs 200-DMA distance, India VIX, daily ADX(14) with DI.

Decision tree (CRASH overrides all):
  CRASH: ANY of (VIX > 22, Nifty > 8% below 200-DMA, gap-down > 3%)
  BEAR:  2 of 3 (Nifty > 3% below 200-DMA, VIX > 16, ADX>25 with -DI>+DI)
  BULL:  default (neither CRASH nor BEAR)
"""

import logging
import time
from datetime import datetime
from enum import Enum

import pandas as pd

log = logging.getLogger("autotheta.regime")


class MarketRegime(str, Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    CRASH = "CRASH"


def classify_regime(api) -> tuple[MarketRegime, dict]:
    """Classify today's market regime using prior day's data.

    Returns (regime, details_dict) where details has all the raw values.
    """
    details = {}
    adx14 = 0
    plus_di_val = 0
    minus_di_val = 0

    # 1. Fetch Nifty daily candles (last 250 days for 200-DMA)
    try:
        from_date = "2025-04-01 09:15"
        to_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        r = api.getCandleData({
            "exchange": "NSE", "symboltoken": "99926000",
            "interval": "ONE_DAY", "fromdate": from_date, "todate": to_date,
        })
        if r and r.get("data") and len(r["data"]) > 50:
            df = pd.DataFrame(r["data"], columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df["high"] = pd.to_numeric(df["high"], errors="coerce")
            df["low"] = pd.to_numeric(df["low"], errors="coerce")

            # 200-DMA
            df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
            nifty_close = df["close"].iloc[-1]
            nifty_200dma = df["ema200"].iloc[-1]
            dma_dist_pct = ((nifty_close - nifty_200dma) / nifty_200dma) * 100

            # ADX(14) with +DI/-DI
            plus_dm = df["high"].diff()
            minus_dm = -df["low"].diff()
            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
            tr = pd.concat([
                df["high"] - df["low"],
                (df["high"] - df["close"].shift(1)).abs(),
                (df["low"] - df["close"].shift(1)).abs(),
            ], axis=1).max(axis=1)
            atr14 = tr.ewm(alpha=1/14, min_periods=14).mean()
            plus_di = 100 * (plus_dm.ewm(alpha=1/14, min_periods=14).mean() / atr14.replace(0, 1e-10))
            minus_di = 100 * (minus_dm.ewm(alpha=1/14, min_periods=14).mean() / atr14.replace(0, 1e-10))
            dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10))
            adx14 = dx.ewm(alpha=1/14, min_periods=14).mean().iloc[-1]
            plus_di_val = plus_di.iloc[-1]
            minus_di_val = minus_di.iloc[-1]

            # Prior day IBS
            prior = df.iloc[-1]  # yesterday's close
            ibs = (prior["close"] - prior["low"]) / max(prior["high"] - prior["low"], 0.01)

            details = {
                "nifty_close": nifty_close,
                "nifty_200dma": nifty_200dma,
                "dma_dist_pct": round(dma_dist_pct, 2),
                "adx14": round(float(adx14), 1),
                "plus_di": round(float(plus_di_val), 1),
                "minus_di": round(float(minus_di_val), 1),
                "prior_day_ibs": round(float(ibs), 3),
            }
        else:
            log.warning("Could not fetch Nifty daily data")
            return MarketRegime.BULL, {"error": "no_data"}
    except Exception as e:
        log.exception("Regime detection failed")
        return MarketRegime.BULL, {"error": str(e)}

    # 2. Fetch India VIX
    time.sleep(0.5)
    try:
        vix_data = api.ltpData("NSE", "India VIX", "26017")
        vix = float(vix_data["data"]["ltp"])
        # Angel One sometimes returns VIX * 100
        if vix > 100:
            vix = vix / 100
        details["india_vix"] = round(vix, 2)
    except Exception:
        vix = 15.0  # default to safe
        details["india_vix"] = vix
        details["vix_error"] = True

    # 3. Check for gap-down at open
    try:
        nifty_ltp = api.ltpData("NSE", "NIFTY", "99926000")
        current_nifty = float(nifty_ltp["data"]["ltp"])
        nifty_close = details.get("nifty_close", current_nifty)
        gap_pct = ((current_nifty - nifty_close) / nifty_close) * 100
        details["gap_pct"] = round(gap_pct, 2)
        details["current_nifty"] = current_nifty
    except Exception:
        gap_pct = 0
        details["gap_pct"] = 0

    # 4. Classification
    dma_dist = details.get("dma_dist_pct", 0)

    # CRASH: ANY one condition
    crash_conditions = [
        vix > 22,
        dma_dist < -8.0,
        gap_pct < -3.0,
    ]
    if any(crash_conditions):
        regime = MarketRegime.CRASH
        details["crash_triggers"] = [
            f"VIX={vix:.1f}>22" if vix > 22 else None,
            f"DMA={dma_dist:.1f}%<-8%" if dma_dist < -8.0 else None,
            f"Gap={gap_pct:.1f}%<-3%" if gap_pct < -3.0 else None,
        ]
        details["crash_triggers"] = [x for x in details["crash_triggers"] if x]
        log.warning("REGIME: CRASH -- %s", details["crash_triggers"])
        return regime, details

    # BEAR: 2 of 3 conditions
    bear_conditions = [
        dma_dist < -3.0,
        vix > 16,
        adx14 > 25 and minus_di_val > plus_di_val,
    ]
    bear_count = sum(bear_conditions)
    details["bear_conditions_met"] = bear_count
    if bear_count >= 2:
        regime = MarketRegime.BEAR
        log.info("REGIME: BEAR -- DMA=%.1f%%, VIX=%.1f, ADX=%.1f (-DI>+DI: %s), %d/3 conditions",
                 dma_dist, vix, adx14, minus_di_val > plus_di_val, bear_count)
        return regime, details

    # BULL: default
    regime = MarketRegime.BULL
    details["bear_conditions_met"] = bear_count
    log.info("REGIME: BULL -- DMA=%.1f%%, VIX=%.1f, ADX=%.1f", dma_dist, vix, adx14)
    return regime, details


def classify_regime_from_data(nifty_daily_df, vix_value=None, current_nifty=None):
    """Classify regime from pre-fetched data (for simulators).

    nifty_daily_df: DataFrame with columns [timestamp, open, high, low, close, volume]
                    covering at least 200 days of Nifty daily candles up to the target date.
    vix_value: India VIX on that day (float), or None to skip VIX check.
    current_nifty: Current Nifty price (for gap calc), or None.

    Returns (regime, details_dict).
    """
    details = {}
    adx14 = 0
    plus_di_val = 0
    minus_di_val = 0

    if nifty_daily_df is None or len(nifty_daily_df) < 50:
        return MarketRegime.BULL, {"error": "insufficient_data"}

    df = nifty_daily_df.copy()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["high"] = pd.to_numeric(df["high"], errors="coerce")
    df["low"] = pd.to_numeric(df["low"], errors="coerce")

    # 200-DMA
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
    nifty_close = df["close"].iloc[-1]
    nifty_200dma = df["ema200"].iloc[-1]
    dma_dist_pct = ((nifty_close - nifty_200dma) / nifty_200dma) * 100

    # ADX(14) with +DI/-DI
    plus_dm = df["high"].diff()
    minus_dm = -df["low"].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.ewm(alpha=1/14, min_periods=14).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/14, min_periods=14).mean() / atr14.replace(0, 1e-10))
    minus_di = 100 * (minus_dm.ewm(alpha=1/14, min_periods=14).mean() / atr14.replace(0, 1e-10))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10))
    adx14 = dx.ewm(alpha=1/14, min_periods=14).mean().iloc[-1]
    plus_di_val = plus_di.iloc[-1]
    minus_di_val = minus_di.iloc[-1]

    # Prior day IBS
    prior = df.iloc[-1]
    ibs = (prior["close"] - prior["low"]) / max(prior["high"] - prior["low"], 0.01)

    details = {
        "nifty_close": nifty_close,
        "nifty_200dma": nifty_200dma,
        "dma_dist_pct": round(dma_dist_pct, 2),
        "adx14": round(float(adx14), 1),
        "plus_di": round(float(plus_di_val), 1),
        "minus_di": round(float(minus_di_val), 1),
        "prior_day_ibs": round(float(ibs), 3),
    }

    # VIX
    vix = vix_value if vix_value is not None else 15.0
    details["india_vix"] = round(vix, 2)
    if vix_value is None:
        details["vix_estimated"] = True

    # Gap
    gap_pct = 0
    if current_nifty is not None:
        gap_pct = ((current_nifty - nifty_close) / nifty_close) * 100
        details["gap_pct"] = round(gap_pct, 2)
        details["current_nifty"] = current_nifty
    else:
        details["gap_pct"] = 0

    dma_dist = details["dma_dist_pct"]

    # CRASH: ANY one condition
    crash_conditions = [
        vix > 22,
        dma_dist < -8.0,
        gap_pct < -3.0,
    ]
    if any(crash_conditions):
        regime = MarketRegime.CRASH
        details["crash_triggers"] = [
            f"VIX={vix:.1f}>22" if vix > 22 else None,
            f"DMA={dma_dist:.1f}%<-8%" if dma_dist < -8.0 else None,
            f"Gap={gap_pct:.1f}%<-3%" if gap_pct < -3.0 else None,
        ]
        details["crash_triggers"] = [x for x in details["crash_triggers"] if x]
        return regime, details

    # BEAR: 2 of 3 conditions
    bear_conditions = [
        dma_dist < -3.0,
        vix > 16,
        adx14 > 25 and minus_di_val > plus_di_val,
    ]
    bear_count = sum(bear_conditions)
    details["bear_conditions_met"] = bear_count
    if bear_count >= 2:
        return MarketRegime.BEAR, details

    # BULL: default
    details["bear_conditions_met"] = bear_count
    return MarketRegime.BULL, details
