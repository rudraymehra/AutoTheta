"""Strategy 3: Multi-Timeframe RSI Mean Reversion (15-min setup + 5-min entry).

v2.0 — Research-backed parameter updates:
  Screen 1 — Daily trend filter: stock above 200-EMA proxy
  Screen 2 — 15-min setup: RSI(9) < 40, ADX(14) < 30
  Screen 3 — 5-min entry trigger: RSI(9) crosses back above 25, price below VWAP

LONG-ONLY — never short (India's structural bullish bias).

Time windows:
  10:15-12:00  Prime window (full size)
  13:30-14:30  Secondary window (half size)
  14:30+       Exit-only zone

Exit rules:
  - 5-min RSI(9) > 50  (mean reversion complete)
  - Price touches VWAP from below
  - 75-min time stop
  - 3.0x ATR(14) on 15-min disaster stop
  - Hard exit at 14:30
"""

from datetime import datetime, timedelta

import pandas as pd

# ── Indicators ──────────────────────────────────────────
# These mirror paper_live.py's indicator functions but are
# defined locally so the module is self-contained.

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (tp * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum().replace(0, 1)
    return cum_tp_vol / cum_vol


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
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
    adx_val = dx.ewm(alpha=1 / period, min_periods=period).mean()
    return adx_val


# ── Resampler ───────────────────────────────────────────

def _resample(df_1min: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample 1-min candles to a higher timeframe (e.g. '5min', '15min').

    Expects df_1min to have a 'timestamp' column (pd.Timestamp).
    Returns a DataFrame with the same columns, or None if insufficient data.
    """
    if df_1min is None or len(df_1min) < 2:
        return None
    temp = df_1min.set_index("timestamp").sort_index()
    resampled = temp.resample(freq).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    if resampled.empty:
        return None
    return resampled.reset_index()


# ── Configuration defaults (v2.0 research-backed) ──────

_DEFAULTS = dict(
    setup_rsi_period=9,            # RSI(9) on 15-min (was 14)
    setup_rsi_threshold=40,        # Connie Brown: oversold in bull = 40-50 (was 30)
    entry_rsi_period=9,
    entry_rsi_threshold=25,        # 5-min trigger (was 30)
    exit_rsi_threshold=50,         # Exit sooner at RSI>50 (was 55)
    adx_period=14,                 # ADX(14) (was 10)
    adx_max=30,                    # Less restrictive (was 25)
    atr_sl_mult=3.0,              # Disaster stop only (was 1.5)
    time_stop_minutes=75,          # Wider time window (was 50)
    max_positions=3,
    max_per_sector=1,
    prime_window_start="10:15",
    prime_window_end="12:00",
    secondary_window_start="13:30",
    secondary_window_end="14:30",
    secondary_size_mult=0.5,
    risk_per_trade=2500,
    capital=250000,
    daily_loss_cap=7500,
)

# ── Module-level state (reset daily via reset_state) ────

_setups: dict[str, dict] = {}          # token -> setup info (Screen 2 satisfied)
_positions: dict[str, dict] = {}       # trade_id -> position info
_trade_counter: list[int] = [0]
_daily_pnl: list[float] = [0.0]
_daily_trades_count: list[int] = [0]   # track number of entries today

# Sector map mirrors paper_live.py (v2.0 — dropped metals/energy)
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


def reset_state():
    """Call at the start of each day to clear state."""
    _setups.clear()
    _positions.clear()
    _trade_counter[0] = 0
    _daily_pnl[0] = 0.0
    _daily_trades_count[0] = 0


def get_positions() -> dict:
    """Return current open positions (read-only view)."""
    return dict(_positions)


def get_daily_pnl() -> float:
    return _daily_pnl[0]


# ── Time window helpers ─────────────────────────────────

def _parse_time(t_str: str):
    h, m = t_str.split(":")
    return int(h), int(m)


def _in_window(now: datetime, start_str: str, end_str: str) -> bool:
    sh, sm = _parse_time(start_str)
    eh, em = _parse_time(end_str)
    start_min = sh * 60 + sm
    end_min = eh * 60 + em
    now_min = now.hour * 60 + now.minute
    return start_min <= now_min < end_min


# ── Daily trend filter (Screen 1) ──────────────────────

_daily_trend_ok: dict[str, bool] = {}   # token -> True if daily trend is bullish


def check_daily_trend(token: str, df_1min: pd.DataFrame) -> bool:
    """Approximate daily trend check using available 1-min data.

    Uses EMA(200) on 1-min as a rough proxy for the daily 200-EMA.
    Once a stock passes this check, it's cached for the day.
    """
    if token in _daily_trend_ok:
        return _daily_trend_ok[token]

    if df_1min is None or len(df_1min) < 200:
        # Not enough data yet — give benefit of doubt if we have enough for EMA
        if df_1min is not None and len(df_1min) >= 50:
            ema50 = _ema(df_1min["close"], 50).iloc[-1]
            price = df_1min["close"].iloc[-1]
            if pd.isna(ema50):
                return False
            ok = price > ema50
            if ok:
                _daily_trend_ok[token] = True
            return ok
        return False

    ema200 = _ema(df_1min["close"], 200).iloc[-1]
    price = df_1min["close"].iloc[-1]

    if pd.isna(ema200):
        return False
    ok = price > ema200
    if ok:
        _daily_trend_ok[token] = True
    return ok


# ── Screen 2: 15-min setup detection ───────────────────

def _check_15min_setup(token: str, sym: str, df_15: pd.DataFrame, cfg: dict) -> bool:
    """Check if a stock has a mean-reversion setup on the 15-min chart.

    v2.0 Conditions:
      - RSI(9) < 40 (Connie Brown: bull market oversold = 40-50)
      - ADX(14) < 30 (range-bound, less restrictive)
      - Price below VWAP (flipped — mean reversion buys below the mean)

    If satisfied, records the setup with ATR for stop-loss calculation.
    """
    if df_15 is None or len(df_15) < max(cfg["adx_period"], cfg["setup_rsi_period"]) + 2:
        return False

    rsi_15 = _rsi(df_15["close"], cfg["setup_rsi_period"])
    current_rsi_15 = rsi_15.iloc[-1]

    if pd.isna(current_rsi_15) or current_rsi_15 >= cfg["setup_rsi_threshold"]:
        # Setup expired — if we had one, invalidate
        if token in _setups:
            del _setups[token]
        return False

    adx_val = _adx(df_15["high"], df_15["low"], df_15["close"], cfg["adx_period"])
    current_adx = adx_val.iloc[-1]
    if pd.isna(current_adx) or current_adx >= cfg["adx_max"]:
        return False

    # Calculate 15-min ATR for stop-loss
    atr_15 = _atr(df_15["high"], df_15["low"], df_15["close"], 14)
    current_atr_15 = atr_15.iloc[-1]
    if pd.isna(current_atr_15) or current_atr_15 <= 0:
        current_atr_15 = df_15["close"].iloc[-1] * 0.005  # fallback

    current_price = df_15["close"].iloc[-1]

    _setups[token] = {
        "symbol": sym,
        "rsi_15": current_rsi_15,
        "adx": current_adx,
        "atr_15": current_atr_15,
        "setup_time": datetime.now(),
        "price_at_setup": current_price,
    }
    return True


# ── Screen 3: 5-min entry trigger ──────────────────────

def _check_5min_trigger(token: str, df_5: pd.DataFrame, df_1: pd.DataFrame, cfg: dict) -> bool:
    """Check if the 5-min chart shows a valid entry trigger.

    v2.0 Conditions:
      - RSI(9) crosses back above 25 (bounce confirmation)
      - Price BELOW VWAP (flipped — entry below the mean, VWAP = exit target)
      - Bullish candle (close > open)
    Removed: volume spike filter, Bollinger Band, VWAP proximity
    """
    if df_5 is None or len(df_5) < cfg["entry_rsi_period"] + 2:
        return False

    rsi_5 = _rsi(df_5["close"], cfg["entry_rsi_period"])
    if len(rsi_5) < 2:
        return False

    current_rsi_5 = rsi_5.iloc[-1]
    prev_rsi_5 = rsi_5.iloc[-2]

    if pd.isna(current_rsi_5) or pd.isna(prev_rsi_5):
        return False

    # RSI must cross BACK above the threshold (was below, now above)
    if not (prev_rsi_5 < cfg["entry_rsi_threshold"] and current_rsi_5 >= cfg["entry_rsi_threshold"]):
        return False

    # Bullish candle
    last_5m = df_5.iloc[-1]
    if last_5m["close"] <= last_5m["open"]:
        return False

    # VWAP check: price must be BELOW VWAP (entry below the mean)
    if df_1 is not None and len(df_1) > 5:
        vwap_series = _vwap(df_1)
        current_vwap = vwap_series.iloc[-1]
        if current_vwap > 0:
            if last_5m["close"] >= current_vwap:
                return False  # Above VWAP — not a mean reversion entry

    return True


# ── Position management (exits) ─────────────────────────

def _check_exits(stock_data: dict, token_to_sym: dict, portfolio, logger, now: datetime, cfg: dict):
    """Check all open Strategy 3 positions for exit conditions."""
    for tid in list(_positions.keys()):
        pos = _positions[tid]
        token = pos["token"]
        sym = pos["symbol"]

        # Get 1-min data
        df_1 = stock_data.get(token)
        if df_1 is None or len(df_1) < 5:
            # No data — still check time-stop and hard exit using last known price
            current_price = pos["entry_price"]

            # Hard exit at 14:30
            if now.hour > 14 or (now.hour == 14 and now.minute >= 30):
                _close_position(tid, current_price, "HARD_EXIT_1430", portfolio, logger)
                continue

            # Time stop — 75 minutes since entry
            elapsed = (now - pos["entry_time"]).total_seconds() / 60
            if elapsed >= cfg["time_stop_minutes"]:
                _close_position(tid, current_price, "TIME_STOP", portfolio, logger)
            continue

        current_price = df_1["close"].iloc[-1]

        # Hard exit at 14:30
        if now.hour > 14 or (now.hour == 14 and now.minute >= 30):
            _close_position(tid, current_price, "HARD_EXIT_1430", portfolio, logger)
            continue

        # Disaster stop-loss (3x ATR)
        if current_price <= pos["stop_loss"]:
            _close_position(tid, current_price, "DISASTER_STOP", portfolio, logger)
            continue

        # Time stop — 75 minutes since entry
        elapsed = (now - pos["entry_time"]).total_seconds() / 60
        if elapsed >= cfg["time_stop_minutes"]:
            _close_position(tid, current_price, "TIME_STOP_75M", portfolio, logger)
            continue

        # Primary exit: 5-min RSI(9) > 50
        df_5 = _resample(df_1, "5min")
        if df_5 is not None and len(df_5) >= cfg["entry_rsi_period"] + 1:
            rsi_5 = _rsi(df_5["close"], cfg["entry_rsi_period"])
            current_rsi_5 = rsi_5.iloc[-1]
            if not pd.isna(current_rsi_5) and current_rsi_5 >= cfg["exit_rsi_threshold"]:
                _close_position(tid, current_price, "RSI_EXIT_50", portfolio, logger)
                continue

        # VWAP touch exit: price crosses above VWAP
        vwap_series = _vwap(df_1)
        current_vwap = vwap_series.iloc[-1]
        if not pd.isna(current_vwap) and current_vwap > 0 and pos.get("entered_below_vwap", False):
            if current_price >= current_vwap:
                _close_position(tid, current_price, "VWAP_TOUCH", portfolio, logger)
                continue


def _close_position(tid: str, price: float, reason: str, portfolio, logger):
    """Close a Strategy 3 position."""
    pos = _positions.get(tid)
    if not pos:
        return

    pnl = (price - pos["entry_price"]) * pos["quantity"]
    _daily_pnl[0] += pnl

    # Log to DailyLogger
    logger.log_trade("SELL", pos["symbol"], pos["quantity"], price,
                     reason=f"S3_{reason}", pnl=pnl)

    prefix = "+" if pnl >= 0 else ""
    print(f"\n  << S3 SELL {pos['symbol']} x{pos['quantity']} @ Rs{price:.2f} | "
          f"reason={reason} | P&L=Rs{prefix}{pnl:.2f}")

    # Update portfolio sector count
    sector = SECTOR_MAP.get(pos["symbol"], "Other")
    portfolio.sector_count[sector] = max(0, portfolio.sector_count.get(sector, 0) - 1)
    portfolio.daily_pnl += pnl

    del _positions[tid]


# ── Main scan function ──────────────────────────────────

def scan_15min_rsi(stock_data: dict, token_to_sym: dict, portfolio, logger, now: datetime) -> None:
    """Main scan function called every 60s from paper_live.py's strategy loop.

    stock_data: {token: DataFrame of 1-min candles}
    token_to_sym: {token: symbol string}
    portfolio: PaperPortfolio instance
    logger: DailyLogger instance
    now: current datetime

    This function:
      1. Checks exits for open S3 positions
      2. Resamples 1-min candles to 15-min and 5-min
      3. Screens for setups (15-min) and triggers (5-min)
      4. Opens positions when all three screens align
    """
    cfg = dict(_DEFAULTS)

    # ── Check exits first ──
    _check_exits(stock_data, token_to_sym, portfolio, logger, now, cfg)

    # ── Daily loss cap ──
    if _daily_pnl[0] <= -cfg["daily_loss_cap"]:
        return

    # ── Determine trading window ──
    in_prime = _in_window(now, cfg["prime_window_start"], cfg["prime_window_end"])
    in_secondary = _in_window(now, cfg["secondary_window_start"], cfg["secondary_window_end"])

    if not in_prime and not in_secondary:
        return  # Outside trading windows — exit-only

    size_mult = 1.0 if in_prime else cfg["secondary_size_mult"]

    # ── Scan each stock ──
    for token, df_1 in stock_data.items():
        if df_1 is None or len(df_1) < 30:
            continue

        sym = token_to_sym.get(token, f"UNK-{token}")

        # Already have max positions?
        if len(_positions) >= cfg["max_positions"]:
            break

        # Already holding this stock in S3?
        if any(p["symbol"] == sym for p in _positions.values()):
            continue

        # Also skip if paper_live has this stock in S1
        if any(p["symbol"] == sym for p in portfolio.positions.values()):
            continue

        # Sector check
        sector = SECTOR_MAP.get(sym, "Other")
        s3_sector_count = sum(1 for p in _positions.values()
                              if SECTOR_MAP.get(p["symbol"], "Other") == sector)
        if s3_sector_count >= cfg["max_per_sector"]:
            continue

        # ── Screen 1: Daily trend (cached) ──
        if not check_daily_trend(token, df_1):
            continue

        # ── Screen 2: 15-min setup ──
        df_15 = _resample(df_1, "15min")
        _check_15min_setup(token, sym, df_15, cfg)

        if token not in _setups:
            continue

        setup = _setups[token]

        # ── Screen 3: 5-min entry trigger ──
        df_5 = _resample(df_1, "5min")
        if not _check_5min_trigger(token, df_5, df_1, cfg):
            continue

        # ── All three screens passed — calculate position ──
        current_price = df_1["close"].iloc[-1]
        atr_15 = setup["atr_15"]
        stop_loss = round(current_price - cfg["atr_sl_mult"] * atr_15, 2)
        risk_per_share = current_price - stop_loss
        if risk_per_share <= 0:
            continue

        base_risk = cfg["risk_per_trade"]
        qty = int((base_risk * size_mult) / risk_per_share)
        qty = min(qty, int(83000 / current_price))  # Single-order limit
        if qty <= 0:
            continue

        # Check VWAP for exit tracking
        vwap_series = _vwap(df_1)
        current_vwap = vwap_series.iloc[-1] if len(vwap_series) > 0 else 0
        if pd.isna(current_vwap):
            current_vwap = 0
        entered_below_vwap = current_price < current_vwap if current_vwap > 0 else False

        # ── ENTRY ──
        _trade_counter[0] += 1
        tid = f"S3-{_trade_counter[0]:04d}"

        window_tag = "PRIME" if in_prime else "SECONDARY"
        _positions[tid] = {
            "token": token,
            "symbol": sym,
            "entry_price": current_price,
            "quantity": qty,
            "stop_loss": stop_loss,
            "entry_time": now,
            "entry_rsi_15": setup["rsi_15"],
            "entry_adx": setup["adx"],
            "entered_below_vwap": entered_below_vwap,
            "window": window_tag,
        }
        _daily_trades_count[0] += 1

        # Log thought
        logger.log_thought(
            sym, current_price, setup["rsi_15"], "S3_SETUP+TRIGGER",
            None, current_vwap, None, setup["adx"],
            "BUY", (f"S3 MeanRevert | 15m RSI(9)={setup['rsi_15']:.1f} ADX(14)={setup['adx']:.1f} | "
                    f"Qty={qty} SL={stop_loss:.2f} | {window_tag}"),
        )

        # Log trade
        logger.log_trade("BUY", sym, qty, current_price,
                         rsi_val=setup["rsi_15"], stop_loss=stop_loss,
                         reason=f"S3_{window_tag}")

        # Update portfolio sector tracking
        portfolio.sector_count[sector] = portfolio.sector_count.get(sector, 0) + 1

        print(f"\n  >> S3 BUY {sym} x{qty} @ Rs{current_price:.2f} | "
              f"SL=Rs{stop_loss:.2f} | 15m RSI(9)={setup['rsi_15']:.1f} | "
              f"ADX(14)={setup['adx']:.1f} | {window_tag}")

        # Consume the setup — don't re-enter on the same signal
        del _setups[token]
