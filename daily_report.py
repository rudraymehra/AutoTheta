"""AutoTheta v2.0 — Daily Report Generator

Reads today's thoughts.csv and trades.csv, generates a human-readable
daily diary for each strategy. Run at end of day or anytime.

Output: logs/YYYY-MM-DD/report.txt
"""

import csv
import os
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
CAPITAL = 250000


def load_csv(path):
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def generate_report(report_date=None):
    report_date = report_date or date.today()
    date_str = report_date.isoformat()
    log_dir = PROJECT_ROOT / "logs" / date_str

    if not log_dir.exists():
        print(f"No logs found for {date_str}")
        return

    thoughts = load_csv(log_dir / "thoughts.csv")
    trades = load_csv(log_dir / "trades.csv")

    # ── Analyze thoughts ──
    # S1 v2.0: RSI(4) on 5-min with hook confirmation
    total_hooks = len([t for t in thoughts if t.get("Signal") == "RSI4_HOOK"])
    total_watching = len([t for t in thoughts if t.get("Decision") == "WATCHING"])
    total_filtered = len([t for t in thoughts if t.get("Decision") == "FILTERED"])
    total_skipped = len([t for t in thoughts if t.get("Decision") == "SKIP"])
    total_bought = len([t for t in thoughts if t.get("Decision") == "BUY" and "S3" not in t.get("Signal", "")])

    # Stocks that triggered
    triggered_stocks = set()
    for t in thoughts:
        if t.get("Signal") == "RSI4_HOOK":
            triggered_stocks.add(t.get("Stock", ""))

    # Filter breakdown
    filter_reasons = defaultdict(int)
    for t in thoughts:
        if t.get("Decision") == "FILTERED":
            reason = t.get("Reason", "")
            if "EMA" in reason:
                filter_reasons["Below EMA(200)"] += 1
            elif "ADX" in reason:
                filter_reasons["ADX too high (trending)"] += 1
            elif "VWAP" in reason:
                filter_reasons["VWAP distance wrong"] += 1
            else:
                filter_reasons["Other"] += 1

    # Stocks that were oversold the most
    oversold_counts = defaultdict(int)
    lowest_rsi = {}
    for t in thoughts:
        stock = t.get("Stock", "")
        rsi_str = t.get("RSI(4)_5m", "50") or "50"
        try:
            rsi_val = float(rsi_str)
        except ValueError:
            rsi_val = 50.0
        if rsi_val < 15:
            oversold_counts[stock] += 1
            if stock not in lowest_rsi or rsi_val < lowest_rsi[stock]:
                lowest_rsi[stock] = rsi_val

    # ── Analyze trades — split by strategy ──
    # S3 trades have "S3_" prefix in their Reason field
    buys = [t for t in trades if t.get("Action") == "BUY"]
    sells = [t for t in trades if t.get("Action") == "SELL"]

    s1_buys = [t for t in buys if not (t.get("Reason", "").startswith("S3_"))]
    s1_sells = [t for t in sells if not (t.get("Reason", "").startswith("S3_"))]
    s3_buys = [t for t in buys if t.get("Reason", "").startswith("S3_")]
    s3_sells = [t for t in sells if t.get("Reason", "").startswith("S3_")]

    total_pnl = 0
    winning_trades = 0
    losing_trades = 0
    for s in sells:
        pnl = float(s.get("P&L", "0") or "0")
        total_pnl += pnl
        if pnl > 0:
            winning_trades += 1
        elif pnl < 0:
            losing_trades += 1

    s3_pnl = 0
    s3_wins = 0
    s3_losses = 0
    for s in s3_sells:
        pnl = float(s.get("P&L", "0") or "0")
        s3_pnl += pnl
        if pnl > 0:
            s3_wins += 1
        elif pnl < 0:
            s3_losses += 1

    s1_pnl = total_pnl - s3_pnl

    pnl_pct = (total_pnl / CAPITAL) * 100

    # S3 thought analysis
    s3_thoughts = [t for t in thoughts if "S3_" in t.get("Signal", "") or "S3" in t.get("Reason", "")]
    s3_setups = [t for t in thoughts if t.get("Signal") == "S3_SETUP+TRIGGER"]

    # ── Determine market mood ──
    below_ema = filter_reasons.get("Below EMA(200)", 0)
    adx_high = filter_reasons.get("ADX too high (trending)", 0)
    vwap_wrong = filter_reasons.get("VWAP distance wrong", 0)

    if total_hooks == 0 and total_watching == 0:
        market_mood = "CALM"
        market_desc = "No stocks showed RSI(4) oversold on 5-min. Market was steady — no deep dips to trade."
    elif below_ema + adx_high > total_hooks * 0.7:
        market_mood = "TRENDING"
        market_desc = (
            f"Stocks were dipping (RSI(4) hook triggered {total_hooks} times) but in a trending "
            f"regime — high ADX or below EMA(200). Mean reversion doesn't work in trends. "
            f"The bot correctly stayed out."
        )
    elif total_bought > 0:
        market_mood = "RANGE-BOUND"
        market_desc = (
            f"Some stocks dipped below VWAP and bounced in a range-bound regime. "
            f"The bot found {total_bought} quality setups where RSI(4) hooked from below 15 "
            f"with price 0.3-1.2% below VWAP in a low-ADX environment."
        )
    elif vwap_wrong > total_hooks * 0.5:
        market_mood = "MISPOSITIONED"
        market_desc = (
            f"Stocks dipped but VWAP positioning was wrong — either too close to VWAP "
            f"(no room to bounce) or too far below (catching falling knife). "
            f"The sweet spot is 0.3-1.2% below VWAP."
        )
    else:
        market_mood = "MIXED"
        market_desc = (
            f"Some signals appeared but didn't pass all 4 filters. "
            f"The market wasn't clearly range-bound enough for mean reversion."
        )

    # ── Build report ──
    lines = []
    lines.append("=" * 60)
    lines.append(f"  AutoTheta v2.0 Daily Diary — {date_str}")
    lines.append("=" * 60)
    lines.append("")

    # Market overview
    lines.append("  MARKET MOOD TODAY: " + market_mood)
    lines.append("  " + "-" * 56)
    lines.append(f"  {market_desc}")
    lines.append("")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Strategy 1: RSI(4) Mean Reversion on 5-min
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    lines.append("  +----------------------------------------------------+")
    lines.append("  |  STRATEGY 1: RSI(4) Mean Reversion on 5-min        |")
    lines.append("  +----------------------------------------------------+")
    lines.append("")
    lines.append("  Entry: RSI(4) drops below 15, hooks back above 15")
    lines.append("  Filters: EMA(200), ADX(14)<25, VWAP 0.3-1.2% below")
    lines.append("  Exit: 60% at VWAP touch, 40% at RSI>50, 75-min timeout")
    lines.append("")

    if total_hooks == 0 and total_watching == 0:
        lines.append("  Today was a quiet day. No stock's RSI(4) dropped below 15")
        lines.append("  on the 5-min chart. This happens on steady, range-bound days")
        lines.append("  without sharp intraday dips.")
    else:
        lines.append(f"  What the bot saw:")
        lines.append(f"    * {total_watching + total_hooks} times a stock's 5-min RSI(4) was near/below 15")
        lines.append(f"    * {total_hooks} RSI hook signals (dropped below 15, bounced back)")
        lines.append(f"    * {len(triggered_stocks)} stocks triggered: {', '.join(sorted(triggered_stocks)) if triggered_stocks else 'none'}")
        lines.append("")

        if oversold_counts:
            most_oversold = sorted(oversold_counts.items(), key=lambda x: -x[1])[:5]
            lines.append(f"  Most oversold stocks today (5-min RSI(4)):")
            for stock, count in most_oversold:
                rsi_low = lowest_rsi.get(stock, 0)
                lines.append(f"    * {stock:18s} — RSI(4) hit {rsi_low:.1f} (triggered {count}x)")
            lines.append("")

        if total_filtered > 0:
            lines.append(f"  Why the bot DIDN'T trade ({total_filtered} signals filtered):")
            for reason, count in sorted(filter_reasons.items(), key=lambda x: -x[1]):
                if reason == "Below EMA(200)":
                    lines.append(f"    * {count}x Below EMA(200) — stock in long-term downtrend")
                    lines.append(f"      (mean reversion fails when the trend is against you)")
                elif reason == "ADX too high (trending)":
                    lines.append(f"    * {count}x ADX(14) >= 25 — market trending, not range-bound")
                    lines.append(f"      (mean reversion needs range-bound conditions)")
                elif reason == "VWAP distance wrong":
                    lines.append(f"    * {count}x VWAP distance outside 0.3-1.2% below")
                    lines.append(f"      (too close = no room to bounce, too far = falling knife)")
                else:
                    lines.append(f"    * {count}x {reason}")
            lines.append("")

        if total_skipped > 0:
            lines.append(f"  Skipped {total_skipped} signals due to position/sector limits")
            lines.append("")

    # S1 Trades
    if s1_buys:
        lines.append(f"  What S1 DID:")
        lines.append(f"    * Entered {len(s1_buys)} trade(s)")
        for b in s1_buys:
            lines.append(f"      BUY {b.get('Stock', '')} x{b.get('Qty', '')} @ Rs{b.get('Price', '')} (RSI(4)={b.get('RSI','')})")
        lines.append("")

    if s1_sells:
        lines.append(f"  How S1 trades ended:")
        for s in s1_sells:
            pnl = float(s.get("P&L", "0") or "0")
            tag = "WIN" if pnl > 0 else "LOSS"
            reason = s.get("Reason", "")
            if "VWAP" in reason:
                reason_desc = "VWAP touch (60% exit)"
            elif "RSI4" in reason:
                reason_desc = "RSI(4) > 50 (final exit)"
            elif "TIME" in reason:
                reason_desc = "75-min timeout"
            elif "DISASTER" in reason:
                reason_desc = "3x ATR disaster stop"
            elif "HARD" in reason:
                reason_desc = "3:00 PM hard exit"
            else:
                reason_desc = reason
            lines.append(f"      [{tag}] SELL {s.get('Stock', '')} x{s.get('Qty', '')} @ Rs{s.get('Price', '')} "
                         f"| {reason_desc} | Rs{pnl:+,.2f}")
        lines.append("")

    if not s1_buys and not s1_sells:
        lines.append(f"  S1 Trades: NONE")
        lines.append(f"  The bot saw opportunities but the 4 filters blocked them all.")
        lines.append(f"  This is the bot protecting your capital. Fewer filters = more")
        lines.append(f"  trades but worse quality. No trade > Bad trade.")
        lines.append("")

    lines.append(f"  S1 P&L: Rs{s1_pnl:+,.2f}")
    lines.append("")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Strategy 3: Multi-Timeframe RSI Mean Reversion
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    lines.append("  +----------------------------------------------------+")
    lines.append("  |  STRATEGY 3: RSI 15-min Mean Reversion (v2.0)      |")
    lines.append("  +----------------------------------------------------+")
    lines.append("")
    lines.append("  Setup: 15-min RSI(9) < 40, ADX(14) < 30")
    lines.append("  Trigger: 5-min RSI(9) crosses above 25, price below VWAP")
    lines.append("  Exit: RSI(9) > 50 or VWAP touch, 75-min timeout, 3x ATR stop")
    lines.append("")

    if not s3_buys and not s3_sells and not s3_setups:
        lines.append("  No setups triggered on the 15-min chart today.")
        lines.append("  This means either:")
        lines.append("    * No stock's 15-min RSI(9) dropped below 40")
        lines.append("    * Or setups appeared but the 5-min entry trigger never fired")
        lines.append("  Patience — mean-reversion needs real pullbacks, not noise.")
    else:
        if s3_setups:
            lines.append(f"  Setups detected: {len(s3_setups)}")
            s3_stocks = set(t.get("Stock", "") for t in s3_setups)
            lines.append(f"    Stocks: {', '.join(sorted(s3_stocks))}")
            lines.append("")

        if s3_buys:
            lines.append(f"  S3 Entries: {len(s3_buys)} trade(s)")
            for b in s3_buys:
                window = b.get("Reason", "").replace("S3_", "")
                lines.append(f"    BUY {b.get('Stock', '')} x{b.get('Qty', '')} @ Rs{b.get('Price', '')} "
                             f"(RSI={b.get('RSI','')}) [{window}]")
            lines.append("")

        if s3_sells:
            lines.append(f"  S3 Exits:")
            for s in s3_sells:
                pnl = float(s.get("P&L", "0") or "0")
                tag = "WIN" if pnl > 0 else "LOSS"
                reason = s.get("Reason", "").replace("S3_", "")
                lines.append(f"    [{tag}] SELL {s.get('Stock', '')} x{s.get('Qty', '')} @ Rs{s.get('Price', '')} "
                             f"| {reason} | Rs{pnl:+,.2f}")
            lines.append("")

    s3_pnl_pct = (s3_pnl / CAPITAL) * 100 if CAPITAL > 0 else 0
    lines.append(f"  S3 P&L: Rs{s3_pnl:+,.2f} ({s3_pnl_pct:+.2f}%)")
    if s3_sells:
        s3_wr = s3_wins / len(s3_sells) * 100
        lines.append(f"  S3 Win Rate: {s3_wins}/{len(s3_sells)} ({s3_wr:.0f}%)")
    lines.append("")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Combined P&L
    lines.append("  " + "-" * 56)
    lines.append(f"  DAILY P&L:          Rs{total_pnl:+,.2f} ({pnl_pct:+.2f}%)")
    if sells:
        win_rate = winning_trades / len(sells) * 100 if sells else 0
        lines.append(f"  Win Rate:           {winning_trades}/{len(sells)} ({win_rate:.0f}%)")
    lines.append(f"  Capital:            Rs{CAPITAL:,} -> Rs{CAPITAL + total_pnl:,.2f}")
    lines.append("")

    # What-if analysis
    if thoughts:
        lines.append("  +----------------------------------------------------+")
        lines.append("  |  WHAT-IF: If the bot ignored ALL filters?          |")
        lines.append("  +----------------------------------------------------+")
        lines.append("")
        lines.append("  v2.0 uses only 4 filters (was 7). If the bot had")
        lines.append("  blindly bought every RSI(4) < 15 hook signal today:")
        lines.append(f"    * It would have entered {total_hooks} trades")
        if market_mood == "TRENDING":
            lines.append(f"    * Most were in trending markets (high ADX)")
            lines.append(f"    * Mean reversion in trends = guaranteed losses")
            lines.append(f"    * The 4 filters saved you from ~{total_filtered} bad trades")
        elif market_mood == "CALM":
            lines.append(f"    * No signals = nothing to trade either way")
        else:
            lines.append(f"    * Mixed results — some would have worked, some wouldn't")
            lines.append(f"    * Filters kept only the highest-quality setups")
        lines.append("")

    # Tomorrow outlook
    lines.append("  +----------------------------------------------------+")
    lines.append("  |  LOOKING AHEAD                                     |")
    lines.append("  +----------------------------------------------------+")
    lines.append("")
    weekday = (report_date.weekday() + 1) % 7  # 0=Sun
    tomorrow_weekday = (weekday + 1) % 7
    if tomorrow_weekday == 2:  # Tuesday
        lines.append("  Tomorrow is TUESDAY — Nifty expiry day!")
        lines.append("  All three strategies will be active: RSI(4) Mean Reversion,")
        lines.append("  Expiry Skew, and RSI 15-min Mean Reversion.")
    elif tomorrow_weekday in (0, 6):  # Weekend
        lines.append("  Tomorrow is weekend — market closed. Rest up.")
    else:
        lines.append("  RSI(4) Mean Reversion + RSI 15-min will run tomorrow.")
        lines.append("  Bot starts automatically at 9:10 AM.")

    if market_mood == "TRENDING":
        lines.append("  If the trend continues tomorrow, expect another quiet day.")
        lines.append("  Mean reversion shines when the trend exhausts and we go range-bound.")
    lines.append("")
    lines.append("=" * 60)

    report_text = "\n".join(lines)

    # Save report
    report_path = log_dir / "report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)

    print(report_text)
    print(f"\n  Saved to: {report_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        d = date.fromisoformat(sys.argv[1])
    else:
        d = date.today()
    generate_report(d)
