"""AutoTheta — Daily Report Generator

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
    total_signals = len([t for t in thoughts if t.get("Signal") == "RSI<20"])
    total_watching = len([t for t in thoughts if t.get("Decision") == "WATCHING"])
    total_filtered = len([t for t in thoughts if t.get("Decision") == "FILTERED"])
    total_skipped = len([t for t in thoughts if t.get("Decision") == "SKIP"])
    total_bought = len([t for t in thoughts if t.get("Decision") == "BUY"])

    # Stocks that triggered
    triggered_stocks = set()
    for t in thoughts:
        if t.get("Signal") == "RSI<20":
            triggered_stocks.add(t["Stock"])

    # Filter breakdown
    filter_reasons = defaultdict(int)
    for t in thoughts:
        if t.get("Decision") == "FILTERED":
            reason = t.get("Reason", "")
            if "VWAP" in reason:
                filter_reasons["Below VWAP"] += 1
            elif "EMA" in reason:
                filter_reasons["Below 5m EMA20"] += 1
            elif "olume" in reason.lower():
                filter_reasons["Low volume"] += 1
            else:
                filter_reasons["Other"] += 1

    # Stocks that were oversold the most
    oversold_counts = defaultdict(int)
    lowest_rsi = {}
    for t in thoughts:
        stock = t.get("Stock", "")
        rsi_val = float(t.get("RSI(7)", "50") or "50")
        if rsi_val < 20:
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
    below_vwap = filter_reasons.get("Below VWAP", 0)
    below_ema = filter_reasons.get("Below 5m EMA20", 0)

    if total_signals == 0:
        market_mood = "CALM"
        market_desc = "No stocks hit RSI oversold territory. Market was steady — no big dips to trade."
    elif below_vwap + below_ema > total_signals * 0.7:
        market_mood = "BEARISH"
        market_desc = (
            f"Stocks were dipping (RSI < 20 triggered {total_signals} times) but the overall market "
            f"was falling — most stocks were below VWAP and EMA. The bot correctly stayed out "
            f"because buying dips in a falling market is how you lose money."
        )
    elif total_bought > 0:
        market_mood = "MIXED"
        market_desc = (
            f"Some stocks dipped while others held up. The bot found {total_bought} quality setups "
            f"where a stock was oversold but still in an overall uptrend (above VWAP and EMA)."
        )
    elif filter_reasons.get("Low volume", 0) > total_signals * 0.5:
        market_mood = "QUIET"
        market_desc = (
            f"Stocks dipped but on thin volume — not real selling pressure, just noise. "
            f"The bot filtered these out because low-volume dips don't bounce reliably."
        )
    else:
        market_mood = "MIXED"
        market_desc = (
            f"Some signals appeared but didn't pass all filters. "
            f"The market wasn't clearly trending in either direction."
        )

    # ── Build report ──
    lines = []
    lines.append("=" * 60)
    lines.append(f"  AutoTheta Daily Diary — {date_str}")
    lines.append("=" * 60)
    lines.append("")

    # Market overview
    lines.append("  MARKET MOOD TODAY: " + market_mood)
    lines.append("  " + "-" * 56)
    lines.append(f"  {market_desc}")
    lines.append("")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Strategy 1: RSI Bounce
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    lines.append("  ┌────────────────────────────────────────────────────┐")
    lines.append("  │  STRATEGY 1: RSI Oversold Bounce                   │")
    lines.append("  └────────────────────────────────────────────────────┘")
    lines.append("")

    if total_signals == 0 and total_watching == 0:
        lines.append("  Today was a quiet day. No stock got oversold enough")
        lines.append("  (RSI < 20) to even consider buying. This happens on")
        lines.append("  steady, range-bound days. The bot had nothing to do.")
    else:
        lines.append(f"  What the bot saw:")
        lines.append(f"    • {total_watching + total_signals} times a stock's RSI dropped near/below 20")
        lines.append(f"    • {len(triggered_stocks)} different stocks triggered: {', '.join(sorted(triggered_stocks)) if triggered_stocks else 'none'}")
        lines.append("")

        if oversold_counts:
            most_oversold = sorted(oversold_counts.items(), key=lambda x: -x[1])[:5]
            lines.append(f"  Most oversold stocks today:")
            for stock, count in most_oversold:
                rsi_low = lowest_rsi.get(stock, 0)
                lines.append(f"    • {stock:18s} — RSI hit {rsi_low:.1f} (triggered {count}x)")
            lines.append("")

        if total_filtered > 0:
            lines.append(f"  Why the bot DIDN'T trade ({total_filtered} signals filtered):")
            for reason, count in sorted(filter_reasons.items(), key=lambda x: -x[1]):
                if reason == "Below VWAP":
                    lines.append(f"    • {count}x Below VWAP — stock was below its fair price for the day")
                    lines.append(f"      (buying below VWAP = catching a falling knife)")
                elif reason == "Below 5m EMA20":
                    lines.append(f"    • {count}x Below 5-min EMA — short-term trend was down")
                    lines.append(f"      (don't buy dips when the trend is against you)")
                elif reason == "Low volume":
                    lines.append(f"    • {count}x Low volume — dip wasn't on real selling pressure")
                    lines.append(f"      (thin volume dips are noise, not opportunity)")
            lines.append("")

        if total_skipped > 0:
            lines.append(f"  Skipped {total_skipped} signals due to position limits or sector rules")
            lines.append("")

    # S1 Trades
    if s1_buys:
        lines.append(f"  What S1 DID:")
        lines.append(f"    • Entered {len(s1_buys)} trade(s)")
        for b in s1_buys:
            lines.append(f"      BUY {b['Stock']} x{b['Qty']} @ Rs{b['Price']} (RSI={b.get('RSI','')})")
        lines.append("")

    if s1_sells:
        lines.append(f"  How S1 trades ended:")
        for s in s1_sells:
            pnl = float(s.get("P&L", "0") or "0")
            tag = "WIN" if pnl > 0 else "LOSS"
            lines.append(f"      [{tag}] SELL {s['Stock']} x{s['Qty']} @ Rs{s['Price']} "
                         f"| {s.get('Reason','')} | Rs{pnl:+,.2f}")
        lines.append("")

    if not s1_buys and not s1_sells:
        lines.append(f"  S1 Trades: NONE")
        lines.append(f"  The bot saw opportunities but the filters blocked them all.")
        lines.append(f"  This is the bot protecting your capital on a bad day.")
        lines.append(f"  No trade > Bad trade.")
        lines.append("")

    lines.append(f"  S1 P&L: Rs{s1_pnl:+,.2f}")
    lines.append("")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Strategy 3: Multi-Timeframe RSI Mean Reversion
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    lines.append("  ┌────────────────────────────────────────────────────┐")
    lines.append("  │  STRATEGY 3: RSI 15-min Mean Reversion             │")
    lines.append("  └────────────────────────────────────────────────────┘")
    lines.append("")
    lines.append("  Triple-screen approach: daily trend + 15-min setup + 5-min trigger")
    lines.append("  Looks for oversold pullbacks within an uptrend (long-only)")
    lines.append("")

    if not s3_buys and not s3_sells and not s3_setups:
        lines.append("  No setups triggered on the 15-min chart today.")
        lines.append("  This means either:")
        lines.append("    • No stock's 15-min RSI(14) dropped below 30")
        lines.append("    • Or setups appeared but the 5-min entry trigger never fired")
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
                lines.append(f"    BUY {b['Stock']} x{b['Qty']} @ Rs{b['Price']} "
                             f"(RSI={b.get('RSI','')}) [{window}]")
            lines.append("")

        if s3_sells:
            lines.append(f"  S3 Exits:")
            for s in s3_sells:
                pnl = float(s.get("P&L", "0") or "0")
                tag = "WIN" if pnl > 0 else "LOSS"
                reason = s.get("Reason", "").replace("S3_", "")
                lines.append(f"    [{tag}] SELL {s['Stock']} x{s['Qty']} @ Rs{s['Price']} "
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
    lines.append(f"  Capital:            Rs{CAPITAL:,} → Rs{CAPITAL + total_pnl:,.2f}")
    lines.append("")

    # What-if analysis
    if thoughts:
        lines.append("  ┌────────────────────────────────────────────────────┐")
        lines.append("  │  WHAT-IF: If the bot ignored ALL filters?          │")
        lines.append("  └────────────────────────────────────────────────────┘")
        lines.append("")
        lines.append("  This shows why the filters matter. If the bot had")
        lines.append("  blindly bought every RSI < 20 signal today:")
        lines.append(f"    • It would have entered {total_signals} trades")
        if market_mood == "BEARISH":
            lines.append(f"    • Most were below VWAP = market was falling")
            lines.append(f"    • Likely LOST money on majority of them")
            lines.append(f"    • The filters saved you from ~{total_filtered} bad trades")
        elif market_mood == "CALM":
            lines.append(f"    • No signals = nothing to trade either way")
        else:
            lines.append(f"    • Mixed results — some would have worked, some wouldn't")
            lines.append(f"    • Filters kept only the highest-quality setups")
        lines.append("")

    # Tomorrow outlook
    lines.append("  ┌────────────────────────────────────────────────────┐")
    lines.append("  │  LOOKING AHEAD                                     │")
    lines.append("  └────────────────────────────────────────────────────┘")
    lines.append("")
    weekday = (report_date.weekday() + 1) % 7  # 0=Sun
    tomorrow_weekday = (weekday + 1) % 7
    if tomorrow_weekday == 2:  # Tuesday
        lines.append("  Tomorrow is TUESDAY — Nifty expiry day!")
        lines.append("  All three strategies will be active: RSI Bounce, Expiry Skew,")
        lines.append("  and RSI 15-min Mean Reversion.")
    elif tomorrow_weekday in (0, 6):  # Weekend
        lines.append("  Tomorrow is weekend — market closed. Rest up.")
    else:
        lines.append("  RSI Bounce + RSI 15-min Mean Reversion will run tomorrow.")
        lines.append("  Bot starts automatically at 9:10 AM.")

    if market_mood == "BEARISH":
        lines.append("  If the selloff continues tomorrow, expect another quiet day.")
        lines.append("  If there's a relief bounce, the bot will catch it.")
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
