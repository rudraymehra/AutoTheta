"""Trading fee calculator for Angel One options trades.

Uses April 2026 rates:
- STT on sell side: 0.15% (up from 0.1%)
- Brokerage: Flat Rs 20 per order
- GST: 18% on brokerage + exchange txn charges
- Exchange txn: 0.053% (NSE)
- SEBI turnover fee: 0.0001%
- Stamp duty on buy side: 0.003%
"""


def calculate_fees(premium: float, quantity: int, side: str) -> dict:
    """Calculate all trading costs for an options trade.

    Args:
        premium: Price per unit
        quantity: Number of units (lot size)
        side: 'BUY' or 'SELL'

    Returns:
        dict with individual fee components and 'total'
    """
    turnover = premium * quantity

    brokerage = 20.0  # Flat per order
    exchange_txn = turnover * 0.00053  # NSE transaction charge
    gst = (brokerage + exchange_txn) * 0.18  # 18% GST on brokerage + exchange txn
    stt = turnover * 0.0015 if side == "SELL" else 0.0  # 0.15% on sell side only
    sebi_fee = turnover * 0.000001  # SEBI turnover fee
    stamp = turnover * 0.00003 if side == "BUY" else 0.0  # Stamp duty on buy side

    total = brokerage + gst + stt + exchange_txn + sebi_fee + stamp

    return {
        "brokerage": round(brokerage, 2),
        "gst": round(gst, 2),
        "stt": round(stt, 2),
        "exchange_txn": round(exchange_txn, 2),
        "sebi_fee": round(sebi_fee, 2),
        "stamp": round(stamp, 2),
        "total": round(total, 2),
    }
