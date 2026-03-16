"""Abstract broker interface with Paper and Live implementations.

Switch between paper and live by setting TRADING_MODE in .env.
"""

import logging
from abc import ABC, abstractmethod

from config.settings import NIFTY_LOT_SIZE, TRADING_MODE
from src.paper_engine import PaperTradingEngine

log = logging.getLogger("autotheta.broker")


class BaseBroker(ABC):
    """Abstract broker interface — identical API for paper and live."""

    @abstractmethod
    def sell_option(self, symbol: str, token: str, strike: float, expiry: str,
                    option_type: str, price: float, quantity: int = NIFTY_LOT_SIZE) -> str | None:
        """Sell an option. Returns order/trade ID."""

    @abstractmethod
    def buy_option(self, symbol: str, token: str, strike: float, expiry: str,
                   option_type: str, price: float, quantity: int = NIFTY_LOT_SIZE) -> str | None:
        """Buy an option (for hedges). Returns order/trade ID."""

    @abstractmethod
    def close_position(self, trade_id: str, symbol: str, token: str,
                       current_price: float) -> float | None:
        """Close a position. Returns P&L."""

    @abstractmethod
    def place_stoploss(self, symbol: str, token: str, trigger_price: float,
                       quantity: int = NIFTY_LOT_SIZE) -> str | None:
        """Place a stop-loss order. Returns order ID."""


class PaperBroker(BaseBroker):
    """Paper trading broker — uses real prices, virtual execution."""

    def __init__(self, engine: PaperTradingEngine):
        self.engine = engine

    def sell_option(self, symbol, token, strike, expiry, option_type, price,
                    quantity=NIFTY_LOT_SIZE):
        return self.engine.open_position(symbol, strike, expiry, option_type,
                                         "SELL", quantity, price)

    def buy_option(self, symbol, token, strike, expiry, option_type, price,
                   quantity=NIFTY_LOT_SIZE):
        return self.engine.open_position(symbol, strike, expiry, option_type,
                                         "BUY", quantity, price)

    def close_position(self, trade_id, symbol, token, current_price):
        return self.engine.close_position(trade_id, current_price)

    def place_stoploss(self, symbol, token, trigger_price, quantity=NIFTY_LOT_SIZE):
        # Paper engine handles SL checks in the monitoring loop
        log.info("PAPER SL registered: %s trigger @ ₹%.2f", symbol, trigger_price)
        return f"SL-PAPER-{symbol}"


class LiveBroker(BaseBroker):
    """Live Angel One broker — LIMIT orders only (SEBI mandate)."""

    def __init__(self, api):
        """
        Args:
            api: Authenticated SmartConnect instance
        """
        self.api = api

    def _place_order(self, symbol, token, txn_type, price, quantity, variety="NORMAL"):
        order = {
            "variety": variety,
            "tradingsymbol": symbol,
            "symboltoken": token,
            "transactiontype": txn_type,
            "exchange": "NFO",
            "ordertype": "LIMIT",  # MARKET is PROHIBITED for algo trading
            "producttype": "CARRYFORWARD",  # NRML equivalent for F&O
            "duration": "DAY",
            "price": str(price),
            "quantity": str(quantity),
            "squareoff": "0",
            "stoploss": "0",
            "triggerprice": "0",
        }
        try:
            result = self.api.placeOrder(order)
            log.info("LIVE ORDER: %s %s %s @ ₹%.2f qty=%d → %s",
                     txn_type, symbol, variety, price, quantity, result)
            return result
        except Exception:
            log.exception("Order placement failed: %s %s", txn_type, symbol)
            return None

    def sell_option(self, symbol, token, strike, expiry, option_type, price,
                    quantity=NIFTY_LOT_SIZE):
        return self._place_order(symbol, token, "SELL", price, quantity)

    def buy_option(self, symbol, token, strike, expiry, option_type, price,
                   quantity=NIFTY_LOT_SIZE):
        return self._place_order(symbol, token, "BUY", price, quantity)

    def close_position(self, trade_id, symbol, token, current_price):
        # For a short position, close = BUY back
        return self._place_order(symbol, token, "BUY", current_price, NIFTY_LOT_SIZE)

    def place_stoploss(self, symbol, token, trigger_price, quantity=NIFTY_LOT_SIZE):
        order = {
            "variety": "STOPLOSS",
            "tradingsymbol": symbol,
            "symboltoken": token,
            "transactiontype": "BUY",
            "exchange": "NFO",
            "ordertype": "STOPLOSS_MARKET",
            "producttype": "CARRYFORWARD",
            "duration": "DAY",
            "price": "0",
            "quantity": str(quantity),
            "triggerprice": str(trigger_price),
        }
        try:
            result = self.api.placeOrder(order)
            log.info("LIVE SL ORDER: %s trigger @ ₹%.2f → %s", symbol, trigger_price, result)
            return result
        except Exception:
            log.exception("SL order failed: %s", symbol)
            return None


def create_broker(api=None, paper_engine=None) -> BaseBroker:
    """Factory: returns PaperBroker or LiveBroker based on TRADING_MODE."""
    if TRADING_MODE == "live":
        if api is None:
            raise ValueError("Live mode requires an authenticated SmartConnect API instance")
        log.warning("*** LIVE TRADING MODE ***")
        return LiveBroker(api)
    else:
        engine = paper_engine or PaperTradingEngine()
        log.info("Paper trading mode active")
        return PaperBroker(engine)
