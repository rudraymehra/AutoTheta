import logging

from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import INDIA_VIX_TOKEN, NIFTY_SPOT_TOKEN

log = logging.getLogger("autotheta.data")


class MarketData:
    """Fetches live market data from Angel One SmartAPI."""

    def __init__(self, api):
        """
        Args:
            api: Authenticated SmartConnect instance (from AngelSession.api)
        """
        self.api = api

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=15), reraise=True)
    def _fetch_ltp(self, exchange: str, symbol: str, token: str) -> float:
        """Fetch last traded price with retry logic."""
        result = self.api.ltpData(exchange, symbol, token)
        if not result or not result.get("status"):
            raise RuntimeError(f"LTP fetch failed for {symbol}: {result}")
        return float(result["data"]["ltp"])

    def get_nifty_spot(self) -> float:
        """Get current Nifty 50 spot price."""
        price = self._fetch_ltp("NSE", "NIFTY", NIFTY_SPOT_TOKEN)
        log.info("Nifty spot: %.2f", price)
        return price

    def get_india_vix(self) -> float:
        """Get current India VIX value."""
        vix = self._fetch_ltp("NSE", "India VIX", INDIA_VIX_TOKEN)
        log.info("India VIX: %.2f", vix)
        return vix

    def get_option_ltp(self, symbol: str, token: str) -> float:
        """Get LTP for an NFO option contract."""
        return self._fetch_ltp("NFO", symbol, token)

    def get_option_premiums(self, put_info: dict, call_info: dict) -> tuple[float, float]:
        """Fetch premiums for a put and call contract pair.

        Args:
            put_info: dict with 'symbol' and 'token' for the put
            call_info: dict with 'symbol' and 'token' for the call

        Returns:
            (put_premium, call_premium)
        """
        put_ltp = self.get_option_ltp(put_info["symbol"], put_info["token"])
        call_ltp = self.get_option_ltp(call_info["symbol"], call_info["token"])
        log.info("Premiums — Put %s: %.2f | Call %s: %.2f",
                 put_info["symbol"], put_ltp, call_info["symbol"], call_ltp)
        return put_ltp, call_ltp
