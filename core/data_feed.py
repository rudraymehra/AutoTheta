"""Real-time data feed via SmartWebSocketV2 + REST candle builder.

Aggregates ticks into 1-min candles and distributes to subscribed strategies.
Also provides REST-based historical candle fetching.
"""

import asyncio
import logging
import threading
from collections import defaultdict
from datetime import datetime, timedelta

import pytz
from SmartApi.smartWebSocketV2 import SmartWebSocketV2

from core.rate_limiter import RateLimiter
from models.types import Candle

log = logging.getLogger("autotheta.data_feed")
IST = pytz.timezone("Asia/Kolkata")

# Max candles to keep in memory per token
MAX_CANDLE_HISTORY = 200


class DataFeed:
    """Manages WebSocket connection and builds 1-min candles from ticks."""

    def __init__(self, api, auth_token: str, api_key: str, client_code: str,
                 feed_token: str, rate_limiter: RateLimiter):
        self.api = api
        self._auth_token = auth_token
        self._api_key = api_key
        self._client_code = client_code
        self._feed_token = feed_token
        self._rate_limiter = rate_limiter

        # Candle storage: token -> list of Candle
        self._candles: dict[str, list[Candle]] = defaultdict(list)
        # Current building candle: (token, minute_key) -> partial candle data
        self._building: dict[tuple[str, str], dict] = {}
        # Subscribers: token -> list of asyncio.Queue
        self._subscribers: dict[str, list[asyncio.Queue]] = defaultdict(list)
        # Token -> symbol mapping
        self._token_symbols: dict[str, str] = {}

        self._ws: SmartWebSocketV2 | None = None
        self._ws_thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_token_map(self, token_map: dict[str, str]):
        """Set token -> symbol mapping (e.g., {"3045": "SBIN-EQ"})."""
        self._token_symbols = token_map

    def subscribe(self, token: str) -> asyncio.Queue:
        """Subscribe to candle updates for a token. Returns an asyncio.Queue."""
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._subscribers[token].append(q)
        return q

    def get_candles(self, token: str, count: int = 200) -> list[Candle]:
        """Get historical candles for a token from memory."""
        return self._candles[token][-count:]

    async def fetch_historical(self, token: str, symbol: str, exchange: str,
                               from_dt: datetime, to_dt: datetime,
                               interval: str = "ONE_MINUTE") -> list[Candle]:
        """Fetch historical candles via REST API."""
        await self._rate_limiter.acquire("historical")
        params = {
            "exchange": exchange,
            "symboltoken": token,
            "interval": interval,
            "fromdate": from_dt.strftime("%Y-%m-%d %H:%M"),
            "todate": to_dt.strftime("%Y-%m-%d %H:%M"),
        }
        try:
            result = self.api.getCandleData(params)
            if not result or not result.get("data"):
                log.warning("No historical data for %s", symbol)
                return []
            candles = []
            for row in result["data"]:
                # [timestamp, open, high, low, close, volume]
                ts = datetime.fromisoformat(row[0]) if isinstance(row[0], str) else row[0]
                candles.append(Candle(
                    timestamp=ts, open=float(row[1]), high=float(row[2]),
                    low=float(row[3]), close=float(row[4]), volume=int(row[5]),
                    token=token, symbol=symbol,
                ))
            # Store in memory
            self._candles[token] = candles[-MAX_CANDLE_HISTORY:]
            return candles
        except Exception:
            log.exception("Historical fetch failed for %s", symbol)
            return []

    def start_websocket(self, tokens: list[str], loop: asyncio.AbstractEventLoop):
        """Start WebSocket in a background thread."""
        self._loop = loop
        self._ws = SmartWebSocketV2(
            self._auth_token, self._api_key, self._client_code, self._feed_token,
        )

        def on_data(wsapp, msg):
            self._on_tick(msg)

        def on_open(wsapp):
            # Subscribe in Quote mode (mode=2) for OHLC + volume
            token_list = [{"exchangeType": 1, "tokens": tokens}]
            self._ws.subscribe("autotheta_feed", 2, token_list)
            log.info("WebSocket subscribed to %d tokens", len(tokens))

        def on_error(wsapp, error):
            log.error("WebSocket error: %s", error)

        def on_close(wsapp):
            log.warning("WebSocket closed")

        self._ws.on_data = on_data
        self._ws.on_open = on_open
        self._ws.on_error = on_error
        self._ws.on_close = on_close

        self._ws_thread = threading.Thread(target=self._ws.connect, daemon=True)
        self._ws_thread.start()
        log.info("WebSocket thread started")

    def stop_websocket(self):
        """Stop the WebSocket connection."""
        if self._ws:
            try:
                self._ws.close_connection()
            except Exception:
                pass
        log.info("WebSocket stopped")

    def _on_tick(self, msg: dict):
        """Process a tick from WebSocket — aggregate into 1-min candles."""
        try:
            token = str(msg.get("token", ""))
            ltp = msg.get("last_traded_price", 0) / 100  # Paise to rupees
            volume = msg.get("volume_trade_for_the_day", 0)

            if not token or ltp <= 0:
                return

            now = datetime.now(IST)
            minute_key = now.strftime("%Y-%m-%d %H:%M")
            key = (token, minute_key)

            if key not in self._building:
                # Finalize previous candle if exists
                self._finalize_candle(token, minute_key)
                self._building[key] = {
                    "o": ltp, "h": ltp, "l": ltp, "c": ltp,
                    "v": volume, "ts": now.replace(second=0, microsecond=0),
                }
            else:
                c = self._building[key]
                c["h"] = max(c["h"], ltp)
                c["l"] = min(c["l"], ltp)
                c["c"] = ltp
                c["v"] = volume
        except Exception:
            log.exception("Tick processing error")

    def _finalize_candle(self, token: str, current_minute: str):
        """Finalize all building candles for this token except the current minute."""
        to_remove = []
        for key, data in self._building.items():
            if key[0] == token and key[1] != current_minute:
                symbol = self._token_symbols.get(token, token)
                candle = Candle(
                    timestamp=data["ts"], open=data["o"], high=data["h"],
                    low=data["l"], close=data["c"], volume=data["v"],
                    token=token, symbol=symbol,
                )
                self._candles[token].append(candle)
                # Trim history
                if len(self._candles[token]) > MAX_CANDLE_HISTORY:
                    self._candles[token] = self._candles[token][-MAX_CANDLE_HISTORY:]
                # Notify subscribers
                self._notify_subscribers(token, candle)
                to_remove.append(key)

        for key in to_remove:
            del self._building[key]

    def _notify_subscribers(self, token: str, candle: Candle):
        """Push completed candle to all subscriber queues."""
        for q in self._subscribers.get(token, []):
            if self._loop:
                self._loop.call_soon_threadsafe(q.put_nowait, candle)
