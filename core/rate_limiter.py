"""Token bucket rate limiter shared across all strategies.

Angel One API limits:
- placeOrder: 20/sec, 1000/hour
- getMarketData/ltpData: 10/sec, 5000/hour
- getCandleData (historical): 3/sec
- searchScrip: 1/sec
"""

import asyncio
import time
from collections import defaultdict


class RateLimiter:
    """Async token bucket rate limiter with per-endpoint limits."""

    # Default limits: (requests_per_second, requests_per_hour)
    LIMITS = {
        "order": (20, 1000),
        "market_data": (10, 5000),
        "historical": (3, None),
        "search": (1, None),
    }

    def __init__(self):
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._last_call: dict[str, float] = defaultdict(float)
        self._hour_counts: dict[str, int] = defaultdict(int)
        self._hour_reset: dict[str, float] = {}

    async def acquire(self, endpoint: str = "market_data"):
        """Wait until a request is allowed for the given endpoint type."""
        async with self._locks[endpoint]:
            rps, rph = self.LIMITS.get(endpoint, (10, None))
            min_interval = 1.0 / rps

            # Per-second throttle
            elapsed = time.monotonic() - self._last_call[endpoint]
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)

            # Per-hour throttle
            if rph:
                now = time.monotonic()
                reset_time = self._hour_reset.get(endpoint, 0)
                if now - reset_time > 3600:
                    self._hour_counts[endpoint] = 0
                    self._hour_reset[endpoint] = now

                if self._hour_counts[endpoint] >= rph:
                    wait = 3600 - (now - reset_time)
                    if wait > 0:
                        raise RuntimeError(
                            f"Hourly rate limit hit for {endpoint} ({rph}/hr). "
                            f"Resets in {wait:.0f}s"
                        )

                self._hour_counts[endpoint] += 1

            self._last_call[endpoint] = time.monotonic()
