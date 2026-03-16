import json
import logging
from datetime import date, datetime

import pandas as pd
import requests

from config.settings import INSTRUMENT_CACHE_PATH, INSTRUMENT_MASTER_URL

log = logging.getLogger("autotheta.instruments")


class InstrumentMaster:
    """Downloads, caches, and queries the Angel One instrument master file."""

    def __init__(self):
        self.df: pd.DataFrame = pd.DataFrame()
        self.nifty_opts: pd.DataFrame = pd.DataFrame()
        self._loaded_date: date | None = None

    def load(self, force_download: bool = False) -> bool:
        """Load instrument master. Downloads fresh copy if cache is stale (not today's)."""
        today = date.today()

        if not force_download and self._is_cache_fresh(today):
            log.info("Loading instruments from cache")
            return self._load_from_cache()

        log.info("Downloading instrument master (~80MB)...")
        return self._download_and_cache(today)

    def _is_cache_fresh(self, today: date) -> bool:
        if not INSTRUMENT_CACHE_PATH.exists():
            return False
        cache_mtime = datetime.fromtimestamp(INSTRUMENT_CACHE_PATH.stat().st_mtime).date()
        return cache_mtime == today

    def _load_from_cache(self) -> bool:
        try:
            with open(INSTRUMENT_CACHE_PATH, "r") as f:
                data = json.load(f)
            self._process(data)
            return True
        except Exception:
            log.exception("Cache load failed, will re-download")
            return self._download_and_cache(date.today())

    def _download_and_cache(self, today: date) -> bool:
        try:
            resp = requests.get(INSTRUMENT_MASTER_URL, timeout=120)
            resp.raise_for_status()
            data = resp.json()

            # Cache to disk
            INSTRUMENT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(INSTRUMENT_CACHE_PATH, "w") as f:
                json.dump(data, f)

            self._process(data)
            self._loaded_date = today
            log.info("Instrument master loaded: %d total records", len(self.df))
            return True
        except Exception:
            log.exception("Failed to download instrument master")
            return False

    def _process(self, data: list[dict]):
        """Parse raw instrument data into filtered DataFrames."""
        self.df = pd.DataFrame(data)
        self.df["expiry_dt"] = pd.to_datetime(
            self.df["expiry"], format="mixed", dayfirst=True
        ).dt.date
        self.df["actual_strike"] = self.df["strike"].astype(float) / 100  # Paise → Rupees

        self.nifty_opts = self.df[
            (self.df["name"] == "NIFTY")
            & (self.df["instrumenttype"] == "OPTIDX")
            & (self.df["exch_seg"] == "NFO")
        ].copy()
        log.info("Nifty options filtered: %d contracts", len(self.nifty_opts))

    def get_nearest_expiry(self, from_date: date | None = None) -> date | None:
        """Get the nearest expiry date on or after from_date."""
        if self.nifty_opts.empty:
            return None
        ref = from_date or date.today()
        future = self.nifty_opts[self.nifty_opts["expiry_dt"] >= ref]
        if future.empty:
            return None
        return future["expiry_dt"].min()

    def get_expiry_chain(self, expiry_date: date) -> pd.DataFrame:
        """Get all Nifty option contracts for a specific expiry date."""
        return self.nifty_opts[self.nifty_opts["expiry_dt"] == expiry_date]

    def lookup(self, strike: float, option_type: str, expiry_date: date) -> dict | None:
        """Look up a specific Nifty option contract.

        Args:
            strike: Strike price in rupees (e.g., 24400)
            option_type: 'CE' or 'PE'
            expiry_date: Expiry date

        Returns:
            dict with 'symbol' and 'token', or None if not found.
        """
        chain = self.get_expiry_chain(expiry_date)
        suffix = option_type.upper()
        matches = chain[
            (chain["actual_strike"] == strike)
            & (chain["symbol"].str.endswith(suffix))
        ]
        if matches.empty:
            log.warning("No instrument found: strike=%s type=%s expiry=%s", strike, suffix, expiry_date)
            return None
        row = matches.iloc[0]
        return {"symbol": row["symbol"], "token": row["token"]}
