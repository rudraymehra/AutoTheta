import logging
import pyotp
from SmartApi import SmartConnect
from config.settings import (
    ANGEL_API_KEY,
    ANGEL_CLIENT_ID,
    ANGEL_PASSWORD,
    ANGEL_TOTP_SECRET,
)

log = logging.getLogger("autotheta.auth")


class AngelSession:
    """Manages Angel One SmartAPI authentication and token lifecycle."""

    def __init__(self):
        self.api = SmartConnect(ANGEL_API_KEY)
        self.auth_token: str | None = None
        self.refresh_token: str | None = None
        self.feed_token: str | None = None

    def login(self) -> bool:
        """Authenticate with Angel One using 4-factor flow."""
        try:
            totp = pyotp.TOTP(ANGEL_TOTP_SECRET).now()
            session = self.api.generateSession(ANGEL_CLIENT_ID, ANGEL_PASSWORD, totp)

            if not session or not session.get("status"):
                log.error("Login failed: %s", session)
                return False

            self.auth_token = session["data"]["jwtToken"]
            self.refresh_token = session["data"]["refreshToken"]
            self.feed_token = self.api.getfeedToken()
            log.info("Login successful for client %s", ANGEL_CLIENT_ID)
            return True

        except Exception:
            log.exception("Login error")
            return False

    def refresh(self) -> bool:
        """Refresh session token mid-day (sessions expire at midnight)."""
        if not self.refresh_token:
            log.warning("No refresh token available, performing full login")
            return self.login()
        try:
            token_data = self.api.generateToken(self.refresh_token)
            if token_data and token_data.get("status"):
                self.auth_token = token_data["data"]["jwtToken"]
                log.info("Token refreshed successfully")
                return True
            log.warning("Token refresh failed, performing full login")
            return self.login()
        except Exception:
            log.exception("Token refresh error")
            return self.login()

    def logout(self):
        """Terminate the session."""
        try:
            self.api.terminateSession(ANGEL_CLIENT_ID)
            log.info("Session terminated")
        except Exception:
            log.exception("Logout error")
