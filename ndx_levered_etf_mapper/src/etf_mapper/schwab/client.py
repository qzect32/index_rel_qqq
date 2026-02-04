from __future__ import annotations

import base64
import os
import urllib.parse
from dataclasses import dataclass
from typing import Any, Optional

import requests

from .tokens import OAuthTokenSet, TokenStore
from .dividends import DividendInfo, extract_ex_dividend


DEFAULT_BASE_URL = os.getenv("SCHWAB_API_BASE_URL", "https://api.schwabapi.com")
DEFAULT_AUTH_URL = os.getenv("SCHWAB_OAUTH_AUTHORIZE_URL", f"{DEFAULT_BASE_URL}/v1/oauth/authorize")
DEFAULT_TOKEN_URL = os.getenv("SCHWAB_OAUTH_TOKEN_URL", f"{DEFAULT_BASE_URL}/v1/oauth/token")


@dataclass(frozen=True)
class SchwabConfig:
    client_id: str
    client_secret: str
    redirect_uri: str
    token_path: str = "data/schwab_tokens.json"
    base_url: str = DEFAULT_BASE_URL
    auth_url: str = DEFAULT_AUTH_URL
    token_url: str = DEFAULT_TOKEN_URL


class SchwabAPI:
    """Thin Schwab Developer Portal API client.

    Notes:
      - Uses OAuth2 Authorization Code grant.
      - Persists tokens locally in a gitignored JSON file.
      - Avoids any trading side-effects by default; order endpoints are exposed but not wired into UI here.
    """

    def __init__(self, cfg: SchwabConfig):
        self.cfg = cfg
        self.tokens = TokenStore(cfg.token_path)
        # Keep a session for connection pooling (noticeably reduces latency on repeated calls).
        self.session = requests.Session()

    # ---------- OAuth ----------
    def build_authorize_url(self, state: str, scope: str = "readonly") -> str:
        # Scope naming depends on Schwab; allow override.
        q = {
            "client_id": self.cfg.client_id,
            "redirect_uri": self.cfg.redirect_uri,
            "response_type": "code",
            "state": state,
            "scope": scope,
        }
        return f"{self.cfg.auth_url}?{urllib.parse.urlencode(q)}"

    def _basic_auth_header(self) -> str:
        raw = f"{self.cfg.client_id}:{self.cfg.client_secret}".encode("utf-8")
        return "Basic " + base64.b64encode(raw).decode("ascii")

    def exchange_code(self, code: str) -> OAuthTokenSet:
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.cfg.redirect_uri,
        }
        headers = {
            "Authorization": self._basic_auth_header(),
            "Content-Type": "application/x-www-form-urlencoded",
        }
        r = requests.post(self.cfg.token_url, data=data, headers=headers, timeout=30)
        if not r.ok:
            raise RuntimeError(f"Token exchange failed: {r.status_code} {r.reason} :: {r.text}")
        return self.tokens.save_from_token_response(r.json())

    def refresh_access_token(self) -> OAuthTokenSet:
        ts = self.tokens.load()
        if not ts or not ts.refresh_token:
            raise RuntimeError("No refresh token available. Re-authorize.")
        data = {
            "grant_type": "refresh_token",
            "refresh_token": ts.refresh_token,
        }
        headers = {
            "Authorization": self._basic_auth_header(),
            "Content-Type": "application/x-www-form-urlencoded",
        }
        r = requests.post(self.cfg.token_url, data=data, headers=headers, timeout=30)
        if not r.ok:
            raise RuntimeError(f"Token refresh failed: {r.status_code} {r.reason} :: {r.text}")
        return self.tokens.save_from_token_response(r.json())

    def get_access_token(self) -> str:
        ts = self.tokens.load()
        if not ts:
            raise RuntimeError("Not authenticated. Connect Schwab OAuth first.")
        if ts.is_expired:
            ts = self.refresh_access_token()
        return ts.access_token

    # ---------- HTTP helpers ----------
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.get_access_token()}",
            "Accept": "application/json",
        }

    def _get(self, path: str, params: Optional[dict[str, Any]] = None) -> Any:
        url = self.cfg.base_url.rstrip("/") + path
        r = self.session.get(url, params=params, headers=self._headers(), timeout=30)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, json_body: dict[str, Any]) -> Any:
        url = self.cfg.base_url.rstrip("/") + path
        r = self.session.post(url, json=json_body, headers=self._headers(), timeout=30)
        r.raise_for_status()
        if r.text:
            return r.json()
        return None

    # ---------- Market data ----------
    def quotes(self, symbols: list[str]) -> Any:
        # Typical endpoint: /marketdata/v1/quotes?symbols=SPY,QQQ
        return self._get("/marketdata/v1/quotes", params={"symbols": ",".join(symbols)})

    def price_history(
        self,
        symbol: str,
        *,
        period_type: str = "day",
        period: int = 3,
        frequency_type: str = "minute",
        frequency: int = 1,
        start_date_ms: Optional[int] = None,
        end_date_ms: Optional[int] = None,
        need_extended_hours_data: bool = True,
    ) -> Any:
        params: dict[str, Any] = {
            "symbol": symbol,
            "periodType": period_type,
            "period": period,
            "frequencyType": frequency_type,
            "frequency": frequency,
            "needExtendedHoursData": "true" if need_extended_hours_data else "false",
        }
        if start_date_ms is not None:
            params["startDate"] = int(start_date_ms)
        if end_date_ms is not None:
            params["endDate"] = int(end_date_ms)
        return self._get("/marketdata/v1/pricehistory", params=params)

    def option_chain(self, symbol: str, contract_type: str = "ALL") -> Any:
        # Typical endpoint: /marketdata/v1/chains?symbol=SPY&contractType=ALL
        return self._get(
            "/marketdata/v1/chains",
            params={
                "symbol": symbol,
                "contractType": contract_type,
            },
        )

    def dividends(self, symbol: str) -> DividendInfo:
        """Best-effort dividend metadata.

        Schwab has multiple market data endpoints; schemas vary. We try an instruments endpoint if available.
        If this fails, we return an empty DividendInfo.
        """
        try:
            js = self._get(
                "/marketdata/v1/instruments",
                params={"symbol": symbol, "projection": "fundamental"},
            )
        except Exception:
            return DividendInfo()

        # Try keyed-by-symbol shape
        if isinstance(js, dict):
            rec = js.get(symbol) or js.get(symbol.upper())
            if isinstance(rec, dict):
                return extract_ex_dividend(rec)
            return extract_ex_dividend(js)
        return DividendInfo()

    # ---------- Trader (orders/accounts) ----------
    def account_numbers(self) -> Any:
        return self._get("/trader/v1/accounts/accountNumbers")

    def place_order(self, account_hash: str, order: dict[str, Any]) -> Any:
        # WARNING: this can place real orders if your app is entitled. Keep UI guarded.
        return self._post(f"/trader/v1/accounts/{account_hash}/orders", json_body=order)
