from __future__ import annotations

"""Schwab/TOS integration scaffold.

This is intentionally non-functional until Schwab Developer Portal credentials are available.

Design goals:
- Keep secrets out of git.
- Provide a stable interface that the Streamlit UI can call.
- Provide broker-grade quotes/options/orders via Schwab Developer Portal.

Planned capabilities (incremental):
1) OAuth connect + token persistence
2) Quotes
3) Options chains
4) Positions / balances
5) Order preview + placement

Nothing in this module should place real trades yet.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol


@dataclass(frozen=True)
class SchwabOAuthConfig:
    enabled: bool = False

    # Provided by Schwab Developer Portal (Trader API app)
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    redirect_uri: Optional[str] = None

    # Where we persist tokens locally (gitignored)
    token_path: str = "data/schwab_tokens.json"


class SchwabBroker(Protocol):
    """Interface the UI can depend on."""

    def is_enabled(self) -> bool: ...

    def is_configured(self) -> bool: ...

    def auth_status(self) -> dict: ...


class SchwabBrokerScaffold:
    """Non-functional broker scaffold.

    Provides a consistent surface area while we wait for portal access.
    """

    name = "schwab_scaffold"

    def __init__(self, cfg: SchwabOAuthConfig):
        self.cfg = cfg

    def is_enabled(self) -> bool:
        return bool(self.cfg.enabled)

    def is_configured(self) -> bool:
        # We consider configured when the user has client_id + redirect_uri.
        return bool(self.cfg.enabled and self.cfg.client_id and self.cfg.redirect_uri)

    def auth_status(self) -> dict:
        p = Path(self.cfg.token_path)
        return {
            "enabled": bool(self.cfg.enabled),
            "client_id_set": bool(self.cfg.client_id),
            "redirect_uri_set": bool(self.cfg.redirect_uri),
            "token_file_present": p.exists(),
            "token_file": str(p),
            "note": "Scaffold only; OAuth connect + API calls not implemented yet.",
        }

    # Future:
    # - build_auth_url()
    # - exchange_code_for_tokens()
    # - refresh_tokens_if_needed()
    # - get_quotes(symbols)
    # - get_option_chain(symbol)
    # - get_positions()
    # - place_order()
