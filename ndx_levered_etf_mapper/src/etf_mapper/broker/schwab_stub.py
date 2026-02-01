from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SchwabConfig:
    # Placeholder for future:
    # client_id, client_secret, redirect_uri, tokens, account_hash, etc.
    enabled: bool = False
    client_id: Optional[str] = None


class SchwabBrokerStub:
    """Scaffold for Schwab/TOS integration.

    This intentionally does nothing today; it's here so the UI can expose a toggle
    and we can fill it in later without refactoring.
    """

    name = "schwab_stub"

    def __init__(self, cfg: SchwabConfig):
        self.cfg = cfg

    def is_configured(self) -> bool:
        return bool(self.cfg.enabled and self.cfg.client_id)

    def place_order(self, *args, **kwargs):
        raise NotImplementedError("Schwab/TOS integration not implemented yet")
