from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class OAuthTokenSet:
    access_token: str
    refresh_token: Optional[str]
    expires_at: float  # unix seconds
    token_type: str = "Bearer"

    @property
    def is_expired(self) -> bool:
        # 60s safety margin
        return time.time() >= (self.expires_at - 60)


class TokenStore:
    """Simple JSON token persistence.

    File is expected to be gitignored (e.g. data/schwab_tokens.json).
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> Optional[OAuthTokenSet]:
        if not self.path.exists():
            return None
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not data.get("access_token"):
            return None
        return OAuthTokenSet(
            access_token=str(data["access_token"]),
            refresh_token=data.get("refresh_token"),
            expires_at=float(data.get("expires_at", 0)),
            token_type=str(data.get("token_type", "Bearer")),
        )

    def save_from_token_response(self, token_json: dict[str, Any]) -> OAuthTokenSet:
        # token_json typically contains: access_token, expires_in, refresh_token, token_type
        access = str(token_json["access_token"])
        refresh = token_json.get("refresh_token")
        expires_in = float(token_json.get("expires_in", 0))
        expires_at = time.time() + expires_in
        token_type = str(token_json.get("token_type", "Bearer"))

        ts = OAuthTokenSet(access_token=access, refresh_token=refresh, expires_at=expires_at, token_type=token_type)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "access_token": ts.access_token,
                    "refresh_token": ts.refresh_token,
                    "expires_at": ts.expires_at,
                    "token_type": ts.token_type,
                    "saved_at": time.time(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return ts
