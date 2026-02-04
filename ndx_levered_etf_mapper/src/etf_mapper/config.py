from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class SchwabSecrets:
    client_id: str
    client_secret: str
    redirect_uri: str
    token_path: str
    oauth_scope: str = "readonly"


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        if not path.exists():
            return None
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def load_schwab_secrets(data_dir: Path) -> Optional[SchwabSecrets]:
    """Load Schwab secrets from (in order):

    1) data_dir / 'schwab_secrets.local.json'
    2) repo root / 'schwab_secrets.local.json' (cwd)
    3) environment variables

    This is intentionally local-first and gitignored.
    """

    candidates = [
        (data_dir / "schwab_secrets.local.json"),
        Path("schwab_secrets.local.json"),
    ]

    for p in candidates:
        obj = _read_json(p)
        if not obj:
            continue

        cid = str(obj.get("SCHWAB_CLIENT_ID") or obj.get("client_id") or "").strip()
        csec = str(obj.get("SCHWAB_CLIENT_SECRET") or obj.get("client_secret") or "").strip()
        ruri = str(obj.get("SCHWAB_REDIRECT_URI") or obj.get("redirect_uri") or "").strip()
        tpath = str(obj.get("SCHWAB_TOKEN_PATH") or obj.get("token_path") or (data_dir / "schwab_tokens.json")).strip()
        scope = str(obj.get("SCHWAB_OAUTH_SCOPE") or obj.get("oauth_scope") or "readonly").strip()

        if cid and csec and ruri:
            return SchwabSecrets(
                client_id=cid,
                client_secret=csec,
                redirect_uri=ruri,
                token_path=tpath,
                oauth_scope=scope or "readonly",
            )

    # Env fallback
    cid = os.getenv("SCHWAB_CLIENT_ID", "").strip()
    csec = os.getenv("SCHWAB_CLIENT_SECRET", "").strip()
    ruri = os.getenv("SCHWAB_REDIRECT_URI", "").strip()
    tpath = os.getenv("SCHWAB_TOKEN_PATH", str(data_dir / "schwab_tokens.json")).strip()
    scope = os.getenv("SCHWAB_OAUTH_SCOPE", "readonly").strip()

    if cid and csec and ruri:
        return SchwabSecrets(
            client_id=cid,
            client_secret=csec,
            redirect_uri=ruri,
            token_path=tpath,
            oauth_scope=scope or "readonly",
        )

    return None
