from __future__ import annotations

import json
from pathlib import Path

from etf_mapper.config import load_schwab_secrets


def test_load_schwab_secrets_missing(tmp_path: Path):
    assert load_schwab_secrets(tmp_path) is None


def test_load_schwab_secrets_from_file(tmp_path: Path):
    p = tmp_path / "schwab_secrets.local.json"
    p.write_text(
        json.dumps(
            {
                "SCHWAB_CLIENT_ID": "cid",
                "SCHWAB_CLIENT_SECRET": "sec",
                "SCHWAB_REDIRECT_URI": "http://127.0.0.1:8501",
                "SCHWAB_TOKEN_PATH": str(tmp_path / "tokens.json"),
                "SCHWAB_OAUTH_SCOPE": "readonly",
            }
        ),
        encoding="utf-8",
    )

    s = load_schwab_secrets(tmp_path)
    assert s is not None
    assert s.client_id == "cid"
    assert s.client_secret == "sec"
    assert s.redirect_uri.startswith("http")
    assert s.token_path.endswith("tokens.json")
