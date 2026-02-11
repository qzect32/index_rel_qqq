"""Refresh Schwab OAuth access token using existing secrets+token files.

Usage (PowerShell):
  python scripts/bench_schwab_refresh.py

This is a pre-step for cross-language benchmarking so other languages can
reuse the freshly-written access_token in data/schwab_tokens.json.
"""

from __future__ import annotations

from pathlib import Path

from etf_mapper.config import load_schwab_secrets
from etf_mapper.schwab.client import SchwabAPI, SchwabConfig


def main() -> int:
    data_dir = Path("data")
    secrets = load_schwab_secrets(data_dir)
    if secrets is None:
        raise SystemExit("Missing Schwab secrets. Expected data/schwab_secrets.local.json")

    tok_path = Path(secrets.token_path)
    if not tok_path.exists():
        raise SystemExit(f"Missing token file: {tok_path}")

    api = SchwabAPI(
        SchwabConfig(
            client_id=secrets.client_id,
            client_secret=secrets.client_secret,
            redirect_uri=secrets.redirect_uri,
            token_path=secrets.token_path,
        )
    )

    api.refresh_access_token()
    print(f"Refreshed token OK -> {tok_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
