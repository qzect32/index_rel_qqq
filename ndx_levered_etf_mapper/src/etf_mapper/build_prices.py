from __future__ import annotations

from pathlib import Path
import os
import sqlite3
from typing import Optional, Literal

import pandas as pd

from .marketdata import SchwabPriceProvider
from .schwab import SchwabConfig
from .config import load_schwab_secrets


PriceProviderName = Literal["schwab"]


def _load_tickers_from_universe(universe_path: str | Path) -> list[str]:
    universe_path = Path(universe_path)
    if not universe_path.exists():
        raise FileNotFoundError(
            f"Universe file not found: {universe_path}. "
            "Provide a universe parquet/CSV with a 'ticker' column (this project is Schwab-only; no Polygon universe fetch)."
        )

    if universe_path.suffix.lower() in {".parquet"}:
        df = pd.read_parquet(universe_path)
    elif universe_path.suffix.lower() in {".csv"}:
        df = pd.read_csv(universe_path)
    else:
        raise ValueError(f"Unsupported universe file type: {universe_path}")

    if "ticker" not in df.columns:
        raise RuntimeError("Universe file must contain a 'ticker' column")

    tickers = (
        df["ticker"].astype(str).str.upper().str.strip().dropna().drop_duplicates().tolist()
    )
    return [t for t in tickers if t and t != "NAN"]


def refresh_prices(
    out_dir: str | Path,
    universe_path: str | Path,
    provider: PriceProviderName = "schwab",
    start: Optional[str] = None,
    end: Optional[str] = None,
    limit: int = 200,
) -> dict[str, Path]:
    """Fetch daily price history for a slice of the ETF universe.

    Notes:
      - Uses Schwab Market Data price history (OAuth required).
      - Designed to be rerun; failures are recorded.

    Outputs:
      - prices.sqlite (table: prices_daily)
      - prices_daily.parquet
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tickers = _load_tickers_from_universe(universe_path)
    if limit:
        tickers = tickers[: int(limit)]

    if provider != "schwab":
        raise ValueError(f"Unknown provider: {provider}")

    # Pull Schwab OAuth config from local secrets file first, env fallback.
    secrets = load_schwab_secrets(Path(out_dir))
    if secrets is None:
        # env fallback for legacy
        client_id = os.getenv("SCHWAB_CLIENT_ID", "")
        client_secret = os.getenv("SCHWAB_CLIENT_SECRET", "")
        redirect_uri = os.getenv("SCHWAB_REDIRECT_URI", "")
        token_path = os.getenv("SCHWAB_TOKEN_PATH", str(Path(out_dir) / "schwab_tokens.json"))
        if not (client_id and client_secret and redirect_uri):
            raise RuntimeError(
                "Missing Schwab OAuth config. Add data/schwab_secrets.local.json (recommended) or set SCHWAB_CLIENT_ID/SCHWAB_CLIENT_SECRET/SCHWAB_REDIRECT_URI."
            )
        secrets = type("S", (), {"client_id": client_id, "client_secret": client_secret, "redirect_uri": redirect_uri, "token_path": token_path})

    p = SchwabPriceProvider(
        SchwabConfig(
            client_id=secrets.client_id,
            client_secret=secrets.client_secret,
            redirect_uri=secrets.redirect_uri,
            token_path=secrets.token_path,
        )
    )

    all_rows: list[pd.DataFrame] = []
    failures: list[str] = []

    for i, t in enumerate(tickers, start=1):
        try:
            res = p.fetch_daily_bars(t, start=start, end=end)
            df = res.prices
            if df is None or df.empty:
                failures.append(t)
            else:
                all_rows.append(df)
        except Exception:
            failures.append(t)

        if i % 25 == 0:
            print(f"Fetched {i}/{len(tickers)} tickers (ok={len(all_rows)}, fail={len(failures)})")

    prices = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()

    # Persist
    db_path = out_dir / "prices.sqlite"
    with sqlite3.connect(db_path) as conn:
        prices.to_sql("prices_daily", conn, if_exists="replace", index=False)

        meta = pd.DataFrame(
            [
                {
                    "provider": provider,
                    "start": start,
                    "end": end,
                    "limit": limit,
                    "ok": len(all_rows),
                    "fail": len(failures),
                }
            ]
        )
        meta.to_sql("prices_meta", conn, if_exists="replace", index=False)

        if failures:
            pd.DataFrame({"ticker": failures}).to_sql(
                "prices_failures", conn, if_exists="replace", index=False
            )

    p_parquet = out_dir / "prices_daily.parquet"
    prices.to_parquet(p_parquet, index=False)

    return {"sqlite": db_path, "prices_daily": p_parquet}
