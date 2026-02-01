from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Iterable, Optional, Literal

import pandas as pd

from .marketdata import YahooPriceProvider, StooqPriceProvider


PriceProviderName = Literal["yahoo", "stooq"]


def _load_tickers_from_universe(universe_path: str | Path) -> list[str]:
    universe_path = Path(universe_path)
    if not universe_path.exists():
        raise FileNotFoundError(
            f"Universe file not found: {universe_path}. "
            "Run: python -m etf_mapper.cli universe --out data --provider polygon"
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
    provider: PriceProviderName = "yahoo",
    start: Optional[str] = None,
    end: Optional[str] = None,
    limit: int = 200,
) -> dict[str, Path]:
    """Fetch daily price history for a slice of the ETF universe.

    Notes:
      - This is a bootstrap step while you wait for Schwab/TOS.
      - Free sources can be flaky; the pipeline is designed to be rerun.

    Outputs:
      - prices.sqlite (table: prices_daily)
      - prices_daily.parquet
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tickers = _load_tickers_from_universe(universe_path)
    if limit:
        tickers = tickers[: int(limit)]

    if provider == "yahoo":
        p = YahooPriceProvider()
    elif provider == "stooq":
        p = StooqPriceProvider()
    else:
        raise ValueError(f"Unknown provider: {provider}")

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
