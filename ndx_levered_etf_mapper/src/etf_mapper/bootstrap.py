from __future__ import annotations

from pathlib import Path
from typing import Optional

from .build_universe import refresh_etf_universe
from .build_prices import refresh_prices


def bootstrap(
    out_dir: str | Path,
    universe_provider: str = "polygon",
    price_provider: str = "yahoo",
    start: Optional[str] = "2024-01-01",
    limit: int = 200,
) -> dict[str, Path]:
    """One-command bootstrap:

    - fetch US ETF universe
    - fetch daily prices for the first N tickers

    This is meant to get the UI running quickly.
    """
    out_dir = Path(out_dir)

    u = refresh_etf_universe(out_dir, provider=universe_provider)  # type: ignore[arg-type]
    universe_path = u["etf_universe"]

    p = refresh_prices(
        out_dir,
        universe_path=universe_path,
        provider=price_provider,  # type: ignore[arg-type]
        start=start,
        end=None,
        limit=limit,
    )

    return {**u, **p}
