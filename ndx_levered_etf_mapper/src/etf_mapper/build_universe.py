from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Literal

import pandas as pd

from .sources.us_etf_universe_polygon import fetch_us_etf_universe_polygon_env


UniverseProvider = Literal["polygon"]


def refresh_etf_universe(out_dir: str | Path, provider: UniverseProvider = "polygon") -> dict[str, Path]:
    """Build a master list of US-listed ETFs.

    This is intentionally separate from the Nasdaq-100 exposure graph builder.

    Outputs:
      - etf_universe.parquet
      - etf_universe.sqlite (table: etf_universe)

    Provider notes:
      - polygon: requires POLYGON_API_KEY
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if provider == "polygon":
        df = fetch_us_etf_universe_polygon_env()
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Minimal serving-friendly normalization
    df = df.copy()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    p_univ = out_dir / "etf_universe.parquet"
    df.to_parquet(p_univ, index=False)

    db_path = out_dir / "etf_universe.sqlite"
    with sqlite3.connect(db_path) as conn:
        df.to_sql("etf_universe", conn, if_exists="replace", index=False)

    return {"etf_universe": p_univ, "sqlite": db_path}
