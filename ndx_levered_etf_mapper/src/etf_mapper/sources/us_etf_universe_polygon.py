from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import os
import time
import requests
import pandas as pd


@dataclass(frozen=True)
class PolygonUniverseConfig:
    api_key: str
    base_url: str = "https://api.polygon.io"
    page_limit: int = 1000
    sleep_s: float = 0.0


def _iter_polygon_pages(url: str, params: dict, session: requests.Session) -> Iterator[dict]:
    """Iterate a Polygon paginated endpoint that returns {results, next_url}.

    IMPORTANT: Polygon's next_url does NOT reliably include the apiKey.
    We always re-attach apiKey on subsequent requests.
    """
    api_key = params.get("apiKey")
    if not api_key:
        raise RuntimeError("Polygon apiKey missing from request params")

    next_url: Optional[str] = url
    next_params: Optional[dict] = params

    while next_url:
        r = session.get(next_url, params=next_params, timeout=60)
        r.raise_for_status()
        payload = r.json()
        yield payload

        next_url = payload.get("next_url")
        # next_url may omit apiKey; keep it in params for subsequent requests.
        next_params = {"apiKey": api_key} if next_url else None


def fetch_us_etf_universe_from_polygon(cfg: PolygonUniverseConfig) -> pd.DataFrame:
    """Fetch a master list of US-listed ETFs from Polygon.

    Returns a dataframe with at least:
      - ticker
      - name
      - issuer
      - exchange
      - primary_exchange
      - active
      - cik
      - composite_figi
      - share_class_figi
      - locale
      - market
      - type
      - source

    Notes:
      - Polygon's /v3/reference/tickers supports pagination.
      - 'type=ETF' is key to getting ETFs.
    """
    if not cfg.api_key:
        raise RuntimeError("POLYGON_API_KEY is required to fetch the ETF universe from Polygon")

    url = f"{cfg.base_url}/v3/reference/tickers"
    params = {
        "market": "stocks",
        "type": "ETF",
        "active": "true",
        "limit": str(cfg.page_limit),
        "apiKey": cfg.api_key,
    }

    rows: list[dict] = []
    with requests.Session() as s:
        for page in _iter_polygon_pages(url, params=params, session=s):
            results = page.get("results") or []
            rows.extend(results)
            if cfg.sleep_s:
                time.sleep(cfg.sleep_s)

    if not rows:
        return pd.DataFrame(
            columns=[
                "ticker",
                "name",
                "issuer",
                "exchange",
                "primary_exchange",
                "active",
                "cik",
                "composite_figi",
                "share_class_figi",
                "locale",
                "market",
                "type",
                "source",
            ]
        )

    df = pd.DataFrame(rows)

    # Normalize key columns
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    if "name" in df.columns:
        df["name"] = df["name"].astype(str).str.strip()

    # Polygon fields: primary_exchange, cik, composite_figi, share_class_figi, locale, market, type, active
    # Derive an 'issuer' best-effort (Polygon doesn't always provide it).
    if "brand" in df.columns and df["brand"].notna().any():
        # brand can be an object; try common shapes
        def _brand_name(x):
            if isinstance(x, dict):
                return x.get("name") or x.get("company")
            return None

        df["issuer"] = df.get("brand").apply(_brand_name)
    else:
        df["issuer"] = None

    df["source"] = "polygon:/v3/reference/tickers?type=ETF"

    # Keep a stable, predictable column order for downstream.
    keep = [
        "ticker",
        "name",
        "issuer",
        "exchange",
        "primary_exchange",
        "active",
        "cik",
        "composite_figi",
        "share_class_figi",
        "locale",
        "market",
        "type",
        "source",
    ]
    for c in keep:
        if c not in df.columns:
            df[c] = None

    return df[keep]


def fetch_us_etf_universe_polygon_env() -> pd.DataFrame:
    return fetch_us_etf_universe_from_polygon(
        PolygonUniverseConfig(api_key=os.getenv("POLYGON_API_KEY", ""))
    )
