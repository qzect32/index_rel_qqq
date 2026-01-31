from __future__ import annotations

from io import StringIO
import pandas as pd
import requests

WIKI_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

def fetch_nasdaq100() -> pd.DataFrame:
    """Return a DataFrame with columns: symbol, company, sector, industry/subindustry.

    Uses Wikipedia's components table; fetches HTML via requests to avoid 403.
    """
    resp = requests.get(WIKI_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    # Parse tables from HTML text
    tables = pd.read_html(StringIO(resp.text))

    target = None
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any("ticker" in c for c in cols) and any("company" in c or "security" in c for c in cols):
            target = t
            break

    if target is None:
        raise RuntimeError("Could not find Nasdaq-100 components table on Wikipedia")

    rename = {}
    for c in target.columns:
        lc = str(c).lower()
        if "ticker" in lc:
            rename[c] = "symbol"
        elif "company" in lc or "security" in lc:
            rename[c] = "company"
        elif "sector" in lc:
            rename[c] = "sector"
        elif "industry" in lc and "sub" not in lc:
            rename[c] = "industry"
        elif "sub-industry" in lc or "subindustry" in lc or "subsector" in lc:
            rename[c] = "subindustry"

    df = target.rename(columns=rename).copy()
    keep = [c for c in ["symbol", "company", "sector", "industry", "subindustry"] if c in df.columns]
    df = df[keep]

    df["symbol"] = df["symbol"].astype(str).str.strip()
    df["company"] = df["company"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["symbol"]).reset_index(drop=True)
    return df
