from __future__ import annotations

import re
import pandas as pd
import requests
from bs4 import BeautifulSoup

URL = "https://graniteshares.com/institutional/us/en-us/etfs/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

UNDERLYING_RE = re.compile(r"\(([A-Z]{1,5})\)")
LEV_RE = re.compile(r"(\d+(?:\.\d+)?)x\s+(long|short)", re.IGNORECASE)

def _infer_target_and_rel(text: str) -> tuple[float | None, str]:
    m = LEV_RE.search(text)
    if not m:
        return None, "unknown"
    mult = float(m.group(1))
    side = m.group(2).lower()
    if side == "long":
        return mult, "leveraged_long"
    return -mult, "inverse"

def fetch_graniteshares_single_stock() -> pd.DataFrame:
    resp = requests.get(URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    rows = []
    seen = set()

    # GraniteShares pages usually have ETF detail links containing "/etfs/" and a ticker slug.
    # We grab candidate links where the visible text looks like a ticker.
    for a in soup.find_all("a", href=True):
        txt = (a.get_text(" ", strip=True) or "").strip()
        if not re.fullmatch(r"[A-Z]{3,5}", txt):
            continue
        ticker = txt
        if ticker in seen:
            continue

        href = a["href"]
        # keep only ETF-ish links
        if "etfs" not in href.lower():
            continue

        # Try to collect context text from the nearest container
        container = a.find_parent(["div", "li", "section", "article"]) or a.parent
        context = container.get_text(" ", strip=True)
        context = re.sub(r"\s+", " ", context)

        # Must mention Daily ETF or GraniteShares + (UNDERLYING)
        if "daily etf" not in context.lower() and "graniteshares" not in context.lower():
            continue

        um = UNDERLYING_RE.search(context)
        if not um:
            continue
        underlying = um.group(1)

        daily_target, relationship = _infer_target_and_rel(context)

        # Best-effort name extraction
        # If the container includes “ETF”, keep the chunk containing it.
        name = context
        mname = re.search(r"(GraniteShares.*?ETF)", context)
        if mname:
            name = mname.group(1).strip()

        rows.append(
            {
                "etf_ticker": ticker,
                "etf_name": name,
                "underlying_symbol": underlying,
                "daily_target": daily_target,
                "relationship": relationship,
                "issuer": "GraniteShares",
                "source_url": URL,
            }
        )
        seen.add(ticker)

    return pd.DataFrame(rows).drop_duplicates(subset=["etf_ticker"]).reset_index(drop=True)
