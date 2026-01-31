from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
import pdfplumber

PDF_URL = "https://www.direxion.com/uploads/Single-Stock-ETFs-List.pdf"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

PAT = re.compile(
    r"""
    (?P<ticker>\b[A-Z]{3,5}\b)
    \s+.*?
    \bDaily\s+(?P<underlying>[A-Z]{1,5})\s+
    (?P<bias>Bull|Bear)\s+
    (?P<lev>\d+(?:\.\d+)?)X\b
    .*?
    (?P<pct>-?\d{2,3})%
    """,
    re.IGNORECASE | re.VERBOSE,
)

def _pdf_text(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()

    with pdfplumber.open(BytesIO(r.content)) as pdf:
        parts = []
        for page in pdf.pages:
            t = page.extract_text() or ""
            parts.append(t)
        return "\n".join(parts)

def debug_dump_pdf(out_path: str | Path = "direxion_pdf_dump.txt") -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = _pdf_text(PDF_URL)
    out_path.write_text(text, encoding="utf-8")
    print(f"Wrote {out_path} ({len(text):,} chars)")

def fetch_direxion_single_stock() -> pd.DataFrame:
    text = _pdf_text(PDF_URL)
    norm = re.sub(r"\s+", " ", text)

    rows = []
    for m in PAT.finditer(norm):
        etf_ticker = m.group("ticker").upper()
        underlying = m.group("underlying").upper()
        bias = m.group("bias").lower()
        pct = float(m.group("pct"))
        daily_target = pct / 100.0

        relationship = "leveraged_long" if bias == "bull" else "inverse"
        direction = "long" if daily_target >= 0 else "short"
        leverage_multiple = abs(daily_target) if daily_target else float(m.group("lev"))

        strategy_group = (
            "leveraged" if (leverage_multiple and leverage_multiple > 1)
            else ("inverse" if direction == "short" else "plain")
        )

        rows.append(
            {
                "etf_ticker": etf_ticker,
                "etf_name": f"Direxion Daily {underlying} {bias.title()} {m.group('lev')}X",
                "underlying_symbol": underlying,
                "daily_target": daily_target,
                "relationship": relationship,
                "issuer": "Direxion",
                "source_url": PDF_URL,
                "direction": direction,
                "leverage_multiple": leverage_multiple,
                "strategy_group": strategy_group,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "etf_ticker",
                "etf_name",
                "underlying_symbol",
                "daily_target",
                "relationship",
                "issuer",
                "source_url",
                "direction",
                "leverage_multiple",
                "strategy_group",
            ]
        )

    return df.drop_duplicates(subset=["etf_ticker"]).reset_index(drop=True)
