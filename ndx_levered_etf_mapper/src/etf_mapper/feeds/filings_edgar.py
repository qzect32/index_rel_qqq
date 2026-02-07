from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
import time
from typing import Optional

import pandas as pd
import requests

from .base import FilingsFeed, FeedStatus


def _normalize_symbol(sym: str) -> str:
    return str(sym or "").upper().strip()


def _data_dir(p: Path | str) -> Path:
    return Path(p).resolve()


def _filings_root(data_dir: Path) -> Path:
    return (data_dir / "filings").resolve()


def _user_agent() -> str:
    # SEC requires a descriptive UA with contact. Keep generic but descriptive.
    # You can override later via config.
    return "MarketHub/0.1 (local research; contact: user)"


def _get_json(url: str, *, headers: dict, timeout: float = 15.0) -> dict:
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    obj = r.json()
    return obj if isinstance(obj, dict) else {}


def _get_text(url: str, *, headers: dict, timeout: float = 30.0) -> str:
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    r.encoding = r.encoding or "utf-8"
    return r.text


def _cik10(cik: str | int) -> str:
    s = str(cik).strip()
    s = re.sub(r"\D", "", s)
    return s.zfill(10)


@dataclass
class EdgarConfig:
    max_items: int = 50
    forms: tuple[str, ...] = ("10-Q", "10-K", "8-K")


class EdgarFilingsFeed(FilingsFeed):
    """SEC EDGAR filings feed.

    Provides:
      - fetch_filings(symbol) -> list recent filings metadata
      - download helpers are implemented in module-level functions below

    This does not require any third-party API keys.
    """

    def __init__(self, *, data_dir: Path, cfg: Optional[EdgarConfig] = None) -> None:
        self.data_dir = _data_dir(data_dir)
        self.cfg = cfg or EdgarConfig()
        self._last_ok = False
        self._last_detail = "not fetched"

    def status(self) -> FeedStatus:
        return FeedStatus(ok=bool(self._last_ok), name=self.name, detail=str(self._last_detail))

    def fetch_filings(self, symbol: str) -> pd.DataFrame:
        sym = _normalize_symbol(symbol)
        if not sym:
            self._last_ok = False
            self._last_detail = "empty symbol"
            return pd.DataFrame(columns=["form", "filed_at", "url", "source"]) 

        headers = {"User-Agent": _user_agent(), "Accept-Encoding": "gzip, deflate"}

        cik = lookup_cik(sym, headers=headers)
        if not cik:
            self._last_ok = False
            self._last_detail = f"CIK lookup failed for {sym}"
            return pd.DataFrame(columns=["form", "filed_at", "url", "source"]) 

        sub = _get_json(f"https://data.sec.gov/submissions/CIK{_cik10(cik)}.json", headers=headers)
        recent = (sub.get("filings") or {}).get("recent") if isinstance(sub.get("filings"), dict) else None
        if not isinstance(recent, dict):
            self._last_ok = False
            self._last_detail = "no recent filings"
            return pd.DataFrame(columns=["form", "filed_at", "url", "source"]) 

        df = pd.DataFrame(recent)
        if df.empty:
            self._last_ok = False
            self._last_detail = "no filings"
            return pd.DataFrame(columns=["form", "filed_at", "url", "source"]) 

        # Normalize a few columns
        if "form" not in df.columns:
            df["form"] = None
        if "filingDate" in df.columns and "filed_at" not in df.columns:
            df["filed_at"] = df["filingDate"]
        elif "filed_at" not in df.columns:
            df["filed_at"] = None

        # Build a document URL (index) when accession + primaryDocument exist
        if "accessionNumber" in df.columns and "primaryDocument" in df.columns:
            def _doc_url(row) -> str:
                try:
                    acc = str(row.get("accessionNumber") or "").replace("-", "")
                    prim = str(row.get("primaryDocument") or "")
                    if not (acc and prim):
                        return ""
                    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{prim}"
                except Exception:
                    return ""

            df["url"] = df.apply(_doc_url, axis=1)
        else:
            df["url"] = ""

        df["source"] = "sec_edgar"

        # Filter forms
        df = df[df["form"].isin(list(self.cfg.forms))]
        df = df.head(int(self.cfg.max_items))

        self._last_ok = True
        self._last_detail = f"ok ({len(df)} items)"

        keep = [c for c in ["form", "filed_at", "reportDate", "accessionNumber", "url", "source"] if c in df.columns]
        return df[keep].copy()


# --- CIK lookup ---

_CIK_CACHE: dict[str, str] = {}


def lookup_cik(symbol: str, *, headers: dict) -> Optional[str]:
    """Best-effort CIK lookup using SEC company_tickers.json.

    We cache in-process to avoid repeated downloads.
    """
    sym = _normalize_symbol(symbol)
    if not sym:
        return None
    if sym in _CIK_CACHE:
        return _CIK_CACHE[sym]

    # SEC publishes a mapping of tickers to CIK
    url = "https://www.sec.gov/files/company_tickers.json"
    try:
        obj = _get_json(url, headers=headers)
    except Exception:
        return None

    # obj is dict keyed by ints as strings
    try:
        for _, rec in obj.items():
            if not isinstance(rec, dict):
                continue
            t = str(rec.get("ticker") or "").upper().strip()
            if t == sym:
                cik = str(rec.get("cik_str") or "").strip()
                if cik:
                    _CIK_CACHE[sym] = cik
                    return cik
    except Exception:
        return None

    return None


# --- Download + extract (storage) ---


def download_filing_primary(
    *,
    data_dir: Path,
    symbol: str,
    accession_number: str,
    cik: str | int,
    primary_url: str,
    min_delay_s: float = 1.0,
) -> Path:
    """Download the primary document for a filing and store it under data/filings/...

    `min_delay_s` enforces a conservative SEC request rate (default 1 req/sec).
    """
    sym = _normalize_symbol(symbol)
    acc = str(accession_number or "").strip()
    headers = {"User-Agent": _user_agent(), "Accept-Encoding": "gzip, deflate"}

    # conservative rate limit
    time.sleep(max(0.0, float(min_delay_s)))
    txt = _get_text(primary_url, headers=headers)

    root = _filings_root(_data_dir(data_dir))
    ts = time.strftime("%Y-%m-%d")
    out_dir = root / sym / ts / acc.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pick extension by URL
    ext = ".html"
    if primary_url.lower().endswith(".txt"):
        ext = ".txt"
    elif primary_url.lower().endswith(".xml"):
        ext = ".xml"
    elif primary_url.lower().endswith(".htm"):
        ext = ".htm"

    out_path = out_dir / ("primary" + ext)
    out_path.write_text(txt, encoding="utf-8", errors="ignore")

    meta = {
        "symbol": sym,
        "cik": str(cik),
        "accessionNumber": acc,
        "primary_url": primary_url,
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return out_path
