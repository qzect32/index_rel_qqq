from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests

from .base import FeedStatus


def _data_dir(p: Path | str) -> Path:
    return Path(p).resolve()


def _cache_dir(data_dir: Path) -> Path:
    return (data_dir / "feeds_cache").resolve()


def _safe_text(x) -> str:
    return str(x or "").strip()


def _parse_rss_or_atom(xml_text: str, *, source: str) -> list[dict]:
    """Parse RSS/Atom into normalized rows.

    Row keys: title, link, published, source
    """
    if not xml_text:
        return []

    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return []

    rows: list[dict] = []

    # RSS: <rss><channel><item>...
    ch = root.find("channel")
    if ch is not None:
        for item in ch.findall("item"):
            title = _safe_text(item.findtext("title"))
            link = _safe_text(item.findtext("link"))
            pub = _safe_text(item.findtext("pubDate")) or _safe_text(item.findtext("date"))
            if not title:
                continue
            rows.append({"title": title, "link": link, "published": pub, "source": source})
        return rows

    # Atom: <feed><entry>...
    if root.tag.endswith("feed"):
        # Atom namespaces can be present; use wildcard search
        for entry in root.findall("{*}entry"):
            title = _safe_text(entry.findtext("{*}title"))
            pub = _safe_text(entry.findtext("{*}updated")) or _safe_text(entry.findtext("{*}published"))
            link = ""
            try:
                lk = entry.find("{*}link")
                if lk is not None:
                    link = _safe_text(lk.attrib.get("href"))
            except Exception:
                link = ""
            if not title:
                continue
            rows.append({"title": title, "link": link, "published": pub, "source": source})

    return rows


@dataclass
class FedRssConfig:
    timeout_s: float = 15.0
    max_items_per_feed: int = 30


class FedRssFeed:
    """Fetch and cache Federal Reserve RSS/Atom feeds listed in data/rss_feeds.json."""

    name = "fed_rss"

    def __init__(self, *, data_dir: Path, urls: list[str]) -> None:
        self.data_dir = _data_dir(data_dir)
        self.urls = [str(u) for u in (urls or []) if str(u).strip()]
        self.cfg = FedRssConfig()
        self._last_ok = False
        self._last_detail = "not fetched"

    def status(self) -> FeedStatus:
        return FeedStatus(ok=bool(self._last_ok), name=self.name, detail=str(self._last_detail))

    def cache_path(self) -> Path:
        return _cache_dir(self.data_dir) / "latest_fed_rss.json"

    def fetch(self) -> pd.DataFrame:
        if not self.urls:
            self._last_ok = False
            self._last_detail = "no urls configured"
            return pd.DataFrame(columns=["title", "published", "link", "source"])

        headers = {
            "User-Agent": "MarketHub/0.1 (macro rss)",
            "Accept": "application/rss+xml, application/atom+xml, text/xml, application/xml;q=0.9, */*;q=0.8",
        }

        all_rows: list[dict] = []
        ok_any = False
        errs = 0

        for url in self.urls:
            try:
                r = requests.get(url, headers=headers, timeout=float(self.cfg.timeout_s))
                r.raise_for_status()
                txt = r.text
                rows = _parse_rss_or_atom(txt, source=url)
                if rows:
                    ok_any = True
                    all_rows.extend(rows[: int(self.cfg.max_items_per_feed)])
                else:
                    errs += 1
            except Exception:
                errs += 1
                continue

        df = pd.DataFrame(all_rows)
        if not df.empty:
            # best-effort published parsing
            df["published_ts"] = pd.to_datetime(df.get("published"), errors="coerce", utc=True)
            df = df.sort_values("published_ts", ascending=False)
            keep = [c for c in ["title", "published", "link", "source", "published_ts"] if c in df.columns]
            df = df[keep]

        self._last_ok = ok_any
        self._last_detail = f"ok ({len(df)} items)" if ok_any else f"failed ({errs} feeds)"

        # cache
        try:
            p = self.cache_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            obj = {
                "fetched_at": time.time(),
                "status": self.status().detail,
                "rows": df.to_dict(orient="records"),
            }
            p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
        except Exception:
            pass

        return df

    def read_cache(self) -> pd.DataFrame:
        p = self.cache_path()
        if not p.exists():
            return pd.DataFrame(columns=["title", "published", "link", "source"])
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            rows = obj.get("rows") if isinstance(obj, dict) else None
            df = pd.DataFrame(rows) if isinstance(rows, list) else pd.DataFrame()
            if "published_ts" in df.columns:
                df["published_ts"] = pd.to_datetime(df["published_ts"], errors="coerce", utc=True)
            return df
        except Exception:
            return pd.DataFrame(columns=["title", "published", "link", "source"])

    def cache_age_s(self) -> float:
        p = self.cache_path()
        if not p.exists():
            return 1e18
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            ts = float(obj.get("fetched_at", 0.0)) if isinstance(obj, dict) else 0.0
            return max(0.0, time.time() - ts)
        except Exception:
            return 1e18
