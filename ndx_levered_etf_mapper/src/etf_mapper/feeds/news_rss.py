from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests

from concurrent.futures import ThreadPoolExecutor, as_completed

from .base import FeedStatus
from .news_rss_state import load_state, save_state


def _data_dir(p: Path | str) -> Path:
    return Path(p).resolve()


def _cache_dir(data_dir: Path) -> Path:
    return (data_dir / "feeds_cache").resolve()


def _safe_text(x) -> str:
    return str(x or "").strip()


def _parse_rss_or_atom(xml_text: str, *, source: str) -> list[dict]:
    if not xml_text:
        return []
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return []

    rows: list[dict] = []

    # RSS
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

    # Atom
    if root.tag.endswith("feed"):
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
class NewsRssConfig:
    timeout_s: float = 15.0
    max_items_per_feed: int = 30
    parallel: bool = True
    max_show: int = 50


class NewsRssFeed:
    name = "news_rss"

    def __init__(self, *, data_dir: Path, urls: list[str], cfg: NewsRssConfig | None = None) -> None:
        self.data_dir = _data_dir(data_dir)
        self.urls = [str(u) for u in (urls or []) if str(u).strip()]
        self.cfg = cfg or NewsRssConfig()
        self._last_ok = False
        self._last_detail = "not fetched"
        self._last_per_feed: dict[str, dict] = {}

    def per_feed_status(self) -> dict[str, dict]:
        return dict(self._last_per_feed)

    def status(self) -> FeedStatus:
        return FeedStatus(ok=bool(self._last_ok), name=self.name, detail=str(self._last_detail))

    def cache_path(self) -> Path:
        return _cache_dir(self.data_dir) / "latest_news_rss.json"

    def fetch(self) -> pd.DataFrame:
        if not self.urls:
            self._last_ok = False
            self._last_detail = "no urls configured"
            return pd.DataFrame(columns=["title", "published", "link", "source"])

        headers = {
            "User-Agent": "MarketHub/0.1 (news rss)",
            "Accept": "application/rss+xml, application/atom+xml, text/xml, application/xml;q=0.9, */*;q=0.8",
        }

        state = load_state(self.data_dir)
        fail_counts = state.get("fail_counts", {}) if isinstance(state.get("fail_counts"), dict) else {}

        all_rows: list[dict] = []
        ok_any = False
        errs = 0
        per_feed: dict[str, dict] = {}

        def _one(url: str) -> tuple[str, bool, str, list[dict]]:
            try:
                r = requests.get(url, headers=headers, timeout=float(self.cfg.timeout_s))
                r.raise_for_status()
                rows = _parse_rss_or_atom(r.text, source=url)
                if rows:
                    return (url, True, f"ok ({len(rows)})", rows[: int(self.cfg.max_items_per_feed)])
                return (url, False, "parsed 0 items", [])
            except Exception as e:
                return (url, False, str(e), [])

        urls = list(self.urls)

        if bool(self.cfg.parallel) and len(urls) > 1:
            with ThreadPoolExecutor(max_workers=min(8, len(urls))) as ex:
                futs = {ex.submit(_one, u): u for u in urls}
                for fut in as_completed(futs):
                    url, ok, detail, rows = fut.result()
                    per_feed[url] = {"ok": ok, "detail": detail, "fail_count": int(fail_counts.get(url, 0))}
                    if ok:
                        ok_any = True
                        all_rows.extend(rows)
                        fail_counts[url] = 0
                    else:
                        errs += 1
                        fail_counts[url] = int(fail_counts.get(url, 0)) + 1
                        per_feed[url]["fail_count"] = int(fail_counts.get(url, 0))
        else:
            for u in urls:
                url, ok, detail, rows = _one(u)
                per_feed[url] = {"ok": ok, "detail": detail, "fail_count": int(fail_counts.get(url, 0))}
                if ok:
                    ok_any = True
                    all_rows.extend(rows)
                    fail_counts[url] = 0
                else:
                    errs += 1
                    fail_counts[url] = int(fail_counts.get(url, 0)) + 1
                    per_feed[url]["fail_count"] = int(fail_counts.get(url, 0))

        state["fail_counts"] = fail_counts
        save_state(self.data_dir, state)

        df = pd.DataFrame(all_rows)
        if not df.empty:
            df["published_ts"] = pd.to_datetime(df.get("published"), errors="coerce", utc=True)
            df = df.sort_values("published_ts", ascending=False)
            keep = [c for c in ["published", "title", "link", "source", "published_ts"] if c in df.columns]
            df = df[keep]
            df = df.head(int(self.cfg.max_show))

        self._last_ok = ok_any
        self._last_detail = f"ok ({len(df)} items)" if ok_any else f"failed ({errs} feeds)"
        self._last_per_feed = per_feed

        try:
            p = self.cache_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            obj = {
                "fetched_at": time.time(),
                "status": self.status().detail,
                "rows": df.to_dict(orient="records"),
                "per_feed": per_feed,
            }
            p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
        except Exception:
            pass

        return df

    def read_cache(self) -> pd.DataFrame:
        p = self.cache_path()
        if not p.exists():
            return pd.DataFrame(columns=["published", "title", "link", "source"])
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            rows = obj.get("rows") if isinstance(obj, dict) else None
            df = pd.DataFrame(rows) if isinstance(rows, list) else pd.DataFrame()
            if isinstance(obj, dict) and isinstance(obj.get("per_feed"), dict):
                self._last_per_feed = obj.get("per_feed")
            if "published_ts" in df.columns:
                df["published_ts"] = pd.to_datetime(df["published_ts"], errors="coerce", utc=True)
            return df
        except Exception:
            return pd.DataFrame(columns=["published", "title", "link", "source"])

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
