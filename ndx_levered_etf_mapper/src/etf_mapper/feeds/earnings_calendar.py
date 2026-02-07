from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time
from typing import Optional

import pandas as pd

from .base import EarningsFeed, FeedStatus


def _rss_feeds_path(data_dir: Path) -> Path:
    return (data_dir / "rss_feeds.json").resolve()


def _read_rss_feeds(data_dir: Path) -> dict:
    p = _rss_feeds_path(data_dir)
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


@dataclass
class EarningsFetchConfig:
    # Placeholder config: we don't yet have a real market-wide earnings calendar provider.
    # This class exists so we can add provider choice later without rewriting the UI.
    max_rows: int = 200


class WebEarningsCalendarFeed(EarningsFeed):
    """Seamless earnings calendar feed (scaffold).

    We intentionally do NOT implement scraping a random site by default.
    The UI/storage is wired so once you pick a provider/source, we just implement it here.

    Returns normalized dataframe with columns:
      symbol, report_date, call_time, eps_est, eps_act, rev_est, rev_act, guidance_flag, notes, url, source

    For now, status is BLOCKED unless a provider/source is configured.
    """

    def __init__(self, *, data_dir: Path, cfg: Optional[EarningsFetchConfig] = None) -> None:
        self.data_dir = Path(data_dir)
        self.cfg = cfg or EarningsFetchConfig()
        self._last_ok = False
        self._last_detail = "not configured"

    def status(self) -> FeedStatus:
        return FeedStatus(ok=bool(self._last_ok), name=self.name, detail=str(self._last_detail))

    def fetch_earnings(self) -> pd.DataFrame:
        rss = _read_rss_feeds(self.data_dir)
        urls = (((rss.get("earnings") or {}).get("urls")) or []) if isinstance(rss, dict) else []

        # If user only provided company IR RSS, that's not a calendar. We keep this blocked.
        if not urls:
            self._last_ok = False
            self._last_detail = "blocked: no earnings source configured (need calendar provider)"
            return self._empty()

        self._last_ok = False
        self._last_detail = "blocked: earnings URLs present but not a calendar provider (needs decision)"
        return self._empty()

    @staticmethod
    def _empty() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "symbol",
                "report_date",
                "call_time",
                "eps_est",
                "eps_act",
                "rev_est",
                "rev_act",
                "guidance_flag",
                "notes",
                "url",
                "source",
            ]
        )


def cache_path(data_dir: Path) -> Path:
    return (Path(data_dir) / "feeds_cache" / "latest_earnings_calendar.json").resolve()


def write_cache(data_dir: Path, df: pd.DataFrame, *, source: str, url: str | None = None) -> None:
    """Write parsed cache (for UI).

    This keeps the UI snappy and makes issues debuggable.
    """
    d = (Path(data_dir) / "feeds_cache").resolve()
    d.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": {
            "ts": time.strftime("%Y%m%d_%H%M%S"),
            "source": source,
            "url": url,
        },
        "rows": json.loads((df if isinstance(df, pd.DataFrame) else pd.DataFrame()).to_json(orient="records")),
    }

    cache_path(data_dir).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_cache(data_dir: Path) -> tuple[Optional[dict], pd.DataFrame]:
    p = cache_path(data_dir)
    if not p.exists():
        return None, pd.DataFrame()
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        rows = obj.get("rows") if isinstance(obj, dict) else None
        df = pd.DataFrame(rows) if isinstance(rows, list) else pd.DataFrame()
        return obj if isinstance(obj, dict) else None, df
    except Exception:
        return None, pd.DataFrame()
