from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time
from typing import Optional
from urllib.parse import urlparse

import pandas as pd
import requests

from .base import HaltsFeed, FeedStatus


def _domain(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower().strip()
    except Exception:
        return ""


@dataclass
class FetchResult:
    ok: bool
    url: str
    status: Optional[int] = None
    error: Optional[str] = None
    text: Optional[str] = None


class HttpFetcher:
    """Tiny HTTP fetcher with allowlist, retries, caching helpers.

    Policy is intentionally simple and conservative.
    """

    def __init__(
        self,
        *,
        allowed_domains: set[str],
        user_agent: str = "MarketHub/0.1 (+local-first)",
        retries: int = 2,
        timeout: float = 12.0,
    ) -> None:
        self.allowed_domains = {d.lower().strip() for d in allowed_domains if d}
        self.user_agent = user_agent
        self.retries = int(retries)
        self.timeout = float(timeout)

    def fetch_text(self, url: str) -> FetchResult:
        dom = _domain(url)
        if dom and (dom not in self.allowed_domains):
            return FetchResult(ok=False, url=url, error=f"domain not allowed: {dom}")

        headers = {"User-Agent": self.user_agent}
        last_err: Optional[str] = None
        for i in range(self.retries + 1):
            try:
                r = requests.get(url, headers=headers, timeout=self.timeout)
                if r.status_code >= 400:
                    last_err = f"HTTP {r.status_code}"
                    # backoff on 429/5xx
                    if r.status_code in (429, 500, 502, 503, 504) and i < self.retries:
                        time.sleep(0.6 + 0.8 * i)
                        continue
                    return FetchResult(ok=False, url=url, status=r.status_code, error=last_err)

                # best-effort decode
                r.encoding = r.encoding or "utf-8"
                return FetchResult(ok=True, url=url, status=r.status_code, text=r.text)
            except Exception as e:
                last_err = str(e)
                if i < self.retries:
                    time.sleep(0.6 + 0.8 * i)
                    continue
        return FetchResult(ok=False, url=url, error=last_err or "fetch failed")


def _feeds_cache_dir(data_dir: Path) -> Path:
    return (data_dir / "feeds_cache").resolve()


def _write_cache(
    data_dir: Path,
    *,
    kind: str,
    url: str,
    raw_text: Optional[str],
    parsed: Optional[pd.DataFrame],
    max_history: int = 20,
) -> None:
    """Write raw + parsed cache.

    - Always overwrite `latest_*.json/txt`
    - Also keep timestamped history (bounded)
    """

    d = _feeds_cache_dir(data_dir)
    d.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    meta = {"ts": ts, "kind": kind, "url": url}

    # raw
    if raw_text is not None:
        (d / f"latest_{kind}.txt").write_text(raw_text, encoding="utf-8", errors="ignore")
        (d / f"{kind}_{ts}.txt").write_text(raw_text, encoding="utf-8", errors="ignore")

    # parsed
    if parsed is not None:
        try:
            payload = {
                "meta": meta,
                "rows": json.loads(parsed.to_json(orient="records")),
            }
            (d / f"latest_{kind}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            (d / f"{kind}_{ts}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass

    # prune history
    try:
        files = sorted(d.glob(f"{kind}_*.*"), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in files[max_history:]:
            try:
                p.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass
    except Exception:
        pass


def _parse_tables_from_html(html: str) -> list[pd.DataFrame]:
    try:
        # pandas read_html is surprisingly effective for these pages.
        return list(pd.read_html(html))
    except Exception:
        return []


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    return out


def _normalize_halts(df: pd.DataFrame, *, source: str) -> pd.DataFrame:
    """Normalize into a stable halts schema.

    Target (decision C):
      symbol, market, reason, halt_time_et, resume_time_et, resumed, notes
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["symbol", "market", "reason", "halt_time_et", "resume_time_et", "resumed", "notes", "source"])  # noqa: E501

    d = _norm_cols(df)

    # common mapping across sources
    col_map = {
        "symbol": ["symbol", "issue_symbol", "issue", "ticker"],
        "market": ["market", "exchange"],
        "reason": ["reason", "reason_code", "reason_codes", "halt_reason", "halt_reason_code"],
        "halt_date": ["halt_date", "date", "halt_date_(from)", "halt_date(from)", "halt_date_from"],
        "halt_time": ["halt_time", "halt_time_et", "halt_time_(from)", "halt_time(from)", "halt_time_from"],
        "resume_date": ["resumption_date", "resume_date", "resumption_date_(to)", "resume_date_(to)", "resume_date(to)", "resume_date_to"],
        "resume_time": ["resumption_quote_time", "resumption_trade_time", "nyse_resume_time", "resume_time", "resume_time_et"],
        "name": ["issue_name", "name", "security_name"],
        "pause_threshold_price": ["pause_threshold_price"],
    }

    def pick(keys: list[str]) -> Optional[str]:
        for k in keys:
            if k in d.columns:
                return k
        return None

    sym_c = pick(col_map["symbol"]) or "symbol"
    mkt_c = pick(col_map["market"]) or None
    rsn_c = pick(col_map["reason"]) or None
    hd_c = pick(col_map["halt_date"]) or None
    ht_c = pick(col_map["halt_time"]) or None
    rd_c = pick(col_map["resume_date"]) or None
    rt_c = pick(col_map["resume_time"]) or None
    nm_c = pick(col_map["name"]) or None
    pt_c = pick(col_map["pause_threshold_price"]) or None

    out = pd.DataFrame()
    out["symbol"] = d.get(sym_c, "").astype(str).str.upper().str.strip()
    out["market"] = d.get(mkt_c, "").astype(str).str.strip() if mkt_c else ""
    out["reason"] = d.get(rsn_c, "").astype(str).str.strip() if rsn_c else ""

    # Build time strings (ET).
    halt_dt = None
    if hd_c and ht_c:
        halt_dt = d[hd_c].astype(str).str.strip() + " " + d[ht_c].astype(str).str.strip()
    elif ht_c:
        halt_dt = d[ht_c].astype(str).str.strip()
    out["halt_time_et"] = halt_dt if halt_dt is not None else ""

    resume_dt = None
    if rd_c and rt_c:
        resume_dt = d[rd_c].astype(str).str.strip() + " " + d[rt_c].astype(str).str.strip()
    elif rt_c:
        resume_dt = d[rt_c].astype(str).str.strip()
    out["resume_time_et"] = resume_dt if resume_dt is not None else ""

    # resumed flag: if we have a resume time string, assume resumed/expected
    out["resumed"] = out["resume_time_et"].astype(str).str.strip().ne("")

    notes = []
    if nm_c:
        notes.append("name")
    if pt_c:
        notes.append("pause_threshold_price")

    if notes:
        # Combine a couple extra fields into notes
        extra_bits = []
        if nm_c:
            extra_bits.append("name=" + d[nm_c].astype(str))
        if pt_c:
            extra_bits.append("pth=" + d[pt_c].astype(str))
        out["notes"] = extra_bits[0]
        for b in extra_bits[1:]:
            out["notes"] = out["notes"].astype(str) + " | " + b.astype(str)
    else:
        out["notes"] = ""

    out["source"] = source

    # drop empty symbol rows
    out = out[out["symbol"].astype(str).str.len() > 0]
    return out


class WebHaltsFeed(HaltsFeed):
    """Fetch halts from public exchange pages (Nasdaq/NYSE/Cboe) and normalize.

    This is intentionally best-effort: HTML parsing can break; failures should surface as status.
    """

    def __init__(
        self,
        *,
        data_dir: Path,
        urls: dict[str, str],
        include_cboe: bool = True,
        source_priority: list[str] | None = None,
        allowed_domains: Optional[set[str]] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.urls = dict(urls)
        self.include_cboe = bool(include_cboe)
        self.source_priority = source_priority or ["cboe", "nasdaq", "nyse"]

        # Allowed domains (decision: include federalreserve.gov too, but here only halts matter)
        allowed = allowed_domains or {
            "www.nasdaqtrader.com",
            "nasdaqtrader.com",
            "beta.nyse.com",
            "www.nyse.com",
            "www.cboe.com",
            "cboe.com",
        }
        self.fetcher = HttpFetcher(allowed_domains=allowed, user_agent="MarketHub/halts")

        self._last_ok = False
        self._last_detail = "not fetched"

    def status(self) -> FeedStatus:
        return FeedStatus(ok=bool(self._last_ok), name=self.name, detail=str(self._last_detail))

    def _fetch_one(self, key: str, url: str) -> tuple[pd.DataFrame, Optional[str]]:
        fr = self.fetcher.fetch_text(url)
        if not fr.ok or not fr.text:
            return pd.DataFrame(), fr.error or "fetch failed"

        txt = fr.text

        # If it's a CSV-ish endpoint (e.g., NYSE download), parse as CSV first.
        raw: pd.DataFrame | None = None
        try:
            head = (txt or "").lstrip()[:200].lower()
            if "," in head and ("halt" in head or "symbol" in head) and ("\n" in txt):
                import io

                raw = pd.read_csv(io.StringIO(txt))
        except Exception:
            raw = None

        if raw is None:
            tables = _parse_tables_from_html(txt)
            if not tables:
                _write_cache(self.data_dir, kind=f"halts_{key}", url=url, raw_text=txt, parsed=None)
                return pd.DataFrame(), "no tables parsed"

            # Heuristic: pick the largest table
            tables = sorted(tables, key=lambda x: x.shape[0] * max(1, x.shape[1]), reverse=True)
            raw = tables[0]

        norm = _normalize_halts(raw, source=key)

        _write_cache(self.data_dir, kind=f"halts_{key}", url=url, raw_text=fr.text, parsed=norm)
        return norm, None

    def fetch_halts(self) -> pd.DataFrame:
        # Pull in priority order and concatenate.
        pieces: list[pd.DataFrame] = []
        errors: list[str] = []

        for k in self.source_priority:
            if k == "cboe" and not self.include_cboe:
                continue
            url = self.urls.get(k)
            if not url:
                continue
            df, err = self._fetch_one(k, url)
            if err:
                errors.append(f"{k}: {err}")
            if not df.empty:
                pieces.append(df)

        if not pieces:
            self._last_ok = False
            self._last_detail = "; ".join(errors)[:240] if errors else "no data"
            return pd.DataFrame(columns=["symbol", "market", "reason", "halt_time_et", "resume_time_et", "resumed", "notes", "source"])  # noqa: E501

        all_df = pd.concat(pieces, axis=0, ignore_index=True)

        # Dedupe (decision: symbol + reason + halt_time)
        key_cols = ["symbol", "reason", "halt_time_et"]
        for c in key_cols:
            if c not in all_df.columns:
                all_df[c] = ""

        all_df = all_df.drop_duplicates(subset=key_cols, keep="first")

        self._last_ok = True
        self._last_detail = f"ok ({len(all_df)} rows)"
        return all_df
