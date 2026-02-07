from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time
from typing import Iterable

import pandas as pd

from etf_mapper.feeds.filings_edgar import EdgarFilingsFeed, lookup_cik, download_filing_primary
from etf_mapper.analysis.edgar_diff import (
    extract_plain_text,
    extract_sections,
    load_risk_phrases,
    phrase_counts,
    diff_sentences,
    score_change,
    write_report,
)


@dataclass
class WatcherConfig:
    interval_s: float = 24 * 60 * 60
    diff_threshold: int = 10
    retention_days: int = 30


def _state_path(data_dir: Path) -> Path:
    return (Path(data_dir).resolve() / "filings_watcher_state.json").resolve()


def load_state(data_dir: Path) -> dict:
    p = _state_path(data_dir)
    if not p.exists():
        return {"last_run": 0.0, "last_seen": {}}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            return {"last_run": 0.0, "last_seen": {}}
        obj.setdefault("last_run", 0.0)
        obj.setdefault("last_seen", {})
        if not isinstance(obj.get("last_seen"), dict):
            obj["last_seen"] = {}
        return obj
    except Exception:
        return {"last_run": 0.0, "last_seen": {}}


def save_state(data_dir: Path, state: dict) -> None:
    p = _state_path(data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _cleanup_retention(data_dir: Path, *, retention_days: int) -> int:
    root = (Path(data_dir).resolve() / "filings").resolve()
    if not root.exists():
        return 0
    cutoff = time.time() - float(retention_days) * 24 * 60 * 60
    removed = 0

    # Only clean report directories
    for p in root.glob("*/reports/*"):
        try:
            if not p.is_dir():
                continue
            mtime = p.stat().st_mtime
            if mtime < cutoff:
                # best-effort delete contents
                for q in sorted(p.rglob("*"), reverse=True):
                    try:
                        if q.is_file():
                            q.unlink(missing_ok=True)
                        elif q.is_dir():
                            q.rmdir()
                    except Exception:
                        pass
                try:
                    p.rmdir()
                    removed += 1
                except Exception:
                    pass
        except Exception:
            continue

    return removed


def maybe_run(
    *,
    data_dir: Path,
    symbols: Iterable[str],
    cfg: WatcherConfig | None = None,
) -> dict:
    """Run once per `cfg.interval_s`.

    Returns a summary dict:
      { ran: bool, alerts: [...], removed_reports: int }
    """
    cfg = cfg or WatcherConfig()
    data_dir = Path(data_dir).resolve()
    state = load_state(data_dir)

    now = time.time()
    last_run = float(state.get("last_run", 0.0) or 0.0)
    if (now - last_run) < float(cfg.interval_s):
        return {"ran": False, "alerts": [], "removed_reports": 0}

    # retention cleanup
    removed = _cleanup_retention(data_dir, retention_days=int(cfg.retention_days))

    feed = EdgarFilingsFeed(data_dir=data_dir)

    phrases = load_risk_phrases(Path(__file__).resolve().parents[1] / "analysis" / "risk_phrases.txt")
    headers = {"User-Agent": "MarketHub/0.1 (local research; contact: user)", "Accept-Encoding": "gzip, deflate"}

    alerts: list[dict] = []

    last_seen: dict = state.get("last_seen", {}) if isinstance(state.get("last_seen"), dict) else {}

    for sym0 in symbols:
        sym = str(sym0 or "").upper().strip()
        if not sym:
            continue
        if sym.startswith("/"):
            # skip futures symbols
            continue

        try:
            fdf = feed.fetch_filings(sym)
        except Exception:
            continue

        if not isinstance(fdf, pd.DataFrame) or fdf.empty:
            continue

        # sort newest first
        fdf2 = fdf.copy()
        if "filed_at" in fdf2.columns:
            fdf2["filed_at"] = pd.to_datetime(fdf2["filed_at"], errors="coerce")
            fdf2 = fdf2.sort_values("filed_at", ascending=False)

        cur = fdf2.iloc[0].to_dict()
        prev = fdf2.iloc[1].to_dict() if len(fdf2) > 1 else None
        if prev is None:
            continue

        acc_cur = str(cur.get("accessionNumber") or "").strip()
        acc_prev = str(prev.get("accessionNumber") or "").strip()
        if not acc_cur or not acc_prev:
            continue

        if str(last_seen.get(sym) or "") == acc_cur:
            continue  # no new filing since last run

        url_cur = str(cur.get("url") or "").strip()
        url_prev = str(prev.get("url") or "").strip()
        if not (url_cur and url_prev):
            continue

        try:
            cik = lookup_cik(sym, headers=headers) or ""

            p_cur = download_filing_primary(
                data_dir=data_dir,
                symbol=sym,
                accession_number=acc_cur,
                cik=cik,
                primary_url=url_cur,
                min_delay_s=1.0,
            )
            p_prev = download_filing_primary(
                data_dir=data_dir,
                symbol=sym,
                accession_number=acc_prev,
                cik=cik,
                primary_url=url_prev,
                min_delay_s=1.0,
            )

            txt_cur = extract_plain_text(p_cur.read_text(encoding="utf-8", errors="ignore"))
            txt_prev = extract_plain_text(p_prev.read_text(encoding="utf-8", errors="ignore"))

            secs_cur = {s.name: s.text for s in extract_sections(txt_cur)}
            secs_prev = {s.name: s.text for s in extract_sections(txt_prev)}

            results = {"sections": {}}
            total_score = 0
            for sec_name in sorted(set(secs_cur.keys()) | set(secs_prev.keys())):
                b = secs_prev.get(sec_name, "")
                a = secs_cur.get(sec_name, "")
                counts_b = phrase_counts(b, phrases)
                counts_a = phrase_counts(a, phrases)
                sc = score_change(counts_b, counts_a)
                ds = diff_sentences(b, a)
                results["sections"][sec_name] = {"score": sc, "sentences": ds}
                try:
                    total_score += int((sc or {}).get("score", 0))
                except Exception:
                    pass

            out_dir = (data_dir / "filings" / sym / "reports" / f"{acc_cur}_vs_{acc_prev}")
            meta = {
                "symbol": sym,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "baseline": acc_prev,
                "current": acc_cur,
                "current_url": url_cur,
                "baseline_url": url_prev,
                "total_score": total_score,
            }
            md_path, js_path = write_report(out_dir=out_dir, meta=meta, results=results)

            if total_score >= int(cfg.diff_threshold):
                alerts.append(
                    {
                        "symbol": sym,
                        "score": total_score,
                        "current": acc_cur,
                        "baseline": acc_prev,
                        "report_md": str(md_path),
                        "report_json": str(js_path),
                    }
                )

            last_seen[sym] = acc_cur
        except Exception:
            # if something fails mid-way, don't advance last_seen for this symbol
            continue

    state["last_run"] = now
    state["last_seen"] = last_seen
    save_state(data_dir, state)

    return {"ran": True, "alerts": alerts, "removed_reports": removed}
