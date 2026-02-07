from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
import time
from typing import Optional

import pandas as pd


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _norm(s: str) -> str:
    s = str(s or "")
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _sentences(text: str) -> list[str]:
    t = _norm(text)
    if not t:
        return []
    # Very lightweight sentence split
    parts = re.split(r"(?<=[\.!\?])\s+", t)
    out = []
    for p in parts:
        p = p.strip()
        if len(p) < 35:
            continue
        if len(p) > 600:
            continue
        out.append(p)
    return out


def _read_text_file(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def extract_plain_text(html_or_text: str) -> str:
    s = str(html_or_text or "")
    if not s:
        return ""

    # If it looks like HTML, strip tags.
    if "<html" in s.lower() or "<body" in s.lower() or "</div>" in s.lower():
        try:
            from bs4 import BeautifulSoup  # type: ignore

            soup = BeautifulSoup(s, "lxml")
            # remove scripts/styles
            for tag in soup(["script", "style", "noscript"]):
                try:
                    tag.decompose()
                except Exception:
                    pass
            txt = soup.get_text("\n")
            return _norm(txt)
        except Exception:
            # fall back to crude removal
            s2 = re.sub(r"<[^>]+>", " ", s)
            return _norm(s2)

    return _norm(s)


@dataclass
class SectionExtract:
    name: str
    text: str


def _slice_between(text: str, start_pat: str, end_pat: str, *, flags=re.I) -> str:
    t = text
    m1 = re.search(start_pat, t, flags)
    if not m1:
        return ""
    t2 = t[m1.start() :]
    m2 = re.search(end_pat, t2, flags)
    if not m2:
        return t2[:60000]
    return t2[: m2.start()]


def extract_sections(plain: str) -> list[SectionExtract]:
    """Best-effort extraction for:
    - Risk Factors
    - MD&A

    Not perfect; intended for change detection, not legal accuracy.
    """
    t = plain
    if not t:
        return []

    # common headings
    risk = _slice_between(
        t,
        start_pat=r"\b(item\s+1a\.?\s+risk\s+factors|risk\s+factors)\b",
        end_pat=r"\b(item\s+1b\b|item\s+2\b|unresolved\s+staff\s+comments)\b",
    )

    mda = _slice_between(
        t,
        start_pat=r"\b(item\s+7\.?\s+management[’']s\s+discussion|management[’']s\s+discussion\s+and\s+analysis|md&a)\b",
        end_pat=r"\b(item\s+7a\b|item\s+8\b|financial\s+statements\s+and\s+supplementary\s+data)\b",
    )

    out: list[SectionExtract] = []
    if risk:
        out.append(SectionExtract("risk_factors", risk[:120000]))
    if mda:
        out.append(SectionExtract("mda", mda[:120000]))

    return out


def load_risk_phrases(path: Path) -> list[str]:
    lines = []
    try:
        for ln in path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            lines.append(ln.lower())
    except Exception:
        return []
    # uniq, stable
    out = []
    for x in lines:
        if x not in out:
            out.append(x)
    return out


def phrase_counts(text: str, phrases: list[str]) -> dict[str, int]:
    t = (text or "").lower()
    out: dict[str, int] = {}
    for ph in phrases:
        if not ph:
            continue
        out[ph] = t.count(ph)
    return out


def diff_sentences(before: str, after: str, *, max_items: int = 40) -> dict:
    b = _sentences(before)
    a = _sentences(after)

    # normalize for set compare
    bn = {re.sub(r"\W+", " ", s.lower()).strip(): s for s in b}
    an = {re.sub(r"\W+", " ", s.lower()).strip(): s for s in a}

    bkeys = set(bn.keys())
    akeys = set(an.keys())

    added_k = list(akeys - bkeys)
    removed_k = list(bkeys - akeys)

    added = [an[k] for k in added_k[:max_items] if k in an]
    removed = [bn[k] for k in removed_k[:max_items] if k in bn]

    return {
        "added": added,
        "removed": removed,
        "n_added": int(len(akeys - bkeys)),
        "n_removed": int(len(bkeys - akeys)),
    }


def score_change(counts_before: dict[str, int], counts_after: dict[str, int]) -> dict:
    keys = sorted(set(counts_before.keys()) | set(counts_after.keys()))
    delta = {k: int(counts_after.get(k, 0) - counts_before.get(k, 0)) for k in keys}
    score = int(sum(max(0, v) for v in delta.values()))
    return {"score": score, "delta": delta}


def write_report(
    *,
    out_dir: Path,
    meta: dict,
    results: dict,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = out_dir / "report.json"
    json_path.write_text(json.dumps({"meta": meta, "results": results}, indent=2), encoding="utf-8")

    # Markdown
    md_path = out_dir / "report.md"

    def _md_list(xs: list[str]) -> str:
        if not xs:
            return "(none)"
        return "\n".join([f"- {x}" for x in xs])

    lines = []
    lines.append(f"# Filing change report — {meta.get('symbol','')}")
    lines.append("")
    lines.append(f"Generated: `{meta.get('generated_at')}`")
    lines.append(f"Baseline: `{meta.get('baseline')}`")
    lines.append("")

    for sec_name, sec in (results.get("sections") or {}).items():
        lines.append(f"## {sec_name}")
        lines.append("")
        sc = sec.get("score")
        if isinstance(sc, dict):
            lines.append(f"Risk phrase score (added mentions): **{sc.get('score',0)}**")
        lines.append("")

        sent = sec.get("sentences") or {}
        lines.append("### Top added sentences")
        lines.append(_md_list(sent.get("added") or []))
        lines.append("")
        lines.append("### Top removed sentences")
        lines.append(_md_list(sent.get("removed") or []))
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path, json_path
