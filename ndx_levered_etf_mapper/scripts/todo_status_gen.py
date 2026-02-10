"""Generate TODO status trackers.

Source of truth is TODO.md.

Outputs:
- TODO_STATUS.md (all items)
- TODO_STATUS_SCAFFOLDS.md (items that are primarily *scaffold/build* work)
- TODO_STATUS_QA.md (items that are primarily *QA/integration/user-testing* work)

Why:
Payne wants the list quantified and to keep "scaffolds" separate from
"QA/testing/OAuth/endpoint verification" work so progress is unambiguous.

Notes:
- This script is intentionally heuristic. You can override classification by
  adding one of these tokens to the TODO item text:
    [SCAFFOLD]  [QA]  [USER]

Statuses (for the status files):
- DONE: checkbox is checked
- IN-PROGRESS: open and scaffoldable without external validation
- BLOCKED: open and likely requires provider choice/docs/endpoint confirmation

Heuristic: mark BLOCKED if the line contains certain hints.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TODO = ROOT / "TODO.md"
OUT_ALL = ROOT / "TODO_STATUS.md"
OUT_SCAFF = ROOT / "TODO_STATUS_SCAFFOLDS.md"
OUT_QA = ROOT / "TODO_STATUS_QA.md"

BLOCKED_HINTS = [
    "source tbd",
    "requires",
    "external",
    "endpoint",
    "feed",
    "if exists",
    "not yet",
    "scope",
    "oauth",
    "entitlement",
    "docs",
    "portal",
]

QA_HINTS = [
    "oauth",
    "scope",
    "endpoint",
    "entitlement",
    "confirm",
    "verify",
    "test",
    "smoke",
    "troubleshoot",
    "debug",
    "url",
    "docs",
    "portal",
    "fixture",
    "contract test",
    "golden-record",
]

USER_HINTS = [
    "payne",
    "open github",
    "github",
    "start",
    "run",
    "click",
    "log in",
    "login",
]


@dataclass
class Item:
    idx: int
    title: str
    is_done: bool
    raw: str
    type: str  # SCAFFOLD|QA|USER
    status: str  # DONE|IN-PROGRESS|BLOCKED


def _iter_items(text: str):
    """Yield (item_text, is_done, raw_line)."""
    for ln in text.splitlines():
        m1 = re.match(r"^\s*- \[( |x)\] (.+)$", ln)
        if m1:
            is_done = (m1.group(1) or "").lower() == "x"
            yield m1.group(2).strip(), is_done, ln
            continue
        m2 = re.match(r"^\s*(\d+)\. \[( |x)\] (.+)$", ln)
        if m2:
            is_done = (m2.group(2) or "").lower() == "x"
            yield m2.group(3).strip(), is_done, ln


def _explicit_type(title: str) -> str | None:
    t = title.upper()
    if "[QA]" in t:
        return "QA"
    if "[USER]" in t:
        return "USER"
    if "[SCAFFOLD]" in t:
        return "SCAFFOLD"
    return None


def _clean_title(title: str) -> str:
    # remove explicit markers from display
    return (
        title.replace("[QA]", "")
        .replace("[qa]", "")
        .replace("[USER]", "")
        .replace("[user]", "")
        .replace("[SCAFFOLD]", "")
        .replace("[scaffold]", "")
        .strip()
    )


def _classify_type(title: str) -> str:
    exp = _explicit_type(title)
    if exp:
        return exp
    low = title.lower()
    if any(h in low for h in USER_HINTS):
        return "USER"
    if any(h in low for h in QA_HINTS):
        return "QA"
    return "SCAFFOLD"


def _classify_status(title: str, *, is_done: bool) -> str:
    if is_done:
        return "DONE"
    low = title.lower()
    blocked = any(h in low for h in BLOCKED_HINTS)
    return "BLOCKED" if blocked else "IN-PROGRESS"


def _render(items: list[Item], *, title: str) -> str:
    lines: list[str] = []
    lines.append(f"# {title}\n")
    lines.append("Generated from TODO.md. Edit TODO.md for canonical tasks; update via scripts/todo_status_gen.py.\n")
    lines.append("\n")

    for it in items:
        lines.append(f"## {it.idx}. {it.title}\n")
        lines.append(f"- TYPE: {it.type}\n")
        lines.append(f"- STATUS: {it.status}\n")
        if it.status == "DONE":
            lines.append("- NEXT: (none)\n")
            lines.append("- BLOCKERS: none\n")
        else:
            lines.append("- NEXT: (fill)\n")
            if it.status == "BLOCKED":
                lines.append("- BLOCKERS: (fill decision/provider/endpoint)\n")
            else:
                lines.append("- BLOCKERS: none (scaffoldable)\n")
        lines.append("\n")

    return "".join(lines)


def main() -> int:
    text = TODO.read_text(encoding="utf-8")
    raw_items = list(_iter_items(text))

    items: list[Item] = []
    for i, (title, is_done, raw) in enumerate(raw_items, start=1):
        t = _classify_type(title)
        st = _classify_status(title, is_done=is_done)
        items.append(Item(idx=i, title=_clean_title(title), is_done=is_done, raw=raw, type=t, status=st))

    scaff = [it for it in items if it.type == "SCAFFOLD"]
    qa = [it for it in items if it.type in ("QA", "USER")]

    OUT_ALL.write_text(_render(items, title="TODO_STATUS"), encoding="utf-8")
    OUT_SCAFF.write_text(_render(scaff, title="TODO_STATUS_SCAFFOLDS"), encoding="utf-8")
    OUT_QA.write_text(_render(qa, title="TODO_STATUS_QA"), encoding="utf-8")

    print(f"Wrote: {OUT_ALL}")
    print(f"Wrote: {OUT_SCAFF}")
    print(f"Wrote: {OUT_QA}")
    print(f"Items (all): {len(items)} | scaffolds: {len(scaff)} | qa/user: {len(qa)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
