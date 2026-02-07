"""Generate a status tracker for TODO.md.

This does not modify TODO.md; it produces TODO_STATUS.md with one entry per checkbox item.

Statuses:
- IN-PROGRESS: scaffold exists or can be scaffolded without external decisions
- BLOCKED: requires a provider choice, endpoint confirmation, or explicit decision

Heuristic: mark BLOCKED if the line contains "source TBD", "requires", "external", "endpoint", "feed", "if exists".

You can refine this later.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TODO = ROOT / "TODO.md"
OUT = ROOT / "TODO_STATUS.md"

BLOCKED_HINTS = [
    "source tbd",
    "requires",
    "external",
    "endpoint",
    "feed",
    "if exists",
    "not yet",
]


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


def main() -> int:
    text = TODO.read_text(encoding="utf-8")
    items = list(_iter_items(text))

    lines = []
    lines.append("# TODO_STATUS\n")
    lines.append("Generated from TODO.md. Edit TODO.md for canonical tasks; update this file by rerunning scripts/todo_status_gen.py.\n")
    lines.append("\n")

    for i, (item, is_done, raw) in enumerate(items, start=1):
        low = item.lower()
        blocked = (not is_done) and any(h in low for h in BLOCKED_HINTS)
        status = "DONE" if is_done else ("BLOCKED" if blocked else "IN-PROGRESS")
        lines.append(f"## {i}. {item}\n")
        lines.append(f"- STATUS: {status}\n")
        if is_done:
            lines.append("- NEXT: (none)\n")
            lines.append("- BLOCKERS: none\n")
        else:
            lines.append("- NEXT: (fill)\n")
            if blocked:
                lines.append("- BLOCKERS: (fill decision/provider/endpoint)\n")
            else:
                lines.append("- BLOCKERS: none (scaffoldable)\n")
        lines.append("\n")

    OUT.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote: {OUT}")
    print(f"Items: {len(items)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
