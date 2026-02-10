"""Generate TODO_MEGA_SPRINT_STATUS.md from TODO_MEGA_SPRINT.md.

Usage:
  python scripts/todo_mega_status_gen.py
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TODO = ROOT / "TODO_MEGA_SPRINT.md"
OUT = ROOT / "TODO_MEGA_SPRINT_STATUS.md"


def _iter_items(text: str):
    for ln in text.splitlines():
        m = re.match(r"^\s*- \[( |x)\] (.+)$", ln)
        if m:
            yield (m.group(1) or " ") == "x", m.group(2).strip()


def main() -> int:
    if not TODO.exists():
        raise SystemExit(f"Missing: {TODO}")

    items = list(_iter_items(TODO.read_text(encoding="utf-8", errors="ignore")))

    lines = []
    lines.append("# TODO_MEGA_SPRINT_STATUS\n")
    lines.append("Generated from TODO_MEGA_SPRINT.md. Edit TODO_MEGA_SPRINT.md; rerun scripts/todo_mega_status_gen.py.\n\n")

    for i, (done, title) in enumerate(items, start=1):
        status = "DONE" if done else "IN-PROGRESS"
        lines.append(f"## {i}. {title}\n")
        lines.append("- TYPE: MEGA\n")
        lines.append(f"- STATUS: {status}\n")
        lines.append("- NEXT: (fill)\n" if not done else "- NEXT: (none)\n")
        lines.append("- BLOCKERS: none\n\n")

    OUT.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote: {OUT} ({len(items)} items)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
