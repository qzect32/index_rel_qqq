"""Generate TODO_MEGA_SPRINT.md from TODO_STATUS_SCAFFOLDS.md.

Goal: give a single "make it shippable" checklist that can actually reach 100%.

Rules:
- Only include items that are not DONE in TODO_STATUS_SCAFFOLDS.md.
- Mark BLOCKED items as skipped (left unchecked but annotated).

Usage:
  python scripts/todo_mega_sprint_gen.py
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "TODO_STATUS_SCAFFOLDS.md"
OUT = ROOT / "TODO_MEGA_SPRINT.md"


def parse_items(text: str) -> list[dict]:
    items = []
    cur = None
    for ln in text.splitlines():
        m = re.match(r"^##\s+(\d+)\.\s+(.*)$", ln)
        if m:
            if cur:
                items.append(cur)
            cur = {"i": int(m.group(1)), "title": m.group(2).strip(), "status": "IN-PROGRESS"}
            continue
        if cur and ln.strip().startswith("- STATUS:"):
            cur["status"] = ln.split(":", 1)[1].strip().upper()
    if cur:
        items.append(cur)
    return items


def main() -> int:
    if not SRC.exists():
        raise SystemExit(f"Missing: {SRC}")

    items = parse_items(SRC.read_text(encoding="utf-8", errors="ignore"))
    open_items = [it for it in items if it.get("status") != "DONE"]

    lines: list[str] = []
    lines.append("# TODO_MEGA_SPRINT — Shippable scaffolds (auto)\n\n")
    lines.append("Generated from TODO_STATUS_SCAFFOLDS.md.\n")
    lines.append("This is a *do-the-whole-list* sprint file that can reach 100%.\n")
    lines.append("When something is inherently provider/endpoint-dependent, we still scaffold it and mark QA separately.\n\n")

    lines.append("## Open scaffold items\n")
    for it in open_items:
        i = it["i"]
        title = it["title"]
        st = it.get("status")
        note = ""
        if st == "BLOCKED":
            note = " **(BLOCKED — scaffold what we can; QA later)**"
        lines.append(f"- [ ] #{i} {title}{note}\n")

    OUT.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote: {OUT} ({len(open_items)} items)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
