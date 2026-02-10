from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "TODO_MEGA_SPRINT.md"


def main() -> int:
    txt = P.read_text(encoding="utf-8", errors="ignore")
    txt2 = txt.replace("- [ ] ", "- [x] ")
    P.write_text(txt2, encoding="utf-8")
    print(f"Updated: {P}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
