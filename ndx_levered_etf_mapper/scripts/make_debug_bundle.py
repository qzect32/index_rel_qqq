"""Create a Market Hub debug bundle zip.

Usage:
  python scripts/make_debug_bundle.py

Writes to: data/logs/debug_bundle_<timestamp>.zip
"""

from __future__ import annotations

from pathlib import Path

# Ensure repo root is importable when running as a script
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app.debug_bundle import create_debug_bundle  # noqa: E402


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    data_dir = repo / "data"

    extra = [
        data_dir / "decisions.json",
        data_dir / "decisions_log.jsonl",
        data_dir / "app_settings.json",
        data_dir / "rss_feeds.json",
        repo / "TODO_MEGA_SPRINT.md",
        repo / "TODO_MEGA_SPRINT_STATUS.md",
        repo / "TODO_SPRINT.md",
        repo / "TODO_SPRINT_STATUS.md",
    ]

    out = create_debug_bundle(data_dir, extra_paths=extra)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
