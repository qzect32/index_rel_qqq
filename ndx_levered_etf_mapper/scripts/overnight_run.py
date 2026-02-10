"""Overnight runner for Market Hub.

Goal: produce measurable progress artifacts (logs, bundles, status files) without
manual babysitting.

It does *not* place orders or touch secrets.

Usage:
  python scripts/overnight_run.py
  python scripts/overnight_run.py --commit

Outputs:
  - data/logs/overnight_<stamp>.log
  - data/logs/debug_bundle_<stamp>.zip
  - refreshed TODO status files

Commit mode:
  If --commit is provided, stages + commits generated artifacts (never secrets).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], *, cwd: Path | None = None, log: Path | None = None, check: bool = False) -> int:
    p = subprocess.run(cmd, cwd=str(cwd or ROOT), capture_output=True, text=True)
    out = (p.stdout or "") + ("\n" if p.stdout and p.stderr else "") + (p.stderr or "")
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text((log.read_text(encoding="utf-8", errors="ignore") if log.exists() else "") + f"\n$ {' '.join(cmd)}\n" + out, encoding="utf-8")
    if check and p.returncode != 0:
        raise SystemExit(p.returncode)
    return p.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true", help="Stage+commit generated artifacts")
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    log = ROOT / "data" / "logs" / f"overnight_{stamp}.log"

    # Basic info
    run([sys.executable, "-c", "import sys; print(sys.version)"] , log=log)
    run(["git", "rev-parse", "--short", "HEAD"], log=log)

    # Fast sanity checks
    run([sys.executable, "scripts/mh_doctor.py"], log=log)

    # Compile everything (cheap)
    run([sys.executable, "-m", "py_compile", "app/streamlit_app.py"], log=log)
    run([sys.executable, "-m", "py_compile", "scripts/decisions_listener.py"], log=log)

    # Refresh status trackers (best-effort)
    run([sys.executable, "scripts/todo_status_gen.py"], log=log)
    # NOTE: todo_mega_sprint_gen.py regenerates TODO_MEGA_SPRINT.md from scaffolds status
    # and will overwrite manual checkmarks. Default: do NOT regenerate the mega checklist.
    # Only refresh the mega STATUS file so the progress meter stays stable.
    run([sys.executable, "scripts/todo_mega_status_gen.py"], log=log)

    # Run tests (if any)
    run([sys.executable, "-m", "pytest", "-q"], log=log)

    # Debug bundle
    run([sys.executable, "scripts/make_debug_bundle.py"], log=log)

    # Git status snapshot
    run(["git", "status", "--porcelain"], log=log)

    if args.commit:
        # Never stage secrets/tokens
        exclude = {
            str((ROOT / "data" / "schwab_secrets.local.json").resolve()),
            str((ROOT / "data" / "schwab_tokens.json").resolve()),
        }

        # Stage safe artifacts
        paths = [
            ROOT / "TODO_STATUS.md",
            ROOT / "TODO_STATUS_SCAFFOLDS.md",
            ROOT / "TODO_STATUS_QA.md",
            ROOT / "TODO_MEGA_SPRINT.md",
            ROOT / "TODO_MEGA_SPRINT_STATUS.md",
        ]
        # log + any debug bundles produced
        paths += [log]
        paths += list((ROOT / "data" / "logs").glob("debug_bundle_*.zip"))

        add = []
        for p in paths:
            try:
                rp = str(p.resolve())
                if rp in exclude:
                    continue
                if p.exists():
                    add.append(str(p))
            except Exception:
                continue

        if add:
            run(["git", "add", "--"] + add, log=log)
            msg = f"Overnight runner artifacts ({stamp})"
            run(["git", "commit", "-m", msg], log=log)

    print(str(log))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
