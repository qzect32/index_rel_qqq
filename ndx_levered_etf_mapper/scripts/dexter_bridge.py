"""Dexter bridge (Market Hub)

Runs the Dexter (virattt/dexter) agent in non-interactive mode and captures output.

This is intentionally a *research-only* integration:
- No live trading orders.
- Dexter runs as an external process.

Prereqs:
- Bun installed (Dexter runtime)
- Dexter repo cloned locally (default: sibling folder "../dexter")
- API keys set via env or ~/.openclaw/.env

Usage:
  python scripts/dexter_bridge.py --query "Analyze AAPL earnings risk" --out-dir data

Exit codes:
  0 success
  2 bun missing
  3 dexter not found
  4 dexter run failed
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _which(exe: str) -> str | None:
    from shutil import which

    return which(exe)


def _read_dotenv(p: Path) -> dict[str, str]:
    if not p.exists():
        return {}
    out: dict[str, str] = {}
    for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k:
            out[k] = v
    return out


def _default_dexter_dir() -> Path:
    # Repo layout preference:
    #   ~/.openclaw/workspace/index_rel_qqq/ndx_levered_etf_mapper (this repo)
    #   ~/.openclaw/workspace/dexter (sibling clone)
    here = Path(__file__).resolve()
    repo = here.parents[1]
    sibling = repo.parents[1] / "dexter"
    return sibling


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--dexter-dir", default=os.getenv("DEXTER_DIR", ""))
    ap.add_argument("--timeout-s", type=int, default=600)
    args = ap.parse_args()

    bun = _which("bun")
    if not bun:
        print("ERROR: bun is not installed or not on PATH. Install from https://bun.sh", file=sys.stderr)
        return 2

    dexter_dir = Path(args.dexter_dir).expanduser() if args.dexter_dir else _default_dexter_dir()
    if not dexter_dir.exists() or not (dexter_dir / "package.json").exists():
        print(f"ERROR: dexter repo not found at: {dexter_dir}", file=sys.stderr)
        return 3

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / "dexter" / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y-%m-%d_%H%M%S", time.localtime())
    out_json = run_dir / f"dexter_{ts}.json"

    # Load OpenClaw-managed env (global fallback) if present.
    openclaw_env = Path.home() / ".openclaw" / ".env"
    env = dict(os.environ)
    for k, v in _read_dotenv(openclaw_env).items():
        env.setdefault(k, v)

    # Prefer running our non-interactive entrypoint if present.
    entry = dexter_dir / "src" / "run_once.ts"
    if not entry.exists():
        print(
            "ERROR: dexter non-interactive entrypoint missing: src/run_once.ts\n"
            "Run the Market Hub integration install step to add it.",
            file=sys.stderr,
        )
        return 4

    cmd = [bun, "run", str(entry), "--query", str(args.query), "--json"]

    p = subprocess.run(
        cmd,
        cwd=str(dexter_dir),
        env=env,
        capture_output=True,
        text=True,
        timeout=int(args.timeout_s),
    )

    if p.returncode != 0:
        sys.stderr.write(p.stdout)
        sys.stderr.write(p.stderr)
        return 4

    # Dexter runner prints JSON to stdout.
    try:
        obj = json.loads(p.stdout)
    except Exception:
        obj = {"raw": p.stdout}

    out_json.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({"ok": True, "out": str(out_json), "result": obj}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
