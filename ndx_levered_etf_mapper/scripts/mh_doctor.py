"""Market Hub doctor.

Offline-first sanity checks for when you git pull.

Usage:
  python scripts/mh_doctor.py
  python scripts/mh_doctor.py --json
  python scripts/mh_doctor.py --live   # optional: includes lightweight network checks

This script does NOT require Streamlit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def check_file(path: Path) -> dict:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size": (path.stat().st_size if path.exists() else None),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--live", action="store_true", help="Attempt optional live checks")
    args = ap.parse_args()

    data = {
        "root": str(ROOT),
        "python": sys.version,
        "cwd": os.getcwd(),
        "checks": [],
        "warnings": [],
    }

    # Repo structure
    for rel in [
        Path("app/streamlit_app.py"),
        Path("src"),
        Path("scripts/decisions_listener.py"),
        Path("scripts/schwab_smoke.py"),
    ]:
        kind = "dir" if (ROOT / rel).is_dir() else "file"
        data["checks"].append({"kind": kind, **check_file(ROOT / rel)})

    # Core package import check (internal module name stays internal)
    try:
        import importlib

        importlib.import_module("etf_mapper")
        data["checks"].append({"kind": "core_package", "ok": True})
    except Exception as e:
        data["checks"].append({"kind": "core_package", "ok": False, "error": str(e)})

    # Data dir + secrets/tokens
    data_dir = ROOT / "data"
    data["checks"].append({"kind": "dir", **check_file(data_dir)})

    secrets = data_dir / "schwab_secrets.local.json"
    tokens = data_dir / "schwab_tokens.json"
    decisions = data_dir / "decisions.json"

    for p in [secrets, tokens, decisions]:
        data["checks"].append({"kind": "file", **check_file(p)})

    # decisions.json quick parse
    if decisions.exists():
        try:
            obj = json.loads(decisions.read_text(encoding="utf-8"))
            schema_v = obj.get("schema_version")
            cats = obj.get("categories") if isinstance(obj.get("categories"), dict) else {}
            data["checks"].append({
                "kind": "decisions",
                "schema_version": schema_v,
                "categories": len(cats),
                "received_at": obj.get("_received_at"),
            })
        except Exception as e:
            data["warnings"].append(f"Could not parse decisions.json: {e}")

    # optional live check: import requests + hit local listener
    if args.live:
        try:
            import requests  # type: ignore

            url = "http://127.0.0.1:8765/status"
            r = requests.get(url, timeout=2)
            data["checks"].append({"kind": "listener", "url": url, "ok": bool(r.ok), "status": r.status_code})
        except Exception as e:
            data["warnings"].append(f"Listener live check failed: {e}")

    # basic warnings
    if not secrets.exists():
        data["warnings"].append("Missing data/schwab_secrets.local.json (OAuth won't work)")
    if secrets.exists() and not tokens.exists():
        data["warnings"].append("Secrets present but tokens missing (need OAuth exchange)")

    if args.json:
        print(json.dumps(data, indent=2))
    else:
        print("Market Hub doctor")
        print("root:", data["root"])
        print("python:", sys.version.split()[0])
        for c in data["checks"]:
            if c.get("kind") == "file":
                print(f"file: {c['path']} exists={c['exists']} size={c['size']}")
            elif c.get("kind") == "dir":
                print(f"dir : {c['path']} exists={c['exists']}")
            elif c.get("kind") == "core_package":
                print(f"core package import: ok={c.get('ok')}")
                if not c.get("ok") and c.get("error"):
                    print(f"  error: {c.get('error')}")
            elif c.get("kind") == "decisions":
                print(f"decisions: schema_version={c.get('schema_version')} categories={c.get('categories')} received_at={c.get('received_at')}")
            elif c.get("kind") == "listener":
                print(f"listener: {c.get('url')} ok={c.get('ok')} status={c.get('status')}")

        if data["warnings"]:
            print("\nWARNINGS")
            for w in data["warnings"]:
                print("-", w)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
