from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path
from typing import Iterable


def _take_last_lines(path: Path, max_lines: int = 2000) -> list[str]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return lines[-max_lines:]
    except Exception:
        return []


def create_debug_bundle(data_dir: Path, extra_paths: Iterable[Path] = ()) -> Path:
    data_dir = Path(data_dir)
    out_dir = data_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    bundle = out_dir / f"debug_bundle_{stamp}.zip"

    session = out_dir / "spade_session.jsonl"
    errors = out_dir / "spade_errors.jsonl"

    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
        # last N lines only
        z.writestr("spade_session.tail.jsonl", "\n".join(_take_last_lines(session)) + "\n")
        z.writestr("spade_errors.tail.jsonl", "\n".join(_take_last_lines(errors)) + "\n")

        # sanitized presence snapshot
        snap = {
            "data_dir": str(data_dir.resolve()),
            "tokens_present": (data_dir / "schwab_tokens.json").exists(),
            "secrets_present": (data_dir / "schwab_secrets.local.json").exists(),
            "last_code_present": (data_dir / "schwab_last_code.txt").exists(),
        }
        z.writestr("snapshot.json", json.dumps(snap, indent=2))

        for p in extra_paths:
            try:
                if p.exists() and p.is_file():
                    z.write(p, arcname=str(p.name))
            except Exception:
                pass

    return bundle
