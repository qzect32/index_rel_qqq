from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path
from typing import Iterable


def _take_last_lines(path: Path, max_lines: int = 2000) -> list[str]:
    """Best-effort tail reader.

    Debug logs can get large; reading the entire file just to take the last few
    lines makes the UI feel sluggish and can spike memory.

    This reads from the end in chunks until we have enough newlines.
    """
    if not path.exists():
        return []

    max_lines = int(max_lines or 0)
    if max_lines <= 0:
        return []

    try:
        # Read enough bytes from the end to cover max_lines newlines.
        # Heuristic: average ~200 bytes/line for JSONL; cap chunk growth.
        chunk_size = 64 * 1024
        max_read = 2 * 1024 * 1024  # safety cap

        data = b""
        with path.open("rb") as f:
            f.seek(0, 2)
            end = f.tell()
            pos = end
            while pos > 0 and data.count(b"\n") <= (max_lines + 5) and len(data) < max_read:
                step = min(chunk_size, pos)
                pos -= step
                f.seek(pos)
                data = f.read(step) + data

        text = data.decode("utf-8", errors="ignore")
        lines = text.splitlines()
        return lines[-max_lines:]
    except Exception:
        return []


def create_debug_bundle(data_dir: Path, extra_paths: Iterable[Path] = ()) -> Path:
    data_dir = Path(data_dir)
    out_dir = data_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    # User-facing artifact name: normalize to the product name.
    # (Avoid legacy/internal names in artifacts users will share.)
    bundle = out_dir / f"Market Hub debug bundle {stamp}.zip"

    # Prefer product-named logs; fall back to legacy filenames if needed.
    session = out_dir / "market_hub_session.jsonl"
    errors = out_dir / "market_hub_errors.jsonl"
    if not session.exists():
        session = out_dir / "spade_session.jsonl"
    if not errors.exists():
        errors = out_dir / "spade_errors.jsonl"

    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
        # last N lines only (internal file names may still be spade_*.jsonl; archive names are normalized)
        z.writestr("Market Hub session.tail.jsonl", "\n".join(_take_last_lines(session)) + "\n")
        z.writestr("Market Hub errors.tail.jsonl", "\n".join(_take_last_lines(errors)) + "\n")

        # sanitized presence snapshot
        snap = {
            "data_dir": str(data_dir.resolve()),
            "tokens_present": (data_dir / "schwab_tokens.json").exists(),
            "secrets_present": (data_dir / "schwab_secrets.local.json").exists(),
            "last_code_present": (data_dir / "schwab_last_code.txt").exists(),
            "decisions_present": (data_dir / "decisions.json").exists(),
            "settings_present": (data_dir / "app_settings.json").exists(),
        }
        z.writestr("snapshot.json", json.dumps(snap, indent=2))

        # Include a few small, non-secret local state files to help reproduce UX.
        # (Never include schwab_tokens.json or schwab_secrets.local.json.)
        for fp in [
            data_dir / "app_settings.json",
            data_dir / "decisions.json",
        ]:
            try:
                fp = Path(fp)
                if not (fp.exists() and fp.is_file()):
                    continue
                if fp.stat().st_size > 200_000:
                    # Avoid accidentally zipping huge files.
                    continue
                z.writestr(fp.name, fp.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                pass

        for p in extra_paths:
            try:
                if p.exists() and p.is_file():
                    z.write(p, arcname=str(p.name))
            except Exception:
                pass

    return bundle
