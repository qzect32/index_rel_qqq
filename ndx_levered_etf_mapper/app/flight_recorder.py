from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


REDACT_KEYS = {
    "access_token",
    "refresh_token",
    "client_secret",
    "SCHWAB_CLIENT_SECRET",
    "code",
}


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()[:12]


def redact_value(key: str, value: Any) -> Any:
    k = str(key).lower()
    if k in REDACT_KEYS or "token" in k or "secret" in k:
        if value is None:
            return None
        s = str(value)
        if len(s) <= 8:
            return {"redacted": True, "len": len(s)}
        return {
            "redacted": True,
            "len": len(s),
            "prefix": s[:4],
            "suffix": s[-4:],
            "sha": _sha(s),
        }
    return value


def redact_obj(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: redact_value(k, redact_obj(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_obj(x) for x in obj]
    return obj


@dataclass
class FlightRecorder:
    data_dir: Path
    session_id: str
    max_bytes: int = 5_000_000
    keep_files: int = 5

    def __post_init__(self):
        self.log_dir = (self.data_dir / "logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session_path = self.log_dir / "spade_session.jsonl"
        self.error_path = self.log_dir / "spade_errors.jsonl"

    def _rotate_if_needed(self, path: Path):
        try:
            if path.exists() and path.stat().st_size > self.max_bytes:
                stamp = time.strftime("%Y%m%d_%H%M%S")
                rotated = path.with_name(path.stem + f"_{stamp}" + path.suffix)
                path.replace(rotated)

                # prune
                files = sorted(path.parent.glob(path.stem + "_*" + path.suffix), key=lambda p: p.stat().st_mtime, reverse=True)
                for p in files[self.keep_files:]:
                    try:
                        p.unlink(missing_ok=True)
                    except Exception:
                        pass
        except Exception:
            pass

    def event(self, kind: str, payload: dict[str, Any]):
        self._rotate_if_needed(self.session_path)
        rec = {
            "ts": _now_iso(),
            "session": self.session_id,
            "kind": kind,
            **redact_obj(payload),
        }
        self.session_path.parent.mkdir(parents=True, exist_ok=True)
        with self.session_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def error(self, where: str, err: Exception, extra: Optional[dict[str, Any]] = None):
        self._rotate_if_needed(self.error_path)
        rec = {
            "ts": _now_iso(),
            "session": self.session_id,
            "where": where,
            "type": type(err).__name__,
            "message": str(err),
            "fingerprint": _sha(type(err).__name__ + ":" + str(err)),
            "extra": redact_obj(extra or {}),
        }
        with self.error_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def make_session_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S") + "_" + _sha(str(time.time()))
