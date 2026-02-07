from __future__ import annotations

from pathlib import Path
import json


def _state_path(data_dir: Path) -> Path:
    return (Path(data_dir).resolve() / "feeds_cache" / "news_rss_state.json").resolve()


def load_state(data_dir: Path) -> dict:
    p = _state_path(Path(data_dir))
    if not p.exists():
        return {"fail_counts": {}}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            return {"fail_counts": {}}
        obj.setdefault("fail_counts", {})
        if not isinstance(obj.get("fail_counts"), dict):
            obj["fail_counts"] = {}
        return obj
    except Exception:
        return {"fail_counts": {}}


def save_state(data_dir: Path, state: dict) -> None:
    p = _state_path(Path(data_dir))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2), encoding="utf-8")
