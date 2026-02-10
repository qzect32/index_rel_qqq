from __future__ import annotations

import json
from typing import Any


def safe_snip(x: Any, max_chars: int = 1200) -> str:
    try:
        s = json.dumps(x, default=str, ensure_ascii=False, indent=2)
    except Exception:
        s = str(x)
    if len(s) > max_chars:
        return s[:max_chars] + "\n…(truncated)…"
    return s
