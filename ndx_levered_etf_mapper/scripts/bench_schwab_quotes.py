"""Benchmark Schwab quotes latency (Python).

Uses existing OAuth token file (data/schwab_tokens.json) via etf_mapper Schwab client.

Rate limit note:
- Schwab request budget is global. Keep iterations low or sleep between calls.

Usage:
  python scripts/bench_schwab_quotes.py --symbols SPY,QQQ,TSLA --iters 20 --sleep-ms 3000
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from etf_mapper.config import load_schwab_secrets
from etf_mapper.schwab.client import SchwabAPI, SchwabConfig


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs2 = sorted(xs)
    k = (len(xs2) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs2) - 1)
    if f == c:
        return xs2[f]
    return xs2[f] * (c - k) + xs2[c] * (k - f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="SPY,QQQ,TSLA,AAPL,NVDA", help="comma-separated")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--sleep-ms", type=int, default=3000)
    args = ap.parse_args()

    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    data_dir = Path("data")
    secrets = load_schwab_secrets(data_dir)
    if secrets is None:
        raise SystemExit("Missing Schwab secrets.")

    api = SchwabAPI(
        SchwabConfig(
            client_id=secrets.client_id,
            client_secret=secrets.client_secret,
            redirect_uri=secrets.redirect_uri,
            token_path=secrets.token_path,
        )
    )

    lats_ms: list[float] = []
    failures = 0

    for i in range(args.iters):
        t0 = time.perf_counter()
        ok = True
        try:
            _ = api.quotes(syms)
        except Exception:
            ok = False
        t1 = time.perf_counter()
        dt = (t1 - t0) * 1000.0
        if ok:
            lats_ms.append(dt)
        else:
            failures += 1
        time.sleep(max(0.0, args.sleep_ms / 1000.0))

    out = {
        "lang": "python",
        "iters": args.iters,
        "ok": len(lats_ms),
        "fail": failures,
        "symbols": syms,
        "lat_ms": {
            "min": min(lats_ms) if lats_ms else None,
            "mean": statistics.mean(lats_ms) if lats_ms else None,
            "p50": pct(lats_ms, 0.50) if lats_ms else None,
            "p95": pct(lats_ms, 0.95) if lats_ms else None,
            "max": max(lats_ms) if lats_ms else None,
        },
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
