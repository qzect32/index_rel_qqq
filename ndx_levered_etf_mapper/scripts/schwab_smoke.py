"""Schwab Market Data smoke test.

Usage:
  python scripts/schwab_smoke.py --symbol QQQ

Reads Schwab OAuth config the same way the app does:
  - data/schwab_secrets.local.json (recommended)
  - env vars fallback

Prints:
  - quote latency + best-effort quote timestamp
  - pricehistory latency + last candle timestamp

This is intentionally minimal and safe (read-only endpoints only).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from etf_mapper.config import load_schwab_secrets
from etf_mapper.schwab import SchwabAPI, SchwabConfig


def _api(data_dir: str = "data") -> SchwabAPI:
    secrets = load_schwab_secrets(Path(data_dir))
    if secrets is None:
        raise SystemExit("Schwab secrets not configured. Create data/schwab_secrets.local.json")

    return SchwabAPI(
        SchwabConfig(
            client_id=secrets.client_id,
            client_secret=secrets.client_secret,
            redirect_uri=secrets.redirect_uri,
            token_path=secrets.token_path,
        )
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="QQQ")
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()

    sym = str(args.symbol).upper().strip()
    api = _api(args.data_dir)

    t0 = time.perf_counter()
    q = api.quotes([sym])
    t1 = time.perf_counter()

    rec = q.get(sym) if isinstance(q, dict) else None
    if not isinstance(rec, dict):
        rec = q if isinstance(q, dict) else {}

    qt_ms = rec.get("quoteTimeInLong") or rec.get("tradeTimeInLong")
    qt = None
    if isinstance(qt_ms, (int, float)) and qt_ms:
        qt = pd.to_datetime(int(qt_ms), unit="ms", utc=True).tz_convert(None)

    print(f"quotes latency: {(t1 - t0) * 1000:.0f} ms")
    print(f"lastPrice={rec.get('lastPrice')} mark={rec.get('mark')} quote_ts={qt}")

    t2 = time.perf_counter()
    h = api.price_history(
        sym,
        period_type="day",
        period=3,
        frequency_type="minute",
        frequency=1,
        need_extended_hours_data=True,
    )
    t3 = time.perf_counter()

    candles = (h or {}).get("candles") or []
    last = None
    if candles:
        last_dt = candles[-1].get("datetime")
        if last_dt is not None:
            last = pd.to_datetime(int(last_dt), unit="ms", utc=True).tz_convert(None)

    print(f"pricehistory latency: {(t3 - t2) * 1000:.0f} ms")
    print(f"candles={len(candles)} last_candle_ts={last}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
