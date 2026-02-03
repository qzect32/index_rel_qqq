from __future__ import annotations

"""Options probe (Yahoo/yfinance) with log artifacts.

Goal: quickly diagnose which tickers return expirations and which fail to load chains.

Default behavior:
- Sample 25 Nasdaq-100 constituents (deterministic: first N sorted)
- Include extra tickers: SPY, IWM, TL, MARA
- Write:
  - JSONL log (one JSON per ticker)
  - CSV summary

Usage:
  python scripts/options_probe.py --data-dir data --sample 25 --include SPY,IWM,TL,MARA

Notes:
- This is best-effort; Yahoo can be rate-limited or temporarily unavailable.
- Retries/backoff included.
"""

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable
import json

import pandas as pd
import yfinance as yf


def _normalize_ticker(ticker: str) -> str:
    return str(ticker or "").upper().strip()


def _qqq_constituents_from_local(data_dir: Path) -> list[str]:
    candidates: list[str] = []
    p = data_dir / "equities.parquet"
    if p.exists():
        try:
            df = pd.read_parquet(p)
            for col in ["ticker", "symbol", "Ticker", "Symbol"]:
                if col in df.columns:
                    candidates = df[col].dropna().astype(str).tolist()
                    break
        except Exception:
            candidates = []

    p2 = data_dir / "nasdaq100_constituents.parquet"
    if (not candidates) and p2.exists():
        try:
            df = pd.read_parquet(p2)
            for col in ["ticker", "symbol", "Ticker", "Symbol"]:
                if col in df.columns:
                    candidates = df[col].dropna().astype(str).tolist()
                    break
        except Exception:
            candidates = []

    out = sorted({_normalize_ticker(x) for x in candidates if _normalize_ticker(x)})
    return out


def _options_probe_one(ticker: str, retries: int = 2) -> dict:
    tkr = _normalize_ticker(ticker)
    out: dict = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "ticker": tkr,
        "has_expirations": False,
        "n_expirations": 0,
        "first_exp": None,
        "has_chain": False,
        "calls": 0,
        "puts": 0,
        "error": None,
    }

    if not tkr:
        out["error"] = "empty ticker"
        return out

    last_err = None
    for i in range(retries + 1):
        try:
            t = yf.Ticker(tkr)
            exps = list(getattr(t, "options", []) or [])
            exps = [str(x) for x in exps if x]
            out["n_expirations"] = len(exps)
            out["has_expirations"] = len(exps) > 0
            out["first_exp"] = exps[0] if exps else None

            if not exps:
                return out

            oc = t.option_chain(exps[0])
            calls = oc.calls
            puts = oc.puts
            out["calls"] = int(len(calls)) if calls is not None else 0
            out["puts"] = int(len(puts)) if puts is not None else 0
            out["has_chain"] = out["calls"] > 0 or out["puts"] > 0
            return out
        except Exception as e:
            last_err = str(e)
            out["error"] = last_err
            time.sleep(0.4 + 0.6 * i)

    out["error"] = last_err
    return out


def _parse_include(val: str) -> list[str]:
    if not val:
        return []
    parts = [p.strip() for p in val.split(",")]
    return [_normalize_ticker(p) for p in parts if _normalize_ticker(p)]


def _write_artifacts(rows: list[dict], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = out_dir / f"options_probe_{stamp}.jsonl"
    csv_path = out_dir / f"options_probe_{stamp}.csv"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return jsonl_path, csv_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--sample", type=int, default=25, help="How many Nasdaq-100 constituents to probe")
    ap.add_argument("--include", default="SPY,IWM,TL,MARA", help="Comma-separated extra tickers")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--out-dir", default="data/logs")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    ndx = _qqq_constituents_from_local(data_dir)
    sample_n = max(0, int(args.sample))
    ndx_sample = ndx[:sample_n]

    tickers = sorted(set(ndx_sample + _parse_include(args.include)))

    rows: list[dict] = []
    for i, tkr in enumerate(tickers, start=1):
        r = _options_probe_one(tkr, retries=int(args.retries))
        rows.append(r)
        if i % 5 == 0 or i == len(tickers):
            print(f"{i}/{len(tickers)} probed")

    jsonl_path, csv_path = _write_artifacts(rows, out_dir)

    # Print quick summary
    df = pd.DataFrame(rows)
    bad = df[(~df["has_expirations"]) | (~df["has_chain"])].copy()
    print("wrote", jsonl_path)
    print("wrote", csv_path)
    print("--- summary ---")
    print(df[["has_expirations", "has_chain"]].value_counts(dropna=False))
    if not bad.empty:
        print("--- failures (top 25) ---")
        print(bad[["ticker", "has_expirations", "n_expirations", "first_exp", "has_chain", "calls", "puts", "error"]].head(25).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
