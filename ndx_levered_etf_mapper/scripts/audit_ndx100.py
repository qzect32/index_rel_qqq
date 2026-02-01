from __future__ import annotations

"""Audit Nasdaq-100 symbols for:
- options availability via yfinance
- relation edges in data/universe.sqlite (if present)

Usage:
  python scripts/audit_ndx100.py --data-dir data --out data/ndx100_audit.csv

Notes:
  - This uses Wikipedia for the Nasdaq-100 list.
  - yfinance is best-effort and will be slow for 100 tickers.
"""

import argparse
from pathlib import Path
import sqlite3

import pandas as pd
import yfinance as yf

from etf_mapper.sources.nasdaq100 import fetch_nasdaq100


def _edges_count(db_path: Path, symbol: str) -> tuple[int, int]:
    if not db_path.exists():
        return (0, 0)
    with sqlite3.connect(db_path) as conn:
        inbound = conn.execute(
            "select count(*) from edges where upper(dst)=upper(?)", (symbol,)
        ).fetchone()[0]
        outbound = conn.execute(
            "select count(*) from edges where upper(src)=upper(?)", (symbol,)
        ).fetchone()[0]
    return int(inbound), int(outbound)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="data/ndx100_audit.csv")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.out)

    ndx = fetch_nasdaq100()
    symbols = ndx["symbol"].astype(str).str.upper().str.strip().tolist()

    rel_db = data_dir / "universe.sqlite"

    rows = []
    for i, sym in enumerate(symbols, start=1):
        has_options = False
        exp_count = 0
        err = None
        try:
            t = yf.Ticker(sym)
            exps = list(getattr(t, "options", []) or [])
            exp_count = len(exps)
            has_options = exp_count > 0
        except Exception as e:
            err = str(e)

        in_edges, out_edges = _edges_count(rel_db, sym)

        rows.append(
            {
                "symbol": sym,
                "has_options": has_options,
                "expirations": exp_count,
                "in_edges": in_edges,
                "out_edges": out_edges,
                "yahoo_error": err,
            }
        )

        if i % 10 == 0:
            print(f"{i}/{len(symbols)} processed")

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print("wrote", out_path)
    print(df[["has_options", "in_edges", "out_edges"]].describe(include="all"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
