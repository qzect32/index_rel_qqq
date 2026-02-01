from __future__ import annotations

import argparse

from .build import refresh_universe
from .build_universe import refresh_etf_universe
from .build_prices import refresh_prices


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="etf-mapper")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("refresh", help="Refresh Nasdaq-100 exposure graph (issuer leveraged/inverse seed)")
    r.add_argument("--out", default="data", help="Output directory")

    u = sub.add_parser("universe", help="Fetch full US ETF universe (master list of ETF tickers)")
    u.add_argument("--out", default="data", help="Output directory")
    u.add_argument("--provider", default="polygon", choices=["polygon"], help="Universe provider")

    pr = sub.add_parser("prices", help="Fetch daily price history for ETFs (bootstrap providers)")
    pr.add_argument("--out", default="data", help="Output directory")
    pr.add_argument(
        "--universe",
        default="data/etf_universe.parquet",
        help="Path to etf_universe.parquet (or CSV with ticker column)",
    )
    pr.add_argument("--provider", default="yahoo", choices=["yahoo", "stooq"], help="Price provider")
    pr.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    pr.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    pr.add_argument("--limit", type=int, default=200, help="Max tickers to fetch (for quick runs)")

    args = p.parse_args(argv)

    if args.cmd == "refresh":
        outputs = refresh_universe(args.out)
        for k, v in outputs.items():
            print(f"{k}: {v}")
        return 0

    if args.cmd == "universe":
        outputs = refresh_etf_universe(args.out, provider=args.provider)
        for k, v in outputs.items():
            print(f"{k}: {v}")
        return 0

    if args.cmd == "prices":
        outputs = refresh_prices(
            args.out,
            universe_path=args.universe,
            provider=args.provider,
            start=args.start,
            end=args.end,
            limit=args.limit,
        )
        for k, v in outputs.items():
            print(f"{k}: {v}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
