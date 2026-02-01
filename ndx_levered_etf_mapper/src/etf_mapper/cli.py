from __future__ import annotations

import argparse

from .build import refresh_universe
from .build_universe import refresh_etf_universe


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="etf-mapper")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("refresh", help="Refresh Nasdaq-100 exposure graph (issuer leveraged/inverse seed)")
    r.add_argument("--out", default="data", help="Output directory")

    u = sub.add_parser("universe", help="Fetch full US ETF universe (master list of ETF tickers)")
    u.add_argument("--out", default="data", help="Output directory")
    u.add_argument("--provider", default="polygon", choices=["polygon"], help="Universe provider")

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

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
