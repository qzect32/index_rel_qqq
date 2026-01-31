from __future__ import annotations
import argparse
from .build import refresh_universe

def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="etf-mapper")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("refresh", help="Refresh constituents + leveraged ETF universe")
    r.add_argument("--out", default="data", help="Output directory")

    args = p.parse_args(argv)

    if args.cmd == "refresh":
        outputs = refresh_universe(args.out)
        for k, v in outputs.items():
            print(f"{k}: {v}")
        return 0

    return 1

if __name__ == "__main__":
    raise SystemExit(main())
