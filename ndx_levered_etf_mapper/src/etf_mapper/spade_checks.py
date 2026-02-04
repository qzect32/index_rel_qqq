from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd


@dataclass(frozen=True)
class Check:
    level: str  # INFO|WARN|FAIL
    code: str
    message: str
    details: dict[str, Any] | None = None


def _is_num(x: Any) -> bool:
    try:
        return pd.notna(x) and float(x) == float(x)
    except Exception:
        return False


def check_option_chain(calls: pd.DataFrame, puts: pd.DataFrame) -> list[Check]:
    out: list[Check] = []

    if (calls is None or calls.empty) and (puts is None or puts.empty):
        out.append(Check("FAIL", "CHAIN_EMPTY", "Both calls and puts are empty"))
        return out

    for side_name, df in [("calls", calls), ("puts", puts)]:
        if df is None or df.empty:
            out.append(Check("WARN", f"{side_name.upper()}_EMPTY", f"{side_name} dataframe is empty"))
            continue

        # Basic required columns
        for col in ["strike", "bid", "ask"]:
            if col not in df.columns:
                out.append(Check("FAIL", f"MISSING_{col.upper()}", f"Missing column '{col}' in {side_name}"))

        # Bid/ask sanity
        if "bid" in df.columns and "ask" in df.columns:
            bad = df[df.apply(lambda r: _is_num(r.get("bid")) and _is_num(r.get("ask")) and float(r.get("bid")) > float(r.get("ask")), axis=1)]
            if not bad.empty:
                out.append(
                    Check(
                        "WARN",
                        "BID_GT_ASK",
                        f"Found {len(bad)} rows where bid > ask in {side_name}",
                        details={"n": int(len(bad))},
                    )
                )

        # Extremely wide spreads
        if "bid" in df.columns and "ask" in df.columns:
            def _spread_ratio(r) -> float | None:
                try:
                    b = float(r.get("bid"))
                    a = float(r.get("ask"))
                    if a <= 0:
                        return None
                    return (a - b) / a
                except Exception:
                    return None

            ratios = df.apply(_spread_ratio, axis=1)
            if ratios.notna().any():
                # flag if many are > 50%
                wide = int((ratios > 0.50).sum())
                if wide >= max(10, int(0.10 * len(df))):
                    out.append(
                        Check(
                            "WARN",
                            "WIDE_SPREADS",
                            f"Many contracts have very wide bid/ask spreads in {side_name}",
                            details={"wide": wide, "rows": int(len(df))},
                        )
                    )

    return out


def check_price_history(df: pd.DataFrame) -> list[Check]:
    out: list[Check] = []
    if df is None or df.empty:
        out.append(Check("FAIL", "HISTORY_EMPTY", "Price history is empty"))
        return out

    for col in ["date", "open", "high", "low", "close"]:
        if col not in df.columns:
            out.append(Check("FAIL", f"MISSING_{col.upper()}", f"Missing column '{col}'"))

    if set(["open", "high", "low", "close"]).issubset(df.columns):
        bad = df[df["low"] > df["high"]]
        if not bad.empty:
            out.append(Check("WARN", "LOW_GT_HIGH", f"Found {len(bad)} rows where low > high"))

    return out


def summarize(checks: Iterable[Check]) -> dict[str, Any]:
    checks = list(checks)
    worst = "OK"
    if any(c.level == "FAIL" for c in checks):
        worst = "FAIL"
    elif any(c.level == "WARN" for c in checks):
        worst = "WARN"

    return {
        "status": worst,
        "n": len(checks),
        "fail": sum(1 for c in checks if c.level == "FAIL"),
        "warn": sum(1 for c in checks if c.level == "WARN"),
        "info": sum(1 for c in checks if c.level == "INFO"),
        "checks": [c.__dict__ for c in checks],
    }
