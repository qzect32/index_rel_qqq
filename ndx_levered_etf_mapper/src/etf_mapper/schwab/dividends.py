from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class DividendInfo:
    ex_dividend_date: Optional[str] = None  # YYYY-MM-DD
    cash_amount: Optional[float] = None
    frequency: Optional[str] = None
    source: str = "unknown"


def extract_ex_dividend(js: Any) -> DividendInfo:
    """Best-effort extraction.

    Schwab's exact schema may vary by endpoint; this function is defensive.
    """
    if not isinstance(js, dict):
        return DividendInfo()

    # Try common patterns
    for k in ["exDividendDate", "ex_dividend_date", "ex_div_date", "exDate", "ex_date"]:
        if js.get(k):
            return DividendInfo(ex_dividend_date=str(js.get(k))[:10], source="schwab")

    # Sometimes nested
    for container in ["fundamental", "fundamentals", "dividend", "dividends"]:
        sub = js.get(container)
        if isinstance(sub, dict):
            for k in ["exDividendDate", "ex_dividend_date", "exDate", "ex_date"]:
                if sub.get(k):
                    amt = sub.get("amount") or sub.get("cashAmount")
                    try:
                        amt_f = float(amt) if amt is not None else None
                    except Exception:
                        amt_f = None
                    return DividendInfo(ex_dividend_date=str(sub.get(k))[:10], cash_amount=amt_f, source="schwab")

    return DividendInfo()
