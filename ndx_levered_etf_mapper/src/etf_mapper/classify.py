from __future__ import annotations
import re
from typing import Any

LEV_RE = re.compile(r"(\d+(?:\.\d+)?)x", re.IGNORECASE)

def classify_etf(etf_name: str, daily_target: float | None) -> dict[str, Any]:
    """Classify an ETF by name + (optional) stated daily leverage target.

    Returns:
      - direction: long|short
      - leverage_multiple: numeric multiple (e.g., 1.25, 2, 3) or None
      - strategy_group: leveraged|inverse|options_covered_call|options_defined_outcome|long_short|plain|unknown
    """
    n = (etf_name or "").lower()

    leverage_multiple = None
    if daily_target is not None:
        leverage_multiple = abs(float(daily_target))
    else:
        m = LEV_RE.search(n)
        if m:
            leverage_multiple = float(m.group(1))

    if daily_target is not None and float(daily_target) < 0:
        direction = "short"
    elif any(k in n for k in [" bear", "short", "inverse", "ultrashort"]):
        direction = "short"
    else:
        direction = "long"

    # Strategy buckets (expand over time)
    if any(k in n for k in ["covered call", "buywrite", "call writing", "income", "yieldmax", "option income"]):
        strategy_group = "options_covered_call"
    elif any(k in n for k in ["buffer", "defined outcome", "outcome", "protect", "hedged", "structured outcome"]):
        strategy_group = "options_defined_outcome"
    elif any(k in n for k in ["long/short", "market neutral", "pairs"]):
        strategy_group = "long_short"
    elif leverage_multiple is not None and leverage_multiple > 1:
        strategy_group = "leveraged"
    elif direction == "short":
        strategy_group = "inverse"
    elif etf_name:
        strategy_group = "plain"
    else:
        strategy_group = "unknown"

    return {
        "direction": direction,
        "leverage_multiple": leverage_multiple,
        "strategy_group": strategy_group,
    }

def relationship_group(src: str, strategy_group):
    # strategy_group can be a scalar or a Series if duplicate column names slip through
    try:
        if hasattr(strategy_group, "iloc"):
            strategy_group = strategy_group.iloc[0] if len(strategy_group) else "unknown"
    except Exception:
        strategy_group = "unknown"

    if src == "^NDX":
        if strategy_group == "leveraged":
            return "index_leveraged"
        if strategy_group == "inverse":
            return "index_inverse"
        return "index_plain"

    if strategy_group == "leveraged":
        return "single_stock_leveraged"
    if strategy_group == "inverse":
        return "single_stock_inverse"
    if str(strategy_group).startswith("options_"):
        return "options_strategy"
    return "single_stock_plain"
