from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class Underlying:
    symbol: str
    name: str
    source: str  # e.g., "wikipedia_nasdaq100"

@dataclass(frozen=True)
class ETF:
    ticker: str
    name: str
    issuer: str
    daily_target: Optional[float]  # e.g. 2.0, -1.0, -2.0, 3.0; None if unknown
    benchmark: str  # free text; for single-stock this will include underlying symbol

@dataclass(frozen=True)
class Edge:
    underlying: str
    etf: str
    relationship: str  # e.g. "leveraged_long", "inverse", "covered_call"
    daily_target: Optional[float]
    source: str
