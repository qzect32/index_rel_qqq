from __future__ import annotations
import pandas as pd
from .base import ProviderResult

class HoldingsStubProvider:
    """Placeholder for holdings-based discovery.

    This is where the 'every ETF that holds NVDA/TSLA/etc' expansion will live.
    For now it returns empty frames but keeps the pipeline/schema ready.
    """
    name = "holdings_stub"

    def fetch(self) -> ProviderResult:
        etfs = pd.DataFrame(columns=["ticker","name","issuer","asset_class","strategy_group","theme_tags","source"])
        edges = pd.DataFrame(columns=[
            "src","dst","edge_type","direction","leverage_multiple",
            "relationship_group","weight","asof","source_url"
        ])
        return ProviderResult(etfs=etfs, edges=edges)
