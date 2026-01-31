from __future__ import annotations
import pandas as pd

def ndx_index_etfs() -> pd.DataFrame:
    """Small curated set of Nasdaq-100 index ETFs (base + common leveraged/inverse)."""
    rows = [
        {"etf_ticker": "QQQ", "etf_name": "Invesco QQQ Trust", "underlying_symbol": "^NDX", "daily_target": 1.0, "relationship": "index", "issuer": "Invesco"},
        {"etf_ticker": "QLD", "etf_name": "ProShares Ultra QQQ", "underlying_symbol": "^NDX", "daily_target": 2.0, "relationship": "leveraged_long", "issuer": "ProShares"},
        {"etf_ticker": "TQQQ", "etf_name": "ProShares UltraPro QQQ", "underlying_symbol": "^NDX", "daily_target": 3.0, "relationship": "leveraged_long", "issuer": "ProShares"},
        {"etf_ticker": "PSQ", "etf_name": "ProShares Short QQQ", "underlying_symbol": "^NDX", "daily_target": -1.0, "relationship": "inverse", "issuer": "ProShares"},
        {"etf_ticker": "QID", "etf_name": "ProShares UltraShort QQQ", "underlying_symbol": "^NDX", "daily_target": -2.0, "relationship": "inverse", "issuer": "ProShares"},
        {"etf_ticker": "SQQQ", "etf_name": "ProShares UltraPro Short QQQ", "underlying_symbol": "^NDX", "daily_target": -3.0, "relationship": "inverse", "issuer": "ProShares"},
    ]
    return pd.DataFrame(rows)
