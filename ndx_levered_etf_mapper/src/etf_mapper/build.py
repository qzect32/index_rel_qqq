from __future__ import annotations

from pathlib import Path
import sqlite3
import math
from typing import Any

import networkx as nx
import pandas as pd

from .classify import classify_etf, relationship_group
from .providers.holdings_stub import HoldingsStubProvider
from .sources.nasdaq100 import fetch_nasdaq100
from .sources.direxion import fetch_direxion_single_stock
from .sources.graniteshares import fetch_graniteshares_single_stock
from .sources.ndx_index_etfs import ndx_index_etfs


def _clean_gexf_value(v: Any) -> Any:
    """GEXF does not allow None and is picky about NaN."""
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    return v


def refresh_universe(out_dir: str | Path) -> dict[str, Path]:
    """Build a local 'exposure graph' for Nasdaq-100 constituents.

    Current coverage:
      - equities: Nasdaq-100 constituents
      - derivative_exposure edges:
          * index-level ETFs (QQQ/TQQQ/SQQQ/etc)
          * single-stock leveraged/inverse ETFs from issuers we can access
      - holds edges: stubbed (pluggable provider; implemented later)

    Outputs:
      - equities.parquet
      - etfs.parquet
      - edges.parquet
      - graph.gexf
      - universe.sqlite
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    equities = fetch_nasdaq100()
    if "symbol" not in equities.columns:
        raise RuntimeError("Nasdaq-100 fetch must return a 'symbol' column")

    # Issuer single-stock derivative products
    direxion = fetch_direxion_single_stock()
    granite = fetch_graniteshares_single_stock()
    index_seed = ndx_index_etfs()

    print("Nasdaq-100 constituents:", len(equities))
    print("Direxion single-stock ETFs:", len(direxion))
    print("GraniteShares single-stock ETFs:", len(granite))
    print("Index ETFs (seed):", len(index_seed))

    # Normalize issuer frames to a common ETF schema
    frames = [direxion, granite, index_seed]
    frames = [f for f in frames if f is not None and not f.empty]
    etfs_raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if etfs_raw.empty:
        etfs_raw = pd.DataFrame(
            columns=[
                "etf_ticker",
                "etf_name",
                "underlying_symbol",
                "daily_target",
                "relationship",
                "issuer",
                "source_url",
            ]
        )

    etfs_raw["etf_ticker"] = etfs_raw["etf_ticker"].astype(str).str.upper().str.strip()
    etfs_raw["underlying_symbol"] = etfs_raw["underlying_symbol"].astype(str).str.upper().str.strip()

    # Classify ETFs (strategy_group, leverage_multiple, direction)
    cls = etfs_raw.apply(
        lambda r: classify_etf(r.get("etf_name", ""), r.get("daily_target", None)), axis=1
    )
    cls_df = (
        pd.DataFrame(list(cls))
        if len(etfs_raw)
        else pd.DataFrame(columns=["direction", "leverage_multiple", "strategy_group"])
    )
    etfs = pd.concat([etfs_raw.reset_index(drop=True), cls_df.reset_index(drop=True)], axis=1)
    # If any source already provided columns like strategy_group/direction/etc,
    # this concat can create duplicate column names. Keep the last occurrence.
    etfs = etfs.loc[:, ~etfs.columns.duplicated(keep="last")]

    # ETF master table (dedup by ticker)
    etf_master = (
        etfs.rename(columns={"etf_ticker": "ticker", "etf_name": "name"})
        .assign(asset_class="etf")
        .assign(theme_tags=None)
        .rename(columns={"source_url": "source"})
    )
    etf_master = etf_master[
        ["ticker", "name", "issuer", "asset_class", "strategy_group", "theme_tags", "source"]
    ].drop_duplicates(subset=["ticker"])

    # Edges: equities/index -> ETF
    edges = etfs.rename(columns={"underlying_symbol": "src", "etf_ticker": "dst"}).copy()
    edges["edge_type"] = "derivative_exposure"
    edges["relationship_group"] = edges.apply(
        lambda r: relationship_group(r["src"], r.get("strategy_group", "unknown")), axis=1
    )

    # Holdings columns reserved for later expansion
    edges["weight"] = None
    edges["asof"] = None

    edges = edges[
        [
            "src",
            "dst",
            "edge_type",
            "direction",
            "leverage_multiple",
            "relationship",
            "relationship_group",
            "daily_target",
            "issuer",
            "source_url",
            "etf_name",
            "strategy_group",
            "weight",
            "asof",
        ]
    ]

    # Holdings provider stub (placeholder for the 'every ETF that holds NVDA/TSLA/etc' expansion)
    holdings_provider = HoldingsStubProvider()
    hp = holdings_provider.fetch()
    if not hp.edges.empty:
        edges = pd.concat([edges, hp.edges], ignore_index=True)

    # Save parquet
    p_eq = out_dir / "equities.parquet"
    p_etfs = out_dir / "etfs.parquet"
    p_edges = out_dir / "edges.parquet"
    equities.to_parquet(p_eq, index=False)
    etf_master.to_parquet(p_etfs, index=False)
    edges.to_parquet(p_edges, index=False)

    # Backward-compatible filenames (keep existing notebook paths working)
    (out_dir / "nasdaq100_constituents.parquet").write_bytes(p_eq.read_bytes())
    (out_dir / "levered_etfs.parquet").write_bytes(p_etfs.read_bytes())

    # Graph (directed edges from src -> dst)
    G = nx.DiGraph()

    for _, r in equities.iterrows():
        G.add_node(r["symbol"], kind="equity", name=str(r.get("company", "") or ""))

    G.add_node("^NDX", kind="index", name="Nasdaq-100 Index")

    for _, r in etf_master.iterrows():
        G.add_node(
            r["ticker"],
            kind="etf",
            name=str(r.get("name", "") or ""),
            issuer=str(r.get("issuer", "") or ""),
        )

    # IMPORTANT: sanitize edge attributes to avoid NoneType in GEXF
    for _, r in edges.iterrows():
        attrs = {
            "edge_type": r.get("edge_type"),
            "relationship": r.get("relationship"),
            "relationship_group": r.get("relationship_group"),
            "direction": r.get("direction"),
            "leverage_multiple": r.get("leverage_multiple"),
            "daily_target": r.get("daily_target"),
            "issuer": r.get("issuer"),
            "strategy_group": r.get("strategy_group"),
            "weight": r.get("weight"),
            "asof": r.get("asof"),
            "source_url": r.get("source_url"),
        }
        attrs = {k: _clean_gexf_value(v) for k, v in attrs.items()}
        G.add_edge(r["src"], r["dst"], **attrs)

    p_graph = out_dir / "graph.gexf"
    nx.write_gexf(G, p_graph)

    # SQLite
    db_path = out_dir / "universe.sqlite"
    with sqlite3.connect(db_path) as conn:
        equities.to_sql("equities", conn, if_exists="replace", index=False)
        etf_master.to_sql("etfs", conn, if_exists="replace", index=False)
        edges.to_sql("edges", conn, if_exists="replace", index=False)

        # Backward compatible table names
        equities.to_sql("nasdaq100_constituents", conn, if_exists="replace", index=False)
        etf_master.to_sql("levered_etfs", conn, if_exists="replace", index=False)

    return {
        "equities": p_eq,
        "etfs": p_etfs,
        "edges": p_edges,
        "graph": p_graph,
        "sqlite": db_path,
    }
