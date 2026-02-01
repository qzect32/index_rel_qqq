from __future__ import annotations

from pathlib import Path
import os
import sqlite3
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from pyvis.network import Network
import yfinance as yf

from etf_mapper.build_universe import refresh_etf_universe
from etf_mapper.build_prices import refresh_prices
from etf_mapper.build import refresh_universe as refresh_relations


# ---------- App boot ----------
st.set_page_config(page_title="ETF Hub", layout="wide")

# Load local .env automatically (kept out of git)
load_dotenv()


def _data_dir() -> Path:
    return Path(st.sidebar.text_input("Data directory", value="data")).resolve()


def _db_path(name: str) -> Path:
    return _data_dir() / name


@st.cache_data(show_spinner=False)
def _read_sql(db_path: str, query: str, params: Optional[tuple] = None) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql(query, conn, params=params)


def _table_cols(db_path: Path, table: str) -> list[str]:
    with sqlite3.connect(db_path) as conn:
        return [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def _ensure_universe(data_dir: Path) -> Path:
    db = data_dir / "etf_universe.sqlite"
    if db.exists():
        return db

    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "Missing etf_universe.sqlite and POLYGON_API_KEY not set. "
            "Add POLYGON_API_KEY to a local .env file or environment variable."
        )

    with st.spinner("Fetching US ETF universe (Polygon)…"):
        refresh_etf_universe(data_dir, provider="polygon")
    return db


def _ensure_relations(data_dir: Path) -> Path:
    # relations come from the existing 'refresh' graph builder
    db = data_dir / "universe.sqlite"
    if db.exists():
        return db

    with st.spinner("Building relations graph (Nasdaq-100 exposure seed)…"):
        refresh_relations(data_dir)
    return db


def _ensure_prices(data_dir: Path, universe_parquet: Path, provider: str, start: str, limit: int) -> Path:
    db = data_dir / "prices.sqlite"
    if db.exists():
        return db

    with st.spinner(f"Fetching daily prices via {provider}…"):
        refresh_prices(
            data_dir,
            universe_path=universe_parquet,
            provider=provider,  # type: ignore[arg-type]
            start=start,
            end=None,
            limit=limit,
        )
    return db


def _load_universe(universe_db: Path) -> pd.DataFrame:
    df = _read_sql(str(universe_db), "select * from etf_universe")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    return df


def _load_prices(prices_db: Path, ticker: str) -> pd.DataFrame:
    # Guard for old malformed DBs
    cols = _table_cols(prices_db, "prices_daily")
    if "date" not in cols:
        raise RuntimeError(
            "prices_daily table is malformed (missing date column). Click 'Reset prices DB' then refetch."
        )

    q = """
    select date, open, high, low, close, adj_close, volume, source
    from prices_daily
    where ticker = ?
    order by date
    """
    df = _read_sql(str(prices_db), q, params=(ticker.upper().strip(),))
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])  # type: ignore
    return df


def _plot_candles(dfp: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=dfp["date"],
                open=dfp["open"],
                high=dfp["high"],
                low=dfp["low"],
                close=dfp["close"],
                name="OHLC",
            )
        ]
    )
    fig.update_layout(
        title=title,
        height=360,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_rangeslider_visible=False,
    )
    return fig


def _infer_intended_usage(name: str) -> str:
    n = (name or "").lower()
    hits = []
    if any(x in n for x in ["2x", "3x", "ultra", "leveraged", "bull"]):
        hits.append("leveraged exposure")
    if any(x in n for x in ["inverse", "bear", "short", "-1x", "-2x", "-3x"]):
        hits.append("inverse/short exposure")
    if "covered call" in n or "buywrite" in n:
        hits.append("income via covered calls")
    if "buffer" in n or "defined outcome" in n:
        hits.append("defined outcome / buffer")
    if not hits:
        hits.append("broad exposure / thematic allocation")
    return ", ".join(hits)


def _yahoo_profile(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    info = {}
    try:
        info = t.get_info() or {}
    except Exception:
        info = {}

    # Keep it stable & readable
    keys = [
        "longName",
        "shortName",
        "quoteType",
        "fundFamily",
        "category",
        "legalType",
        "currency",
        "exchange",
        "market",
        "website",
        "longBusinessSummary",
    ]
    out = {k: info.get(k) for k in keys if info.get(k) not in (None, "")}
    return out


def _yahoo_options_chain(ticker: str, expiration: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = yf.Ticker(ticker)
    oc = t.option_chain(expiration)
    calls = oc.calls.copy()
    puts = oc.puts.copy()
    # normalize
    for df, side in [(calls, "call"), (puts, "put")]:
        df["side"] = side
        df["ticker"] = ticker
        df["expiration"] = expiration
    return calls, puts


def _net_html(nodes: list[dict], edges: list[dict]) -> str:
    net = Network(height="520px", width="100%", bgcolor="#0b1220", font_color="#e5e7eb")
    net.barnes_hut(gravity=-24000, central_gravity=0.25, spring_length=190, spring_strength=0.02)

    for n in nodes:
        net.add_node(**n)
    for e in edges:
        net.add_edge(**e)

    return net.generate_html(notebook=False)


# ---------- UI ----------
st.title("ETF Hub")
st.caption("Universe • Relations • Prices • Options • Cart (Schwab/TOS integration scaffolded)")

data_dir = _data_dir()

# Sidebar controls
st.sidebar.markdown("### Data ops")
universe_provider = st.sidebar.selectbox("Universe provider", ["polygon"], index=0)
price_provider = st.sidebar.selectbox("Price provider", ["yahoo", "stooq"], index=0)
price_start = st.sidebar.text_input("Price start", value="2024-01-01")
price_limit = st.sidebar.slider("Price fetch limit", min_value=25, max_value=1000, value=200, step=25)

st.sidebar.markdown("### Secrets")
if not os.getenv("POLYGON_API_KEY"):
    st.sidebar.warning("POLYGON_API_KEY not detected. Add it to a local .env file.")

# Ensure universe exists (or explain why not)
try:
    universe_db = _ensure_universe(data_dir)
except Exception as e:
    st.error(str(e))
    st.info("Create a file named .env next to ndx_levered_etf_mapper/pyproject.toml with POLYGON_API_KEY=... (do not commit it).")
    st.stop()

universe = _load_universe(universe_db)

# Search + selection
with st.sidebar:
    st.markdown("### Explore")
    query = st.text_input("Search (ticker or name)", value="QQQ")

q = query.strip().lower()
view = universe
if q:
    mask = universe["ticker"].str.lower().str.contains(q) | universe["name"].fillna("").str.lower().str.contains(q)
    view = universe[mask].copy()

st.sidebar.caption(f"Universe rows (filtered): {len(view):,}")

# Pick ticker
default_ticker = "QQQ" if "QQQ" in set(universe["ticker"]) else str(universe.iloc[0]["ticker"])
selected = st.sidebar.text_input("Selected ticker", value=default_ticker).upper().strip()

# Top layout
tab_overview, tab_rel, tab_opts, tab_cart, tab_admin = st.tabs(
    ["Overview", "Relations", "Options", "Cart", "Admin"]
)

with tab_overview:
    colA, colB = st.columns([0.42, 0.58], gap="large")

    with colA:
        st.subheader("Universe")
        st.dataframe(
            view[["ticker", "name", "primary_exchange", "active"]].head(250),
            use_container_width=True,
            height=420,
        )

        meta = universe[universe["ticker"] == selected].head(1)
        if not meta.empty:
            st.markdown("#### ETF record")
            st.json(meta.iloc[0].dropna().to_dict())

        st.markdown("#### Intended usage (heuristic)")
        nm = str(meta.iloc[0]["name"]) if not meta.empty else selected
        st.write(_infer_intended_usage(nm))

        st.markdown("#### Provider description (Yahoo best-effort)")
        with st.spinner("Loading Yahoo profile…"):
            prof = _yahoo_profile(selected)
        if prof:
            # pull the summary out for nicer display
            summary = prof.pop("longBusinessSummary", None)
            st.json(prof)
            if summary:
                st.write(summary)
        else:
            st.caption("No Yahoo profile available.")

    with colB:
        st.subheader(f"Price chart: {selected}")

        universe_parquet = data_dir / "etf_universe.parquet"
        if not universe_parquet.exists():
            st.info("Universe parquet missing (should exist after universe fetch). Re-run universe or refresh page.")
            st.stop()

        prices_db = data_dir / "prices.sqlite"
        if not prices_db.exists():
            st.info("No prices DB yet.")
            if st.button("Fetch bootstrap prices now", type="primary"):
                _ensure_prices(data_dir, universe_parquet, provider=price_provider, start=price_start, limit=price_limit)
                st.rerun()
            st.stop()

        try:
            dfp = _load_prices(prices_db, selected)
        except Exception as e:
            st.error(str(e))
            if st.button("Reset prices DB", type="primary"):
                prices_db.unlink(missing_ok=True)
                st.success("Deleted prices.sqlite. Now fetch prices again.")
            st.stop()

        if dfp.empty:
            st.warning("No prices found for this ticker yet.")

            colX, colY = st.columns([0.34, 0.66])
            with colX:
                if st.button("Fetch THIS ticker now", type="primary"):
                    # Fetch on-demand for the selected ticker (no batch assumptions)
                    res = yf.Ticker(selected).history(start=price_start, end=None, interval="1d", auto_adjust=False)
                    if res is None or res.empty:
                        st.error("No data returned from Yahoo for this ticker.")
                    else:
                        df = res.reset_index()
                        if "Date" in df.columns:
                            df = df.rename(columns={"Date": "date"})
                        elif "Datetime" in df.columns:
                            df = df.rename(columns={"Datetime": "date"})
                        df = df.rename(
                            columns={
                                "Open": "open",
                                "High": "high",
                                "Low": "low",
                                "Close": "close",
                                "Adj Close": "adj_close",
                                "Volume": "volume",
                            }
                        )
                        df["ticker"] = selected
                        df["source"] = "yahoo:yfinance"
                        if pd.api.types.is_datetime64_any_dtype(df["date"]):
                            df["date"] = df["date"].dt.date.astype(str)
                        else:
                            df["date"] = df["date"].astype(str)

                        keep = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume", "source"]
                        for c in keep:
                            if c not in df.columns:
                                df[c] = None

                        with sqlite3.connect(prices_db) as conn:
                            # Append to existing DB
                            df[keep].to_sql("prices_daily", conn, if_exists="append", index=False)
                    st.rerun()

            with colY:
                if st.button("Fetch prices batch now", type="secondary"):
                    prices_db.unlink(missing_ok=True)
                    _ensure_prices(
                        data_dir,
                        universe_parquet,
                        provider=price_provider,
                        start=price_start,
                        limit=price_limit,
                    )
                    st.rerun()

            st.stop()

        st.plotly_chart(_plot_candles(dfp, title=f"{selected} OHLC"), use_container_width=True)

        with st.expander("Raw prices"):
            st.dataframe(dfp, use_container_width=True, height=320)

with tab_rel:
    st.subheader("Relationship map")
    st.caption("Current graph = seeded derivative exposures (index + single-stock leveraged/inverse ETFs). Holdings-based map is a future step.")

    try:
        rel_db = _ensure_relations(data_dir)
    except Exception as e:
        st.error(str(e))
        st.stop()

    edges = _read_sql(str(rel_db), "select * from edges")

    # Build local ego-network around selected ticker
    # If selected is an ETF, show its underlyings and sibling ETFs for those underlyings.
    nodes: dict[str, dict] = {}
    vis_edges: list[dict] = []

    def _add_node(i: str, label: str, group: str, color: str, size: int = 16):
        if i not in nodes:
            nodes[i] = {"n_id": i, "label": label, "group": group, "color": color, "size": size}

    # Normalize
    edges["src"] = edges["src"].astype(str)
    edges["dst"] = edges["dst"].astype(str)

    # inbound edges to ETF
    inbound = edges[edges["dst"].astype(str).str.upper() == selected.upper()]
    # if none, try outbound (selected underlying)
    outbound = edges[edges["src"].astype(str).str.upper() == selected.upper()]

    if inbound.empty and outbound.empty:
        st.info("No relations found for this ticker in the current graph seed.")
        st.stop()

    center = selected
    _add_node(center, label=center, group="center", color="#22c55e", size=24)

    underlyings = set(inbound["src"].tolist()) if not inbound.empty else set([center])
    if not inbound.empty:
        for _, r in inbound.iterrows():
            u = str(r["src"])
            rel = str(r.get("relationship", ""))
            strat = str(r.get("strategy_group", ""))
            _add_node(u, label=u, group="underlying", color="#60a5fa", size=20)
            vis_edges.append({"source": u, "to": center, "title": f"{rel} / {strat}"})

    # Add sibling ETFs for each underlying
    sib = edges[edges["src"].isin(list(underlyings))].copy()
    sib = sib[sib["dst"].astype(str).str.upper() != selected.upper()]
    sib = sib.head(120)

    for _, r in sib.iterrows():
        u = str(r["src"])
        e = str(r["dst"])
        rel = str(r.get("relationship", ""))
        strat = str(r.get("strategy_group", ""))
        _add_node(u, label=u, group="underlying", color="#60a5fa", size=20)
        _add_node(e, label=e, group="etf", color="#f59e0b", size=16)
        vis_edges.append({"source": u, "to": e, "title": f"{rel} / {strat}"})

    html = _net_html(
        nodes=[{"id": v["n_id"], "label": v["label"], "group": v["group"], "color": v["color"], "size": v["size"]} for v in nodes.values()],
        edges=vis_edges,
    )
    st.components.v1.html(html, height=540, scrolling=True)

    with st.expander("Edges (raw)"):
        st.dataframe(inbound.head(200), use_container_width=True)

with tab_opts:
    st.subheader(f"Options: {selected}")
    st.caption("Bootstrap via Yahoo (yfinance). For serious options + greeks/history, we’ll switch to Schwab/TOS API later.")

    t = yf.Ticker(selected)
    try:
        expirations = list(t.options)
    except Exception:
        expirations = []

    if not expirations:
        st.info("No options expirations found (or Yahoo unavailable) for this symbol.")
        st.stop()

    exp = st.selectbox("Expiration", expirations, index=0)

    with st.spinner("Loading options chain…"):
        calls, puts = _yahoo_options_chain(selected, exp)

    st.markdown("#### Calls / Puts")
    tabC, tabP = st.tabs(["Calls", "Puts"])

    def _editor(df: pd.DataFrame) -> pd.DataFrame:
        show = df.copy()
        # Add selection + qty
        if "select" not in show.columns:
            show.insert(0, "select", False)
        if "qty" not in show.columns:
            show.insert(1, "qty", 1)
        cols = [
            "select",
            "qty",
            "contractSymbol",
            "strike",
            "lastPrice",
            "bid",
            "ask",
            "impliedVolatility",
            "openInterest",
            "volume",
            "inTheMoney",
        ]
        cols = [c for c in cols if c in show.columns]
        return st.data_editor(
            show[cols],
            use_container_width=True,
            height=420,
            column_config={
                "qty": st.column_config.NumberColumn(min_value=1, max_value=500, step=1),
                "impliedVolatility": st.column_config.NumberColumn(format="%.4f"),
            },
            disabled=[c for c in cols if c not in ("select", "qty")],
        )

    if "cart" not in st.session_state:
        st.session_state["cart"] = []

    def _add_selected_to_cart(ed: pd.DataFrame):
        picked = ed[ed["select"] == True]  # noqa: E712
        if picked.empty:
            st.warning("Select at least one contract.")
            return
        for _, r in picked.iterrows():
            st.session_state["cart"].append(
                {
                    "contractSymbol": r.get("contractSymbol"),
                    "ticker": selected,
                    "expiration": exp,
                    "strike": float(r.get("strike")),
                    "side": str(r.get("side", "")),
                    "qty": int(r.get("qty", 1)),
                    "lastPrice": float(r.get("lastPrice")) if pd.notna(r.get("lastPrice")) else None,
                    "bid": float(r.get("bid")) if pd.notna(r.get("bid")) else None,
                    "ask": float(r.get("ask")) if pd.notna(r.get("ask")) else None,
                }
            )

    with tabC:
        calls = calls.sort_values(["strike"], ascending=True)
        calls_ed = _editor(calls)
        if st.button("Add selected calls to cart", type="primary"):
            calls_ed["side"] = "call"
            _add_selected_to_cart(calls_ed)
            st.success("Added to cart.")

    with tabP:
        puts = puts.sort_values(["strike"], ascending=True)
        puts_ed = _editor(puts)
        if st.button("Add selected puts to cart", type="primary"):
            puts_ed["side"] = "put"
            _add_selected_to_cart(puts_ed)
            st.success("Added to cart.")

with tab_cart:
    st.subheader("Cart")
    cart = st.session_state.get("cart", [])
    if not cart:
        st.info("Cart is empty. Add options contracts from the Options tab.")
        st.stop()

    dfc = pd.DataFrame(cart)

    # Compute estimated premium cost
    def _row_cost(r) -> float:
        px = r.get("ask") or r.get("lastPrice") or 0.0
        qty = int(r.get("qty") or 1)
        return float(px) * 100.0 * qty

    dfc["est_cost"] = dfc.apply(_row_cost, axis=1)

    st.dataframe(dfc, use_container_width=True, height=360)
    st.metric("Estimated premium (USD)", f"${dfc['est_cost'].sum():,.2f}")

    col1, col2 = st.columns([0.25, 0.75])
    with col1:
        if st.button("Clear cart", type="secondary"):
            st.session_state["cart"] = []
            st.rerun()
    with col2:
        st.info("Schwab/TOS order placement is not live yet — this is a functional staging cart.")

with tab_admin:
    st.subheader("Admin / Data pipelines")

    st.markdown("#### 1) Universe")
    st.code("python -m etf_mapper.cli universe --out data --provider polygon", language="bash")
    if st.button("Rebuild universe now"):
        # wipe + rebuild
        (data_dir / "etf_universe.sqlite").unlink(missing_ok=True)
        (data_dir / "etf_universe.parquet").unlink(missing_ok=True)
        _ensure_universe(data_dir)
        st.success("Universe rebuilt.")
        st.rerun()

    st.markdown("#### 2) Prices")
    st.code(
        "python -m etf_mapper.cli prices --out data --universe data/etf_universe.parquet --provider yahoo --limit 200 --start 2024-01-01",
        language="bash",
    )
    if st.button("Reset prices DB"):
        (data_dir / "prices.sqlite").unlink(missing_ok=True)
        st.success("Deleted prices.sqlite.")

    st.markdown("#### 3) Relations graph seed")
    st.code("python -m etf_mapper.cli refresh --out data", language="bash")
    if st.button("Rebuild relations now"):
        (data_dir / "universe.sqlite").unlink(missing_ok=True)
        (data_dir / "equities.parquet").unlink(missing_ok=True)
        (data_dir / "etfs.parquet").unlink(missing_ok=True)
        (data_dir / "edges.parquet").unlink(missing_ok=True)
        _ensure_relations(data_dir)
        st.success("Relations rebuilt.")

    st.markdown("#### Schwab/TOS")
    st.caption("Integration scaffolded; toggle will appear here later.")
