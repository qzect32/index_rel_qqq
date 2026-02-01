from __future__ import annotations

from pathlib import Path
import sqlite3

import pandas as pd
import streamlit as st


st.set_page_config(page_title="ETF Hub", layout="wide")


def _load_etf_universe(db_path: Path) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql("select * from etf_universe", conn)


def _load_prices(db_path: Path, ticker: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        q = "select date, open, high, low, close, adj_close, volume, source from prices_daily where ticker = ? order by date"
        df = pd.read_sql(q, conn, params=(ticker,))
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"])  # type: ignore


st.title("ETF Hub")
st.caption("US ETF universe + bootstrap market data (Yahoo/Stooq) — Schwab/TOS provider later")

colL, colR = st.columns([0.35, 0.65], gap="large")

with colL:
    st.subheader("Universe")
    data_dir = Path(st.sidebar.text_input("Data directory", value="data"))
    universe_db = data_dir / "etf_universe.sqlite"
    prices_db = data_dir / "prices.sqlite"

    if not universe_db.exists():
        st.warning(f"Missing {universe_db}. Run: python -m etf_mapper.cli universe --out data")
        st.stop()

    universe = _load_etf_universe(universe_db)
    universe["ticker"] = universe["ticker"].astype(str).str.upper().str.strip()

    query = st.text_input("Search (ticker or name)", value="QQQ")
    q = query.strip().lower()
    if q:
        mask = universe["ticker"].str.lower().str.contains(q) | universe["name"].fillna("").str.lower().str.contains(q)
        view = universe[mask].copy()
    else:
        view = universe.copy()

    st.write(f"Rows: {len(view):,}")
    st.dataframe(view[["ticker", "name", "primary_exchange", "active"]].head(200), use_container_width=True, height=420)

    default_ticker = "QQQ" if "QQQ" in set(universe["ticker"]) else universe.iloc[0]["ticker"]
    ticker = st.text_input("Selected ticker", value=str(default_ticker)).upper().strip()

    meta = universe[universe["ticker"] == ticker].head(1)
    if not meta.empty:
        st.markdown("**Metadata**")
        st.json(meta.iloc[0].dropna().to_dict())

with colR:
    st.subheader(f"Chart: {ticker}")

    if not prices_db.exists():
        st.info(
            f"No {prices_db} yet. Run: python -m etf_mapper.cli prices --out data --universe data/etf_universe.parquet --provider yahoo --limit 200"
        )
        st.stop()

    dfp = _load_prices(prices_db, ticker)
    if dfp.empty:
        st.warning("No prices found for this ticker yet. Fetch it via the prices command (increase --limit or add targeting later).")
        st.stop()

    st.line_chart(dfp.set_index("date")["close"], height=320)

    with st.expander("Raw prices", expanded=False):
        st.dataframe(dfp, use_container_width=True, height=360)
