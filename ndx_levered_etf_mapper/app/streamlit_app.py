from __future__ import annotations

from pathlib import Path
import os
import sqlite3
import json
from datetime import datetime
from typing import Optional, Iterable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from pyvis.network import Network
from streamlit_autorefresh import st_autorefresh

from etf_mapper.schwab import SchwabAPI, SchwabConfig
from etf_mapper.config import load_schwab_secrets

# Schwab API removed (replaced with Schwab Market Data)

from etf_mapper.build_universe import refresh_etf_universe
from etf_mapper.build_prices import refresh_prices
from etf_mapper.build import refresh_universe as refresh_relations


# ---------- App boot ----------
st.set_page_config(page_title="ETF Hub", layout="wide")

# Load local .env automatically (kept out of git)
load_dotenv()

# Small styling pass to make the UI feel tighter / more "app-like".
st.markdown(
    """
<style>
  /* Reduce top whitespace */
  .block-container { padding-top: 1.0rem; padding-bottom: 2.0rem; }

  /* Make Streamlit metrics look like cards */
  div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03);
    padding: 0.75rem 0.9rem;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.08);
  }

  /* Slightly tighter tabs */
  .stTabs [data-baseweb="tab"] { font-size: 0.95rem; padding: 0.35rem 0.75rem; }

  /* Dataframe header contrast */
  div[data-testid="stDataFrame"] div[role="columnheader"] {
    background: rgba(255,255,255,0.04);
  }
</style>
    """,
    unsafe_allow_html=True,
)


def _data_dir() -> Path:
    # Stored in session_state so it doesn't "jump" on reruns.
    d = st.session_state.get("data_dir", "data")
    return Path(d).resolve()


def _db_path(name: str) -> Path:
    return _data_dir() / name


def _schwab_api() -> SchwabAPI | None:
    secrets = load_schwab_secrets(_data_dir())
    if secrets is None:
        return None

    return SchwabAPI(
        SchwabConfig(
            client_id=secrets.client_id,
            client_secret=secrets.client_secret,
            redirect_uri=secrets.redirect_uri,
            token_path=secrets.token_path,
        )
    )


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
        template="plotly_dark",
    )
    return fig


def _fetch_history(ticker: str, tos_style: str) -> pd.DataFrame:
    """Live intraday candles via Schwab Market Data.

    Scope is intentionally reduced to 1-minute bars:
      - 1m over 3D
      - 1m over ~4H (today)

    The UI still calls this "TOS-like" but the source is Schwab.
    """
    api = _schwab_api()
    if api is None:
        return pd.DataFrame()

    ticker = _normalize_ticker(ticker)
    if not ticker:
        return pd.DataFrame()

    presets = {
        "1m (3D)": {"periodType": "day", "period": 3, "frequencyType": "minute", "frequency": 1},
        "1m (4H)": {"periodType": "day", "period": 1, "frequencyType": "minute", "frequency": 1, "clip_hours": 4},
    }

    cfg = presets.get(tos_style, presets["1m (3D)"])

    try:
        js = api.price_history(
            ticker,
            period_type=cfg["periodType"],
            period=int(cfg["period"]),
            frequency_type=cfg["frequencyType"],
            frequency=int(cfg["frequency"]),
            need_extended_hours_data=True,
        )
    except Exception:
        return pd.DataFrame()

    candles = js.get("candles") or []
    if not candles:
        return pd.DataFrame()

    df = pd.DataFrame(candles)
    if "datetime" in df.columns:
        df["date"] = pd.to_datetime(df["datetime"], unit="ms", utc=True).dt.tz_convert(None)
    else:
        df["date"] = pd.NaT

    df = df.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"})
    keep = ["date", "open", "high", "low", "close", "volume"]
    for c in keep:
        if c not in df.columns:
            df[c] = None

    df["adj_close"] = df.get("close")
    df = df.dropna(subset=["date"])  # type: ignore

    # Optional clip to last N hours for the 4H view.
    clip_hours = cfg.get("clip_hours")
    if clip_hours:
        cutoff = pd.Timestamp.now() - pd.Timedelta(hours=int(clip_hours))
        df = df[df["date"] >= cutoff]

    return df[["date", "open", "high", "low", "close", "adj_close", "volume"]]


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


@st.cache_data(show_spinner=False, ttl=60 * 5)
def _schwab_profile(ticker: str) -> dict:
    """Lightweight symbol/quote metadata via Schwab quotes endpoint."""
    tkr = _normalize_ticker(ticker)
    if not tkr:
        return {}

    api = _schwab_api()
    if api is None:
        return {}

    try:
        js = api.quotes([tkr])
    except Exception:
        return {}

    # Schwab returns a dict keyed by symbol in some variants; be defensive.
    if isinstance(js, dict):
        rec = js.get(tkr) or js.get(tkr.upper()) or js.get("quotes", {}).get(tkr) or js
    else:
        rec = {}

    if not isinstance(rec, dict):
        return {}

    # Try a few common fields; keep as JSON-ish so UI can render it.
    keys = [
        "symbol",
        "description",
        "assetType",
        "exchangeName",
        "quoteType",
        "cusip",
        "lastPrice",
        "mark",
        "openPrice",
        "highPrice",
        "lowPrice",
        "closePrice",
        "netChange",
        "netPercentChangeInDouble",
        "totalVolume",
    ]
    out = {k: rec.get(k) for k in keys if rec.get(k) not in (None, "")}
    return out


def _normalize_ticker(ticker: str) -> str:
    return str(ticker or "").upper().strip()


def _looks_rate_limited(err: str) -> bool:
    e = (err or "").lower()
    return any(x in e for x in ["429", "too many", "rate limit", "ratelimit", "throttle", "temporarily blocked"])


@st.cache_data(show_spinner=False, ttl=60 * 10)
def _schwab_expirations(ticker: str) -> list[str]:
    # Kept for compatibility; silent failure -> empty list.
    exps, _ = _schwab_expirations_dbg(ticker)
    return exps


def _schwab_expirations_dbg(ticker: str) -> tuple[list[str], Optional[str]]:
    """Return option expiration dates discovered from Schwab option chain."""
    tkr = _normalize_ticker(ticker)
    if not tkr:
        return [], "empty ticker"

    api = _schwab_api()
    if api is None:
        return [], "Schwab OAuth not configured"

    try:
        js = api.option_chain(tkr, contract_type="ALL")
    except Exception as e:
        return [], str(e)

    exps: set[str] = set()
    for k in ["callExpDateMap", "putExpDateMap"]:
        m = js.get(k) if isinstance(js, dict) else None
        if isinstance(m, dict):
            for exp_key in m.keys():
                # exp_key often looks like '2026-02-20:17' (date:dte)
                d = str(exp_key).split(":")[0]
                if d:
                    exps.add(d)

    out = sorted(exps)
    return out, None


@st.cache_data(show_spinner=False, ttl=60 * 1)
def _schwab_option_chain(ticker: str, expiration: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch one expiration's calls/puts from Schwab option chain response."""
    tkr = _normalize_ticker(ticker)
    exp = str(expiration or "").strip()
    if not (tkr and exp):
        return pd.DataFrame(), pd.DataFrame()

    api = _schwab_api()
    if api is None:
        return pd.DataFrame(), pd.DataFrame()

    try:
        js = api.option_chain(tkr, contract_type="ALL")
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

    def _flatten(side_key: str, side: str) -> pd.DataFrame:
        m = js.get(side_key)
        if not isinstance(m, dict):
            return pd.DataFrame()

        rows: list[dict] = []
        for exp_key, strikes in m.items():
            exp_date = str(exp_key).split(":")[0]
            if exp_date != exp:
                continue
            if not isinstance(strikes, dict):
                continue
            for strike_key, contracts in strikes.items():
                if not isinstance(contracts, list) or not contracts:
                    continue
                for c in contracts:
                    if not isinstance(c, dict):
                        continue
                    row = dict(c)
                    row["strike"] = float(row.get("strikePrice") or strike_key or 0)
                    row["side"] = side
                    row["ticker"] = tkr
                    row["expiration"] = exp_date
                    rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # Normalize a few commonly-used columns so the ladder logic keeps working.
        ren = {
            "bid": "bid",
            "ask": "ask",
            "last": "lastPrice",
            "mark": "mark",
            "totalVolume": "volume",
            "openInterest": "openInterest",
            "volatility": "impliedVolatility",
            "delta": "delta",
            "gamma": "gamma",
            "theta": "theta",
            "vega": "vega",
            "rho": "rho",
            "inTheMoney": "inTheMoney",
            "description": "description",
            "symbol": "symbol",
        }
        for out_col, in_col in ren.items():
            if in_col in df.columns and out_col not in df.columns:
                df[out_col] = df[in_col]

        # Keep a minimal stable surface
        if "impliedVolatility" in df.columns and "impliedVolatility" not in df.columns:
            df["impliedVolatility"] = df.get("volatility")

        return df

    calls = _flatten("callExpDateMap", "call")
    puts = _flatten("putExpDateMap", "put")
    return calls, puts


def _options_probe(ticker: str, retries: int = 2) -> dict:
    """Probe Schwab options availability with retries/backoff.

    Returns a dict safe to drop into a dataframe.
    """
    import time

    tkr = _normalize_ticker(ticker)
    out: dict = {
        "ticker": tkr,
        "has_expirations": False,
        "n_expirations": 0,
        "first_exp": None,
        "has_chain": False,
        "calls": 0,
        "puts": 0,
        "error": None,
    }

    if not tkr:
        out["error"] = "empty ticker"
        return out

    last_err = None
    for i in range(retries + 1):
        try:
            exps = _schwab_expirations(tkr)
            out["n_expirations"] = len(exps)
            out["has_expirations"] = len(exps) > 0
            out["first_exp"] = exps[0] if exps else None

            if not exps:
                return out

            calls, puts = _schwab_option_chain(tkr, exps[0])
            out["calls"] = int(len(calls))
            out["puts"] = int(len(puts))
            out["has_chain"] = out["calls"] > 0 or out["puts"] > 0
            return out
        except Exception as e:
            last_err = str(e)
            out["error"] = last_err
            # backoff (small)
            time.sleep(0.4 + 0.6 * i)

            # Clear caches on retry to avoid pinning a transient empty response
            try:
                st.cache_data.clear()
            except Exception:
                pass

    out["error"] = last_err
    return out


def _qqq_constituents_from_local(data_dir: Path) -> list[str]:
    """Best-effort: load Nasdaq-100 constituents from our refresh pipeline outputs."""
    candidates: list[str] = []
    # preferred: equities.parquet (written by refresh)
    p = data_dir / "equities.parquet"
    if p.exists():
        try:
            df = pd.read_parquet(p)
            for col in ["ticker", "symbol", "Ticker", "Symbol"]:
                if col in df.columns:
                    candidates = df[col].dropna().astype(str).tolist()
                    break
        except Exception:
            candidates = []

    # fallback: legacy parquet
    p2 = data_dir / "nasdaq100_constituents.parquet"
    if (not candidates) and p2.exists():
        try:
            df = pd.read_parquet(p2)
            for col in ["ticker", "symbol", "Ticker", "Symbol"]:
                if col in df.columns:
                    candidates = df[col].dropna().astype(str).tolist()
                    break
        except Exception:
            candidates = []

    # normalize + uniq
    out = sorted({ _normalize_ticker(x) for x in candidates if _normalize_ticker(x) })
    return out


def _build_probe_list(data_dir: Path, sample_n: int = 25, include: Optional[Iterable[str]] = None) -> list[str]:
    # Deterministic sample from local Nasdaq-100 list (sorted then first N)
    ndx = _qqq_constituents_from_local(data_dir)
    ndx_sample = ndx[: max(0, int(sample_n))]

    base = set(ndx_sample)

    # Always include these (explicit request)
    base.update(["SPY", "IWM", "TL", "TLT", "MARA"])

    if include:
        for x in include:
            base.add(_normalize_ticker(str(x)))

    return sorted({t for t in base if t})


def _estimate_atm_iv(ticker: str) -> Optional[float]:
    """Best-effort ATM IV estimate from Schwab options (earliest expiry, nearest strike).

    Returns decimal IV (e.g., 0.22) or None.
    """
    expirations = _schwab_expirations(ticker)
    if not expirations:
        return None

    prof = _schwab_profile(ticker)
    spot = None
    try:
        if prof.get("lastPrice") is not None:
            spot = float(prof.get("lastPrice"))
        elif prof.get("mark") is not None:
            spot = float(prof.get("mark"))
    except Exception:
        spot = None

    exp = expirations[0]
    calls, puts = _schwab_option_chain(ticker, exp)
    if calls.empty and puts.empty:
        return None

    # choose strike nearest spot if available; else choose median strike
    strikes = pd.concat([calls.get("strike", pd.Series(dtype=float)), puts.get("strike", pd.Series(dtype=float))], axis=0)
    strikes = strikes.dropna()
    if strikes.empty:
        return None

    if spot is not None:
        target_strike = float(strikes.iloc[(strikes - spot).abs().argmin()])
    else:
        target_strike = float(strikes.sort_values().iloc[len(strikes) // 2])

    def _iv_near(df: pd.DataFrame) -> Optional[float]:
        if df.empty or "strike" not in df.columns or "impliedVolatility" not in df.columns:
            return None
        x = df.loc[(df["strike"] - target_strike).abs().idxmin()]
        v = x.get("impliedVolatility")
        try:
            return float(v) if pd.notna(v) else None
        except Exception:
            return None

    ivs = [v for v in [_iv_near(calls), _iv_near(puts)] if v is not None]
    if not ivs:
        return None
    return float(sum(ivs) / len(ivs))


def _schwab_options_chain(ticker: str, expiration: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Backwards-compatible wrapper (old name) → cached implementation.
    return _schwab_option_chain(ticker, expiration)


def _tos_options_ladder(calls: pd.DataFrame, puts: pd.DataFrame) -> pd.DataFrame:
    """Build a Thinkorswim-ish ladder: one row per strike, calls on left, puts on right."""
    c = calls.copy()
    p = puts.copy()

    def _pick(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        cols = {
            "lastPrice": f"{prefix}_last",
            "bid": f"{prefix}_bid",
            "ask": f"{prefix}_ask",
            "impliedVolatility": f"{prefix}_iv",
            "openInterest": f"{prefix}_oi",
            "volume": f"{prefix}_vol",
            "inTheMoney": f"{prefix}_itm",
            "contractSymbol": f"{prefix}_sym",
        }
        keep = ["strike"] + [c for c in cols.keys() if c in df.columns]
        out = df[keep].rename(columns=cols)
        return out

    c2 = _pick(c, "call")
    p2 = _pick(p, "put")

    ladder = pd.merge(c2, p2, on="strike", how="outer").sort_values("strike")

    # Add selection + qty for both sides
    for side in ["call", "put"]:
        sel = f"{side}_select"
        qty = f"{side}_qty"
        if sel not in ladder.columns:
            ladder.insert(0, sel, False)
        if qty not in ladder.columns:
            ladder.insert(1, qty, 1)

    return ladder


def _net_html(nodes: list[dict], edges: list[dict]) -> str:
    """Render an interactive force graph using pyvis.

    Also inject a `network.fit()` call so the graph is centered/fit-to-view by default.
    """
    net = Network(height="520px", width="100%", bgcolor="#0b1220", font_color="#e5e7eb")
    net.barnes_hut(gravity=-24000, central_gravity=0.25, spring_length=190, spring_strength=0.02)

    for n in nodes:
        nid = n.get("id") or n.get("n_id")
        if nid is None:
            raise RuntimeError("Graph node is missing id/n_id")
        attrs = dict(n)
        attrs.pop("id", None)
        attrs.pop("n_id", None)
        net.add_node(nid, **attrs)

    for e in edges:
        net.add_edge(e.get("source"), e.get("to"), title=e.get("title"))

    html = net.generate_html(notebook=False)

    # pyvis exposes a JS variable named `network` in the generated HTML.
    fit_js = "\n<script>try { network.fit({animation:true}); } catch(e) {}</script>\n"
    if "</body>" in html:
        html = html.replace("</body>", fit_js + "</body>")
    else:
        html = html + fit_js

    return html


# ---------- UI ----------

# -------- Sidebar (streamlined) --------
# Keep the sidebar focused on exploration; move operational controls behind an expander.
with st.sidebar:
    st.markdown("### Explore")

    # Advanced settings (collapsed by default)
    with st.expander("Advanced", expanded=False):
        st.text_input("Data directory", value=st.session_state.get("data_dir", "data"), key="data_dir")

        universe_provider = st.selectbox("Universe provider", ["polygon"], index=0)
        price_provider = st.selectbox("Price provider", ["schwab"], index=0)
        price_start = st.text_input("Price start", value="2024-01-01")
        price_limit = st.slider("Price fetch limit", min_value=25, max_value=1000, value=200, step=25)

    with st.expander("Live", expanded=False):
        auto_refresh = st.toggle("Auto-refresh UI", value=False)
        auto_refresh_s = st.slider("Refresh interval (seconds)", 5, 60, 15, 5)
        if auto_refresh:
            # lightweight refresh loop (useful during market hours)
            st_autorefresh(interval=auto_refresh_s * 1000, key="autorefresh")

    with st.expander("Secrets", expanded=False):
        poly_present = bool(os.getenv("POLYGON_API_KEY"))
        st.write(f"POLYGON_API_KEY: {'set' if poly_present else 'missing'}")
        if not poly_present:
            st.warning("POLYGON_API_KEY not detected. Add it to a local .env file.")

# Need data_dir after sidebar inputs are bound
data_dir = _data_dir()

# Ensure universe exists (or explain why not)
try:
    universe_db = _ensure_universe(data_dir)
except Exception as e:
    st.error(str(e))
    st.info(
        "Create a file named .env next to ndx_levered_etf_mapper/pyproject.toml with POLYGON_API_KEY=... (do not commit it)."
    )
    st.stop()

universe = _load_universe(universe_db)

# Single source of truth: one ticker box.
# We'll use that ticker value both for selection and for filtering the universe table.
with st.sidebar:
    default_ticker = "QQQ" if "QQQ" in set(universe["ticker"]) else str(universe.iloc[0]["ticker"])
    selected = st.text_input("Ticker", value=default_ticker).upper().strip()

if not selected:
    st.warning("Enter a ticker symbol (e.g. TSLA, TSLL, QQQ).")
    st.stop()

q = selected.strip().lower()
view = universe
if q:
    mask = universe["ticker"].str.lower().str.contains(q) | universe["name"].fillna("").str.lower().str.contains(q)
    view = universe[mask].copy()

st.sidebar.caption(f"Ticker bank rows (filtered ETFs): {len(view):,}")

# Header row: title left, provider description right (uses otherwise-empty space)
headerL, headerR = st.columns([0.42, 0.58], vertical_alignment="top")
with headerL:
    st.markdown("# ETF Hub")
    st.caption("Universe • Relations • Prices • Options • Cart")

with headerR:
    # Right-justified, compact description block for the current symbol
    prof_hdr = _schwab_profile(selected)
    summary_hdr = (prof_hdr.get("longBusinessSummary") or "").strip()

    title_bits = []
    if prof_hdr.get("longName"):
        title_bits.append(str(prof_hdr.get("longName")))
    elif prof_hdr.get("shortName"):
        title_bits.append(str(prof_hdr.get("shortName")))

    # Always show the symbol itself
    title_bits.append(selected)

    st.markdown(
        f"<div style='text-align:right; font-size: 1.15rem; font-weight: 600;'>{' — '.join(title_bits)}</div>",
        unsafe_allow_html=True,
    )

    if summary_hdr:
        # Keep the header tight; full text available on Overview tab.
        short = summary_hdr
        if len(short) > 420:
            short = short[:420].rsplit(" ", 1)[0] + "…"
        st.markdown(
            f"<div style='text-align:right; color: #9ca3af; line-height: 1.25;'>{short}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='text-align:right; color:#9ca3af;'>No provider description available.</div>",
            unsafe_allow_html=True,
        )

# Quick facts row
meta = universe[universe["ticker"] == selected].head(1)
nm = str(meta.iloc[0]["name"]) if not meta.empty else selected

c1, c2, c3, c4, c5, c6 = st.columns([0.14, 0.22, 0.18, 0.18, 0.14, 0.14], vertical_alignment="top")
c1.metric("Symbol", selected)
c2.metric("Category", str(prof_hdr.get("category") or "—"))
c3.metric("Fund family", str(prof_hdr.get("fundFamily") or "—"))
c4.metric("Exchange", str(prof_hdr.get("exchange") or "—"))

# Stats (best-effort; placeholders when missing)
inst = prof_hdr.get("heldPercentInstitutions")
try:
    inst_pct = f"{float(inst) * 100.0:,.1f}%" if inst is not None else "—"
except Exception:
    inst_pct = "—"

flt = prof_hdr.get("floatShares")
try:
    flt_txt = f"{int(flt):,}" if flt is not None else "—"
except Exception:
    flt_txt = "—"

iv = _estimate_atm_iv(selected)
iv_txt = "—" if iv is None else f"{iv * 100.0:,.1f}%"

c5.metric("ATM IV (est)", iv_txt)
c6.metric("Inst. own", inst_pct)

with st.expander("More stats", expanded=False):
    st.write({"floatShares": flt_txt, "sharesOutstanding": prof_hdr.get("sharesOutstanding", "—")})

st.caption(f"Heuristic usage: {_infer_intended_usage(nm)}")

# Context menus
(tab_overview, tab_rel, tab_opts, tab_cart, tab_admin) = st.tabs(
    ["Overview", "Relations", "Options", "Cart", "Admin"]
)

with tab_overview:
    colA, colB = st.columns([0.42, 0.58], gap="large")

    with colA:
        st.subheader("Ticker bank")
        st.caption("Currently populated from the US ETF universe (Polygon). You can still type any symbol above.")
        st.dataframe(
            view[["ticker", "name", "primary_exchange", "active"]].head(250),
            use_container_width=True,
            height=420,
        )

        st.markdown("#### Intended usage (heuristic)")
        st.write(_infer_intended_usage(nm))

        # Provider description tucked under Intended usage (collapsed)
        with st.expander("Provider description (full)", expanded=False):
            prof = _schwab_profile(selected)
            if prof:
                desc = (prof.get("description") or "").strip()
                if desc:
                    st.write(desc)

                with st.expander("Provider fields (JSON)", expanded=False):
                    st.json(dict(prof))
            else:
                st.caption("No provider description available (Schwab returned nothing for this symbol).")

    with colB:
        # Timeframe control should live beneath the chart (TOS-like). We'll render it later.
        tos_tf_options = [
            "1m (3D)",
            "1m (4H)",
        ]
        tos_tf_default_index = 0
        # Single source of truth for timeframe: stored in Streamlit session state
        tos_tf = st.session_state.get("tos_tf", tos_tf_options[tos_tf_default_index])
        if tos_tf not in tos_tf_options:
            tos_tf = tos_tf_options[tos_tf_default_index]

        # Prefer DB (fast, stable). If missing, pull from Schwab on-demand.
        universe_parquet = data_dir / "etf_universe.parquet"
        prices_db = data_dir / "prices.sqlite"

        dfp: pd.DataFrame
        used_db = False

        if prices_db.exists():
            try:
                dfp = _load_prices(prices_db, selected)
                used_db = True
            except Exception as e:
                st.error(str(e))
                if st.button("Reset prices DB", type="primary"):
                    prices_db.unlink(missing_ok=True)
                    st.success("Deleted prices.sqlite. Refresh and refetch.")
                dfp = pd.DataFrame()
        else:
            dfp = pd.DataFrame()

        # Let user choose timeframe *below* the chart; but we still need a value for the initial fetch.
        # If the UI reruns, Streamlit will preserve the selectbox state.

        if dfp.empty:
            with st.spinner("Fetching chart data from Schwab…"):
                dfp = _fetch_history(selected, tos_tf)

        if dfp.empty:
            st.warning("No price history found (Schwab may be down or symbol unsupported).")
            if universe_parquet.exists() and st.button("Fetch bootstrap prices DB now", type="secondary"):
                _ensure_prices(data_dir, universe_parquet, provider=price_provider, start=price_start, limit=price_limit)
                st.rerun()
            st.stop()

        # Current price (best-effort): use last close from the chart dataframe.
        try:
            cur_px = float(dfp["close"].dropna().iloc[-1])
        except Exception:
            cur_px = None

        title_price = f"Price" if cur_px is None else f"Price — ${cur_px:,.2f}"
        st.subheader(title_price)

        st.plotly_chart(_plot_candles(dfp, title=f"{selected}"), use_container_width=True)

        # Timeframe selector *below* the chart.
        # Changing it triggers a rerun automatically; on rerun we fetch using st.session_state["tos_tf"].
        st.selectbox(
            "Timeframe",
            tos_tf_options,
            index=tos_tf_options.index(tos_tf),
            key="tos_tf",
        )

        st.caption("Source: local prices.sqlite (daily bars)." if used_db else "Source: Schwab (on-demand).")

        # Move ETF record under raw prices/source area (collapsed by default)
        meta = universe[universe["ticker"] == selected].head(1)
        with st.expander("ETF record (from universe)", expanded=False):
            if not meta.empty:
                st.json(meta.iloc[0].dropna().to_dict())
            else:
                st.caption("No ETF record found in the ETF universe table (might be a stock symbol).")

        with st.expander("Raw prices", expanded=False):
            st.dataframe(dfp, use_container_width=True, height=320)

with tab_rel:
    st.subheader("Relationship map")
    st.caption(
        "Current graph = seeded derivative exposures (index + single-stock leveraged/inverse ETFs). "
        "Many mainstream symbols (e.g., SPY) won't appear until we add holdings-based discovery or seed more index mappings."
    )

    colS, colM = st.columns([0.34, 0.66])
    with colS:
        max_edges = st.slider("Graph complexity", 20, 300, 120, 10)
        max_siblings = st.slider("Sibling ETFs", 0, 300, 120, 10)
        max_backfill = st.slider("Backfill underlyings", 0, 300, 140, 10)
    with colM:
        st.caption("Use these sliders to keep the graph compact so you don’t have to pan/zoom as much.")

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

    # inbound edges to ETF (underlying -> ETF)
    inbound = edges[edges["dst"].astype(str).str.upper() == selected.upper()]
    # outbound edges (selected underlying -> ETFs)
    outbound = edges[edges["src"].astype(str).str.upper() == selected.upper()]

    # Some mainstream symbols (SPY, IVV, VOO, etc.) aren't in the initial seed.
    # Provide a small synthetic fallback mapping so the UI doesn't feel "broken".
    SYNTHETIC_INDEX_MAP = {
        "SPY": ["^GSPC", "SPX"],
        "IVV": ["^GSPC", "SPX"],
        "VOO": ["^GSPC", "SPX"],
        "QQQ": ["^NDX", "NDX"],
        "IWM": ["^RUT", "RUT"],
        "DIA": ["^DJI", "DJI"],
    }
    SYNTHETIC_ETF_FOR_INDEX = {
        "^GSPC": ["SPY", "IVV", "VOO"],
        "SPX": ["SPY", "IVV", "VOO"],
        "^NDX": ["QQQ"],
        "NDX": ["QQQ"],
        "^RUT": ["IWM"],
        "RUT": ["IWM"],
        "^DJI": ["DIA"],
        "DJI": ["DIA"],
    }

    if inbound.empty and outbound.empty:
        fx = str(selected).upper().strip()
        synth_nodes: dict[str, dict] = {}
        synth_edges: list[dict] = []

        def _add_synth_node(i: str, group: str, color: str, size: int):
            if i not in synth_nodes:
                synth_nodes[i] = {"n_id": i, "label": i, "group": group, "color": color, "size": size}

        _add_synth_node(fx, group="center", color="#22c55e", size=26)

        if fx in SYNTHETIC_INDEX_MAP:
            for idx in SYNTHETIC_INDEX_MAP[fx]:
                _add_synth_node(idx, group="index", color="#93c5fd", size=18)
                synth_edges.append({"source": idx, "to": fx, "title": "synthetic: index tracking"})

        if fx in SYNTHETIC_ETF_FOR_INDEX:
            for etf in SYNTHETIC_ETF_FOR_INDEX[fx]:
                _add_synth_node(etf, group="etf", color="#f59e0b", size=18)
                synth_edges.append({"source": fx, "to": etf, "title": "synthetic: tradable proxy"})

        if synth_edges:
            st.info("No relations found in the current seed. Showing a small synthetic index/ETF mapping fallback.")
            html = _net_html(
                nodes=[{"id": v["n_id"], "label": v["label"], "group": v["group"], "color": v["color"], "size": v["size"]} for v in synth_nodes.values()],
                edges=synth_edges,
            )
            st.components.v1.html(html, height=420, scrolling=True)
            st.stop()

        st.info("No relations found for this ticker in the current graph seed.")
        st.stop()

    # Allow "click-expand" style exploration via a focus dropdown.
    # We'll default focus to the sidebar ticker, but you can pivot to any neighbor.
    candidate_nodes = set([selected])
    candidate_nodes.update(inbound["src"].astype(str).tolist())
    candidate_nodes.update(outbound["dst"].astype(str).tolist())

    focus = st.selectbox("Focus node (expand graph)", sorted(candidate_nodes), index=0)

    center = focus
    _add_node(center, label=center, group="center", color="#22c55e", size=26)

    # Determine if focus is acting like an ETF or an underlying, and build an ego network.
    in_f = edges[edges["dst"].astype(str).str.upper() == str(focus).upper()]
    out_f = edges[edges["src"].astype(str).str.upper() == str(focus).upper()]

    # If focus is an ETF: show its underlyings + sibling ETFs for those underlyings.
    if not in_f.empty:
        underlyings = set(in_f["src"].astype(str).tolist())
        for _, r in in_f.iterrows():
            u = str(r["src"])
            rel = str(r.get("relationship", ""))
            strat = str(r.get("strategy_group", ""))
            _add_node(u, label=u, group="underlying", color="#60a5fa", size=20)
            vis_edges.append({"source": u, "to": center, "title": f"{rel} / {strat}"})

        sib = edges[edges["src"].isin(list(underlyings))].copy()
        sib = sib[sib["dst"].astype(str).str.upper() != str(focus).upper()]
        sib = sib.head(int(max_siblings))

        for _, r in sib.iterrows():
            u = str(r["src"])
            e = str(r["dst"])
            rel = str(r.get("relationship", ""))
            strat = str(r.get("strategy_group", ""))
            _add_node(u, label=u, group="underlying", color="#60a5fa", size=20)
            _add_node(e, label=e, group="etf", color="#f59e0b", size=16)
            vis_edges.append({"source": u, "to": e, "title": f"{rel} / {strat}"})

    # If focus is an underlying: show ETFs tied to it + (optionally) other underlyings for those ETFs.
    elif not out_f.empty:
        etfs = set(out_f["dst"].astype(str).tolist())
        for _, r in out_f.iterrows():
            e = str(r["dst"])
            rel = str(r.get("relationship", ""))
            strat = str(r.get("strategy_group", ""))
            _add_node(e, label=e, group="etf", color="#f59e0b", size=18)
            vis_edges.append({"source": center, "to": e, "title": f"{rel} / {strat}"})

        # pull in other underlyings connected to those ETFs (helps discover structured products)
        back = edges[edges["dst"].isin(list(etfs))].copy().head(int(max_backfill))
        for _, r in back.iterrows():
            u = str(r["src"])
            e = str(r["dst"])
            if u.upper() == str(focus).upper():
                continue
            _add_node(u, label=u, group="underlying", color="#60a5fa", size=16)
            vis_edges.append({"source": u, "to": e, "title": str(r.get("relationship", ""))})

    # Cap edge count to keep the graph readable
    vis_edges = vis_edges[: int(max_edges)]

    html = _net_html(
        nodes=[{"id": v["n_id"], "label": v["label"], "group": v["group"], "color": v["color"], "size": v["size"]} for v in nodes.values()],
        edges=vis_edges,
    )
    st.components.v1.html(html, height=540, scrolling=True)

    with st.expander("Edges (raw)"):
        st.dataframe(inbound.head(200), use_container_width=True)

with tab_opts:
    st.subheader(f"Options: {selected}")
    st.caption("Bootstrap via Schwab (Schwab API). TOS-like ladder view; will migrate to Schwab/TOS chain later.")

    # Make failures obvious: Schwab is sometimes flaky / rate-limited.
    with st.spinner("Checking options expirations…"):
        expirations, exp_err = _schwab_expirations_dbg(selected)

    if not expirations:
        if exp_err and _looks_rate_limited(exp_err):
            st.warning("Schwab appears rate-limited right now (429 / throttling). Try again in a minute.")
        else:
            st.warning("No options expirations returned. This can mean: no options, Schwab outage, or rate-limit.")

        with st.expander("Diagnostics", expanded=False):
            st.write({"ticker": selected, "error": exp_err, "note": "If this keeps happening for SPY/QQQ, it's almost certainly Schwab flakiness."})

        colr1, colr2 = st.columns([0.35, 0.65])
        with colr1:
            if st.button("Retry options lookup", type="secondary"):
                st.cache_data.clear()
                st.rerun()
        with colr2:
            st.caption("Tip: the Admin → Options troubleshooting probe can confirm whether Schwab is working across a basket.")
        st.stop()

    exp = st.selectbox("Expiration", expirations, index=0)

    with st.spinner("Loading options chain…"):
        try:
            calls, puts = _schwab_option_chain(selected, exp)
        except Exception as e:
            st.error(f"Options chain load failed: {e}")
            if st.button("Retry chain load", type="secondary"):
                st.cache_data.clear()
                st.rerun()
            st.stop()

    if "cart" not in st.session_state:
        st.session_state["cart"] = []

    ladder = _tos_options_ladder(calls, puts)

    st.markdown("#### Options ladder (calls left, puts right)")

    # Make the ladder more readable
    display_cols = [
        "call_select",
        "call_qty",
        "call_bid",
        "call_ask",
        "call_last",
        "call_iv",
        "call_oi",
        "call_vol",
        "strike",
        "put_bid",
        "put_ask",
        "put_last",
        "put_iv",
        "put_oi",
        "put_vol",
        "put_qty",
        "put_select",
    ]
    display_cols = [c for c in display_cols if c in ladder.columns]

    edited_view = st.data_editor(
        ladder[display_cols],
        use_container_width=True,
        height=520,
        column_config={
            "call_qty": st.column_config.NumberColumn(min_value=1, max_value=500, step=1),
            "put_qty": st.column_config.NumberColumn(min_value=1, max_value=500, step=1),
            "call_iv": st.column_config.NumberColumn(format="%.4f"),
            "put_iv": st.column_config.NumberColumn(format="%.4f"),
        },
        disabled=[c for c in display_cols if c not in ("call_select", "call_qty", "put_select", "put_qty")],
    )

    # Merge edited selection/qty back with the hidden contract symbols (call_sym/put_sym)
    edited = pd.merge(
        edited_view,
        ladder[[c for c in ["strike", "call_sym", "put_sym"] if c in ladder.columns]],
        on="strike",
        how="left",
    )

    colA, colB = st.columns([0.35, 0.65])

    def _add_contract(side: str, r: pd.Series):
        sym = r.get(f"{side}_sym")
        if not sym:
            return
        qty = int(r.get(f"{side}_qty") or 1)
        bid = r.get(f"{side}_bid")
        ask = r.get(f"{side}_ask")
        last = r.get(f"{side}_last")
        st.session_state["cart"].append(
            {
                "contractSymbol": sym,
                "ticker": selected,
                "expiration": exp,
                "strike": float(r.get("strike")),
                "side": side,
                "qty": qty,
                "lastPrice": float(last) if pd.notna(last) else None,
                "bid": float(bid) if pd.notna(bid) else None,
                "ask": float(ask) if pd.notna(ask) else None,
            }
        )

    with colA:
        if st.button("Add selected to cart", type="primary"):
            added = 0
            for _, r in edited.iterrows():
                if bool(r.get("call_select")):
                    _add_contract("call", r)
                    added += 1
                if bool(r.get("put_select")):
                    _add_contract("put", r)
                    added += 1
            if added:
                st.success(f"Added {added} contract(s) to cart.")
            else:
                st.warning("Select at least one call/put in the ladder.")

    with colB:
        st.caption("Tip: pick strikes like TOS — calls on the left, puts on the right. Cart uses ask/last for premium estimate.")

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

    st.markdown("#### Schwab OAuth")

    api = _schwab_api()
    secrets = load_schwab_secrets(_data_dir())
    client_id_set = bool(secrets and secrets.client_id)
    client_secret_set = bool(secrets and secrets.client_secret)
    redirect_set = bool(secrets and secrets.redirect_uri)
    token_path = (secrets.token_path if secrets else str(_db_path("schwab_tokens.json")))
    secrets_path = str((_data_dir() / "schwab_secrets.local.json").resolve())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SCHWAB_CLIENT_ID", "set" if client_id_set else "missing")
    c2.metric("SCHWAB_CLIENT_SECRET", "set" if client_secret_set else "missing")
    c3.metric("SCHWAB_REDIRECT_URI", "set" if redirect_set else "missing")
    c4.metric("Token file", "present" if Path(token_path).exists() else "missing")

    if api is None:
        st.warning(
            "Schwab is not configured yet. Create a local secrets file (gitignored) and restart the app."
        )
        st.code(
            secrets_path,
            language="text",
        )
        st.caption("Example file contents:")
        st.code(
            """{
  \"SCHWAB_CLIENT_ID\": \"...\",
  \"SCHWAB_CLIENT_SECRET\": \"...\",
  \"SCHWAB_REDIRECT_URI\": \"http://127.0.0.1:8501\",
  \"SCHWAB_TOKEN_PATH\": \"data/schwab_tokens.json\",
  \"SCHWAB_OAUTH_SCOPE\": \"readonly\"
}""",
            language="json",
        )

        if st.button("Create empty secrets file", type="secondary"):
            p = Path(secrets_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            if not p.exists():
                p.write_text("{}\n", encoding="utf-8")
            st.success(f"Wrote: {p}")
    else:
        st.caption(
            "Authorize once, then the app will refresh tokens automatically. Paste the full redirect URL (or just the code) below."
        )

        import secrets
        import urllib.parse

        state = st.session_state.get("schwab_oauth_state")
        if not state:
            state = secrets.token_urlsafe(16)
            st.session_state["schwab_oauth_state"] = state

        scope = (secrets.oauth_scope if secrets else "readonly")
        auth_url = api.build_authorize_url(state=state, scope=scope)
        st.code(auth_url, language="text")

        oauth_in = st.text_input("Paste redirect URL (or code)", value=st.session_state.get("schwab_oauth_in", ""))
        st.session_state["schwab_oauth_in"] = oauth_in

        def _extract_code(x: str) -> str | None:
            x = (x or "").strip()
            if not x:
                return None
            if x.startswith("http"):
                try:
                    q = urllib.parse.urlparse(x).query
                    params = urllib.parse.parse_qs(q)
                    code = params.get("code", [None])[0]
                    return code
                except Exception:
                    return None
            return x

        if st.button("Exchange code for tokens", type="primary"):
            code = _extract_code(oauth_in)
            if not code:
                st.error("Could not extract code.")
            else:
                try:
                    api.exchange_code(code)
                    st.success("Saved Schwab tokens.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Token exchange failed: {e}")

        if st.button("Refresh access token now"):
            try:
                api.refresh_access_token()
                st.success("Refreshed token.")
            except Exception as e:
                st.error(f"Refresh failed: {e}")

    st.divider()

    st.markdown("#### Options troubleshooting")
    st.caption(
        "Schwab options can fail transiently (rate-limits/outages). Use this to probe expirations/chains across a basket. "
        "For Nasdaq-100, run `refresh` first so equities.parquet exists."
    )

    probe_default = "SPY, IWM, TL, MARA"
    custom_probe = st.text_input("Custom tickers (comma-separated)", value=probe_default)

    include_n100 = st.toggle("Include Nasdaq-100 sample (from local parquet)", value=True)
    sample_n = st.slider("Nasdaq-100 sample size", 5, 100, 25, 5)

    probe_retries = st.slider("Retries per ticker", 0, 4, 2, 1)

    def _save_probe_logs(df: pd.DataFrame) -> tuple[Path, Path]:
        out_dir = data_dir / "logs"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = out_dir / f"options_probe_{stamp}.csv"
        jsonl_path = out_dir / f"options_probe_{stamp}.jsonl"

        df.to_csv(csv_path, index=False)
        with jsonl_path.open("w", encoding="utf-8") as f:
            for _, r in df.iterrows():
                # convert to plain python types for JSON
                obj = {k: (None if pd.isna(v) else v) for k, v in r.to_dict().items()}
                f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")

        return jsonl_path, csv_path

    if st.button("Run options probe", type="primary"):
        # Build list
        custom_list = [_normalize_ticker(x) for x in (custom_probe or "").split(",") if _normalize_ticker(x)]
        tickers = set(custom_list)
        if include_n100:
            tickers.update(_build_probe_list(data_dir, sample_n=int(sample_n)))
        tick_list = sorted({t for t in tickers if t})

        st.write({"tickers_to_probe": len(tick_list), "sample_n": int(sample_n) if include_n100 else 0})

        prog = st.progress(0)
        rows = []
        for i, tkr in enumerate(tick_list, start=1):
            rows.append(_options_probe(tkr, retries=int(probe_retries)))
            prog.progress(int(i / max(1, len(tick_list)) * 100))

        df = pd.DataFrame(rows)
        # Add timestamp for per-row traversal
        df.insert(0, "ts", datetime.now().isoformat())
        df = df.sort_values(["has_chain", "has_expirations", "ticker"], ascending=[True, True, True])

        jsonl_path, csv_path = _save_probe_logs(df)
        st.success(f"Wrote logs: {jsonl_path} and {csv_path}")

        st.dataframe(df, use_container_width=True, height=420)

        bad = df[(~df["has_expirations"]) | (~df["has_chain"])].copy()
        if not bad.empty:
            st.warning("Some tickers did not return expirations and/or a chain. This is often Schwab flakiness; retry later.")
            st.dataframe(bad[["ticker", "has_expirations", "n_expirations", "first_exp", "has_chain", "calls", "puts", "error"]], use_container_width=True)

    st.markdown("#### Status")
    univ_db = data_dir / "etf_universe.sqlite"
    rel_db = data_dir / "universe.sqlite"
    prices_db = data_dir / "prices.sqlite"

    cols = st.columns(3)
    cols[0].metric("Universe DB", "present" if univ_db.exists() else "missing")
    cols[1].metric("Relations DB", "present" if rel_db.exists() else "missing")
    cols[2].metric("Prices DB", "present" if prices_db.exists() else "missing")

    with st.expander("Quick diagnostics"):
        try:
            if univ_db.exists():
                dfu = _read_sql(str(univ_db), "select count(*) as n from etf_universe")
                st.write({"etf_universe_rows": int(dfu.iloc[0]["n"])})
            if rel_db.exists():
                dfe = _read_sql(str(rel_db), "select count(*) as n from edges")
                st.write({"edges_rows": int(dfe.iloc[0]["n"])})
            if prices_db.exists():
                # count distinct tickers w/ prices
                dfp = _read_sql(str(prices_db), "select count(distinct ticker) as n from prices_daily")
                st.write({"priced_tickers": int(dfp.iloc[0]["n"])})
        except Exception as e:
            st.error(f"Diagnostics error: {e}")

    st.markdown("#### 1) Universe")
    st.code("python -m etf_mapper.cli universe --out data --provider polygon", language="bash")
    if st.button("Rebuild universe now"):
        (data_dir / "etf_universe.sqlite").unlink(missing_ok=True)
        (data_dir / "etf_universe.parquet").unlink(missing_ok=True)
        _ensure_universe(data_dir)
        st.success("Universe rebuilt.")
        st.rerun()

    st.markdown("#### 2) Prices")
    st.code(
        "python -m etf_mapper.cli prices --out data --universe data/etf_universe.parquet --provider schwab --limit 200 --start 2024-01-01",
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

    st.markdown("#### Test tickers")
    st.caption("Use these during troubleshooting to sanity-check options/relations quickly.")
    st.code("AMZN  TSLA  AAPL  QQQ  TSLL  SPY", language="text")

    st.markdown("#### Orders (optional)")
    st.caption("Order placement endpoints exist in the client, but the UI does not submit live orders yet.")
