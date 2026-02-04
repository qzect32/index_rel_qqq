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

from ladder_styles import style_ladder_with_changes
from local_oauth import CallbackServerState, ensure_localhost_cert, start_https_callback_server, stop_callback_server
from schwab_diagnostics import safe_snip
from dotenv import load_dotenv
from pyvis.network import Network
from streamlit_autorefresh import st_autorefresh

from etf_mapper.schwab import SchwabAPI, SchwabConfig
from etf_mapper.config import load_schwab_secrets
from etf_mapper.spade_checks import check_option_chain, check_price_history, summarize

# Schwab API removed (replaced with Schwab Market Data)

# Polygon universe builder kept in codebase but not required by Schwab-only UI.
# from etf_mapper.build_universe import refresh_etf_universe
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


def _ensure_universe(data_dir: Path) -> Path | None:
    """Universe is optional in Schwab-only mode.

    We keep the builder code around, but the UI must not require Polygon or a universe DB.
    """
    db = data_dir / "etf_universe.sqlite"
    if db.exists():
        return db

    # Schwab-only mode: do not hard-stop the app.
    return None


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
    """Build a Thinkorswim-ish ladder: one row per strike, calls on left, puts on right.

    Must be tolerant of empty/partial data (e.g., before Schwab OAuth is completed).
    """
    c = calls.copy() if isinstance(calls, pd.DataFrame) else pd.DataFrame()
    p = puts.copy() if isinstance(puts, pd.DataFrame) else pd.DataFrame()

    # If neither side has a strike column, return an empty ladder with the UI-expected columns.
    if (c.empty or "strike" not in c.columns) and (p.empty or "strike" not in p.columns):
        base_cols = [
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
        return pd.DataFrame(columns=base_cols)

    def _pick(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        if df is None or df.empty or "strike" not in df.columns:
            return pd.DataFrame(columns=["strike"])

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
        keep = ["strike"] + [k for k in cols.keys() if k in df.columns]
        out = df[keep].rename(columns=cols)
        return out

    c2 = _pick(c, "call")
    p2 = _pick(p, "put")

    ladder = pd.merge(c2, p2, on="strike", how="outer")
    if "strike" in ladder.columns:
        ladder = ladder.sort_values("strike")

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

        # Schwab-only mode: hide universe provider.
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
        # Schwab-only UI: polygon is not required.
        secrets = load_schwab_secrets(_data_dir())
        st.write(
            {
                "schwab_secrets_file": str((_data_dir() / "schwab_secrets.local.json").resolve()),
                "schwab_configured": bool(secrets),
                "schwab_tokens_present": (_data_dir() / "schwab_tokens.json").exists(),
            }
        )

# Need data_dir after sidebar inputs are bound
data_dir = _data_dir()

# Universe is optional (Schwab-only UI).
universe_db = _ensure_universe(data_dir)
if universe_db is not None:
    universe = _load_universe(universe_db)
else:
    universe = pd.DataFrame(columns=["ticker", "name", "primary_exchange", "active"])

# Single source of truth: one ticker box.
with st.sidebar:
    selected = st.text_input("Ticker", value="QQQ").upper().strip()

if not selected:
    st.warning("Enter a ticker symbol (e.g. TSLA, TSLL, QQQ).")
    st.stop()

# No universe table filtering in Schwab-only mode.

# Header row: title left, provider description right (uses otherwise-empty space)
headerL, headerR = st.columns([0.42, 0.58], vertical_alignment="top")
with headerL:
    st.markdown("# ETF Hub")
    st.caption("Schwab • Relations • Prices • Options • Position Builder")

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

# Quick facts row (Schwab quotes are lightweight; universe metadata may be absent)
nm = selected

c1, c2, c3, c4, c5, c6 = st.columns([0.14, 0.22, 0.18, 0.18, 0.14, 0.14], vertical_alignment="top")
c1.metric("Symbol", selected)
c2.metric("Asset type", str(prof_hdr.get("assetType") or "—"))
c3.metric("Exchange", str(prof_hdr.get("exchangeName") or prof_hdr.get("exchange") or "—"))
c4.metric("Last", str(prof_hdr.get("lastPrice") or prof_hdr.get("mark") or "—"))

iv = _estimate_atm_iv(selected)
iv_txt = "—" if iv is None else f"{iv * 100.0:,.1f}%"

c5.metric("ATM IV (est)", iv_txt)
c6.metric("Volume", str(prof_hdr.get("totalVolume") or "—"))

with st.expander("More stats", expanded=False):
    st.json(dict(prof_hdr) if prof_hdr else {})

st.caption(f"Heuristic usage: {_infer_intended_usage(nm)}")

# Context menus
(tab_overview, tab_rel, tab_opts, tab_cart, tab_admin) = st.tabs(
    ["Overview", "Relations", "Options", "Position Builder", "Admin"]
)

with tab_overview:
    colA, colB = st.columns([0.42, 0.58], gap="large")

    # Spade checks summary (updates with live fetches)
    st.session_state.setdefault("spade", {})

    with colA:
        st.subheader("Symbol")
        st.caption("Schwab-only mode: type any symbol (e.g., SPY, QQQ, /ES).")

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

        # Spade checks: price history
        try:
            checks = check_price_history(dfp)
            st.session_state["spade"]["history"] = summarize(checks)
        except Exception:
            st.session_state["spade"]["history"] = None

        if dfp.empty:
            st.warning("No price history found (Schwab may be down, tokens missing, or symbol unsupported).")
            st.caption("This does not block the rest of the app. Use Admin → Schwab OAuth to connect.")
            if universe_parquet.exists() and st.button("Fetch bootstrap prices DB now", type="secondary"):
                _ensure_prices(data_dir, universe_parquet, provider=price_provider, start=price_start, limit=price_limit)
                st.rerun()
            # IMPORTANT: don't st.stop() here; it prevents other tabs (Admin) from rendering.
            dfp = pd.DataFrame()

        # Current price (best-effort): use last close from the chart dataframe.
        try:
            cur_px = float(dfp["close"].dropna().iloc[-1])
        except Exception:
            cur_px = None

        title_price = f"Price" if cur_px is None else f"Price — ${cur_px:,.2f}"
        st.subheader(title_price)

        if not dfp.empty:
            st.plotly_chart(_plot_candles(dfp, title=f"{selected}"), use_container_width=True)

        # Spade status (history)
        sp = st.session_state.get("spade", {}).get("history")
        if sp and isinstance(sp, dict):
            msg = f"Spade checks: {sp.get('status')} (warn={sp.get('warn')}, fail={sp.get('fail')})"
            if sp.get("status") == "FAIL":
                st.error(msg)
            elif sp.get("status") == "WARN":
                st.warning(msg)
            else:
                st.caption(msg)
            with st.expander("Spade details (price history)", expanded=False):
                st.json(sp)

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
    if str(selected).startswith("/"):
        st.info("Futures symbols like /ES often require a different options symbol convention. Schwab equity options chain endpoint may not return options for futures roots.")
    st.caption("Schwab options chain (equity-style) with TOS-like ladder layout.")

    # Make failures obvious: Schwab is sometimes flaky / rate-limited.
    expirations, exp_err = [], None
    if not str(selected).startswith("/"):
        with st.spinner("Checking options expirations…"):
            expirations, exp_err = _schwab_expirations_dbg(selected)

    if not expirations:
        if exp_err and _looks_rate_limited(exp_err):
            st.warning("Schwab appears rate-limited right now (429 / throttling). Try again in a minute.")
        else:
            st.warning("No options expirations returned. This can mean: no options, Schwab outage, tokens missing, or rate-limit.")

        with st.expander("Diagnostics", expanded=False):
            st.write({"ticker": selected, "error": exp_err, "note": "If this keeps happening for SPY/QQQ, confirm OAuth tokens in Admin."})

        colr1, colr2 = st.columns([0.35, 0.65])
        with colr1:
            if st.button("Retry options lookup", type="secondary"):
                st.cache_data.clear()
                st.rerun()
        with colr2:
            st.caption("Tip: Admin → Schwab OAuth should be completed before expecting options.")
        # IMPORTANT: don't st.stop(); allow other tabs (Admin) to render.
        expirations = []

    # TOS-like: pick expiration; strikes are shown around ATM by default.
    if expirations:
        exp = st.selectbox("Expiration", expirations, index=0)
    else:
        exp = None

    # Strike window (around ATM)
    strike_window = st.selectbox(
        "Strike window (around ATM)",
        options=["4", "8", "16", "32", "ALL"],
        index=2,
        help="Shows N strikes above and below ATM (approx).",
    )

    calls, puts = pd.DataFrame(), pd.DataFrame()
    if exp:
        with st.spinner("Loading options chain…"):
            try:
                calls, puts = _schwab_option_chain(selected, exp)
            except Exception as e:
                st.error(f"Options chain load failed: {e}")
                if st.button("Retry chain load", type="secondary"):
                    st.cache_data.clear()
                    st.rerun()
                # IMPORTANT: don't stop entire app.
                calls, puts = pd.DataFrame(), pd.DataFrame()

    # Spade checks: option chain
    try:
        sp_checks = check_option_chain(calls, puts)
        sp_sum = summarize(sp_checks)
        st.session_state["spade"]["chain"] = sp_sum
        if sp_sum.get("status") == "FAIL":
            st.error(f"Spade checks: FAIL (options chain) — see details below")
        elif sp_sum.get("status") == "WARN":
            st.warning(f"Spade checks: WARN (options chain) — see details below")
        else:
            st.caption("Spade checks: OK (options chain)")

        with st.expander("Spade details (options chain)", expanded=False):
            st.json(sp_sum)
    except Exception:
        st.session_state["spade"]["chain"] = None

    # Position Builder legs (backwards compatible with older key name)
    if "builder_legs" not in st.session_state:
        st.session_state["builder_legs"] = st.session_state.get("cart", [])

    ladder = _tos_options_ladder(calls, puts)

    # Filter strikes around ATM by default
    prof = _schwab_profile(selected)
    spot = None
    try:
        spot = float(prof.get("lastPrice") or prof.get("mark"))
    except Exception:
        spot = None

    # Ex-dividend warning (best-effort)
    try:
        api = _schwab_api()
        if api is not None:
            div = api.dividends(_normalize_ticker(selected))
            exd = (div.ex_dividend_date or "").strip()
            if exd:
                # Warn if expiration is on/after ex-div date (especially relevant if you might be short calls)
                try:
                    exp_dt = pd.Timestamp(str(exp)).date()
                    exd_dt = pd.Timestamp(exd).date()
                    if exp_dt >= exd_dt:
                        st.warning(f"Ex-dividend warning: ex-div date {exd} is on/before selected expiration {exp}. Early assignment risk can increase for short ITM calls.")
                except Exception:
                    pass
    except Exception:
        pass

    if spot is not None and (strike_window or "").upper() != "ALL" and "strike" in ladder.columns and not ladder.empty:
        try:
            n = int(strike_window)
        except Exception:
            n = 16
        strikes = ladder["strike"].astype(float)
        atm_idx = int((strikes - float(spot)).abs().idxmin())
        # Convert idx into position in the sorted ladder
        ladder = ladder.sort_values("strike").reset_index(drop=True)
        strikes2 = ladder["strike"].astype(float)
        atm_pos = int((strikes2 - float(spot)).abs().idxmin())
        lo = max(0, atm_pos - n)
        hi = min(len(ladder), atm_pos + n + 1)
        ladder = ladder.iloc[lo:hi].copy().reset_index(drop=True)

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

    # Keep previous ladder snapshot for change highlighting (live-ish)
    prev_ladder = st.session_state.get("prev_ladder")
    st.session_state["prev_ladder"] = ladder[[c for c in ladder.columns if c in display_cols]].copy()

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

    # Read-only styled view (call/put shading + change intensity)
    with st.expander("Ladder (styled / live changes)", expanded=False):
        try:
            st.dataframe(style_ladder_with_changes(ladder[display_cols], prev_ladder), use_container_width=True, height=520)
            st.caption("Change highlighting is best-effort (compares current snapshot to previous refresh).")
        except Exception as e:
            st.caption(f"Styled ladder unavailable: {e}")

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
        st.session_state["builder_legs"].append(
            {
                "contractSymbol": sym,
                "ticker": selected,
                "expiration": exp,
                "strike": float(r.get("strike")),
                "side": side,
                "action": "BUY",
                "qty": qty,
                "lastPrice": float(last) if pd.notna(last) else None,
                "bid": float(bid) if pd.notna(bid) else None,
                "ask": float(ask) if pd.notna(ask) else None,
            }
        )

    with colA:
        if st.button("Add selected to position builder", type="primary"):
            added = 0
            for _, r in edited.iterrows():
                if bool(r.get("call_select")):
                    _add_contract("call", r)
                    added += 1
                if bool(r.get("put_select")):
                    _add_contract("put", r)
                    added += 1
            if added:
                st.success(f"Added {added} contract(s) to position builder.")
            else:
                st.warning("Select at least one call/put in the ladder.")

    with colB:
        st.caption("Tip: pick strikes like TOS — calls on the left, puts on the right. Position Builder uses bid/ask for debit/credit estimates.")

with tab_cart:
    st.subheader("Position Builder")

    legs = st.session_state.get("builder_legs", [])
    if not legs:
        st.info("No legs yet. Add option contracts from the Options tab.")
        legs = []

    df = pd.DataFrame(legs)
    if "action" not in df.columns:
        df["action"] = "BUY"

    st.caption("Set BUY/SELL for each leg. Debit/Credit uses ask for BUY and bid for SELL (best-effort).")

    edited = st.data_editor(
        df,
        use_container_width=True,
        height=320,
        column_config={
            "action": st.column_config.SelectboxColumn(options=["BUY", "SELL"]),
            "qty": st.column_config.NumberColumn(min_value=1, max_value=500, step=1),
        },
        disabled=[c for c in df.columns if c not in ("action", "qty")],
    )

    # Persist edits back
    st.session_state["builder_legs"] = edited.to_dict(orient="records")

    def _leg_price(r) -> float:
        # BUY -> ask (pay); SELL -> bid (receive); fallback to last
        a = str(r.get("action") or "BUY").upper()
        if a == "SELL":
            px = r.get("bid") or r.get("lastPrice") or 0.0
            return float(px or 0.0)
        px = r.get("ask") or r.get("lastPrice") or 0.0
        return float(px or 0.0)

    def _leg_sign(r) -> int:
        return -1 if str(r.get("action") or "BUY").upper() == "SELL" else 1

    edited["est_px"] = edited.apply(_leg_price, axis=1)
    edited["est_premium"] = edited.apply(lambda r: _leg_sign(r) * float(r.get("est_px") or 0.0) * 100.0 * int(r.get("qty") or 1), axis=1)

    net = float(edited["est_premium"].sum())
    net_txt = f"${abs(net):,.2f}" + (" debit" if net > 0 else " credit" if net < 0 else "")

    c1, c2, c3 = st.columns(3)
    c1.metric("Net premium", net_txt)
    c2.metric("Legs", str(len(edited)))
    c3.metric("Expiration", str(edited["expiration"].iloc[0]) if "expiration" in edited.columns and len(edited) else "—")

    # Payoff at expiration (intrinsic only) — a "good enough" first scaffold.
    st.markdown("#### Payoff at expiration (scaffold)")

    prof = _schwab_profile(selected)
    spot = None
    try:
        spot = float(prof.get("lastPrice") or prof.get("mark"))
    except Exception:
        spot = None

    # Scenario controls (time + underlying move) for quick intuition
    with st.expander("Scenario (time + underlying move)", expanded=False):
        days_fwd = st.slider("Days forward", 0, 30, 0, 1)
        move = st.slider("Underlying move ($)", -200.0, 200.0, 0.0, 1.0)
        st.caption("Greeks-based estimate is best-effort. Use for intuition, not guarantees.")

    if spot is None:
        st.warning("Spot price unavailable; payoff graph needs lastPrice/mark.")
    else:
        import numpy as np

        s0 = float(spot)
        xs = np.linspace(s0 * 0.7, s0 * 1.3, 121)

        def _payoff_leg(r, s: float) -> float:
            side = str(r.get("side") or "").lower()
            k = float(r.get("strike") or 0.0)
            qty = int(r.get("qty") or 1)
            sign = 1 if str(r.get("action") or "BUY").upper() == "BUY" else -1
            px = float(r.get("est_px") or 0.0)

            if side == "call":
                intrinsic = max(s - k, 0.0)
            elif side == "put":
                intrinsic = max(k - s, 0.0)
            else:
                intrinsic = 0.0

            # Profit = (intrinsic - premium) for BUY; (premium - intrinsic) for SELL
            if sign == 1:
                pnl = (intrinsic - px) * 100.0 * qty
            else:
                pnl = (px - intrinsic) * 100.0 * qty
            return float(pnl)

        ys = []
        for s in xs:
            total = 0.0
            for _, r in edited.iterrows():
                total += _payoff_leg(r, float(s))
            ys.append(total)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="PnL @ exp"))

        # Scenario overlay (greeks-based approximation)
        try:
            s1 = s0 + float(move)
        except Exception:
            s1 = s0

        if int(days_fwd) > 0:
            # Use theta/delta when present; otherwise skip.
            def _est_price(r, s: float) -> float:
                px0 = float(r.get("est_px") or 0.0)
                d = r.get("delta")
                th = r.get("theta")
                try:
                    d = float(d) if d is not None else None
                except Exception:
                    d = None
                try:
                    th = float(th) if th is not None else None
                except Exception:
                    th = None

                px = px0
                if d is not None:
                    px += d * (s - s0)
                # theta is commonly quoted per day; assume per day
                if th is not None:
                    px += th * float(days_fwd)
                return float(max(px, 0.0))

            def _pnl_leg_scenario(r, s: float) -> float:
                side = str(r.get("side") or "").lower()
                k = float(r.get("strike") or 0.0)
                qty = int(r.get("qty") or 1)
                sign = 1 if str(r.get("action") or "BUY").upper() == "BUY" else -1
                px = _est_price(r, s)

                if side == "call":
                    intrinsic = max(s - k, 0.0)
                elif side == "put":
                    intrinsic = max(k - s, 0.0)
                else:
                    intrinsic = 0.0

                if sign == 1:
                    pnl = (intrinsic - px) * 100.0 * qty
                else:
                    pnl = (px - intrinsic) * 100.0 * qty
                return float(pnl)

            ys2 = []
            for s in xs:
                total = 0.0
                for _, r in edited.iterrows():
                    total += _pnl_leg_scenario(r, float(s))
                ys2.append(total)

            fig.add_trace(go.Scatter(x=xs, y=ys2, mode="lines", name=f"PnL est (+{days_fwd}d)", line=dict(dash="dash")))

        fig.add_vline(x=s0, line_width=1, line_dash="dot", line_color="#94a3b8")
        fig.add_vline(x=s1, line_width=1, line_dash="dot", line_color="#22c55e")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns([0.25, 0.75])
    with col1:
        if st.button("Clear position builder", type="secondary"):
            st.session_state["builder_legs"] = []
            st.rerun()
    with col2:
        st.info("Order placement is intentionally not wired yet. This tab is for building/inspecting positions.")

with tab_admin:
    st.subheader("Admin / Data pipelines")

    st.markdown("#### Schwab OAuth")

    # One-click local OAuth support (HTTPS callback on 127.0.0.1:8000)
    st.session_state.setdefault("oauth_server", None)
    st.session_state.setdefault("oauth_thread", None)
    st.session_state.setdefault(
        "oauth_state",
        CallbackServerState(out_path=_data_dir() / "schwab_last_code.txt"),
    )

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
            "Authorize once, then the app will refresh tokens automatically. "
            "If Schwab forces an HTTPS localhost callback (127.0.0.1:8000), you can use the one-click controls below."
        )

        st.markdown("**One-click local HTTPS callback (127.0.0.1:8000)**")
        col_cb1, col_cb2, col_cb3 = st.columns([0.34, 0.33, 0.33])

        cert_path = _data_dir() / "localhost.pem"
        key_path = _data_dir() / "localhost-key.pem"

        with col_cb1:
            if st.button("Generate localhost cert", type="secondary"):
                try:
                    ensure_localhost_cert(cert_path, key_path)
                    st.success("Generated data/localhost.pem and data/localhost-key.pem")
                except Exception as e:
                    st.error(f"Cert generation failed: {e}")

        with col_cb2:
            if st.button("Start HTTPS callback server", type="primary"):
                try:
                    ensure_localhost_cert(cert_path, key_path)
                    state = st.session_state["oauth_state"]
                    t, httpd = start_https_callback_server(state, cert_path, key_path)
                    st.session_state["oauth_thread"] = t
                    st.session_state["oauth_server"] = httpd
                    st.success("Callback server started on https://127.0.0.1:8000")
                except Exception as e:
                    st.error(f"Failed to start callback server: {e}")

        with col_cb3:
            if st.button("Stop callback server"):
                try:
                    stop_callback_server(st.session_state.get("oauth_server"))
                    st.session_state["oauth_server"] = None
                    st.session_state["oauth_thread"] = None
                    st.success("Stopped callback server")
                except Exception as e:
                    st.error(f"Failed to stop server: {e}")

        # Status
        st.write(
            {
                "callback_running": bool(getattr(st.session_state.get("oauth_state"), "running", False)),
                "cert_present": cert_path.exists(),
                "key_present": key_path.exists(),
                "last_code_present": (_data_dir() / "schwab_last_code.txt").exists(),
                "last_error": getattr(st.session_state.get("oauth_state"), "last_error", None),
            }
        )

        import secrets as py_secrets
        import urllib.parse

        state = st.session_state.get("schwab_oauth_state")
        if not state:
            state = py_secrets.token_urlsafe(16)
            st.session_state["schwab_oauth_state"] = state

        # Use configured scope from loaded Schwab secrets (not the stdlib secrets module)
        scope = (load_schwab_secrets(_data_dir()).oauth_scope if load_schwab_secrets(_data_dir()) else "readonly")
        auth_url = api.build_authorize_url(state=state, scope=scope)
        st.code(auth_url, language="text")

        oauth_in = st.text_input("Paste redirect URL (or code)", value=st.session_state.get("schwab_oauth_in", ""))
        st.session_state["schwab_oauth_in"] = oauth_in

        # Convenience: if you use the local callback catcher script (port 8000), it can write the code to a file.
        code_file = st.text_input("(Optional) Local code file", value=str((_data_dir() / "schwab_last_code.txt").resolve()))
        if st.button("Load code from file"):
            try:
                p = Path(code_file)
                if p.exists():
                    st.session_state["schwab_oauth_in"] = p.read_text(encoding="utf-8").strip()
                    st.success("Loaded code into input.")
                    st.rerun()
                else:
                    st.error(f"File not found: {p}")
            except Exception as e:
                st.error(f"Failed to read file: {e}")

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

    st.markdown("#### Schwab diagnostics")
    st.caption("Use this to debug symbol support (e.g., SPY vs /ES).")

    diag_sym = st.text_input("Diagnostic symbol", value=selected)
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        if st.button("Test quotes"):
            try:
                js = api.quotes([_normalize_ticker(diag_sym)])
                st.code(safe_snip(js), language="json")
            except Exception as e:
                st.error(str(e))

    with col_d2:
        if st.button("Test 1m/3D price history"):
            try:
                js = api.price_history(
                    _normalize_ticker(diag_sym),
                    period_type="day",
                    period=3,
                    frequency_type="minute",
                    frequency=1,
                    need_extended_hours_data=True,
                )
                candles = js.get("candles") if isinstance(js, dict) else None
                st.write({"candles": 0 if not candles else len(candles), "keys": list(js.keys()) if isinstance(js, dict) else type(js).__name__})
                st.code(safe_snip(js), language="json")
            except Exception as e:
                st.error(str(e))

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

    st.markdown("#### 1) Universe (disabled)")
    st.caption("Polygon/universe fetch is disabled in Schwab-only mode. Use manual symbols.")

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
