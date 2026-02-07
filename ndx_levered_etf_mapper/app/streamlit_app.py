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

from etf_mapper.feeds import (
    StubHaltsFeed,
    StubNewsFeed,
    StubCalendarFeed,
    StubEarningsFeed,
    StubFilingsFeed,
    StubActivesFeed,
    StubInternalsFeed,
    WebHaltsFeed,
)

from ladder_styles import style_ladder_with_changes
from local_oauth import CallbackServerState, ensure_localhost_cert, start_https_callback_server, stop_callback_server
from schwab_diagnostics import safe_snip
from flight_recorder import FlightRecorder, make_session_id
from debug_bundle import create_debug_bundle
from dotenv import load_dotenv
from pyvis.network import Network
from streamlit_autorefresh import st_autorefresh

# PDF exports (reportlab)
from io import BytesIO
import time
import zipfile

from etf_mapper.schwab import SchwabAPI, SchwabConfig
from etf_mapper.config import load_schwab_secrets
from etf_mapper.spade_checks import check_option_chain, check_price_history, summarize

# Schwab API removed (replaced with Schwab Market Data)

# Schwab-only build (no Polygon universe builder).
from etf_mapper.build_prices import refresh_prices
from etf_mapper.build import refresh_universe as refresh_relations


# ---------- App boot ----------
st.set_page_config(page_title="Market Hub", layout="wide")

# Flight recorder (local logs)
if "session_id" not in st.session_state:
    st.session_state["session_id"] = make_session_id()
if "rec" not in st.session_state:
    st.session_state["rec"] = FlightRecorder(_data_dir(), session_id=st.session_state["session_id"])

# Settings load (local-only)
if "_settings_loaded" not in st.session_state:
    st.session_state["_settings_loaded"] = True
    loaded = _settings_defaults() | _load_settings()
    for k, v in loaded.items():
        st.session_state.setdefault(k, v)

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

  /* Casino mode hooks */
  .casino-wrap {
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 14px;
    padding: 0.75rem 0.9rem;
    background: rgba(0,0,0,0.25);
    backdrop-filter: blur(6px);
  }

  .ticker-tape {
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 12px;
    padding: 0.45rem 0.7rem;
    background: linear-gradient(90deg, rgba(14,16,24,0.90), rgba(8,10,18,0.90));
    overflow: hidden;
    white-space: nowrap;
    box-shadow: 0 0 22px rgba(0,255,209,0.05);
  }

  .ticker-tape .marquee {
    display: inline-block;
    padding-left: 100%;
    animation: marquee 18s linear infinite;
  }

  @keyframes marquee {
    0% { transform: translateX(0); }
    100% { transform: translateX(-100%); }
  }

  .neon-green { color: #33ffcc; text-shadow: 0 0 12px rgba(51,255,204,0.35); }
  .neon-red   { color: #ff4d6d; text-shadow: 0 0 12px rgba(255,77,109,0.25); }
  .neon-gold  { color: #ffd166; text-shadow: 0 0 12px rgba(255,209,102,0.18); }
  .neon-blue  { color: #74c0fc; text-shadow: 0 0 12px rgba(116,192,252,0.20); }

  .price-card {
    border-radius: 16px;
    padding: 0.85rem 1.0rem;
    border: 1px solid rgba(255,255,255,0.12);
    background: radial-gradient(1200px 160px at 15% 20%, rgba(51,255,204,0.10), transparent 55%),
                radial-gradient(800px 220px at 85% 30%, rgba(255,209,102,0.08), transparent 60%),
                rgba(8,10,18,0.65);
  }

  .price-big { font-size: 2.1rem; font-weight: 750; letter-spacing: 0.2px; }
  .muted { color: #9ca3af; }
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


def _settings_path() -> Path:
    return _data_dir() / "app_settings.json"


def _scanners_dir() -> Path:
    return _data_dir() / "scanners"


def _decisions_path() -> Path:
    return _data_dir() / "decisions.json"


def _load_decisions() -> dict:
    p = _decisions_path()
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _save_decisions(obj: dict) -> None:
    try:
        p = _decisions_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        pass


def _settings_defaults() -> dict:
    # Keep this small and non-secret.
    return {
        "selected_ticker": "QQQ",
        "watchlist": "QQQ,SPY,TSLA,AAPL,NVDA,/ES",
        "event_mode": "Normal",
        "api_budget_cap": 60,
        "dash_hot_spark_window": "1h",
        "heat_rv_on": False,
        "heat_rv_method": "returns",
        "heat_rv_k": 0.35,
        "heat_weights": _default_heat_weights(),
        # Scanner UI
        "scanner_preset": "Watchlist",
        "scanner_custom_symbols": "",
        "scanner_max_symbols": 80,
    }


def _load_settings() -> dict:
    p = _settings_path()
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _save_settings(obj: dict) -> None:
    try:
        p = _settings_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        pass


def _settings_snapshot() -> dict:
    # Single source of truth: pull from session_state.
    keys = list(_settings_defaults().keys())
    snap = {}
    for k in keys:
        v = st.session_state.get(k)
        # Only store JSON-safe items.
        try:
            json.dumps(v)
            snap[k] = v
        except Exception:
            pass
    return snap


def _autosave_settings() -> None:
    """Auto-save settings when they change."""
    defaults = _settings_defaults()
    cur = defaults | _settings_snapshot()
    try:
        blob = json.dumps(cur, sort_keys=True)
    except Exception:
        return

    last = st.session_state.get("_settings_last_blob")
    if last == blob:
        return

    _save_settings(cur)
    st.session_state["_settings_last_blob"] = blob


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


@st.cache_data(show_spinner=False, ttl=90)
def _sparkline_history(ticker: str, *, window: str = "1h") -> pd.DataFrame:
    """Cached micro-history for sparklines.

    window:
      - "1h": last 60 x 1m bars
      - "4h": last 240 x 1m bars

    Cached to reduce Schwab call volume (user preference: 60–120s).
    """
    if _budget_blocked():
        return pd.DataFrame()

    api = _schwab_api()
    if api is None:
        return pd.DataFrame()

    tkr = _normalize_ticker(ticker)
    if not tkr:
        return pd.DataFrame()

    _budget_note_call(1)
    try:
        js = api.price_history(
            tkr,
            period_type="day",
            period=1,
            frequency_type="minute",
            frequency=1,
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
        return pd.DataFrame()

    for c in ["open", "high", "low", "close", "volume"]:
        if c not in df.columns:
            df[c] = None

    df = df.dropna(subset=["date"])  # type: ignore

    n = 60 if window == "1h" else 240
    df = df.sort_values("date").tail(int(n))

    return df[["date", "open", "high", "low", "close", "volume"]]


def _realized_vol_1m(df1m: pd.DataFrame, *, method: str = "returns", lookback: int = 60) -> Optional[float]:
    """Compute a small realized-vol proxy from 1m candles.

    method:
      - returns: stddev of 1m log returns (lookback bars)
      - atr: ATR-ish using true range / close (lookback bars)
    Returns a scalar (not annualized)."""
    if df1m is None or df1m.empty:
        return None

    df = df1m.dropna(subset=["close"]).copy()
    df = df.sort_values("date").tail(int(lookback) + 1)
    if len(df) < max(10, int(lookback) // 3):
        return None

    try:
        close = df["close"].astype(float)
    except Exception:
        return None

    if method == "atr":
        # True range, normalized by close
        for c in ["high", "low"]:
            if c not in df.columns:
                return None
        try:
            high = df["high"].astype(float)
            low = df["low"].astype(float)
        except Exception:
            return None

        prev_close = close.shift(1)
        tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        tr = tr.dropna()
        if tr.empty:
            return None

        # Mean TR over lookback, normalized by last close
        tr_mean = float(tr.tail(int(lookback)).mean())
        last_close = float(close.dropna().iloc[-1])
        if last_close <= 0:
            return None
        return tr_mean / last_close

    # default: returns
    r = close.astype(float).pct_change().dropna()
    r = r.tail(int(lookback))
    if r.empty:
        return None
    try:
        return float(r.std(ddof=0))
    except Exception:
        return None


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


def _plot_sparkline(dfp: pd.DataFrame, *, height: int = 90) -> go.Figure:
    df = dfp.copy()
    if df.empty or "date" not in df.columns or "close" not in df.columns:
        fig = go.Figure()
        fig.update_layout(height=height, margin=dict(l=4, r=4, t=6, b=6), template="plotly_dark")
        return fig

    fig = go.Figure(
        data=[
            go.Scatter(
                x=df["date"],
                y=df["close"].astype(float),
                mode="lines",
                line=dict(width=1.6, color="#33ffcc"),
                fill="tozeroy",
                fillcolor="rgba(51,255,204,0.10)",
                hoverinfo="skip",
            )
        ]
    )
    fig.update_layout(
        height=height,
        margin=dict(l=4, r=4, t=6, b=6),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        template="plotly_dark",
    )
    return fig


def _fetch_history(ticker: str, tos_style: str) -> pd.DataFrame:
    """Live intraday candles via Schwab Market Data.

    Scope is intentionally reduced to 1-minute bars:
      - 1m over 3D
      - 1m over ~4H (today)

    The UI still calls this "TOS-like" but the source is Schwab.

    NOTE: this function is intentionally *not cached* because it is used for
    interactive focus charts. For tiny sparklines, use `_sparkline_history()`.
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


@st.cache_data(show_spinner=False, ttl=30)
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


@st.cache_data(show_spinner=False, ttl=15)
def _schwab_quote(ticker: str) -> dict:
    """Fetch a live quote snapshot from Schwab.

    Returned dict is normalized and includes best-effort timestamps so the UI can show "data age".

    Guardrails:
      - default quote TTL is 15s
      - budget cap enforced (60 calls/min default)
      - on suspected 429, a short cooldown is applied
    """
    tkr = _normalize_ticker(ticker)
    if not tkr:
        return {}

    if _budget_blocked():
        return {"symbol": tkr, "raw": {}, "_blocked": True}

    api = _schwab_api()
    if api is None:
        return {}

    _budget_note_call(1)
    try:
        js = api.quotes([tkr])
    except Exception as e:
        if _looks_rate_limited(str(e)):
            # exponential-ish cooldown based on how far over budget we are
            over = max(0, _budget_calls_last_minute() - _budget_cap_per_minute())
            _budget_set_cooldown(min(90.0, 10.0 + 5.0 * over))
        return {}

    if isinstance(js, dict):
        rec = js.get(tkr) or js.get(tkr.upper()) or js.get("quotes", {}).get(tkr) or js
    else:
        rec = {}

    if not isinstance(rec, dict):
        return {}

    # Common time fields (Schwab schemas vary). Keep them if present.
    # Prefer "...InLong" (epoch ms) when available.
    time_keys = [
        "quoteTimeInLong",
        "tradeTimeInLong",
        "regularMarketTradeTimeInLong",
        "regularMarketLastPriceTimeInLong",
    ]
    ts_ms = None
    for k in time_keys:
        v = rec.get(k)
        if isinstance(v, (int, float)) and v > 0:
            ts_ms = int(v)
            break

    out = {
        "symbol": rec.get("symbol") or tkr,
        "last": rec.get("lastPrice") or rec.get("last") or rec.get("lastPriceInDouble"),
        "mark": rec.get("mark") or rec.get("markPrice") or rec.get("markPriceInDouble"),
        "bid": rec.get("bidPrice") or rec.get("bid"),
        "ask": rec.get("askPrice") or rec.get("ask"),
        "volume": rec.get("totalVolume") or rec.get("volume"),
        "netChange": rec.get("netChange"),
        "netPct": rec.get("netPercentChangeInDouble") or rec.get("netPercentChange"),
        "ts_ms": ts_ms,
        "raw": rec,
    }

    # Trim raw a bit (some schemas are huge)
    if isinstance(out.get("raw"), dict) and len(out["raw"]) > 120:
        # keep only a subset if it's gigantic
        keep = set(
            [
                "symbol",
                "description",
                "assetType",
                "exchangeName",
                "quoteType",
                "lastPrice",
                "mark",
                "bidPrice",
                "askPrice",
                "totalVolume",
            ]
            + time_keys
        )
        out["raw"] = {k: rec.get(k) for k in keep if k in rec}

    return out


def _normalize_ticker(ticker: str) -> str:
    return str(ticker or "").upper().strip()


def _default_heat_weights() -> dict:
    # Weighting is per event mode (configurable in Scanner).
    # Values should sum to 1.0.
    return {
        "Normal": {"w_move": 0.62, "w_dollar_vol": 0.38},
        "Fed day": {"w_move": 0.70, "w_dollar_vol": 0.30},
        "CPI/NFP day": {"w_move": 0.70, "w_dollar_vol": 0.30},
        "Earnings week": {"w_move": 0.58, "w_dollar_vol": 0.42},
    }


def _heat_weights_for_mode(event_mode: str) -> dict:
    w = st.session_state.get("heat_weights")
    if not isinstance(w, dict):
        w = _default_heat_weights()
        st.session_state["heat_weights"] = w

    em = str(event_mode or "Normal")
    cur = w.get(em)
    if not isinstance(cur, dict):
        cur = w.get("Normal") or {"w_move": 0.62, "w_dollar_vol": 0.38}

    # Defensive normalization
    try:
        wm = float(cur.get("w_move", 0.62))
        wd = float(cur.get("w_dollar_vol", 0.38))
    except Exception:
        wm, wd = 0.62, 0.38

    tot = wm + wd
    if tot <= 0:
        wm, wd = 0.62, 0.38
        tot = 1.0

    wm /= tot
    wd /= tot

    return {"w_move": wm, "w_dollar_vol": wd}


@st.cache_data(show_spinner=False, ttl=15)
def _schwab_account_numbers() -> list[dict]:
    """Return Schwab accountNumbers list (best-effort)."""
    api = _schwab_api()
    if api is None:
        return []
    try:
        js = api.account_numbers()
    except Exception:
        return []
    if isinstance(js, list):
        return [x for x in js if isinstance(x, dict)]
    if isinstance(js, dict) and isinstance(js.get("accounts"), list):
        return [x for x in js.get("accounts") if isinstance(x, dict)]
    return []


@st.cache_data(show_spinner=False, ttl=15)
def _schwab_account_details(account_hash: str, *, fields: str = "positions") -> dict:
    api = _schwab_api()
    if api is None:
        return {}
    try:
        js = api.account_details(account_hash, fields=fields)
    except Exception:
        return {}
    return js if isinstance(js, dict) else {}


def _looks_rate_limited(err: str) -> bool:
    e = (err or "").lower()
    return any(x in e for x in ["429", "too many", "rate limit", "ratelimit", "throttle", "temporarily blocked"])


def _budget_note_call(n: int = 1) -> None:
    """Track API call timestamps for a rolling calls/min counter."""
    now = time.time()
    st.session_state.setdefault("api_calls_ts", [])
    ts = st.session_state.get("api_calls_ts")
    if not isinstance(ts, list):
        ts = []
    # add n timestamps
    for _ in range(max(1, int(n))):
        ts.append(now)
    # prune >60s
    cutoff = now - 60.0
    ts = [t for t in ts if isinstance(t, (int, float)) and t >= cutoff]
    st.session_state["api_calls_ts"] = ts


def _budget_calls_last_minute() -> int:
    now = time.time()
    ts = st.session_state.get("api_calls_ts")
    if not isinstance(ts, list):
        return 0
    cutoff = now - 60.0
    return int(sum(1 for t in ts if isinstance(t, (int, float)) and t >= cutoff))


def _budget_cap_per_minute() -> int:
    return int(st.session_state.get("api_budget_cap", 60))


def _budget_blocked() -> bool:
    # If we recently hit a rate limit, honor a cooldown.
    until = st.session_state.get("api_cooldown_until")
    try:
        if until is not None and time.time() < float(until):
            return True
    except Exception:
        pass

    # Soft cap: if over cap, treat as blocked.
    return _budget_calls_last_minute() >= _budget_cap_per_minute()


def _budget_set_cooldown(seconds: float) -> None:
    st.session_state["api_cooldown_until"] = time.time() + float(seconds)


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


def _local_timestamp_compact() -> str:
    """America/New_York timestamp for filenames."""
    from zoneinfo import ZoneInfo

    tz = ZoneInfo("America/New_York")
    return datetime.now(tz).strftime("%Y-%m-%d_%H%M")


def _pdf_exposure_summary(
    expo: pd.DataFrame,
    *,
    top_n: int = 20,
    title: str = "Exposure summary",
) -> bytes:
    """Generate a 1-page Exposure PDF (redacted: no account ids)."""
    # Lazy import so app runs even if reportlab missing.
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.piecharts import Pie

    df = expo.copy() if isinstance(expo, pd.DataFrame) else pd.DataFrame()
    if df.empty:
        df = pd.DataFrame(columns=["symbol", "market_value", "pct", "day_pl"])

    # normalize expected columns
    for c in ["market_value", "pct", "day_pl"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = None
    if "symbol" not in df.columns:
        df["symbol"] = ""

    df = df.sort_values("market_value", ascending=False)
    total_mv = float(df["market_value"].fillna(0.0).sum())

    top = df.head(int(top_n)).copy()
    other_mv = float(df.iloc[int(top_n):]["market_value"].fillna(0.0).sum()) if len(df) > int(top_n) else 0.0

    # compute pct if absent/garbage
    if "pct" not in top.columns or top["pct"].isna().all():
        top["pct"] = (top["market_value"].fillna(0.0) / total_mv) if total_mv else 0.0

    # build pie dataset
    pie_labels = list(top["symbol"].astype(str))
    pie_vals = [float(x) for x in top["market_value"].fillna(0.0).tolist()]
    if other_mv > 0:
        pie_labels.append("Other")
        pie_vals.append(other_mv)

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    w, h = letter

    # header
    ts = _local_timestamp_compact().replace("_", " ")
    c.setFont("Helvetica-Bold", 16)
    c.drawString(0.75 * inch, h - 0.75 * inch, title)
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.grey)
    c.drawRightString(w - 0.75 * inch, h - 0.72 * inch, f"Generated {ts} ET")
    c.setFillColor(colors.black)

    # totals row
    c.setFont("Helvetica", 11)
    c.drawString(0.75 * inch, h - 1.05 * inch, f"Total market value: ${total_mv:,.0f}")

    # pie chart
    d = Drawing(320, 220)
    pie = Pie()
    pie.x = 10
    pie.y = 10
    pie.width = 200
    pie.height = 200
    pie.data = pie_vals
    pie.labels = ["" for _ in pie_labels]  # keep clean; labels handled in table
    pie.slices.strokeWidth = 0.5
    pie.slices.strokeColor = colors.white
    d.add(pie)

    from reportlab.graphics import renderPDF

    renderPDF.draw(d, c, 0.75 * inch, h - 3.6 * inch)

    # table (top N)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(3.4 * inch, h - 1.45 * inch, f"Top {min(int(top_n), len(df))} holdings")

    c.setFont("Helvetica", 9)
    y = h - 1.70 * inch
    c.setFillColor(colors.grey)
    c.drawString(3.4 * inch, y, "Symbol")
    c.drawRightString(w - 1.65 * inch, y, "MV")
    c.drawRightString(w - 0.90 * inch, y, "%")
    c.setFillColor(colors.black)

    y -= 0.14 * inch
    for r in top.itertuples(index=False):
        sym = str(getattr(r, "symbol", ""))
        mv = getattr(r, "market_value", 0.0)
        pct = getattr(r, "pct", 0.0)
        try:
            mv_f = float(mv) if mv == mv else 0.0
        except Exception:
            mv_f = 0.0
        try:
            pct_f = float(pct) * 100.0 if pct == pct else 0.0
        except Exception:
            pct_f = 0.0

        c.drawString(3.4 * inch, y, sym)
        c.drawRightString(w - 1.65 * inch, y, f"${mv_f:,.0f}")
        c.drawRightString(w - 0.90 * inch, y, f"{pct_f:,.2f}%")
        y -= 0.14 * inch
        if y < 0.9 * inch:
            break

    c.setFont("Helvetica-Oblique", 8)
    c.setFillColor(colors.grey)
    c.drawString(0.75 * inch, 0.65 * inch, "Redacted export: no account numbers/hashes.")
    c.setFillColor(colors.black)

    c.showPage()
    c.save()
    return buf.getvalue()


def _pdf_scanner_snapshot(
    sdf2: pd.DataFrame,
    *,
    title: str = "Scanner snapshot",
    top_n: int = 30,
) -> bytes:
    """Generate a 1-page Scanner PDF (rankings + hot list)."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas

    df = sdf2.copy() if isinstance(sdf2, pd.DataFrame) else pd.DataFrame()
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    w, h = letter

    ts = _local_timestamp_compact().replace("_", " ")
    c.setFont("Helvetica-Bold", 16)
    c.drawString(0.75 * inch, h - 0.75 * inch, title)
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.grey)
    c.drawRightString(w - 0.75 * inch, h - 0.72 * inch, f"Generated {ts} ET")
    c.setFillColor(colors.black)

    if df.empty:
        c.setFont("Helvetica", 11)
        c.drawString(0.75 * inch, h - 1.2 * inch, "No scanner data available.")
        c.showPage()
        c.save()
        return buf.getvalue()

    # normalize
    for ccol in ["px", "chg_%", "dollar_vol", "heat"]:
        if ccol in df.columns:
            df[ccol] = pd.to_numeric(df[ccol], errors="coerce")

    # choose top by heat (already pre-sorted in UI often, but be explicit)
    if "heat" in df.columns:
        df = df.sort_values("heat", ascending=False)

    top = df.head(int(top_n)).copy()

    c.setFont("Helvetica-Bold", 11)
    c.drawString(0.75 * inch, h - 1.15 * inch, f"Top {min(int(top_n), len(df))} by heat")

    c.setFont("Helvetica", 8.5)
    y = h - 1.35 * inch
    c.setFillColor(colors.grey)
    c.drawString(0.75 * inch, y, "Symbol")
    c.drawRightString(2.35 * inch, y, "Px")
    c.drawRightString(3.25 * inch, y, "%")
    c.drawRightString(4.65 * inch, y, "$Vol")
    c.drawRightString(5.60 * inch, y, "Heat")
    c.setFillColor(colors.black)

    y -= 0.14 * inch
    for r in top.itertuples(index=False):
        sym = str(getattr(r, "symbol", ""))
        px = getattr(r, "px", None)
        chgp = getattr(r, "chg_%", None)
        dv = getattr(r, "dollar_vol", None)
        heat = getattr(r, "heat", None)

        def _f(x, fmt, default="—"):
            try:
                if x is None or (isinstance(x, float) and x != x):
                    return default
                return fmt.format(float(x))
            except Exception:
                return default

        c.drawString(0.75 * inch, y, sym)
        c.drawRightString(2.35 * inch, y, _f(px, "{:.2f}"))
        c.drawRightString(3.25 * inch, y, _f(chgp, "{:+.2f}%"))
        c.drawRightString(4.65 * inch, y, _f(dv, "${:,.0f}"))
        c.drawRightString(5.60 * inch, y, _f(heat, "{:.1f}"))
        y -= 0.14 * inch
        if y < 1.1 * inch:
            break

    c.setFont("Helvetica-Oblique", 8)
    c.setFillColor(colors.grey)
    c.drawString(0.75 * inch, 0.65 * inch, "Local snapshot export.")
    c.setFillColor(colors.black)

    c.showPage()
    c.save()
    return buf.getvalue()


# ---------- UI ----------

# -------- Sidebar (streamlined) --------
# Keep the sidebar focused on exploration; move operational controls behind an expander.
with st.sidebar:
    st.markdown("### Explore")

    # Event mode (affects scanner/dashboard defaults)
    st.session_state["event_mode"] = st.selectbox(
        "Event mode",
        ["Normal", "Fed day", "CPI/NFP day", "Earnings week"],
        index=["Normal", "Fed day", "CPI/NFP day", "Earnings week"].index(st.session_state.get("event_mode", "Normal")),
        key="event_mode",
    )

    with st.expander("Calculator", expanded=False):
        a = st.number_input("A", value=0.0, step=1.0, format="%.6f")
        op = st.selectbox("Op", ["+", "-", "*", "/", "%"], index=0)
        b = st.number_input("B", value=0.0, step=1.0, format="%.6f")
        out = None
        try:
            if op == "+":
                out = a + b
            elif op == "-":
                out = a - b
            elif op == "*":
                out = a * b
            elif op == "/":
                out = None if b == 0 else a / b
            elif op == "%":
                out = None if b == 0 else a % b
        except Exception:
            out = None
        st.markdown(f"**Result:** `{out if out is not None else '—'}`")

    # Advanced settings (collapsed by default)
    with st.expander("Advanced", expanded=False):
        st.text_input("Data directory", value=st.session_state.get("data_dir", "data"), key="data_dir")

        # Schwab-only mode: hide universe provider.
        price_provider = st.selectbox("Price provider", ["schwab"], index=0)
        price_start = st.text_input("Price start", value="2024-01-01")
        price_limit = st.slider("Price fetch limit", min_value=25, max_value=1000, value=200, step=25)

    with st.expander("Live", expanded=False):
        st.caption("Live mode is Schwab-only. Enable to refresh the UI every 60 seconds.")
        auto_refresh = st.toggle("Auto-refresh (60s)", value=True)
        if auto_refresh:
            # lightweight refresh loop (useful during market hours)
            st_autorefresh(interval=60 * 1000, key="autorefresh")

    with st.expander("Guardrails", expanded=False):
        st.caption("Request budget + rate-limit backoff. Default: 60 calls/min soft cap.")
        st.session_state.setdefault("api_budget_cap", 60)
        st.session_state["api_budget_cap"] = st.slider(
            "API soft cap (calls/min)",
            min_value=20,
            max_value=240,
            value=int(st.session_state.get("api_budget_cap", 60)),
            step=10,
        )

        calls = _budget_calls_last_minute()
        cap = _budget_cap_per_minute()
        blocked = _budget_blocked()
        st.write({"calls_last_minute": calls, "cap": cap, "blocked": bool(blocked)})

        if st.button("Reset call counter / clear cooldown", key="guardrails_reset"):
            st.session_state["api_calls_ts"] = []
            st.session_state["api_cooldown_until"] = None
            st.rerun()

    with st.expander("Secrets", expanded=False):
        # Schwab-only UI.
        secrets = load_schwab_secrets(_data_dir())
        st.write(
            {
                "schwab_secrets_file": str((_data_dir() / "schwab_secrets.local.json").resolve()),
                "schwab_configured": bool(secrets),
                "schwab_tokens_present": (_data_dir() / "schwab_tokens.json").exists(),
            }
        )

# Auto-save settings on change
_autosave_settings()

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
    st.session_state.setdefault("selected_ticker", "QQQ")
    selected = st.text_input("Ticker", value=st.session_state.get("selected_ticker", "QQQ"), key="selected_ticker").upper().strip()

if not selected:
    st.warning("Enter a ticker symbol (e.g. TSLA, TSLL, QQQ).")
    st.stop()

# No universe table filtering in Schwab-only mode.

# Header row: title left, provider description right (uses otherwise-empty space)
headerL, headerR = st.columns([0.42, 0.58], vertical_alignment="top")
with headerL:
    st.markdown("# Market Hub")
    st.caption("Schwab-only • Live quotes • 1m candles • Options ladder • Exposure")

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

# Top-of-page rolling tape (watchlist) — Schwab-only
watch = st.session_state.get("watchlist", "QQQ,SPY,TSLA,AAPL,NVDA,/ES")
watch = st.text_input("Watchlist (comma-separated)", value=watch, key="watchlist")
watch_syms = [s.strip().upper() for s in str(watch).split(",") if s.strip()]
watch_syms = watch_syms[:12]
if watch_syms:
    bits = []
    for s in watch_syms:
        q = _schwab_quote(s)
        px = q.get("mark") or q.get("last")
        try:
            pxs = f"{float(px):,.2f}" if px not in (None, "") else "—"
        except Exception:
            pxs = str(px) if px is not None else "—"
        bits.append(f"{s} ${pxs}")

    st.markdown(
        f"""
<div class='ticker-tape'>
  <span class='marquee'>
    <span class='neon-gold'>TAPE</span> • {'  •  '.join(bits)}
  </span>
</div>
""",
        unsafe_allow_html=True,
    )


def _beta_binomial_up_prob(prices_1m: pd.DataFrame, lookback_bars: int = 240) -> dict:
    """Toy Bayesian read: probability next bar is up.

    Uses Beta-Binomial on sign(close_t - close_{t-1}) over last N bars.
    Prior is Beta(1,1) (uniform).

    Returns dict with posterior parameters and a 90% credible interval.
    """
    df = prices_1m.copy()
    if df.empty or "close" not in df.columns:
        return {}

    df = df.dropna(subset=["close"]).tail(int(lookback_bars) + 1)
    if len(df) < 10:
        return {}

    d = df["close"].astype(float).diff().dropna()
    ups = int((d > 0).sum())
    downs = int((d <= 0).sum())

    a0, b0 = 1.0, 1.0
    a1 = a0 + ups
    b1 = b0 + downs

    # mean = a/(a+b); approximate interval via quantiles if scipy exists, else normal approx
    mean = a1 / (a1 + b1)

    ci = None
    try:
        from scipy.stats import beta  # type: ignore

        lo = float(beta.ppf(0.05, a1, b1))
        hi = float(beta.ppf(0.95, a1, b1))
        ci = (lo, hi)
    except Exception:
        # normal approx to beta
        var = (a1 * b1) / (((a1 + b1) ** 2) * (a1 + b1 + 1.0))
        sd = float(var ** 0.5)
        lo = max(0.0, mean - 1.645 * sd)
        hi = min(1.0, mean + 1.645 * sd)
        ci = (float(lo), float(hi))

    return {
        "lookback_bars": int(lookback_bars),
        "ups": ups,
        "downs": downs,
        "posterior_a": float(a1),
        "posterior_b": float(b1),
        "p_up_mean": float(mean),
        "p_up_ci90": [float(ci[0]), float(ci[1])],
    }


def _backtest_1m(prices_1m: pd.DataFrame, strategy: str, *, fee_bps: float = 0.0, slippage_bps: float = 0.0) -> pd.DataFrame:
    """Very small backtest harness (single-position, long/flat) on 1m closes.

    Strategies (toy):
      - mean_reversion: buy when close below SMA by k*std, exit at SMA
      - breakout: buy when close breaks N-bar high, exit on N-bar low

    Returns a dataframe with equity curve columns.
    """
    df = prices_1m.copy()
    if df.empty:
        return pd.DataFrame()

    df = df.dropna(subset=["date", "close"]).copy()
    df = df.sort_values("date")
    df["close"] = df["close"].astype(float)

    # returns
    df["ret"] = df["close"].pct_change().fillna(0.0)

    pos = pd.Series(0.0, index=df.index)

    if strategy == "mean_reversion":
        n = 60
        k = 1.25
        sma = df["close"].rolling(n).mean()
        sd = df["close"].rolling(n).std()
        z = (df["close"] - sma) / sd
        entry = z < -k
        exit_ = df["close"] >= sma

        in_pos = False
        for i in range(len(df)):
            if not in_pos and bool(entry.iloc[i]) and pd.notna(sma.iloc[i]) and pd.notna(sd.iloc[i]) and sd.iloc[i] != 0:
                in_pos = True
            elif in_pos and bool(exit_.iloc[i]):
                in_pos = False
            pos.iloc[i] = 1.0 if in_pos else 0.0

    elif strategy == "breakout":
        n = 45
        hh = df["close"].rolling(n).max().shift(1)
        ll = df["close"].rolling(n).min().shift(1)
        entry = df["close"] > hh
        exit_ = df["close"] < ll

        in_pos = False
        for i in range(len(df)):
            if not in_pos and bool(entry.iloc[i]) and pd.notna(hh.iloc[i]):
                in_pos = True
            elif in_pos and bool(exit_.iloc[i]) and pd.notna(ll.iloc[i]):
                in_pos = False
            pos.iloc[i] = 1.0 if in_pos else 0.0

    else:
        return pd.DataFrame()

    # apply costs on position changes
    df["pos"] = pos
    df["pos_chg"] = df["pos"].diff().abs().fillna(0.0)

    cost = (fee_bps + slippage_bps) / 10000.0
    df["cost"] = df["pos_chg"] * cost

    # strategy return: position held from previous bar close to current close
    df["strat_ret"] = df["pos"].shift(1).fillna(0.0) * df["ret"] - df["cost"]
    df["equity"] = (1.0 + df["strat_ret"]).cumprod()

    # drawdown
    df["equity_peak"] = df["equity"].cummax()
    df["drawdown"] = df["equity"] / df["equity_peak"] - 1.0

    return df

# Context menus
(tab_dash, tab_scanner, tab_halts, tab_signals, tab_overview, tab_rel, tab_opts, tab_cart, tab_casino, tab_exposure, tab_exports, tab_decisions, tab_admin) = st.tabs(
    [
        "Dashboard",
        "Scanner",
        "Halts",
        "Signals",
        "Overview",
        "Relations",
        "Options",
        "Position Builder",
        "Casino Lab",
        "Exposure",
        "Exports",
        "Decisions",
        "Admin",
    ]
)

with tab_dash:
    st.subheader("Dashboard")
    st.caption("Tiles: watchlist • selected chart • countdown • alerts • headlines. Schwab-only.")


def _parse_symbols(s: str) -> list[str]:
    raw = str(s or "")
    # allow commas, spaces, newlines
    parts = [p.strip().upper() for p in raw.replace("\n", ",").replace(" ", ",").split(",")]
    out = []
    for p in parts:
        if not p:
            continue
        if p not in out:
            out.append(p)
    return out


def _hotlist_path() -> Path:
    return _data_dir() / "scanner_hotlist.json"


def _load_hotlist() -> list[str]:
    try:
        p = _hotlist_path()
        if not p.exists():
            return []
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return [str(x).upper().strip() for x in obj if str(x).strip()]
    except Exception:
        return []
    return []


def _save_hotlist(syms: list[str]) -> None:
    try:
        p = _hotlist_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        clean = []
        for x in syms:
            t = str(x).upper().strip()
            if t and t not in clean:
                clean.append(t)
        p.write_text(json.dumps(clean, indent=2), encoding="utf-8")
    except Exception:
        pass


    # Ensure Hot List is available across tabs (Dashboard + Scanner).
    st.session_state.setdefault("scanner_hotlist", _load_hotlist())

    # ---- Watchlist tile ----
    dL, dM, dR = st.columns([0.36, 0.40, 0.24], gap="large")

    with dL:
        st.markdown("### Watchlist")
        watch = st.session_state.get("watchlist", "QQQ,SPY,TSLA,AAPL,NVDA,/ES")
        watch = st.text_input("Symbols", value=watch, key="watchlist")
        syms = [s.strip().upper() for s in str(watch).split(",") if s.strip()][:20]

        rows = []
        for s in syms:
            q = _schwab_quote(s)
            px = q.get("mark") or q.get("last")
            net = q.get("netChange")
            netp = q.get("netPct")
            rows.append({"symbol": s, "px": px, "chg": net, "chg_%": netp})

        wdf = pd.DataFrame(rows)
        for c in ["px", "chg", "chg_%"]:
            if c in wdf.columns:
                wdf[c] = pd.to_numeric(wdf[c], errors="coerce")

        st.dataframe(
            wdf,
            use_container_width=True,
            height=360,
            hide_index=True,
        )

        st.caption("Tip: click a row isn't supported by Streamlit dataframe yet; copy symbol into the Ticker box on the left.")

        st.markdown("### Hot List")
        st.caption("Pins from Scanner (saved locally).")

        # Read from session_state if present; fall back to persisted file.
        hot = st.session_state.get("scanner_hotlist")
        if hot is None:
            hot = _load_hotlist()
            st.session_state["scanner_hotlist"] = hot

        hot_syms = [s for s in (hot or []) if str(s).strip()][:25]
        if not hot_syms:
            st.write("(empty)")
        else:
            spark_window = st.selectbox(
                "Sparkline window",
                ["1h", "4h", "off"],
                index=0,
                help="Default is 1h. 4h is available if you need more context. Cached ~90s to reduce calls.",
                key="dash_hot_spark_window",
            )

            for sym in hot_syms:
                q = _schwab_quote(sym)
                px = q.get("mark") or q.get("last")
                netp = q.get("netPct")

                try:
                    pxs = f"${float(px):,.2f}" if px not in (None, "") else "—"
                except Exception:
                    pxs = "—"

                try:
                    cps = f"{float(netp):+.2f}%" if netp == netp else "—"
                except Exception:
                    cps = "—"

                rowL, rowR = st.columns([0.28, 0.72], vertical_alignment="center")
                rowL.markdown(f"**{sym}**  \\n{pxs}  \\n{cps}")

                if spark_window != "off":
                    df_s = _sparkline_history(sym, window=spark_window)
                    if df_s.empty:
                        rowR.caption("(no 1m candles)")
                    else:
                        rowR.plotly_chart(_plot_sparkline(df_s), use_container_width=True)
                else:
                    rowR.caption("sparklines off")

    # ---- Selected chart tile ----
    with dM:
        st.markdown("### Selected")
        q = _schwab_quote(selected)
        px = q.get("mark") or q.get("last")
        try:
            pxs = f"${float(px):,.2f}" if px not in (None, "") else "—"
        except Exception:
            pxs = str(px) if px is not None else "—"

        st.markdown(f"<div class='price-card'><div class='muted'>{selected}</div><div class='price-big neon-green'>{pxs}</div></div>", unsafe_allow_html=True)

        tf = st.selectbox("Timeframe", ["1m (4H)", "1m (3D)"], index=0, key="dash_tf")
        dfp = _fetch_history(selected, tf)
        if dfp.empty:
            st.warning("No 1m candles.")
        else:
            st.plotly_chart(_plot_candles(dfp, title=f"{selected} — {tf}"), use_container_width=True)

    # ---- Countdown + alerts + headlines tile ----
    with dR:
        st.markdown("### Countdown")

        # Countdown to margin-trading restart (Payne's discipline timer)
        from datetime import datetime
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/New_York")
        target = datetime(2026, 3, 14, 9, 30, 0, tzinfo=tz)
        now_dt = datetime.now(tz)
        delta = target - now_dt

        live_countdown = st.toggle("Live seconds", value=False, help="Off by default to keep things fast.")
        if live_countdown:
            st_autorefresh(interval=1000, key="countdown_1s")

        if delta.total_seconds() <= 0:
            st.success("It’s Mar 14 @ 9:30 — margin discipline timer is done.")
        else:
            s = int(delta.total_seconds())
            days = s // 86400
            s -= days * 86400
            hrs = s // 3600
            s -= hrs * 3600
            mins = s // 60
            secs = s - mins * 60

            st.markdown(
                f"<div class='price-card'><div class='muted'>Margin trading restarts</div>"
                f"<div class='price-big neon-gold'>{days}d {hrs:02d}h {mins:02d}m {secs:02d}s</div></div>",
                unsafe_allow_html=True,
            )
            st.caption("Target: 2026-03-14 09:30 America/New_York")

        st.markdown("### Alerts")
        st.caption(
            "Target state: create/read *Schwab-native* (TOS/Schwab mobile) alerts via Schwab API so Schwab handles SMS/push. "
            "I can’t confirm the alerts endpoints until we’re in your Schwab developer account docs."
        )

        with st.expander("Local alerts (temporary placeholder)", expanded=False):
            st.session_state.setdefault("alerts", [])
            a_sym = st.text_input("Symbol", value=selected, key="alert_sym")
            a_op = st.selectbox("Condition", [">=", "<=", ">", "<", "=="], index=0, key="alert_op")
            a_px = st.number_input("Trigger price", min_value=0.0, value=0.0, step=0.5, key="alert_px")
            a_exp_min = st.slider("Expires in (minutes)", 5, 24 * 60, 60, 5, key="alert_exp")

            if st.button("Add local alert"):
                exp_at = pd.Timestamp.now() + pd.Timedelta(minutes=int(a_exp_min))
                st.session_state["alerts"].append(
                    {
                        "symbol": str(a_sym).upper().strip(),
                        "op": a_op,
                        "price": float(a_px),
                        "expires_at": exp_at.isoformat(),
                        "armed": True,
                    }
                )
                st.success("Local alert added (UI only).")

            st.json(st.session_state.get("alerts", []))
            if st.button("Clear local alerts"):
                st.session_state["alerts"] = []
                st.rerun()

        st.markdown("### Headlines")
        st.caption("Placeholder for now (fastest path). Paste headlines you care about; later we’ll wire a feed.")
        st.session_state.setdefault("headlines", "")
        st.text_area("Headlines", key="headlines", height=180)

with tab_scanner:
    st.subheader("Scanner")
    em = st.session_state.get("event_mode", "Normal")
    st.caption(
        f"Broad scan scaffold. Quotes/candles come from Schwab. News/halts feeds are placeholders for now.  |  Event mode: {em}"
    )

    # User preference: scanner is OFF by default (avoid unnecessary API calls).
    scan_on = st.toggle("Enable scanning", value=bool(st.session_state.get("scan_on", False)), key="scan_on")
    if not scan_on:
        st.info("Scanner is off. Toggle 'Enable scanning' when you want to run it.")
        st.stop()

    # Persistent hot list (saved under data/)
    st.session_state.setdefault("scanner_hotlist", _load_hotlist())

    # Universe selection
    base = _parse_symbols(st.session_state.get("watchlist", "QQQ,SPY,TSLA,AAPL,NVDA"))
    actives_feed = StubActivesFeed()

    with st.expander("Universe", expanded=True):
        st.caption(
            "Start fast. Add symbols here. Later we can auto-source 'most active' and themed universes. "
            "Keep it under ~200 symbols if you want it to stay snappy."
        )
        preset = st.selectbox(
            "Preset",
            [
                "Watchlist",
                "Core liquid (starter)",
                "Semis / AI-ish (starter)",
                "Energy / Oil & Gas (starter)",
                "Market-wide actives (feed stub)",
                "Custom (paste)",
            ],
            index=[
                "Watchlist",
                "Core liquid (starter)",
                "Semis / AI-ish (starter)",
                "Energy / Oil & Gas (starter)",
                "Market-wide actives (feed stub)",
                "Custom (paste)",
            ].index(st.session_state.get("scanner_preset", "Watchlist")),
            key="scanner_preset",
        )
        st.caption(f"Actives feed status: {actives_feed.status().detail}")

        core_liquid = _parse_symbols("SPY,QQQ,IWM,DIA,TQQQ,SQQQ,TLT,GLD,SLV,USO,XLF,XLK,XLE,XLI,XLY,XLP")
        semis = _parse_symbols("NVDA,AMD,INTC,TSM,AVGO,QCOM,AMAT,LRCX,SMH,SOXL,SOXS")
        energy = _parse_symbols("XLE,CVX,XOM,OXY,SLB,HAL,BKR,UNG,BOIL,KOLD")

        custom = st.text_area(
            "Custom symbols",
            value=st.session_state.get("scanner_custom_symbols", ""),
            height=120,
            placeholder="TSLA, AAPL, /ES, ...",
            key="scanner_custom_symbols",
        )

        if preset == "Watchlist":
            uni = base
        elif preset == "Core liquid (starter)":
            uni = core_liquid
        elif preset == "Semis / AI-ish (starter)":
            uni = semis
        elif preset == "Energy / Oil & Gas (starter)":
            uni = energy
        elif preset == "Market-wide actives (feed stub)":
            try:
                adf = actives_feed.fetch_actives()
                if isinstance(adf, pd.DataFrame) and not adf.empty and "symbol" in adf.columns:
                    uni = _parse_symbols(",".join(adf["symbol"].astype(str).tolist()))
                else:
                    uni = []
            except Exception:
                uni = []
        else:
            uni = _parse_symbols(custom)

        max_n = st.slider(
            "Max symbols to scan",
            5,
            300,
            int(st.session_state.get("scanner_max_symbols", 80)),
            5,
            key="scanner_max_symbols",
        )
        uni = uni[: int(max_n)]

    # Scan (manual, by design)
    st.session_state.setdefault("scanner_ran", False)

    scan_cols = st.columns([0.25, 0.75], vertical_alignment="center")
    if scan_cols[0].button("Scan now", key="scan_now"):
        st.session_state["scanner_ran"] = True

    if _budget_blocked():
        st.warning("API budget hit (or cooldown). Scanner is paused. Clear cooldown in Sidebar → Guardrails.")

    sdf = st.session_state.get("scanner_last_sdf")
    if not isinstance(sdf, pd.DataFrame):
        sdf = pd.DataFrame()

    if st.session_state.get("scanner_ran") and (not _budget_blocked()):
        with st.spinner(f"Scanning {len(uni)} symbols via Schwab quotes…"):
            rows = []
            for s in uni:
                q = _schwab_quote(s)
                rec = q.get("raw") if isinstance(q.get("raw"), dict) else {}
                px = q.get("mark") or q.get("last")
                vol = q.get("volume")
                chg = q.get("netChange")
                chgp = q.get("netPct")

                rows.append(
                    {
                        "symbol": s,
                        "px": px,
                        "volume": vol,
                        "chg": chg,
                        "chg_%": chgp,
                        "assetType": rec.get("assetType"),
                        "exchange": rec.get("exchangeName"),
                    }
                )

            sdf = pd.DataFrame(rows)
            for c in ["px", "volume", "chg", "chg_%"]:
                if c in sdf.columns:
                    sdf[c] = pd.to_numeric(sdf[c], errors="coerce")
            if "px" in sdf.columns and "volume" in sdf.columns:
                sdf["dollar_vol"] = sdf["px"].fillna(0.0) * sdf["volume"].fillna(0.0)

            st.session_state["scanner_last_sdf"] = sdf

    if sdf.empty:
        st.info("Scanner is idle. Click 'Scan now' to run a scan.")
        # Create empty placeholder frame so downstream UI doesn't crash
        sdf = pd.DataFrame(columns=["symbol", "px", "volume", "chg", "chg_%", "dollar_vol"])
    st.markdown("### Rankings")
    metric = st.selectbox(
        "Rank by",
        ["heat", "dollar_vol", "volume", "chg_%", "abs_chg_%"],
        index=0,
        help="heat is a composite: abs % change + dollar volume rank (fast proxy for 'what's popping').",
    )
    # Derived columns
    sdf2 = sdf.dropna(subset=["symbol"]).copy()
    # Save for Exports tab
    st.session_state["last_scanner"] = sdf2.copy()

    sdf2["abs_chg_%"] = sdf2["chg_%"].abs()

    with st.expander("Heat score tuning", expanded=False):
        st.caption("Heat = weighted blend of (abs % move rank) + (dollar volume rank). Per-event-mode weights.")

        w = _heat_weights_for_mode(em)
        wm = st.slider(
            f"Move weight ({em})",
            min_value=0.0,
            max_value=1.0,
            value=float(w["w_move"]),
            step=0.01,
            key=f"heat_w_move_{em}",
        )
        wd = 1.0 - float(wm)
        st.write({"w_move": round(float(wm), 2), "w_dollar_vol": round(float(wd), 2)})

        # Optional realized-vol multiplier (throttled)
        st.session_state.setdefault("heat_rv_on", False)
        rv_on = st.toggle(
            "Realized-vol boost (uses 1m candles)",
            value=bool(st.session_state.get("heat_rv_on", False)),
            help="Default off. When on, we fetch 1m candles ONLY for Focus + Top 5 strip + Hot List.",
            key="heat_rv_on",
        )
        rv_method = st.selectbox(
            "RV method",
            ["returns", "atr"],
            index=0,
            help="returns = stddev of 1m returns; atr = ATR-ish true range / close.",
            key="heat_rv_method",
        )
        rv_k = st.slider(
            "RV multiplier strength (k)",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.get("heat_rv_k", 0.35)),
            step=0.05,
            help="Final heat = base_heat * (1 + k * rv_rank).",
            key="heat_rv_k",
        )

        # Persist back into session weights map
        st.session_state.setdefault("heat_weights", _default_heat_weights())
        if isinstance(st.session_state.get("heat_weights"), dict):
            st.session_state["heat_weights"][em] = {"w_move": float(wm), "w_dollar_vol": float(wd)}

        if st.button("Reset weights to defaults", key=f"heat_reset_{em}"):
            st.session_state["heat_weights"] = _default_heat_weights()
            st.rerun()

    # Heat score: blend of abs % move + dollar volume percentile
    # (no candles needed; stays fast)
    try:
        weights = _heat_weights_for_mode(em)
        dv_rank = sdf2["dollar_vol"].rank(pct=True).fillna(0.0)
        mv_rank = sdf2["abs_chg_%"].rank(pct=True).fillna(0.0)
        sdf2["heat_base"] = (weights["w_move"] * mv_rank + weights["w_dollar_vol"] * dv_rank) * 100.0
    except Exception:
        sdf2["heat_base"] = None

    # Optional realized-vol multiplier (fetch ONLY for Focus + Top5 strip + Hot List)
    sdf2["rv_ret"] = None
    sdf2["rv_atr"] = None
    sdf2["rv"] = None
    sdf2["rv_rank"] = 0.0

    if bool(st.session_state.get("heat_rv_on", False)) and ("heat_base" in sdf2.columns):
        try:
            focus_sym = str(st.session_state.get("scanner_focus") or "").upper().strip()
            hot_syms = [str(x).upper().strip() for x in st.session_state.get("scanner_hotlist", []) if str(x).strip()]
            top5_syms = [str(x).upper().strip() for x in strip_syms if str(x).strip()]

            eligible = []
            for x in [focus_sym] + top5_syms + hot_syms:
                x = str(x).upper().strip()
                if x and x not in eligible:
                    eligible.append(x)

            # compute rv for eligible only
            rv_map: dict[str, dict] = {}
            for sym in eligible[:60]:
                df1m = _sparkline_history(sym, window="1h")  # 60 bars
                rv_ret = _realized_vol_1m(df1m, method="returns", lookback=60)
                rv_atr = _realized_vol_1m(df1m, method="atr", lookback=60)
                if rv_ret is None and rv_atr is None:
                    continue

                if st.session_state.get("heat_rv_method") == "atr":
                    rv = rv_atr
                else:
                    rv = rv_ret

                if rv is None:
                    continue

                rv_map[sym] = {"rv_ret": rv_ret, "rv_atr": rv_atr, "rv": rv}

            if rv_map:
                # rank among the eligible rvs only
                rv_series = pd.Series({k: v["rv"] for k, v in rv_map.items()}).rank(pct=True)
                for sym, rec in rv_map.items():
                    m = sdf2["symbol"] == sym
                    sdf2.loc[m, "rv_ret"] = rec.get("rv_ret")
                    sdf2.loc[m, "rv_atr"] = rec.get("rv_atr")
                    sdf2.loc[m, "rv"] = rec.get("rv")
                    sdf2.loc[m, "rv_rank"] = float(rv_series.get(sym, 0.0))
        except Exception:
            pass

    try:
        k = float(st.session_state.get("heat_rv_k", 0.35))
        sdf2["heat"] = sdf2["heat_base"] * (1.0 + k * sdf2["rv_rank"].fillna(0.0))
    except Exception:
        sdf2["heat"] = sdf2.get("heat_base")

    if metric in sdf2.columns:
        sdf2 = sdf2.sort_values(metric, ascending=False)

    # Hot list controls
    st.markdown("### Hot List")
    hl = list(st.session_state.get("scanner_hotlist", []))
    hL, hR = st.columns([0.72, 0.28])
    with hL:
        st.caption("Persistent (saved locally). Use it as your pinboard.")

        if not hl:
            st.write("(empty)")
        else:
            # Remove buttons (Scanner-only)
            for sym in hl[:50]:
                rL, rR = st.columns([0.78, 0.22])
                rL.write(sym)
                if rR.button("Remove", key=f"hot_rm_{sym}"):
                    nxt = [x for x in hl if x != sym]
                    st.session_state["scanner_hotlist"] = nxt
                    _save_hotlist(nxt)
                    st.rerun()
    with hR:
        if st.button("Clear hot list"):
            st.session_state["scanner_hotlist"] = []
            _save_hotlist([])
            st.rerun()

    # --- Top 5 strip by dollar volume ---
    by_dv = sdf2.copy()
    if "dollar_vol" in by_dv.columns:
        by_dv = by_dv.sort_values("dollar_vol", ascending=False)

    top5 = by_dv.head(5).copy()
    strip_syms = top5["symbol"].tolist() if not top5.empty else []

    st.markdown("#### Top 5 (dollar volume)")
    if strip_syms:
        cols = st.columns(len(strip_syms))
        for i, sym in enumerate(strip_syms):
            r = top5.iloc[i]
            px = r.get("px")
            chgp = r.get("chg_%")
            dv = r.get("dollar_vol")

            try:
                pxs = f"${float(px):,.2f}" if px == px else "—"
            except Exception:
                pxs = "—"

            try:
                cps = f"{float(chgp):+.2f}%" if chgp == chgp else "—"
            except Exception:
                cps = "—"

            try:
                dvs = f"${float(dv)/1e9:,.2f}B" if dv == dv else "—"
            except Exception:
                dvs = "—"

            cols[i].markdown(
                f"<div class='casino-wrap'><div class='neon-blue'><b>{sym}</b></div>"
                f"<div class='neon-green'>{pxs}</div>"
                f"<div class='muted'>{cps} • {dvs}</div></div>",
                unsafe_allow_html=True,
            )
            b1, b2 = cols[i].columns([0.5, 0.5])
            if b1.button("Focus", key=f"scanner_focus_btn_{sym}"):
                st.session_state["scanner_focus"] = sym
                st.session_state["scanner_pin"] = True
                st.rerun()
            if b2.button("Hot+", key=f"scanner_hot_btn_{sym}"):
                cur = list(st.session_state.get("scanner_hotlist", []))
                if sym not in cur:
                    cur.insert(0, sym)
                st.session_state["scanner_hotlist"] = cur[:50]
                _save_hotlist(st.session_state["scanner_hotlist"])
                st.rerun()
    else:
        st.caption("No scan results yet.")

    topk = st.slider("Show top", 5, 50, 15, 5)
    show_cols = [c for c in ["symbol", "px", "chg_%", "volume", "dollar_vol", "heat", "assetType", "exchange"] if c in sdf2.columns]
    st.dataframe(sdf2[show_cols].head(int(topk)), use_container_width=True, hide_index=True, height=360)

    # --- Focus selection: auto-rotate unless pinned ---
    st.session_state.setdefault("scanner_pin", False)
    st.session_state.setdefault("scanner_focus", "")
    st.session_state.setdefault("scanner_rot_i", 0)
    st.session_state.setdefault("scanner_rot_tick", 0)

    cA, cB, cC = st.columns([0.34, 0.36, 0.30])
    pin = cA.toggle("Pin focus", value=bool(st.session_state.get("scanner_pin")), key="scanner_pin")
    rotate_every = cB.slider("Rotate every (refreshes)", 1, 10, 2, 1)
    auto_refresh = cC.toggle("Auto-refresh Scanner", value=False)
    if auto_refresh:
        # keep it modest; Scanner does many quote calls
        st_autorefresh(interval=15 * 1000, key="scanner_autorefresh")

    focus_pool_mode = st.selectbox(
        "Focus pool",
        ["Top 5 by $ volume", "Top 5 by current rank"],
        index=0,
        help="$ volume is usually the best 'what matters right now' proxy.",
    )

    # pick pool
    if focus_pool_mode == "Top 5 by current rank":
        pool = sdf2.head(5)["symbol"].tolist() if not sdf2.empty else []
    else:
        pool = strip_syms

    if pin and st.session_state.get("scanner_focus"):
        focus = str(st.session_state.get("scanner_focus"))
    else:
        # rotate through pool every N refreshes
        st.session_state["scanner_rot_tick"] = int(st.session_state.get("scanner_rot_tick", 0)) + 1
        tick = int(st.session_state.get("scanner_rot_tick", 0))

        if pool:
            if tick % int(rotate_every) == 0:
                i = int(st.session_state.get("scanner_rot_i", 0))
                st.session_state["scanner_rot_i"] = (i + 1) % len(pool)

            i = int(st.session_state.get("scanner_rot_i", 0))
            focus = pool[i % len(pool)]
        else:
            focus = str(sdf2.head(1).iloc[0]["symbol"]) if not sdf2.empty else selected

        st.session_state["scanner_focus"] = focus

    st.markdown("### Focus")
    fL, fR = st.columns([0.62, 0.38], gap="large")
    with fL:
        st.caption(f"In-focus: {focus} (rotates across Top 5 unless pinned)")
        a1, a2 = st.columns([0.5, 0.5])
        if a1.button("Set as main ticker", key="scanner_set_main"):
            st.session_state["selected_ticker"] = str(focus).upper().strip()
            st.toast(f"Main ticker set: {focus}")
            st.rerun()
        if a2.button("Hot+", key="scanner_hot_focus"):
            cur = list(st.session_state.get("scanner_hotlist", []))
            if focus not in cur:
                cur.insert(0, focus)
            st.session_state["scanner_hotlist"] = cur[:50]
            _save_hotlist(st.session_state["scanner_hotlist"])
            st.rerun()

        tf = st.selectbox("Focus timeframe", ["1m (4H)", "1m (3D)"], index=0, key="scanner_tf")
        dfp = _fetch_history(focus, tf)
        if dfp.empty:
            st.warning("No 1m candles for focus symbol.")
        else:
            st.plotly_chart(_plot_candles(dfp, title=f"{focus} — {tf}"), use_container_width=True)

    with fR:
        st.markdown("#### Exposure overlap")
        st.caption("If you're already holding it, we flag it here (best-effort).")

        held = []
        try:
            accts = _schwab_account_numbers()
            hashes = []
            for a in accts:
                h = str(a.get("hashValue") or a.get("accountHash") or a.get("hash") or "")
                if h:
                    hashes.append(h)
            hashes = hashes[:3]  # keep it cheap

            for h in hashes:
                js = _schwab_account_details(h)
                acct = js.get("securitiesAccount") if isinstance(js.get("securitiesAccount"), dict) else js
                pos = acct.get("positions") if isinstance(acct, dict) else []
                if isinstance(pos, list):
                    for p in pos:
                        instr = p.get("instrument") if isinstance(p, dict) and isinstance(p.get("instrument"), dict) else {}
                        sym = (instr.get("underlyingSymbol") or instr.get("symbol") or "").upper().strip()
                        if sym and sym not in held:
                            held.append(sym)
        except Exception:
            held = []

        pool_now = list(set((pool or []) + [focus] + list(st.session_state.get("scanner_hotlist", []))))
        overlap = sorted([s for s in pool_now if s in held])
        if overlap:
            st.warning(f"Already exposed to: {', '.join(overlap[:12])}")
        else:
            st.caption("No detected overlap (or positions unavailable).")

        st.markdown("#### Why is it moving? (placeholder)")
        st.caption("Paste a headline or catalyst here for now. Later we’ll wire a news source.")
        note = st.text_area("Catalyst", value="", height=160, key="scanner_catalyst")
        st.markdown("#### Related headlines (manual)")
        st.session_state.setdefault("scanner_headlines", "")
        st.text_area("Headlines", key="scanner_headlines", height=220, placeholder="Paste bullet headlines here…")

with tab_halts:
    st.subheader("Trading halts")

    # Auto-fetch feed (Nasdaq + NYSE + optional Cboe)
    st.session_state.setdefault("halts_auto", True)

    # Load URLs from decisions.json / rss_feeds.json if present (best-effort)
    halts_urls = {
        "nasdaq": "https://www.nasdaqtrader.com/trader.aspx?id=tradehalts",
        "nyse": "https://beta.nyse.com/trade/trading-halts",
        "cboe": "https://www.cboe.com/us/equities/market_statistics/halts/",
    }
    try:
        dec_p = _data_dir() / "decisions.json"
        if dec_p.exists():
            dec = json.loads(dec_p.read_text(encoding="utf-8"))
            pu = (((dec.get("provided_urls") or {}).get("halts")) or {}) if isinstance(dec, dict) else {}
            if isinstance(pu, dict):
                if pu.get("nasdaq"):
                    halts_urls["nasdaq"] = str(pu.get("nasdaq"))
                if pu.get("nyse"):
                    halts_urls["nyse"] = str(pu.get("nyse"))
                if pu.get("cboe_halts"):
                    halts_urls["cboe"] = str(pu.get("cboe_halts"))
    except Exception:
        pass

    # Priority: Cboe → Nasdaq → NYSE (your latest pick)
    feed = WebHaltsFeed(
        data_dir=_data_dir(),
        urls=halts_urls,
        include_cboe=True,
        source_priority=["cboe", "nasdaq", "nyse"],
    )

    topL, topR = st.columns([0.55, 0.45], vertical_alignment="top")
    with topL:
        st.caption(f"Feed status: {feed.status().detail}")
        st.session_state["halts_auto"] = st.toggle("Auto-fetch", value=bool(st.session_state.get("halts_auto", True)))
        refresh = st.button("Refresh now")

    with topR:
        st.caption("Sources: NasdaqTrader + NYSE + Cboe (best-effort HTML parsing).")

    # throttle: only fetch when tab visible and either refresh clicked or 60s elapsed
    st.session_state.setdefault("halts_last_fetch", 0.0)
    now = time.time()
    should_fetch = False
    if refresh:
        should_fetch = True
    elif bool(st.session_state.get("halts_auto", True)):
        if (now - float(st.session_state.get("halts_last_fetch", 0.0))) >= 60.0:
            should_fetch = True

    if should_fetch:
        hdf = feed.fetch_halts()
        st.session_state["halts_last_fetch"] = time.time()
        st.session_state["halts_last_df"] = hdf
    else:
        hdf = st.session_state.get("halts_last_df")
        if not isinstance(hdf, pd.DataFrame):
            hdf = pd.DataFrame()

    if hdf is None or hdf.empty:
        st.info("No halts data (yet). Click Refresh now.")
    else:
        # Default filter: active halts only
        st.markdown("### Halts")
        active_only = st.toggle("Active only", value=True, help="Default ON")
        show = hdf.copy()

        # Active-only heuristic: missing resume time or not resumed
        if active_only:
            if "resumed" in show.columns:
                show = show[show["resumed"] == False]  # noqa: E712
            else:
                show = show[show.get("resume_time_et", "").astype(str).str.strip() == ""]

        # Highlight resumes within 15 minutes (best-effort parse)
        def _mins_until_resume(s: str) -> float:
            try:
                if not s:
                    return 1e9
                # Expect formats like '02/07/2026 10:49:38' or '10:49:38'
                stxt = str(s).strip()
                if "/" in stxt:
                    dt = pd.to_datetime(stxt, errors="coerce")
                else:
                    # today
                    dt = pd.to_datetime(pd.Timestamp.now().strftime("%Y-%m-%d") + " " + stxt, errors="coerce")
                if pd.isna(dt):
                    return 1e9
                return float((dt - pd.Timestamp.now()).total_seconds() / 60.0)
            except Exception:
                return 1e9

        if "resume_time_et" in show.columns:
            show["mins_to_resume"] = show["resume_time_et"].astype(str).map(_mins_until_resume)

        st.dataframe(show, use_container_width=True, height=520)

    with st.expander("Legacy paste/parse (fallback)", expanded=False):
        st.info("If auto-fetch breaks, you can still paste a CSV.")
        raw = st.text_area(
            "Paste halts CSV",
            value="",
            height=180,
            placeholder="symbol,market,reason,halt_time,resume_time\nTICKER,NASDAQ,T1,09:45:01,10:15:00",
        )
        if raw.strip():
            try:
                import io

                hdf2 = pd.read_csv(io.StringIO(raw))
                st.dataframe(hdf2, use_container_width=True, height=420)
            except Exception as e:
                st.error(f"Could not parse CSV: {e}")

    st.markdown("### What’s needed next")
    st.write(
        "- Choose a halts source (Nasdaq Trader + NYSE).\n"
        "- Implement fetch + parse + reason-code mapping.\n"
        "- Add resume countdown + highlight newly halted/resumed symbols."
    )

with tab_signals:
    st.subheader("Signals")

    news_feed = StubNewsFeed()
    cal_feed = StubCalendarFeed()
    earn_feed = StubEarningsFeed()
    filings_feed = StubFilingsFeed()

    st.caption(
        "Scaffold mode: feeds are stubbed. Paste or type what you care about now; wire providers later."
    )
    st.write(
        {
            "news_feed": news_feed.status().detail,
            "calendar_feed": cal_feed.status().detail,
            "earnings_feed": earn_feed.status().detail,
            "filings_feed": filings_feed.status().detail,
        }
    )

    sL, sR = st.columns([0.55, 0.45], gap="large")

    with sL:
        st.markdown("### Fed / macro calendar (manual)")
        st.caption("Paste upcoming Fed events / speakers / CPI / NFP. We'll make it prettier later.")
        st.session_state.setdefault("macro_events", "")
        st.text_area("Events", key="macro_events", height=260, placeholder="YYYY-MM-DD HH:MM — Event — Notes")

        st.markdown("### Themes (scaffold)")
        theme = st.selectbox(
            "Theme universe (starter)",
            [
                "None",
                "Halal / Islamic ETFs (starter)",
                "Oil / Gas (starter)",
                "Metals (starter)",
            ],
            index=0,
        )
        if theme != "None":
            if theme.startswith("Halal"):
                syms = _parse_symbols("HLAL,SPUS,SPSK")
            elif theme.startswith("Oil"):
                syms = _parse_symbols("XLE,CVX,XOM,OXY,USO,UNG")
            else:
                syms = _parse_symbols("GLD,SLV,COPX,PPLT,PALL")

            st.caption(f"Theme symbols: {', '.join(syms)}")
            trows = []
            for s in syms:
                q = _schwab_quote(s)
                px = q.get("mark") or q.get("last")
                trows.append({"symbol": s, "px": px, "chg_%": q.get("netPct")})
            tdf = pd.DataFrame(trows)
            for c in ["px", "chg_%"]:
                if c in tdf.columns:
                    tdf[c] = pd.to_numeric(tdf[c], errors="coerce")
            st.dataframe(tdf, use_container_width=True, hide_index=True, height=200)

    with sR:
        st.markdown("### News board (manual)")
        st.caption("Paste headlines. We’ll auto-detect tickers (basic).")
        raw_news = st.text_area("Headlines", value="", height=260, placeholder="- TSLA jumps on delivery beat…")

        import re

        tickers = []
        if raw_news.strip():
            # detect $TSLA or TSLA (2-5 caps) — intentionally simple
            tickers = re.findall(r"\$([A-Z]{1,6})\b|\b([A-Z]{2,5})\b", raw_news.upper())
            flat = []
            for a, b in tickers:
                t = a or b
                if t and t not in flat:
                    flat.append(t)
            tickers = flat[:25]

        if tickers:
            st.markdown("#### Detected tickers")
            st.write(tickers)

            qrows = []
            for t in tickers[:10]:
                q = _schwab_quote(t)
                qrows.append({"symbol": t, "px": q.get("mark") or q.get("last"), "chg_%": q.get("netPct")})
            qdf = pd.DataFrame(qrows)
            for c in ["px", "chg_%"]:
                if c in qdf.columns:
                    qdf[c] = pd.to_numeric(qdf[c], errors="coerce")
            st.dataframe(qdf, use_container_width=True, hide_index=True, height=240)

        st.markdown("### Oil / Metals intel (placeholder)")
        st.caption("Later: wire tanker flows, inventory reports, mines, etc. For now: notes.")
        st.session_state.setdefault("intel_notes", "")
        st.text_area("Notes", key="intel_notes", height=210)

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

        # Schwab-only: chart + quote always come from Schwab Market Data.
        # (Daily bootstrap DB is still available for offline exploration, but not used for the live chart.)

        q = _schwab_quote(selected)

        with st.spinner("Fetching 1m candles from Schwab…"):
            dfp = _fetch_history(selected, tos_tf)

        # Spade checks: price history
        try:
            checks = check_price_history(dfp)
            st.session_state["spade"]["history"] = summarize(checks)
        except Exception:
            st.session_state["spade"]["history"] = None

        if dfp.empty:
            st.warning("No 1m candle history returned (Schwab may be down, tokens missing, or symbol unsupported).")
            st.caption("This does not block the rest of the app. Use Admin → Schwab OAuth to connect.")

        # Current price: prefer quote mark/last; fallback to last candle close.
        cur_px = None
        try:
            if q.get("mark") not in (None, ""):
                cur_px = float(q.get("mark"))
            elif q.get("last") not in (None, ""):
                cur_px = float(q.get("last"))
        except Exception:
            cur_px = None

        if cur_px is None:
            try:
                cur_px = float(dfp["close"].dropna().iloc[-1])
            except Exception:
                cur_px = None

        title_price = "Price" if cur_px is None else f"Price — ${cur_px:,.2f}"
        st.subheader(title_price)

        # Data age indicators
        now = pd.Timestamp.now()

        qt_ms = q.get("ts_ms")
        qt = None
        if isinstance(qt_ms, (int, float)) and qt_ms:
            try:
                qt = pd.to_datetime(int(qt_ms), unit="ms", utc=True).tz_convert(None)
            except Exception:
                qt = None

        last_candle = None
        if not dfp.empty and "date" in dfp.columns:
            try:
                last_candle = pd.to_datetime(dfp["date"].dropna().max(), errors="coerce")
            except Exception:
                last_candle = None

        age_bits = []
        if qt is not None:
            age_bits.append(f"Quote: {qt.strftime('%Y-%m-%d %H:%M:%S')} (age {(now-qt).total_seconds():.0f}s)")
        else:
            age_bits.append("Quote: (no timestamp)")

        if last_candle is not None and not pd.isna(last_candle):
            age_bits.append(f"Last 1m candle: {pd.Timestamp(last_candle).strftime('%Y-%m-%d %H:%M:%S')} (age {(now-pd.Timestamp(last_candle)).total_seconds():.0f}s)")

        st.caption(" • ".join(age_bits))

        if not dfp.empty:
            st.plotly_chart(_plot_candles(dfp, title=f"{selected} — 1m"), use_container_width=True)

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

        st.caption("Source: Schwab Market Data (quotes + 1m candles).")

        with st.expander("Raw quote (Schwab)", expanded=False):
            st.json(dict(q.get("raw") or {}))

        with st.expander("Raw prices", expanded=False):
            st.dataframe(dfp, use_container_width=True, height=320)

with tab_casino:
    st.subheader("Casino Lab")
    st.caption("Quant playground. Toy models for exploration only — not trading advice. Data source is Schwab.")

    casino_on = st.toggle("Casino visuals", value=True)
    q_tape = _schwab_quote(selected)

    if casino_on:
        last = q_tape.get("last")
        mark = q_tape.get("mark")
        net = q_tape.get("netChange")
        netp = q_tape.get("netPct")

        def _fmt(x, nd=2):
            try:
                if x is None or x == "":
                    return "—"
                return f"{float(x):,.{nd}f}"
            except Exception:
                return str(x)

        px_txt = _fmt(mark) if mark not in (None, "") else _fmt(last)
        net_txt = "—"
        try:
            if net not in (None, ""):
                net_txt = f"{float(net):+.2f}"
        except Exception:
            net_txt = str(net)

        netp_txt = "—"
        try:
            if netp not in (None, ""):
                netp_txt = f"{float(netp):+.2f}%"
        except Exception:
            netp_txt = str(netp)

        st.markdown(
            f"""
<div class='ticker-tape'>
  <span class='marquee'>
    <span class='neon-gold'>LIVE TAPE</span>  •  
    <span class='neon-blue'>{selected}</span>  
    <span class='neon-green'>${px_txt}</span>  
    <span class='muted'>({net_txt} / {netp_txt})</span>
    &nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
    <span class='muted'>Bayes + Backtests + Toys</span>
  </span>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown("### Modules")
    mod = st.selectbox(
        "Pick a module",
        [
            "Bayes: Next 1m bar up probability",
            "Backtest: Mean reversion (toy)",
            "Backtest: Breakout (toy)",
        ],
        index=0,
    )

    st.markdown("#### Data")
    st.caption("Uses the same 1m candles as the Overview chart. Adjust timeframe there, or we can add separate controls.")
    prices_1m = _fetch_history(selected, st.session_state.get("tos_tf", "1m (3D)"))

    if prices_1m.empty:
        st.warning("No 1m candles available for this symbol right now.")
    else:
        st.caption(f"Bars loaded: {len(prices_1m):,} (last={pd.to_datetime(prices_1m['date'].max())})")

    if mod.startswith("Bayes"):
        st.markdown("### Bayesian quick read")
        st.caption("Beta-Binomial on up/down bars. This is intentionally simple and fast.")
        lookback = st.slider("Lookback bars", 30, 780, 240, 30)
        res = _beta_binomial_up_prob(prices_1m, lookback_bars=int(lookback))
        if not res:
            st.warning("Not enough data to compute.")
        else:
            cA, cB, cC = st.columns(3)
            cA.metric("P(up) mean", f"{res['p_up_mean']*100:.1f}%")
            cB.metric("CI90 low", f"{res['p_up_ci90'][0]*100:.1f}%")
            cC.metric("CI90 high", f"{res['p_up_ci90'][1]*100:.1f}%")
            with st.expander("Details", expanded=False):
                st.json(res)

            st.markdown("#### What I need from you (to make this real)")
            st.write(
                "- Which horizon matters: next 1m, next 5m, next hour?\n"
                "- Do you care about direction only, or magnitude too?\n"
                "- Do you want priors based on regime (trend/vol), or keep it dumb-simple?"
            )

    if mod.startswith("Backtest"):
        st.markdown("### Backtest sandbox")
        st.caption("Single-position long/flat toy backtests on 1m closes.")
        fee = st.number_input("Fees (bps per entry/exit)", min_value=0.0, max_value=50.0, value=0.0, step=0.5)
        slip = st.number_input("Slippage (bps per entry/exit)", min_value=0.0, max_value=50.0, value=0.0, step=0.5)

        strat = "mean_reversion" if "Mean reversion" in mod else "breakout"
        bt = _backtest_1m(prices_1m, strat, fee_bps=float(fee), slippage_bps=float(slip))
        if bt.empty:
            st.warning("Backtest could not run (not enough bars).")
        else:
            # KPIs
            total = float(bt["equity"].iloc[-1] - 1.0)
            dd = float(bt["drawdown"].min())
            trades = int((bt["pos_chg"] > 0).sum() / 2)  # entry+exit pairs

            k1, k2, k3 = st.columns(3)
            k1.metric("Total return", f"{total*100:+.2f}%")
            k2.metric("Max drawdown", f"{dd*100:.2f}%")
            k3.metric("Trades (approx)", f"{trades}")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=bt["date"], y=bt["equity"], name="equity", line=dict(color="#33ffcc")))
            fig.update_layout(height=280, margin=dict(l=10, r=10, t=25, b=10), template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Equity + signals (raw)", expanded=False):
                st.dataframe(bt[["date", "close", "pos", "strat_ret", "equity", "drawdown"]].tail(400), use_container_width=True, height=320)

            st.markdown("#### What I need from you (to build your real backtester)")
            st.write(
                "- Are we backtesting equities/ETFs only, or options too?\n"
                "- What’s your typical entry trigger (breakout, VWAP reclaim, ORB, gamma levels, etc.)?\n"
                "- Risk model: fixed stop, ATR stop, time stop, max loss/day?\n"
                "- Execution: market, limit at mid, bid/ask model?"
            )

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

with tab_exposure:
    st.subheader("Exposure")
    st.caption("Schwab-only view of accounts, balances, and positions. Designed to be fast + high-signal.")

    api = _schwab_api()
    if api is None:
        st.warning("Schwab OAuth is not configured. Go to Admin → Schwab OAuth.")
    else:
        accts = _schwab_account_numbers()
        if not accts:
            st.warning("No accounts returned (Schwab may be down, tokens missing, or app not entitled for trader endpoints).")
        else:
            # Build label -> hash mapping (avoid showing hashes/ids in normal UI)
            opts = []
            acct_i = 0
            for a in accts:
                num = str(a.get("accountNumber") or a.get("account") or "")
                h = str(a.get("hashValue") or a.get("accountHash") or a.get("hash") or "")
                if h:
                    acct_i += 1
                    label = f"…{num[-4:]}" if num else f"Acct {acct_i}"
                    opts.append((label, h, num))

            if not opts:
                st.warning("Accounts list did not include hashes. Can't fetch positions.")
            else:
                # Decision: default scope = All accounts
                mode = st.radio("Scope", ["All accounts", "Single account"], horizontal=True, index=0)

                if mode == "Single account":
                    pick = st.selectbox("Account", opts, format_func=lambda x: x[0])
                    hashes = [pick[1]]
                else:
                    hashes = [x[1] for x in opts]

                group_underlying = st.toggle(
                    "Group under underlying",
                    value=True,
                    help="On by default. Groups options/futures under their underlying when Schwab provides underlyingSymbol.",
                    key="expo_group_underlying",
                )

                # Pull account details (positions + best-effort balances)
                per_account = []
                raw_payloads = {}
                for h in hashes:
                    js = _schwab_account_details(h, fields="positions")
                    raw_payloads[h] = js

                    # Schwab payload can be nested; handle a few common shapes
                    acct = None
                    if isinstance(js.get("securitiesAccount"), dict):
                        acct = js.get("securitiesAccount")
                    elif isinstance(js.get("account"), dict):
                        acct = js.get("account")
                    else:
                        acct = js

                    if not isinstance(acct, dict):
                        continue

                    # balances (best-effort; varies by entitlement/account type)
                    bals = acct.get("currentBalances") if isinstance(acct.get("currentBalances"), dict) else {}
                    net_liq = bals.get("liquidationValue") or bals.get("equity") or bals.get("accountValue")
                    cash = bals.get("cashBalance") or bals.get("cashAvailableForTrading")
                    bp = bals.get("buyingPower") or bals.get("availableFunds")

                    pos = acct.get("positions") or []
                    if not isinstance(pos, list):
                        pos = []

                    rows = []
                    for p in pos:
                        if not isinstance(p, dict):
                            continue

                        instr = p.get("instrument") if isinstance(p.get("instrument"), dict) else {}
                        sym = instr.get("symbol") or instr.get("cusip") or "(unknown)"
                        asset = instr.get("assetType") or instr.get("type")
                        underlying = (
                            instr.get("underlyingSymbol")
                            or instr.get("underlying")
                            or instr.get("underlyingSymbolId")
                            or sym
                        )

                        # Quantity: show shorts as negative when possible
                        long_qty = p.get("longQuantity")
                        short_qty = p.get("shortQuantity")
                        qty = p.get("quantity")
                        try:
                            if short_qty not in (None, 0, 0.0):
                                qty_out = -abs(float(short_qty))
                            elif long_qty not in (None, 0, 0.0):
                                qty_out = float(long_qty)
                            elif qty not in (None, 0, 0.0):
                                qty_out = float(qty)
                            else:
                                qty_out = None
                        except Exception:
                            qty_out = qty

                        mv = p.get("marketValue")  # mv preferred
                        avg = p.get("averagePrice") or p.get("avgPrice")
                        pl = p.get("currentDayProfitLoss")

                        rows.append(
                            {
                                "account_hash": h,
                                "symbol": str(sym),
                                "underlying": str(underlying),
                                "asset": asset,
                                "qty": qty_out,
                                "avg_price": avg,
                                "market_value": mv,
                                "day_pl": pl,
                            }
                        )

                    per_account.append(
                        {
                            "hash": h,
                            "label": next((o[0] for o in opts if o[1] == h), h[:6]),
                            "net_liq": net_liq,
                            "cash": cash,
                            "buying_power": bp,
                            "positions": pd.DataFrame(rows),
                        }
                    )

                # Combined positions df
                all_rows = []
                for a in per_account:
                    d = a.get("positions")
                    if isinstance(d, pd.DataFrame) and not d.empty:
                        all_rows.append(d)
                df = pd.concat(all_rows, axis=0, ignore_index=True) if all_rows else pd.DataFrame()

                if df.empty:
                    st.warning("No positions found (or schema mismatch).")
                    with st.expander("Raw account payloads (debug)", expanded=False):
                        st.json(raw_payloads)
                else:
                    # Normalize numeric columns
                    for c in ["qty", "avg_price", "market_value", "day_pl"]:
                        if c in df.columns:
                            df[c] = pd.to_numeric(df[c], errors="coerce")

                    key_col = "underlying" if group_underlying else "symbol"

                    expo = df.groupby([key_col], as_index=False).agg(
                        market_value=("market_value", "sum"),
                        day_pl=("day_pl", "sum"),
                    )
                    expo = expo.rename(columns={key_col: "symbol"})

                    total_mv = float(expo["market_value"].fillna(0.0).sum())
                    expo["pct"] = (expo["market_value"].fillna(0.0) / total_mv) if total_mv else 0.0

                    # Decision: default sort by day P/L
                    expo = expo.sort_values("day_pl", ascending=False)

                    topn = st.slider("Top symbols", 5, 60, 20, 5)
                    expo_top = expo.head(int(topn)).copy()
                    other_mv = float(expo["market_value"].fillna(0.0).sum() - expo_top["market_value"].fillna(0.0).sum())
                    if other_mv > 0:
                        expo_top = pd.concat(
                            [
                                expo_top,
                                pd.DataFrame([{ "symbol": "Other", "market_value": other_mv, "day_pl": None, "pct": other_mv / total_mv if total_mv else 0.0 }]),
                            ],
                            ignore_index=True,
                        )

                    # Combined summary
                    k1, k2, k3 = st.columns(3)
                    k1.metric("Total market value (sum)", f"${total_mv:,.0f}")
                    k2.metric("Grouped symbols" if group_underlying else "Symbols", f"{len(expo):,}")
                    k3.metric("Accounts in scope", f"{len(hashes)}")

                    fig = go.Figure(
                        data=[
                            go.Pie(
                                labels=expo_top["symbol"],
                                values=expo_top["market_value"].fillna(0.0),
                                hole=0.45,
                                textinfo="label+percent",
                            )
                        ]
                    )
                    fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("### Exposure table")
                    expo_tbl = expo[["symbol", "market_value", "pct", "day_pl"]].copy()
                    expo_tbl["pct_%"] = (expo_tbl["pct"] * 100.0).round(2)
                    expo_tbl = expo_tbl.drop(columns=["pct"])

                    # Set-as-main via data editor selection
                    expo_tbl.insert(0, "set_main", False)
                    edited = st.data_editor(
                        expo_tbl,
                        use_container_width=True,
                        height=420,
                        hide_index=True,
                        column_config={
                            "set_main": st.column_config.CheckboxColumn("Set main"),
                        },
                    )
                    try:
                        picks = edited[edited["set_main"] == True]["symbol"].astype(str).tolist()  # noqa: E712
                    except Exception:
                        picks = []
                    if picks:
                        st.session_state["selected_ticker"] = str(picks[0]).upper().strip()
                        st.toast(f"Main ticker set: {picks[0]}")

                    # Account cards (collapsible) with balances in header (best-effort)
                    st.markdown("### Accounts")
                    st.caption("Balances are best-effort; if missing, we stay positions-only.")
                    for a in per_account:
                        label = a.get("label")
                        nl = a.get("net_liq")
                        cash = a.get("cash")
                        bp = a.get("buying_power")

                        def _fmt(x):
                            try:
                                return f"${float(x):,.0f}" if x not in (None, "") else "—"
                            except Exception:
                                return "—"

                        hdr = f"{label} • NetLiq {_fmt(nl)} • Cash {_fmt(cash)} • BP {_fmt(bp)}"
                        with st.expander(hdr, expanded=False):
                            d = a.get("positions")
                            if not isinstance(d, pd.DataFrame) or d.empty:
                                st.write("(no positions)")
                            else:
                                # Raw positions view includes qty + avg_price (trade price)
                                st.dataframe(
                                    d[["symbol", "underlying", "asset", "qty", "avg_price", "market_value", "day_pl"]],
                                    use_container_width=True,
                                    height=260,
                                    hide_index=True,
                                )

                    # Save for Exports tab
                    st.session_state["last_expo"] = expo.copy()

                    st.markdown("### Shareable snapshot (redacted)")
                    st.caption("Markdown format, no account ids/numbers.")
                    snap = expo.copy()
                    snap["market_value"] = snap["market_value"].fillna(0.0).round(2)
                    snap["pct"] = (snap["pct"] * 100.0).round(2)
                    snap = snap[["symbol", "market_value", "pct", "day_pl"]]

                    md_lines = ["| symbol | mv | pct | dayPL |", "|---:|---:|---:|---:|"]
                    for r in snap.itertuples(index=False):
                        daypl = "" if pd.isna(r.day_pl) else f"{float(r.day_pl):+.2f}"
                        md_lines.append(f"| {r.symbol} | ${float(r.market_value):,.0f} | {float(r.pct):.2f}% | {daypl} |")
                    md = "\n".join(md_lines)

                    st.text_area("Snapshot (markdown)", value=md, height=260)
                    st.download_button("Download snapshot.md", data=md.encode("utf-8"), file_name="exposure_snapshot.md")

                    with st.expander("Raw account payloads (debug)", expanded=False):
                        st.json(raw_payloads)

                    st.caption("PDF export: we can add a one-click PDF once you tell me what format you like (1-page summary vs full tables).")

with tab_exports:
    st.subheader("Exports")
    st.caption("Local-first exports. PDFs are generated locally (ReportLab) and saved to data/exports/.")

    export_dir = _data_dir() / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    expL, expR = st.columns([0.5, 0.5], vertical_alignment="top")
    with expL:
        st.markdown("### Exposure PDF")
        st.caption("1-page summary • Top 20 • pie chart • redacted")

        expo = st.session_state.get("last_expo")
        if not isinstance(expo, pd.DataFrame) or expo.empty:
            st.info("Open the Exposure tab once to load data, then come back here.")
        else:
            if st.button("Generate exposure PDF", key="gen_expo_pdf"):
                try:
                    pdf = _pdf_exposure_summary(expo, top_n=20, title="Exposure summary")
                    fname = f"exposure_{_local_timestamp_compact()}.pdf"
                    out_path = export_dir / fname
                    out_path.write_bytes(pdf)
                    st.success(f"Wrote: {out_path}")
                    st.download_button(
                        "Download exposure PDF",
                        data=pdf,
                        file_name=fname,
                        mime="application/pdf",
                        key="dl_expo_pdf",
                    )
                except Exception as e:
                    st.error(str(e))

    with expR:
        st.markdown("### Scanner PDF")
        st.caption("1-page snapshot • rankings + hot list context")

        sdf2 = st.session_state.get("last_scanner")
        if not isinstance(sdf2, pd.DataFrame) or sdf2.empty:
            st.info("Open the Scanner tab once to load data, then come back here.")
        else:
            if st.button("Generate scanner PDF", key="gen_scan_pdf"):
                try:
                    pdf = _pdf_scanner_snapshot(sdf2, title="Scanner snapshot")
                    fname = f"scanner_{_local_timestamp_compact()}.pdf"
                    out_path = export_dir / fname
                    out_path.write_bytes(pdf)
                    st.success(f"Wrote: {out_path}")
                    st.download_button(
                        "Download scanner PDF",
                        data=pdf,
                        file_name=fname,
                        mime="application/pdf",
                        key="dl_scan_pdf",
                    )
                except Exception as e:
                    st.error(str(e))

    st.markdown("### Recent exports")
    try:
        paths = sorted(export_dir.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)[:12]
        if not paths:
            st.write("(none yet)")
        else:
            for p in paths:
                st.write(p.name)
    except Exception:
        st.write("(could not list exports)")


with tab_decisions:
    st.subheader("Decisions")
    st.caption("Use this page as a lightweight multiple-choice form. Decisions are saved locally to data/decisions.json")

    decisions = _load_decisions()

    # --- Current decision set: Integrations / Feeds ---
    st.markdown("### Integrations / Feeds (current block)")

    q = [
        ("alerts_priority", "1) Schwab-native Alerts priority", {"A": "Do Alerts next (big unlock)", "B": "Do something else first"}),
        ("alerts_scope", "2) Alerts scope", {"A": "Price alerts only", "B": "Price + % change + time-based (if available)"}),
        ("outbound_domains", "3) Allowed outbound domains policy", {"A": "Schwab-only (no external HTTP)", "B": "Allow specific public domains (allowlist)"}),
        ("halts_source", "4) Halts feed source", {"A": "Nasdaq Trader", "B": "NYSE", "C": "Both (merge/dedupe)"}),
        ("news_mode", "5) News", {"A": "Keep manual paste for now", "B": "Wire a basic RSS/news feed"}),
        ("macro_calendar", "6) Macro calendar", {"A": "Manual paste stays", "B": "Add an auto calendar feed"}),
        ("earnings_calendar", "7) Earnings calendar", {"A": "Manual CSV/paste only", "B": "Add an auto earnings feed"}),
        ("actives_movers", "8) Most active / movers universe", {"A": "Keep curated universe only", "B": "Add actives feed"}),
        ("balances_now", "9) Exposure balances reliability", {"A": "Positions-only primary; balances best-effort", "B": "Normalize balances across account types now"}),
        ("oauth_arch", "10) OAuth/token architecture", {"A": "One app/token set if possible", "B": "Separate read-only vs trader scopes/apps"}),
    ]

    with st.form("decisions_form"):
        picks = {}
        for key, label, opts in q:
            # default to last saved, else first key
            default = decisions.get(key)
            if default not in opts:
                default = list(opts.keys())[0]
            picks[key] = st.radio(label, list(opts.keys()), index=list(opts.keys()).index(default), format_func=lambda k, o=opts: f"{k}) {o[k]}")
            st.divider()

        submitted = st.form_submit_button("Save decisions")

    if submitted:
        for k, v in picks.items():
            decisions[k] = v
        decisions["_updated_at"] = _local_timestamp_compact()
        _save_decisions(decisions)
        st.success(f"Saved to: {_decisions_path()}")

    st.markdown("### Current saved answers")
    st.json(decisions)

    st.markdown("### Next step helper")
    if decisions.get("outbound_domains") == "B":
        st.info("Next: we need an allowlist of outbound domains (Nasdaq/NYSE, calendar provider, news RSS domains, etc.).")
    else:
        st.info("Outbound stays Schwab-only. Next: focus on Schwab Alerts endpoints/scopes.")


with tab_admin:
    st.subheader("Admin / Data pipelines")

    st.markdown("### App settings")
    st.caption("Local-only settings (no secrets). Auto-saved to data/app_settings.json")

    sL, sR = st.columns([0.5, 0.5], vertical_alignment="top")
    with sL:
        if st.button("Reset settings to defaults", key="settings_reset"):
            defaults = _settings_defaults()
            for k, v in defaults.items():
                st.session_state[k] = v
            _save_settings(defaults)
            st.toast("Settings reset to defaults")
            st.rerun()

        st.download_button(
            "Download settings (app_settings.json)",
            data=json.dumps(_settings_defaults() | _settings_snapshot(), indent=2, sort_keys=True).encode("utf-8"),
            file_name="app_settings.json",
            mime="application/json",
            key="settings_dl_json",
        )

        # Also offer a zip that includes scanner presets.
        buf = BytesIO()
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("app_settings.json", json.dumps(_settings_defaults() | _settings_snapshot(), indent=2, sort_keys=True))
            # include scanner presets if they exist
            try:
                sdir = _scanners_dir()
                if sdir.exists():
                    for p in sdir.glob("*.json"):
                        z.writestr(f"scanners/{p.name}", p.read_text(encoding="utf-8"))
            except Exception:
                pass

        st.download_button(
            "Download settings bundle (.zip)",
            data=buf.getvalue(),
            file_name="market_hub_settings.zip",
            mime="application/zip",
            key="settings_dl_zip",
        )

    with sR:
        up = st.file_uploader("Import settings (JSON or ZIP)", type=["json", "zip"], key="settings_upload")
        if up is not None:
            if st.button("Import now", key="settings_import"):
                try:
                    if up.name.lower().endswith(".zip"):
                        zf = zipfile.ZipFile(BytesIO(up.getvalue()))
                        if "app_settings.json" in zf.namelist():
                            settings_obj = json.loads(zf.read("app_settings.json").decode("utf-8"))
                        else:
                            settings_obj = {}
                        # restore scanner presets
                        sdir = _scanners_dir()
                        sdir.mkdir(parents=True, exist_ok=True)
                        for name in zf.namelist():
                            if name.startswith("scanners/") and name.endswith(".json"):
                                outp = sdir / Path(name).name
                                outp.write_text(zf.read(name).decode("utf-8"), encoding="utf-8")
                    else:
                        settings_obj = json.loads(up.getvalue().decode("utf-8"))

                    if not isinstance(settings_obj, dict):
                        raise ValueError("settings file must be a JSON object")

                    merged = _settings_defaults() | settings_obj
                    _save_settings(merged)
                    for k, v in merged.items():
                        st.session_state[k] = v
                    st.success(f"Imported settings into: {_settings_path()}")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

    st.write({"settings_path": str(_settings_path()), "scanners_dir": str(_scanners_dir())})

    # Debug bundle
    with st.expander("Support / Debug bundle", expanded=False):
        st.caption("Creates a sanitized zip with recent logs (no secrets/tokens).")
        if st.button("Create debug bundle"):
            try:
                bundle = create_debug_bundle(_data_dir())
                st.success(f"Wrote: {bundle}")
            except Exception as e:
                st.error(str(e))

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

    st.markdown("#### 1) Universe")
    st.caption("Universe fetch has been removed. This UI is Schwab-only; use manual symbols.")

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
