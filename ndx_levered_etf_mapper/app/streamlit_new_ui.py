"""Experimental UI (new interface) for Market Hub.

Goal:
- Keep existing app/streamlit_app.py intact.
- Reuse the already-configured Schwab OAuth (data/schwab_secrets.local.json + data/schwab_tokens.json).

Run:
  streamlit run app/streamlit_new_ui.py

Notes:
- This file is intentionally small and modular. We'll grow it iteratively.
- No live order placement.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from etf_mapper.config import load_schwab_secrets
from etf_mapper.schwab.client import SchwabAPI, SchwabConfig


def _data_dir() -> Path:
    # Keep consistent with the main app: default to repo-local data/
    # (streamlit runs with CWD at repo root typically)
    return Path("data")


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


def _quotes(api: SchwabAPI, symbols: list[str]) -> dict:
    # Simple wrapper; keep it defensive.
    try:
        return api.quotes([s.strip().upper() for s in symbols if str(s).strip()])
    except Exception:
        return {}


def _inject_matrix_css() -> None:
    st.markdown(
        """
<style>
  /* Matrix theme */
  :root {
    --mx-bg: #050805;
    --mx-panel: #071007;
    --mx-border: rgba(56,255,154,0.18);
    --mx-text: #38ff9a;
    --mx-muted: rgba(56,255,154,0.55);
    --mx-warn: #ffd166;
    --mx-down: #ff3b6b;
  }

  .stApp {
    background: var(--mx-bg) !important;
    color: var(--mx-text) !important;
  }

  /* General text */
  html, body, [class*="css"], p, li, label, span { color: var(--mx-text) !important; }
  small, .stCaption, .stMarkdown small { color: var(--mx-muted) !important; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #040604 !important;
    border-right: 1px solid var(--mx-border);
  }

  /* Cards/containers */
  div[data-testid="stMetric"], div[data-testid="stExpander"], div[data-testid="stForm"], div[data-testid="stContainer"] {
    background: var(--mx-panel) !important;
    border: 1px solid var(--mx-border) !important;
    border-radius: 12px !important;
  }

  /* Inputs */
  input, textarea {
    background: #040a04 !important;
    border: 1px solid var(--mx-border) !important;
    color: var(--mx-text) !important;
  }

  /* Selectbox / multiselect */
  div[data-baseweb="select"] > div {
    background: #040a04 !important;
    border: 1px solid var(--mx-border) !important;
    color: var(--mx-text) !important;
  }

  /* Buttons */
  button[kind], .stButton > button {
    background: #061206 !important;
    border: 1px solid rgba(56,255,154,0.35) !important;
    color: var(--mx-text) !important;
    border-radius: 10px !important;
  }
  .stButton > button:hover {
    border-color: rgba(56,255,154,0.70) !important;
    box-shadow: 0 0 18px rgba(56,255,154,0.12);
  }

  /* Tabs */
  .stTabs [data-baseweb="tab"] { color: var(--mx-muted) !important; }
  .stTabs [aria-selected="true"] { color: var(--mx-text) !important; }

  /* Dataframe */
  div[data-testid="stDataFrame"] {
    border: 1px solid var(--mx-border);
    border-radius: 12px;
    overflow: hidden;
  }
</style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Market Hub (New UI)", layout="wide")
    _inject_matrix_css()

    st.title("Market Hub — New UI")
    st.caption("Experimental interface. Reuses Schwab OAuth tokens from data/. (Matrix mode)")

    api = _schwab_api()
    data_dir = _data_dir().resolve()

    with st.sidebar:
        st.subheader("Connection")
        st.write(
            {
                "data_dir": str(data_dir),
                "secrets_present": (data_dir / "schwab_secrets.local.json").exists(),
                "tokens_present": (data_dir / "schwab_tokens.json").exists(),
            }
        )
        watch = st.text_input("Watchlist", value="SPY,QQQ,TSLA,AAPL,NVDA")
        symbols = [s.strip().upper() for s in watch.split(",") if s.strip()]

    if api is None:
        st.error("Schwab OAuth not configured. Ensure data/schwab_secrets.local.json exists.")
        return

    if not (data_dir / "schwab_tokens.json").exists():
        st.error("Schwab tokens missing. Complete OAuth in the main app or create tokens.")
        return

    st.markdown("### Live tape (batch quotes)")
    js = _quotes(api, symbols)

    rows = []
    for s in symbols:
        rec = js.get(s) if isinstance(js, dict) else None
        q = (rec or {}).get("quote") if isinstance((rec or {}).get("quote"), dict) else {}
        rows.append(
            {
                "symbol": s,
                "last": q.get("lastPrice"),
                "mark": q.get("mark"),
                "net%": q.get("netPercentChange"),
                "vol": q.get("totalVolume"),
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, height=280)

    st.info("Next: we’ll build the new scanner-first UI here (Meters v2 / multi-panels / chart grid).")


if __name__ == "__main__":
    main()
