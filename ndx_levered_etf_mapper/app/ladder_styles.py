from __future__ import annotations

import pandas as pd


def style_ladder_with_changes(df: pd.DataFrame, prev: pd.DataFrame | None) -> pd.io.formats.style.Styler:
    """Return a Styler that highlights call vs put areas and shows price change intensity.

    Note: Streamlit's st.data_editor doesn't support full cell styling.
    We use this for a read-only styled view, while selection/qty editing stays in data_editor.
    """

    d = df.copy()

    # Compute deltas vs previous snapshot (by strike)
    if prev is not None and not prev.empty and "strike" in d.columns and "strike" in prev.columns:
        p = prev.set_index("strike")
        d = d.set_index("strike")
        for col in ["call_bid", "call_ask", "put_bid", "put_ask"]:
            if col in d.columns and col in p.columns:
                d[f"{col}_chg"] = d[col] - p[col]
        d = d.reset_index()

    def _bg_call_put(s: pd.Series):
        # subtle background for calls vs puts columns
        cols = list(s.index)
        out = ["" for _ in cols]
        for i, c in enumerate(cols):
            if str(c).startswith("call_"):
                out[i] = "background-color: rgba(34,197,94,0.06);"  # green-ish
            elif str(c).startswith("put_"):
                out[i] = "background-color: rgba(239,68,68,0.06);"  # red-ish
            elif str(c) == "strike":
                out[i] = "background-color: rgba(148,163,184,0.06); font-weight: 600;"
        return out

    def _chg_color(v):
        try:
            x = float(v)
        except Exception:
            return ""
        if x == 0:
            return ""
        # intensity scaling
        a = min(0.35, 0.10 + 0.05 * min(5.0, abs(x) / 0.05))
        if x > 0:
            return f"background-color: rgba(34,197,94,{a});"  # light green
        return f"background-color: rgba(239,68,68,{a});"  # light red

    sty = d.style.apply(_bg_call_put, axis=1)

    # Highlight price changes if chg cols exist
    for col in [c for c in d.columns if str(c).endswith("_chg")]:
        sty = sty.applymap(_chg_color, subset=[col])

    return sty
