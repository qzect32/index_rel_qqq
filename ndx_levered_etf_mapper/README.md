# Market Hub (Schwab-only)

A local-first **Streamlit trading cockpit** powered by **Charles Schwab Market Data**.

This repo is intentionally practical: fast symbol lookup, live 1‑minute candles, options inspection, portfolio exposure, and a sandbox (“Casino Lab”) for quant toys/backtests.

> Status: actively evolving. The UI is usable today; Schwab-only market/trader data is wired, and several intel feeds (halts/macro/news) are now cached locally. Some Schwab endpoints (notably Alerts + potential Schwab earnings/news endpoints) are still unconfirmed.

---

## Changelog (running)

### 2026-02-08
- **Decisions Hub**: shipped multiple decision batches (performance/guardrails → feed quality → UI/UX+debugging) via `config/decisions_inbox_schema.json`
- **Decisions listener** (`scripts/decisions_listener.py`):
  - Filters out empty categories (prevents "notes-only" tabs after completing a batch)
  - Injects TODO triage questions dynamically
  - Adds decisions progress stats into schema meta (answered/total/%/remaining)
  - Adds `GET /debug_bundle` endpoint to download a sanitized debug zip
- **Decisions form** (`decisions_form.html`):
  - Progress pill now combines Decisions progress + normalized TODO progress
  - Build stamp + submit guard hardening (reduces submit-time "not rendered" edge cases)
- **Streamlit app** (`app/streamlit_app.py`):
  - Debugging UX upgraded: create bundle + download button + verbose tails for `spade_errors.jsonl` and `spade_session.jsonl`
- **Project ops**:
  - Refreshed `TODO_STATUS.md` (normalized tracking)
  - Added `scripts/metrics_24h.py` to compute last-24h commits/LOC/files + decisions/todo stats
  - README updated with Decisions Hub + debugging bundle docs

### 2026-02-07
- **Signals hub**: 3-panel layout (Halts | Earnings | Macro) + status badges
- **Earnings**: new Earnings tab + SEC EDGAR filings list + download → diff → score → report (md+json)
- **Filings watcher**: daily best-effort watcher while app is running + Signals badge + inline drilldown
- **Macro**: Fed RSS auto-feed cached locally + Signals panel + Dashboard highlight
- **News**: RSS auto-feed cached locally + Signals section + News tab + per-feed status + failure counts

### 2026-02-04
- Renamed project to **Market Hub** (from a prior internal name)
- Enforced **Schwab-only** outbound (Polygon removed)
- Live quote + **1m candles** with visible data-age indicators
- New **Dashboard** tab (watchlist + selected chart tiles)
- New **Scanner** tab (universe scan + Top 5 strip + focus rotation + Heat score + Hot List)
- New **Halts** tab (auto-fetch wired)
- New **Signals** tab (intel hub scaffold)
- New **Casino Lab** tab (Bayes module + toy backtests)
- New **Exposure** tab (accounts/positions, donut chart, redacted snapshot)
- Added sidebar calculator + watchlist tape
- Added margin discipline **countdown** (currently hard-coded for owner)
- Added `TODO.md` to track pending work, especially items requiring additional APIs/providers
- Known placeholders (not wired yet):
  - Alerts: Schwab-native/TOS alerts (endpoints/scopes not confirmed)
  - Schwab earnings/news endpoints: still unconfirmed (we currently use EDGAR + RSS)

---

## What it does (today)

### Dashboard
- **Watchlist** quotes table (Schwab quotes)
- **Selected symbol**: big price card + **1m candles** (Schwab price history)
- **Halts highlight** tile (reads cached halts)
- **Next macro event** highlight (reads cached Fed RSS; can be overridden manually)
- **Alerts**: UI placeholder for Schwab-native/TOS alerts (waiting on endpoint docs)

### Trading / Overview (single-symbol)
- Symbol profile/metadata (Schwab quotes)
- Live **1‑minute candles** (Schwab `pricehistory`)
- Data-age indicators (quote time + last candle time)
- **Countdown**: a configurable “line in the sand” timer (currently hard-coded for the project owner)

### Options
- Options expirations + chain (Schwab `chains`)
- Ladder-style view and position building helpers (no live order placement wired)

### Exposure
- Pulls accounts + positions (Schwab Trader endpoints)
- Aggregates exposure and renders a donut chart + table
- Groups options/futures **under the underlying** when Schwab provides underlying metadata
- Generates a **redacted shareable snapshot** (text + download)
- PDF exports available under **Exports** tab

### Casino Lab
- Quant playground (Bayes + toy backtests) using Schwab 1m candles

### Relations / Dataset export
- Builds a relationship graph and exports local datasets:
  - Parquet + SQLite (for notebooks, research, and agentic workflows)
  - Graph export (GEXF)

### Signals / News / Earnings
- **Signals** hub (halts + earnings + macro + filings alerts)
- **News** tab: cached RSS board + ticker detection (quotes via Schwab)
- **Earnings** tab: EDGAR filings list + diff/scoring reports (md+json)

---

## Primary rule: Schwab-only outbound

All outbound market/trader requests are Schwab endpoints:
- Quotes → Schwab Market Data (`/marketdata/v1/quotes`)
- 1-minute candles → Schwab Market Data (`/marketdata/v1/pricehistory`)
- Options chains → Schwab Market Data (`/marketdata/v1/chains`)
- Accounts/positions → Schwab Trader (`/trader/v1/...`)

Polygon support has been removed.

---

## Installation

### 1) Prerequisites
- **Python 3.10+**
- A Schwab Developer Portal app (client id/secret)

### 2) Create a virtualenv and install

```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
```

### 3) Configure Schwab OAuth secrets (local-only)

Create a gitignored file:

`data/schwab_secrets.local.json`

Example:

```json
{
  "SCHWAB_CLIENT_ID": "...",
  "SCHWAB_CLIENT_SECRET": "...",
  "SCHWAB_REDIRECT_URI": "http://127.0.0.1:8501",
  "SCHWAB_TOKEN_PATH": "data/schwab_tokens.json",
  "SCHWAB_OAUTH_SCOPE": "readonly"
}
```

Notes:
- This file is intentionally **local-first** and **gitignored**.
- Tokens are stored at `data/schwab_tokens.json` by default.

### 4) Run the UI

```bash
python -m streamlit run app/streamlit_app.py
```

Then open **Admin → Schwab OAuth** and run the OAuth authorization flow.

---

## Support / Debugging (recommended)

### Debug bundle (Streamlit)
In the app, open **Admin → Support / Debug bundle**:
- Click **Create debug bundle**
- Then **Download last bundle**
- Review **Recent errors** + **Session events** tails (sanitized)

### Debug bundle (listener endpoint)
If you're running the local decisions listener on port 8765, you can download a sanitized bundle at:

- `http://127.0.0.1:8765/debug_bundle`

---

## Decisions Hub (local decisions inbox)

A tiny local listener drives `decisions_form.html` so you can rapidly answer “batch” questions and persist them to `data/decisions.json`.

Run the listener:

```bash
python scripts/decisions_listener.py --port 8765
```

Then open `decisions_form.html` and click **Reload schema** / **Submit**.

---

## CLI (datasets)

Build the relationships graph and export datasets to `data/`:

```bash
market-hub refresh --out data
```

Fetch daily bars via Schwab for a provided universe file:

```bash
market-hub prices --out data --universe data/etf_universe.parquet --provider schwab --start 2024-01-01 --limit 200
```

---

## Troubleshooting

- If quotes/options/candles return empty: check **Admin → Schwab OAuth** and confirm tokens exist.
- If **Exposure** shows accounts but not positions: your Schwab app may not be entitled for trader endpoints, or the account payload schema differs. Expand the “debug/raw payload” section and open an issue with the sanitized output.

---

## Safety / non-goals

- The UI does **not** place live orders.
- Casino Lab modules are exploratory. They are not trading advice.

---

## Roadmap (short)

- Schwab-native alerts (create/list/delete) once endpoint docs are confirmed
- Better exposure analytics (risk, concentration, hedges)
- Options/Greeks-aware analytics + backtesting extensions
- Real headlines feed (provider TBD)
