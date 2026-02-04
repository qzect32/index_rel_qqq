# ETF Hub (ETF Mapper)

Local-first Python project that builds a **relationship graph** and serves it through a lightweight **Streamlit UI**.

## Primary rule: Schwab-only

This repo is intentionally **Schwab-only** for anything outbound:
- Quotes → Schwab Market Data
- 1-minute candles → Schwab Market Data (`pricehistory`)
- Options chains → Schwab Market Data (`chains`)

Polygon support has been removed.

## What it produces

The `refresh` pipeline outputs both Parquet and SQLite:

- `data/equities.parquet` — Nasdaq-100 constituents (expected ~100 rows)
- `data/etfs.parquet` — ETF master table (deduped by ticker)
- `data/edges.parquet` — relationship edges (src → dst)
- `data/graph.gexf` — graph for Gephi
- `data/universe.sqlite` — same tables in SQLite

Backward-compatible artifacts may also be written:
- `data/nasdaq100_constituents.parquet`
- `data/levered_etfs.parquet`

## Quickstart

Create and activate a virtualenv, then install deps:

```bash
python -m pip install -r requirements.txt
```

Build the relationships graph:

```bash
python -m etf_mapper.cli refresh --out data
```

Run the UI (Streamlit):

```bash
python -m streamlit run app/streamlit_app.py
```

## Schwab OAuth config

Recommended: create a local, gitignored secrets file:

- `data/schwab_secrets.local.json`

Expected keys (either env-style or snake_case are accepted):

```json
{
  "SCHWAB_CLIENT_ID": "...",
  "SCHWAB_CLIENT_SECRET": "...",
  "SCHWAB_REDIRECT_URI": "http://127.0.0.1:8501",
  "SCHWAB_TOKEN_PATH": "data/schwab_tokens.json",
  "SCHWAB_OAUTH_SCOPE": "readonly"
}
```

## Notes / roadmap

- The UI focuses on a single ticker input, with live Schwab quote + 1m candles.
- Relationship graph is currently seeded from derivative-style exposures; holdings-based discovery can come later.
