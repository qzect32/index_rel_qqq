# ETF Hub (ETF Mapper)

Local-first Python project that builds a **US ticker universe + relationship graph** and serves it through a lightweight **Streamlit UI**.

Originally this started as a Nasdaq-100 exposure graph. It now aims to become a general-purpose “hub” to explore **stocks, ETFs, and their relationships**, with bootstrap market data (prices/options) today and broker-grade integrations (Schwab/TOS) later.

## What changed today (high-level)

- Added a **US ETF universe fetch** (Polygon reference tickers) → `data/etf_universe.{parquet,sqlite}`.
- Switched price history + live chart data to **Schwab Market Data** (OAuth).
- Built a **Streamlit UI** with:
  - search + single “Ticker” source of truth
  - price chart with **TOS-like timeframe presets**
  - options chain presented as a **TOS-style ladder** + a functional **cart** with premium estimates
  - a **relations graph** (interactive/force layout) with focus/expand controls
  - admin utilities + diagnostics
- Hardened common failure modes (empty ticker, malformed prices DB, missing provider fields).

## Near-term expectations

- Schwab Developer Portal integration is now the primary market data source (quotes + option chains + 1m candles). Order placement remains optional/guarded.
- Expand “relations” from derivative-style exposures to true **holdings-based discovery** (SEC N-PORT / vendor holdings):
  - ETF → holdings (equities)
  - equity → ETFs that hold it
- Improve graph UX (better centering/fit, click-to-focus, and compact default views).


## What it produces

The `refresh` pipeline outputs both Parquet and SQLite (so notebooks can run without extra parquet engines):

- `data/equities.parquet` — Nasdaq-100 constituents (expected ~100 rows)
- `data/etfs.parquet` — ETF master table (deduped by ticker)
- `data/edges.parquet` — relationship edges (src → dst), with classification fields
- `data/graph.gexf` — a graph you can open in Gephi
- `data/universe.sqlite` — same tables in SQLite

Backward-compatible artifacts are also written:
- `data/nasdaq100_constituents.parquet`
- `data/levered_etfs.parquet`

## Current edge types

- `derivative_exposure` — single-stock or index products that explicitly target an underlying (e.g. TSLA → TSLL, ^NDX → TQQQ)

Planned:
- `holds` — ETF holdings relationships (equity ∈ ETF) with `weight` (% holding) and `asof` date.

## Classification fields

Edges include fields that make the dataset human-queryable:

- `direction`: `long` / `short`
- `leverage_multiple`: `1.25`, `2`, `3`, etc (or null)
- `strategy_group`: `leveraged`, `inverse`, `options_covered_call`, `options_defined_outcome`, `plain`, …
- `relationship_group`: higher-level grouping (single-stock vs index, options strategy vs leveraged, etc.)

## Quickstart

Create and activate a virtualenv, then install deps:

```bash
python -m pip install -r requirements.txt
```

Build the Nasdaq-100 exposure graph universe:

```bash
python -m etf_mapper.cli refresh --out data
```

Fetch the full US ETF universe (master list of ETF tickers):

```bash
# requires POLYGON_API_KEY in your environment
python -m etf_mapper.cli universe --out data --provider polygon
```

Fetch daily price history via Schwab (OAuth required):

```bash
python -m etf_mapper.cli prices --out data --universe data/etf_universe.parquet --provider schwab --limit 200 --start 2024-01-01
```

One-shot bootstrap (universe + first batch of prices):

```bash
# requires POLYGON_API_KEY + Schwab OAuth env vars
python -m etf_mapper.cli bootstrap --out data --universe-provider polygon --price-provider schwab --start 2024-01-01 --limit 200
```

Run the UI (Streamlit):

```bash
python -m streamlit run app/streamlit_app.py
```

---

**Partner update (1-liner):** Follow progress in chat — we pushed a working ETF Hub UI (universe + prices + options ladder + relations graph + cart) and will wire Schwab/TOS next.

Explore:

```bash
jupyter lab
# open notebooks/01_explore_universe.ipynb
```

## Notes

- Direxion is fetched from an official PDF list (more reliable than HTML scraping).
- Some issuer sites block scraping in certain environments. The pipeline prints row counts per provider to make failures obvious.
- The "full gambit" (all ETFs related to every Nasdaq-100 stock) requires holdings-based discovery; a provider stub is included so you can drop in SEC N-PORT or a vendor API without changing the schema.
