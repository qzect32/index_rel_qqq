# Nasdaq-100 Exposure Graph (ETF Mapper)

Local-first Python project that builds a **relationship graph** from the Nasdaq-100 constituents to ETFs that reference them.

Right now the project focuses on **derivative-style exposure** (index + single-stock daily leveraged/inverse products) and is structured to expand into **holdings-based discovery** ("every ETF that holds NVDA/TSLA/etc") next.

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

Build the local universe:

```bash
python -m etf_mapper.cli refresh --out data
```

Explore:

```bash
jupyter lab
# open notebooks/01_explore_universe.ipynb
```

## Notes

- Direxion is fetched from an official PDF list (more reliable than HTML scraping).
- Some issuer sites block scraping in certain environments. The pipeline prints row counts per provider to make failures obvious.
- The "full gambit" (all ETFs related to every Nasdaq-100 stock) requires holdings-based discovery; a provider stub is included so you can drop in SEC N-PORT or a vendor API without changing the schema.
