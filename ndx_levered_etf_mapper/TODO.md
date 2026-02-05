# TODO (Actionable)

This file tracks **pending work** with emphasis on items that require **new external APIs / data sources** beyond the current Schwab-only build.

Guiding rule:
- **Broker + market data** stays Schwab.
- Anything else is a **separate feed** (public/free/paid) and must be explicitly chosen.

---

## 0) Inventory: what currently requires external APIs / sources

### A) Schwab APIs we *haven’t confirmed / wired* yet

1) **Schwab-native Alerts (TOS/Schwab Mobile)**
   - Goal: create/read/delete price alerts inside Schwab so **Schwab** sends push/SMS.
   - Status: UI has placeholders; endpoints not confirmed.
   - Needed:
     - Identify Schwab Developer Portal endpoints (paths, scopes, payloads)
     - Implement client methods + UI flows
     - Read existing alerts + delete/disable + expiry

2) **Schwab market movers / most active** (if exists)
   - Goal: “Top 5 most active trading stocks” without manually curating symbols.
   - Status: not implemented.
   - Needed:
     - Confirm whether Schwab has a movers/actives endpoint
     - If not: choose another source (see below)

3) **Schwab news endpoints** (if exists)
   - Goal: “Why is it moving?” + headline list tied to symbols.
   - Status: manual paste areas only.
   - Needed:
     - Confirm whether Schwab provides news via API
     - If not: choose another provider (RSS, paid news API, etc.)

4) **Schwab account balances fields** (Exposure)
   - Goal: show account cards: account name/type, net liq, cash, buying power, etc.
   - Status: positions view is best-effort; balances not yet standardized.
   - Needed:
     - Confirm correct `fields=` query or endpoint(s) to retrieve balances
     - Normalize schemas across account types

5) **Futures contract metadata / continuous futures mapping**
   - Goal: handle `/ES`, `/NQ`, micro contracts cleanly (symbols, trading hours, roll logic).
   - Status: candles are attempted; metadata is inconsistent.
   - Needed:
     - Confirm Schwab symbol conventions + supported contracts


### B) Non-Schwab sources we need to choose (public/free/paid)

1) **Trading halts feed (Nasdaq / NYSE)**
   - Goal: dedicated “Halts” tab that auto-updates.
   - Status: paste-CSV scaffold.
   - Candidates:
     - Nasdaq Trader halt list
     - NYSE halt list
   - Needed:
     - Decide allowed outbound domains
     - Implement fetch + parse + reason-code classification + highlight resumptions

2) **Macro calendar (Fed, CPI, NFP, speakers, etc.)**
   - Goal: event mode planning + countdowns + weekly planning.
   - Status: manual paste scaffold.
   - Candidates:
     - FOMC schedule + Federal Reserve speeches
     - Economic calendar source (free or paid)
   - Needed:
     - Choose a canonical feed format
     - Add “next event” tile + link-outs

3) **Earnings calendar**
   - Goal: plan the week by earnings; attach call links.
   - Status: idea only.
   - Candidates:
     - Public earnings calendar (web scrape/RSS)
     - Paid calendar API
   - Needed:
     - Choose source
     - Add import (CSV/paste) as no-API fallback

4) **Broad market breadth / trending / volatility scanners**
   - Goal: true market-wide “trending / popping” lists (beyond our curated universe).
   - Status: Scanner scans a user-defined universe only.
   - Candidates:
     - Exchange-provided lists
     - Paid market data vendor
   - Needed:
     - Decide whether to keep this curated or plug in a real breadth feed

5) **Commodities / shipping / tanker flows / inventories**
   - Goal: oil/nat gas intelligence, tanker tracking, inventory reports.
   - Status: Signals tab is manual notes.
   - Needed:
     - Choose data sources (AIS/tanker APIs, EIA inventories, etc.)

6) **Metals / mining discoveries / geopolitical supply shocks**
   - Goal: metals dashboard + event detection.
   - Status: Signals tab placeholder.
   - Needed:
     - Choose sources (commodities pricing + news/filings)

7) **Government / Congress / contracts intelligence**
   - Goal: detect policy/contract headlines that move tickers.
   - Status: manual notes.
   - Needed:
     - Choose sources (congressional feed, press releases, procurement data)

---

## 1) Product/UI TODO (doesn’t require new APIs)

- [ ] **TODO tracking in README changelog**: add a “Known placeholders” note per release.
- [ ] **Scanner “Hot List” improvements**
  - [ ] Show hot list as a tile on Dashboard
  - [ ] Add “remove” buttons per hot list symbol
  - [ ] Add mini sparkline (last N 1m bars) for each hot symbol (Schwab)
- [ ] **Heat score tuning**
  - [ ] Make weights configurable per Event mode
  - [ ] Add optional realized-vol component (requires 1m candles; keep throttled)
- [ ] **Exposure UX upgrades**
  - [ ] Collapsible account cards
  - [ ] Group positions under underlying (already best-effort)
  - [ ] Add “set as main ticker” buttons from exposure rows
- [ ] **PDF export** (local-only)
  - [ ] Export Exposure summary to PDF (one-page)
  - [ ] Export Scanner snapshot to PDF
- [ ] **Performance guardrails**
  - [ ] Hard caps + rate-limit backoff for quote loops
  - [ ] Cache strategy per tile (dashboard vs scanner)

---

## 2) Nice-to-haves (feature wishlist)

These are not required for “tomorrow’s API wiring” but are high ROI once the core endpoints are stable.

### Scanner / Dashboard
- [ ] **Composite scanner presets** (save/load named scanners)
  - [ ] Persist scanner configs to `data/scanners/*.json`
  - [ ] Quick-switch dropdown ("Momentum", "Vol", "Fed Day", "Earnings", "Semis", "Energy", etc.)
- [ ] **Gappers / Range / Volatility tiles**
  - [ ] Top gappers (pre/after-hours if supported)
  - [ ] Largest 1m range expansion (today)
  - [ ] Vol spike detector (realized vol vs trailing)
- [ ] **Circuit breaker proximity** (LULD bands / halt risk) once we have the data
- [ ] **Watchlist tape** improvements
  - [ ] Colorize by % change
  - [ ] Flash on threshold crossings

### Exposure / Risk
- [ ] **Always-on risk strip (“Oh sh*t bar”)**
  - [ ] Toggleable bar showing: risk state (G/Y/R), next catalyst countdown, quick scenario P&L
  - [ ] Flags: wide spread / low volume / stale data / missing quote time
- [ ] **Risk budget / guardrails**
  - [ ] Daily loss limit, per-trade loss limit, max % in one underlying
  - [ ] Warn if adding a trade would exceed guardrails
- [ ] **Exit plan required (lightweight)**
  - [ ] Store thesis, stop, target, time stop, invalidation note
  - [ ] Overlay stop/target on chart
  - [ ] “What changed since entry?” diff view
- [ ] **Exposure by category**
  - [ ] By sector
  - [ ] By asset class (equity/ETF/options/futures)
  - [ ] Leveraged vs unleveraged buckets
- [ ] **Concentration warnings**
  - [ ] "Top 1", "Top 3", "Top 5" concentration
  - [ ] Account-level vs total portfolio
- [ ] **Correlation / overlap warnings**
  - [ ] Detect clustered exposures (e.g., TSLA+TSLL+QQQ+TQQQ)
  - [ ] Simple correlation matrix from recent returns (throttled)
- [ ] **Scenario overlays**
  - [ ] "If TSLA -1%/-2%/-5%" → approximate impact on portfolio MV and day P/L
  - [ ] Event-mode stress templates (Fed day / CPI day)
- [ ] **Leverage / margin stress** (approx)
  - [ ] Gross exposure / equity
  - [ ] Maintenance/breach heuristics (best-effort)

### Options / Greeks
- [ ] **Greeks panel**
  - [ ] Delta/Gamma/Theta/Vega totals per underlying
  - [ ] Expiration ladder summaries
- [ ] **PnL surface** (price × time) for multi-leg positions
- [ ] **Assignment/exercise risk flags** for short legs near ITM

### Planning / Journaling
- [ ] **Weekly plan board**
  - [ ] Earnings week planner (import/paste calendar)
  - [ ] Fed week planner (auto countdowns per event)
  - [ ] “Binary days” tagging (Fed decision, CPI, NFP, big earnings)
  - [ ] One-click add countdown for selected event
- [ ] **Trade journal / post-mortems**
  - [ ] One-click snapshot of chart + exposure + notes saved locally
  - [ ] Export to markdown/PDF
  - [ ] "What changed since entry?" auto-diff block (price/IV/events/exposure)

### Macro / Signals (thought process expansion)
- [ ] **Macro regime dashboard**
  - [ ] Risk-on/off gauge (based on a small basket: SPY/QQQ/IWM, TLT, DXY proxy, GLD, oil proxy)
  - [ ] Vol proxy tile (VIX proxy ETFs + realized vol from 1m bars)
  - [ ] Credit stress proxy tile (HYG/LQD vs SPY spread)
- [ ] **Event heatmap**
  - [ ] Calendar view: Fed speakers, FOMC, CPI, PPI, NFP, auctions
  - [ ] Countdown widgets per event
- [ ] **Policy / Congress board (manual → feed later)**
  - [ ] Paste headlines + auto-detect tickers + affected sectors
  - [ ] “Likely beneficiaries / losers” note template
- [ ] **Commodities signal panels**
  - [ ] Oil: WTI/Brent proxies, nat gas, EIA weekly inventory placeholder
  - [ ] Metals: gold/silver/copper proxies + mining ETF proxies
- [ ] **"Storm watch" presets**
  - [ ] One-click switch event mode + watchlist + risk budgets
  - [ ] Pre-market checklist + post-event checklist

### UX / Ops
- [ ] **Session persistence**
  - [ ] Persist dashboard layout preferences
  - [ ] Persist pinned symbols per tab
- [ ] **Performance instrumentation**
  - [ ] Display per-tile latency (quotes/candles/options)
  - [ ] Simple request budget (calls/min) counters
- [ ] **Single-position cockpit mode ("Tournament Blue")**
  - [ ] Dedicated tab/view optimized for one symbol (minimal distractions)
  - [ ] Calm colorway option (tournament-blue felt / ocean vibe) for “in a trade” headspace
  - [ ] Optional chart overlays: stop/target, time-to-catalyst, position P&L

---

## 3) Implementation notes / decisions to make later

- Allowed outbound domains list (if we start pulling halts/news/macro from public sources)
- Whether we want an internal “data repo” or keep everything live + local snapshots
- Whether we want to support live order placement (currently intentionally NOT wired)
- Whether this becomes a personal cockpit only vs a sharable product (impacts secrets handling + multi-user)
