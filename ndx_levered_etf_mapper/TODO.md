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
- [ ] **Wide monitor wall (“50-ticker view”)**
  - [ ] Grid view (e.g., 5×10) with minimal per-ticker info
  - [ ] Price + % change + tiny sparkline (“micro squiggle”)
  - [ ] Quick filter: only show movers / only show hotlist / only show watchlist
  - [ ] Optional pattern tagger (basic: trend up/down, range, spike)
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
- [ ] **Chart indicators (first-class)**
  - [ ] Overlay indicators (VWAP, EMAs, RSI, MACD, ATR, volume profile-ish proxy)
  - [ ] Indicator presets per Event mode
  - [ ] Scanner filters driven by indicators (e.g., RSI>70, VWAP reclaim)
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

## 3) Fringe cases / edge conditions (add tests + UI behavior)

These are common failure modes in broker-grade data and trading workflows.

- [ ] **Stale/partial quote payloads**
  - Some Schwab quote schemas omit timestamps or return `None` fields; ensure UI shows “stale/unknown” rather than misleading values.
- [ ] **Market hours vs extended hours**
  - Pre/after-hours can have thin prints; ensure candles/quotes clearly indicate session and don’t mix RTH/ETH unintentionally.
- [ ] **Symbol normalization oddities**
  - Futures (`/ES`), indices (`$SPX`, `^NDX`), options symbols, and OTC symbols may require different formatting; add a normalization map + clear errors.
- [ ] **Rate limit / throttle behavior**
  - Handle 429/backoff centrally; Scanner should degrade gracefully (reduce universe, increase cache TTL, show “limited”).
- [ ] **Entitlement gaps (accounts but no positions / options but no chains)**
  - Detect “not entitled” vs “empty” and show a precise next-step message in Admin.
- [ ] **Corporate actions & symbol changes**
  - Splits, reverse splits, ticker changes, delistings: ensure exposure snapshots and scanner don’t silently break; add a “symbol changed” warning when detected.

---

## 4) Quant-grade trading interface wishlist (50)

1. [ ] **DOM/Level II view** (if data source allows)
2. [ ] **Time & Sales** tape (prints, size, aggressor side if available)
3. [ ] **VWAP bands** (VWAP ±1/±2 stdev)
4. [ ] **Multi-timeframe chart grid** (1m/5m/15m/1h/1d)
5. [ ] **Session shading** (premarket/RTH/after-hours)
6. [ ] **Anchored VWAP** (from event/date/high/low)
7. [ ] **ORH/ORL (Opening Range)** indicator + breakout signals
8. [ ] **ATR-based stops** calculator + overlays
9. [ ] **Volume profile** proxy (intraday histogram)
10. [ ] **Market internals tile** (A/D, TRIN proxies, ticks — source TBD)
11. [ ] **Breadth by sector** heatmap (requires sector mapping)
12. [ ] **Relative strength vs SPY/QQQ** line
13. [ ] **Beta / correlation vs index** (rolling)
14. [ ] **Volatility cone** (realized vol vs history)
15. [ ] **IV rank / IV percentile** (requires options IV history)
16. [ ] **Skew chart** (put/call IV skew by strike)
17. [ ] **Term structure** (IV by expiration)
18. [ ] **Greeks by underlying** (delta/gamma/theta/vega totals)
19. [ ] **Gamma exposure (GEX) estimate** (source/model TBD)
20. [ ] **Dealer positioning proxy** (if feasible)
21. [ ] **Max pain** estimate (options OI-based)
22. [ ] **OI + volume by strike** ladder visualization
23. [ ] **Unusual options activity detector** (sweeps/blocks — source TBD)
24. [ ] **News-catalyst timeline** pinned to chart (source TBD)
25. [ ] **Event markers** (earnings, Fed, CPI) on charts (source TBD)
26. [ ] **Earnings whisper vs reported** (source TBD)
27. [ ] **Halt/LULD markers** on chart (halts feed required)
28. [ ] **Gap scanner** (premarket gap %, volume)
29. [ ] **Top movers** (gainers/losers) with filters
30. [ ] **Top active** (volume + dollar volume) market-wide (source TBD)
31. [ ] **Multi-symbol alert rules** (e.g., SPY down AND VIX up)
32. [ ] **Trailing stop automation suggestions** (UI-only)
33. [ ] **Position sizing calculator** (risk per trade → shares/contracts)
34. [ ] **Kelly fraction estimator** (optional, guarded)
35. [ ] **Expected value (EV) calculator** for setups
36. [ ] **R-multiple tracking** for each trade
37. [ ] **Trade replay** (bar-by-bar playback for a day)
38. [ ] **Backtest library** with saved strategies + parameters
39. [ ] **Walk-forward validation** (basic)
40. [ ] **Monte Carlo on trade outcomes** (equity curve distribution)
41. [ ] **Slippage model** (spread/volume-based)
42. [ ] **Commission/fees model** per asset
43. [ ] **Portfolio Greeks** (aggregate across accounts)
44. [ ] **Risk parity / rebalancing suggestions** (optional)
45. [ ] **Hedge finder** (what reduces delta/beta fastest)
46. [ ] **Drawdown guard** (auto-warn and lock suggestions when DD spikes)
47. [ ] **Correlation breakdown under stress** (tail correlation proxy)
48. [ ] **Regime switch detector** (trend vs chop classifier)
49. [ ] **Execution checklist** (pre-trade, during, post-trade)
50. [ ] **Audit log** of user actions (what changed, when)

## 5) QA / quality methodologies (6)

- [ ] **Golden-record fixtures**
  - Capture sanitized JSON responses for quotes/pricehistory/chains/accounts and replay them in tests.
- [ ] **Contract tests for API clients**
  - Validate required keys/types per endpoint and fail loudly when Schwab schema changes.
- [ ] **Latency + request-budget monitoring**
  - Log per-tile timings and request counts; set thresholds that fail CI when regressions occur.
- [ ] **Chaos testing (simulated bad data)**
  - Inject missing fields, NaNs, empty candles, and entitlement errors to verify graceful UI degradation.
- [ ] **Repro bundles for bug reports**
  - One-click “debug bundle” includes recent logs + anonymized payload samples + app config (no secrets).
- [ ] **Manual smoke checklist**
  - A short checklist for releases: OAuth connect, quote load, 1m candles, options chain, exposure load, scanner scan.

---

## 5) Implementation notes / decisions to make later

- Allowed outbound domains list (if we start pulling halts/news/macro from public sources)
- Whether we want an internal “data repo” or keep everything live + local snapshots
- Whether we want to support live order placement (currently intentionally NOT wired)
  - If yes: build a HARD safety gate (paper mode toggle, confirm screens, allowlist symbols)
- Whether this becomes a personal cockpit only vs a sharable product (impacts secrets handling + multi-user)
- OAuth/token architecture decision:
  - [ ] Confirm whether Schwab marketdata + trader + alerts (if any) can share the same token set/scopes
  - [ ] Confirm whether we need separate apps/scopes for read-only vs trading endpoints
  - [ ] Build a single “connection status” panel that validates: quotes, pricehistory, chains, trader, alerts
