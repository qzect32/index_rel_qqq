# TODO (Actionable)

This file tracks **pending work** with emphasis on items that require **new external APIs / data sources** beyond the current Schwab-only build.

Guiding rule:
- **Broker + market data** stays Schwab.
- Anything else is a **separate feed** (public/free/paid) and must be explicitly chosen.

---

## Workflow: scaffold-first + status tracking

Yes, we can implement *most* of these as scaffolds first (UI panels, placeholders, local calculations) and then progressively “light them up” with real endpoints.

**Status tags (use one):**
- `STATUS: SCaffolded` (UI exists / placeholders exist)
- `STATUS: IN-PROGRESS` (actively being wired)
- `STATUS: BLOCKED` (missing endpoint/provider/decision)
- `STATUS: DONE`

**For each TODO that matters, add 3 sub-bullets:**
- `STATUS:` one of the above
- `NEXT:` the next concrete step to finish
- `BLOCKERS:` only if something is missing (endpoint docs, entitlement, provider choice)

Decision-related blocks should be captured as explicit TODOs under “Implementation notes / decisions”.

---

## 0) Inventory: what currently requires external APIs / sources

### A) Schwab APIs we *haven’t confirmed / wired* yet

1) **Schwab-native Alerts (TOS/Schwab Mobile)**
   - Goal: create/read/delete price alerts inside Schwab so **Schwab** sends push/SMS.
   - STATUS: SCaffolded (UI placeholders exist)
   - NEXT: confirm Schwab alerts endpoints + required OAuth scopes, then implement create/list/delete in `SchwabAPI` + wire UI.
   - BLOCKERS: Schwab Developer Portal docs/endpoints not yet accessible in this workspace.
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
   - STATUS: SCaffolded (paste/parse UI exists)
   - NEXT: choose halt feed source(s), implement scheduled fetch + parse + reason-code classifier, render "resuming soon" highlights.
   - BLOCKERS: outbound source choice + allowed domains policy.
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
- [ ] **Earnings + filings + votes ingestion (sentiment)**
  - [ ] Pull earnings call transcripts/audio links, 10-Q/10-K, and proxy/vote items (DEF 14A)
  - [ ] Extract key sections (guidance, risks, MD&A, buybacks, dilution, comp changes)
  - [ ] Run sentiment + change detection quarter-over-quarter (tone shift, new risk language)
  - [ ] Output: per-ticker “what changed” summary + catalysts + red flags
  - [ ] Source TBD: SEC EDGAR + earnings transcript provider (free/paid)- [ ] **Commodities signal panels**
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

## 4) Quant-grade trading interface wishlist (50) — fleshed out

Each line item below is written as a **unique UI need** with a brief rationale and the likely data dependency.

1. [ ] **DOM/Level II view**
   - Need: see depth/liquidity to avoid entering/exiting into thin books.
   - Requires: Level II feed (Schwab if available; otherwise external).
2. [ ] **Time & Sales tape**
   - Need: confirm real participation/urgency (prints) rather than just candles.
   - Requires: tick/print stream (data source dependent).
3. [ ] **VWAP bands**
   - Need: quantify mean-reversion vs trend strength around VWAP.
   - Requires: intraday candles (Schwab 1m).
4. [ ] **Multi-timeframe chart grid**
   - Need: avoid getting trapped by 1m noise; confirm alignment across frames.
   - Requires: multiple bar sizes (Schwab price history config).
5. [ ] **Session shading (PM/RTH/AH)**
   - Need: prevent misreading illiquid pre/after-hours moves.
   - Requires: timestamps + trading session rules.
6. [ ] **Anchored VWAP (AVWAP)**
   - Need: measure price relative to a specific event anchor (earnings, breakout).
   - Requires: intraday candles + chosen anchor.
7. [ ] **Opening Range (ORH/ORL) tool**
   - Need: classic intraday playbook support with consistent definitions.
   - Requires: intraday candles + market open time.
8. [ ] **ATR-based stops calculator**
   - Need: stops sized to volatility so you don’t get wicked out by noise.
   - Requires: recent candles (1m/5m) + ATR calculation.
9. [ ] **Volume profile proxy**
   - Need: identify high-volume nodes (support/resistance) intraday.
   - Requires: intraday volume by price (approx via candle bins).
10. [ ] **Market internals tile**
   - Need: detect “index is propped but internals are weak/strong”.
   - Requires: internals feed (A/D, ticks) — likely external.
11. [ ] **Sector breadth heatmap**
   - Need: know if move is isolated or broad across a sector.
   - Requires: sector mapping + sector constituents data.
12. [ ] **Relative strength vs SPY/QQQ**
   - Need: trade winners (outperformers) not just absolute movers.
   - Requires: candles for symbol + benchmark.
13. [ ] **Rolling beta/correlation vs index**
   - Need: understand hidden leverage to market beta.
   - Requires: returns history for both series.
14. [ ] **Volatility cone**
   - Need: contextualize realized vol vs historical distribution.
   - Requires: longer-term price history.
15. [ ] **IV rank / IV percentile**
   - Need: decide option selling/buying regimes rationally.
   - Requires: historical IV time series (external unless broker provides).
16. [ ] **IV skew chart (strike)**
   - Need: see tail pricing and put/call demand.
   - Requires: option chain IV across strikes.
17. [ ] **IV term structure (expiration)**
   - Need: spot event risk vs long-dated pricing.
   - Requires: option chain IV across expirations.
18. [ ] **Greeks totals per underlying**
   - Need: know your true delta/gamma/theta exposure at a glance.
   - Requires: chain greeks or model greeks per leg.
19. [ ] **Gamma exposure (GEX) estimate**
   - Need: anticipate pinning/accelerations near key strikes.
   - Requires: OI by strike + IV + model assumptions (often external).
20. [ ] **Dealer positioning proxy**
   - Need: map “who’s trapped” scenarios; helps fade/chase decisions.
   - Requires: specialized data/model; likely external.
21. [ ] **Max pain estimate**
   - Need: rough magnet level into expiry for short-term options context.
   - Requires: OI by strike (chain) + calc.
22. [ ] **OI + volume by strike ladder**
   - Need: quickly find “where the fight is” on a chain.
   - Requires: chain OI/volume by strike.
23. [ ] **Unusual options activity detector**
   - Need: identify flow-driven names without staring at everything.
   - Requires: options prints/sweeps feed — usually external.
24. [ ] **News/catalyst timeline on chart**
   - Need: explain price action and anchor risk to time.
   - Requires: news/events feed with timestamps.
25. [ ] **Event markers (earnings/Fed/CPI)**
   - Need: avoid holding risk unknowingly into binary events.
   - Requires: macro/earnings calendar feed.
26. [ ] **Earnings whisper vs reported**
   - Need: measure surprise magnitude vs expectations.
   - Requires: estimates + reported numbers (external).
27. [ ] **Halt/LULD markers on chart**
   - Need: understand discontinuities and resume behavior.
   - Requires: halts feed + timestamps.
28. [ ] **Gap scanner**
   - Need: surf the premarket opportunity set fast.
   - Requires: premarket quotes + volume; often external.
29. [ ] **Top movers with filters**
   - Need: surface tradeable movers and exclude junk.
   - Requires: market-wide mover feed OR curated universe.
30. [ ] **Top active (market-wide)**
   - Need: always know where liquidity is today.
   - Requires: market-wide volume/$ volume feed.
31. [ ] **Multi-symbol alert rules**
   - Need: alert on regimes (SPY down + vol up) not single tickers.
   - Requires: alerts engine + multi-symbol quotes.
32. [ ] **Trailing stop suggestions (UI-only)**
   - Need: take emotion out of managing winners.
   - Requires: candle stream + trailing logic.
33. [ ] **Position sizing calculator**
   - Need: translate risk ($) → shares/contracts instantly.
   - Requires: entry/stop + account equity/budget.
34. [ ] **Kelly fraction estimator (guarded)**
   - Need: sizing model for edges with known winrate/odds.
   - Requires: winrate/EV inputs (from journal/backtests).
35. [ ] **Expected Value (EV) calculator**
   - Need: compare setups on math, not vibes.
   - Requires: probability + payoff assumptions.
36. [ ] **R-multiple tracking**
   - Need: measure performance independent of instrument price.
   - Requires: entry/stop + exits + journal.
37. [ ] **Trade replay (bar-by-bar)**
   - Need: post-mortems and skill-building.
   - Requires: stored intraday candles.
38. [ ] **Backtest library**
   - Need: re-run proven systems quickly with parameters.
   - Requires: historical data + strategy definitions.
39. [ ] **Walk-forward validation**
   - Need: reduce overfitting by testing across regimes.
   - Requires: historical data + splits.
40. [ ] **Monte Carlo on trade outcomes**
   - Need: understand drawdown distributions and risk of ruin.
   - Requires: trade outcome series.
41. [ ] **Slippage model**
   - Need: keep backtests honest and position sizing realistic.
   - Requires: spread/volume proxies; ideally prints.
42. [ ] **Commission/fees model**
   - Need: net P&L realism across asset types.
   - Requires: fee schedules.
43. [ ] **Portfolio Greeks (aggregate)**
   - Need: total risk across accounts, not per-position.
   - Requires: positions + greeks per leg.
44. [ ] **Risk parity/rebalancing suggestions**
   - Need: long-term portfolio hygiene and defense.
   - Requires: vol/corr estimates + holdings.
45. [ ] **Hedge finder**
   - Need: quickest way to reduce delta/beta when storms hit.
   - Requires: correlation/beta estimates + hedge instruments.
46. [ ] **Drawdown guard**
   - Need: prevent spiral trading; enforce cool-down rules.
   - Requires: P&L tracking + thresholds.
47. [ ] **Stress correlation (“tail corr”) proxy**
   - Need: correlations change in selloffs; warn early.
   - Requires: longer history + regime logic.
48. [ ] **Regime switch detector**
   - Need: know when trend systems vs mean reversion should be active.
   - Requires: return features + classifier.
49. [ ] **Execution checklist**
   - Need: reduce dumb mistakes (size, stop, catalyst) before entry.
   - Requires: UI workflow + journal.
50. [ ] **Audit log of user actions**
   - Need: reproduce “what did I change?” across settings/trades.
   - Requires: local logging + timestamps.

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

## 6) Implementation notes / decisions to make later

- Allowed outbound domains list (if we start pulling halts/news/macro from public sources)
- Whether we want an internal “data repo” or keep everything live + local snapshots
- Whether we want to support live order placement (currently intentionally NOT wired)
  - If yes: build a HARD safety gate (paper mode toggle, confirm screens, allowlist symbols)
- Whether this becomes a personal cockpit only vs a sharable product (impacts secrets handling + multi-user)
- OAuth/token architecture decision:
  - [ ] Confirm whether Schwab marketdata + trader + alerts (if any) can share the same token set/scopes
  - [ ] Confirm whether we need separate apps/scopes for read-only vs trading endpoints
  - [ ] Build a single “connection status” panel that validates: quotes, pricehistory, chains, trader, alerts
