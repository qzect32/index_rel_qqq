# TODO_SPRINT — Batches 7–12 implementation (data reader)

This is the *active* sprint checklist. When this file is 100% DONE, we are ready for QA/debug in Market Hub.

## Scanner
- [x] Apply scanner7–12 defaults from decisions.json into session_state
- [x] Scan budget meter UI (calls/min + cap + cooldown timer)
- [x] Secondary sort tie-break (symbol / %move / $vol)
- [x] Focus: show reason/why-hot (rank factors)
- [x] Focus: Prev/Next navigation controls
- [x] Focus: multi-pin list + remove + pin limit + autorotate
- [x] Focus: set News filter + Halts/Filings highlights based on focus

## Signals
- [x] Apply signals7–12 defaults from decisions.json into session_state
- [x] Section order (Halts → News → Filings → Macro)
- [x] Halts: include resumed within timewindow
- [x] Focus symbol filtering controls
- [x] Perf caps: rows per section
- [x] 1-click copy markdown snapshot
- [x] Onepager export (markdown)

## News
- [x] Apply news7–12 defaults from decisions.json into session_state
- [x] Reader-mode panel (best-effort extract + cache + timeout)
- [x] Ticker highlight + tagging (earnings/filings/macro)
- [x] Failures threshold display
- [x] Grouped-by-symbol compact view
- [x] Strict dedupe + limits
- [x] Export markdown (limit + include snippets/sources)

## Wall
- [x] Apply wall7–12 defaults from decisions.json into session_state
- [x] Stale definition (2m/5m/15m) + hide-stale
- [x] Movers-only toggle + filter buttons
- [x] Export bundle options + include badges/heat/stale
- [x] Focus sync: clicking tile updates scanner focus + news filter
- [x] Layout mode: Grid (masonry treated as scaffold/no-op)

## Done gate
- [x] Sprint complete: all items checked and status shows 100%
