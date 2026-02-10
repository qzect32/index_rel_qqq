# Changelog

## 0.2.0 (2026-02-09)

### UX
- Reduced Streamlit "rerun jitter" by moving high-churn controls behind forms and debounced snapshots.
- Centralized/normalized empty-state copy.

### Persistence & hardening
- Atomic JSON writes with `.bak` fallback for multiple persisted artifacts (settings, decisions, scanner views, hotlist).
- Settings import now only accepts known keys.
- More robust data-dir resolution + auto-create data folder.
- Offline quote fallback from local daily prices.
- SQLite introspection hardening (missing/corrupt DB => safe empty).

### Debuggability
- Debug bundle branding normalization to "Market Hub" + tail reader improvements.
- Instrumentation panel improvements (cache hit/miss stats).
- FlightRecorder made fail-safe (never crash app on IO).

### Tooling
- "doctor" script hides internal package paths.
- Overnight runner wrappers.

