# DEBUG_PLAYBOOK — Market Hub (when you git pull)

This is the "when it breaks, do this" guide.

## 0) Fastest triage (90 seconds)

### Overnight runner (recommended before you report bugs)

```bash
python scripts/overnight_run.py
```

Optional (auto-commit safe artifacts):

```bash
python scripts/overnight_run.py --commit
```

1) Confirm you pulled the right repo/branch:

```bash
git status
python --version
```

2) Run the doctor (offline-first):

```bash
python scripts/mh_doctor.py
```

3) If OAuth/Schwab data is involved, run live smoke (read-only):

```bash
python scripts/schwab_smoke.py --symbol QQQ
```

4) Create a debug bundle:

```bash
python scripts/make_debug_bundle.py
```

Upload/send the resulting zip path.

---

## 1) Common failure modes

### A) "Nothing updates" / listener version mismatch
- Ensure listener is running from THIS repo root.
- Use absolute paths:

```powershell
& "C:\Users\pwpat\AppData\Local\Programs\Python\Python312\python.exe" "<REPO>\scripts\decisions_listener.py" --port 8765 --repo "<REPO>"
```

- Verify:
  - http://127.0.0.1:8765/status
  - http://127.0.0.1:8765/schema

### B) Rate limit / budget blocked
- Look for banner "Guardrails: API budget/cooldown active"
- Clear in Sidebar → Guardrails → "Reset call counter / clear cooldown"

### C) OAuth configured but quotes fail
- Check:
  - `data/schwab_secrets.local.json` exists
  - `data/schwab_tokens.json` exists
  - token scope matches features used

### D) Feeds empty (halts/news/macro)
- These are cached.
- Check cache files under:
  - `data/feeds_cache/`

---

## 2) What to collect for a bug report

- Exact error text (copy/paste)
- A debug bundle zip (see above)
- Which tab you were on + which symbol(s)
- Whether this was after a fresh pull
