r"""Local Decisions Listener (no copy/paste).

Runs a tiny HTTP server on localhost that accepts POST /submit with JSON body.
Writes the payload to data/decisions.json and appends to data/decisions_log.jsonl.

Usage (PowerShell):
  cd <repo>
  python scripts\decisions_listener.py --port 8765

Or run from anywhere:
  python scripts\decisions_listener.py --port 8765 --repo <repo>

Then open decisions_form.html and click Submit.

Security: binds to 127.0.0.1 only.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import time
import re
import subprocess
import zipfile
from urllib.parse import urlparse, parse_qs


def _root(repo: str | None) -> Path:
    if repo:
        return Path(repo).resolve()
    return Path(__file__).resolve().parents[1]


def _data_dir(repo: str | None) -> Path:
    return (_root(repo) / "data").resolve()


def _todo_status_path(repo: str | None) -> Path:
    return (_root(repo) / "TODO_STATUS.md").resolve()


def _todo_stats(repo: str | None) -> dict:
    """Return TODO stats from TODO_STATUS.md if present.

    Raw pct_done counts only DONE.
    Normalized pct_norm treats IN-PROGRESS as partial credit to better reflect reality.
    """
    p = _todo_status_path(repo)
    if not p.exists():
        return {
            "total": 0,
            "done": 0,
            "in_progress": 0,
            "blocked": 0,
            "pct_done": 0.0,
            "pct_norm": 0.0,
        }

    total = 0
    done = 0
    in_progress = 0
    blocked = 0
    cur_status: str | None = None

    def _bump(status: str | None):
        nonlocal done, in_progress, blocked
        if not status:
            return
        s = status.strip().upper()
        if s == "DONE":
            done += 1
        elif s in ("IN-PROGRESS", "IN PROGRESS", "INPROGRESS"):
            in_progress += 1
        elif s == "BLOCKED":
            blocked += 1

    try:
        for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            if ln.startswith("## "):
                # finalize previous
                if cur_status is not None:
                    _bump(cur_status)
                total += 1
                cur_status = "IN-PROGRESS"  # default if missing
                continue
            if ln.strip().startswith("- STATUS:"):
                cur_status = ln.strip().split(":", 1)[1].strip()

        # finalize last
        if cur_status is not None:
            _bump(cur_status)
    except Exception:
        return {
            "total": 0,
            "done": 0,
            "in_progress": 0,
            "blocked": 0,
            "pct_done": 0.0,
            "pct_norm": 0.0,
        }

    pct_done = (float(done) / float(total) * 100.0) if total else 0.0

    # Normalization heuristic: DONE=1.0, IN-PROGRESS=0.35, BLOCKED=0.0
    credit = float(done) + 0.35 * float(in_progress)
    pct_norm = (credit / float(total) * 100.0) if total else 0.0

    return {
        "total": int(total),
        "done": int(done),
        "in_progress": int(in_progress),
        "blocked": int(blocked),
        "pct_done": round(pct_done, 1),
        "pct_norm": round(pct_norm, 1),
    }


def _todo_open_items(repo: str | None, *, limit: int = 15, answered: set[str] | None = None) -> list[dict]:
    """Parse TODO_STATUS.md and return a list of open items (not DONE).

    If `answered` is provided, skip items whose key `todo.todo_<n>` has been answered before.
    """
    p = _todo_status_path(repo)
    if not p.exists():
        return []

    items: list[dict] = []
    cur = None
    try:
        for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = re.match(r"^##\s+(\d+)\.\s+(.*)$", ln)
            if m:
                if cur and cur.get("status") != "DONE":
                    items.append(cur)
                cur = {"i": int(m.group(1)), "title": m.group(2).strip(), "status": "IN-PROGRESS"}
                continue
            if ln.strip().startswith("- STATUS:") and cur is not None:
                cur["status"] = ln.strip().split(":", 1)[1].strip()

        if cur and cur.get("status") != "DONE":
            items.append(cur)
    except Exception:
        return []

    if answered:
        items = [it for it in items if f"todo.todo_{it.get('i')}" not in answered]

    return items[: int(limit)]


def _load_schema(repo: str | None) -> dict:
    """Load schema from config/.. preferred, else data/.., else {}."""
    repo_root = _root(repo)
    schema_path = (repo_root / "config" / "decisions_inbox_schema.json").resolve()
    if not schema_path.exists():
        schema_path = (repo_root / "data" / "decisions_inbox_schema.json").resolve()
    if not schema_path.exists():
        return {}
    try:
        obj = json.loads(schema_path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _todo_md(repo: str | None) -> dict:
    """Return markdown contents for TODO.md and TODO_STATUS.md (best-effort)."""
    repo_root = _root(repo)
    out: dict = {"todo_md": "", "todo_status_md": ""}
    try:
        p1 = (repo_root / "TODO.md").resolve()
        if p1.exists():
            out["todo_md"] = p1.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        pass
    try:
        p2 = (repo_root / "TODO_STATUS.md").resolve()
        if p2.exists():
            out["todo_status_md"] = p2.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        pass
    return out


def _git_velocity(repo: str | None) -> dict:
    """Best-effort git stats since local midnight."""
    repo_root = _root(repo)
    try:
        since = time.strftime("%Y-%m-%d 00:00")
        # commits
        r1 = subprocess.run(
            ["git", "log", f"--since={since}", "--pretty=oneline", "--no-merges"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=8,
        )
        commits = 0
        if r1.returncode == 0:
            commits = len([ln for ln in (r1.stdout or "").splitlines() if ln.strip()])

        # numstat
        r2 = subprocess.run(
            ["git", "log", f"--since={since}", "--pretty=tformat:", "--numstat"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=8,
        )
        ins = dels = 0
        if r2.returncode == 0:
            for ln in (r2.stdout or "").splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                parts = ln.split("\t")
                if len(parts) < 3:
                    continue
                a, b, _ = parts[0], parts[1], parts[2]
                if a.isdigit():
                    ins += int(a)
                if b.isdigit():
                    dels += int(b)

        net = ins - dels
        # Simple "effort multiplier" heuristic: net_kLOC + commits/10
        mult = round((net / 1000.0) + (commits / 10.0), 2)
        return {
            "since": since,
            "commits": int(commits),
            "insertions": int(ins),
            "deletions": int(dels),
            "net": int(net),
            "effort_multiplier": float(mult),
        }
    except Exception:
        return {"since": "", "commits": 0, "insertions": 0, "deletions": 0, "net": 0, "effort_multiplier": 0.0}


def _schema_with_todo(repo: str | None, *, include_answered: bool) -> dict:
    sch = _load_schema(repo)
    if not isinstance(sch, dict):
        sch = {}

    cats = sch.get("categories") if isinstance(sch.get("categories"), list) else []

    answered = _answered_keys(repo)

    # Decisions progress (based on the static schema only; excludes injected TODO/meta).
    base_total = 0
    base_cat_ids: set[str] = set()
    base_keys: set[str] = set()
    try:
        for c in cats:
            if not isinstance(c, dict):
                continue
            cid = str(c.get("id") or "").strip()
            if not cid:
                continue
            base_cat_ids.add(cid)
            items = c.get("items") if isinstance(c.get("items"), list) else []
            for it in items:
                if not isinstance(it, dict):
                    continue
                k = str(it.get("key") or "").strip()
                if not k:
                    continue
                base_total += 1
                base_keys.add(f"{cid}.{k}")
    except Exception:
        base_total = 0
        base_cat_ids = set()
        base_keys = set()

    base_answered = len([k for k in answered if k in base_keys]) if base_keys else 0
    base_pct = (float(base_answered) / float(base_total) * 100.0) if base_total else 0.0

    # Filter out any already-answered questions unless explicitly requested.
    if not include_answered:
        new_cats = []
        for c in cats:
            if not isinstance(c, dict):
                continue
            cid = str(c.get("id") or "").strip()
            items = c.get("items") if isinstance(c.get("items"), list) else []
            filt = []
            for it in items:
                if not isinstance(it, dict):
                    continue
                k = str(it.get("key") or "").strip()
                if not k:
                    continue
                if f"{cid}.{k}" in answered:
                    continue
                filt.append(it)
            c2 = dict(c)
            c2["items"] = filt
            # Drop empty categories so the UI doesn't show 'notes-only' tabs.
            if filt:
                new_cats.append(c2)
        cats = new_cats

    # Inject a dynamic TODO decisions category.
    # This keeps the Decisions Hub useful even when all static batches are answered.
    stats = _todo_stats(repo)
    open_items = _todo_open_items(repo, limit=12, answered=answered)
    todo_items = []
    for it in open_items:
        i = int(it.get("i") or 0)
        title = str(it.get("title") or "").strip()
        todo_items.append(
            {
                "key": f"todo_{i}",
                "q": f"TODO #{i}: {title}",
                "opts": {"A": "Do now", "B": "Queue", "C": "Skip"},
                "default": "B",
            }
        )

    if todo_items:
        cats = list(cats) + [
            {
                "id": "todo",
                "name": "TODO — next up",
                "notesKey": "todo_notes",
                "items": todo_items,
            }
        ]

    # If everything is answered and TODO injection had nothing new, keep the UI usable.
    if not cats:
        cats = [
            {
                "id": "meta",
                "name": "No unanswered questions",
                "notesKey": "meta_notes",
                "items": [
                    {
                        "key": "no_unanswered",
                        "q": "Nothing new to decide right now. Next step: toggle 'Show answered' or bump schema to a new batch.",
                        "opts": {"A": "OK"},
                        "default": "A",
                    }
                ],
            }
        ]

    sch["categories"] = list(cats)
    sch.setdefault("meta", {})
    if isinstance(sch.get("meta"), dict):
        sch["meta"]["answered_keys"] = len(list(answered))
        sch["meta"]["include_answered"] = bool(include_answered)
        sch["meta"]["todo"] = stats

        sch["meta"]["decisions_total"] = int(base_total)
        sch["meta"]["decisions_answered"] = int(base_answered)
        sch["meta"]["decisions_pct"] = round(float(base_pct), 1)
        sch["meta"]["decisions_remaining"] = int(max(0, base_total - base_answered))

        # Add file freshness to help explain 'stuck %' situations.
        try:
            p = _todo_status_path(repo)
            sch["meta"]["todo_status_mtime"] = p.stat().st_mtime if p.exists() else None
        except Exception:
            sch["meta"]["todo_status_mtime"] = None

    return sch


def _decisions_path(repo: str | None) -> Path:
    return _data_dir(repo) / "decisions.json"


def _log_path(repo: str | None) -> Path:
    return _data_dir(repo) / "decisions_log.jsonl"


def _answered_keys(repo: str | None) -> set[str]:
    """Return a set of 'category.key' strings ever submitted."""
    out: set[str] = set()

    def _ingest_obj(obj: dict):
        cats = obj.get("categories") if isinstance(obj.get("categories"), dict) else {}
        for cat, payload in cats.items():
            if not isinstance(payload, dict):
                continue
            for k in payload.keys():
                if k.endswith("_notes") or k == "notes":
                    continue
                out.add(f"{cat}.{k}")

    # latest
    try:
        p = _decisions_path(repo)
        if p.exists():
            obj = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                _ingest_obj(obj)
    except Exception:
        pass

    # full log
    try:
        lp = _log_path(repo)
        if lp.exists():
            for ln in lp.read_text(encoding="utf-8", errors="ignore").splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                    if isinstance(obj, dict):
                        _ingest_obj(obj)
                except Exception:
                    continue
    except Exception:
        pass

    return out


def _write_latest(obj: dict, *, repo: str | None) -> None:
    d = _data_dir(repo)
    d.mkdir(parents=True, exist_ok=True)
    _decisions_path(repo).write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _append_log(obj: dict, *, repo: str | None) -> None:
    d = _data_dir(repo)
    d.mkdir(parents=True, exist_ok=True)
    with _log_path(repo).open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


def _take_last_lines(path: Path, max_lines: int = 2000) -> list[str]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return lines[-max_lines:]
    except Exception:
        return []


def _create_debug_bundle(repo: str | None) -> bytes:
    """Return a sanitized debug zip (as bytes).

    Contains tails of local logs and a few non-secret snapshots. Intended for
    'what failed' troubleshooting.
    """
    data_dir = _data_dir(repo)
    logs = data_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    session = logs / "spade_session.jsonl"
    errors = logs / "spade_errors.jsonl"

    out = Path("debug_bundle_tmp.zip")

    # We build in-memory (BytesIO would be ideal, but keep it simple/robust here).
    # Use a temp file then read bytes.
    try:
        import io

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("spade_session.tail.jsonl", "\n".join(_take_last_lines(session)) + "\n")
            z.writestr("spade_errors.tail.jsonl", "\n".join(_take_last_lines(errors)) + "\n")

            # decisions (useful for reproducing schema filtering issues)
            try:
                p = _decisions_path(repo)
                if p.exists():
                    z.writestr("decisions.json", p.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                pass
            try:
                lp = _log_path(repo)
                if lp.exists():
                    z.writestr("decisions_log.tail.jsonl", "\n".join(_take_last_lines(lp)) + "\n")
            except Exception:
                pass

            # schema snapshot
            try:
                sch = _load_schema(repo)
                z.writestr("decisions_inbox_schema.json", json.dumps(sch, indent=2, ensure_ascii=False, default=str))
            except Exception:
                pass

            # todo snapshot
            try:
                tp = _todo_status_path(repo)
                if tp.exists():
                    z.writestr("TODO_STATUS.md", "\n".join(_take_last_lines(tp, max_lines=1200)) + "\n")
            except Exception:
                pass

            # simple runtime snapshot (no secrets)
            snap = {
                "repo": str(_root(repo)),
                "data_dir": str(data_dir),
                "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "tokens_present": (data_dir / "schwab_tokens.json").exists(),
                "secrets_present": (data_dir / "schwab_secrets.local.json").exists(),
            }
            z.writestr("snapshot.json", json.dumps(snap, indent=2))

        return buf.getvalue()
    except Exception:
        return b""


def _decorate_latest_for_client(latest: dict) -> dict:
    """Add per-category _received_at stamps for UI convenience.

    The UI computes 'unread' state per-tab, but the saved decisions payload has
    a single top-level _received_at. We copy it onto each category dict when
    serving /status so tabs can compare stamps.
    """
    if not isinstance(latest, dict):
        return {}
    ra = latest.get("_received_at")
    cats = latest.get("categories") if isinstance(latest.get("categories"), dict) else {}
    out = dict(latest)
    out_cats: dict = {}
    for cid, payload in cats.items():
        if isinstance(payload, dict):
            p2 = dict(payload)
            if ra is not None and "_received_at" not in p2:
                p2["_received_at"] = ra
            out_cats[cid] = p2
        else:
            out_cats[cid] = payload
    out["categories"] = out_cats
    return out


class Handler(BaseHTTPRequestHandler):
    repo: str | None = None

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS, GET")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_GET(self):  # noqa: N802
        if self.path in ("/status", "/status/"):
            try:
                latest = {}
                p = _decisions_path(self.repo)
                if p.exists():
                    latest = json.loads(p.read_text(encoding="utf-8"))
                    if not isinstance(latest, dict):
                        latest = {}

                latest = _decorate_latest_for_client(latest)

                todo = _todo_stats(self.repo)
                vel = _git_velocity(self.repo)

                self.send_response(200)
                self._cors()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps(
                        {"ok": True, "latest": latest, "todo": todo, "velocity": vel},
                        ensure_ascii=False,
                        default=str,
                    ).encode("utf-8")
                )
            except Exception as e:
                self.send_response(500)
                self._cors()
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode("utf-8"))
            return

        if self.path in ("/todo", "/todo/"):
            try:
                obj = _todo_md(self.repo)
                self.send_response(200)
                self._cors()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": True, **obj}, ensure_ascii=False, default=str).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self._cors()
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode("utf-8"))
            return

        if self.path in ("/debug_bundle", "/debug_bundle/"):
            try:
                blob = _create_debug_bundle(self.repo)
                if not blob:
                    raise RuntimeError("failed to create debug bundle")
                stamp = time.strftime("%Y%m%d_%H%M%S")
                fn = f"debug_bundle_{stamp}.zip"
                self.send_response(200)
                self._cors()
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Disposition", f"attachment; filename=\"{fn}\"")
                self.end_headers()
                self.wfile.write(blob)
            except Exception as e:
                self.send_response(500)
                self._cors()
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode("utf-8"))
            return

        # Note: supports query param ?includeAnswered=1
        if self.path.startswith("/schema"):
            try:
                q = parse_qs(urlparse(self.path).query)
                include_answered = str((q.get("includeAnswered") or [""])[0]).strip() in ("1", "true", "yes")
                obj = _schema_with_todo(self.repo, include_answered=include_answered)
                self.send_response(200)
                self._cors()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": True, "schema": obj}, ensure_ascii=False, default=str).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self._cors()
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode("utf-8"))
            return

        self.send_response(404)
        self._cors()
        self.end_headers()
        self.wfile.write(b"not found")

    def do_OPTIONS(self):  # noqa: N802
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_POST(self):  # noqa: N802
        if self.path not in ("/submit", "/submit/"):
            self.send_response(404)
            self._cors()
            self.end_headers()
            self.wfile.write(b"not found")
            return

        n = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(n) if n > 0 else b"{}"
        try:
            obj = json.loads(raw.decode("utf-8"))
            if not isinstance(obj, dict):
                raise ValueError("payload must be a JSON object")
        except Exception as e:
            self.send_response(400)
            self._cors()
            self.end_headers()
            self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode("utf-8"))
            return

        obj.setdefault("_received_at", time.strftime("%Y-%m-%dT%H:%M:%S"))

        # Server-side validation: ensure every question in the *served schema mode* is accounted for.
        # If the client hid already-answered questions, we validate against include_answered=False.
        try:
            inc = bool(obj.get("schema_include_answered"))
            sch = _schema_with_todo(self.repo, include_answered=inc)
            sch_cats = sch.get("categories") if isinstance(sch.get("categories"), list) else []
            cats = obj.get("categories") if isinstance(obj.get("categories"), dict) else {}
            missing: list[str] = []
            for c in sch_cats:
                if not isinstance(c, dict):
                    continue
                cid = str(c.get("id") or "").strip()
                items = c.get("items") if isinstance(c.get("items"), list) else []
                got = cats.get(cid)
                if not isinstance(got, dict):
                    got = {}
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    k = str(it.get("key") or "").strip()
                    if not k:
                        continue
                    if k not in got:
                        missing.append(f"{cid}.{k}")
            if missing:
                self.send_response(400)
                self._cors()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": "missing schema keys", "missing": missing[:200]}, ensure_ascii=False).encode("utf-8"))
                return
        except Exception:
            # best-effort only
            pass

        try:
            _write_latest(obj, repo=self.repo)
            _append_log(obj, repo=self.repo)
        except Exception as e:
            self.send_response(500)
            self._cors()
            self.end_headers()
            self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode("utf-8"))
            return

        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(
            json.dumps(
                {
                    "ok": True,
                    "saved": str(_decisions_path(self.repo)),
                    "logged": str(_log_path(self.repo)),
                }
            ).encode("utf-8")
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--repo", type=str, default="", help="Path to repo root (where data/ lives)")
    args = ap.parse_args()

    Handler.repo = (args.repo or "").strip() or None

    srv = HTTPServer(("127.0.0.1", int(args.port)), Handler)
    print(f"Decisions Listener running on http://127.0.0.1:{args.port}/submit")
    print(f"Repo: {_root(Handler.repo)}")
    print(f"Writes: {_decisions_path(Handler.repo)}")
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
