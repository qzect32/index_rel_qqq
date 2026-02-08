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


def _root(repo: str | None) -> Path:
    if repo:
        return Path(repo).resolve()
    return Path(__file__).resolve().parents[1]


def _data_dir(repo: str | None) -> Path:
    return (_root(repo) / "data").resolve()


def _decisions_path(repo: str | None) -> Path:
    return _data_dir(repo) / "decisions.json"


def _log_path(repo: str | None) -> Path:
    return _data_dir(repo) / "decisions_log.jsonl"


def _write_latest(obj: dict, *, repo: str | None) -> None:
    d = _data_dir(repo)
    d.mkdir(parents=True, exist_ok=True)
    _decisions_path(repo).write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _append_log(obj: dict, *, repo: str | None) -> None:
    d = _data_dir(repo)
    d.mkdir(parents=True, exist_ok=True)
    with _log_path(repo).open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


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
                self.send_response(200)
                self._cors()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": True, "latest": latest}, ensure_ascii=False, default=str).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self._cors()
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode("utf-8"))
            return

        if self.path in ("/schema", "/schema/"):
            try:
                repo_root = _root(self.repo)
                obj = {}
                # Prefer git-tracked config schema; fall back to local data schema
                schema_path = (repo_root / "config" / "decisions_inbox_schema.json").resolve()
                if not schema_path.exists():
                    schema_path = (repo_root / "data" / "decisions_inbox_schema.json").resolve()

                if schema_path.exists():
                    obj = json.loads(schema_path.read_text(encoding="utf-8"))
                    if not isinstance(obj, dict):
                        obj = {}
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
