from __future__ import annotations

"""Local Schwab OAuth callback catcher (no extra deps).

Runs a tiny HTTP server on https://127.0.0.1:8000 (actually HTTP, not HTTPS).

Why:
- Schwab redirect URIs are often registered on port 8000 with specific paths.
- Streamlit typically runs on 8501 and isn't a general-purpose callback router.

This server captures the OAuth `code` and writes it to:
  data/schwab_last_code.txt

Then in the Streamlit Admin → Schwab OAuth tab you can paste the code, or read it from file.

NOTE:
- This uses plain HTTP. If you registered an https:// callback, you must either:
  (a) change the registration to http://127.0.0.1:8000/..., or
  (b) put a local TLS terminator in front (not provided here).

Usage:
  python scripts/schwab_callback_server.py --port 8000 --out data/schwab_last_code.txt
"""

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


class Handler(BaseHTTPRequestHandler):
    out_path: Path

    def do_GET(self):
        u = urlparse(self.path)
        qs = parse_qs(u.query)
        code = (qs.get("code") or [None])[0]
        state = (qs.get("state") or [None])[0]
        err = (qs.get("error") or [None])[0]

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

        if err:
            body = f"<h2>OAuth Error</h2><pre>{err}</pre>"
            self.wfile.write(body.encode("utf-8"))
            return

        if not code:
            body = "<h2>Waiting for OAuth code…</h2><p>No <code>code</code> parameter found.</p>"
            self.wfile.write(body.encode("utf-8"))
            return

        # Persist code for easy pickup
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(str(code).strip() + "\n", encoding="utf-8")

        body = (
            "<h2>Schwab OAuth code captured ✅</h2>"
            f"<p>Saved to: <code>{self.out_path}</code></p>"
            "<p>Now return to the Streamlit app → Admin → Schwab OAuth and exchange the code.</p>"
            "<hr/>"
            "<details><summary>Details</summary>"
            f"<pre>path={u.path}\nstate={state}\ncode={code}</pre>"
            "</details>"
        )
        self.wfile.write(body.encode("utf-8"))

    def log_message(self, format, *args):
        # quiet
        return


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", default="data/schwab_last_code.txt")
    args = ap.parse_args()

    Handler.out_path = Path(args.out)

    srv = HTTPServer((args.host, args.port), Handler)
    print(f"Listening on http://{args.host}:{args.port} …")
    print(f"Will write latest code to: {Handler.out_path}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
