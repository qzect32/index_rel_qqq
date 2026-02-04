from __future__ import annotations

import datetime
import ipaddress
import ssl
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


@dataclass
class CallbackServerState:
    host: str = "127.0.0.1"
    port: int = 8000
    out_path: Path = Path("data/schwab_last_code.txt")
    last_code: str | None = None
    last_error: str | None = None
    last_path: str | None = None
    running: bool = False


def ensure_localhost_cert(cert_path: Path, key_path: Path) -> tuple[Path, Path]:
    """Create a self-signed cert for 127.0.0.1 + localhost if missing.

    Uses cryptography. Does not modify OS trust store.
    """

    cert_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)

    if cert_path.exists() and key_path.exists():
        return cert_path, key_path

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, u"127.0.0.1")])

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow() - datetime.timedelta(minutes=1))
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                    x509.DNSName("localhost"),
                ]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )

    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))

    return cert_path, key_path


def start_https_callback_server(state: CallbackServerState, cert_path: Path, key_path: Path) -> tuple[threading.Thread, HTTPServer]:
    """Start an HTTPS server in a background thread.

    Captures `code` from query string and writes it to state.out_path.
    """

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            u = urlparse(self.path)
            qs = parse_qs(u.query)
            code = (qs.get("code") or [None])[0]
            err = (qs.get("error") or [None])[0]

            state.last_path = self.path
            if err:
                state.last_error = str(err)

            if code:
                state.last_code = str(code).strip()
                state.out_path.parent.mkdir(parents=True, exist_ok=True)
                state.out_path.write_text(state.last_code + "\n", encoding="utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()

            if err:
                self.wfile.write(f"<h2>OAuth Error</h2><pre>{err}</pre>".encode("utf-8"))
                return

            if not code:
                self.wfile.write(
                    (
                        "<h2>Schwab OAuth callback ready</h2>"
                        "<p>No <code>code</code> param found yet.</p>"
                        "<p>Return to the Schwab consent flow; it should redirect back here.</p>"
                    ).encode("utf-8")
                )
                return

            self.wfile.write(
                (
                    "<h2>Code captured ✅</h2>"
                    f"<p>Saved to <code>{state.out_path}</code></p>"
                    "<p>You can close this tab and return to the app.</p>"
                ).encode("utf-8")
            )

        def log_message(self, format, *args):
            return

    httpd = HTTPServer((state.host, state.port), Handler)

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))
    httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)

    def _run():
        state.running = True
        try:
            httpd.serve_forever()
        except Exception as e:
            state.last_error = str(e)
        finally:
            state.running = False

    t = threading.Thread(target=_run, name="schwab_callback_https", daemon=True)
    t.start()
    return t, httpd


def stop_callback_server(httpd: HTTPServer | None) -> None:
    if not httpd:
        return
    try:
        httpd.shutdown()
    except Exception:
        pass
    try:
        httpd.server_close()
    except Exception:
        pass
