"""Tiny local HTTP callback helper for OAuth desktop flows."""

from __future__ import annotations

import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable


class LocalOAuthSession:
    """Bind a localhost callback, then wait for the browser redirect."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        path: str = "/callback/",
    ) -> None:
        if not path.startswith("/"):
            path = "/" + path
        if not path.endswith("/"):
            path = path + "/"
        self.host = host
        self.port = port
        self.path = path
        self.result: dict[str, Any] = {"params": None, "error": None}
        self._done = threading.Event()
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.redirect_uri = ""

    def start(self) -> str:
        result = self.result
        done = self._done
        path = self.path

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
                return

            def do_GET(self) -> None:  # noqa: N802
                parsed = urllib.parse.urlparse(self.path)
                if parsed.path.rstrip("/") != path.rstrip("/"):
                    self.send_response(404)
                    self.end_headers()
                    self.wfile.write(b"Not found")
                    return
                params = urllib.parse.parse_qs(parsed.query)
                flat = {k: (v[0] if isinstance(v, list) and v else "") for k, v in params.items()}
                result["params"] = flat
                body = (
                    b"<html><body style='font-family:sans-serif;padding:2rem'>"
                    b"<h2>Connected</h2>"
                    b"<p>You can close this window and return to the Connections page.</p>"
                    b"</body></html>"
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                done.set()

        self._server = HTTPServer((self.host, self.port), Handler)
        actual_port = int(self._server.server_address[1])
        self.redirect_uri = f"http://{self.host}:{actual_port}{self.path}"

        def _serve() -> None:
            assert self._server is not None
            while not done.is_set():
                self._server.handle_request()

        self._thread = threading.Thread(target=_serve, daemon=True)
        self._thread.start()
        return self.redirect_uri

    def wait(self, timeout_sec: float = 300.0) -> dict[str, str]:
        if not self._done.wait(timeout_sec):
            self.close()
            raise TimeoutError("Timed out waiting for browser authorization")
        self.close()
        params = self.result.get("params") or {}
        if params.get("error"):
            desc = params.get("error_description") or params.get("error")
            raise RuntimeError(f"OAuth denied: {desc}")
        return params

    def close(self) -> None:
        if self._server is not None:
            try:
                self._server.server_close()
            except Exception:  # noqa: BLE001
                pass
            self._server = None


def run_local_oauth_callback(
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    path: str = "/callback/",
    timeout_sec: float = 300.0,
    before_wait: Callable[[str], None] | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Start a one-shot localhost server, optionally run before_wait(redirect_uri),
    then wait for the OAuth redirect.
    """
    session = LocalOAuthSession(host=host, port=port, path=path)
    redirect_uri = session.start()
    if before_wait is not None:
        before_wait(redirect_uri)
    params = session.wait(timeout_sec=timeout_sec)
    return redirect_uri, params
