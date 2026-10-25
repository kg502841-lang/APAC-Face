"""Serve the Faceify web app with cross-origin isolation, so ONNX Runtime Web can use several threads.

Browsers only allow WebAssembly threads (SharedArrayBuffer) on cross-origin-isolated pages, which need the
COOP and COEP response headers that `python -m http.server` does not send. COEP "credentialless" keeps the CDN
scripts and Google Fonts loading without changes to the page.

    python -m src.serve_web                 # http://127.0.0.1:8080
    python -m src.serve_web --port 8090
"""
from __future__ import annotations

import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from .utils import load_config, resolve_path

ISOLATION_HEADERS = {
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "credentialless",
}


class IsolatedHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        for k, v in ISOLATION_HEADERS.items():
            self.send_header(k, v)
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Faceify web app (cross-origin isolated)")
    parser.add_argument("--config", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    web = resolve_path(load_config(args.config)["paths"]["web"])
    server = ThreadingHTTPServer((args.host, args.port), functools.partial(IsolatedHandler, directory=str(web)))
    print(f"Serving {web} at http://{args.host}:{args.port} (cross-origin isolated). Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
