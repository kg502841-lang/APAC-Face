"""Serve the Faceify web app with cross-origin isolation, so ONNX Runtime Web can use several threads.

Browsers only allow WebAssembly threads (SharedArrayBuffer) on cross-origin-isolated pages, which need the
COOP and COEP response headers that `python -m http.server` does not send. COEP "credentialless" keeps the CDN
scripts and Google Fonts loading without changes to the page.

It also stores the landmark labels made in the labelling tool (web/label.html):

    GET  /api/labels?annotator=NAME   that annotator's saved labels, plus which faces anyone has labelled
    POST /api/labels                  save one label (JSON) to data/apac_labels/<annotator>/<id>.json

    python -m src.serve_web                 # http://127.0.0.1:8080  (labelling tool: /label.html)
    python -m src.serve_web --port 8090
"""
from __future__ import annotations

import argparse
import functools
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .utils import load_config, resolve_path

ISOLATION_HEADERS = {
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "credentialless",
}
MAX_LABEL_BYTES = 64_000


class IsolatedHandler(SimpleHTTPRequestHandler):
    labels_root: Path = resolve_path("data/apac_labels")
    manifest_path: Path = resolve_path("web/labeling/manifest.json")

    def end_headers(self) -> None:
        for k, v in ISOLATION_HEADERS.items():
            self.send_header(k, v)
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _manifest_ids(self) -> set[str]:
        if not self.manifest_path.is_file():
            return set()
        return {it["id"] for it in json.loads(self.manifest_path.read_text(encoding="utf-8"))["items"]}

    def do_GET(self) -> None:
        url = urlparse(self.path)
        if url.path != "/api/labels":
            return super().do_GET()
        from .apac_labels import ANNOTATOR_RE, load_labels
        annotator = parse_qs(url.query).get("annotator", [""])[0]
        if annotator and not ANNOTATOR_RE.match(annotator):
            return self._json(400, {"error": "bad annotator name"})
        labels = load_labels(self.labels_root)
        mine = {i: recs[annotator] for i, recs in labels.items() if annotator in recs}
        anyone = {i: sorted(a for a, r in recs.items() if not r["skipped"]) for i, recs in labels.items()}
        return self._json(200, {"labels": mine, "labelled_by": anyone})

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/labels":
            return self._json(404, {"error": "not found"})
        from .apac_labels import save_label, validate_label
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_LABEL_BYTES:
            return self._json(413, {"error": "label too large or empty"})
        try:
            record = validate_label(json.loads(self.rfile.read(length)), self._manifest_ids())
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            return self._json(400, {"error": str(e)})
        path = save_label(self.labels_root, record)
        return self._json(200, {"saved": path.relative_to(self.labels_root.parent.parent).as_posix(),
                                "record": record})


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Faceify web app (cross-origin isolated)")
    parser.add_argument("--config", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    web = resolve_path(load_config(args.config)["paths"]["web"])
    IsolatedHandler.manifest_path = web / "labeling" / "manifest.json"
    server = ThreadingHTTPServer((args.host, args.port), functools.partial(IsolatedHandler, directory=str(web)))
    print(f"Serving {web} at http://{args.host}:{args.port} (cross-origin isolated). "
          f"Labelling tool: http://{args.host}:{args.port}/label.html. Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
