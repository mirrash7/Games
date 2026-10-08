"""Serve web/ locally with the headers the browser build needs.

Cross-origin isolation (COOP + COEP) unlocks SharedArrayBuffer, which
onnxruntime-web needs to run on more than one CPU thread. GitHub Pages cannot
set these headers, so the site ships a service worker that adds them; this
server sets them directly for local development. Game art is mounted from
assets/ exactly as tools/build_web.py lays it out for Pages.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_web import MOUNTS, WEB  # noqa: E402  (same site layout as the Pages build)

ROOT = WEB


class Handler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path: str) -> str:
        clean = path.split("?", 1)[0].split("#", 1)[0].lstrip("/")
        for site_path, src in MOUNTS.items():
            if clean == site_path or clean.startswith(site_path + "/"):
                rest = clean[len(site_path):].lstrip("/")
                return str(src / rest)
        return super().translate_path(path)

    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".wasm": "application/wasm", ".mjs": "text/javascript",
                      ".onnx": "application/octet-stream"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    handler = functools.partial(Handler, directory=str(ROOT))
    print(f"serving {ROOT} on http://localhost:{args.port}")
    http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
