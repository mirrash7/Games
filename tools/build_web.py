"""Assemble the static site for GitHub Pages into build/site/.

The site is web/ plus the game art, which lives once in assets/<game>/generated/
and is mounted into the site rather than copied in the repo, so the desktop
and browser builds can never drift apart. tools/serve_web.py serves the same
layout live for development.

    uv run python tools/build_web.py            # -> build/site/
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"

# Site path -> repo directory.
MOUNTS = {
    "assets/snack": REPO / "assets" / "fruitninja" / "generated",
    "assets/flappy": REPO / "assets" / "flappy" / "generated",
}

# Development-only files that should not be published.
EXCLUDE = ("tests", "bench.html", "dev", "*.onnx")


def build(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(WEB, out, ignore=shutil.ignore_patterns(*EXCLUDE))
    for site_path, src in MOUNTS.items():
        if not src.is_dir():
            raise SystemExit(f"missing {src}; run the asset generators in tools/ first")
        shutil.copytree(src, out / site_path)
    parts = sorted((out / "models").glob("*.part*"))
    if not parts:
        raise SystemExit("no model chunks in web/models; run tools/export_web_model.py first")
    (out / ".nojekyll").write_text("")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"built {out} ({size / 1e6:.1f} MB, {len(parts)} model chunks)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "build" / "site")
    build(ap.parse_args().out)


if __name__ == "__main__":
    main()
