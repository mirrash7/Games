"""The browser build's own tests (node:test), run as part of the Python suite.

The browser port re-implements the per-frame code in JavaScript; its tests
live in web/tests and pin the same behaviour as the Python ones. Skipped when
Node.js is not installed.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[1] / "web"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_web_suite_passes():
    files = sorted(str(p.relative_to(WEB)) for p in (WEB / "tests").glob("*.test.mjs"))
    assert files, "no browser tests found"
    run = subprocess.run(["node", "--test", *files], cwd=WEB, capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stdout[-4000:] + run.stderr[-2000:]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_leaderboard_server_rules_pass():
    root = WEB.parent / "leaderboard"
    files = sorted(str(p.relative_to(root)) for p in (root / "test").glob("*.test.mjs"))
    run = subprocess.run(["node", "--test", *files], cwd=root, capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stdout[-4000:] + run.stderr[-2000:]


def test_site_build_layout(tmp_path):
    """The Pages build mounts the game art and ships the model chunks."""
    import sys

    sys.path.insert(0, str(WEB.parent / "tools"))
    from build_web import build

    if not any((WEB / "models").glob("*.part*")):
        pytest.skip("model chunks not exported")
    out = tmp_path / "site"
    build(out)
    assert (out / "index.html").exists() and (out / "coi-sw.js").exists()
    assert (out / "assets" / "snack" / "manifest.json").exists()
    assert (out / "assets" / "flappy" / "bird_0.png").exists()
    assert (out / "models" / "manifest_fp16.json").exists()
    assert not (out / "tests").exists() and not (out / "dev").exists()
    assert not list(out.rglob("*.onnx")), "unsplit models must not be published"
