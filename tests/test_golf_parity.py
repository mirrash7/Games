"""The browser port's golf parity fixture must match the Python it was made from.

web/tests/golf_parity.test.mjs replays web/tests/fixtures/golf_parity.json
(swing streams, physics shots, course lookups, scripted holes) against the
JavaScript port. If swing.py, physics.py, course.py, the course or the game's
rules change, the fixture goes stale and the browser would be checked against
old answers: this test catches that. Regenerate with

    uv run python tools/make_golf_parity_fixture.py

and then run the browser tests (pytest runs them too, in test_web.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import make_golf_parity_fixture as fixture  # noqa: E402


def test_golf_parity_fixture_is_up_to_date():
    committed = json.loads(fixture.OUT.read_text())
    fresh = json.loads(json.dumps(fixture.build(), separators=(",", ":")))
    for section in ("swing", "physics", "course", "game"):
        assert fresh[section] == committed[section], (
            f"web/tests/fixtures/golf_parity.json is stale ({section}): "
            "run uv run python tools/make_golf_parity_fixture.py")
