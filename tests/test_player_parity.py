"""The browser port's PlayerLock fixture must match the Python it was made from.

web/tests/player.test.mjs replays web/tests/fixtures/player_parity.json
(crowds from tests/crowd_poses.py) against web/js/core/player.js. If
src/kpapp/player.py changes, the fixture goes stale and the browser would be
checked against old answers: this test catches that. Regenerate with

    uv run python tools/make_player_parity_fixture.py

and then run the browser tests (pytest runs them too, in test_web.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import make_player_parity_fixture as fixture  # noqa: E402


def test_player_parity_fixture_is_up_to_date():
    committed = json.loads(fixture.OUT.read_text())
    fresh = json.loads(json.dumps(fixture.build(), separators=(",", ":")))
    assert [s["name"] for s in fresh["scenarios"]] == [s["name"] for s in committed["scenarios"]]
    for new, old in zip(fresh["scenarios"], committed["scenarios"]):
        assert new == old, (f"web/tests/fixtures/player_parity.json is stale ({new['name']}): "
                            "run uv run python tools/make_player_parity_fixture.py")
