"""The reference game must keep working - it is what new games are copied from."""

from __future__ import annotations

import numpy as np

from kpapp.config import KP
from kpapp.controls import ControlState
from kpapp.game import REGISTRY
from kpapp.game.template.game import TouchTargetsGame
from kpapp.inference import Pose
from kpapp.shell import Screen, Shell

W, H, DT = 1280, 720, 1 / 60


def palm_at(p) -> ControlState:
    """The player's real right hand (left_wrist on a mirrored feed) at p."""
    xy = np.zeros((17, 2), np.float32)
    c = np.zeros(17, np.float32)
    xy[KP["left_wrist"]] = p
    c[KP["left_wrist"]] = 0.9
    return ControlState(present=True, pose=Pose(xy=xy, confidence=c, score=0.9))


def test_registered_but_kept_off_the_menu():
    assert REGISTRY["template"] is TouchTargetsGame
    assert "template" not in Shell((W, H), REGISTRY).menu_games


def test_plays_through_the_real_shell():
    # Hidden from the menu, so reach it the way `kp game --game template` does.
    shell = Shell((W, H), {"template": TouchTargetsGame}, selected="template")
    shell.handle_key(ord("1"))
    for _ in range(200):  # countdown
        shell.update(ControlState(present=False), DT)
    assert shell.screen is Screen.PLAYING
    game = shell.game
    for _ in range(300):
        shell.update(palm_at(game.target.pos.copy()), DT)
        shell.render(np.zeros((H, W, 3), np.uint8))
    assert game.score >= 5, "holding the palm on targets should score"
    for _ in range(int(31 / DT)):
        shell.update(ControlState(present=False), DT)
    assert shell.game_over, "the shell must see the round end"


def test_a_passing_hand_does_not_score():
    game = TouchTargetsGame((W, H), seed=1)
    start = game.target.pos.copy()
    for k in range(10):  # sweep across the target without stopping
        game.update(palm_at(start + np.float32([-200 + 40 * k, 0])), DT)
    assert game.score == 0


def test_seed_reproduces_target_positions():
    a, b = TouchTargetsGame((W, H), seed=3), TouchTargetsGame((W, H), seed=3)
    assert np.allclose(a.target.pos, b.target.pos)
