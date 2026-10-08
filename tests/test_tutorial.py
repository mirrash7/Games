"""How-to-play cards: the Python port of the browser's tutorial demos."""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pytest

from kpapp import figure
from kpapp.game.flappy import tutorial as flappy_tut
from kpapp.game.flappy.game import FlappyGame
from kpapp.game.fruitninja import tutorial as snack_tut
from kpapp.game.fruitninja.game import FruitNinjaGame

WEB = Path(__file__).resolve().parents[1] / "web" / "js" / "games"
SIZE = (720, 1280)
# The demo areas of the shell's three cards (342 x 200 each).
RECTS = [(69 + i * 400, 134, 411 + i * 400, 334) for i in range(3)]
TIMES = [0.0, 0.3, 0.71, 0.95, 1.2, 1.6, 2.05, 2.9, 3.15, 4.4, 7.77]

GAMES = [
    (FruitNinjaGame, "snack", ["STAND BACK", "YOUR HAND IS THE RACCOON", "AVOID THE TRASH"]),
    (FlappyGame, "flappy", ["STAND BACK", "FLAP TO FLY", "FLY THROUGH THE GAPS"]),
]


def _noise_frame(seed: int = 0) -> np.ndarray:
    # Noise, so any pixel a demo writes is very likely to change.
    return np.random.default_rng(seed).integers(0, 256, (*SIZE, 3), dtype=np.uint8)


def _js_strings(game: str, key: str) -> list[str]:
    src = (WEB / game / "tutorial.js").read_text()
    return re.findall(rf'{key}: "([^"]+)"', src)


@pytest.mark.parametrize("cls,web_name,titles", GAMES)
def test_steps_match_the_browser(cls, web_name, titles):
    steps = cls.tutorial
    assert len(steps) == 3
    assert [s.title for s in steps] == titles
    assert all(isinstance(s, figure.TutorialStep) for s in steps)
    # Same words as the browser cards, so the two arcades teach the same thing.
    assert [s.title for s in steps] == _js_strings(web_name, "title")
    assert [s.text for s in steps] == _js_strings(web_name, "text")


@pytest.mark.parametrize("cls,web_name", [(c, w) for c, w, _ in GAMES])
def test_tip_matches_the_browser(cls, web_name):
    src = (WEB / web_name / "game.js").read_text()
    assert cls.tip == re.search(r'static tip = "([^"]+)"', src).group(1)


@pytest.mark.parametrize("cls", [c for c, _, _ in GAMES])
def test_demos_draw_only_inside_their_rect(cls):
    for i, (step, rect) in enumerate(zip(cls.tutorial, RECTS)):
        x0, y0, x1, y1 = rect
        for t in TIMES:
            frame = _noise_frame(i)
            before = frame.copy()
            step.draw(frame, rect, t)
            changed = np.any(frame != before, axis=2)
            inside = np.zeros(SIZE, bool)
            inside[y0:y1, x0:x1] = True
            assert not changed[~inside].any(), f"{step.title} at t={t} drew outside its rect"
            # Not a no-op: a real demo covers a good part of its card.
            assert changed[inside].mean() > 0.03, f"{step.title} at t={t} drew almost nothing"


@pytest.mark.parametrize("cls", [c for c, _, _ in GAMES])
def test_demos_survive_a_rect_off_the_frame(cls):
    frame = _noise_frame()
    before = frame.copy()
    for step in cls.tutorial:
        step.draw(frame, (1100, 600, 1442, 800), 1.3)  # hangs off the bottom-right
        step.draw(frame, (-200, -100, 142, 100), 1.3)  # and the top-left
    changed = np.any(frame != before, axis=2)
    assert changed[600:, 1100:].any() and changed[:100, :142].any()
    assert not changed[100:600].any()


@pytest.mark.parametrize("module", [snack_tut, flappy_tut])
def test_demos_draw_the_figure_without_art(module, monkeypatch):
    # Missing generated art must not break the card: the figure still teaches.
    monkeypatch.setattr(module, "_art", None)
    monkeypatch.setattr(module, "_art_missing", True)
    steps = module.SNACK_TUTORIAL if module is snack_tut else module.FLAPPY_TUTORIAL
    for step, rect in zip(steps, RECTS):
        frame = _noise_frame()
        before = frame.copy()
        step.draw(frame, rect, 1.0)
        assert np.any(frame != before)


def test_reach_puts_the_wrist_on_target():
    for target in [(0.5, -1.0), (0.9, -0.2), (0.6, 0.0)]:
        a1, a2 = figure.reach(target)
        sh = (0.3, -0.5)
        wrist = (sh[0] + np.cos(a1) * figure.UPPER + np.cos(a2) * figure.FORE,
                 sh[1] + np.sin(a1) * figure.UPPER + np.sin(a2) * figure.FORE)
        assert np.allclose(wrist, target, atol=1e-6)
        # The elbow is the lower of the two solutions.
        assert sh[1] + np.sin(a1) * figure.UPPER >= min(target[1], sh[1]) - 1e-9


@pytest.mark.parametrize("cls", [c for c, _, _ in GAMES])
def test_three_demos_cost_under_4ms_per_frame(cls):
    frame = _noise_frame()
    times = [k / 60 for k in range(0, 60 * 7)]  # 7 s covers every loop's period
    for t in times:  # warm-up: art load, pre-scaled and rotated sprites
        for step, rect in zip(cls.tutorial, RECTS):
            step.draw(frame, rect, t)
    start = time.perf_counter()
    for t in times:
        for step, rect in zip(cls.tutorial, RECTS):
            step.draw(frame, rect, t)
    per_frame_ms = (time.perf_counter() - start) * 1000 / len(times)
    assert per_frame_ms < 4.0, f"{cls.__name__} tutorial: {per_frame_ms:.2f} ms per frame"
