"""The "how to play" cards shown before a player's first Flappy Raccoon run.

Port of web/js/games/flappy/tutorial.js. The art loads at the first draw;
sprites are pre-scaled once and reused.
"""

from __future__ import annotations

import math

import numpy as np

from ... import theme
from ...gfx import alpha_composite
from ...figure import (DOWN, SpriteCache, TutorialStep, demo, draw_figure, ease, lerp, mirror,
                       stand_back_step, text)
from .art import AssetsMissingError, load_art

ARM_DOWN = (DOWN - 0.3, DOWN - 0.15)
ARM_UP = (-0.45, -1.15)
SKY = (232, 197, 112)  # rgb(112,197,232)

_sprites = SpriteCache()
_art = None  # tests may set this to stub art; None loads the real art at first draw
_art_missing = False


def _get_art():
    global _art, _art_missing
    if _art is None and not _art_missing:
        try:
            _art = load_art()
        except AssetsMissingError:
            _art_missing = True
    return _art


def _wingbeat(phase: float) -> float:
    """Arms over one 1.1 s wing-beat: rise slowly, beat down fast, rest. 0 (down) .. 1 (up)."""
    if phase < 0.55:
        return ease(phase / 0.55)
    if phase < 0.72:
        return 1 - ease((phase - 0.55) / 0.17)
    return 0.0


def _arm_at(up: float):
    return (lerp(ARM_DOWN[0], ARM_UP[0], up), lerp(ARM_DOWN[1], ARM_UP[1], up))


def _bird(art, phase: float):
    frames = getattr(art, "bird_frames", None) if art is not None else None
    if not frames:
        return None
    return frames[int(math.floor(phase * 12)) % len(frames)]


def _draw_flap(img, rect, t: float) -> None:
    art = _get_art()
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    period = 1.1
    phase = (t % period) / period
    up = _wingbeat(phase)
    s = h * 0.33
    right = _arm_at(up)
    draw_figure(img, x0 + w * 0.33, y0 + h * 0.62, s, right=right,
                left=(mirror(right[0]), mirror(right[1])))

    # One hop per downstroke: the flap fires part-way down (phase ~0.62).
    tau = ((phase - 0.62 + 1) % 1) * period
    hop = max(0.0, 230 * tau - 0.5 * 420 * tau * tau)
    bird = _bird(art, phase)
    if bird is not None:
        bw = h * 0.34
        _sprites.blit(img, bird, x0 + w * 0.76, y0 + h * 0.72 - hop * (h / 230),
                      bw, bw * bird.shape[0] / bird.shape[1], angle=-0.25 if tau < 0.25 else 0.15)
    if tau < 0.3:
        text(img, "FLAP!", x0 + w * 0.33, y0 + h * 0.16, 0.7, theme.CYAN, alpha=1 - tau / 0.3)


def _pipe_body(art, pw: int, height: int) -> np.ndarray:
    """The body sprite at pipe width, `height` rows tall (it is uniform vertically)."""
    body = _sprites.get(art.pipe_body, pw, art.pipe_body.shape[0] * pw / art.pipe_body.shape[1])
    key = ("pipe_tall", pw, height)
    tall = _tall.get(key)
    if tall is None:
        reps = height // body.shape[0] + 1
        tall = np.ascontiguousarray(np.tile(body, (reps, 1, 1))[:height])
        _tall[key] = tall
    return tall


_tall: dict[tuple, np.ndarray] = {}


def _draw_pipes(img, rect, t: float) -> None:
    art = _get_art()
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    img[:] = SKY
    pw, gap, spacing, speed = 46, h * 0.42, w * 0.55, 90

    def gap_y(i):
        return y0 + h * (0.4 + 0.18 * math.sin(i * 2.1))

    offset = (t * speed) % spacing
    first = math.floor((t * speed) / spacing)
    bird_x = x0 + w * 0.3
    bird_y = y0 + h * 0.5
    have_pipes = art is not None and art.pipe_body is not None and art.pipe_cap is not None
    if have_pipes:
        body = _pipe_body(art, pw, int(math.ceil(h)) + 2)
        cap_w = pw * 1.18
        cap_h = cap_w * art.pipe_cap.shape[0] / art.pipe_cap.shape[1]
        cap = _sprites.get(art.pipe_cap, cap_w, cap_h)
    for k in range(-1, 4):
        i = first + k
        px = x0 + w * 0.65 + k * spacing - offset
        gy = gap_y(i)
        # The bird steers for the gap of the next pipe ahead of it.
        if px + pw / 2 > bird_x - 30 and px - spacing + pw / 2 <= bird_x - 30:
            prev_y = gap_y(i - 1)
            u = 1 - (px - (bird_x - 30)) / spacing
            bird_y = lerp(prev_y, gy, ease(min(1.0, max(0.0, u * 1.6))))
        if have_pipes:
            left = int(round(px - pw / 2))
            top_h = int(round(gy - gap / 2 - y0))
            bot_y = int(round(gy + gap / 2))
            bot_h = int(round(y1 - gy - gap / 2))
            if top_h > 0:
                alpha_composite(img, body[-top_h:], left, int(round(y0)))
            if bot_h > 0:
                alpha_composite(img, body[:bot_h], left, bot_y)
            cx = int(round(px - pw * 0.59))
            alpha_composite(img, cap, cx, int(round(gy - gap / 2 - cap_h)))
            alpha_composite(img, cap, cx, bot_y)
        # +1 as each pipe goes past the bird.
        past = (bird_x - px) / speed
        if 0 < past < 0.7:
            text(img, "+1", bird_x, bird_y - 40 - past * 50, 0.7, theme.ACCENT, alpha=1 - past / 0.7)
    bird = _bird(art, (t % 1.1) / 1.1)
    if bird is not None:
        _sprites.blit(img, bird, bird_x, bird_y, 54, 54 * bird.shape[0] / bird.shape[1])


FLAPPY_TUTORIAL = [
    stand_back_step(
        "STAND BACK",
        "Far enough that both arms stay in the picture, even raised above your head.",
        arms_up=True,
    ),
    TutorialStep(
        "FLAP TO FLY",
        "Raise both arms, then beat them down like wings. Every flap is one hop.",
        demo(_draw_flap),
    ),
    TutorialStep(
        "FLY THROUGH THE GAPS",
        "Each pipe you pass is a point. Hitting a pipe or the ground ends the run.",
        demo(_draw_pipes),
    ),
]
