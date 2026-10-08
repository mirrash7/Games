"""The "how to play" cards shown before a player's first Snack Attack round.

Port of web/js/games/snack/tutorial.js. Each card animates a stick figure with
the game's own sprites; the art loads at the first draw, and every sprite is
pre-scaled (and rotated in angle buckets) once, then reused.
"""

from __future__ import annotations

import math

from ... import theme
from ...figure import (DOWN, SpriteCache, TutorialStep, cross, demo, draw_figure, ease,
                       hand_pos, lerp, mirror, reach, stand_back_step, stroke_alpha, text)
from .art import AssetsMissingError, load_art

REST_LEFT = (mirror(DOWN - 0.25), mirror(DOWN - 0.1))
TRAIL = (255, 110, 190)  # rgb(190,110,255)
X3_COLOUR = (245, 230, 235)  # rgb(235,230,245)

_sprites = SpriteCache()
_art = None  # tests may set this to stub art; None loads the real art at first draw
_art_missing = False


def _get_art():
    """The game's art, loaded at the first draw. None if it was never generated."""
    global _art, _art_missing
    if _art is None and not _art_missing:
        try:
            _art = load_art()
        except AssetsMissingError:
            _art_missing = True  # draw the figure alone, like the browser before preload
    return _art


def _swing(u: float):
    """The right arm's swipe: a diagonal slash out to the side, high to low."""
    return reach((lerp(0.42, 1.0, u), lerp(-1.12, -0.02, u)))


def _draw_swipe(img, rect, t: float) -> None:
    art = _get_art()
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    s, cx, cy = h * 0.4, x0 + w * 0.36, y0 + h * 0.6

    def P(p):
        return (cx + p[0] * s, cy + p[1] * s)

    phase = t % 2.2
    u = ease((phase - 0.35) / 0.75)

    # The snack waits at the top of the swing; once the hand has passed it, it
    # breaks in two and falls, with a +1.
    snack_at = P(hand_pos(_swing(0.5)))
    eaten = u > 0.5
    since = phase - (0.35 + 0.75 * 0.5) if eaten else 0.0
    fruit = None
    if art is not None and art.fruits:
        fruit = art.fruits[3] if len(art.fruits) > 3 else art.fruits[0]
    if fruit is not None:
        size = s * 0.62
        if not eaten:
            wh = fruit.whole
            _sprites.blit(img, wh, snack_at[0], snack_at[1], size, size * wh.shape[0] / wh.shape[1],
                          angle=math.sin(t * 2) * 0.2)
        else:
            fall = 300 * since * since
            for piece, dx in ((fruit.half_a, -1), (fruit.half_b, 1)):
                _sprites.blit(img, piece, snack_at[0] + dx * 70 * since, snack_at[1] + fall - 40 * since,
                              size, size * piece.shape[0] / piece.shape[1],
                              angle=dx * since * 3, alpha=max(0.0, 1 - since / 1.2))
            text(img, "+1", snack_at[0], snack_at[1] - 30 - 40 * since, 0.8, theme.ACCENT,
                 alpha=max(0.0, 1 - since / 1.0))

    # Purple trail behind the hand while it moves fast.
    for k in range(7, 0, -1):
        ua = ease((phase - 0.35 - k * 0.035) / 0.75)
        ub = ease((phase - 0.35 - (k - 1) * 0.035) / 0.75)
        if ub - ua < 0.004:
            continue
        a, b = P(hand_pos(_swing(ua))), P(hand_pos(_swing(ub)))
        stroke_alpha(img, [a, b], TRAIL, s * 0.16 * (1 - k / 9), 0.75 * (1 - k / 8))

    hands = draw_figure(img, cx, cy, s, left=REST_LEFT, right=_swing(u))
    if art is not None:
        head = art.cursor_chomp if eaten and since < 0.35 else art.cursor_idle
        if head is not None:
            _sprites.blit(img, head, hands.right[0], hands.right[1], s * 0.62, s * 0.62)


def _draw_trash(img, rect, t: float) -> None:
    art = _get_art()
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    bx, by = x0 + w * 0.32, y0 + h * 0.46 + math.sin(t * 2.4) * 8
    if art is not None and art.bomb is not None:
        b = art.bomb
        _sprites.blit(img, b, bx, by, h * 0.5, h * 0.5 * b.shape[0] / b.shape[1],
                      angle=math.sin(t * 1.7) * 0.15)
    cross(img, bx, by, h * 0.2)
    # Lives: a dropped snack costs one; the third one blinks out.
    lx, ly, size = x0 + w * 0.72, y0 + h * 0.5, h * 0.17
    lost = int((t % 3) // 1)  # 0, 1, 2 lives lost, looping
    if art is not None:
        for i in range(3):
            icon = art.life_lost if i >= 3 - lost else art.life_full
            if icon is not None:
                _sprites.blit(img, icon, lx + (i - 1) * size * 1.15, ly, size, size)
    text(img, "x3", lx, ly + size * 1.4, 0.6, X3_COLOUR, body=True)


SNACK_TUTORIAL = [
    stand_back_step(
        "STAND BACK",
        "About 2 m from the screen, with your head, shoulders and hands in the picture.",
    ),
    TutorialStep(
        "YOUR HAND IS THE RACCOON",
        "Swipe your right hand fast through the snacks to gobble them. A slow hand won't bite.",
        demo(_draw_swipe),
    ),
    TutorialStep(
        "AVOID THE TRASH",
        "Grabbing a bag of trash ends the game. Let three snacks fall and it's over too.",
        demo(_draw_trash),
    ),
]
