"""The "how to play" cards shown before a player's first Raccoon Dance song.

Three cards in the house style (figure.py): stand back, copy the coach like a
mirror, hit the poses on the beat. The second and third draw the real coach and
pictogram cards from the game's renderer, so the tutorial teaches what the
player will actually see. The art loads at the first draw; without it the cards
fall back to the plain figure. All three together cost well under 1.5 ms a frame:
everything is a cached sprite or a handful of anti-aliased polygons.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ... import theme
from ...figure import (TutorialStep, demo, draw_figure, ease, fill_alpha, lerp, reach,
                       stand_back_step, stroke, text)
from .art import AssetsMissingError, load_art
from .pose import REST
from .render import CoachPainter, PictogramCards, ScaledSprites, Sprite, _resize_bgra, draw_pictogram_figure

STAGE_BG = (40, 16, 30)  # the stage's dark violet, BGR
FLOOR = (90, 40, 110)
COACH_ARM_DOWN, COACH_ARM_UP = [14.0, 8.0], [158.0, 172.0]
PLAYER_REST, PLAYER_UP = (0.42, 0.3), (0.62, -1.15)

_art = None  # tests may set this to stub art; None loads the real art at first draw
_art_missing = False
_painters: dict[int, CoachPainter] = {}
_cards: PictogramCards | None = None
_words: dict[str, ScaledSprites] = {}
_rings: dict[tuple[int, bool], Sprite] = {}


def _get_art():
    global _art, _art_missing
    if _art is None and not _art_missing:
        try:
            _art = load_art()
        except AssetsMissingError:
            _art_missing = True  # draw the figure alone
    return _art


def _painter(art, scale: float) -> CoachPainter:
    key = int(round(scale))
    p = _painters.get(key)
    if p is None or p.c is None:
        # Flat, as before the stage coach got its lighting: the cards are small
        # instructions (and their 1.5 ms budget covers three of them).
        p = _painters[key] = CoachPainter(art, "neon", scale=key, shade=False)
    return p


def _word(art, name: str, scale: float) -> Sprite | None:
    if art is None or name not in art.words:
        return None
    w = _words.get(name)
    if w is None:
        w = _words[name] = ScaledSprites(art.words[name], step=0.05)
    return w.get(scale)


def _ring(art, size: int, gold: bool) -> Sprite | None:
    if art is None:
        return None
    key = (size, gold)
    r = _rings.get(key)
    if r is None:
        tint = (70, 205, 255) if gold else (255, 235, 70)
        r = _rings[key] = Sprite(_resize_bgra(art.ring, size, size), tint=tint)
    return r


def _arm_up(t: float, delay: float = 0.0) -> float:
    """0 (down) .. 1 (up) over a 2.8 s loop: rise, hold, drop."""
    p = (t - delay) % 2.8
    if p < 0.25:
        return 0.0
    if p < 0.8:
        return ease((p - 0.25) / 0.55)
    if p < 1.9:
        return 1.0
    if p < 2.4:
        return 1.0 - ease((p - 1.9) / 0.5)
    return 0.0


def _backdrop(img, rect) -> None:
    x0, y0, x1, y1 = rect
    fill_alpha(img, STAGE_BG, 0.92)
    fy = y0 + (y1 - y0) * 0.86
    stroke(img, [(x0, fy), (x1, fy)], FLOOR, 2)


def _draw_mirror(img, rect, t: float) -> None:
    """The coach raises its screen-right arm; the player, a beat behind, does the same."""
    art = _get_art()
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    _backdrop(img, rect)
    feet = y0 + h * 0.86
    u_coach, u_you = _arm_up(t), _arm_up(t, 0.3)
    ccx = x0 + w * 0.27
    if art is not None:
        p = _painter(art, h * 0.175)
        pose = dict(REST)
        pose["ra"] = [lerp(COACH_ARM_DOWN[0], COACH_ARM_UP[0], u_coach), lerp(COACH_ARM_DOWN[1], COACH_ARM_UP[1], u_coach)]
        pose["la"] = [16.0, 12.0]
        geo = p.layout(pose, beat=t * 2.0, origin=p.origin_for(ccx, feet))
        p.paint(img, geo)
    else:
        right = reach(_lerp2(PLAYER_REST, PLAYER_UP, u_coach))
        draw_figure(img, ccx, feet - h * 0.42, h * 0.38, right=right, colour=(200, 80, 255))
    text(img, "COACH", ccx, y1 - 6, 0.48, theme.DIM, body=True, thick=1)

    # The mirror: a dashed line between them.
    mx = x0 + w * 0.5
    for yy in np.arange(y0 + 14, feet, 14):
        stroke(img, [(mx, yy), (mx, min(feet, yy + 7))], (190, 150, 210), 2)

    ycx = x0 + w * 0.73
    target = _lerp2(PLAYER_REST, PLAYER_UP, u_you)
    left_rest = reach((-PLAYER_REST[0], PLAYER_REST[1]), side="left")
    right = reach(target)
    s = h * 0.29
    draw_figure(img, ycx, feet - s * 0.95, s, left=left_rest, right=right)
    text(img, "YOU", ycx, y1 - 6, 0.48, theme.DIM, body=True, thick=1)

    # Both arms up on the same side of the screen: PERFECT.
    p = (t - 0.3) % 2.8
    if 0.8 <= p < 1.9:
        age = p - 0.8
        spr = _word(art, "perfect", 0.34 * _pop(age))
        fade = 1.0 if age < 0.8 else max(0.0, 1 - (age - 0.8) / 0.3)
        if spr is not None:
            spr.draw_c(img, mx, y0 + h * 0.11, fade=fade)
        else:
            text(img, "PERFECT", mx, y0 + h * 0.17, 0.7, theme.ACCENT, alpha=fade)


def _lerp2(a, b, u):
    return (lerp(a[0], b[0], u), lerp(a[1], b[1], u))


def _pop(age: float) -> float:
    if age < 0.1:
        return 0.5 + 0.8 * age / 0.1
    if age < 0.22:
        return 1.3 - 0.3 * (age - 0.1) / 0.12
    return 1.0


DEMO_POSES = [
    {"la": [90, 90], "ra": [90, 90]},
    {"ra": [150, 165], "la": [20, 10], "lean": 8},
    {"la": [170, 175], "ra": [170, 175]},
    {"la": [60, -20], "ra": [60, -20], "squat": 0.8},
]
BEAT = 0.6  # seconds per beat in the demo


def _draw_lane(img, rect, t: float) -> None:
    """Cards scroll onto the ring; each scores as it lands; the gold one counts double."""
    global _cards
    art = _get_art()
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    _backdrop(img, rect)
    ly = y0 + h * 0.6
    rx = x0 + w * 0.24
    spacing = w * 0.3
    card_w = int(round(h * 0.34 / 4) * 4)
    ring = _ring(art, int(card_w * 1.3), False)
    beat = t / BEAT
    kick = math.exp(-(beat - math.floor(beat)) * 5)
    if ring is not None:
        ring.draw_c(img, rx, ly, fade=0.7 + 0.3 * kick)
    else:
        cv2.circle(img, (int(rx), int(ly)), int(card_w * 0.62), (255, 235, 70), 2, cv2.LINE_AA)
    # Beat dots under the lane.
    for b in range(math.floor(beat) - 2, math.floor(beat) + 6):
        x = rx + (b - beat) * spacing
        if x0 + 8 < x < x1 - 8:
            cv2.circle(img, (int(x), int(ly + card_w * 0.82)), 3 if b % 4 == 0 else 2, (240, 160, 210), -1, cv2.LINE_AA)
    if art is not None and _cards is None:
        _cards = PictogramCards(art)
    n0 = math.floor(beat) - 1
    for n in range(n0 + 5, n0 - 1, -1):  # farthest first
        d = n - beat
        x = rx + d * spacing
        if x - card_w / 2 > x1 or d < -0.6:
            continue
        gold = n % 3 == 2
        pose = DEMO_POSES[n % len(DEMO_POSES)]
        fade = 1.0 if d >= 0 else max(0.0, 1 + d / 0.6)
        size = card_w + (int(card_w * 0.25 / 4) * 4 if 0 <= d < 1 else 0)
        if _cards is not None:
            _cards.get(pose, gold, size).draw_c(img, x, ly, fade=fade)
        else:
            half = size / 2
            cv2.rectangle(img, (int(x - half), int(ly - half)), (int(x + half), int(ly + half)),
                          (70, 205, 255) if gold else (255, 90, 230), 2, cv2.LINE_AA)
            draw_pictogram_figure(img, pose, (x - half * 0.7, ly - half * 0.8, x + half * 0.7, ly + half * 0.8), gold)
        if gold and d > 0.2:
            text(img, "x2", x, ly - size * 0.62, 0.5, theme.ACCENT, thick=1)
    # The card on the ring scores.
    last = math.floor(beat)
    age = (beat - last) * BEAT
    gold = last % 3 == 2
    name = "perfect" if last % 2 == 0 or gold else "super"
    spr = _word(art, name, 0.52 * _pop(age))
    fade = 1.0 if age < 0.35 else max(0.0, 1 - (age - 0.35) / 0.2)
    if spr is not None:
        spr.draw_c(img, rx + w * 0.1, y0 + h * 0.17, fade=fade)
        if gold:
            text(img, "x2!", rx + w * 0.1 + spr.w * 0.5 + 24, y0 + h * 0.2, 0.75, theme.ACCENT, alpha=fade)
    else:
        text(img, name.upper(), rx + w * 0.16, y0 + h * 0.2, 0.6,
             theme.ACCENT if name == "perfect" else theme.CYAN, alpha=fade)


DANCE_TUTORIAL = [
    stand_back_step(
        "STAND BACK",
        "About 2 m from the screen, so your head, both arms and your body are in the picture.",
        arms_up=True,
    ),
    TutorialStep(
        "COPY LIKE A MIRROR",
        "The coach is your mirror: when its arm on the right of the screen goes up, raise your right arm.",
        demo(_draw_mirror),
    ),
    TutorialStep(
        "HIT THE POSES ON THE BEAT",
        "Strike each pose as its card reaches the ring. Gold moves count double.",
        demo(_draw_lane),
    ),
]
