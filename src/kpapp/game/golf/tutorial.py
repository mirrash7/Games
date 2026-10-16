"""The "how to play" cards shown before a player's first Raccoon Golf round.

Three animated cards, drawn with the shared stick figure (figure.py): address
(hands together, low and still), backswing (the gauge fills with hand height,
red above the head), and the swing through (with putting as a pendulum).
Nothing here loads art, so the cards draw even before the assets exist.
"""

from __future__ import annotations

import math

import cv2

from ... import theme
from ...figure import TutorialStep, demo, disc, draw_figure, ease, lerp, reach, stroke, text

GREEN = (88, 170, 70)  # BGR grass under the figure
SHAFT = (215, 220, 230)
GRIP_LOW = (0.0, 0.42)  # figure units: hands together at the waist


def _arms_to(grip):
    """Both arms reaching the same grip point (two-bone IK per arm)."""
    return dict(left=reach(grip, "left"), right=reach(grip, "right"))


def _swing_grip(angle: float, radius: float = 0.92):
    """The grip on an arc round the chest: 0 hanging at the waist, + up to the viewer's left."""
    return (-math.sin(angle) * radius, -0.5 + math.cos(angle) * radius)


def _club(img, cx, cy, s, grip, length=1.0, colour=SHAFT):
    """A club from the grip, pointing away from the shoulders' centre."""
    gx, gy = grip
    dx, dy = gx - 0.0, gy - (-0.5)
    n = math.hypot(dx, dy) or 1.0
    hx, hy = gx + dx / n * length, gy + dy / n * length
    a = (cx + gx * s, cy + gy * s)
    b = (cx + hx * s, cy + hy * s)
    stroke(img, [a, b], (20, 16, 24), s * 0.07)
    stroke(img, [a, b], colour, s * 0.035)
    disc(img, b, s * 0.06, (60, 56, 70))


def _ground(img, rect, y):
    x0, _, x1, y1 = rect
    cv2.rectangle(img, (int(x0), int(y)), (int(x1), int(y1)), GREEN, -1)


def _draw_address(img, rect, t: float) -> None:
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    s = h * 0.36
    cx, cy = x0 + w * 0.42, y0 + h * 0.52
    _ground(img, rect, cy + 0.95 * s)
    period = 3.0
    u = (t % period) / period
    # Hands come together at the waist (0..0.35), then hold still (0.35..).
    k = ease(u / 0.35)
    apart_l, apart_r = (-0.55, 0.25), (0.55, 0.25)
    gl = (lerp(apart_l[0], GRIP_LOW[0], k), lerp(apart_l[1], GRIP_LOW[1], k))
    gr = (lerp(apart_r[0], GRIP_LOW[0], k), lerp(apart_r[1], GRIP_LOW[1], k))
    draw_figure(img, cx, cy, s, left=reach(gl, "left"), right=reach(gr, "right"))
    hold = max(0.0, (u - 0.35) / 0.25)
    if 0.0 < hold < 1.0:
        c = (int(cx + GRIP_LOW[0] * s), int(cy + GRIP_LOW[1] * s))
        cv2.ellipse(img, c, (int(s * 0.22),) * 2, -90, 0, 360 * hold, theme.ACCENT, 3, cv2.LINE_AA)
    if hold >= 1.0:
        _club(img, cx, cy, s, GRIP_LOW, 0.9)
        text(img, "READY", x0 + w * 0.78, y0 + h * 0.3, 0.75, theme.CYAN, alpha=min(1.0, (u - 0.6) * 6))
    else:
        text(img, "HOLD STILL", x0 + w * 0.76, y0 + h * 0.3, 0.55, theme.WARN, alpha=min(1.0, u * 4), body=True)


def _gauge(img, x, y0, y1, power: float) -> None:
    """A small power gauge: fill to `power` (1.15 = top), red zone above 100%."""
    full = y1 - (y1 - y0) / 1.15
    cv2.rectangle(img, (int(x - 9), int(y0 - 3)), (int(x + 9), int(y1 + 3)), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(img, (int(x - 7), int(y0)), (int(x + 7), int(full)), (50, 40, 150), -1)
    top = y1 - (y1 - y0) * min(1.15, power) / 1.15
    colour = theme.DANGER if power > 1.0 else theme.ACCENT if power > 0.8 else theme.CYAN
    cv2.rectangle(img, (int(x - 7), int(top)), (int(x + 7), int(y1)), colour, -1)
    cv2.line(img, (int(x - 12), int(full)), (int(x + 12), int(full)), (255, 255, 255), 2, cv2.LINE_AA)


def _draw_backswing(img, rect, t: float) -> None:
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    s = h * 0.36
    cx, cy = x0 + w * 0.42, y0 + h * 0.54
    _ground(img, rect, cy + 0.95 * s)
    period = 3.2
    u = (t % period) / period
    # Two backswings: a full one to head height, then an overswing above it.
    big = u >= 0.5
    v = (u % 0.5) / 0.5
    top = 2.45 if big else 2.05
    a = top * (ease(v / 0.55) if v < 0.75 else 1 - ease((v - 0.75) / 0.25))
    grip = _swing_grip(a)
    draw_figure(img, cx, cy, s, **_arms_to(grip))
    _club(img, cx, cy, s, grip, 0.9)
    head_y = -0.86
    power = min(1.15, max(0.0, (GRIP_LOW[1] - grip[1]) / (GRIP_LOW[1] - head_y)))
    _gauge(img, x0 + w * 0.86, y0 + h * 0.12, y0 + h * 0.84, power)
    if power > 1.0:
        text(img, "OVERSWING", x0 + w * 0.66, y0 + h * 0.13, 0.5, theme.DANGER, body=True)
    elif power > 0.85:
        text(img, "FULL POWER", x0 + w * 0.64, y0 + h * 0.13, 0.5, theme.ACCENT, body=True)


def _draw_swing(img, rect, t: float) -> None:
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    s = h * 0.34
    period = 3.4
    u = (t % period) / period
    putt = u >= 0.55
    if not putt:
        cx, cy = x0 + w * 0.36, y0 + h * 0.54
        _ground(img, rect, cy + 0.95 * s)
        v = u / 0.55
        # Hold at the top, then a fast swing down through the ball to a high finish.
        if v < 0.3:
            a = 2.05
        else:
            k = min(1.0, (v - 0.3) / 0.25)
            a = 2.05 + (-2.0 - 2.05) * (k * k * (3 - 2 * k))
        grip = _swing_grip(a)
        draw_figure(img, cx, cy, s, **_arms_to(grip))
        _club(img, cx, cy, s, grip, 0.9)
        impact = 0.3 + 0.25 * 0.5
        bx, by = cx + 0.95 * s, cy + 0.9 * s
        if v < impact:
            disc(img, (bx, by), s * 0.06, (255, 255, 255))
        else:
            f = (v - impact) / (1 - impact)
            px = bx + f * w * 0.55
            py = by - math.sin(min(1.0, f * 1.2) * math.pi) * h * 0.45
            disc(img, (px, py), s * 0.06, (255, 255, 255))
            text(img, "NICE SHOT!", x0 + w * 0.62, y0 + h * 0.16, 0.65, theme.ACCENT, alpha=min(1.0, f * 4))
    else:
        cx, cy = x0 + w * 0.42, y0 + h * 0.5
        _ground(img, rect, cy + 0.95 * s)
        v = (u - 0.55) / 0.45
        swing = math.sin(v * 2 * math.pi) * 0.42
        grip = (GRIP_LOW[0] + swing, GRIP_LOW[1] - abs(swing) * 0.1)
        draw_figure(img, cx, cy, s, **_arms_to(grip))
        _club(img, cx, cy, s, grip, 0.55)
        text(img, "PUTT: SIDE TO SIDE", x0 + w * 0.5, y0 + h * 0.13, 0.5, theme.CYAN, body=True)


GOLF_TUTORIAL = [
    TutorialStep(
        "HANDS TOGETHER, LOW",
        "Stand back so your hands show. Hold them together at your waist, still, until READY.",
        demo(_draw_address),
    ),
    TutorialStep(
        "SWING BACK FOR POWER",
        "Lift your hands up to one side. Head height is full power - higher overswings and curves.",
        demo(_draw_backswing),
    ),
    TutorialStep(
        "SWING THROUGH TO HIT",
        "Swing down fast across your body. On the green, rock your hands side to side to putt.",
        demo(_draw_swing),
    ),
]
