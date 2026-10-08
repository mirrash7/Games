"""A stick figure and drawing helpers for the "how to play" tutorial cards.

Port of web/js/core/figure.js. The figure is drawn procedurally so it scales
to any card and animates from a couple of joint angles.

Units: 1 = roughly a shoulder width. Origin at the chest. Angles are screen
angles in radians (0 = pointing right, pi/2 = straight down). "left"/"right"
are the viewer's sides - on the mirrored camera feed the player's right hand
is on the viewer's right, exactly as in a mirror.

Everything draws in place on a BGR uint8 image, anti-aliased, with sub-pixel
coordinates. Tutorial demos draw through `clipped`, which hands them a scratch
copy of their rect and copies it back through a rounded mask, so a demo can
never touch a pixel outside its rect.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, NamedTuple

import cv2
import numpy as np

from . import theme
from .gfx import alpha_composite, blit as _gfx_blit

DOWN = math.pi / 2
UPPER, FORE = 0.36, 0.32

FIGURE_COLOUR = (246, 232, 236)  # rgb(236,232,246), BGR
OUTLINE = (16, 8, 10)  # rgba(10,8,16,0.9), drawn opaque

_SHIFT = 4  # sub-pixel bits for cv2 drawing
_ONE = 1 << _SHIFT


@dataclass(frozen=True)
class TutorialStep:
    """One how-to-play card: a title, a sentence, and an animated demo.

    draw(frame, rect, t) draws the demo for time t (seconds since the tutorial
    opened) inside rect = (x0, y0, x1, y1), touching no pixel outside it.
    """

    title: str
    text: str
    draw: Callable[[np.ndarray, tuple[int, int, int, int], float], None]


class Hands(NamedTuple):
    left: tuple[float, float]
    right: tuple[float, float]


# --- maths (same as figure.js) ---

def _at(o, a, length):
    return (o[0] + math.cos(a) * length, o[1] + math.sin(a) * length)


def _shoulder(side: str):
    return (0.3, -0.5) if side == "right" else (-0.3, -0.5)


def hand_pos(arm, side: str = "right"):
    """Hand position in figure units for the right or left arm [upper, fore]."""
    a1, a2 = arm
    return _at(_at(_at(_shoulder(side), a1, UPPER), a2, FORE), a2, 0.1)


def reach(target, side: str = "right"):
    """Arm angles that put the wrist at `target` (figure units), elbow low: two-bone IK."""
    sh = _shoulder(side)
    dx, dy = target[0] - sh[0], target[1] - sh[1]
    d = min(max(math.hypot(dx, dy), abs(UPPER - FORE) + 1e-3), UPPER + FORE - 1e-3)
    phi = math.atan2(dy, dx)
    alpha = math.acos((UPPER * UPPER + d * d - FORE * FORE) / (2 * UPPER * d))
    # Of the two elbow solutions, the lower one (larger y).
    a1 = max((phi + alpha, phi - alpha), key=lambda a: _at(sh, a, UPPER)[1])
    e = _at(sh, a1, UPPER)
    w = _at(sh, phi, d)
    return (a1, math.atan2(w[1] - e[1], w[0] - e[0]))


def mirror(a: float) -> float:
    """Mirror a right-arm angle to the left side."""
    return math.pi - a


def ease(u: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * min(1.0, max(0.0, u)))


def lerp(a: float, b: float, u: float) -> float:
    return a + (b - a) * u


# --- primitive drawing (sub-pixel, anti-aliased) ---

def _fix(points) -> np.ndarray:
    return np.round(np.asarray(points, np.float64) * _ONE).astype(np.int32).reshape(-1, 1, 2)


def stroke(img, points, colour, width: float, closed: bool = False) -> None:
    """A round-capped, round-joined polyline `width` px wide."""
    cv2.polylines(img, [_fix(points)], closed, colour, max(1, int(round(width))), cv2.LINE_AA, _SHIFT)


def disc(img, centre, radius: float, colour) -> None:
    c = (int(round(centre[0] * _ONE)), int(round(centre[1] * _ONE)))
    cv2.circle(img, c, max(1, int(round(radius * _ONE))), colour, -1, cv2.LINE_AA, _SHIFT)


_flat: dict[tuple, np.ndarray] = {}


def fill_alpha(img, colour, alpha: float) -> None:
    """Blend a flat colour over the whole image (e.g. a translucent backdrop)."""
    key = (img.shape, tuple(colour))
    flat = _flat.get(key)
    if flat is None:
        flat = _flat[key] = np.full(img.shape, colour, np.uint8)
    cv2.addWeighted(flat, alpha, img, 1.0 - alpha, 0, img)


def stroke_alpha(img, points, colour, width: float, alpha: float) -> None:
    """A translucent stroke, blended only inside its own bounding box."""
    if alpha <= 0.003:
        return
    if alpha >= 0.997:
        stroke(img, points, colour, width)
        return
    pts = np.asarray(points, np.float64)
    pad = width / 2 + 2
    x0, y0 = (int(math.floor(v)) for v in pts.min(axis=0) - pad)
    x1, y1 = (int(math.ceil(v)) for v in pts.max(axis=0) + pad)
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    over = roi.copy()
    stroke(over, pts - (x0, y0), colour, width)
    cv2.addWeighted(over, alpha, roi, 1.0 - alpha, 0, roi)


def text(img, s: str, cx: float, y: float, scale: float, colour, alpha: float = 1.0,
         body: bool = False, thick: int = 2) -> None:
    """Centred, outlined text (theme.centred_text), optionally faded."""
    if alpha <= 0.003:
        return
    font = theme.BODY_FONT if body else theme.TITLE_FONT
    if alpha >= 0.997:
        theme.centred_text(img, s, cx, int(round(y)), scale, colour, thick=thick, font=font)
        return
    (tw, th), base = cv2.getTextSize(s, font, scale, thick)
    pad = 4
    x0, y0 = int(cx - tw / 2) - pad, int(round(y)) - th - pad
    x1, y1 = x0 + tw + 2 * pad, int(round(y)) + base + pad
    h, w = img.shape[:2]
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    roi = img[cy0:cy1, cx0:cx1]
    over = roi.copy()
    theme.centred_text(over, s, cx - cx0, int(round(y)) - cy0, scale, colour, thick=thick, font=font)
    cv2.addWeighted(over, alpha, roi, 1.0 - alpha, 0, roi)


def dashed_rect(img, rect, colour, width: int = 3, dash=(10.0, 7.0)) -> None:
    """strokeRect with setLineDash: dashes run clockwise from the top-left corner."""
    x0, y0, x1, y1 = rect
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    on, off = dash
    cap = width / 2  # cv2 caps are round; shorten so a dash reads `on` px long
    pos = 0.0  # distance into the dash pattern
    dashes = []
    for (ax, ay), (bx, by) in zip(corners, corners[1:]):
        seg = math.hypot(bx - ax, by - ay)
        ux, uy = (bx - ax) / seg, (by - ay) / seg
        d = 0.0
        while d < seg - 1e-9:
            p = pos % (on + off)
            if p < on:
                n = min(on - p, seg - d)
                a = d + (cap if p < 1e-6 else 0)
                b = d + n - (cap if p + n >= on - 1e-6 else 0)
                if b <= a:
                    a = b = d + n / 2
                dashes.append([(ax + ux * a, ay + uy * a), (ax + ux * b, ay + uy * b)])
            else:
                n = min(on + off - p, seg - d)
            d += n
            pos += n
    if dashes:  # one call for every dash
        cv2.polylines(img, [_fix(dd) for dd in dashes], False, colour, max(1, int(round(width))),
                      cv2.LINE_AA, _SHIFT)


def cross(img, cx: float, cy: float, r: float) -> None:
    """A bold red X, for "don't"."""
    lines = [[(cx - r, cy - r), (cx + r, cy + r)], [(cx + r, cy - r), (cx - r, cy + r)]]
    for colour, w in ((OUTLINE, r * 0.42), (theme.DANGER, r * 0.26)):
        for ln in lines:
            stroke(img, ln, colour, w)


# --- sprites: pre-scaled and rotated once, cached ---

def _premul(img: np.ndarray) -> np.ndarray:
    out = img.astype(np.float32)
    out[:, :, :3] *= out[:, :, 3:4] / 255.0
    return out


def _unpremul(img: np.ndarray) -> np.ndarray:
    a = img[:, :, 3:4]
    out = img.copy()
    out[:, :, :3] = np.where(a > 0, img[:, :, :3] * 255.0 / np.maximum(a, 1e-3), 0)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


class SpriteCache:
    """Sprites resized to a target size and rotated in angle buckets.

    Resizing and rotation run on premultiplied colour, so transparent edges
    never pick up a dark fringe. Each (sprite, size, bucket) is built once.
    """

    STEP_DEG = 3.0

    def __init__(self) -> None:
        self._cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    def get(self, img: np.ndarray, w: float, h: float, angle: float = 0.0) -> np.ndarray:
        w, h = max(1, int(round(w))), max(1, int(round(h)))
        bucket = int(round(math.degrees(angle) / self.STEP_DEG)) % int(round(360 / self.STEP_DEG))
        key = (id(img), w, h, bucket)
        hit = self._cache.get(key)
        if hit is not None:
            return hit[1]
        pm = _premul(img)
        interp = cv2.INTER_AREA if w < img.shape[1] else cv2.INTER_LINEAR
        pm = cv2.resize(pm, (w, h), interpolation=interp)
        if bucket:
            d = int(math.ceil(math.hypot(w, h))) + 2
            pad = np.zeros((d, d, 4), np.float32)
            oy, ox = (d - h) // 2, (d - w) // 2
            pad[oy:oy + h, ox:ox + w] = pm
            # Canvas rotate(angle) turns clockwise on screen; OpenCV's positive
            # angle turns counter-clockwise, hence the minus.
            m = cv2.getRotationMatrix2D((d / 2, d / 2), -bucket * self.STEP_DEG, 1.0)
            pm = cv2.warpAffine(pad, m, (d, d), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
        out = _unpremul(pm)
        self._cache[key] = (img, out)  # keep img alive so its id stays unique
        return out

    def blit(self, dst, img, cx, cy, w, h, angle: float = 0.0, alpha: float = 1.0) -> None:
        """Draw `img` centred on (cx, cy) at w x h, rotated and faded (like gfx.js blit)."""
        if alpha <= 0.003:
            return
        _gfx_blit(dst, self.get(img, w, h, angle), (cx, cy), alpha=alpha)

    def blit_corner(self, dst, img, x, y, w, h) -> None:
        """Draw `img` with its top-left at (x, y), like drawImage."""
        alpha_composite(dst, self.get(img, w, h), int(round(x)), int(round(y)))


# --- clipping ---

_masks: dict[tuple, tuple[np.ndarray, tuple]] = {}


def _round_mask(shape, rect, radius):
    """A rounded rect's fully covered pixels (row block + spans) and its anti-aliased edge pixels."""
    key = (shape, tuple(round(v * _ONE) for v in rect), radius)
    hit = _masks.get(key)
    if hit is not None:
        return hit
    h, w = shape
    x0, y0, x1, y1 = rect
    r = max(0.0, min(radius, (x1 - x0) / 2, (y1 - y0) / 2))
    # Fill the rounded rect at 4x and area-average down: an anti-aliased edge.
    k = 4
    big = np.zeros((h * k, w * k), np.uint8)
    # Canvas pixel i spans [i, i+1); supersampled pixel j's centre is at j.
    S = lambda v: int(round((v * k - 0.5) * _ONE))  # noqa: E731
    pts = []
    for (cx, cy), a0 in (((x1 - r, y0 + r), -90), ((x1 - r, y1 - r), 0),
                         ((x0 + r, y1 - r), 90), ((x0 + r, y0 + r), 180)):
        for a in np.linspace(math.radians(a0), math.radians(a0 + 90), 12):
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    poly = np.array([[S(x), S(y)] for x, y in pts], np.int32)
    cv2.fillPoly(big, [poly], 255, cv2.LINE_8, _SHIFT)
    m = cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)
    full = m >= 255
    ys, xs = np.nonzero((m > 0) & ~full)
    edge = (ys, xs, m[ys, xs].astype(np.uint16)[:, None])
    # Fully covered pixels as row spans: one block of rows sharing the widest
    # span, plus one span per corner row. Slice copies are ~10x cheaper than a
    # masked copyto.
    row_spans = {}
    for y in np.nonzero(full.any(axis=1))[0]:
        xs_full = np.nonzero(full[y])[0]
        row_spans[int(y)] = (int(xs_full[0]), int(xs_full[-1]) + 1)
    block = (0, 0, 0, 0)
    if row_spans:
        widest = max(row_spans.values(), key=lambda sp: sp[1] - sp[0])
        rows = [y for y, sp in row_spans.items() if sp == widest]
        block = (rows[0], rows[-1] + 1, *widest)  # rounded rect: these rows are contiguous
    spans = [(y, xa, xb) for y, (xa, xb) in row_spans.items() if not block[0] <= y < block[1]]
    hit = (block, spans, edge)
    _masks[key] = hit
    return hit


def clipped(img, rect, draw: Callable[[np.ndarray, tuple], None], radius: float = 14) -> None:
    """Run draw(buf, local_rect) on a scratch copy, then copy back through a rounded mask.

    Like canvas clipTo: nothing drawn outside the rounded rect reaches `img`.
    `local_rect` is `rect` in the scratch buffer's coordinates.
    """
    x0, y0, x1, y1 = rect
    H, W = img.shape[:2]
    ix0, iy0 = max(0, int(math.floor(x0))), max(0, int(math.floor(y0)))
    ix1, iy1 = min(W, int(math.ceil(x1))), min(H, int(math.ceil(y1)))
    if ix1 <= ix0 or iy1 <= iy0:
        return
    roi = img[iy0:iy1, ix0:ix1]
    buf = roi.copy()
    local = (x0 - ix0, y0 - iy0, x1 - ix0, y1 - iy0)
    draw(buf, local)
    (ya, yb, xa0, xb0), spans, (ys, xs, a) = _round_mask(buf.shape[:2], local, radius)
    roi[ya:yb, xa0:xb0] = buf[ya:yb, xa0:xb0]
    for y, xa, xb in spans:
        roi[y, xa:xb] = buf[y, xa:xb]
    if len(ys):
        roi[ys, xs] = ((roi[ys, xs].astype(np.uint16) * (255 - a)
                        + buf[ys, xs].astype(np.uint16) * a + 127) // 255).astype(np.uint8)


def demo(fn: Callable[[np.ndarray, tuple, float], None], radius: float = 14):
    """Wrap a demo fn(img, rect, t) so it draws clipped to its (rounded) rect."""
    def draw(frame: np.ndarray, rect, t: float) -> None:
        clipped(frame, rect, lambda buf, local: fn(buf, local, t), radius)
    draw.__name__ = getattr(fn, "__name__", "draw")
    draw.__doc__ = fn.__doc__
    return draw


# --- the figure ---

def draw_figure(img, cx: float, cy: float, s: float, left=None, right=None,
                colour=FIGURE_COLOUR, alpha: float = 1.0) -> Hands:
    """Draw the figure centred at (cx, cy), `s` px per unit. Returns the hands' pixel positions.

    Arms are (upper_angle, fore_angle) for each side.
    """
    left = left if left is not None else (DOWN - 0.2, DOWN - 0.1)
    right = right if right is not None else (DOWN + 0.2, DOWN + 0.1)
    if alpha < 0.997:
        over = img.copy()
        hands = draw_figure(over, cx, cy, s, left, right, colour)
        cv2.addWeighted(over, max(0.0, alpha), img, 1.0 - max(0.0, alpha), 0, img)
        return hands

    def P(p):
        return (cx + p[0] * s, cy + p[1] * s)

    ls, rs = (-0.3, -0.5), (0.3, -0.5)
    le = _at(ls, left[0], UPPER)
    lw = _at(le, left[1], FORE)
    re = _at(rs, right[0], UPPER)
    rw = _at(re, right[1], FORE)

    def limb(pts, w):
        pp = [P(p) for p in pts]
        stroke(img, pp, OUTLINE, w * s * 1.7)
        stroke(img, pp, colour, w * s)

    # Torso and legs first, so the arms read in front of them.
    limb([(-0.2, 0.95), (-0.17, 0.3), (0.17, 0.3), (0.2, 0.95)], 0.12)
    limb([ls, (-0.17, 0.3), (0.17, 0.3), rs, ls], 0.12)
    limb([(0, -0.52), (0, -0.66)], 0.12)
    head = P((0, -0.86))
    disc(img, head, 0.19 * s + 0.035 * s, OUTLINE)  # stroke 0.07 s, centred on the edge
    disc(img, head, 0.19 * s - 0.035 * s, colour)
    limb([ls, le, lw], 0.11)
    limb([rs, re, rw], 0.11)
    return Hands(P(_at(lw, left[1], 0.1)), P(_at(rw, right[1], 0.1)))


def draw_stand_back(img, rect, t: float, arms_up: bool = False) -> None:
    """The "stand back" demo shared by every game: too close (cropped), then in frame.

    Draws in place, unclipped at the outer rect; use `stand_back_step` for a card.
    """
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    u = ease(((t % 3.2) - 0.4) / 1.4)
    s = lerp(h * 0.95, h * 0.33, u)
    fx, fy, fw, fh = x0 + w * 0.2, y0 + h * 0.08, w * 0.6, h * 0.84
    if arms_up:
        arms = dict(left=(mirror(-0.5), mirror(-1.3)), right=(-0.5, -1.3))
    else:
        arms = dict(left=(mirror(0.9), mirror(-1.2)), right=(0.9, -1.2))

    def inner(buf, r):
        bx0, by0, bx1, by1 = r
        fill_alpha(buf, (58, 34, 40), 0.9)  # rgba(40,34,58,0.9)
        draw_figure(buf, (bx0 + bx1) / 2, by0 + (by1 - by0) * 0.42 + s * 0.12, s, **arms)

    clipped(img, (fx, fy, fx + fw, fy + fh), inner, radius=8)
    fits = u > 0.98
    dashed_rect(img, (fx, fy, fx + fw, fy + fh), theme.GOOD if fits else theme.WARN, 3)
    if fits:
        cx, cy = fx + fw + 26, fy + 26
        stroke(img, [(cx - 13, cy), (cx - 3, cy + 11), (cx + 15, cy - 12)], theme.GOOD, 7)


def stand_back_step(title: str, text_: str, arms_up: bool = False) -> TutorialStep:
    return TutorialStep(title, text_, demo(lambda img, r, t: draw_stand_back(img, r, t, arms_up)))
