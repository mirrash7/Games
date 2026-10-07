#!/usr/bin/env python3
"""Procedurally generate every sprite and backdrop for the Flappy game.

Run once (or whenever the look changes):

    uv run python tools/generate_flappy_assets.py

numpy + opencv only - no downloads, no binary sources, no extra dependencies,
and nothing traced from any existing game. Output goes to
``assets/flappy/generated/`` together with ``manifest.json``; the schema
(file names, formats, manifest keys) is defined by
``kpapp.game.flappy.art``.

Look target: bright, cheerful, chunky retro arcade. Bold dark outlines, flat
colour plus a couple of hard-edged shading bands, read from two metres away.
The hero is "Pip", a round raspberry songbird with a cream belly, a three
feather crest, a pointed sunflower beak, rosy cheeks and one huge eye. The
backdrop is deliberately paler and cooler than the gameplay layer (pipes,
bird) so the things that matter pop.

Technique notes:

* Every sprite is drawn on a :class:`Canvas` supersampled by ``SS`` and box
  filtered down, with shape coverage computed from signed distance fields.
  Clean anti-aliased edges with no ``LINE_AA`` stair-steps.
* The canvas stores *premultiplied* colour, so painting is a plain ``over``
  and the downscale cannot drag a dark fringe into soft edges. Unpremultiply
  bleeds colour outward into transparent pixels, so the game can rotate or
  bilinear-stretch a sprite without picking up a halo.
* Outlines come from a precise distance transform of the part's coverage
  (:meth:`Canvas.grow`), which gives uniform stroke width on any silhouette.
* Tiling assets (``skyline``, ``ground``, and the sky for good measure) use a
  wrapping canvas: analytic primitives are evaluated at ``x`` and ``x +- w``,
  polygons are drawn three times and morphological ops pad with the opposite
  edge, so the left and right columns are true neighbours.
* Gradients are quantised through triangular dither (kills banding). Flat
  sprites are not dithered, so ``pipe_body`` rows are bit-identical and the
  pipe tiles vertically with no seam at all.
* Determinism: one seed, per-tag generators keyed by crc32 (``hash`` is
  salted per process). Re-running gives byte-identical files.
"""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

SEED = 20241006
SS = 4  # supersampling factor for sprites
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "assets" / "flappy" / "generated"

FRAME_W, FRAME_H = 1280, 720
GROUND_H = 96
SKYLINE_H = 240
PIPE_W = 110
CAP_W, CAP_H = 130, 44
# Collision radius at native size, centred on the sprite. The raccoon's head
# is ~29 px in radius; 24 lets ears, wing tips and tail graze a pipe without a
# crash, in keeping with the game's latency-forgiving tuning.
BIRD_HIT_RADIUS = 24


def rgb(r: float, g: float, b: float) -> tuple[float, float, float]:
    """Palette entries are written as RGB for sanity and stored as BGR."""
    return (float(b), float(g), float(r))


# ---------------------------------------------------------------------------
# palette
# ---------------------------------------------------------------------------

# bird ("Pip")
INK = rgb(46, 22, 44)  # plum-black outline shared by the characters

# pipes
PIPE_INK = rgb(22, 52, 30)
PIPE_SHADE2 = rgb(40, 124, 50)
PIPE_SHADE1 = rgb(62, 156, 56)
PIPE_BASE = rgb(96, 192, 64)
PIPE_LIGHT = rgb(148, 222, 90)
PIPE_HI = rgb(222, 250, 170)

# sky
SKY_STOPS = [
    (0.00, rgb(78, 156, 230)),
    (0.45, rgb(122, 192, 242)),
    (0.78, rgb(178, 222, 250)),
    (1.00, rgb(210, 238, 252)),
]
SUN = rgb(255, 246, 200)
SUN_HALO = rgb(255, 252, 228)
CLOUD = rgb(255, 255, 255)
CLOUD_SHADE = rgb(222, 238, 252)

# skyline
CITY = rgb(176, 202, 238)
CITY_EDGE = rgb(146, 176, 222)
CITY_WIN = rgb(204, 224, 248)
BUSH = rgb(128, 198, 150)
BUSH_LIGHT = rgb(166, 222, 172)
BUSH_EDGE = rgb(82, 150, 116)
BUSH_BACK = rgb(150, 206, 184)
BUSH_BACK_LIGHT = rgb(182, 226, 206)
BUSH_BACK_EDGE = rgb(110, 166, 160)

# ground
GROUND_INK = rgb(44, 60, 28)
GRASS_HI = rgb(184, 238, 104)
GRASS = rgb(122, 204, 62)
GRASS_SH = rgb(88, 164, 50)
GRASS_EDGE = rgb(54, 112, 36)
DIRT = rgb(238, 208, 142)
DIRT_STRIPE = rgb(222, 184, 114)
DIRT_DEEP = rgb(204, 162, 96)
DIRT_DEEP_STRIPE = rgb(190, 146, 84)
PEBBLE = rgb(186, 142, 86)
PEBBLE_HI = rgb(248, 226, 172)

# panel
PANEL_INK = rgb(70, 42, 30)
PANEL_FACE = rgb(252, 238, 196)
PANEL_HI = rgb(255, 250, 230)
PANEL_SH = rgb(226, 196, 140)
PANEL_WELL = rgb(238, 214, 158)
PANEL_WELL_SH = rgb(214, 184, 126)
PANEL_RIVET = rgb(232, 170, 80)

MEDALS = {
    # dark ink, shade, base, light, highlight
    "bronze": (rgb(84, 40, 18), rgb(168, 92, 42), rgb(206, 128, 62), rgb(236, 168, 104),
               rgb(255, 218, 176)),
    "silver": (rgb(54, 60, 76), rgb(146, 156, 174), rgb(194, 202, 216), rgb(224, 230, 240),
               rgb(255, 255, 255)),
    "gold": (rgb(104, 62, 0), rgb(216, 146, 18), rgb(252, 196, 38), rgb(255, 226, 104),
             rgb(255, 250, 206)),
    "platinum": (rgb(34, 64, 92), rgb(142, 196, 218), rgb(198, 236, 246), rgb(230, 250, 255),
                 rgb(255, 255, 255)),
}
PLATINUM_GEM = (rgb(40, 120, 220), rgb(110, 190, 255), rgb(220, 246, 255))


# ---------------------------------------------------------------------------
# small numeric helpers
# ---------------------------------------------------------------------------


def _rng(tag: str) -> np.random.Generator:
    """Deterministic per-tag generator (``hash()`` is salted, crc32 is not)."""
    return np.random.default_rng(SEED + zlib.crc32(tag.encode()))


def _smoothstep(t):
    t = np.clip(np.asarray(t, dtype=np.float32), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _blur(img: np.ndarray, sigma: float, wrap: bool = False) -> np.ndarray:
    if sigma <= 0:
        return img
    if not wrap:
        return cv2.GaussianBlur(img, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
    p = int(np.ceil(sigma * 4)) + 1
    padded = np.concatenate([img[:, -p:], img, img[:, :p]], axis=1)
    out = cv2.GaussianBlur(padded, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
    return out[:, p:-p]


def _vramp(h: int, w: int, stops) -> np.ndarray:
    ts = np.array([s[0] for s in stops], np.float32)
    cols = np.array([s[1] for s in stops], np.float32)
    y = (np.arange(h, dtype=np.float32) + 0.5) / h
    ramp = np.stack([np.interp(y, ts, cols[:, c]) for c in range(3)], axis=1).astype(np.float32)
    return np.repeat(ramp[:, None, :], w, axis=1)


def _dither_u8(arr: np.ndarray, tag: str, amount: float = 0.8) -> np.ndarray:
    """Quantise float 0..255 to uint8 with triangular dither (kills banding)."""
    r = _rng("dither" + tag)
    noise = (r.random(arr.shape, dtype=np.float32) - r.random(arr.shape, dtype=np.float32))
    return np.clip(arr + noise * amount + 0.5, 0, 255).astype(np.uint8)


def _ellipse_sd(x: np.ndarray, y: np.ndarray, rx: float, ry: float) -> np.ndarray:
    """First-order signed distance to an axis-aligned ellipse (exact for circles)."""
    k = np.sqrt((x / rx) ** 2 + (y / ry) ** 2)
    g = np.sqrt((x / (rx * rx)) ** 2 + (y / (ry * ry)) ** 2)
    d = (k - 1.0) * k / np.maximum(g, 1e-6)
    return np.where(k < 0.25, -(1.0 - k) * min(rx, ry), d)


def _box_sd(x: np.ndarray, y: np.ndarray, hw: float, hh: float, r: float) -> np.ndarray:
    qx = np.abs(x) - (hw - r)
    qy = np.abs(y) - (hh - r)
    outside = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2)
    inside = np.minimum(np.maximum(qx, qy), 0)
    return outside + inside - r


# ---------------------------------------------------------------------------
# canvas
# ---------------------------------------------------------------------------


class Canvas:
    """Supersampled premultiplied BGRA drawing surface in *final* pixel units.

    Primitives return coverage masks at supersampled resolution; ``paint``
    composites a colour through a mask. ``wrap=True`` makes everything
    periodic in x with period ``w`` so the result tiles horizontally.
    """

    def __init__(self, w: int, h: int, ss: int = SS, wrap: bool = False,
                 origin: tuple[float, float] = (0.0, 0.0)):
        self.w, self.h, self.ss, self.wrap = w, h, ss, wrap
        self.ox, self.oy = origin  # added to every primitive's coordinates
        self.W, self.H = w * ss, h * ss
        self.xs = (np.arange(self.W, dtype=np.float32) + 0.5) / ss
        self.ys = (np.arange(self.H, dtype=np.float32) + 0.5) / ss
        self.pm = np.zeros((self.H, self.W, 3), np.float32)
        self.a = np.zeros((self.H, self.W), np.float32)

    # -- mask helpers -------------------------------------------------------

    def zeros(self) -> np.ndarray:
        return np.zeros((self.H, self.W), np.float32)

    def _cov(self, d: np.ndarray) -> np.ndarray:
        return np.clip(0.5 - d * self.ss, 0.0, 1.0)

    def _field(self, cx: float, cy: float, ext: float,
               fn: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> np.ndarray:
        """Evaluate a local SDF inside a bounding box (with wrap copies)."""
        out = self.zeros()
        cx, cy = cx + self.ox, cy + self.oy
        offsets = (0.0, -float(self.w), float(self.w)) if self.wrap else (0.0,)
        for off in offsets:
            x0 = int(max(0, np.floor((cx + off - ext) * self.ss)))
            x1 = int(min(self.W, np.ceil((cx + off + ext) * self.ss)))
            y0 = int(max(0, np.floor((cy - ext) * self.ss)))
            y1 = int(min(self.H, np.ceil((cy + ext) * self.ss)))
            if x0 >= x1 or y0 >= y1:
                continue
            dx = self.xs[None, x0:x1] - (cx + off)
            dy = self.ys[y0:y1, None] - cy
            dx, dy = np.broadcast_arrays(dx, dy)
            cov = self._cov(fn(dx, dy))
            out[y0:y1, x0:x1] = np.maximum(out[y0:y1, x0:x1], cov)
        return out

    def ellipse(self, cx: float, cy: float, rx: float, ry: float, ang: float = 0.0) -> np.ndarray:
        t = np.deg2rad(ang)
        ca, sa = float(np.cos(t)), float(np.sin(t))

        def fn(dx, dy):
            return _ellipse_sd(dx * ca + dy * sa, -dx * sa + dy * ca, rx, ry)

        return self._field(cx, cy, max(rx, ry) + 2.0, fn)

    def circle(self, cx: float, cy: float, r: float) -> np.ndarray:
        return self.ellipse(cx, cy, r, r)

    def rrect(self, x0: float, y0: float, x1: float, y1: float, r: float) -> np.ndarray:
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        hw, hh = (x1 - x0) / 2.0, (y1 - y0) / 2.0
        r = min(r, hw, hh)
        return self._field(cx, cy, max(hw, hh) + 2.0, lambda dx, dy: _box_sd(dx, dy, hw, hh, r))

    def poly(self, pts) -> np.ndarray:
        m = np.zeros((self.H, self.W), np.uint8)
        arr = np.asarray(pts, np.float64)
        offsets = (0.0, -float(self.w), float(self.w)) if self.wrap else (0.0,)
        for off in offsets:
            p = (arr + [off + self.ox, self.oy]) * self.ss * 16.0
            cv2.fillPoly(m, [np.round(p).astype(np.int32)], 255, cv2.LINE_AA, shift=4)
        return m.astype(np.float32) / 255.0

    def halfplane_y(self, y: float, below: bool = True) -> np.ndarray:
        """Coverage of y' > y (``below``) or y' < y."""
        y = y + self.oy
        d = (y - self.ys) if below else (self.ys - y)
        return np.repeat(self._cov(d)[:, None], self.W, axis=1)

    def shift(self, mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
        """Translate a mask by (dx, dy) final px; wraps in x on wrap canvases."""
        sx, sy = int(round(dx * self.ss)), int(round(dy * self.ss))
        out = np.zeros_like(mask)
        if sy >= 0:
            out[sy:] = mask[: self.H - sy] if sy else mask
        else:
            out[:sy] = mask[-sy:]
        if self.wrap:
            return np.roll(out, sx, axis=1)
        res = np.zeros_like(out)
        if sx >= 0:
            res[:, sx:] = out[:, : self.W - sx] if sx else out
        else:
            res[:, :sx] = out[:, -sx:]
        return res

    def grow(self, mask: np.ndarray, t: float) -> np.ndarray:
        """Coverage of ``mask`` dilated by ``t`` final px (uniform outline)."""
        pad = int(np.ceil(t * self.ss)) + 3
        m = mask
        if self.wrap:
            m = np.concatenate([mask[:, -pad:], mask, mask[:, :pad]], axis=1)
        outside = (m < 0.5).astype(np.uint8)
        dist = cv2.distanceTransform(outside, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        cov = np.clip(t * self.ss - dist + 0.5, 0.0, 1.0)
        if self.wrap:
            cov = cov[:, pad:-pad]
        return np.maximum(mask, cov)

    def edge_band(self, mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
        """Inner band of ``mask`` on the side facing (-dx, -dy)."""
        return mask * (1.0 - self.shift(mask, dx, dy))

    # -- painting -----------------------------------------------------------

    def paint(self, mask: np.ndarray, color, opacity: float = 1.0) -> None:
        m = mask * opacity if opacity != 1.0 else mask
        c = np.asarray(color, np.float32)
        if c.ndim == 1:
            c = c[None, None, :]
        self.pm *= 1.0 - m[..., None]
        self.pm += c * m[..., None]
        self.a *= 1.0 - m
        self.a += m

    def inked(self, mask: np.ndarray, fill, ink, t: float) -> None:
        """Paint ``mask`` with a bold outline of width ``t`` around it."""
        self.paint(self.grow(mask, t), ink)
        self.paint(mask, fill)

    # -- output -------------------------------------------------------------

    def _down(self, img: np.ndarray) -> np.ndarray:
        if self.ss == 1:
            return img
        return cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_AREA)

    def to_bgra(self, tag: str, dither: float = 0.0) -> np.ndarray:
        pm = self._down(self.pm)
        a = np.clip(self._down(self.a), 0.0, 1.0)
        # Unpremultiply; where alpha is tiny, bleed in the alpha-weighted local
        # colour so bilinear resampling never pulls black into an edge.
        num = _blur(pm, 1.5, self.wrap)
        den = _blur(a, 1.5, self.wrap)
        bleed = num / np.maximum(den, 1e-4)[..., None]
        direct = pm / np.maximum(a, 1e-4)[..., None]
        w = np.clip(a / 0.3, 0.0, 1.0)[..., None]
        col = np.clip(direct * w + bleed * (1.0 - w), 0.0, 255.0)
        a8 = np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8)
        if dither > 0:
            bgr = _dither_u8(col, tag, dither)
        else:
            bgr = np.clip(col + 0.5, 0, 255).astype(np.uint8)
        bgr[den < 1e-4] = 0  # far-away transparent pixels: constant (small PNG)
        return np.concatenate([bgr, a8[..., None]], axis=2)

    def to_bgr(self, tag: str, dither: float = 0.0) -> np.ndarray:
        pm = self._down(self.pm)
        if dither > 0:
            return _dither_u8(pm, tag, dither)
        return np.clip(pm + 0.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# the flying raccoon
# ---------------------------------------------------------------------------

# The hero is the same raccoon as Snack Attack (its head is drawn by
# tools/generate_fruitninja_scene.py, imported below), given a small tucked
# body, a ringed tail and feathered wings. Front-facing like a mascot, so the
# game only tilts it gently. Kept compact - big head, small body - so one
# collision circle can fit it fairly; wings and tail tips are not solid.

RACCOON_DEEP = rgb(104, 63, 155)  # mask purple, shared with Snack Attack
RACCOON_LAV = rgb(185, 165, 215)  # cheek-fluff lavender
RACCOON_LAV_DK = rgb(150, 125, 196)
RACCOON_WHITE = rgb(250, 248, 255)
WING_SHADE = rgb(212, 198, 240)
WING_TIP = rgb(150, 110, 210)

FLYER_W, FLYER_H = 128, 112
HEAD_SIZE = 66  # raccoon head canvas, px
HEAD_C = (64.0, 44.0)
BODY_C, BODY_R = (64.0, 76.0), (15.0, 13.0)
SHOULDERS = ((52.0, 70.0), (76.0, 70.0))  # left, right
# Right-wing angle per frame (degrees, screen space: 0 = right, - = up).
WING_ANGLES = (-46.0, -12.0, 22.0)  # up, mid, down (leading feather)
T_MAIN = 2.4


def _head(size: int = HEAD_SIZE) -> np.ndarray:
    """The Snack Attack raccoon head, rendered by its own generator."""
    import importlib.util

    path = ROOT / "tools" / "generate_fruitninja_scene.py"
    spec = importlib.util.spec_from_file_location("_fn_scene", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.make_raccoon(size=size, chomp=False)


def _dir(ang: float, mirror: bool) -> tuple[float, float]:
    t = np.deg2rad(180.0 - ang if mirror else ang)
    return float(np.cos(t)), float(np.sin(t))


def _draw_wing(c: "Canvas", shoulder, ang: float, mirror: bool) -> None:
    """A cartoon wing: four long primary feathers fanned out behind a rounded
    covert. Each feather is outlined on its own, so the gaps between them read
    as feathers rather than one blob."""
    sx, sy = shoulder
    lengths = (42.0, 39.0, 34.0, 28.0)  # leading edge -> trailing edge
    # Back to front, so the leading feather overlaps the ones behind it.
    for i in reversed(range(4)):
        fa = ang + 13.0 * i  # fan toward the trailing edge (down/back)
        ux, uy = _dir(fa, mirror)
        half = lengths[i] / 2.0
        cx, cy = sx + ux * (half + 3.0), sy + uy * (half + 3.0)
        feather = c.ellipse(cx, cy, half, 4.6, float(np.rad2deg(np.arctan2(uy, ux))))
        c.inked(feather, RACCOON_WHITE, INK, 1.9)
        # Lavender toward the tip, purple at the very end.
        tip1 = c.circle(sx + ux * lengths[i] * 0.86, sy + uy * lengths[i] * 0.86, lengths[i] * 0.34)
        c.paint(feather * tip1, WING_SHADE)
        tip2 = c.circle(sx + ux * (lengths[i] + 2.0), sy + uy * (lengths[i] + 2.0), 6.5)
        c.paint(feather * tip2, WING_TIP, 0.9)
    ux, uy = _dir(ang + 14.0, mirror)
    covert = c.ellipse(sx + ux * 9.0, sy + uy * 9.0, 12.0, 8.5,
                       float(np.rad2deg(np.arctan2(uy, ux))))
    c.inked(covert, RACCOON_WHITE, INK, T_MAIN)
    c.paint(covert * (1.0 - c.shift(covert, 0.0, -2.2)), WING_SHADE)


def _tail(c: "Canvas") -> tuple[np.ndarray, np.ndarray]:
    """Fluffy ringed tail trailing behind (to the left) and down."""
    p0, p1, p2 = np.array([52.0, 80.0]), np.array([36.0, 84.0]), np.array([20.0, 100.0])
    tail, rings = c.zeros(), c.zeros()
    n = 7
    for i in range(n):
        t = i / (n - 1)
        q = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t * t * p2
        r = 7.6 - 2.6 * t
        seg = c.ellipse(float(q[0]), float(q[1]), r, r * 0.92)
        tail = np.maximum(tail, seg)
        if i % 2 == 1 or i == n - 1:
            rings = np.maximum(rings, seg)
    return tail, rings


def make_raccoon_flyer(frame: int) -> np.ndarray:
    c = Canvas(FLYER_W, FLYER_H)
    ang = WING_ANGLES[frame]

    # Wings behind everything, the only part that changes between frames.
    for k, shoulder in enumerate(SHOULDERS):
        _draw_wing(c, shoulder, ang, mirror=(k == 0))

    tail, rings = _tail(c)
    c.inked(tail, RACCOON_LAV, INK, T_MAIN)
    c.paint(tail * rings, RACCOON_DEEP)

    bx, by = BODY_C
    body = c.ellipse(bx, by, *BODY_R)
    c.inked(body, RACCOON_LAV, INK, T_MAIN)
    c.paint(body * c.ellipse(bx, by + 3.5, 8.5, 8.0), RACCOON_WHITE)  # belly
    c.paint(body * (1.0 - c.ellipse(bx - 2.0, by - 2.5, 14.0, 12.0)), RACCOON_LAV_DK)
    for px in (bx - 7.0, bx + 7.0):  # tucked paws
        c.inked(c.ellipse(px, by + 12.0, 4.6, 3.6), RACCOON_DEEP, INK, 1.8)

    out = c.to_bgra(f"flyer{frame}")

    # The head goes on top, straight-alpha "over".
    head = _head()
    hh, hw = head.shape[:2]
    x0, y0 = int(round(HEAD_C[0] - hw / 2)), int(round(HEAD_C[1] - hh / 2))
    roi = out[y0:y0 + hh, x0:x0 + hw].astype(np.float32)
    src = head.astype(np.float32)
    sa, da = src[..., 3:4] / 255.0, roi[..., 3:4] / 255.0
    oa = sa + da * (1.0 - sa)
    rgb_ = np.where(oa > 1e-6, (src[..., :3] * sa + roi[..., :3] * da * (1.0 - sa)) / np.maximum(oa, 1e-6), 0.0)
    out[y0:y0 + hh, x0:x0 + hw] = np.concatenate([rgb_, oa * 255.0], axis=2).round().clip(0, 255).astype(np.uint8)
    return out

# ---------------------------------------------------------------------------
# pipes
# ---------------------------------------------------------------------------

# Banded horizontal profile, fractions of the inner width (lit from the left).
PIPE_BANDS = [
    (0.00, PIPE_BASE),
    (0.05, PIPE_LIGHT),
    (0.11, PIPE_HI),
    (0.21, PIPE_LIGHT),
    (0.29, PIPE_BASE),
    (0.34, PIPE_HI),
    (0.37, PIPE_BASE),
    (0.62, PIPE_SHADE1),
    (0.80, PIPE_SHADE2),
]


def _pipe_profile(w: int, ink: int, body_w: int = PIPE_W, body_ink: int = 5) -> np.ndarray:
    """One row of a pipe, w px wide, ``ink`` px outline each side (float BGR).

    Bands are positioned in *body* columns: a wider profile (the cap) is
    centred on the body and its overhang just continues the edge bands, so the
    highlight stripes on cap and body line up exactly.
    """
    row = np.zeros((w, 3), np.float32)
    off = (w - body_w) // 2
    inner = body_w - 2 * body_ink
    for x in range(w):
        if x < ink or x >= w - ink:
            row[x] = PIPE_INK
            continue
        f = min(max((x - off - body_ink + 0.5) / inner, 0.0), 0.999)
        col = PIPE_BANDS[0][1]
        for start, c in PIPE_BANDS:
            if f >= start:
                col = c
        row[x] = col
    return row


def make_pipe_body(h: int = 64) -> np.ndarray:
    row = _pipe_profile(PIPE_W, 5)
    bgr = np.clip(np.repeat(row[None], h, axis=0) + 0.5, 0, 255).astype(np.uint8)
    a = np.full((h, PIPE_W, 1), 255, np.uint8)
    return np.concatenate([bgr, a], axis=2)


def make_pipe_cap() -> np.ndarray:
    c = Canvas(CAP_W, CAP_H)
    ink = 5.0
    outer = c.rrect(0.0, 0.0, CAP_W, CAP_H, 7.0)
    inner = c.rrect(ink, ink, CAP_W - ink, CAP_H - ink, 3.0)
    c.paint(outer, PIPE_INK)
    # Banded fill from the same profile so cap and body read as one object.
    prof = _pipe_profile(CAP_W, int(ink))
    field = np.repeat(np.repeat(prof[None], CAP_H, axis=0), SS, axis=1)
    field = np.repeat(field, SS, axis=0)
    c.paint(inner, field)
    # Rim: bright lip along the top, dark underside where it meets the body.
    c.paint(inner * (1.0 - c.halfplane_y(ink + 4.0)), PIPE_HI, 0.55)
    c.paint(inner * c.halfplane_y(CAP_H - ink - 7.0), PIPE_SHADE2, 0.9)
    c.paint(inner * c.halfplane_y(CAP_H - ink - 7.0) * (1.0 - c.halfplane_y(CAP_H - ink - 5.5)),
            PIPE_INK, 0.5)
    return c.to_bgra("cap")


# ---------------------------------------------------------------------------
# sky
# ---------------------------------------------------------------------------

# (x, y, scale, opacity) - hand placed; lower clouds are fainter for depth.
CLOUDS = [
    (170, 120, 1.05, 1.00),
    (520, 64, 0.70, 0.95),
    (800, 190, 1.25, 1.00),
    (1150, 300, 0.85, 0.85),
    (350, 290, 0.70, 0.80),
    (640, 400, 0.95, 0.65),
    (60, 430, 0.75, 0.60),
    (1010, 470, 0.80, 0.55),
]
CLOUD_PUFFS = [(-52, 4, 26), (-22, -14, 34), (16, -20, 30), (46, -2, 26), (6, 4, 32), (68, 10, 18)]


def make_sky() -> np.ndarray:
    w, h = FRAME_W, FRAME_H
    ss = 2
    c = Canvas(w, h, ss=ss, wrap=True)
    full = np.ones((c.H, c.W), np.float32)
    ramp = _vramp(c.H, c.W, SKY_STOPS)
    c.paint(full, ramp)
    edge_col = ramp * 0.86  # cloud keyline: a deeper shade of the sky behind it

    # Sun, upper right: soft glow, two flat halo bands, a disc.
    sx, sy = 1040.0, 118.0
    dx = c.xs[None, :] - sx
    dx = (dx + w / 2.0) % w - w / 2.0
    dist = np.sqrt(dx ** 2 + (c.ys[:, None] - sy) ** 2)
    glow = np.exp(-(dist / 230.0) ** 2) * 0.30
    c.paint(glow, SUN_HALO)
    c.paint(c.circle(sx, sy, 104), SUN_HALO, 0.14)
    c.paint(c.circle(sx, sy, 74), SUN_HALO, 0.22)
    c.paint(c.circle(sx, sy, 46), SUN)

    for i, (cx, cy, s, op) in enumerate(CLOUDS):
        m = c.zeros()
        for (px, py, r) in CLOUD_PUFFS:
            m = np.maximum(m, c.circle(cx + px * s, cy + py * s, r * s))
        m *= 1.0 - c.halfplane_y(cy + 22 * s)  # flat bottom
        m = np.maximum(m, c.rrect(cx - 66 * s, cy + 4 * s, cx + 80 * s, cy + 22 * s, 9 * s) * 1.0)
        edge = c.grow(m, 2.0)
        c.paint(edge, edge_col, op * 0.8)
        c.paint(m, CLOUD, op)
        c.paint(c.edge_band(m, 0.0, -7.0 * s), CLOUD_SHADE, op)
    return c.to_bgr("sky", dither=0.9)


# ---------------------------------------------------------------------------
# skyline (mid layer, tiles horizontally)
# ---------------------------------------------------------------------------


def make_skyline() -> np.ndarray:
    w, h = FRAME_W, SKYLINE_H
    c = Canvas(w, h, wrap=True)
    r = _rng("skyline")

    # Far city: pale periwinkle blocks with a few lit windows.
    x = 0.0
    while x < w - 30:
        bw = float(r.integers(46, 92))
        bh = float(r.integers(78, 178))
        top = h - bh
        kind = int(r.integers(0, 4))
        m = c.rrect(x, top, x + bw, h + 10, 4.0)
        if kind == 1:  # dome
            m = np.maximum(m, c.ellipse(x + bw / 2, top, bw * 0.34, bw * 0.30))
        elif kind == 2:  # stepped tower
            m = np.maximum(m, c.rrect(x + bw * 0.25, top - 22, x + bw * 0.75, top + 4, 3.0))
            m = np.maximum(m, c.rrect(x + bw * 0.47, top - 40, x + bw * 0.53, top - 18, 1.0))
        elif kind == 3:  # slanted roof
            m = np.maximum(m, c.poly([(x, top + 1), (x + bw, top - 18), (x + bw, top + 1)]))
        c.inked(m, CITY, CITY_EDGE, 2.0)
        win = c.zeros()
        for wy in np.arange(top + 14, h - 30, 18.0):
            for wx in np.arange(x + 9, x + bw - 14, 15.0):
                if r.random() < 0.45:
                    win = np.maximum(win, c.rrect(wx, wy, wx + 7, wy + 9, 1.5))
        c.paint(win * m, CITY_WIN)
        x += bw + float(r.integers(-6, 14))

    # Hedge: two rows of rounded bushes, the back row bluer (atmosphere),
    # both lighter and cooler than the pipes so the pipes stay the hero.
    for row, (base_y, rmin, rmax, step, fill, light, edge) in enumerate((
        (h - 74, 22.0, 32.0, 32.0, BUSH_BACK, BUSH_BACK_LIGHT, BUSH_BACK_EDGE),
        (h - 46, 24.0, 36.0, 36.0, BUSH, BUSH_LIGHT, BUSH_EDGE),
    )):
        # Spacing < 2 * min radius, so neighbours always overlap and the base
        # slab below never shows a corner between two bushes.
        bush = c.rrect(-20, base_y + 2, w + 20, h + 20, 0.0)
        n = int(round(w / step))
        for i in range(n):
            bx = i * w / n + float(r.uniform(-4, 4))
            rad = float(r.uniform(rmin, rmax))
            by = base_y + float(r.uniform(-7, 7)) + 6.0 * np.sin(2 * np.pi * (bx / w * 2 + row * 0.3))
            bush = np.maximum(bush, c.circle(bx, by, rad))
        c.paint(c.grow(bush, 2.4), edge)
        c.paint(bush, fill)
        c.paint(c.edge_band(bush, 3.0, 6.0), light)
    return c.to_bgra("skyline", dither=0.6)


# ---------------------------------------------------------------------------
# ground (tiles horizontally)
# ---------------------------------------------------------------------------


def make_ground() -> np.ndarray:
    w, h = FRAME_W, GROUND_H
    c = Canvas(w, h, wrap=True)
    full = np.ones((c.H, c.W), np.float32)

    # Dirt with chunky diagonal stripes (period 40 divides 1280).
    c.paint(full, DIRT)
    X = c.xs[None, :]
    Y = c.ys[:, None]
    f = np.abs(((X + Y) % 40.0) - 20.0) - 10.0
    stripes = np.broadcast_to(c._cov(f / np.sqrt(2.0)), (c.H, c.W)).astype(np.float32)
    c.paint(stripes, DIRT_STRIPE)
    deep = c.halfplane_y(h - 14.0)
    c.paint(deep, DIRT_DEEP)
    c.paint(deep * stripes, DIRT_DEEP_STRIPE)
    c.paint(c.halfplane_y(h - 14.0) * (1.0 - c.halfplane_y(h - 12.0)), GROUND_INK, 0.35)

    # Pebbles.
    r = _rng("pebbles")
    for _ in range(26):
        px = float(r.uniform(0, w))
        py = float(r.uniform(42, h - 22))
        rx = float(r.uniform(3.0, 5.5))
        p = c.ellipse(px, py, rx, rx * 0.62)
        c.paint(p, PEBBLE)
        c.paint(p * c.ellipse(px - rx * 0.3, py - rx * 0.25, rx * 0.4, rx * 0.22), PEBBLE_HI)

    # Grass lip with scalloped tufts hanging over the dirt (period 40).
    grass = c.rrect(-20, -10, w + 20, 18.0, 0.0)
    for gx in np.arange(20.0, w + 1, 40.0):
        grass = np.maximum(grass, c.ellipse(gx, 18.0, 15.0, 8.5))
    c.paint(c.grow(grass, 2.6), GRASS_EDGE)
    c.paint(grass, GRASS)
    c.paint(c.edge_band(grass, 0.0, -3.0), GRASS_SH)
    c.paint(grass * (1.0 - c.halfplane_y(9.0)), GRASS_HI)
    # Blade ticks along the bright band.
    ticks = c.zeros()
    for tx in np.arange(8.0, w, 20.0):
        ticks = np.maximum(ticks, c.poly([(tx, 9.4), (tx + 3.0, 5.6), (tx + 4.4, 9.4)]))
    c.paint(ticks, GRASS)
    # Bold top edge where the ground meets the scenery.
    c.paint(1.0 - c.halfplane_y(3.5), GROUND_INK)
    return c.to_bgr("ground", dither=0.6)


# ---------------------------------------------------------------------------
# feather particle
# ---------------------------------------------------------------------------


def make_feather(size: int = 24) -> np.ndarray:
    c = Canvas(size, size)
    p0, p1, p2 = np.array([5.0, 20.0]), np.array([7.5, 7.0]), np.array([19.5, 3.8])
    ts = np.linspace(0.0, 1.0, 40)
    spine = [(1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t * t * p2 for t in ts]
    left, right = [], []
    for i, t in enumerate(ts):
        j0, j1 = max(0, i - 1), min(len(ts) - 1, i + 1)
        d = spine[j1] - spine[j0]
        d /= np.linalg.norm(d) + 1e-6
        n = np.array([-d[1], d[0]])
        wdt = 4.6 * np.sin(np.pi * min(1.0, t * 1.05)) ** 0.7 * (1.0 - 0.15 * t)
        notch = 0.45 if 0.5 < t < 0.6 else 1.0
        left.append(spine[i] + n * wdt * notch)
        right.append(spine[i] - n * wdt)
    vane = c.poly(left + right[::-1])
    c.paint(vane, rgb(255, 255, 255))
    c.paint(vane * c.edge_band(vane, -1.0, 1.0), rgb(232, 236, 244))
    quill = c.zeros()
    pts = [p0 + (p0 - spine[3]) * 0.9] + spine[:34]
    for a, b in zip(pts[:-1], pts[1:]):
        dd = b - a
        nn = np.array([-dd[1], dd[0]]) / (np.linalg.norm(dd) + 1e-6) * 0.5
        quill = np.maximum(quill, c.poly([a + nn, b + nn, b - nn, a - nn]))
    c.paint(quill, rgb(208, 214, 226))
    return c.to_bgra("feather")


# ---------------------------------------------------------------------------
# medals
# ---------------------------------------------------------------------------


def _star(cx: float, cy: float, ro: float, ri: float, rot: float = -90.0):
    pts = []
    for k in range(10):
        a = np.deg2rad(rot + k * 36.0)
        r = ro if k % 2 == 0 else ri
        pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    return pts


def make_medal(kind: str, size: int = 88) -> np.ndarray:
    dark, shade, base, light, hi = MEDALS[kind]
    c = Canvas(size, size)
    cx = cy = size / 2.0

    # Scalloped rim (12 rounded lobes).
    X = c.xs[None, :] - cx
    Y = c.ys[:, None] - cy
    rad = np.sqrt(X ** 2 + Y ** 2)
    th = np.arctan2(Y, X)
    R, amp, n = 36.0, 2.6, 12
    rr = R + amp * np.cos(n * th)
    slope = np.sqrt(1.0 + (amp * n / np.maximum(rad, 1.0)) ** 2)
    rim = c._cov((rad - rr) / slope)

    # Soft drop shadow so it sits on the panel.
    c.paint(_blur(c.shift(rim, 1.5, 3.0), 2.0 * SS), rgb(40, 20, 10), 0.35)
    c.inked(rim, base, dark, 3.0)
    lit = c.circle(cx - 3.0, cy - 3.5, 37.0)
    c.paint(rim * (1.0 - lit), shade)
    c.paint(rim * c.ellipse(cx - 12, cy - 16, 16, 9, -40), light)

    # Recessed inner disc: inverted lighting.
    disc = c.circle(cx, cy, 25.0)
    c.paint(c.grow(disc, 2.0), dark, 0.85)
    c.paint(disc, shade)
    c.paint(disc * c.circle(cx + 1.6, cy + 2.0, 24.2), base)

    # Embossed star.
    star = c.poly(_star(cx, cy + 0.5, 17.0, 7.4))
    c.inked(star, light, dark, 1.8)
    left = c.poly([(cx, cy + 0.5)] + _star(cx, cy + 0.5, 17.0, 7.4)[5:] + [(cx, cy - 16.5)])
    c.paint(star * left, hi, 0.9)
    c.paint(star * (1.0 - c.shift(star, 0, -1.6)), shade, 0.8)

    if kind == "platinum":
        gdk, gmid, ghi = PLATINUM_GEM
        gem = c.poly([(cx, cy - 5.5), (cx + 5.0, cy + 0.5), (cx, cy + 6.5), (cx - 5.0, cy + 0.5)])
        c.inked(gem, gmid, dark, 1.2)
        c.paint(gem * c.poly([(cx, cy - 5.5), (cx, cy + 0.5), (cx - 5.0, cy + 0.5)]), ghi)
        c.paint(gem * c.poly([(cx, cy + 6.5), (cx, cy + 0.5), (cx + 5.0, cy + 0.5)]), gdk)

    # Glint sparkle(s).
    def sparkle(x, y, s):
        m = np.maximum(c.ellipse(x, y, s, s * 0.22), c.ellipse(x, y, s * 0.22, s))
        c.paint(m, hi)

    sparkle(cx - 19.0, cy - 18.0, 5.0)
    if kind in ("gold", "platinum"):
        sparkle(cx + 22.0, cy + 15.0, 3.6)
    if kind == "platinum":
        sparkle(cx + 18.0, cy - 25.0, 3.0)
    return c.to_bgra(f"medal_{kind}")


# ---------------------------------------------------------------------------
# game-over panel
# ---------------------------------------------------------------------------


def make_panel(w: int = 520, h: int = 300) -> np.ndarray:
    c = Canvas(w, h)
    x0, y0, x1, y1 = 6.0, 4.0, w - 6.0, h - 12.0
    outer = c.rrect(x0, y0, x1, y1, 28.0)
    # Chunky drop shadow.
    c.paint(_blur(c.shift(outer, 0.0, 8.0), 3.0 * SS), rgb(20, 12, 30), 0.38)
    c.inked(outer, PANEL_FACE, PANEL_INK, 6.0)
    c.paint(c.edge_band(outer, 0.0, -9.0), PANEL_SH)
    c.paint(c.edge_band(outer, 0.0, 6.0), PANEL_HI)

    # Recessed well for SCORE / BEST / medal.
    well = c.rrect(x0 + 26, y0 + 26, x1 - 26, y1 - 30, 16.0)
    c.paint(c.grow(well, 3.0), PANEL_INK, 0.85)
    c.paint(well, PANEL_WELL)
    c.paint(c.edge_band(well, 0.0, 6.0), PANEL_WELL_SH)
    c.paint(c.edge_band(well, 0.0, -3.0), PANEL_HI, 0.6)

    # Rivets in the four corners of the frame.
    for (rx, ry) in ((x0 + 16, y0 + 16), (x1 - 16, y0 + 16), (x0 + 16, y1 - 18), (x1 - 16, y1 - 18)):
        rv = c.circle(rx, ry, 5.0)
        c.inked(rv, PANEL_RIVET, PANEL_INK, 2.0)
        c.paint(c.circle(rx - 1.5, ry - 1.5, 1.6), PANEL_HI)
    return c.to_bgra("panel", dither=0.5)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def build() -> dict[str, np.ndarray]:
    assets: dict[str, np.ndarray] = {}
    for i in range(3):
        assets[f"bird_{i}.png"] = make_raccoon_flyer(i)
    assets["pipe_body.png"] = make_pipe_body()
    assets["pipe_cap.png"] = make_pipe_cap()
    assets["sky.png"] = make_sky()
    assets["skyline.png"] = make_skyline()
    assets["ground.png"] = make_ground()
    assets["feather.png"] = make_feather()
    for m in ("bronze", "silver", "gold", "platinum"):
        assets[f"medal_{m}.png"] = make_medal(m)
    assets["panel.png"] = make_panel()
    return assets


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Flappy game art.")
    ap.add_argument("--out", type=Path, default=OUT_DIR, help="output directory")
    args = ap.parse_args()
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, str, str, float]] = []
    for name, img in build().items():
        path = out_dir / name
        if not cv2.imwrite(str(path), img):
            raise RuntimeError(f"failed to write {path}")
        hh, ww = img.shape[:2]
        fmt = "BGRA" if img.ndim == 3 and img.shape[2] == 4 else "BGR"
        rows.append((name, f"{ww}x{hh}", fmt, path.stat().st_size / 1024.0))

    manifest = {"bird_frames": 3, "bird_hit_radius": BIRD_HIT_RADIUS, "ground_height": GROUND_H}
    mpath = out_dir / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2) + "\n")
    rows.append(("manifest.json", "-", "json", mpath.stat().st_size / 1024.0))

    width = max(len(r[0]) for r in rows)
    print(f"\nwrote {len(rows)} files to {out_dir}\n")
    print(f"{'name'.ljust(width)}  {'size':>10}  {'fmt':>4}  {'KB':>8}")
    print("-" * (width + 28))
    for name, size, fmt, kb in rows:
        print(f"{name.ljust(width)}  {size:>10}  {fmt:>4}  {kb:>8.1f}")
    print("-" * (width + 28))
    print(f"{'total'.ljust(width)}  {'':>10}  {'':>4}  {sum(r[3] for r in rows):>8.1f}\n")


if __name__ == "__main__":
    main()
