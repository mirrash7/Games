#!/usr/bin/env python3
"""Procedurally generate the stage, coach and UI art for Raccoon Dance.

Run once (or whenever the look changes):

    uv run python tools/generate_dance_assets.py

numpy + OpenCV only: no downloads, no fonts beyond OpenCV's Hershey set, no
binary sources. Output goes to ``assets/dance/generated/`` with a
``manifest.json`` that carries the numbers that belong to the art (outfit and
rating colours, where the stage's lights hang, where the coach stands).

Look target: a dark neon club stage. A deep navy LED wall, a lighting truss
with can lights, haze, a neon ring behind the dancer, a cyan tube where the
wall meets a glossy floor with a perspective grid, and a spotlight pool where
the coach stands. The coach is the arcade's purple raccoon (the same head as
Snack Attack's, from ``generate_fruitninja_scene.make_raccoon``) wearing
headphones; its body is drawn live by the renderer from the outfit colours here.

Technique notes (shared with the other generators):

* Everything is composed in float32 and quantised once with a triangular
  dither (`_to_u8`): a dark 720p gradient bands badly without it.
* Sprites are drawn supersampled and box-filtered down, premultiplied, so soft
  edges never pick up a dark fringe.
* Outlines come from a distance transform: exact round keylines, never a
  per-frame `cv2.dilate`.
* Sprites the renderer tints (beams, glows, sparkles, the ring, the head rim)
  are pure white with the shape carried by alpha.
* Light layers that are only ever added on top (trim glow, floor glow, the
  gold edge glow) are stored at quarter resolution: they are soft, and the
  renderer composites all of them in one quarter-resolution buffer.
* Deterministic: one seed, no clock. Two runs are byte-identical (md5).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import cv2
import numpy as np

SEED = 20261014
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "assets" / "dance" / "generated"
W, H = 1280, 720
QUARTER = 4  # light layers are stored at 1/4 resolution (320 x 180)

HORIZON = 470  # where the back wall meets the floor
EYE_Y = 300  # perspective vanishing height for the floor grid
COACH_X, COACH_FEET = 520, 660
SPOT = (520, 652, 250, 56)  # spotlight pool: centre x, y, radius x, y
LIGHTS = [170, 345, 520, 695, 870]  # can lights on the truss (x); the PiP hides x > 940
LENS_Y = 94
RING = (520, 292, 188)  # neon ring behind the coach: centre x, y, radius


def rgb(r, g, b) -> tuple[float, float, float]:
    """Author colours in RGB, store BGR (OpenCV order)."""
    return (float(b), float(g), float(r))


INK = (46.0, 24.0, 34.0)  # the arcade's sticker keyline (same as the raccoon's)
MAGENTA = rgb(255, 50, 190)
CYAN_NEON = rgb(70, 235, 255)
VIOLET = rgb(150, 90, 255)
GOLD = rgb(255, 205, 70)

# The coach's outfits, keyed by song.json "coach": {"outfit": ...}. BGR.
OUTFITS = {
    "neon": {
        "fur": rgb(160, 118, 222),  # arms and neck: the raccoon's lavender, a step deeper
        "fur_dark": rgb(92, 58, 150),  # tail rings
        "top": rgb(255, 46, 170),  # magenta crop jacket
        "top_hi": rgb(255, 150, 215),
        "pants": rgb(84, 48, 168),  # violet
        "accent": rgb(70, 235, 255),  # cyan: cuffs, belt, stripes, soles
        "glove": rgb(250, 246, 255),
        "shoe": rgb(250, 248, 255),
        "ink": INK,
        "rim": rgb(110, 245, 255),  # crisp neon line around the silhouette
        "glow": rgb(200, 70, 255),  # soft halo behind it
    },
    # "Pumpkin Patch Bounce": pumpkin-orange jacket, deep plum trousers, a
    # lime-green stem accent, a warm orange rim. The fur and gloves stay the
    # raccoon's own, and the headphones (baked into the head sprite) keep neon's.
    "pumpkin": {
        "fur": rgb(160, 118, 222),
        "fur_dark": rgb(92, 58, 150),
        "top": rgb(255, 128, 24),  # pumpkin orange
        "top_hi": rgb(255, 196, 110),
        "pants": rgb(58, 30, 84),  # deep plum, near black
        "accent": rgb(150, 236, 60),  # stem green: cuffs, belt, stripes, soles
        "glove": rgb(250, 246, 255),
        "shoe": rgb(250, 248, 255),
        "ink": INK,
        "rim": rgb(255, 196, 90),  # warm candle-light line
        "glow": rgb(255, 110, 20),  # orange halo
    },
}

# Rating words: fill colour (BGR) and sprite. PERFECT gold, SUPER cyan, GOOD
# green, OK violet, X red: the arcade theme's own meanings (theme.py).
RATINGS = {
    "perfect": {"text": "PERFECT", "colour": rgb(255, 210, 70)},
    "super": {"text": "SUPER", "colour": rgb(110, 220, 255)},
    "good": {"text": "GOOD", "colour": rgb(150, 235, 140)},
    "ok": {"text": "OK", "colour": rgb(200, 150, 255)},
    "x": {"text": "X", "colour": rgb(255, 70, 70)},
}
WORDS = {  # other pre-rendered words: name -> (text, colour, cap height px)
    "get_ready": ("GET READY", rgb(110, 220, 255), 50),
    "count_3": ("3", rgb(255, 80, 200), 120),
    "count_2": ("2", rgb(255, 80, 200), 120),
    "count_1": ("1", rgb(255, 80, 200), 120),
    "count_go": ("DANCE!", rgb(255, 210, 70), 84),
    "superstar": ("SUPERSTAR", rgb(255, 210, 70), 40),
}
RATING_CAP = 52  # cap height of the rating words in px


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_scene():
    """Snack Attack's scene generator: the raccoon head and its drawing helpers."""
    path = ROOT / "tools" / "generate_fruitninja_scene.py"
    spec = importlib.util.spec_from_file_location("_kp_fruitninja_scene", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rng(stream: int) -> np.random.Generator:
    return np.random.default_rng(SEED + stream * 7919)


def _to_u8(arr: np.ndarray, dither: float = 0.8, stream: int = 0) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    if dither > 0.0:
        r = _rng(stream + 101)
        a = a + (r.random(a.shape, dtype=np.float32) - r.random(a.shape, dtype=np.float32)) * dither
    return np.clip(a + 0.5, 0, 255).astype(np.uint8)


def _smooth(t) -> np.ndarray:
    t = np.clip(np.asarray(t, dtype=np.float32), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)


def _over(dst: np.ndarray, colour, alpha: np.ndarray) -> np.ndarray:
    a = alpha if alpha.ndim == 3 else alpha[..., None]
    return dst * (1.0 - a) + np.asarray(colour, np.float32) * a


def _add(dst: np.ndarray, colour, amount: np.ndarray) -> np.ndarray:
    a = amount if amount.ndim == 3 else amount[..., None]
    return dst + np.asarray(colour, np.float32) * a


def _ring(mask: np.ndarray, width: float) -> np.ndarray:
    """Anti-aliased band `width` px wide grown outward from a 0..1 mask."""
    outside = (mask < 0.5).astype(np.uint8)
    dist = cv2.distanceTransform(outside, cv2.DIST_L2, 5)
    return np.clip(width - dist + 0.5, 0.0, 1.0)


def _inside_dist(mask: np.ndarray) -> np.ndarray:
    return cv2.distanceTransform((mask >= 0.5).astype(np.uint8), cv2.DIST_L2, 5)


def _bgra(rgb_f: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return np.concatenate([np.clip(rgb_f, 0, 255), np.clip(alpha, 0, 1)[..., None] * 255.0], axis=2)


def _down(bgra: np.ndarray, ss: int) -> np.ndarray:
    """Premultiplied box-filter downscale of a float BGRA (alpha 0..255) buffer."""
    if ss == 1:
        return bgra
    a = bgra[..., 3:4] / 255.0
    pm = np.concatenate([bgra[..., :3] * a, bgra[..., 3:4]], axis=2)
    h, w = pm.shape[:2]
    small = cv2.resize(pm, (w // ss, h // ss), interpolation=cv2.INTER_AREA)
    sa = np.maximum(small[..., 3:4] / 255.0, 1e-4)
    return np.concatenate([np.clip(small[..., :3] / sa, 0, 255), small[..., 3:4]], axis=2)


def _white(alpha: np.ndarray) -> np.ndarray:
    """A tintable sprite: white, shape in alpha (0..1 float)."""
    out = np.empty((*alpha.shape, 4), np.float32)
    out[..., :3] = 255.0
    out[..., 3] = np.clip(alpha, 0, 1) * 255.0
    return out


def _vramp(h: int, w: int, stops) -> np.ndarray:
    ts = np.array([s[0] for s in stops], np.float32)
    cols = np.array([s[1] for s in stops], np.float32)
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)
    ramp = np.stack([np.interp(y, ts, cols[:, c]) for c in range(3)], axis=1).astype(np.float32)
    return np.repeat(ramp[:, None, :], w, axis=1)


def _radial(h, w, cx, cy, rx, ry) -> np.ndarray:
    ys = (np.arange(h, dtype=np.float32) + 0.5 - cy) / max(ry, 1e-3)
    xs = (np.arange(w, dtype=np.float32) + 0.5 - cx) / max(rx, 1e-3)
    return np.sqrt(xs[None, :] ** 2 + ys[:, None] ** 2)


def _noise(h, w, sigma, stream) -> np.ndarray:
    n = _blur(_rng(stream).standard_normal((h, w)).astype(np.float32), sigma)
    return n / max(float(n.std()), 1e-6)


def _line_mask(h, w, pts, thick, ss=4) -> np.ndarray:
    """An anti-aliased polyline mask (0..1), drawn with sub-pixel precision."""
    m = np.zeros((h, w), np.float32)
    arr = np.round(np.asarray(pts, np.float64) * 16).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(m, [arr], False, 1.0, max(1, int(round(thick))), cv2.LINE_AA, 4)
    return m


def _neon(h, w, mask: np.ndarray, colour, core=0.9, glow=(6.0, 22.0), gains=(0.9, 0.45)):
    """Light from a neon tube mask: a white-hot core, the tube colour, and two glows."""
    col = np.asarray(colour, np.float32)
    light = np.zeros((h, w, 3), np.float32)
    for sigma, g in zip(glow, gains):
        light = _add(light, col, _blur(mask, sigma) * g * 2.2)
    light = _add(light, col, mask * 0.9)
    light = _add(light, (255.0, 255.0, 255.0), _blur(mask, 0.7) ** 2 * core * 0.8)
    return light


# ---------------------------------------------------------------------------
# the stage
# ---------------------------------------------------------------------------

def _tube_masks():
    """The neon tubes, as masks at full resolution: (mask, colour) pairs."""
    floor_tube = _line_mask(H, W, [(0, HORIZON - 3), (W, HORIZON - 3)], 4)
    cx, cy, r = RING
    t = np.radians(np.linspace(0, 360, 361))
    ring_pts = np.stack([cx + np.cos(t) * r, cy + np.sin(t) * r], axis=1)
    ring = _line_mask(H, W, ring_pts, 5)
    ring[HORIZON - 6:] = 0.0  # the floor hides the bottom of the ring
    # Two slanted side tubes frame the stage like wings.
    side = np.maximum(_line_mask(H, W, [(150, 150), (226, HORIZON - 30)], 4),
                      _line_mask(H, W, [(890, 150), (814, HORIZON - 30)], 4))
    return [(floor_tube, CYAN_NEON), (ring, MAGENTA), (side, VIOLET)]


def make_stage() -> tuple[np.ndarray, np.ndarray]:
    """The 1280x720 stage (BGR) and its quarter-res trim-glow layer (BGR)."""
    h, w = H, W
    hz = HORIZON / h
    img = _vramp(h, w, [(0.0, rgb(6, 6, 18)), (0.25, rgb(10, 9, 30)), (hz - 0.002, rgb(26, 16, 58)),
                        (hz, rgb(16, 10, 34)), (0.82, rgb(10, 7, 22)), (1.0, rgb(5, 4, 12))])

    # --- LED wall: rounded tiles, lit by a slow field brightest behind the coach.
    tile, gap = 44, 4
    ys, xs = np.mgrid[0:HORIZON, 0:w].astype(np.float32)
    u = (xs + 10) % tile
    v = (ys + 6) % tile
    edge = np.minimum(np.minimum(u, tile - gap - u), np.minimum(v, tile - gap - v))
    tiles = _smooth((edge + 0.5) / 2.0) * ((u < tile - gap) & (v < tile - gap))
    field = np.clip(1.0 - _radial(HORIZON, w, COACH_X, 300, 560, 330), 0, 1) ** 1.4
    field = field * (0.75 + 0.25 * _noise(HORIZON, w, 60, 1)) + 0.12
    # LED dots: a faint dot matrix inside each tile reads as "screen", not "bricks".
    dots = (((u % 11) - 5.0) ** 2 + ((v % 11) - 5.0) ** 2 < 4.5).astype(np.float32)
    dots = _blur(dots, 0.6)
    wall_light = tiles * (0.55 + 0.45 * dots) * np.clip(field, 0, 1.2)
    img[:HORIZON] = _add(img[:HORIZON], rgb(52, 34, 110), wall_light * 0.55)

    # --- back light: a broad violet glow behind the dancer.
    back = np.clip(1.0 - _radial(h, w, COACH_X, 320, 470, 300), 0, 1) ** 2
    back[HORIZON:] *= 0.0
    img = _add(img, rgb(90, 36, 160), back * 0.55)

    # --- haze: soft drifting fog, thicker low on the wall and up in the rig.
    haze = np.clip(_noise(h, w, 46, 2) * 0.5 + 0.5, 0, 1)
    prof = np.interp(np.arange(h), [0, 120, 300, HORIZON, HORIZON + 40, h], [0.8, 0.5, 0.35, 0.8, 0.3, 0.0])
    img = _add(img, rgb(70, 50, 130), haze * prof[:, None].astype(np.float32) * 0.25)

    # --- neon tubes (and a quarter-res copy of their glow for beat pulses).
    tubes = _tube_masks()
    pulse = np.zeros((h, w, 3), np.float32)
    for k, (mask, colour) in enumerate(tubes):
        # The ring sits right behind the coach: dimmer, so the coach's own glow wins.
        img += _neon(h, w, mask, colour) * (0.55 if k == 1 else 1.0)
        pulse = _add(pulse, colour, _blur(mask, 14.0) * 3.0 + _blur(mask, 3.0) * 0.8)

    # --- the spotlight: a faint cone from the centre light down to its pool.
    sx, sy, srx, sry = SPOT
    yy = np.arange(h, dtype=np.float32)[:, None]
    xx = np.arange(w, dtype=np.float32)[None, :]
    tcone = np.clip((yy - LENS_Y) / (sy - LENS_Y), 0, 1)
    half = 18 + (srx * 0.92 - 18) * tcone
    cone = _smooth(1.0 - np.abs(xx - sx) / half) * (yy > LENS_Y) * (yy < sy + 10)
    cone *= 0.55 + 0.45 * np.clip(_noise(h, w, 30, 3) * 0.5 + 0.5, 0, 1)
    img = _add(img, rgb(150, 120, 230), cone * 0.11)

    # --- the floor: glossy, reflecting the wall, with a perspective grid.
    fh = h - HORIZON
    wall = img[:HORIZON].copy()
    refl = wall[::-1][:fh]
    refl = cv2.GaussianBlur(refl, (0, 0), sigmaX=3, sigmaY=10)
    fall = np.exp(-np.arange(fh, dtype=np.float32) / 120.0)[:, None, None]
    img[HORIZON:] += refl * fall * 0.32

    grid = np.zeros((fh, w), np.float32)
    for k in range(-9, 10):  # receding lines converge on the eye point
        xb = COACH_X + k * 150
        x_at_h = COACH_X + (xb - COACH_X) * (HORIZON - EYE_Y) / (h - EYE_Y)
        grid = np.maximum(grid, _line_mask(fh, w, [(x_at_h, 0), (xb, fh)], 2))
    for z in (0.88, 0.76, 0.64, 0.52, 0.40):  # cross lines at equal depth steps
        y = EYE_Y + (HORIZON - EYE_Y) / z - HORIZON
        grid = np.maximum(grid, _line_mask(fh, w, [(0, y), (w, y)], 2))
    depth = np.clip(np.arange(fh, dtype=np.float32) / 150.0, 0.25, 1.0)[:, None]
    side_fade = np.clip(1.0 - np.abs(np.arange(w, dtype=np.float32) - COACH_X) / 760.0, 0.15, 1)[None, :]
    g = grid * depth * side_fade
    img[HORIZON:] = _add(img[HORIZON:], MAGENTA, _blur(g, 4.0) * 0.55 + g * 0.30)

    # --- the spotlight pool where the coach stands.
    pool_d = _radial(h, w, sx, sy, srx, sry)
    pool = _smooth((1.0 - pool_d) / 0.55) * 0.55 + np.exp(-((pool_d - 0.97) / 0.06) ** 2) * 0.35
    pool *= (yy > HORIZON)
    img = _add(img, rgb(175, 150, 255), pool * 0.62)

    # --- the light rig: truss, then the can lights hanging from it.
    img = _draw_truss(img)
    for i, x in enumerate(LIGHTS):
        img = _draw_can(img, x, [MAGENTA, CYAN_NEON, rgb(255, 245, 235), CYAN_NEON, MAGENTA][i])

    # --- vignette: darker corners keep the HUD and the shell's footer readable.
    vig = _radial(h, w, w * 0.45, h * 0.48, w * 0.70, h * 0.78)
    img *= (1.0 - 0.55 * _smooth((vig - 0.55) / 0.6))[..., None]

    stage = _to_u8(img, dither=1.0, stream=11)
    small = cv2.resize(pulse, (w // QUARTER, h // QUARTER), interpolation=cv2.INTER_AREA)
    return stage, _to_u8(small, dither=0.6, stream=12)


def _draw_truss(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    top, bot = 16, 52
    mask = np.zeros((h, w), np.float32)
    for y in (top, bot):
        cv2.rectangle(mask, (0, y - 3), (w, y + 3), 1.0, -1)
    pts = []
    for i, x in enumerate(range(-20, w + 40, 36)):
        pts.append((x, top if i % 2 == 0 else bot))
    mask = np.maximum(mask, _line_mask(h, w, pts, 3))
    mask = _blur(mask, 0.6)
    shade = np.interp(np.arange(h), [top - 4, top, bot, bot + 4], [1.0, 0.7, 0.45, 0.3])[:, None]
    metal = np.zeros_like(img)
    metal[:] = rgb(64, 60, 88)
    metal *= shade[..., None].astype(np.float32)
    # A specular glint along the top rail.
    glint = np.zeros((h, w), np.float32)
    cv2.line(glint, (0, top - 2), (w, top - 2), 1.0, 1, cv2.LINE_AA)
    img = _over(img, (0, 0, 0), _blur(mask, 3.0) * 0.6)  # soft shadow first
    img = _over(img, metal, mask[..., None] * np.ones((1, 1, 3), np.float32))
    return _add(img, rgb(160, 150, 200), glint * mask * 0.6)


def _draw_can(img: np.ndarray, x: int, colour) -> np.ndarray:
    """A can light: yoke, housing and a glowing lens pointing down."""
    h, w = img.shape[:2]
    body = np.zeros((h, w), np.float32)
    cv2.rectangle(body, (x - 17, 56), (x + 17, LENS_Y - 2), 1.0, -1, cv2.LINE_AA)
    cv2.ellipse(body, (x, LENS_Y - 2), (17, 6), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    yoke = _line_mask(h, w, [(x - 24, 70), (x - 24, 52), (x + 24, 52), (x + 24, 70)], 4)
    housing = np.zeros_like(img)
    xs = np.arange(w, dtype=np.float32)
    lit = np.clip(1.0 - np.abs(xs - (x - 6)) / 20.0, 0, 1)[None, :]  # rounded highlight
    housing[:] = rgb(26, 24, 40)
    housing = _add(housing, rgb(80, 74, 110), np.broadcast_to(lit, (h, w)) * 0.8)
    img = _over(img, rgb(40, 38, 58), yoke)
    img = _over(img, housing, body[..., None] * np.ones((1, 1, 3), np.float32))
    lens = np.zeros((h, w), np.float32)
    cv2.ellipse(lens, (x, LENS_Y), (13, 5), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    img += _neon(h, w, lens, colour, core=1.4, glow=(5.0, 16.0), gains=(1.0, 0.5))
    return img


def make_floor_glow() -> np.ndarray:
    """Quarter-res grey light (BGRA white): the pool and grid flaring on the beat."""
    h, w = H // QUARTER, W // QUARTER
    sx, sy, srx, sry = SPOT
    d = _radial(h, w, sx / QUARTER, sy / QUARTER, srx * 1.25 / QUARTER, sry * 1.6 / QUARTER)
    glow = np.exp(-d * d * 2.2) * 0.85 + np.exp(-((d - 0.78) / 0.12) ** 2) * 0.35
    glow[: HORIZON // QUARTER] *= 0.0
    # A faint wash up the wall behind the coach, as the floor bounces light.
    up = np.clip(1.0 - _radial(h, w, sx / QUARTER, HORIZON / QUARTER, 90, 34), 0, 1) ** 2
    glow = np.maximum(glow, up * 0.35)
    return _to_u8(_white(np.clip(glow, 0, 1)), dither=0.0)


def make_edge_glow() -> np.ndarray:
    """Quarter-res screen-edge glow (BGRA white), for gold moves."""
    h, w = H // QUARTER, W // QUARTER
    ys = np.arange(h, dtype=np.float32)[:, None]
    xs = np.arange(w, dtype=np.float32)[None, :]
    d = np.minimum(np.minimum(xs, w - 1 - xs), np.minimum(ys, h - 1 - ys))
    a = np.exp(-d / 9.0) * 0.95 + np.exp(-d / 28.0) * 0.25
    return _to_u8(_white(np.clip(a, 0, 1)), dither=0.0)


def make_beam() -> np.ndarray:
    """A light beam (BGRA white): narrow at the lens, wide and faint at the floor."""
    bw, bh = 260, 640
    ys = np.arange(bh, dtype=np.float32)[:, None]
    xs = np.arange(bw, dtype=np.float32)[None, :] - bw / 2
    half = 9.0 + ys * (bw * 0.42 - 9.0) / bh
    across = np.clip(1.0 - np.abs(xs) / half, 0, 1)
    core = across ** 1.6
    a = core * (0.95 * np.exp(-ys / 520.0) + 0.05)
    a *= np.clip(ys / 10.0, 0, 1)
    texture = 0.75 + 0.25 * np.clip(_noise(bh, bw, 9, 21) * 0.5 + 0.5, 0, 1)  # haze in the beam
    return _to_u8(_white(np.clip(a * texture, 0, 1)), dither=0.0)


# ---------------------------------------------------------------------------
# small sprites
# ---------------------------------------------------------------------------

def make_glow(size: int = 64) -> np.ndarray:
    d = _radial(size, size, size / 2, size / 2, size / 2, size / 2)
    return _to_u8(_white(np.exp(-d * d * 4.0) * (d < 1)), dither=0.0)


def make_sparkle(size: int = 72) -> np.ndarray:
    """A four-point twinkle: long thin rays, short diagonals, a hot core."""
    ss = 4
    n = size * ss
    c = n / 2
    ys, xs = np.mgrid[0:n, 0:n].astype(np.float32)
    dx, dy = np.abs(xs - c) / c, np.abs(ys - c) / c
    ray = np.exp(-(dy / 0.035) ** 2) * np.clip(1 - dx, 0, 1) ** 1.5
    ray = np.maximum(ray, np.exp(-(dx / 0.035) ** 2) * np.clip(1 - dy, 0, 1) ** 1.5)
    u, v = (dx + dy) / np.sqrt(2), np.abs(dx - dy) / np.sqrt(2)
    diag = np.exp(-(v / 0.03) ** 2) * np.clip(1 - u / 0.5, 0, 1) ** 2 * 0.7
    r = np.sqrt(dx * dx + dy * dy)
    a = np.clip(np.maximum(ray, diag) + np.exp(-(r / 0.16) ** 2), 0, 1)
    return _to_u8(_down(_white(a), ss), dither=0.0)


def make_burst(size: int = 420) -> np.ndarray:
    """Radial rays for gold moves (BGRA white): 18 rays, alternating widths."""
    c = size / 2
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    ang = np.arctan2(ys - c, xs - c)
    r = np.sqrt((xs - c) ** 2 + (ys - c) ** 2) / c
    k = 18
    phase = (ang * k / (2 * np.pi)) % 1.0
    wide = (np.arange(size)[None, :] * 0 + 0)  # placeholder for clarity
    del wide
    ray = np.where(np.floor(ang * k / (2 * np.pi)) % 2 == 0,
                   _smooth(1 - np.abs(phase - 0.5) / 0.32), _smooth(1 - np.abs(phase - 0.5) / 0.18) * 0.7)
    fade = np.clip(1 - r, 0, 1) ** 1.3 * _smooth(r / 0.18)
    a = ray * fade * 0.9 + np.exp(-(r / 0.22) ** 2) * 0.6
    return _to_u8(_white(np.clip(a, 0, 1)), dither=0.0)


def make_ring(size: int = 140) -> np.ndarray:
    """The pictogram target ring (BGRA white): a crisp ring, soft glow, four ticks."""
    ss = 4
    n = size * ss
    c = n / 2
    r = _radial(n, n, c, c, c, c)
    line = np.exp(-((r - 0.80) / 0.028) ** 2)
    glow = np.exp(-((r - 0.80) / 0.13) ** 2) * 0.35
    ticks = np.zeros((n, n), np.float32)
    for ang in (0, 90, 180, 270):
        a = np.radians(ang)
        p0 = (c + np.cos(a) * c * 0.86, c + np.sin(a) * c * 0.86)
        p1 = (c + np.cos(a) * c * 0.98, c + np.sin(a) * c * 0.98)
        cv2.line(ticks, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), 1.0, int(n * 0.03), cv2.LINE_AA)
    a = np.clip(np.maximum(np.maximum(line, glow), ticks), 0, 1)
    return _to_u8(_down(_white(a), ss), dither=0.0)


def _star_mask(n: int, r_out: float, r_in: float, round_px: float) -> np.ndarray:
    c = n / 2
    pts = []
    for i in range(10):
        a = -np.pi / 2 + i * np.pi / 5
        r = r_out if i % 2 == 0 else r_in
        pts.append((c + np.cos(a) * r * n, c + 0.04 * n + np.sin(a) * r * n))
    m = np.zeros((n, n), np.float32)
    cv2.fillPoly(m, [np.round(np.array(pts) * 16).astype(np.int32)], 1.0, cv2.LINE_AA, 4)
    b = _blur(m, round_px)
    return np.clip((b - 0.5) * 3 + 0.5, 0, 1)


def make_star(size: int = 64, full: bool = True) -> np.ndarray:
    ss = 4
    n = size * ss
    m = _star_mask(n, 0.40, 0.18, n * 0.02)
    key = _ring(m, n * 0.055)
    out = np.zeros((n, n, 3), np.float32)
    out[:] = INK
    if full:
        fill = _vramp(n, n, [(0.0, rgb(255, 250, 200)), (0.45, rgb(255, 214, 80)), (1.0, rgb(245, 150, 40))])
        inner = _inside_dist(m)
        hi = _smooth(1 - inner / (n * 0.05)) * (np.arange(n)[:, None] < n * 0.5)
        fill = _add(fill, rgb(255, 255, 240), hi * 0.35)
        out = _over(out, fill, m[..., None] * np.ones((1, 1, 3), np.float32))
        glow = _blur(_ring(m, n * 0.12), n * 0.05) * 0.55
        alpha = np.maximum(np.maximum(m, key), glow)
        glow_col = np.zeros_like(out)
        glow_col[:] = GOLD
        out = np.where((np.maximum(m, key) > 0.01)[..., None], out, glow_col)
    else:
        out = _over(out, rgb(60, 44, 96), m * 0.9)
        rim = _ring(m, n * 0.03) * 0 + (m > 0.5) * (_inside_dist(m) < n * 0.03)
        out = _over(out, rgb(190, 170, 235), _blur(rim.astype(np.float32), 1.5))
        alpha = np.maximum(m * 0.92, key * 0.85)
    return _to_u8(_down(_bgra(out, alpha), ss), dither=0.0)


def _rounded_rect(n_w: int, n_h: int, r: float, inset: float = 0.0) -> np.ndarray:
    m = np.zeros((n_h, n_w), np.float32)
    x0, y0, x1, y1 = inset, inset, n_w - 1 - inset, n_h - 1 - inset
    ri = max(1.0, r - inset)
    cv2.rectangle(m, (int(x0 + ri), int(y0)), (int(x1 - ri), int(y1)), 1.0, -1)
    cv2.rectangle(m, (int(x0), int(y0 + ri)), (int(x1), int(y1 - ri)), 1.0, -1)
    for cx, cy in ((x0 + ri, y0 + ri), (x1 - ri, y0 + ri), (x0 + ri, y1 - ri), (x1 - ri, y1 - ri)):
        cv2.circle(m, (int(cx), int(cy)), int(ri), 1.0, -1, cv2.LINE_AA)
    return m


def make_card(gold: bool, w: int = 192, h: int = 208) -> np.ndarray:
    """A pictogram card: dark translucent tile with a neon border (gold for gold moves)."""
    ss = 2
    nw, nh = w * ss, h * ss
    pad = 10 * ss
    body = _rounded_rect(nw, nh, 30 * ss, inset=pad)
    border_c = GOLD if gold else rgb(235, 80, 255)
    fill = _vramp(nh, nw, [(0, rgb(70, 36, 120) if not gold else rgb(110, 70, 40)),
                           (1, rgb(24, 12, 46) if not gold else rgb(46, 24, 14))])
    if gold:
        fill = _add(fill, rgb(255, 200, 80), np.clip(1 - _radial(nh, nw, nw / 2, nh * 0.45, nw * 0.45, nh * 0.45), 0, 1) ** 2 * 0.35)
    inner = _inside_dist(body)
    border = _smooth(1 - np.abs(inner - 6 * ss) / (3.2 * ss))
    hot = _smooth(1 - np.abs(inner - 6 * ss) / (1.2 * ss))
    out = fill.copy()
    out = _over(out, INK, (inner < 2.5 * ss).astype(np.float32) * body)
    out = _over(out, border_c, border)
    out = _add(out, (255.0, 255.0, 255.0), hot * 0.45)
    # Glow outside the tile, so it reads on any stage colour.
    halo = _blur(_ring(body, 9 * ss), 5 * ss) * 0.6
    halo_img = np.zeros_like(out)
    halo_img[:] = border_c
    out = np.where((body > 0.01)[..., None], out, halo_img)
    alpha = np.maximum(body * 0.94, halo)
    return _to_u8(_down(_bgra(out, alpha), ss), dither=0.5, stream=31 if gold else 30)


# ---------------------------------------------------------------------------
# words
# ---------------------------------------------------------------------------

def make_word(text: str, colour, cap: int, slant: float = 0.18, stream: int = 40) -> np.ndarray:
    """A bold, slanted arcade word: gradient fill, gloss, dark keyline, coloured glow."""
    ss = 4
    font = cv2.FONT_HERSHEY_DUPLEX
    capn = cap * ss
    scale = capn / 22.0
    thick = max(2, int(round(capn * 0.16)))
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    keyw, glow_w = capn * 0.11, capn * 0.32
    pad = int(glow_w * 1.6 + capn * slant) + 4 * ss
    nw, nh = tw + 2 * pad, th + base + 2 * pad
    nw += (-nw) % ss
    nh += (-nh) % ss
    m8 = np.zeros((nh, nw), np.uint8)
    cv2.putText(m8, text, (pad, pad + th), font, scale, 255, thick, cv2.LINE_AA)
    m = m8.astype(np.float32) / 255.0
    # Slant to the right around the baseline: energy without an italic font.
    shear = np.float32([[1, -slant, slant * (pad + th)], [0, 1, 0]])
    m = cv2.warpAffine(m, shear, (nw, nh), flags=cv2.INTER_LINEAR)
    m = np.clip((_blur(m, capn * 0.012) - 0.5) * 3 + 0.5, 0, 1)  # round the stroke ends
    rows = np.nonzero(m.max(axis=1) > 0.5)[0]
    top, bot = (rows[0], rows[-1]) if len(rows) else (0, nh)
    yy = (np.arange(nh, dtype=np.float32)[:, None] - top) / max(1, bot - top)
    col = np.asarray(colour, np.float32)
    light = col + (255 - col) * 0.72
    deep = col * 0.62
    t = np.clip(yy, 0, 1)
    fill = (light[None, None, :] * (1 - t[..., None]) + col * t[..., None])
    lower = np.clip((t - 0.52) / 0.48, 0, 1)[..., None]
    fill = fill * (1 - lower) + (col * (1 - lower) + deep * lower) * lower
    # Gloss: a lighter upper half with a crisp split, the arcade-logo look.
    gloss = (t < 0.46).astype(np.float32)
    gloss = _blur(gloss, ss * 0.8)
    fill = fill + (255 - fill) * gloss[..., None] * 0.28
    fill = np.broadcast_to(fill, (nh, nw, 3)).copy()
    key = _ring(m, keyw)
    shadow = np.roll(np.roll(np.maximum(m, key), int(capn * 0.06), 0), int(capn * 0.05), 1)
    glow = _blur(_ring(m, glow_w * 0.5), glow_w * 0.45) * 0.85
    out = np.zeros((nh, nw, 3), np.float32)
    out[:] = col
    out = _over(out, INK, np.clip(shadow + key, 0, 1))
    out = _over(out, fill, m[..., None] * np.ones((1, 1, 3), np.float32))
    alpha = np.maximum(np.maximum(m, key), np.maximum(shadow * 0.9, glow))
    sprite = _down(_bgra(out, alpha), ss)
    # Crop to the visible pixels plus a little room for the pop-in scale.
    a = sprite[..., 3] > 2
    ys, xs = np.nonzero(a)
    y0, y1, x0, x1 = max(0, ys.min() - 2), ys.max() + 3, max(0, xs.min() - 2), xs.max() + 3
    return _to_u8(sprite[y0:y1, x0:x1], dither=0.0, stream=stream)


# ---------------------------------------------------------------------------
# the coach's head
# ---------------------------------------------------------------------------

HEAD_SIZE = 256


def make_head(scene) -> tuple[np.ndarray, np.ndarray]:
    """The arcade raccoon with DJ headphones (BGRA), and a white rim around it (BGRA)."""
    head = scene.make_raccoon(size=HEAD_SIZE, chomp=False).astype(np.float32)
    ss = 4
    n = HEAD_SIZE * ss
    outfit = OUTFITS["neon"]
    # Headphones: a band over the crown between the ears, cups over the cheeks.
    band = scene._arc(n, (0.5, 0.56), (0.315, 0.285), 196, 344, 0.05)
    cups = np.maximum(scene._ellipse(n, (0.145, 0.585), (0.062, 0.098), -8),
                      scene._ellipse(n, (0.855, 0.585), (0.062, 0.098), 8))
    pads = np.maximum(scene._ellipse(n, (0.185, 0.585), (0.03, 0.085), -8),
                      scene._ellipse(n, (0.815, 0.585), (0.03, 0.085), 8))
    shape = np.maximum(band, cups)
    key = _ring(shape, 0.028 * n)
    rgbf = np.zeros((n, n, 3), np.float32)
    rgbf[:] = INK
    rgbf = _over(rgbf, outfit["top"], np.maximum(band, cups))
    # Band and cups get a cyan stripe and a highlight, like the outfit.
    stripe = scene._arc(n, (0.5, 0.56), (0.315, 0.285), 200, 340, 0.014)
    rgbf = _over(rgbf, outfit["accent"], stripe)
    hi = np.maximum(scene._ellipse(n, (0.13, 0.55), (0.02, 0.04), -8), scene._ellipse(n, (0.84, 0.55), (0.02, 0.04), 8))
    rgbf = _over(rgbf, outfit["top_hi"], hi * 0.9)
    rgbf = _over(rgbf, outfit["accent"], pads)
    alpha = np.maximum(shape, key)
    phones = _down(_bgra(rgbf, alpha), ss)
    a = phones[..., 3:4] / 255.0
    out = head.copy()
    out[..., :3] = head[..., :3] * (1 - a) + phones[..., :3] * a
    out[..., 3:4] = np.maximum(head[..., 3:4], phones[..., 3:4])
    # Drop the sticker's soft dark halo: the renderer draws a neon rim there.
    out, rim = _head_rim(out)
    return _to_u8(out, dither=0.0), _to_u8(_white(rim), dither=0.0)


def _head_rim(out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop the sticker's soft halo (alpha below the solid head) and return the
    white rim (0..1) the renderer tints around the silhouette."""
    solid = (out[..., 3] / 255.0)
    solid_mask = (solid > 0.6).astype(np.float32)
    rim = np.clip(_ring(solid_mask, 4.5) - solid_mask, 0, 1)
    rim = np.maximum(rim, _blur(_ring(solid_mask, 7.0), 2.0) * 0.5 * (1 - solid_mask))
    out = out.copy()
    out[..., 3] = np.where(solid > 0.6, out[..., 3], np.where(out[..., 3] > 120, out[..., 3], 0))
    return out, rim


# Head turns (degrees of yaw, + = the face turns to screen-right). The
# renderer picks the nearest one as the coach twists (docs/DANCE.md 4a).
HEAD_YAWS = (-20, -10, 0, 10, 20)
HEAD_TURN_SHIFT = 0.6  # interior shift, as a fraction of sin(yaw) times the face's half width
HEAD_LIGHT = (-0.55, -0.83)  # key light from the upper left (unit vector, y down)


def _turn_head(head: np.ndarray, yaw: float) -> np.ndarray:
    """Turn the head about its vertical axis: the features slide towards the
    turn and bunch up on that side, the far side opens up, and the silhouette
    (ears, cups, outline) stays nearly put. A cylinder-ish remap that keeps the
    face's edge columns fixed (x_src = x - s R (1 - u^2), monotonic for s < 0.5)."""
    if yaw == 0:
        return head.copy()
    n = head.shape[0]
    cx, half = 0.5 * n, 0.335 * n
    s = HEAD_TURN_SHIFT * math.sin(math.radians(yaw))
    xs = np.arange(n, dtype=np.float32) + 0.5
    u = (xs - cx) / half
    src = np.where(np.abs(u) < 1.0, xs - s * half * (1.0 - u * u), xs) - 0.5
    mapx = np.repeat(src[None, :], n, axis=0).astype(np.float32)
    mapy = np.repeat((np.arange(n, dtype=np.float32))[:, None], n, axis=1)
    pm = head.astype(np.float32).copy()
    pm[..., :3] *= pm[..., 3:4] / 255.0
    pm = cv2.remap(pm, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    a = np.maximum(pm[..., 3:4], 1e-3)
    pm[..., :3] = np.where(pm[..., 3:4] > 0.01, pm[..., :3] * 255.0 / a, 0)
    return pm


def _shade_head(head: np.ndarray) -> np.ndarray:
    """Bake the coach's key light into the head: a soft core shadow hugging the
    lower-right of the silhouette and a highlight on the upper left, so the
    head reads as a volume like the shaded body (render.py)."""
    n = head.shape[0]
    solid = (head[..., 3] > 150).astype(np.float32)
    inside = _inside_dist(solid)
    cx, cy, r = 0.5 * n, 0.565 * n, 0.36 * n
    ys, xs = np.mgrid[0:n, 0:n].astype(np.float32) + 0.5
    t = -((xs - cx) * HEAD_LIGHT[0] + (ys - cy) * HEAD_LIGHT[1]) / r  # +1 away from the light
    edge = np.clip(1.0 - inside / (0.32 * r), 0.0, 1.0)
    core = _smooth((t - 0.05) / 0.8) * (0.45 + 0.55 * edge) * solid
    core = _blur(core, 2.0)
    hi = _smooth((-t - 0.45) / 0.45) * np.clip(1.0 - np.abs(inside - 0.09 * r) / (0.07 * r), 0, 1) * solid
    hi = _blur(hi, 1.5)
    out = head.copy()
    ambient = np.asarray(rgb(70, 30, 120), np.float32)
    rgbf = out[..., :3]
    rgbf = rgbf * (1.0 - 0.34 * core[..., None]) + ambient * (0.10 * core[..., None])
    rgbf = rgbf + (255.0 - rgbf) * (0.22 * hi[..., None])
    out[..., :3] = rgbf
    return out


def make_head_turns(head_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The head (make_head's) at each of HEAD_YAWS, shaded, side by side in one
    strip (BGRA), and the matching strip of white rims."""
    head = head_u8.astype(np.float32)
    frames, rims = [], []
    for yaw in HEAD_YAWS:
        turned = _shade_head(_turn_head(head, yaw))
        out, rim = _head_rim(turned)
        frames.append(_to_u8(out, dither=0.0))
        rims.append(_to_u8(_white(rim), dither=0.0))
    return np.concatenate(frames, axis=1), np.concatenate(rims, axis=1)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def build() -> dict[str, np.ndarray]:
    scene = _load_scene()
    stage, trims = make_stage()
    head, rim = make_head(scene)
    turns, turns_rim = make_head_turns(head)
    out = {
        "stage.png": stage,
        "glow_trims.png": trims,
        "glow_floor.png": make_floor_glow(),
        "glow_edge.png": make_edge_glow(),
        "beam.png": make_beam(),
        "glow.png": make_glow(),
        "sparkle.png": make_sparkle(),
        "burst.png": make_burst(),
        "ring.png": make_ring(),
        "star_full.png": make_star(full=True),
        "star_empty.png": make_star(full=False),
        "card.png": make_card(False),
        "card_gold.png": make_card(True),
        "coach_head.png": head,
        "coach_head_rim.png": rim,
        "coach_head_turns.png": turns,
        "coach_head_turns_rim.png": turns_rim,
    }
    for i, (name, spec) in enumerate(RATINGS.items()):
        out[f"word_{name}.png"] = make_word(spec["text"], spec["colour"], RATING_CAP, stream=40 + i)
    for i, (name, (text, colour, cap)) in enumerate(WORDS.items()):
        out[f"word_{name}.png"] = make_word(text, colour, cap, stream=50 + i)
    return out


def manifest() -> dict:
    as_list = lambda c: [int(round(v)) for v in c]  # noqa: E731
    return {
        "version": 1,
        "size": [W, H],
        "light_scale": QUARTER,
        "stage": {
            "horizon": HORIZON,
            "lights": [[x, LENS_Y] for x in LIGHTS],
            "spot": list(SPOT),
            "coach": [COACH_X, COACH_FEET],
        },
        "head": {
            # In head-sprite units (0..1): the face centre and where the neck meets it.
            "centre": [0.5, 0.565],
            "chin": [0.5, 0.80],
            "face_width": 0.67,  # the head ellipse, as a fraction of the sprite width
            # coach_head_turns.png: one HEAD_SIZE frame per yaw (degrees, + = turned to screen-right),
            # shaded by the key light from the upper left.
            "turns": list(HEAD_YAWS),
        },
        "outfits": {k: {n: as_list(c) for n, c in v.items()} for k, v in OUTFITS.items()},
        "ratings": {k: {"text": v["text"], "colour": as_list(v["colour"]), "sprite": f"word_{k}.png"}
                    for k, v in RATINGS.items()},
        "words": {k: f"word_{k}.png" for k in WORDS},
        "colours": {"magenta": as_list(MAGENTA), "cyan": as_list(CYAN_NEON),
                    "violet": as_list(VIOLET), "gold": as_list(GOLD)},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--md5", action="store_true", help="print an md5 per file")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    files = build()
    for name, img in files.items():
        if not cv2.imwrite(str(args.out / name), img):
            raise RuntimeError(f"failed to write {name}")
    (args.out / "manifest.json").write_text(json.dumps(manifest(), indent=2) + "\n")
    for name, img in files.items():
        line = f"{name:24s} {img.shape[1]:>5}x{img.shape[0]:<5} {'BGRA' if img.ndim == 3 and img.shape[2] == 4 else 'BGR '}"
        if args.md5:
            line += "  " + hashlib.md5((args.out / name).read_bytes()).hexdigest()
        print(line)
    print(f"wrote {len(files)} images + manifest.json to {args.out}")


if __name__ == "__main__":
    main()
