#!/usr/bin/env python3
"""Procedurally generate the hero, scene and UI art for Snack Attack.

Run once (or whenever the look changes):

    uv run python tools/generate_fruitninja_scene.py

numpy + opencv only - no downloads, no binary sources, no extra dependencies.
Output goes to ``assets/fruitninja/generated/`` (the game package is still
called fruitninja). The snack sprites, the trash bag and the manifest come from
a separate generator; this script only writes the seven files in :data:`ASSETS`
and never touches anything else in that directory.

The hero is a raccoon head that rides on the player's palm. It is an original
character drawn in the spirit of a purple-and-white sticker mascot: thick dark
keyline, white face, deep-purple bandit mask with a stripe up the forehead,
lavender cheek fluff, glossy black eyes. Every shape here is built from
ellipses, curves and polygons in this file; nothing is traced.

Look target for the scene: a raccoon's moonlit back yard. Indigo sky, a low
moon, a string of party lights, a wooden fence and a bin. The camera feed is
blended over this and bright snacks fly across it, so the backdrop is dark, low
contrast and low frequency - atmosphere, not detail.

Technique notes:

* Everything is composed in float32 and quantised once through :func:`_to_u8`,
  which adds a triangular dither. A dark 720p gradient bands horribly without
  it, and the backdrop here is nothing but dark gradients.
* Sprites are drawn supersampled and box-filtered down (:func:`_downscale`).
* Alpha sprites are premultiplied before downscaling so soft edges do not pick
  up a dark fringe.
* ``splat.png`` is pure white with the shape carried entirely by alpha, because
  gameplay tints it with each snack's crumb colour at runtime.
* Outlines come from a distance transform rather than ``cv2.dilate``: exact
  round keylines of any width, at a cost paid once here, never per frame.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

SEED = 20240811
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "assets" / "fruitninja" / "generated"

# ---------------------------------------------------------------------------
# palette (BGR, because that is what OpenCV writes)
# ---------------------------------------------------------------------------

INK = (46.0, 24.0, 34.0)  # the sticker keyline: near-black with a purple cast
MASK_PURPLE = (155.0, 63.0, 104.0)  # RGB 104,63,155
EAR_PURPLE = (132.0, 50.0, 86.0)  # a step deeper, for the inner ear
LAVENDER = (215.0, 165.0, 185.0)  # RGB 185,165,215: crown and cheek fluff
LAVENDER_HI = (240.0, 214.0, 226.0)
FUR_WHITE = (255.0, 249.0, 251.0)
EYE_BLACK = (30.0, 18.0, 22.0)
MOUTH_DARK = (52.0, 20.0, 86.0)
TONGUE = (138.0, 104.0, 240.0)
LOST_RED = (54.0, 52.0, 214.0)


# ---------------------------------------------------------------------------
# small numeric helpers
# ---------------------------------------------------------------------------


def _rng(stream: int = 0) -> np.random.Generator:
    return np.random.default_rng(SEED + stream * 7919)


def _to_u8(arr: np.ndarray, dither: float = 0.8, stream: int = 0) -> np.ndarray:
    """Quantise float 0..255 to uint8 with triangular dither (kills banding)."""
    a = np.asarray(arr, dtype=np.float32)
    if dither > 0.0:
        r = _rng(stream + 101)
        noise = (r.random(a.shape, dtype=np.float32) - r.random(a.shape, dtype=np.float32)) * dither
        a = a + noise
    return np.clip(a + 0.5, 0, 255).astype(np.uint8)


def _smoothstep(t: np.ndarray | float) -> np.ndarray:
    t = np.clip(np.asarray(t, dtype=np.float32), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)


def _downscale(img: np.ndarray, ss: int) -> np.ndarray:
    """Box-filter a supersampled buffer back to final size."""
    if ss == 1:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (w // ss, h // ss), interpolation=cv2.INTER_AREA)


def _downscale_bgra(bgra: np.ndarray, ss: int) -> np.ndarray:
    """Premultiplied downscale, so soft edges keep their colour."""
    if ss == 1:
        return bgra
    a = bgra[..., 3:4] / 255.0
    pm = np.concatenate([bgra[..., :3] * a, bgra[..., 3:4]], axis=2)
    small = _downscale(pm, ss)
    sa = np.maximum(small[..., 3:4] / 255.0, 1e-4)
    rgb = np.clip(small[..., :3] / sa, 0, 255)
    return np.concatenate([rgb, small[..., 3:4]], axis=2)


def _vramp(h: int, w: int, stops: list[tuple[float, tuple]]) -> np.ndarray:
    """Multi-stop vertical colour ramp as float32 HxWx3."""
    ts = np.array([s[0] for s in stops], np.float32)
    cols = np.array([s[1] for s in stops], np.float32)
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)
    ramp = np.stack([np.interp(y, ts, cols[:, c]).astype(np.float32) for c in range(3)], axis=1)
    return np.repeat(ramp[:, None, :], w, axis=1)


def _radial(h: int, w: int, cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
    """Normalised radial distance field (0 at centre, 1 on the ellipse)."""
    ys = (np.arange(h, dtype=np.float32) + 0.5 - cy) / max(ry, 1e-3)
    xs = (np.arange(w, dtype=np.float32) + 0.5 - cx) / max(rx, 1e-3)
    return np.sqrt(xs[None, :] ** 2 + ys[:, None] ** 2)


def _noise(h: int, w: int, sigma: float, amp: float, stream: int = 0) -> np.ndarray:
    """Seeded low-frequency grain, used to keep flat fills from looking dead."""
    n = _rng(stream).standard_normal((h, w)).astype(np.float32)
    n = _blur(n, sigma)
    s = float(n.std())
    return (n / s * amp) if s > 1e-6 else n


def _alpha_over(dst: np.ndarray, src_rgb, alpha: np.ndarray) -> np.ndarray:
    """src over dst, all float32; alpha is HxW or HxWx1 in 0..1."""
    a = alpha if alpha.ndim == 3 else alpha[..., None]
    src = np.asarray(src_rgb, np.float32)
    return dst * (1.0 - a) + src * a


def _emboss(mask: np.ndarray, radius: float) -> np.ndarray:
    """Signed -1..1 lighting map for a mask lit from the upper left."""
    m = _blur(mask.astype(np.float32), radius)
    gx = cv2.Sobel(m, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(m, cv2.CV_32F, 0, 1, ksize=5)
    light = gx * 0.55 + gy * 0.85
    peak = float(np.abs(light).max())
    return light / peak if peak > 1e-6 else light


# ---------------------------------------------------------------------------
# vector-ish drawing on a supersampled unit canvas
# ---------------------------------------------------------------------------
#
# Sprite geometry is written in unit coordinates (0..1 across the sprite), so
# the numbers read as proportions of the head and survive any change of
# output size or supersampling factor.


def _pt(n: int, p) -> tuple[int, int]:
    return int(round(p[0] * n)), int(round(p[1] * n))


def _ellipse(n: int, c, axes, angle: float = 0.0) -> np.ndarray:
    m = np.zeros((n, n), np.float32)
    cv2.ellipse(m, _pt(n, c), (max(1, int(round(axes[0] * n))), max(1, int(round(axes[1] * n)))),
                angle, 0, 360, 1.0, -1, cv2.LINE_AA)
    return m


def _poly(n: int, pts) -> np.ndarray:
    m = np.zeros((n, n), np.float32)
    arr = np.round(np.asarray(pts, np.float32) * n).astype(np.int32)
    cv2.fillPoly(m, [arr], 1.0, cv2.LINE_AA)
    return m


def _arc(n: int, c, axes, a0: float, a1: float, thick: float, angle: float = 0.0) -> np.ndarray:
    """A round-capped stroke along an elliptical arc (degrees, y down)."""
    m = np.zeros((n, n), np.float32)
    t = np.radians(np.linspace(a0, a1, 64))
    ca, sa = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    x = np.cos(t) * axes[0]
    y = np.sin(t) * axes[1]
    pts = np.stack([c[0] + x * ca - y * sa, c[1] + x * sa + y * ca], axis=1)
    arr = np.round(pts * n).astype(np.int32)
    cv2.polylines(m, [arr], False, 1.0, max(1, int(round(thick * n))), cv2.LINE_AA)
    return m


def _bezier(p0, p1, p2, steps: int = 48) -> np.ndarray:
    """Quadratic Bezier from p0 to p2 pulled toward p1."""
    t = np.linspace(0.0, 1.0, steps, dtype=np.float32)[:, None]
    p0, p1, p2 = (np.asarray(p, np.float32) for p in (p0, p1, p2))
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2


def _round(mask: np.ndarray, sigma: float) -> np.ndarray:
    """Round off a mask's corners: blur, then re-threshold at the midline."""
    b = _blur(mask, sigma)
    return np.clip((b - 0.5) * 3.0 + 0.5, 0.0, 1.0)


def _ring(mask: np.ndarray, width: float) -> np.ndarray:
    """Anti-aliased band of `width` pixels grown outward from `mask`."""
    outside = (mask < 0.5).astype(np.uint8)
    dist = cv2.distanceTransform(outside, cv2.DIST_L2, 5)
    return np.clip(width - dist + 0.5, 0.0, 1.0)


def _mirror(pts) -> list[tuple[float, float]]:
    return [(1.0 - x, y) for x, y in pts]


def _both(fn, *args, **kw) -> np.ndarray:
    """Draw a left-side feature and its mirror image on the right."""
    return np.maximum(fn(*args, side=0, **kw), fn(*args, side=1, **kw))


# ---------------------------------------------------------------------------
# the raccoon
# ---------------------------------------------------------------------------
#
# Layout (unit coords, y down), left side; the right side is the mirror image.
# Head ellipse centred (0.5, 0.56). Ears lean outward from the crown. Eyes sit
# at y 0.545 inside a drooping bandit mask whose lobes meet over the nose; a
# tapering stripe runs from the bridge up the forehead. Cheek fluff fans out
# below the mask in three points per side.

_HEAD_C = (0.5, 0.565)
_HEAD_AX = (0.335, 0.262)
_EYE = (0.362, 0.556)


def _ear_outer(n: int, side: int) -> np.ndarray:
    base_out, tip, base_in = (0.155, 0.50), (0.19, 0.105), (0.46, 0.33)
    a = _bezier(base_out, (0.105, 0.26), tip)
    b = _bezier(tip, (0.33, 0.15), base_in)
    pts = np.concatenate([a, b, [(0.40, 0.52)]])
    if side:
        pts = _mirror(pts)
    return _poly(n, pts)


def _ear_inner(n: int, side: int) -> np.ndarray:
    base_out, tip, base_in = (0.215, 0.43), (0.205, 0.165), (0.395, 0.335)
    a = _bezier(base_out, (0.175, 0.28), tip)
    b = _bezier(tip, (0.30, 0.21), base_in)
    pts = np.concatenate([a, b, [(0.33, 0.45)]])
    if side:
        pts = _mirror(pts)
    return _poly(n, pts)


def _ear_tuft(n: int, side: int) -> np.ndarray:
    """Fluff in the bottom of the ear: a fan of soft points."""
    pts = [(0.205, 0.46), (0.215, 0.33), (0.245, 0.40), (0.268, 0.285),
           (0.292, 0.385), (0.33, 0.30), (0.335, 0.39), (0.39, 0.345), (0.37, 0.46)]
    if side:
        pts = _mirror(pts)
    return _poly(n, pts)


def _cheek_fluff(n: int, side: int) -> np.ndarray:
    pts = [(0.215, 0.52), (0.04, 0.655), (0.165, 0.665), (0.075, 0.775),
           (0.215, 0.745), (0.215, 0.835), (0.325, 0.775), (0.36, 0.64)]
    if side:
        pts = _mirror(pts)
    return _poly(n, pts)


def _mask_lobe(n: int, side: int) -> np.ndarray:
    cx = 0.348 if side == 0 else 0.652
    lobe = _ellipse(n, (cx, 0.56), (0.17, 0.10), -16.0 if side == 0 else 16.0)
    # The outer end droops into a point toward the cheek: the bandit look.
    tail = [(0.215, 0.515), (0.135, 0.635), (0.255, 0.60)]
    if side:
        tail = _mirror(tail)
    return np.maximum(lobe, _poly(n, tail))


def _brow(n: int, side: int) -> np.ndarray:
    cx = 0.352 if side == 0 else 0.648
    return _arc(n, (cx, 0.452), (0.056, 0.030), 200, 340, 0.026, -12.0 if side == 0 else 12.0)


def _eye_open(n: int, side: int) -> np.ndarray:
    cx = _EYE[0] if side == 0 else 1.0 - _EYE[0]
    return _ellipse(n, (cx, _EYE[1]), (0.058, 0.064))


def _eye_glints(n: int, side: int) -> np.ndarray:
    cx = _EYE[0] if side == 0 else 1.0 - _EYE[0]
    big = _ellipse(n, (cx + 0.018, _EYE[1] - 0.022), (0.021, 0.021))
    small = _ellipse(n, (cx - 0.017, _EYE[1] + 0.024), (0.009, 0.009))
    return np.maximum(big, small)


def _eye_happy(n: int, side: int) -> np.ndarray:
    """Squeezed-shut ^ ^ eyes for the chomp frame."""
    cx = _EYE[0] if side == 0 else 1.0 - _EYE[0]
    return _arc(n, (cx, _EYE[1] + 0.022), (0.048, 0.042), 198, 342, 0.034)


def _silhouette(n: int, chomp: bool) -> np.ndarray:
    s = _ellipse(n, _HEAD_C, _HEAD_AX)
    # Ears get a bigger corner radius than the rest: soft, rounded tips read
    # as a furry raccoon rather than a fox or a cat.
    s = np.maximum(s, _round(_both(_ear_outer, n), n * 0.016))
    s = np.maximum(s, _both(_cheek_fluff, n))
    if chomp:
        # The jaw drops a touch, so the open mouth has room without the
        # crown moving: both frames stay registered on the palm.
        s = np.maximum(s, _ellipse(n, (0.5, 0.70), (0.15, 0.135)))
    return _round(s, n * 0.006)


def make_raccoon(size: int = 128, chomp: bool = False) -> np.ndarray:
    """Front-facing raccoon head, sticker style. Idle or mid-chomp.

    Both frames share one canvas and one head position, so swapping them on a
    bite pops the mouth open without the head jumping on the palm.
    """
    ss = 4
    n = size * ss
    rgb = np.zeros((n, n, 3), np.float32)

    sil = _silhouette(n, chomp)
    keyline = _ring(sil, 0.05 * n)  # ~6 px at 128, ~5 px at the 104 px it is drawn
    alpha = np.maximum(sil, keyline)

    # A soft dark halo just outside the keyline, so the head lifts off bright
    # snacks and busy camera frames alike.
    halo = _blur(_ring(sil, 0.062 * n), 0.008 * n) * 0.35
    alpha = np.maximum(alpha, halo)
    rgb[:] = INK

    # ---- fur fills --------------------------------------------------------
    fur = np.zeros_like(rgb)
    fur[:] = LAVENDER
    # Lighter crown, so the lavender reads as fur catching light from above.
    ys = (np.arange(n, dtype=np.float32) / n)[:, None, None]
    fur = fur + (np.array(LAVENDER_HI, np.float32) - LAVENDER) * _smoothstep((0.42 - ys) / 0.30) * 0.55

    # Ears: deep purple inside, then fluff.
    fur = _alpha_over(fur, EAR_PURPLE, _round(_both(_ear_inner, n), n * 0.014))
    fur = _alpha_over(fur, LAVENDER_HI, _round(_both(_ear_tuft, n), n * 0.003))

    # White face: a broad lower face plus a white brow patch over each eye,
    # leaving a lavender notch at the top centre for the stripe to run into.
    face = _ellipse(n, (0.5, 0.635), (0.29, 0.185))
    face = np.maximum(face, _ellipse(n, (0.345, 0.47), (0.122, 0.092), -10))
    face = np.maximum(face, _ellipse(n, (0.655, 0.47), (0.122, 0.092), 10))
    fur = _alpha_over(fur, FUR_WHITE, _round(face, n * 0.006))

    # Lavender cheek fluff under the mask.
    cheeks = np.maximum(_both(_cheek_fluff, n) * (1.0 - _ellipse(n, (0.5, 0.70), (0.17, 0.12))), 0)
    cheeks = np.maximum(cheeks, _ellipse(n, (0.225, 0.655), (0.085, 0.07)))
    cheeks = np.maximum(cheeks, _ellipse(n, (0.775, 0.655), (0.085, 0.07)))
    fur = _alpha_over(fur, LAVENDER, _round(cheeks, n * 0.004) * sil)

    # ---- the mask ---------------------------------------------------------
    mask = _both(_mask_lobe, n)
    mask = np.maximum(mask, _ellipse(n, (0.5, 0.548), (0.09, 0.05)))  # bridge over the nose
    stripe = [(0.452, 0.53), (0.548, 0.53), (0.53, 0.36), (0.5, 0.30), (0.47, 0.36)]
    mask = np.maximum(mask, _poly(n, stripe))
    mask = _round(mask, n * 0.006) * sil
    fur = _alpha_over(fur, MASK_PURPLE, mask)

    # ---- muzzle -----------------------------------------------------------
    muzzle_c, muzzle_ax = ((0.5, 0.688), (0.135, 0.118)) if chomp else ((0.5, 0.672), (0.122, 0.088))
    muzzle = _ellipse(n, muzzle_c, muzzle_ax)
    fur = _alpha_over(fur, FUR_WHITE, muzzle)

    # ---- eyes and brows ---------------------------------------------------
    fur = _alpha_over(fur, INK, _both(_brow, n))
    if chomp:
        fur = _alpha_over(fur, EYE_BLACK, _both(_eye_happy, n))
    else:
        fur = _alpha_over(fur, EYE_BLACK, _both(_eye_open, n))
        fur = _alpha_over(fur, FUR_WHITE, _both(_eye_glints, n))

    # ---- mouth ------------------------------------------------------------
    nose_y = 0.598
    if chomp:
        mouth = _ellipse(n, (0.5, 0.728), (0.092, 0.082))
        mouth = mouth * _poly(n, [(0, nose_y + 0.048), (1, nose_y + 0.048), (1, 1), (0, 1)])
        mouth = _round(mouth, n * 0.008)
        fur = _alpha_over(fur, INK, _ring(mouth, 0.014 * n) * muzzle)
        fur = _alpha_over(fur, MOUTH_DARK, mouth)
        tongue = _ellipse(n, (0.5, 0.79), (0.066, 0.045)) * mouth
        fur = _alpha_over(fur, TONGUE, tongue)
        fur = _alpha_over(fur, (110.0, 72.0, 200.0), _arc(n, (0.5, 0.775), (0.001, 0.02), 90, 270, 0.008) * tongue)
        fangs = np.maximum(_poly(n, [(0.43, 0.645), (0.462, 0.645), (0.447, 0.685)]),
                           _poly(n, _mirror([(0.43, 0.645), (0.462, 0.645), (0.447, 0.685)])))
        fur = _alpha_over(fur, FUR_WHITE, _round(fangs, n * 0.002) * mouth)
    else:
        smile = _arc(n, (0.471, 0.672), (0.029, 0.024), 10, 175, 0.017)
        smile = np.maximum(smile, _arc(n, (0.529, 0.672), (0.029, 0.024), 5, 170, 0.017))
        smile = np.maximum(smile, _arc(n, (0.5, nose_y + 0.04), (0.001, 0.03), 270, 450, 0.017))
        fur = _alpha_over(fur, INK, smile)

    nose = _round(_poly(n, [(0.436, nose_y - 0.004), (0.564, nose_y - 0.004), (0.5, nose_y + 0.062)]), n * 0.012)
    nose = np.maximum(nose, _ellipse(n, (0.5, nose_y + 0.006), (0.06, 0.025)))
    fur = _alpha_over(fur, EYE_BLACK, nose)
    fur = _alpha_over(fur, (150.0, 128.0, 140.0), _ellipse(n, (0.482, nose_y + 0.006), (0.017, 0.008), -8))

    # Gentle form shading: the fur darkens a little toward the keyline, which
    # rounds the head without adding detail that would muddy it at 100 px.
    inner = _blur(sil, 0.03 * n)
    fur *= (0.86 + 0.14 * _smoothstep((inner - 0.35) / 0.6))[..., None]

    rgb = _alpha_over(rgb, fur, sil)
    alpha = np.maximum(alpha, sil)
    out = np.concatenate([np.clip(rgb, 0, 255), (np.clip(alpha, 0, 1) * 255.0)[..., None]], axis=2)
    return _to_u8(_downscale_bgra(out, ss), dither=0.4, stream=7 if chomp else 6)


def make_raccoon_idle() -> np.ndarray:
    return make_raccoon(chomp=False)


def make_raccoon_chomp() -> np.ndarray:
    return make_raccoon(chomp=True)


# ---------------------------------------------------------------------------
# background
# ---------------------------------------------------------------------------


def _catenary(x0: float, y0: float, x1: float, y1: float, sag: float, n: int = 400) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    x = x0 + (x1 - x0) * t
    y = y0 + (y1 - y0) * t + sag * 4.0 * t * (1.0 - t)
    return np.stack([x, y], axis=1)


def make_background(w: int = 1280, h: int = 720) -> np.ndarray:
    rng = _rng(21)
    yy = np.arange(h, dtype=np.float32)[:, None]
    xx = np.arange(w, dtype=np.float32)[None, :]

    # ---- sky --------------------------------------------------------------
    img = _vramp(
        h,
        w,
        [
            (0.00, (24.0, 9.0, 12.0)),
            (0.30, (44.0, 16.0, 22.0)),
            (0.52, (66.0, 28.0, 40.0)),
            (0.60, (60.0, 26.0, 38.0)),
            (1.00, (30.0, 13.0, 18.0)),
        ],
    )

    # Moon, high on the right, with a broad, faint halo.
    mx, my, mr = w * 0.80, h * 0.20, 52.0
    halo = np.exp(-(_radial(h, w, mx, my, 1, 1) / 260.0) ** 1.4)
    img += np.array((40.0, 26.0, 30.0), np.float32) * halo[..., None]
    d = _radial(h, w, mx, my, 1, 1)
    disc = np.clip(mr - d + 0.5, 0.0, 1.0)
    moon = np.broadcast_to(np.array((140.0, 118.0, 124.0), np.float32), img.shape).copy()
    # Soft maria: a couple of darker blotches, then a lit upper-left limb.
    for ox, oy, r in ((-14, -8, 17), (16, 10, 13), (-4, 22, 10)):
        moon -= (np.exp(-(_radial(h, w, mx + ox, my + oy, r, r * 0.8)) ** 2 * 1.2) * 16.0)[..., None]
    moon += (_smoothstep(1.0 - _radial(h, w, mx - 16, my - 16, mr * 1.1, mr * 1.1)) * 14.0)[..., None]
    img = _alpha_over(img, moon, disc)

    # A sparse handful of faint stars, kept off the moon and the score corner.
    stars = np.zeros((h, w), np.float32)
    for _ in range(46):
        sx, sy = float(rng.uniform(0, w)), float(rng.uniform(0, h * 0.42))
        if np.hypot(sx - mx, sy - my) < 120 or (sx < 260 and sy < 110):
            continue
        cv2.circle(stars, (int(sx), int(sy)), 1, float(rng.uniform(0.25, 0.8)), -1, cv2.LINE_AA)
    stars = _blur(stars, 0.7)
    img += stars[..., None] * np.array((70.0, 52.0, 56.0), np.float32)

    # ---- far rooftops -----------------------------------------------------
    roof = np.zeros((h, w), np.float32)
    horizon = h * 0.555
    pts = [(0, h), (0, horizon - 20)]
    x = 0.0
    while x < w + 40:
        bw = float(rng.uniform(110, 210))
        top = horizon - float(rng.uniform(30, 75))
        if rng.random() < 0.55:  # gabled roof
            pts += [(x, top + 26), (x + bw * 0.5, top - 10), (x + bw, top + 26)]
        else:  # flat roof, maybe a chimney
            pts += [(x, top), (x + bw * 0.62, top)]
            if rng.random() < 0.6:
                pts += [(x + bw * 0.62, top - 22), (x + bw * 0.72, top - 22), (x + bw * 0.72, top)]
            pts += [(x + bw, top)]
        x += bw
    pts += [(w, h)]
    cv2.fillPoly(roof, [np.round(np.array(pts, np.float32)).astype(np.int32)], 1.0, cv2.LINE_AA)
    img = _alpha_over(img, (40.0, 16.0, 22.0), _blur(roof, 1.2))
    # A few sleepy lit windows, barely there.
    for _ in range(7):
        wx, wy = float(rng.uniform(40, w - 40)), float(rng.uniform(horizon - 35, horizon + 10))
        win = np.zeros((h, w), np.float32)
        cv2.rectangle(win, (int(wx), int(wy)), (int(wx + 12), int(wy + 15)), 1.0, -1)
        img += (_blur(win, 1.5) * roof)[..., None] * np.array((24.0, 44.0, 64.0), np.float32)

    # ---- string lights ----------------------------------------------------
    # Strung from off the left edge to a post rising behind the fence.
    post_x = w * 0.665
    post = np.zeros((h, w), np.float32)
    cv2.rectangle(post, (int(post_x - 6), int(h * 0.095)), (int(post_x + 6), int(h * 0.6)), 1.0, -1)
    cv2.rectangle(post, (int(post_x - 22), int(h * 0.105)), (int(post_x + 22), int(h * 0.118)), 1.0, -1)
    img = _alpha_over(img, (30.0, 12.0, 17.0), _blur(post, 0.8))
    wire = _catenary(-30, h * 0.20, post_x - 18, h * 0.112, 92)
    lines = np.zeros((h, w), np.float32)
    cv2.polylines(lines, [np.round(wire).astype(np.int32)], False, 1.0, 2, cv2.LINE_AA)
    img = _alpha_over(img, (22.0, 10.0, 14.0), lines * 0.9)
    bulb_cols = [(70.0, 150.0, 230.0), (190.0, 120.0, 235.0), (225.0, 150.0, 175.0)]
    glow = np.zeros((h, w, 3), np.float32)
    bulbs = np.zeros((h, w, 3), np.float32)
    for i, k in enumerate(range(22, len(wire) - 10, 34)):
        bx, by = wire[k]
        by += 9.0
        col = np.array(bulb_cols[i % 3], np.float32)
        g = np.exp(-(_radial(h, w, bx, by, 1, 1) / 22.0) ** 2)
        glow += g[..., None] * col * 0.20
        core = np.zeros((h, w), np.float32)
        cv2.ellipse(core, (int(bx), int(by)), (4, 6), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
        bulbs = _alpha_over(bulbs, col * 0.62, core)
        bulbs = np.maximum(bulbs, 0)
    img += glow
    lit = bulbs.max(axis=2, keepdims=True) > 0
    img = np.where(lit, np.maximum(img, bulbs), img)

    # ---- fence ------------------------------------------------------------
    fence_top, fence_bot = h * 0.535, h * 0.875
    fence = np.zeros((h, w), np.float32)
    shade = np.zeros((h, w), np.float32)
    x = -18.0
    while x < w:
        pw = 62.0
        top = fence_top + float(rng.uniform(-6, 8))
        lean = float(rng.uniform(-2.5, 2.5))
        poly = [(x + lean, top + 20), (x + pw * 0.5 + lean, top), (x + pw + lean, top + 20),
                (x + pw, fence_bot), (x, fence_bot)]
        plank = np.zeros((h, w), np.float32)
        cv2.fillPoly(plank, [np.round(np.array(poly, np.float32)).astype(np.int32)], 1.0, cv2.LINE_AA)
        fence = np.maximum(fence, plank)
        # Moonlight from the upper right: each plank's right edge is lit.
        across = np.clip((xx - x) / pw, 0.0, 1.0)
        shade += plank * (across - 0.5) * 2.0
        shade += plank * float(rng.uniform(-0.6, 0.6))
        x += pw + 7.0
    grain = _blur(_rng(22).standard_normal((h, w)).astype(np.float32), 0.6)
    grain = cv2.GaussianBlur(grain, (0, 0), sigmaX=1.2, sigmaY=18.0)
    grain /= max(float(grain.std()), 1e-6)
    wood = np.broadcast_to(np.array((50.0, 24.0, 33.0), np.float32), (h, w, 3)).copy()
    wood += (shade * 6.0 + grain * 3.0)[..., None] * np.array((1.0, 0.7, 0.8), np.float32)
    # The fence falls off into shadow toward its foot.
    wood *= (1.0 - 0.45 * _smoothstep((yy - fence_top) / (fence_bot - fence_top)))[..., None]
    img = _alpha_over(img, wood, fence)
    # Two rails, each with a soft shadow below it.
    for ry in (h * 0.635, h * 0.80):
        rail = ((yy > ry) & (yy < ry + 16)).astype(np.float32) * np.ones((1, w), np.float32)
        sh = ((yy > ry + 16) & (yy < ry + 30)).astype(np.float32) * np.ones((1, w), np.float32)
        img *= (1.0 - 0.25 * _blur(sh, 3.0) * fence)[..., None]
        rcol = np.array((48.0, 23.0, 31.0), np.float32) * (1.0 - 0.4 * (ry - fence_top) / (fence_bot - fence_top))
        img = _alpha_over(img, rcol, _blur(rail, 0.8) * fence)

    # ---- ground and the bin -----------------------------------------------
    ground = _smoothstep((yy - fence_bot + 2) / 6.0) * np.ones((1, w), np.float32)
    img = _alpha_over(img, (22.0, 10.0, 14.0), ground)

    bin_m = np.zeros((h, w), np.float32)
    bx0, bx1, by0, by1 = w * 0.055, w * 0.155, h * 0.69, h * 0.905
    body = [(bx0 + 6, by0), (bx1 - 6, by0), (bx1, by1), (bx0, by1)]
    cv2.fillPoly(bin_m, [np.round(np.array(body, np.float32)).astype(np.int32)], 1.0, cv2.LINE_AA)
    cv2.ellipse(bin_m, (int((bx0 + bx1) / 2), int(by0)), (int((bx1 - bx0) / 2 + 8), 11),
                -4, 0, 360, 1.0, -1, cv2.LINE_AA)
    cv2.ellipse(bin_m, (int((bx0 + bx1) / 2), int(by0 - 10)), (14, 6), -4, 0, 360, 1.0, -1, cv2.LINE_AA)
    bin_col = np.broadcast_to(np.array((30.0, 13.0, 18.0), np.float32), img.shape).copy()
    ribs = (np.sin((xx - bx0) / (bx1 - bx0) * np.pi * 7.0) * 0.5 + 0.5) * np.ones((h, 1), np.float32)
    bin_col += (ribs * 3.0 + _smoothstep((xx - bx0) / (bx1 - bx0) - 0.55) * 22.0)[..., None]
    img = _alpha_over(img, bin_col, bin_m)

    # ---- grading ----------------------------------------------------------
    img += (_noise(h, w, 30.0, 1.0, stream=23) * 2.5)[..., None] * np.array((1.0, 0.6, 0.8), np.float32)
    img += _noise(h, w, 1.0, 1.0, stream=24)[..., None]

    # HUD band: the bottom eighth is pulled down so lives and combo text stay
    # legible no matter what the camera is doing. A smoothstep over ~70px;
    # a hard edge would read as a UI bar.
    band = _smoothstep((yy - h * 0.845) / (h * 0.10))
    img *= (1.0 - 0.45 * band)[..., None]
    img *= (1.0 - 0.22 * _smoothstep((h * 0.09 - yy) / (h * 0.09)))[..., None]

    vig = 1.0 - 0.55 * _smoothstep((_radial(h, w, w * 0.5, h * 0.48, w * 0.78, h * 0.88) - 0.35) / 0.8)
    img *= vig[..., None]

    return _to_u8(np.clip(img, 0, 255), dither=0.9, stream=1)


# ---------------------------------------------------------------------------
# crumb burst
# ---------------------------------------------------------------------------


def _blob_polygon(
    cx: float,
    cy: float,
    radius: float,
    rng: np.random.Generator,
    lobes: int = 7,
    rough: float = 0.34,
    squash: float = 1.0,
    tilt: float = 0.0,
) -> np.ndarray:
    """Closed polygon with a wobbly, asymmetric radius. Returns int32 points."""
    th = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False, dtype=np.float32)
    r = np.ones_like(th)
    weights = rng.uniform(0.35, 1.0, lobes)
    weights /= weights.sum()
    for k in range(lobes):
        harm = int(rng.integers(2, 7))
        phase = float(rng.uniform(0, 2 * np.pi))
        r += np.sin(harm * th + phase) * rough * float(weights[k]) * 2.4
    r = np.clip(r, 0.45, 1.8) * radius
    x = np.cos(th) * r
    y = np.sin(th) * r * squash
    ct, st = np.cos(tilt), np.sin(tilt)
    pts = np.stack([cx + x * ct - y * st, cy + x * st + y * ct], axis=1)
    return np.round(pts).astype(np.int32)


def _crumb(cx: float, cy: float, r: float, rng: np.random.Generator) -> np.ndarray:
    """An irregular chunky polygon: 5-7 corners at uneven radii."""
    k = int(rng.integers(5, 8))
    th = np.sort(rng.uniform(0, 2 * np.pi, k))
    rr = r * rng.uniform(0.55, 1.15, k)
    return np.round(np.stack([cx + np.cos(th) * rr, cy + np.sin(th) * rr], axis=1)).astype(np.int32)


def make_splat(size: int = 256) -> np.ndarray:
    """A burst of crumbs and sugar: white, with the shape all in alpha.

    Gameplay tints this with each snack's crumb colour, so any colour baked in
    would multiply against that tint. Built from three things: a soft dusty
    puff, chunky crumbs thrown outward (bigger near the bite), and a few
    four-point sugar sparkles. The throw is biased to one side so it never
    reads as a stamped symmetric decal.
    """
    ss = 3
    n = size * ss
    rng = _rng(11)
    c = n * 0.48
    base = n * 0.16

    # ---- the puff ---------------------------------------------------------
    puff = np.zeros((n, n), np.float32)
    cv2.fillPoly(puff, [_blob_polygon(c, c + n * 0.01, base, rng, lobes=6, rough=0.42, squash=0.85, tilt=0.4)],
                 1.0, cv2.LINE_AA)
    for _ in range(5):
        a = float(rng.uniform(0, 2 * np.pi))
        d = base * float(rng.uniform(0.5, 0.9))
        cv2.fillPoly(puff, [_blob_polygon(c + np.cos(a) * d, c + np.sin(a) * d, base * float(rng.uniform(0.35, 0.6)),
                                          rng, lobes=4, rough=0.4)], 1.0, cv2.LINE_AA)
    puff = _blur(puff, ss * 3.5)
    puff = puff * (0.78 + _noise(n, n, ss * 4.0, 0.12, stream=12))
    alpha = np.clip(puff, 0, 1) * 0.70

    # ---- crumbs -----------------------------------------------------------
    crumbs = np.zeros((n, n), np.float32)
    bias = 0.7  # radians: the side the bite throws toward
    for _ in range(30):
        a = float(rng.normal(bias, 1.25))
        t = float(rng.random()) ** 0.8
        d = base * (0.35 + 1.75 * t)
        r = n * (0.058 - 0.034 * t) * float(rng.uniform(0.7, 1.2))
        px, py = c + np.cos(a) * d, c + np.sin(a) * d * 0.9
        if not (r * 2 < px < n - r * 2 and r * 2 < py < n - r * 2):
            continue
        cv2.fillPoly(crumbs, [_crumb(px, py, r, rng)], 1.0, cv2.LINE_AA)
    # Fine dust between the big crumbs.
    for _ in range(40):
        a = float(rng.uniform(0, 2 * np.pi))
        d = base * float(rng.uniform(0.3, 2.3))
        px, py = c + np.cos(a) * d, c + np.sin(a) * d * 0.9
        if 6 < px < n - 6 and 6 < py < n - 6:
            cv2.circle(crumbs, (int(px), int(py)), int(rng.uniform(3.0, 7.0)), 1.0, -1, cv2.LINE_AA)
    alpha = np.maximum(alpha, crumbs)

    # ---- sugar sparkles ---------------------------------------------------
    spark = np.zeros((n, n), np.float32)
    for _ in range(9):
        a = float(rng.uniform(0, 2 * np.pi))
        d = base * float(rng.uniform(0.9, 2.2))
        px, py = c + np.cos(a) * d, c + np.sin(a) * d * 0.9
        L = n * float(rng.uniform(0.045, 0.075))
        wdt = L * 0.22
        if not (L + 2 < px < n - L - 2 and L + 2 < py < n - L - 2):
            continue
        rot = float(rng.uniform(0, np.pi / 4))
        pts = []
        for i in range(8):
            ang = rot + i * np.pi / 4
            rr = L if i % 2 == 0 else wdt
            pts.append((px + np.cos(ang) * rr, py + np.sin(ang) * rr))
        cv2.fillPoly(spark, [np.round(np.array(pts)).astype(np.int32)], 1.0, cv2.LINE_AA)
    alpha = np.maximum(alpha, spark)

    rgb = np.full((n, n, 3), 255.0, np.float32)
    out = np.concatenate([rgb, (np.clip(alpha, 0, 1) * 255.0)[..., None]], axis=2)
    return _to_u8(_downscale_bgra(out, ss), dither=0.5, stream=2)


# ---------------------------------------------------------------------------
# bite flash
# ---------------------------------------------------------------------------


def make_flash(size: int = 192) -> np.ndarray:
    """Soft white flare, composited additively at the instant of a chomp."""
    ss = 2
    n = size * ss
    c = n / 2.0
    r = _radial(n, n, c, c, n * 0.5, n * 0.5)
    f = np.clip(1.0 - r, 0.0, 1.0)

    # Three stacked falloffs: a wide haze, a body, and a small hot core.
    a = 0.30 * f**2.0 + 0.46 * f**5.0 + 0.75 * f**13.0

    # Four short, soft rays: a hint of a "pop" without a hard sparkle shape.
    ys = (np.arange(n, dtype=np.float32) + 0.5 - c)[:, None]
    xs = (np.arange(n, dtype=np.float32) + 0.5 - c)[None, :]
    reach = np.exp(-((np.sqrt(xs**2 + ys**2) / (n * 0.20)) ** 1.6))
    d0 = (xs + ys) / np.sqrt(2.0)
    d1 = (xs - ys) / np.sqrt(2.0)
    a += np.exp(-((ys / (n * 0.028)) ** 2)) * reach * 0.12
    a += np.exp(-((xs / (n * 0.028)) ** 2)) * reach * 0.12
    a += np.exp(-((d0 / (n * 0.024)) ** 2)) * reach * 0.06
    a += np.exp(-((d1 / (n * 0.024)) ** 2)) * reach * 0.06
    a = np.clip(a * np.clip(1.0 - r * 0.92, 0.0, 1.0) ** 0.6, 0.0, 1.0)

    # White core cooling to a faint lavender rim, in the raccoon's palette.
    core = np.clip(f**3.0, 0, 1)[..., None]
    rgb = np.array((255.0, 222.0, 238.0), np.float32) * (1 - core) + np.array(
        (255.0, 255.0, 255.0), np.float32
    ) * core

    out = np.concatenate([rgb, (a * 255.0)[..., None]], axis=2)
    return _to_u8(_downscale_bgra(out, ss), dither=0.4, stream=3)


# ---------------------------------------------------------------------------
# lives: raccoon paw prints
# ---------------------------------------------------------------------------


def _paw_mask(n: int) -> np.ndarray:
    """A raccoon print: a lobed palm pad and five long, splayed fingers.

    Raccoon prints look like little hands - five separate, elongated toes -
    which is what makes this read as "raccoon" and not "dog" even at 40 px.
    """
    m = np.zeros((n, n), np.float32)
    pc = (0.5, 0.66)
    # Palm: three overlapping lobes, wider than tall.
    m = np.maximum(m, _ellipse(n, (pc[0], pc[1] + 0.02), (0.20, 0.135)))
    m = np.maximum(m, _ellipse(n, (pc[0] - 0.10, pc[1] - 0.03), (0.105, 0.105)))
    m = np.maximum(m, _ellipse(n, (pc[0] + 0.10, pc[1] - 0.03), (0.105, 0.105)))
    for ang, dist, ax in ((-74, 0.29, 0.085), (-37, 0.345, 0.092), (0, 0.365, 0.095),
                          (37, 0.345, 0.092), (74, 0.29, 0.085)):
        t = np.radians(ang)
        cx = pc[0] + np.sin(t) * dist
        cy = pc[1] - 0.07 - np.cos(t) * dist
        m = np.maximum(m, _ellipse(n, (cx, cy), (ax * 0.66, ax * 1.15), ang))
    return _round(m, n * 0.006)


def _paw_icon(size: int, fill_top, fill_bot, ink, glow_col, glow_amt: float, opacity: float):
    ss = 4
    n = size * ss
    paw = _paw_mask(n)
    key = _ring(paw, 0.055 * n)
    ys = (np.arange(n, dtype=np.float32) / n)[:, None, None]
    t = _smoothstep((ys - 0.15) / 0.75)
    fill = np.array(fill_top, np.float32) * (1 - t) + np.array(fill_bot, np.float32) * t
    bev = _emboss(paw, radius=1.5 * ss)
    fill = fill + (np.clip(bev, 0, None) * 30.0)[..., None]

    rgb = np.zeros((n, n, 3), np.float32)
    alpha = np.zeros((n, n), np.float32)
    if glow_amt > 0:
        g = _blur(np.maximum(paw, key), 3.5 * ss)
        g /= max(float(g.max()), 1e-5)
        rgb = _alpha_over(rgb, glow_col, g)
        alpha = np.maximum(alpha, g * glow_amt)
    rgb = _alpha_over(rgb, ink, key)
    alpha = np.maximum(alpha, key)
    rgb = _alpha_over(rgb, np.clip(fill, 0, 255), paw)
    alpha = np.maximum(alpha, paw)
    return n, ss, rgb, alpha * opacity


def make_life_full(size: int = 64) -> np.ndarray:
    """A bright lavender-white paw print: "you still have this one"."""
    n, ss, rgb, alpha = _paw_icon(size, FUR_WHITE, LAVENDER, INK, MASK_PURPLE, 0.55, 1.0)
    out = np.concatenate([np.clip(rgb, 0, 255), (np.clip(alpha, 0, 1) * 255.0)[..., None]], axis=2)
    return _to_u8(_downscale_bgra(out, ss), dither=0.4, stream=4)


def make_life_lost(size: int = 64) -> np.ndarray:
    """The spent version: same print gone cold grey, struck through in red."""
    n, ss, rgb, alpha = _paw_icon(size, (92.0, 88.0, 92.0), (70.0, 66.0, 70.0), (12.0, 10.0, 14.0),
                                  (0.0, 0.0, 0.0), 0.0, 0.85)
    c = n / 2.0
    cross = np.zeros((n, n), np.float32)
    r = n * 0.33
    for dx, dy in ((1.0, 1.0), (1.0, -1.0)):
        p0 = (int(round(c - r * dx)), int(round(c + n * 0.04 - r * dy)))
        p1 = (int(round(c + r * dx)), int(round(c + n * 0.04 + r * dy)))
        cv2.line(cross, p0, p1, 1.0, int(round(n * 0.085)), cv2.LINE_AA)
    cross = np.clip(cross, 0, 1)
    key = _ring(cross, 0.035 * n)
    rgb = _alpha_over(rgb, (10.0, 8.0, 16.0), key)
    alpha = np.maximum(alpha, key)
    rgb = _alpha_over(rgb, LOST_RED, cross)
    alpha = np.maximum(alpha, cross)
    xbev = _emboss(cross, radius=1.2 * ss)
    rgb += (np.clip(xbev, 0, None) * 60.0)[..., None] * cross[..., None]
    out = np.concatenate([np.clip(rgb, 0, 255), (np.clip(alpha, 0, 1) * 255.0)[..., None]], axis=2)
    return _to_u8(_downscale_bgra(out, ss), dither=0.4, stream=5)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

ASSETS = (
    ("raccoon_idle.png", make_raccoon_idle),
    ("raccoon_chomp.png", make_raccoon_chomp),
    ("background.png", make_background),
    ("splat.png", make_splat),
    ("flash.png", make_flash),
    ("life_full.png", make_life_full),
    ("life_lost.png", make_life_lost),
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT_DIR, help="output directory")
    args = ap.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, fn in ASSETS:
        img = fn()
        path = out_dir / name
        if not cv2.imwrite(str(path), img):
            raise RuntimeError(f"failed to write {path}")
        h, w = img.shape[:2]
        ch = img.shape[2] if img.ndim == 3 else 1
        rows.append((name, f"{w}x{h}", "BGRA" if ch == 4 else "BGR", path.stat().st_size / 1024.0))

    width = max(len(r[0]) for r in rows)
    print(f"\nwrote {len(rows)} assets to {out_dir}\n")
    print(f"{'name'.ljust(width)}  {'size':>10}  {'fmt':>4}  {'KB':>8}")
    print("-" * (width + 28))
    for name, size, fmt, kb in rows:
        print(f"{name.ljust(width)}  {size:>10}  {fmt:>4}  {kb:>8.1f}")
    print("-" * (width + 28))
    print(f"{'total'.ljust(width)}  {'':>10}  {'':>4}  {sum(r[3] for r in rows):>8.1f}\n")


if __name__ == "__main__":
    main()
