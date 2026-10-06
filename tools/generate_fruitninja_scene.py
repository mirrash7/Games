#!/usr/bin/env python3
"""Procedurally generate the scene and UI art for the Fruit Ninja game.

Run once (or whenever the look changes):

    uv run python tools/generate_fruitninja_scene.py

numpy + opencv only - no downloads, no binary sources, no extra dependencies,
and nothing traced from any existing game. Output goes to
``assets/fruitninja/generated/``. The fruit sprites, the bomb and the manifest
are produced by a separate generator; this script only writes the five scene /
UI files listed in :data:`ASSETS`.

Look target: a dim dojo at dusk. A woven tatami wall lit by one warm lantern,
everything else falling away into shadow. The camera feed is blended over this
and saturated fruit flies across it, so the backdrop is deliberately dark, low
contrast and low frequency - atmosphere, not detail. Any bright or busy pixel
here is a pixel that fights the gameplay.

Technique notes:

* Everything is composed in float32 and quantised once through :func:`_to_u8`,
  which adds a triangular dither. A dark 720p gradient bands horribly without
  it, and the backdrop here is nothing but dark gradients.
* Sprites are drawn supersampled and box-filtered down (:func:`_downscale`),
  which beats ``LINE_AA`` alone for the concave points of the shuriken and the
  filaments of the splat.
* Alpha sprites are premultiplied before downscaling so soft edges do not pick
  up a dark fringe.
* ``splat.png`` is pure white with the shape carried entirely by alpha, because
  gameplay tints it with each fruit's juice colour at runtime.
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

MAT_HI = (30.0, 48.0, 68.0)  # tatami catching the lantern
MAT_MID = (22.0, 36.0, 53.0)
MAT_LO = (12.0, 20.0, 30.0)  # deep shadow at the very top / bottom
LANTERN = (46.0, 96.0, 150.0)  # warm key light spill
STEEL_HI = (238.0, 242.0, 246.0)
STEEL_LO = (124.0, 134.0, 148.0)
BLADE_EDGE = (18.0, 20.0, 26.0)
LOST_RED = (54.0, 52.0, 196.0)


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


def _alpha_over(dst: np.ndarray, src_rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """src over dst, all float32; alpha is HxW or HxWx1 in 0..1."""
    a = alpha if alpha.ndim == 3 else alpha[..., None]
    return dst * (1.0 - a) + src_rgb * a


def _emboss(mask: np.ndarray, radius: float) -> np.ndarray:
    """Signed -1..1 lighting map for a mask lit from the upper left."""
    m = _blur(mask.astype(np.float32), radius)
    gx = cv2.Sobel(m, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(m, cv2.CV_32F, 0, 1, ksize=5)
    light = gx * 0.55 + gy * 0.85
    peak = float(np.abs(light).max())
    return light / peak if peak > 1e-6 else light


# ---------------------------------------------------------------------------
# background
# ---------------------------------------------------------------------------


def _weave(h: int, w: int, cell: int = 56, stripe: float = 11.0) -> np.ndarray:
    """Basket-weave relief, signed roughly -1..1.

    Alternating cells run their reeds horizontally and vertically, the way a
    tatami mat does. The pattern is deliberately coarse (56px cells): finer
    weave turns into shimmering aliasing once the camera feed is blended on
    top and the whole thing is re-encoded by the display pipeline.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    hor = np.sin(2.0 * np.pi * yy / stripe)
    ver = np.sin(2.0 * np.pi * xx / stripe)
    checker = (np.floor(xx / cell) + np.floor(yy / cell)) % 2.0
    tex = np.where(checker < 0.5, hor, ver)

    # Groove between cells: distance to the nearest cell boundary.
    dx = np.minimum(xx % cell, cell - (xx % cell))
    dy = np.minimum(yy % cell, cell - (yy % cell))
    groove = np.exp(-((np.minimum(dx, dy) / 2.4) ** 2))
    tex = tex * (1.0 - 0.45 * groove) - groove * 0.35

    # Per-cell tone jitter so the mat is not mechanically uniform.
    cells_y, cells_x = int(np.ceil(h / cell)) + 1, int(np.ceil(w / cell)) + 1
    jitter = _rng(3).standard_normal((cells_y, cells_x)).astype(np.float32) * 0.30
    jitter = cv2.resize(jitter, (w + cell, h + cell), interpolation=cv2.INTER_NEAREST)[:h, :w]

    return _blur(tex + jitter, 0.9)


def make_background(w: int = 1280, h: int = 720) -> np.ndarray:
    img = _vramp(
        h,
        w,
        [
            (0.00, MAT_LO),
            (0.22, MAT_MID),
            (0.52, MAT_HI),
            (0.80, MAT_MID),
            (1.00, MAT_LO),
        ],
    )

    # One warm lantern, high and slightly left of centre. Everything else in
    # the frame is lit by its falloff, which is what keeps the backdrop reading
    # as a lit space rather than a flat texture swatch.
    key = 1.0 - _smoothstep(_radial(h, w, w * 0.44, h * 0.34, w * 0.86, h * 1.20))
    key = key**1.35
    img += np.array(LANTERN, np.float32) * (key * 0.40)[..., None]

    # A second, much weaker cool bounce from the lower right stops the shadow
    # side going dead flat.
    bounce = 1.0 - _smoothstep(_radial(h, w, w * 0.92, h * 0.80, w * 0.55, h * 0.55))
    img += np.array((26.0, 20.0, 14.0), np.float32) * (bounce**2.0 * 0.30)[..., None]

    # Weave relief, modulated by the key so the shadow side loses its texture
    # the way a real mat does.
    relief = _weave(h, w)
    lit = 0.22 + 0.78 * key
    img += (relief * lit * 3.2)[..., None] * np.array((0.55, 0.85, 1.0), np.float32)

    # Broad fibre streaks: very low frequency, just enough to break the ramp.
    fibre = _noise(h, w, 26.0, 1.0, stream=5)
    img += (fibre * 3.4)[..., None] * np.array((0.5, 0.8, 1.0), np.float32)
    img += _noise(h, w, 1.0, 1.3, stream=6)[..., None]

    # ---- HUD band ---------------------------------------------------------
    # The bottom eighth is pulled down so lives and combo text stay legible no
    # matter what the camera feed is doing behind them. The transition is a
    # smoothstep over ~70px; a hard edge would read as a UI bar, and the point
    # is for the player never to notice it.
    y = np.arange(h, dtype=np.float32)
    band = _smoothstep((y - h * 0.845) / (h * 0.10))
    img *= (1.0 - 0.42 * band)[:, None, None]
    # Faint warm lip where the band starts, so it reads as a shadowed ledge.
    lip = np.exp(-(((y - h * 0.845) / 3.4) ** 2))
    img += (lip * 4.5)[:, None, None] * np.array((0.5, 0.8, 1.0), np.float32)

    # Top strip gets a gentler version of the same treatment for the score.
    img *= (1.0 - 0.20 * _smoothstep((h * 0.085 - y) / (h * 0.085)))[:, None, None]

    # ---- vignette ---------------------------------------------------------
    vig = 1.0 - 0.66 * _smoothstep((_radial(h, w, w * 0.48, h * 0.46, w * 0.78, h * 0.86) - 0.30) / 0.85)
    img *= vig[..., None]

    return _to_u8(np.clip(img, 0, 255), dither=0.9, stream=1)


# ---------------------------------------------------------------------------
# juice splat
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


def make_splat(size: int = 256) -> np.ndarray:
    """White sprite whose shape lives entirely in the alpha channel.

    Gameplay tints this with the sliced fruit's juice colour, so any colour
    baked in here would multiply against that tint and come out wrong. The
    silhouette is deliberately lopsided - a symmetric splat reads as a bug.
    """
    ss = 3
    n = size * ss
    rng = _rng(11)
    mask = np.zeros((n, n), np.float32)

    cx, cy = n * 0.47, n * 0.50
    base = n * 0.215

    # Main body, plus two overlapping lobes so the silhouette has bays and
    # peninsulas instead of one continuous outline.
    cv2.fillPoly(
        mask,
        [_blob_polygon(cx, cy, base, rng, lobes=7, rough=0.40, squash=0.88, tilt=0.5)],
        1.0,
        cv2.LINE_AA,
    )
    for ox, oy, sc in ((0.62, -0.42, 0.55), (-0.55, 0.40, 0.48), (0.30, 0.66, 0.38)):
        cv2.fillPoly(
            mask,
            [
                _blob_polygon(
                    cx + base * ox,
                    cy + base * oy,
                    base * sc,
                    rng,
                    lobes=5,
                    rough=0.46,
                    squash=0.80,
                    tilt=float(rng.uniform(0, np.pi)),
                )
            ],
            1.0,
            cv2.LINE_AA,
        )

    # ---- runs and drips ---------------------------------------------------
    # A tapering chain of circles from the rim outward, ending in a fat bead:
    # the shape juice makes when it hits a surface and keeps going.
    for ang in rng.permutation(np.linspace(0, 2 * np.pi, 11, endpoint=False))[:5]:
        ang = float(ang) + float(rng.uniform(-0.16, 0.16))
        curve = float(rng.uniform(-0.45, 0.45))
        length = base * float(rng.uniform(0.70, 1.55))
        head = base * float(rng.uniform(0.16, 0.26))
        steps = 46
        for i in range(steps):
            t = i / (steps - 1.0)
            a = ang + curve * t * t
            d = base * 0.78 + length * t
            px = cx + np.cos(a) * d
            py = cy + np.sin(a) * d * 0.94
            rad = head * (1.0 - t) ** 1.35 + n * 0.0022
            cv2.circle(mask, (int(round(px)), int(round(py))), int(round(rad)), 1.0, -1, cv2.LINE_AA)
        # Bead at the tip.
        a = ang + curve
        d = base * 0.78 + length * 1.02
        cv2.fillPoly(
            mask,
            [
                _blob_polygon(
                    cx + np.cos(a) * d,
                    cy + np.sin(a) * d * 0.94,
                    head * float(rng.uniform(0.32, 0.55)),
                    rng,
                    lobes=4,
                    rough=0.32,
                )
            ],
            1.0,
            cv2.LINE_AA,
        )

    # ---- satellite droplets ----------------------------------------------
    sats = np.zeros((n, n), np.float32)
    for _ in range(22):
        a = float(rng.uniform(0, 2 * np.pi))
        d = base * float(rng.uniform(1.30, 2.05))
        px = cx + np.cos(a) * d
        py = cy + np.sin(a) * d * 0.92
        rad = base * float(rng.uniform(0.030, 0.115))
        if not (rad * 2.5 < px < n - rad * 2.5 and rad * 2.5 < py < n - rad * 2.5):
            continue
        # Streak the droplet along its flight direction; round dots look like
        # polka dots, elongated ones look thrown.
        axes = (int(round(rad * float(rng.uniform(1.1, 2.3)))), int(round(rad)))
        cv2.ellipse(
            sats,
            (int(round(px)), int(round(py))),
            axes,
            np.degrees(a),
            0,
            360,
            1.0,
            -1,
            cv2.LINE_AA,
        )
    mask = np.maximum(mask, sats)

    mask = np.clip(_blur(mask, ss * 0.6), 0.0, 1.0)
    mask = _smoothstep((mask - 0.30) / 0.34)

    # ---- alpha shaping ----------------------------------------------------
    # Thick and opaque in the body, thinning towards the rim and along the
    # runs, which is where a real splat is only a film.
    body = np.clip(_blur(mask, ss * 4.0), 0.0, 1.0)
    thickness = 0.87 + 0.13 * _smoothstep(body * 1.55)
    mottle = 1.0 + _noise(n, n, ss * 5.0, 0.045, stream=12)
    alpha = np.clip(mask * thickness * mottle, 0.0, 1.0)

    # A couple of thin spots inside the body: juice does not pool evenly.
    thin = np.zeros((n, n), np.float32)
    for ox, oy, sc in ((-0.30, -0.20, 0.22), (0.34, 0.12, 0.16)):
        cv2.circle(
            thin,
            (int(cx + base * ox), int(cy + base * oy)),
            int(base * sc),
            1.0,
            -1,
            cv2.LINE_AA,
        )
    alpha *= 1.0 - _blur(thin, ss * 3.0) * 0.07

    rgb = np.full((n, n, 3), 255.0, np.float32)
    out = np.concatenate([rgb, (np.clip(alpha, 0, 1) * 255.0)[..., None]], axis=2)
    return _to_u8(_downscale_bgra(out, ss), dither=0.5, stream=2)


# ---------------------------------------------------------------------------
# slice flash
# ---------------------------------------------------------------------------


def make_flash(size: int = 192) -> np.ndarray:
    """Soft white flare, composited additively at the instant of a cut."""
    ss = 2
    n = size * ss
    c = n / 2.0
    r = _radial(n, n, c, c, n * 0.5, n * 0.5)
    f = np.clip(1.0 - r, 0.0, 1.0)

    # Three stacked falloffs: a wide haze, a body, and a small hot core.
    a = 0.30 * f**2.0 + 0.46 * f**5.0 + 0.75 * f**13.0

    # Four soft rays, brighter on the horizontal pair, so the flare has a hint
    # of a blade streak without turning into a sparkle sprite.
    ys = (np.arange(n, dtype=np.float32) + 0.5 - c)[:, None]
    xs = (np.arange(n, dtype=np.float32) + 0.5 - c)[None, :]
    reach = np.exp(-((np.sqrt(xs**2 + ys**2) / (n * 0.24)) ** 1.6))
    a += np.exp(-((ys / (n * 0.030)) ** 2)) * reach * 0.17
    a += np.exp(-((xs / (n * 0.030)) ** 2)) * reach * 0.17
    d0 = (xs + ys) / np.sqrt(2.0)
    d1 = (xs - ys) / np.sqrt(2.0)
    a += np.exp(-((d0 / (n * 0.024)) ** 2)) * reach * 0.07
    a += np.exp(-((d1 / (n * 0.024)) ** 2)) * reach * 0.07
    a = np.clip(a * np.clip(1.0 - r * 0.92, 0.0, 1.0) ** 0.6, 0.0, 1.0)

    # White core cooling to a warm rim; additive, so this reads as heat.
    core = np.clip(f**3.0, 0, 1)[..., None]
    rgb = np.array((214.0, 238.0, 255.0), np.float32) * (1 - core) + np.array(
        (255.0, 255.0, 255.0), np.float32
    ) * core

    out = np.concatenate([rgb, (a * 255.0)[..., None]], axis=2)
    return _to_u8(_downscale_bgra(out, ss), dither=0.4, stream=3)


# ---------------------------------------------------------------------------
# lives
# ---------------------------------------------------------------------------


def _shuriken_mask(n: int, radius: float, inner: float, rot: float, points: int = 4) -> np.ndarray:
    """Filled four-point throwing star with slightly concave blades."""
    mask = np.zeros((n, n), np.float32)
    c = n / 2.0
    pts = []
    for i in range(points):
        a_tip = rot + i * 2.0 * np.pi / points
        a_mid = a_tip + np.pi / points
        # Waist pulled in between tip and valley to hollow the blade edges.
        for a, rr in (
            (a_tip, radius),
            (a_tip + np.pi / points * 0.55, inner * 1.18),
            (a_mid, inner),
            (a_mid + np.pi / points * 0.45, inner * 1.18),
        ):
            pts.append((c + np.cos(a) * rr, c + np.sin(a) * rr))
    cv2.fillPoly(mask, [np.round(np.array(pts, np.float32)).astype(np.int32)], 1.0, cv2.LINE_AA)
    return mask


def _life_base(size: int, ss: int) -> tuple[int, float, np.ndarray, np.ndarray]:
    n = size * ss
    c = n / 2.0
    star = _shuriken_mask(n, n * 0.470, n * 0.180, rot=-np.pi / 2.0 + 0.30)
    hole = np.zeros((n, n), np.float32)
    cv2.circle(hole, (int(c), int(c)), int(n * 0.088), 1.0, -1, cv2.LINE_AA)
    return n, c, star, hole


def make_life_full(size: int = 64) -> np.ndarray:
    """A bright steel shuriken - unmistakably "you still have this one"."""
    ss = 4
    n, c, star, hole = _life_base(size, ss)
    body = np.clip(star - hole, 0.0, 1.0)

    # Brushed-steel ramp along the upper-left / lower-right diagonal.
    ys = np.arange(n, dtype=np.float32)[:, None]
    xs = np.arange(n, dtype=np.float32)[None, :]
    t = _smoothstep(((xs * 0.6 + ys * 0.8) / n - 0.10) / 0.85)
    rgb = np.array(STEEL_LO, np.float32) * t[..., None] + np.array(STEEL_HI, np.float32) * (
        1.0 - t
    )[..., None]
    # Two specular bands crossing the blades.
    diag = (xs * 0.55 + ys * 0.84) / n
    rgb += (np.exp(-(((diag - 0.36) / 0.055) ** 2)) * 70.0)[..., None]
    rgb += (np.exp(-(((diag - 0.68) / 0.030) ** 2)) * 40.0)[..., None]

    bev = _emboss(body, radius=1.6 * ss)
    rgb += (np.clip(bev, 0, None) * 120.0)[..., None]
    rgb -= (np.clip(-bev, 0, None) * 95.0)[..., None] * np.array((0.6, 0.85, 1.0), np.float32)

    # Dark keyline: grow the silhouette, then punch the body back into it. The
    # marker sits on a dark backdrop, so the keyline is what separates the
    # steel from the glow rather than from the background.
    k = max(3, int(2.0 * ss) | 1)
    outline = cv2.dilate(star, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    canvas = np.zeros((n, n, 3), np.float32)
    alpha = np.zeros((n, n), np.float32)

    glow = _blur(outline, 5.0 * ss)
    glow /= max(float(glow.max()), 1e-5)
    canvas = _alpha_over(canvas, np.broadcast_to(np.array(LANTERN, np.float32) * 2.0, canvas.shape), glow)
    alpha = np.maximum(alpha, glow * 0.40)

    canvas = _alpha_over(canvas, np.broadcast_to(np.array(BLADE_EDGE, np.float32), canvas.shape), outline)
    alpha = np.maximum(alpha, outline)

    canvas = _alpha_over(canvas, np.clip(rgb, 0, 255), body)
    alpha = np.maximum(alpha, body)

    # Bright ring around the centre hole reads as a punched, chamfered edge.
    ring = np.clip(_blur(hole, 1.2 * ss) - hole, 0, 1) * star
    canvas += (ring * 150.0)[..., None]

    out = np.concatenate([np.clip(canvas, 0, 255), (np.clip(alpha, 0, 1) * 255.0)[..., None]], axis=2)
    return _to_u8(_downscale_bgra(out, ss), dither=0.4, stream=4)


def make_life_lost(size: int = 64) -> np.ndarray:
    """The spent version: same silhouette gone cold, struck through in red."""
    ss = 4
    n, c, star, hole = _life_base(size, ss)
    body = np.clip(star - hole, 0.0, 1.0)

    # Flat, desaturated and dim - it must lose every fight for attention with
    # the full marker sitting next to it.
    rgb = np.broadcast_to(np.array((88.0, 90.0, 98.0), np.float32), (n, n, 3)).copy()
    bev = _emboss(body, radius=1.6 * ss)
    rgb += (np.clip(bev, 0, None) * 46.0)[..., None]
    rgb -= (np.clip(-bev, 0, None) * 20.0)[..., None]

    k = max(3, int(2.0 * ss) | 1)
    outline = cv2.dilate(star, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    canvas = np.zeros((n, n, 3), np.float32)
    alpha = np.zeros((n, n), np.float32)
    canvas = _alpha_over(canvas, np.broadcast_to(np.array((10.0, 10.0, 14.0), np.float32), canvas.shape), outline)
    alpha = np.maximum(alpha, outline * 0.92)
    canvas = _alpha_over(canvas, np.clip(rgb, 0, 255), body)
    alpha = np.maximum(alpha, body * 0.92)

    # Red cross, keylined in near-black so it stays readable where it crosses
    # the pale part of the star.
    cross = np.zeros((n, n), np.float32)
    r = n * 0.36
    for dx, dy in ((1.0, 1.0), (1.0, -1.0)):
        p0 = (int(round(c - r * dx)), int(round(c - r * dy)))
        p1 = (int(round(c + r * dx)), int(round(c + r * dy)))
        cv2.line(cross, p0, p1, 1.0, int(round(n * 0.066)), cv2.LINE_AA)
    cross = np.clip(cross, 0, 1)
    key = cv2.dilate(cross, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    canvas = _alpha_over(canvas, np.broadcast_to(np.array((8.0, 6.0, 14.0), np.float32), canvas.shape), key)
    alpha = np.maximum(alpha, key)
    cross_rgb = np.broadcast_to(np.array(LOST_RED, np.float32), canvas.shape)
    canvas = _alpha_over(canvas, cross_rgb, cross)
    alpha = np.maximum(alpha, cross)
    # Lift the top-left face of each stroke so the cross is not a flat decal.
    xbev = _emboss(cross, radius=1.2 * ss)
    canvas += (np.clip(xbev, 0, None) * 70.0)[..., None] * cross[..., None]

    out = np.concatenate([np.clip(canvas, 0, 255), (np.clip(alpha, 0, 1) * 255.0)[..., None]], axis=2)
    return _to_u8(_downscale_bgra(out, ss), dither=0.4, stream=5)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

ASSETS = (
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
