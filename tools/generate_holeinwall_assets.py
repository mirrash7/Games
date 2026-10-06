#!/usr/bin/env python3
"""Procedurally generate every visual asset for the Hole in the Wall game.

Run once (or whenever the look changes):

    uv run python tools/generate_holeinwall_assets.py

Everything here is numpy + opencv only - no downloads, no binary sources, no
extra dependencies. Output goes to ``assets/holeinwall/generated/``.

Look target: the TV show's saturated yellow wall and magenta LED studio, drawn
with Kinect Adventures' clean vector polish - soft vertical gradients, glossy
sheen bands, chunky bevels, rounded corners, bright primaries.

Technique notes:

* Everything is composed in float32 and quantised once at the end through
  :func:`_to_u8`, which adds a triangular dither. Smooth 720p gradients band
  badly without it.
* Shapes that need clean edges are drawn supersampled and box-filtered down
  (:func:`_downscale`), which beats ``LINE_AA`` alone for thick strokes and
  text.
* Alpha assets are premultiplied before downscaling so semi-transparent edges
  do not pick up a dark fringe.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

SEED = 20240518
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "assets" / "holeinwall" / "generated"

# ---------------------------------------------------------------------------
# palette (BGR, because that is what OpenCV writes)
# ---------------------------------------------------------------------------

YELLOW_HI = (95.0, 238.0, 255.0)  # top of the wall, warm and bright
YELLOW_LO = (10.0, 168.0, 236.0)  # bottom of the wall, deeper amber
YELLOW_MID = (30.0, 205.0, 250.0)
MAGENTA = (200.0, 20.0, 255.0)
LED_RED = (70.0, 30.0, 255.0)
CYAN = (255.0, 236.0, 150.0)
STAGE_TOP = (26.0, 12.0, 16.0)
STAGE_BOT = (46.0, 20.0, 26.0)
FLOOR_NEAR = (30.0, 14.0, 18.0)


# ---------------------------------------------------------------------------
# small numeric helpers
# ---------------------------------------------------------------------------


def _rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


def _to_u8(arr: np.ndarray, dither: float = 0.75) -> np.ndarray:
    """Quantise float 0..255 to uint8 with triangular dither (kills banding)."""
    a = np.asarray(arr, dtype=np.float32)
    if dither > 0.0:
        r = _rng()
        noise = (r.random(a.shape, dtype=np.float32) - r.random(a.shape, dtype=np.float32)) * dither
        a = a + noise
    return np.clip(a + 0.5, 0, 255).astype(np.uint8)


def _smoothstep(t: np.ndarray | float) -> np.ndarray:
    t = np.clip(np.asarray(t, dtype=np.float32), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _vgrad(h: int, w: int, top: tuple, bot: tuple, gamma: float = 1.0) -> np.ndarray:
    """Vertical colour ramp as float32 HxWx3."""
    t = np.linspace(0.0, 1.0, h, dtype=np.float32) ** gamma
    ramp = np.array(top, np.float32)[None, :] * (1 - t)[:, None] + np.array(bot, np.float32)[
        None, :
    ] * t[:, None]
    return np.repeat(ramp[:, None, :], w, axis=1)


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


def _rounded_rect_mask(h: int, w: int, radius: int, ss: int = 4) -> np.ndarray:
    """Anti-aliased rounded-rectangle coverage mask, float32 0..1."""
    big = np.zeros((h * ss, w * ss), np.uint8)
    r = radius * ss
    cv2.rectangle(big, (r, 0), (w * ss - r, h * ss), 255, -1)
    cv2.rectangle(big, (0, r), (w * ss, h * ss - r), 255, -1)
    for cx, cy in ((r, r), (w * ss - r, r), (r, h * ss - r), (w * ss - r, h * ss - r)):
        cv2.circle(big, (cx, cy), r, 255, -1, cv2.LINE_AA)
    return _downscale(big.astype(np.float32), ss) / 255.0


def _emboss(mask: np.ndarray, radius: float) -> np.ndarray:
    """Signed -1..1 lighting map for a mask lit from the upper left."""
    m = _blur(mask.astype(np.float32), radius)
    gx = cv2.Sobel(m, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(m, cv2.CV_32F, 0, 1, ksize=5)
    light = gx * 0.55 + gy * 0.85
    peak = float(np.abs(light).max())
    return light / peak if peak > 1e-6 else light


def _alpha_over(dst: np.ndarray, src_rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """src over dst, all float32; alpha is HxW or HxWx1 in 0..1."""
    a = alpha if alpha.ndim == 3 else alpha[..., None]
    return dst * (1.0 - a) + src_rgb * a


def _radial(h: int, w: int, cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
    """Normalised radial distance field (0 at centre, 1 on the ellipse)."""
    ys = (np.arange(h, dtype=np.float32) + 0.5 - cy) / max(ry, 1e-3)
    xs = (np.arange(w, dtype=np.float32) + 0.5 - cx) / max(rx, 1e-3)
    return np.sqrt(xs[None, :] ** 2 + ys[:, None] ** 2)


def _noise(h: int, w: int, sigma: float, amp: float) -> np.ndarray:
    """Seeded low-frequency grain, used to keep flat fills from looking dead."""
    r = _rng()
    n = r.standard_normal((h, w)).astype(np.float32)
    n = _blur(n, sigma)
    s = float(n.std())
    return (n / s * amp) if s > 1e-6 else n


# ---------------------------------------------------------------------------
# arena background
# ---------------------------------------------------------------------------


def make_arena_bg(w: int = 1280, h: int = 720) -> np.ndarray:
    horizon = int(h * 0.62)

    img = _vgrad(h, w, STAGE_TOP, STAGE_BOT, gamma=1.3)

    # Cool spill from a big soft key light behind the wall.
    glow = 1.0 - _smoothstep(_radial(h, w, w * 0.5, horizon * 0.72, w * 0.52, h * 0.55))
    img += np.array((70.0, 34.0, 30.0), np.float32) * (glow**1.7)[..., None]

    # Back wall: faint vertical pilasters so the dark is not empty.
    for i in range(1, 8):
        x = int(w * i / 8.0)
        band = np.exp(-(((np.arange(w, dtype=np.float32) - x) / 26.0) ** 2))
        col = band[None, :, None] * np.array((22.0, 10.0, 12.0), np.float32)
        fade = _smoothstep(np.linspace(0.15, 1.0, horizon, dtype=np.float32))[:, None, None]
        img[:horizon] += col * fade

    # ---- floor -----------------------------------------------------------
    fh = h - horizon
    floor = _vgrad(fh, w, tuple(c * 0.55 for c in STAGE_BOT), FLOOR_NEAR, gamma=0.75)

    # Perspective grid converging on the vanishing point.
    grid = np.zeros((fh, w), np.float32)
    vx = w * 0.5
    for i in range(-14, 15):
        x_far = vx + i * w * 0.030
        x_near = vx + i * w * 0.30
        cv2.line(grid, (int(round(x_far)), 0), (int(round(x_near)), fh), 1.0, 1, cv2.LINE_AA)
    for j in range(1, 9):
        t = (j / 9.0) ** 2.4
        y = int(t * fh)
        cv2.line(grid, (0, y), (w, y), 1.0, 1, cv2.LINE_AA)
    grid = _blur(grid, 0.7)
    depth = _smoothstep(np.linspace(0.05, 1.0, fh, dtype=np.float32))[:, None] ** 0.7
    floor += (grid * depth * 1.15)[..., None] * np.array((150.0, 34.0, 115.0), np.float32)

    # Glossy sheen: a wide specular pool right under the wall.
    pool = 1.0 - _smoothstep(_radial(fh, w, w * 0.5, -fh * 0.05, w * 0.46, fh * 0.95))
    floor += (pool**1.7)[..., None] * np.array((96.0, 50.0, 56.0), np.float32)

    img[horizon:] = floor

    # Soft horizon seam so floor and backdrop meet without a hard line.
    seam = np.exp(-((np.arange(h, dtype=np.float32) - horizon) / 9.0) ** 2)
    img += seam[:, None, None] * np.array((26.0, 14.0, 18.0), np.float32)

    # ---- LED arrow strips -------------------------------------------------
    # Stacked chevrons pointing up the strip, on a dim rail. Drawn as filled
    # polygons so they stay crisp arrows rather than a continuous zigzag.
    ss = 3
    strip = np.zeros((h * ss, w * ss, 3), np.float32)
    cx_left, cx_right = w * 0.040, w * 0.960
    span = w * 0.026
    rise = h * 0.036
    bar = h * 0.020
    n = 9
    for cx in (cx_left, cx_right):
        for k in range(n):
            t = k / (n - 1)
            y = (0.075 + t * 0.855) * h
            mix = _smoothstep(t) ** 1.3
            col = np.array(
                [MAGENTA[c] * (1 - mix) + LED_RED[c] * mix for c in range(3)], np.float32
            )
            col *= 0.62 + 0.38 * (1.0 - abs(t - 0.40) * 1.4)
            pts = np.array(
                [
                    [cx - span, y + rise],
                    [cx, y],
                    [cx + span, y + rise],
                    [cx + span, y + rise + bar],
                    [cx, y + bar],
                    [cx - span, y + rise + bar],
                ],
                np.float32,
            )
            cv2.fillPoly(
                strip,
                [np.round(pts * ss).astype(np.int32)],
                tuple(float(c) for c in col),
                cv2.LINE_AA,
            )
    strip = _downscale(strip, ss)

    # Dim LED rail running the full height behind the chevrons.
    rail = np.zeros((h, w), np.float32)
    for cx in (cx_left, cx_right):
        x = int(round(cx))
        rail[:, x - 1 : x + 2] = 1.0
    rail = _blur(rail, 1.6) * 0.55
    strip += rail[..., None] * np.array((130.0, 30.0, 190.0), np.float32)

    img += strip * 1.0 + _blur(strip, 9.0) * 0.75 + _blur(strip, 34.0) * 0.55

    # Reflect the strips into the floor.
    refl = cv2.flip(strip[:horizon], 0)
    rh = min(fh, refl.shape[0])
    refl = refl[-rh:] if refl.shape[0] > rh else refl
    refl = _blur(refl, 5.0)
    fade = (np.linspace(1.0, 0.0, rh, dtype=np.float32) ** 1.6)[:, None, None]
    img[horizon : horizon + rh] += refl * fade * 0.55

    img += _noise(h, w, 1.1, 2.0)[..., None]

    # Vignette.
    vig = 1.0 - 0.62 * _smoothstep((_radial(h, w, w * 0.5, h * 0.48, w * 0.72, h * 0.78) - 0.45) / 0.9)
    img *= vig[..., None]

    return _to_u8(img)


# ---------------------------------------------------------------------------
# wall panel
# ---------------------------------------------------------------------------


def make_wall_panel(size: int = 1024) -> np.ndarray:
    s = size
    img = _vgrad(s, s, YELLOW_HI, YELLOW_LO, gamma=1.15)

    # Very soft horizontal falloff so the middle reads as lit.
    hx = 1.0 - _smoothstep(np.abs(np.linspace(-1, 1, s, dtype=np.float32)) ** 1.6)
    img += (hx[None, :, None] * 0.10) * np.array((30.0, 22.0, 8.0), np.float32)

    # ---- panel seams ------------------------------------------------------
    ss = 2
    seam = np.zeros((s * ss, s * ss), np.float32)
    divs = 4
    for i in range(1, divs):
        p = int(s * ss * i / divs)
        cv2.line(seam, (p, 0), (p, s * ss), 1.0, int(3.0 * ss), cv2.LINE_AA)
        cv2.line(seam, (0, p), (s * ss, p), 1.0, int(3.0 * ss), cv2.LINE_AA)
    seam = _downscale(seam, ss)
    groove = _blur(seam, 1.6)
    img -= groove[..., None] * np.array((40.0, 62.0, 58.0), np.float32)
    # Lit lip just below/right of each groove.
    lip = np.roll(np.roll(groove, 3, axis=0), 3, axis=1) - groove
    img += np.clip(lip, 0, None)[..., None] * np.array((60.0, 40.0, 18.0), np.float32)

    # ---- glossy sheen band -----------------------------------------------
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
    diag = (xx * 0.55 + yy * 0.85) / (s * 1.4)
    sheen = np.exp(-(((diag - 0.34) / 0.16) ** 2)) * 0.55
    sheen += np.exp(-(((diag - 0.62) / 0.05) ** 2)) * 0.22
    img += sheen[..., None] * np.array((58.0, 42.0, 26.0), np.float32)

    # Top highlight, like a strip light above the wall.
    top = np.exp(-((np.arange(s, dtype=np.float32) / (s * 0.13)) ** 2))
    img += top[:, None, None] * np.array((46.0, 30.0, 13.0), np.float32)

    # ---- beveled border ---------------------------------------------------
    border = int(s * 0.035)
    inner = _rounded_rect_mask(s, s, int(s * 0.02))
    frame_mask = _rounded_rect_mask(s, s, int(s * 0.02))
    core = np.zeros((s, s), np.float32)
    cv2.rectangle(core, (border, border), (s - border, s - border), 1.0, -1)
    core = _blur(core, 1.2)
    ring = np.clip(frame_mask - core, 0, 1)

    dark = np.array((0.0, 96.0, 168.0), np.float32)
    img = _alpha_over(img, np.broadcast_to(dark, img.shape), ring * 0.85)
    bev = _emboss(core, radius=border * 0.42)
    img += (np.clip(bev, 0, None) * 130.0)[..., None] * np.array((0.55, 0.85, 1.0), np.float32)
    img -= (np.clip(-bev, 0, None) * 110.0)[..., None] * np.array((0.9, 0.75, 0.45), np.float32)

    # Corner rounding: fill the outside with the border amber, not black. This
    # asset is 3-channel, so anything outside the rounding is visible.
    img = _alpha_over(img, np.broadcast_to(dark * 0.55, img.shape), 1.0 - inner)
    img += _noise(s, s, 1.4, 2.2)[..., None]
    return _to_u8(img)


# ---------------------------------------------------------------------------
# wall edge (chrome rim strip)
# ---------------------------------------------------------------------------


def make_wall_edge(w: int = 512, h: int = 64) -> np.ndarray:
    """Horizontal chrome bevel strip; tile or stretch along the wall border."""
    t = np.linspace(0.0, 1.0, h, dtype=np.float32)

    # Classic chrome ramp: dark lip, hot specular band near the top, a cool
    # mid, a second softer catch light, then a deep shadow at the base. High
    # contrast is what sells metal - a smooth grey ramp reads as plastic.
    lum = (
        0.30
        + 0.95 * np.exp(-(((t - 0.24) / 0.115) ** 2))
        + 0.40 * np.exp(-(((t - 0.50) / 0.18) ** 2))
        + 0.50 * np.exp(-(((t - 0.72) / 0.075) ** 2))
    )
    lum *= 1.0 - 0.60 * _smoothstep((t - 0.80) / 0.20)
    # Warm the highlight, cool the shadow: the studio key is warm, fill is cool.
    tint_hi = np.array((236.0, 244.0, 252.0), np.float32)
    tint_lo = np.array((120.0, 96.0, 84.0), np.float32)
    mix = _smoothstep(np.clip((lum - 0.25) / 1.0, 0, 1))[:, None]
    col = tint_lo[None, :] * (1 - mix) + tint_hi[None, :] * mix
    strip = (col * np.clip(lum, 0, 1.30)[:, None])[:, None, :].repeat(w, axis=1)

    # Slow specular travel along the length so a stretched strip is not uniform.
    travel = 1.0 + 0.16 * np.sin(np.linspace(0, np.pi * 2.0, w, dtype=np.float32))
    strip *= travel[None, :, None]

    # Dark lines top and bottom to seat it against the wall.
    strip[:2] *= 0.30
    strip[-3:] *= 0.22

    alpha = np.full((h, w), 255.0, np.float32)
    alpha[0] = 120.0
    alpha[-1] = 120.0
    out = np.concatenate([np.clip(strip, 0, 255), alpha[..., None]], axis=2)
    return _to_u8(out)


# ---------------------------------------------------------------------------
# glow / particle sprites
# ---------------------------------------------------------------------------


def make_hole_glow(size: int = 256) -> np.ndarray:
    r = _radial(size, size, size / 2, size / 2, size / 2, size / 2)
    falloff = np.clip(1.0 - r, 0.0, 1.0)
    a = falloff**1.5 * 0.52 + falloff**4.0 * 0.34 + falloff**11.0 * 0.40
    a = np.clip(a, 0, 1)
    core = np.clip(falloff**7.0, 0, 1)[..., None]
    rgb = np.array(CYAN, np.float32)[None, None, :] * (1 - core) + np.array(
        (255.0, 255.0, 255.0), np.float32
    ) * core
    out = np.concatenate([rgb, (a * 255.0)[..., None]], axis=2)
    return _to_u8(out, dither=0.4)


def make_particle(size: int = 64) -> np.ndarray:
    ss = 4
    n = size * ss
    r = _radial(n, n, n / 2, n / 2, n / 2, n / 2)
    falloff = np.clip(1.0 - r, 0.0, 1.0)
    a = np.clip(falloff**2.6 * 0.85 + falloff**9.0 * 0.5, 0, 1)
    rgb = np.full((n, n, 3), 255.0, np.float32)
    rgb *= (0.80 + 0.20 * falloff)[..., None]
    out = np.concatenate([rgb, (a * 255.0)[..., None]], axis=2)
    return _to_u8(_downscale_bgra(out, ss), dither=0.4)


# ---------------------------------------------------------------------------
# word badges
# ---------------------------------------------------------------------------


def _draw_bang(dst: np.ndarray, cx: float, baseline: float, cap: float, gap: float) -> None:
    """A tapered exclamation mark that survives being fattened."""
    top = baseline - cap
    dot_r = cap * 0.105
    dot_cy = baseline - dot_r
    stem_bot = dot_cy - dot_r - max(gap * 1.6, cap * 0.09)
    half_top = cap * 0.10
    half_bot = cap * 0.045
    pts = np.array(
        [
            [cx - half_top, top],
            [cx + half_top, top],
            [cx + half_bot, stem_bot],
            [cx - half_bot, stem_bot],
        ],
        np.float32,
    )
    cv2.fillPoly(dst, [np.round(pts).astype(np.int32)], 255, cv2.LINE_AA)
    cv2.circle(dst, (int(round(cx)), int(round(dot_cy))), int(round(dot_r)), 255, -1, cv2.LINE_AA)


def _text_mask(text: str, w: int, h: int, ss: int, weight: float, skew: float) -> np.ndarray:
    """Chunky, slightly italic letterforms as a 0..1 coverage mask.

    Characters are placed one at a time with extra tracking. Fattening the
    glyphs (below) eats the natural side bearings, so without the tracking the
    letters weld into each other and read as a smear.
    """
    big = np.zeros((h * ss, w * ss), np.uint8)
    font = cv2.FONT_HERSHEY_DUPLEX
    base_thick = max(1, int(3 * ss))

    # Fit: measure at scale 1, then solve for the scale that fills the box
    # allowing for the tracking and the weight we are about to add.
    widths = [cv2.getTextSize(c, font, 1.0, base_thick)[0][0] for c in text]
    (_, th1), _ = cv2.getTextSize(text, font, 1.0, base_thick)
    track_frac = 0.24  # of cap height
    natural = sum(widths) + track_frac * th1 * (len(text) - 1)
    grow = weight * ss  # extra px the dilate adds on each side, at final scale
    scale = min(
        (w * ss * 0.80 - grow) / max(natural, 1e-3),
        (h * ss * 0.44 - grow) / max(th1, 1e-3),
    )

    widths = [cv2.getTextSize(c, font, scale, base_thick)[0][0] for c in text]
    (_, th), _ = cv2.getTextSize(text, font, scale, base_thick)
    track = track_frac * th
    total = sum(widths) + track * (len(text) - 1)
    x = (w * ss - total) * 0.5
    y = (h * ss + th) * 0.5
    for c, cw in zip(text, widths):
        if c == "!":
            # Hershey's bang has a hairline gap between stem and dot that the
            # fattening below closes, turning it into an "I". Draw a tapered
            # wedge and a well-separated dot instead - it also just looks
            # punchier, which is the point of the character.
            _draw_bang(big, x + cw * 0.5, y, th, grow / ss)
        else:
            cv2.putText(
                big, c, (int(round(x)), int(round(y))), font, scale, 255, base_thick, cv2.LINE_AA
            )
        x += cw + track

    m = big.astype(np.float32) / 255.0
    # Fatten with a round kernel: gives the heavy arcade weight and rounded
    # terminals that raw putText never has.
    k = max(3, int(weight * ss) | 1)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    if abs(skew) > 1e-6:
        cy = h * ss * 0.5
        mat = np.array([[1.0, skew, -skew * cy], [0.0, 1.0, 0.0]], np.float32)
        m = cv2.warpAffine(
            m, mat, (w * ss, h * ss), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
    return np.clip(m, 0, 1)


def _make_badge(
    text: str,
    top: tuple,
    bot: tuple,
    outline_col: tuple,
    rim_col: tuple,
    w: int = 512,
    h: int = 200,
) -> np.ndarray:
    ss = 3
    W, H = w * ss, h * ss
    core = _text_mask(text, w, h, ss, weight=6.0, skew=-0.15)

    def grow(mask: np.ndarray, px: float) -> np.ndarray:
        k = max(3, int(px * ss) | 1)
        return np.clip(
            cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))), 0, 1
        )

    dark_edge = grow(core, 7.0)  # heavy dark keyline
    rim = grow(core, 12.0)  # bright outer rim
    shadow = grow(core, 9.0)

    shadow = np.roll(shadow, int(7 * ss), axis=0)
    shadow = np.roll(shadow, int(4 * ss), axis=1)
    shadow = _blur(shadow, 6.0 * ss) * 0.85

    rgb = np.zeros((H, W, 3), np.float32)
    alpha = np.zeros((H, W), np.float32)

    # drop shadow
    rgb = _alpha_over(rgb, np.zeros_like(rgb), shadow)
    alpha = np.clip(alpha + shadow * 0.80, 0, 1)

    # bright outer rim
    rim_rgb = np.broadcast_to(np.array(rim_col, np.float32), rgb.shape)
    rgb = _alpha_over(rgb, rim_rgb, rim)
    alpha = np.maximum(alpha, rim)

    # dark keyline
    out_rgb = np.broadcast_to(np.array(outline_col, np.float32), rgb.shape)
    rgb = _alpha_over(rgb, out_rgb, dark_edge)
    alpha = np.maximum(alpha, dark_edge)

    # letter fill: vertical gradient + a hard gloss split near the top third
    fill = _vgrad(H, W, top, bot, gamma=1.0)
    split = _smoothstep((np.arange(H, dtype=np.float32) - H * 0.44) / (H * 0.06))[:, None, None]
    fill = fill * (1.0 - split) + fill * split * 0.80
    fill += (1.0 - split) * np.array((36.0, 36.0, 36.0), np.float32)
    rgb = _alpha_over(rgb, fill, core)
    alpha = np.maximum(alpha, core)

    # inner bevel on the letters
    bev = _emboss(core, radius=3.0 * ss)
    lit = np.clip(bev, 0, None) * core
    shd = np.clip(-bev, 0, None) * core
    rgb += (lit * 165.0)[..., None]
    rgb -= (shd * 120.0)[..., None] * np.array((0.65, 0.85, 1.0), np.float32)

    # specular streak across the upper letters
    yy = np.arange(H, dtype=np.float32)[:, None]
    xx = np.arange(W, dtype=np.float32)[None, :]
    streak = np.exp(-(((yy * 1.0 + xx * 0.22) / H - 0.30) / 0.055) ** 2) * core
    rgb += (streak * 90.0)[..., None]

    # soft glow halo so it pops off a busy frame
    halo = _blur(rim, 9.0 * ss) * 0.55
    alpha = np.clip(np.maximum(alpha, halo * 0.55), 0, 1)
    rgb = _alpha_over(rgb, np.broadcast_to(np.array(rim_col, np.float32), rgb.shape), halo * (1 - alpha))

    out = np.concatenate([np.clip(rgb, 0, 255), (alpha * 255.0)[..., None]], axis=2)
    return _to_u8(_downscale_bgra(out, ss), dither=0.5)


def make_badge_pass() -> np.ndarray:
    # Cyan -> blue. Reads instantly as "good", and it is the one hue guaranteed
    # to pop when the badge lands on top of the yellow wall.
    return _make_badge(
        "PERFECT!",
        top=(255.0, 246.0, 176.0),
        bot=(238.0, 118.0, 18.0),
        outline_col=(46.0, 18.0, 10.0),
        rim_col=(252.0, 252.0, 252.0),
    )


def make_badge_fail() -> np.ndarray:
    # Magenta -> red, matching the studio LEDs.
    return _make_badge(
        "SPLASH!",
        top=(255.0, 176.0, 250.0),
        bot=(52.0, 22.0, 226.0),
        outline_col=(34.0, 12.0, 34.0),
        rim_col=(252.0, 252.0, 252.0),
    )


# ---------------------------------------------------------------------------
# HUD panel
# ---------------------------------------------------------------------------


def make_hud_panel(w: int = 448, h: int = 120) -> np.ndarray:
    radius = 26
    mask = _rounded_rect_mask(h, w, radius)
    inner = _rounded_rect_mask(h - 6, w - 6, radius - 3)
    inner_full = np.zeros((h, w), np.float32)
    inner_full[3 : h - 3, 3 : w - 3] = inner

    body = _vgrad(h, w, (78.0, 34.0, 26.0), (34.0, 14.0, 12.0), gamma=1.2)

    # Glossy upper half, cut with a soft horizontal split.
    split = 1.0 - _smoothstep((np.arange(h, dtype=np.float32) - h * 0.40) / (h * 0.10))
    body += split[:, None, None] * np.array((44.0, 24.0, 20.0), np.float32)
    body += (split**3.0)[:, None, None] * np.array((30.0, 22.0, 20.0), np.float32)

    # Cyan rim light along the bottom, magenta hairline at the top.
    bottom = _smoothstep((np.arange(h, dtype=np.float32) - h * 0.86) / (h * 0.14))
    body += bottom[:, None, None] * np.array((60.0, 30.0, 14.0), np.float32)

    ring = np.clip(mask - inner_full, 0, 1)
    border = np.broadcast_to(np.array((255.0, 180.0, 110.0), np.float32), body.shape)
    body = _alpha_over(body, border, ring * 0.90)

    bev = _emboss(inner_full, radius=3.0)
    body += (np.clip(bev, 0, None) * 70.0)[..., None]
    body -= (np.clip(-bev, 0, None) * 45.0)[..., None]

    body += _noise(h, w, 1.2, 1.6)[..., None]

    alpha = mask * 0.86
    alpha = np.maximum(alpha, ring * 0.98)
    out = np.concatenate([np.clip(body, 0, 255), (alpha * 255.0)[..., None]], axis=2)
    return _to_u8(out, dither=0.5)


# ---------------------------------------------------------------------------
# countdown ring
# ---------------------------------------------------------------------------


def make_countdown_ring(size: int = 256) -> np.ndarray:
    ss = 4
    n = size * ss
    c = n / 2.0
    outer = n * 0.46
    inner = n * 0.355

    r = _radial(n, n, c, c, 1.0, 1.0)
    band = np.clip(1.0 - np.abs(r - (outer + inner) * 0.5) / ((outer - inner) * 0.5), 0, 1)
    ring = np.clip(band * 60.0, 0, 1)  # crisp band, AA comes from the downscale

    # Tick marks around the ring.
    ticks = np.zeros((n, n), np.float32)
    for i in range(12):
        ang = i * np.pi / 6.0 - np.pi / 2.0
        p0 = (c + np.cos(ang) * inner * 1.02, c + np.sin(ang) * inner * 1.02)
        p1 = (c + np.cos(ang) * outer * 0.98, c + np.sin(ang) * outer * 0.98)
        cv2.line(
            ticks,
            (int(round(p0[0])), int(round(p0[1]))),
            (int(round(p1[0])), int(round(p1[1]))),
            1.0,
            int(3 * ss),
            cv2.LINE_AA,
        )
    ticks *= ring

    # Ring shading: bright at the top, deeper at the bottom, glossy inner lip.
    t = np.clip((np.arange(n, dtype=np.float32) / n), 0, 1)[:, None]
    rgb = np.repeat(
        (np.array(CYAN, np.float32)[None, None, :] * (1.0 - t * 0.45)[..., None]), n, axis=1
    )
    rgb += np.array((255.0, 255.0, 255.0), np.float32) * (
        np.exp(-((r - (inner + (outer - inner) * 0.30)) / (n * 0.012)) ** 2) * 0.55
    )[..., None]
    rgb *= (0.75 + 0.45 * np.clip(1.0 - np.abs(r - (outer + inner) * 0.5) / ((outer - inner) * 0.6), 0, 1))[
        ..., None
    ]

    bev = _emboss(ring, radius=2.5 * ss)
    rgb += (np.clip(bev, 0, None) * 150.0)[..., None]
    rgb -= (np.clip(-bev, 0, None) * 90.0)[..., None] * np.array((0.4, 0.8, 1.0), np.float32)
    rgb *= (1.0 - ticks * 0.62)[..., None]

    glow = _blur(ring, 7.0 * ss) * 0.55
    alpha = np.clip(np.maximum(ring, glow * 0.5), 0, 1)
    rgb = _alpha_over(
        rgb, np.broadcast_to(np.array(CYAN, np.float32) * 0.8, rgb.shape), glow * (1 - ring)
    )

    out = np.concatenate([np.clip(rgb, 0, 255), (alpha * 255.0)[..., None]], axis=2)
    return _to_u8(_downscale_bgra(out, ss), dither=0.5)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

ASSETS = (
    ("arena_bg.png", make_arena_bg),
    ("wall_panel.png", make_wall_panel),
    ("wall_edge.png", make_wall_edge),
    ("hole_glow.png", make_hole_glow),
    ("particle.png", make_particle),
    ("badge_pass.png", make_badge_pass),
    ("badge_fail.png", make_badge_fail),
    ("hud_panel.png", make_hud_panel),
    ("countdown_ring.png", make_countdown_ring),
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
        ok = cv2.imwrite(str(path), img)
        if not ok:
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
