#!/usr/bin/env python3
"""Procedurally generate the fruit, halves and bomb sprites for Fruit Ninja.

Run once (or whenever the look changes):

    uv run python tools/generate_fruitninja_fruit.py

Everything here is numpy + opencv only - no downloads, no binary sources, no
extra dependencies, no reference art of any kind. Output goes to
``assets/fruitninja/generated/`` together with ``manifest.json``, which is the
file ``kpapp.game.fruitninja.art`` parses.

Look target: bright, glossy, saturated arcade fruit. Every sprite is a shaded
solid, not a flat vector: a diffuse term from a fake surface normal, a Blinn
specular, a soft sheen blob, a bounce light on the shadow rim, a darker rim
keyline so the sprite separates from a busy background, and a small drop
shadow.

Technique notes:

* Geometry is written in *unit* coordinates (1.0 = the fruit's nominal radius)
  and rasterised through :class:`Canvas`, which supersamples by ``SS`` and box
  filters back down. Readable maths, clean edges.
* The fake normal comes from the distance transform of the silhouette
  (:func:`shade_body`): ``z = sqrt(t(2-t))`` with the horizontal component
  along the (blurred) distance gradient. That shades *any* silhouette like a
  rounded solid - sphere, apple, banana tube - from one code path.
* Halves are cut out of a *full-size* cross-section render of the same
  silhouette, so half_a stacked on half_b reconstructs the whole outline
  exactly. Only the rim keyline is recomputed per half, which is what gives the
  cut edge its dark lip.
* Alpha is premultiplied before the downscale and unpremultiplied with an
  alpha-weighted colour bleed, so soft edges never pick up a white or black
  fringe.
"""

from __future__ import annotations

import argparse
import json
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

SEED = 20240907
SS = 4  # supersampling factor
PAD = 9  # transparent margin around a sprite, in final pixels
CUT_PAD = 4  # margin on the cut side of a half

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "assets" / "fruitninja" / "generated"

# Key light: upper left, slightly in front of the fruit.
LIGHT = np.array([-0.44, -0.60, 0.67], np.float32)
LIGHT /= np.linalg.norm(LIGHT)
HALFV = LIGHT + np.array([0.0, 0.0, 1.0], np.float32)
HALFV /= np.linalg.norm(HALFV)

SHADOW_RGB = np.array([12.0, 10.0, 14.0], np.float32)
SHADOW_ALPHA = 0.40
SHADOW_OFFSET = (2.0, 2.6)  # final pixels, (dx, dy)
SHADOW_SIGMA = 2.2  # final pixels


# ---------------------------------------------------------------------------
# small numeric helpers
# ---------------------------------------------------------------------------


def _rng(tag: str = "") -> np.random.Generator:
    """Deterministic per-tag generator (``hash()`` is salted, crc32 is not)."""
    return np.random.default_rng(SEED + zlib.crc32(tag.encode()))


def _smoothstep(t: np.ndarray | float) -> np.ndarray:
    t = np.clip(np.asarray(t, dtype=np.float32), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)


def _col(c) -> np.ndarray:
    return np.asarray(c, np.float32)


def over(rgb: np.ndarray, mask: np.ndarray, color) -> np.ndarray:
    """Blend a flat colour (or colour field) into ``rgb`` through ``mask``."""
    a = mask[..., None] if mask.ndim == 2 else mask
    c = _col(color)
    if c.ndim == 1:
        c = c[None, None, :]
    return rgb * (1.0 - a) + c * a


def _unpremultiply(pm: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Undo premultiplication, extending colour outward where alpha fades.

    Dividing straight through by a near-zero alpha explodes to white, which
    shows up as a halo in compositors that ignore alpha (and in any nearest
    neighbour rescale). Blending toward an alpha-weighted local average keeps
    the fringe the colour of the fruit instead.
    """
    num = _blur(pm, 2.0)
    den = _blur(a, 2.0) + 1e-6
    bleed = num / den[..., None]
    direct = pm / np.maximum(a, 1e-3)[..., None]
    w = np.clip(a / 0.35, 0.0, 1.0)[..., None]
    return np.clip(direct * w + bleed * (1.0 - w), 0.0, 255.0)


def _pack(rgb: np.ndarray, alpha: np.ndarray, tag: str) -> np.ndarray:
    """Quantise to BGRA uint8. RGB gets triangular dither, alpha does not."""
    r = _rng("pack" + tag)
    noise = (r.random(rgb.shape, dtype=np.float32) - r.random(rgb.shape, dtype=np.float32)) * 0.7
    bgr = np.clip(rgb + noise + 0.5, 0, 255).astype(np.uint8)
    a = np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return np.concatenate([bgr, a[..., None]], axis=2)


# ---------------------------------------------------------------------------
# canvas
# ---------------------------------------------------------------------------


class Canvas:
    """A supersampled drawing surface with unit-space coordinates.

    ``unit`` is how many *final* pixels one unit spans, so a fruit drawn with
    radius 1.0 ends up ``2 * unit`` pixels across.
    """

    def __init__(self, w: int, h: int, unit: float, oy: float = 0.0, ox: float = 0.0):
        self.w, self.h, self.unit = w, h, unit
        self.W, self.H = w * SS, h * SS
        self.S = unit * SS
        self.cx = self.W / 2.0 + ox * unit * SS
        self.cy = self.H / 2.0 + oy * unit * SS
        xs = (np.arange(self.W, dtype=np.float32) + 0.5 - self.cx) / self.S
        ys = (np.arange(self.H, dtype=np.float32) + 0.5 - self.cy) / self.S
        self.X, self.Y = np.meshgrid(xs, ys)
        self.rgb = np.zeros((self.H, self.W, 3), np.float32)

    # -- coordinate helpers -------------------------------------------------

    def to_px(self, u: float, v: float) -> tuple[float, float]:
        return self.cx + u * self.S, self.cy + v * self.S

    def ipx(self, u: float, v: float) -> tuple[int, int]:
        x, y = self.to_px(u, v)
        return int(round(x)), int(round(y))

    def zeros(self) -> np.ndarray:
        return np.zeros((self.H, self.W), np.float32)

    # -- primitives ---------------------------------------------------------

    def ellipse(self, rx: float, ry: float, ux: float = 0.0, uy: float = 0.0) -> np.ndarray:
        """Analytically anti-aliased ellipse coverage mask."""
        f = np.sqrt(((self.X - ux) / rx) ** 2 + ((self.Y - uy) / ry) ** 2)
        return np.clip((1.0 - f) * min(rx, ry) * self.S + 0.5, 0.0, 1.0)

    def poly(self, pts) -> np.ndarray:
        m = np.zeros((self.H, self.W), np.uint8)
        arr = np.array([self.to_px(u, v) for u, v in pts], np.float32)
        cv2.fillPoly(m, [np.round(arr).astype(np.int32)], 255, cv2.LINE_AA)
        return m.astype(np.float32) / 255.0

    def blob(self, ux: float, uy: float, ru: float, rv: float, ang: float = 0.0) -> np.ndarray:
        """Soft gaussian ellipse - the sheen / glow workhorse."""
        dx, dy = self.X - ux, self.Y - uy
        ca, sa = np.cos(ang), np.sin(ang)
        xr, yr = dx * ca + dy * sa, -dx * sa + dy * ca
        return np.exp(-((xr / ru) ** 2 + (yr / rv) ** 2))

    def dot(self, layer: np.ndarray, u: float, v: float, ru: float, rv: float, ang: float = 0.0,
            value: float = 1.0) -> None:
        """Filled ellipse straight into ``layer`` (unit coords, degrees)."""
        cx, cy = self.to_px(u, v)
        cv2.ellipse(
            layer,
            (int(round(cx)), int(round(cy))),
            (max(1, int(round(ru * self.S))), max(1, int(round(rv * self.S)))),
            ang, 0, 360, float(value), -1, cv2.LINE_AA,
        )

    def stroke(self, layer: np.ndarray, pts, width: float, value: float = 1.0) -> None:
        arr = np.array([self.to_px(u, v) for u, v in pts], np.float32)
        cv2.polylines(
            layer, [np.round(arr).astype(np.int32)], False, float(value),
            max(1, int(round(width * self.S))), cv2.LINE_AA,
        )

    # -- fields -------------------------------------------------------------

    def dist(self, mask: np.ndarray) -> np.ndarray:
        """Distance (in units) from the outside of ``mask``."""
        d = cv2.distanceTransform((mask > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
        return d / self.S

    def radial(self, rx: float, ry: float, ux: float = 0.0, uy: float = 0.0):
        """(normalised radius, ellipse angle) for a citrus-style layout."""
        dx, dy = (self.X - ux) / rx, (self.Y - uy) / ry
        return np.sqrt(dx * dx + dy * dy), np.arctan2(dy, dx)

    def lon(self, rx: float) -> np.ndarray:
        """Longitude in -1..1 across a sphere of radius ``rx`` (edge crowding)."""
        return np.arcsin(np.clip(self.X / rx, -1.0, 1.0)) / (np.pi / 2.0)

    def grain(self, tag: str, sigma: float, amp: float) -> np.ndarray:
        r = _rng(tag)
        n = r.standard_normal((self.H, self.W)).astype(np.float32)
        n = _blur(n, sigma * self.S)
        s = float(n.std())
        return (n / s * amp) if s > 1e-6 else n


# ---------------------------------------------------------------------------
# shading
# ---------------------------------------------------------------------------


def surface_normal(c: Canvas, mask: np.ndarray, radius: float):
    """Fake normal for an arbitrary silhouette, plus the 0..1 inset field."""
    d = c.dist(mask)
    db = _blur(d, max(radius * 0.06, 0.6 / SS) * c.S)
    t = np.clip(db / radius, 0.0, 1.0)
    z = np.sqrt(np.clip(t * (2.0 - t), 0.0, 1.0))
    gx = cv2.Sobel(db, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(db, cv2.CV_32F, 0, 1, ksize=5)
    gn = np.sqrt(gx * gx + gy * gy) + 1e-9
    flat = 1.0 - t
    return (-gx / gn * flat, -gy / gn * flat, z), t


def shade_body(
    c: Canvas,
    mask: np.ndarray,
    base: np.ndarray,
    *,
    radius: float,
    ambient: float = 0.60,
    kd: float = 0.62,
    spec: float = 0.62,
    shin: float = 24.0,
    sheen: tuple | None = (-0.34, -0.40, 0.40, 0.26, -0.6, 0.42),
    bounce: float = 0.34,
    rim: float = 0.62,
    rim_w: float = 0.10,
    rim_tint: tuple = (0.34, 0.30, 0.28),
) -> np.ndarray:
    """Light ``base`` (an HxWx3 colour field) as a rounded glossy solid."""
    (nx, ny, nz), t = surface_normal(c, mask, radius)
    lam = nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2]
    diff = np.clip(lam, 0.0, 1.0)

    out = base * (ambient + kd * diff)[..., None]

    # Bounce light: a cool sliver on the unlit rim. Cheap, sells the roundness.
    b = np.clip(-lam, 0.0, 1.0) * (1.0 - t) ** 2.4
    out += (b * bounce * 105.0)[..., None] * _col((1.05, 0.95, 0.88))

    # Blinn specular.
    s = np.clip(nx * HALFV[0] + ny * HALFV[1] + nz * HALFV[2], 0.0, 1.0) ** shin
    out += (s * spec * 255.0)[..., None]

    # A broad soft sheen, kept off the silhouette edge so it never leaks.
    if sheen is not None:
        ux, uy, ru, rv, ang, amt = sheen
        blob = c.blob(ux, uy, ru, rv, ang) * _smoothstep(t / 0.07)
        out += (blob * amt * 255.0)[..., None]

    # Darker rim keyline: keeps the fruit readable on a busy background.
    e = 1.0 - _smoothstep(t / rim_w)
    out = out * (1.0 - (e * rim)[..., None]) + base * _col(rim_tint) * (e * rim)[..., None]
    return out


def shade_flat(
    c: Canvas,
    mask: np.ndarray,
    base: np.ndarray,
    *,
    radius: float,
    ambient: float = 0.94,
    kd: float = 0.16,
    gloss: float = 0.10,
) -> np.ndarray:
    """Gentle shading for a cut face: it is flat and faces the camera."""
    (nx, ny, nz), t = surface_normal(c, mask, radius)
    lam = nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2]
    out = base * (ambient + kd * np.clip(lam, 0.0, 1.0))[..., None]
    # Wet sheen sweeping across the upper left of the face.
    sweep = c.blob(-0.30, -0.34, 0.62, 0.40, -0.7) * _smoothstep(t / 0.10)
    out += (sweep * gloss * 255.0)[..., None]
    return out


def rind_light(c: Canvas, mask: np.ndarray, band: np.ndarray, radius: float, amount: float = 0.34):
    """Round the skin band of a cut face so the peel does not read as paint."""
    (nx, ny, nz), t = surface_normal(c, mask, radius)
    lam = nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2]
    return band * (np.clip(lam, -1.0, 1.0) * amount)


def rim_key(c: Canvas, mask: np.ndarray, rgb: np.ndarray, width: float = 0.055,
            strength: float = 0.55, tint: tuple = (0.36, 0.32, 0.30)) -> np.ndarray:
    """Dark contrast line hugging the given silhouette."""
    t = np.clip(c.dist(mask) / width, 0.0, 1.0)
    e = (1.0 - _smoothstep(t)) * strength
    return rgb * (1.0 - e[..., None]) + rgb * _col(tint) * e[..., None]


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------


def _shadow(alpha: np.ndarray, w: int, h: int) -> np.ndarray:
    m = np.float32([[1, 0, SHADOW_OFFSET[0] * SS], [0, 1, SHADOW_OFFSET[1] * SS]])
    sh = cv2.warpAffine(alpha, m, (w * SS, h * SS), flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    return _blur(sh, SHADOW_SIGMA * SS) * SHADOW_ALPHA


def finish(rgb: np.ndarray, alpha: np.ndarray, w: int, h: int, tag: str,
           rows: tuple[int, int] | None = None) -> np.ndarray:
    """Shadow, downscale, unpremultiply and quantise one sprite."""
    sh = _shadow(alpha, w, h) * (1.0 - alpha)
    out_a = np.clip(alpha + sh, 0.0, 1.0)
    pm = rgb * alpha[..., None] + SHADOW_RGB[None, None, :] * sh[..., None]

    if rows is not None:
        y0, y1 = rows[0] * SS, rows[1] * SS
        pm, out_a = pm[y0:y1], out_a[y0:y1]
        h = rows[1] - rows[0]

    pm_s = cv2.resize(pm, (w, h), interpolation=cv2.INTER_AREA)
    a_s = cv2.resize(out_a, (w, h), interpolation=cv2.INTER_AREA)
    return _pack(_unpremultiply(pm_s, a_s), a_s, tag)


def mask_margins(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Free pixels (final scale) between the silhouette and each canvas edge."""
    ys, xs = np.nonzero(mask > 0.02)
    if ys.size == 0:
        return (0, 0, 0, 0)
    h, w = mask.shape
    return (int(ys.min()) // SS, int(h - 1 - ys.max()) // SS,
            int(xs.min()) // SS, int(w - 1 - xs.max()) // SS)


def assert_inside(c: Canvas, mask: np.ndarray, name: str, margin: int = 3) -> None:
    top, bot, left, right = mask_margins(mask)
    if min(top, bot, left, right) < margin:
        raise RuntimeError(
            f"{name}: silhouette too close to the canvas border "
            f"(top={top} bottom={bot} left={left} right={right}, need >= {margin}). "
            f"Grow the canvas or adjust oy."
        )


def cut_lip(c: Canvas, rgb: np.ndarray, alpha: np.ndarray, y_ss: int, above: bool) -> np.ndarray:
    """Darken the flesh right at the cut so the straight edge reads as an edge."""
    rows = np.arange(c.H, dtype=np.float32)
    d = (y_ss - rows) if above else (rows - y_ss)
    prof = np.clip(1.0 - d / (1.8 * SS), 0.0, 1.0) * (d >= 0)
    prof = (prof ** 1.5)[:, None] * alpha
    return rgb * (1.0 - (prof * 0.42)[..., None])


def stamp_cut_face(c: Canvas, rgb: np.ndarray, mask: np.ndarray, y_ss: int,
                   flesh, ring, core=None) -> np.ndarray:
    """Elliptical exposed face where a solid body crosses the cut line.

    Used for fruit whose halves keep their peel (banana): the cross section is
    a small oval on the cut, not the whole silhouette.
    """
    row = mask[min(y_ss, c.H - 1)] > 0.5
    idx = np.flatnonzero(row)
    if idx.size == 0:
        return rgb
    breaks = np.flatnonzero(np.diff(idx) > 1)
    runs = np.split(idx, breaks + 1)
    for run in runs:
        if run.size < 4:
            continue
        x0, x1 = float(run[0]), float(run[-1])
        u0 = (x0 - c.cx) / c.S
        u1 = (x1 - c.cx) / c.S
        um = (u0 + u1) * 0.5
        ru = (u1 - u0) * 0.5
        vm = (y_ss - c.cy) / c.S
        rv = ru * 0.58
        face = c.ellipse(ru, rv, um, vm)
        inner = c.ellipse(ru * 0.80, rv * 0.78, um, vm)
        rgb = over(rgb, face * 0.98, ring)
        rgb = over(rgb, inner, flesh)
        if core is not None:
            heart = c.ellipse(ru * 0.30, rv * 0.34, um, vm)
            rgb = over(rgb, heart * 0.65, core)
        # a little modelling on the oval so it is not a flat sticker
        lightf = c.blob(um - ru * 0.30, vm - rv * 0.35, ru * 0.55, rv * 0.60)
        rgb += (lightf * inner * 40.0)[..., None]
        edge = np.clip(face - c.ellipse(ru * 0.90, rv * 0.88, um, vm), 0, 1)
        rgb = rgb * (1.0 - (edge * 0.22)[..., None])
    return rgb


# ---------------------------------------------------------------------------
# fruit: shared ingredients
# ---------------------------------------------------------------------------


def citrus_face(
    c: Canvas,
    mask: np.ndarray,
    rx: float,
    ry: float,
    *,
    segments: int,
    flesh_in,
    flesh_out,
    pith,
    skin,
    veins: float = 0.55,
    pith_w: float = 0.165,
    tag: str = "citrus",
) -> np.ndarray:
    """Radial segment cross-section (orange, lime): pith ring + wedges."""
    f, ang = c.radial(rx, ry)
    t = c.dist(mask)

    skin_w = 0.038
    seg = (ang / (2.0 * np.pi) * segments) % 1.0
    dtheta = np.minimum(seg, 1.0 - seg) / segments * 2.0 * np.pi
    # constant *physical* membrane width -> widens into a core near the middle
    memb = 1.0 - _smoothstep((dtheta * np.maximum(f, 1e-3) - 0.022) / 0.030)

    # juice vesicles: 1-D noise over the angle, stretched radially
    r = _rng(tag)
    bins = 720
    noise = _blur(r.standard_normal(bins).astype(np.float32)[None, :], 1.6)[0]
    streak = np.interp((ang + np.pi) / (2 * np.pi) * bins, np.arange(bins), noise, period=bins)
    streak = streak.astype(np.float32) * veins

    body = np.clip((f - pith_w * 0.0) / 1.0, 0, 1)
    flesh = _col(flesh_in)[None, None, :] * (1 - body[..., None]) + _col(flesh_out)[None, None, :] * body[..., None]
    flesh = flesh + (streak * 16.0)[..., None]
    # a touch of doming per wedge
    flesh = flesh * (1.0 + 0.10 * _smoothstep((dtheta * f) / 0.09))[..., None]

    rgb = flesh
    rgb = over(rgb, memb * 0.88, pith)

    band_pith = 1.0 - _smoothstep((t - skin_w) / (pith_w - skin_w))
    band_skin = 1.0 - _smoothstep((t - skin_w * 0.55) / (skin_w * 0.45))
    rgb = over(rgb, band_pith * 0.99, pith)
    rgb = over(rgb, band_skin, skin)
    dark = 1.0 - _smoothstep((t - 0.012) / 0.020)
    rgb = over(rgb, dark * 0.85, tuple(v * 0.62 for v in skin))
    rgb += (rind_light(c, mask, band_pith, min(rx, ry), 0.34) * 75.0)[..., None]
    return rgb


def leaf_blade(c: Canvas, base_u, base_v, tip_u, tip_v, width, curve=0.0, n=18):
    """Tapered leaf/blade polygon from base to tip, bowed by ``curve``."""
    s = np.linspace(0.0, 1.0, n, dtype=np.float32)
    bx = base_u + (tip_u - base_u) * s
    by = base_v + (tip_v - base_v) * s
    dx, dy = tip_u - base_u, tip_v - base_v
    ln = np.hypot(dx, dy) + 1e-6
    nx, ny = -dy / ln, dx / ln
    bow = np.sin(s * np.pi) * curve
    bx, by = bx + nx * bow, by + ny * bow
    hw = width * np.clip(1.0 - s, 0, 1) ** 0.75 * (
        0.35 + 0.65 * np.clip(np.sin(np.clip(s, 0, 1) * np.pi), 0, 1) ** 0.4
    )
    left = [(bx[i] + nx * hw[i], by[i] + ny * hw[i]) for i in range(n)]
    right = [(bx[i] - nx * hw[i], by[i] - ny * hw[i]) for i in range(n - 1, -1, -1)]
    return c.poly(left + right)


# ---------------------------------------------------------------------------
# watermelon
# ---------------------------------------------------------------------------

WM_RX, WM_RY = 1.0, 0.90


def watermelon_mask(c: Canvas) -> np.ndarray:
    return c.ellipse(WM_RX, WM_RY)


def watermelon_whole(c: Canvas) -> tuple[np.ndarray, np.ndarray]:
    mask = watermelon_mask(c)
    lon = c.lon(WM_RX)
    lat = np.clip(c.Y / WM_RY, -1, 1)

    wob = 0.10 * np.sin(lat * 3.4 + 0.7) + 0.05 * np.sin(lat * 6.1 - 1.1)
    s = np.sin((lon + wob) * np.pi * 3.2 + 0.35)
    stripe = _smoothstep((s - 0.05) / 0.55)

    light = _col((78.0, 168.0, 66.0))
    dark = _col((26.0, 84.0, 26.0))
    base = light[None, None, :] * (1 - stripe[..., None]) + dark[None, None, :] * stripe[..., None]
    base += (c.grain("wm", 0.012, 5.0))[..., None]
    # faint hairline between stripes, like the real skin's edge
    base *= (1.0 - 0.10 * np.exp(-(((s - 0.05) / 0.10) ** 2)))[..., None]

    rgb = shade_body(c, mask, base, radius=WM_RY, spec=0.55, shin=30.0, rim=0.60,
                     sheen=(-0.36, -0.40, 0.28, 0.15, -0.6, 0.20))
    return rgb, mask


def watermelon_cross(c: Canvas) -> tuple[np.ndarray, np.ndarray]:
    mask = watermelon_mask(c)
    t = c.dist(mask)
    f, ang = c.radial(WM_RX, WM_RY)

    flesh_c = _col((70.0, 52.0, 232.0))
    flesh_e = _col((70.0, 60.0, 200.0))
    base = flesh_c[None, None, :] * (1 - f[..., None] ** 1.6) + flesh_e[None, None, :] * (f[..., None] ** 1.6)
    base += (c.grain("wmx", 0.010, 4.5))[..., None]
    # radial fibres
    r = _rng("wmfib")
    bins = 540
    n1 = _blur(r.standard_normal(bins).astype(np.float32)[None, :], 2.0)[0]
    fib = np.interp((ang + np.pi) / (2 * np.pi) * bins, np.arange(bins), n1, period=bins)
    base += (fib.astype(np.float32) * 9.0 * f)[..., None]

    rgb = shade_flat(c, mask, base, radius=WM_RY, gloss=0.13)

    # pale flesh -> rind transition, white rind, thin dark skin
    rind = 1.0 - _smoothstep((t - 0.055) / 0.075)
    pale = 1.0 - _smoothstep((t - 0.10) / 0.075)
    skin_o = 1.0 - _smoothstep((t - 0.020) / 0.035)
    rgb = over(rgb, pale * 0.55, (150.0, 175.0, 225.0))
    rgb = over(rgb, rind * 0.95, (150.0, 214.0, 176.0))
    rgb = over(rgb, skin_o * 0.96, (44.0, 108.0, 40.0))
    rgb += (rind_light(c, mask, rind, WM_RY, 0.32) * 70.0)[..., None]

    # seeds on two rings, pointing at the centre
    seeds = c.zeros()
    hi = c.zeros()
    r = _rng("wmseed")
    for ring, count, jit in ((0.46, 7, 0.05), (0.74, 11, 0.045)):
        for i in range(count):
            a = (i + (0.5 if ring > 0.6 else 0.0)) / count * 2 * np.pi + float(r.normal(0, 0.06))
            rr = ring + float(r.normal(0, jit))
            u, v = np.cos(a) * rr * WM_RX, np.sin(a) * rr * WM_RY
            if abs(v) > WM_RY * 0.90 or np.hypot(u / WM_RX, v / WM_RY) > 0.86:
                continue
            deg = np.degrees(np.arctan2(v / WM_RY, u / WM_RX))
            c.dot(seeds, u, v, 0.072, 0.040, deg, 1.0)
            c.dot(hi, u - 0.012, v - 0.012, 0.030, 0.016, deg, 1.0)
    rgb = over(rgb, seeds * 0.97, (26.0, 24.0, 30.0))
    rgb = over(rgb, hi * seeds * 0.35, (120.0, 118.0, 128.0))
    return rgb, mask


# ---------------------------------------------------------------------------
# orange
# ---------------------------------------------------------------------------

OR_RX, OR_RY = 1.0, 0.95


def orange_mask(c: Canvas) -> np.ndarray:
    return c.ellipse(OR_RX, OR_RY)


def orange_whole(c: Canvas) -> tuple[np.ndarray, np.ndarray]:
    mask = orange_mask(c)
    base = np.broadcast_to(_col((30.0, 150.0, 252.0)), (c.H, c.W, 3)).copy()
    # deeper toward the bottom, warmer on top
    base += (np.clip(-c.Y, -1, 1) * 16.0)[..., None] * _col((0.2, 0.5, 0.2))
    # pores: fine dimpling
    pores = c.grain("orpore", 0.0080, 1.0)
    base += (pores * 9.0)[..., None] * _col((0.3, 0.9, 1.0))
    base -= (np.clip(-pores, 0, None) * 6.0)[..., None]

    rgb = shade_body(c, mask, base, radius=OR_RY, spec=0.55, shin=26.0, rim=0.58,
                     rim_tint=(0.36, 0.30, 0.36),
                     sheen=(-0.34, -0.40, 0.30, 0.16, -0.6, 0.21))

    # navel dimple at the top with a tiny stem scar
    dim = c.blob(0.06, -0.72, 0.16, 0.10, 0.0)
    rgb -= (dim * mask * 46.0)[..., None]
    scar = c.zeros()
    c.dot(scar, 0.06, -0.74, 0.075, 0.048, -12.0, 1.0)
    rgb = over(rgb, scar * 0.85, (48.0, 92.0, 140.0))
    star = c.zeros()
    for k in range(5):
        a = k / 5 * 2 * np.pi
        c.stroke(star, [(0.06, -0.74), (0.06 + np.cos(a) * 0.06, -0.74 + np.sin(a) * 0.045)], 0.016)
    rgb = over(rgb, star * scar * 0.6, (30.0, 60.0, 96.0))
    return rgb, mask


def orange_cross(c: Canvas) -> tuple[np.ndarray, np.ndarray]:
    mask = orange_mask(c)
    rgb = citrus_face(
        c, mask, OR_RX, OR_RY, segments=10,
        flesh_in=(60.0, 178.0, 252.0), flesh_out=(28.0, 140.0, 246.0),
        pith=(212.0, 236.0, 252.0), skin=(24.0, 132.0, 244.0), tag="orange",
    )
    rgb = shade_flat(c, mask, rgb, radius=OR_RY, ambient=0.97, kd=0.12, gloss=0.14)
    return rgb, mask


# ---------------------------------------------------------------------------
# lime
# ---------------------------------------------------------------------------

LM_RX, LM_RY = 1.0, 0.82


def lime_mask(c: Canvas) -> np.ndarray:
    m = c.ellipse(LM_RX, LM_RY)
    m = np.maximum(m, c.ellipse(0.14, 0.11, -0.96, 0.0))
    m = np.maximum(m, c.ellipse(0.14, 0.11, 0.96, 0.0))
    return np.clip(m, 0, 1)


def lime_whole(c: Canvas) -> tuple[np.ndarray, np.ndarray]:
    mask = lime_mask(c)
    base = np.broadcast_to(_col((38.0, 178.0, 112.0)), (c.H, c.W, 3)).copy()
    base += (np.clip(-c.Y / LM_RY, -1, 1) * 16.0)[..., None] * _col((0.3, 1.0, 0.6))
    pores = c.grain("lmpore", 0.0075, 1.0)
    base += (pores * 7.0)[..., None] * _col((0.4, 1.0, 0.7))
    base -= (np.clip(-pores, 0, None) * 5.0)[..., None]

    rgb = shade_body(c, mask, base, radius=LM_RY, spec=0.60, shin=28.0, rim=0.60,
                     sheen=(-0.34, -0.34, 0.34, 0.16, -0.5, 0.30))
    # blossom scars on the little end nubs
    for u in (-0.98, 0.98):
        nub = c.zeros()
        c.dot(nub, u, 0.0, 0.055, 0.045, 0.0, 1.0)
        rgb = over(rgb, nub * 0.55, (60.0, 170.0, 150.0))
    return rgb, mask


def lime_cross(c: Canvas) -> tuple[np.ndarray, np.ndarray]:
    mask = lime_mask(c)
    rgb = citrus_face(
        c, mask, LM_RX, LM_RY, segments=9,
        flesh_in=(146.0, 240.0, 206.0), flesh_out=(58.0, 208.0, 138.0),
        pith=(228.0, 248.0, 240.0), skin=(36.0, 152.0, 84.0), pith_w=0.125, tag="lime",
    )
    rgb = shade_flat(c, mask, rgb, radius=LM_RY, ambient=0.97, kd=0.12, gloss=0.16)
    return rgb, mask


# ---------------------------------------------------------------------------
# apple
# ---------------------------------------------------------------------------

AP_R = 0.86


def apple_body(c: Canvas) -> np.ndarray:
    lobes = np.maximum(c.ellipse(AP_R, AP_R * 1.02, -0.22, 0.06),
                       c.ellipse(AP_R, AP_R * 1.02, 0.22, 0.06))
    bound = c.ellipse(1.03, 0.99, 0.0, 0.06)
    return np.clip(np.minimum(lobes, bound), 0, 1)


def apple_stem(c: Canvas) -> np.ndarray:
    pts = [(-0.045, -0.80), (0.02, -0.94), (0.10, -1.10), (0.17, -1.16)]
    m = c.zeros()
    c.stroke(m, pts, 0.075)
    c.dot(m, 0.17, -1.16, 0.045, 0.038, 0.0, 1.0)
    return np.clip(m, 0, 1)


def apple_leaf(c: Canvas) -> np.ndarray:
    return leaf_blade(c, 0.10, -1.02, 0.62, -1.16, 0.15, curve=0.16)


def apple_mask(c: Canvas) -> np.ndarray:
    return np.clip(np.maximum(np.maximum(apple_body(c), apple_stem(c)), apple_leaf(c)), 0, 1)


def _apple_trim(c: Canvas, rgb: np.ndarray) -> np.ndarray:
    stem, leaf = apple_stem(c), apple_leaf(c)
    st = shade_body(c, stem, np.broadcast_to(_col((48.0, 84.0, 122.0)), (c.H, c.W, 3)).copy(),
                    radius=0.075, ambient=0.62, kd=0.7, spec=0.3, shin=18.0, sheen=None,
                    rim=0.5, rim_w=0.03)
    rgb = over(rgb, stem, st)
    lbase = np.broadcast_to(_col((58.0, 166.0, 86.0)), (c.H, c.W, 3)).copy()
    lbase += (c.blob(0.34, -1.10, 0.26, 0.06, -0.2) * 22.0)[..., None]
    lf = shade_body(c, leaf, lbase, radius=0.15, ambient=0.66, kd=0.62, spec=0.35, shin=22.0,
                    sheen=(0.30, -1.14, 0.20, 0.05, -0.2, 0.30), rim=0.55, rim_w=0.05)
    vein = c.zeros()
    c.stroke(vein, [(0.12, -1.02), (0.36, -1.10), (0.60, -1.15)], 0.018)
    lf = over(lf, vein * leaf * 0.45, (40.0, 128.0, 66.0))
    return over(rgb, leaf, lf)


def apple_whole(c: Canvas) -> tuple[np.ndarray, np.ndarray]:
    body = apple_body(c)
    lon = c.lon(1.02)
    lat = np.clip((c.Y - 0.06) / 1.0, -1, 1)

    deep = _col((36.0, 30.0, 190.0))
    bright = _col((58.0, 58.0, 232.0))
    blush = np.clip(0.55 - lat * 0.5 - np.abs(lon) * 0.25, 0, 1)
    base = deep[None, None, :] * (1 - blush[..., None]) + bright[None, None, :] * blush[..., None]
    # golden-green cheek low on the shadow side
    gold = np.clip(_smoothstep((lat - 0.05) / 0.9) * (0.55 + 0.45 * _smoothstep((lon + 0.2) / 0.9)), 0, 1)
    base = over(base, gold * 0.42, (60.0, 168.0, 236.0))
    # lenticels
    r = _rng("aplent")
    spots = c.zeros()
    for _ in range(46):
        u = float(r.uniform(-0.95, 0.95))
        v = float(r.uniform(-0.85, 0.9))
        if np.hypot(u / 1.0, (v - 0.06) / 0.96) > 0.92:
            continue
        c.dot(spots, u, v, float(r.uniform(0.014, 0.026)), float(r.uniform(0.012, 0.02)), 0.0, 1.0)
    base = over(base, _blur(spots, 0.4 * SS) * 0.40, (150.0, 200.0, 246.0))

    rgb = shade_body(c, body, base, radius=0.80, spec=0.66, shin=24.0, rim=0.62,
                     rim_tint=(0.40, 0.32, 0.34),
                     sheen=(-0.36, -0.34, 0.28, 0.17, -0.6, 0.32))
    # stem well
    well = c.blob(0.0, -0.86, 0.30, 0.12)
    rgb -= (well * body * 52.0)[..., None]
    rgb = _apple_trim(c, rgb)
    return rgb, apple_mask(c)


def apple_cross(c: Canvas) -> tuple[np.ndarray, np.ndarray]:
    body = apple_body(c)
    t = c.dist(body)
    f, ang = c.radial(1.0, 0.98, 0.0, 0.06)

    base = np.broadcast_to(_col((186.0, 226.0, 244.0)), (c.H, c.W, 3)).copy()
    base += (np.clip(1.0 - f, 0, 1)[..., None] ** 1.4) * _col((18.0, 12.0, 6.0))
    base += ((1.0 - _smoothstep((t - 0.10) / 0.10))[..., None]) * _col((20.0, 14.0, 8.0))
    r = _rng("apfib")
    bins = 480
    n1 = _blur(r.standard_normal(bins).astype(np.float32)[None, :], 2.2)[0]
    fib = np.interp((ang + np.pi) / (2 * np.pi) * bins, np.arange(bins), n1, period=bins)
    base += (fib.astype(np.float32) * 7.0 * f)[..., None]
    base += (c.grain("apx", 0.010, 3.0))[..., None]

    rgb = shade_flat(c, body, base, radius=0.80, ambient=0.97, kd=0.10, gloss=0.11)

    # skin ring: thin red band with a darker outer line
    ring = 1.0 - _smoothstep((t - 0.030) / 0.030)
    outer = 1.0 - _smoothstep((t - 0.010) / 0.020)
    rgb = over(rgb, ring * 0.96, (46.0, 44.0, 214.0))
    rgb = over(rgb, outer * 0.85, (30.0, 26.0, 150.0))
    rgb += (rind_light(c, body, ring, 0.80, 0.26) * 60.0)[..., None]

    # core: two lens-shaped chambers stacked, plus stem channel and calyx
    core = c.zeros()
    for sgn in (-1.0, 1.0):
        pts = []
        n = 24
        for i in range(n + 1):
            s = i / n
            v = 0.06 + sgn * (0.02 + s * 0.40)
            hw = 0.24 * np.sin(s * np.pi) ** 0.75 + 0.012
            pts.append((hw, v))
        for i in range(n, -1, -1):
            s = i / n
            v = 0.06 + sgn * (0.02 + s * 0.40)
            hw = 0.24 * np.sin(s * np.pi) ** 0.75 + 0.012
            pts.append((-hw, v))
        core = np.maximum(core, c.poly(pts))
    chan = c.zeros()
    c.stroke(chan, [(0.0, -0.92), (0.0, 0.90)], 0.045)
    core = np.clip(np.maximum(core, chan * 0.9) * body, 0, 1)
    k = int(0.016 * c.S) * 2 + 1
    edge = np.clip(core - cv2.erode(core, np.ones((k, k), np.uint8)), 0, 1)
    rgb = over(rgb, core * 0.92, (216.0, 240.0, 250.0))
    rgb = over(rgb, edge * 0.80, (104.0, 158.0, 198.0))

    pips = c.zeros()
    for u, v, a in ((-0.115, -0.16, 28.0), (0.115, -0.16, -28.0),
                    (-0.115, 0.28, -28.0), (0.115, 0.28, 28.0)):
        c.dot(pips, u, v, 0.078, 0.048, a, 1.0)
    halo = _blur(pips, 0.9 * SS)
    rgb = over(rgb, np.clip(halo * 0.45, 0, 1), (150.0, 196.0, 226.0))
    rgb = over(rgb, pips * 0.96, (26.0, 48.0, 84.0))
    php = c.zeros()
    for u, v, a in ((-0.13, -0.185, 28.0), (0.10, -0.185, -28.0),
                    (-0.13, 0.255, -28.0), (0.10, 0.255, 28.0)):
        c.dot(php, u, v, 0.030, 0.018, a, 1.0)
    rgb = over(rgb, php * pips * 0.40, (110.0, 150.0, 190.0))

    rgb = _apple_trim(c, rgb)
    return rgb, apple_mask(c)


# ---------------------------------------------------------------------------
# strawberry
# ---------------------------------------------------------------------------


def _straw_outline(n: int = 46):
    pts = []
    for i in range(n + 1):
        s = i / n
        v = -0.92 + s * 2.12
        u = (1.0 - s) ** 0.42 * (s + 0.16) ** 0.55 * 1.65
        pts.append((u, v))
    right = pts
    left = [(-u, v) for u, v in reversed(pts)]
    return right + left


def straw_body(c: Canvas) -> np.ndarray:
    m = c.poly(_straw_outline())
    m = np.maximum(m, c.ellipse(0.62, 0.30, 0.0, -0.74))
    return np.clip(m, 0, 1)


def straw_calyx(c: Canvas) -> np.ndarray:
    m = c.zeros()
    specs = ((-0.90, 0.62, 0.15), (-0.50, 0.80, 0.17), (0.0, 0.86, 0.18),
             (0.50, 0.80, 0.17), (0.90, 0.62, 0.15), (-1.16, 0.30, 0.12), (1.16, 0.30, 0.12))
    for ux, uy, wd in specs:
        m = np.maximum(m, leaf_blade(c, 0.0, -0.80, ux, -0.80 - uy * 0.62, wd, curve=0.05))
    st = c.zeros()
    c.stroke(st, [(0.0, -0.86), (-0.03, -1.20), (0.02, -1.34)], 0.070)
    return np.clip(np.maximum(m, st), 0, 1)


def straw_mask(c: Canvas) -> np.ndarray:
    return np.clip(np.maximum(straw_body(c), straw_calyx(c)), 0, 1)


def _straw_trim(c: Canvas, rgb: np.ndarray) -> np.ndarray:
    cal = straw_calyx(c)
    base = np.broadcast_to(_col((56.0, 172.0, 96.0)), (c.H, c.W, 3)).copy()
    base += (c.blob(0.0, -1.05, 0.7, 0.35) * 24.0)[..., None] * _col((0.4, 1.0, 0.6))
    sh = shade_body(c, cal, base, radius=0.15, ambient=0.66, kd=0.58, spec=0.30, shin=20.0,
                    sheen=(-0.25, -1.10, 0.28, 0.10, -0.4, 0.26), rim=0.55, rim_w=0.045)
    return over(rgb, cal, sh)


def straw_whole(c: Canvas) -> tuple[np.ndarray, np.ndarray]:
    body = straw_body(c)
    lon = c.lon(1.15)
    base = np.broadcast_to(_col((44.0, 40.0, 224.0)), (c.H, c.W, 3)).copy()
    base += (np.clip(-c.Y, -1, 1) * 16.0)[..., None] * _col((0.4, 0.5, 0.4))
    base -= (_smoothstep((c.Y - 0.35) / 0.7) * 30.0)[..., None] * _col((0.2, 0.3, 0.6))

    # achenes on a quincunx lattice mapped through the sphere longitude
    seeds = c.zeros()
    dimple = c.zeros()
    r = _rng("straw")
    rows = 11
    for j in range(rows):
        v = -0.84 + j * (1.92 / (rows - 1))
        # local half width from the outline
        s = (v + 0.92) / 2.12
        hw = max((1.0 - s) ** 0.42 * (s + 0.16) ** 0.55 * 1.65, 0.02)
        cols = max(2, int(round(hw * 4.6)))
        for i in range(cols):
            uu = (i + 0.5 + (0.5 if j % 2 else 0.0)) / cols * 2.0 - 1.0
            if abs(uu) > 0.94:
                continue
            u = np.sin(uu * np.pi / 2.0) * hw * 0.97
            jitter = float(r.normal(0, 0.012))
            c.dot(seeds, u + jitter, v + jitter, 0.040, 0.028, np.degrees(np.arctan2(0.4, u)), 1.0)
            c.dot(dimple, u + jitter, v + jitter + 0.030, 0.062, 0.045, 0.0, 1.0)
    dimple = _blur(dimple, 0.9 * SS)
    base -= (dimple * 26.0)[..., None] * _col((0.3, 0.4, 0.9))
    base = over(base, seeds * 0.92, (86.0, 196.0, 246.0))

    rgb = shade_body(c, body, base, radius=0.72, spec=0.62, shin=26.0, rim=0.60,
                     rim_tint=(0.40, 0.30, 0.32),
                     sheen=(-0.34, -0.42, 0.26, 0.16, -0.6, 0.32))
    rgb += (seeds * 0.30 * 40.0)[..., None]
    return _straw_trim(c, rgb), straw_mask(c)


def straw_cross(c: Canvas) -> tuple[np.ndarray, np.ndarray]:
    body = straw_body(c)
    t = c.dist(body)
    f, ang = c.radial(1.05, 1.05, 0.0, 0.10)

    core = _col((214.0, 222.0, 252.0))
    outer = _col((58.0, 58.0, 224.0))
    mix = np.clip((f - 0.06) / 0.72, 0, 1) ** 1.05
    base = core[None, None, :] * (1 - mix[..., None]) + outer[None, None, :] * mix[..., None]
    # white radiating fibres from the core
    r = _rng("strfib")
    bins = 420
    n1 = _blur(r.standard_normal(bins).astype(np.float32)[None, :], 1.3)[0]
    fib = np.interp((ang + np.pi) / (2 * np.pi) * bins, np.arange(bins), n1, period=bins)
    fibf = np.clip(fib.astype(np.float32), 0, None) * np.clip(1.15 - f, 0, 1) ** 0.9
    base = over(base, np.clip(fibf * 0.60, 0, 0.62), (226.0, 232.0, 252.0))
    base += (c.grain("strx", 0.009, 3.0))[..., None]

    rgb = shade_flat(c, body, base, radius=0.72, ambient=0.97, kd=0.10, gloss=0.13)

    ring = 1.0 - _smoothstep((t - 0.026) / 0.030)
    rgb = over(rgb, ring * 0.95, (40.0, 36.0, 206.0))
    rgb += (rind_light(c, body, ring, 0.72, 0.24) * 55.0)[..., None]

    # achene dots just inside the skin
    seeds = c.zeros()
    edge_t = np.clip((t - 0.045) / 0.02, 0, 1)
    rr = _rng("strseed")
    n = 34
    for i in range(n):
        a = i / n * 2 * np.pi
        u = np.cos(a) * 1.03 + float(rr.normal(0, 0.01))
        v = 0.10 + np.sin(a) * 1.03
        c.dot(seeds, u, v, 0.030, 0.022, np.degrees(a), 1.0)
    seeds = seeds * body * (1.0 - edge_t * 0.0)
    rgb = over(rgb, np.clip(seeds * (t < 0.085) * 0.9, 0, 1), (74.0, 186.0, 240.0))

    return _straw_trim(c, rgb), straw_mask(c)


# ---------------------------------------------------------------------------
# banana
# ---------------------------------------------------------------------------

BAN_TILT = np.radians(18.0)
BAN_A, BAN_B = 0.72, 0.92
BAN_TH = 0.38


def _ban_center(n: int = 220):
    th = np.linspace(-1.72, 1.72, n, dtype=np.float32)
    x = BAN_A * np.cos(th)
    y = BAN_B * np.sin(th)
    ca, sa = np.cos(BAN_TILT), np.sin(BAN_TILT)
    return np.stack([x * ca - y * sa, x * sa + y * ca], axis=1), th


def banana_mask(c: Canvas) -> np.ndarray:
    p, th = _ban_center()
    s = (th - th[0]) / (th[-1] - th[0])
    hw = BAN_TH * (1.0 - (2 * s - 1) ** 2) ** 0.40
    hw = np.maximum(hw, 0.012)
    d = np.gradient(p, axis=0)
    ln = np.hypot(d[:, 0], d[:, 1]) + 1e-9
    nx, ny = -d[:, 1] / ln, d[:, 0] / ln
    left = [(p[i, 0] + nx[i] * hw[i], p[i, 1] + ny[i] * hw[i]) for i in range(len(p))]
    right = [(p[i, 0] - nx[i] * hw[i], p[i, 1] - ny[i] * hw[i]) for i in range(len(p) - 1, -1, -1)]
    return c.poly(left + right)


def banana_whole(c: Canvas) -> tuple[np.ndarray, np.ndarray]:
    mask = banana_mask(c)
    p, _ = _ban_center()
    base = np.broadcast_to(_col((48.0, 208.0, 246.0)), (c.H, c.W, 3)).copy()
    # warmer on the inner curve, paler along the outer ridge
    ridge = c.dist(mask)
    base += (np.clip(1.0 - ridge / BAN_TH, 0, 1)[..., None] ** 2) * _col((-10.0, 10.0, -6.0))
    base += (c.grain("ban", 0.010, 3.0))[..., None]

    # longitudinal ridge lines
    lines = c.zeros()
    for off in (-0.36, 0.34):
        pts = []
        d = np.gradient(p, axis=0)
        ln = np.hypot(d[:, 0], d[:, 1]) + 1e-9
        for i in range(0, len(p), 4):
            nx, ny = -d[i, 1] / ln[i], d[i, 0] / ln[i]
            pts.append((p[i, 0] + nx * BAN_TH * off, p[i, 1] + ny * BAN_TH * off))
        c.stroke(lines, pts, 0.022)
    lines = _blur(lines, 0.8 * SS) * mask
    base -= (lines * 22.0)[..., None] * _col((0.2, 0.7, 1.0))

    # brown tips and a green-brown stem
    tipa = c.blob(p[0, 0], p[0, 1], 0.16, 0.16)
    tipb = c.blob(p[-1, 0], p[-1, 1], 0.14, 0.14)
    base = over(base, np.clip(tipa ** 1.6 * 0.95, 0, 1), (26.0, 44.0, 72.0))
    base = over(base, np.clip(tipb ** 1.6 * 0.85, 0, 1), (34.0, 60.0, 92.0))
    spots = c.zeros()
    r = _rng("banspot")
    for i in range(7):
        k = int(r.integers(40, 180))
        off = float(r.uniform(-0.5, 0.5))
        d = np.gradient(p, axis=0)
        ln = np.hypot(d[k, 0], d[k, 1]) + 1e-9
        nx, ny = -d[k, 1] / ln, d[k, 0] / ln
        c.dot(spots, p[k, 0] + nx * BAN_TH * off, p[k, 1] + ny * BAN_TH * off,
              float(r.uniform(0.018, 0.032)), float(r.uniform(0.014, 0.026)),
              float(r.uniform(0, 180)), 1.0)
    base = over(base, _blur(spots, 0.5 * SS) * 0.35, (44.0, 96.0, 140.0))

    rgb = shade_body(c, mask, base, radius=BAN_TH, ambient=0.62, kd=0.60, spec=0.58, shin=22.0,
                     rim=0.58, rim_w=0.055, rim_tint=(0.40, 0.42, 0.34),
                     sheen=(-0.55, -0.30, 0.30, 0.10, 0.9, 0.34))
    return rgb, mask


def banana_face(c: Canvas, rgb: np.ndarray, mask: np.ndarray, y_ss: int) -> np.ndarray:
    return stamp_cut_face(c, rgb, mask, y_ss,
                          flesh=(214.0, 244.0, 252.0), ring=(44.0, 178.0, 220.0),
                          core=(182.0, 228.0, 246.0))


# ---------------------------------------------------------------------------
# pineapple
# ---------------------------------------------------------------------------

PA_RX, PA_RY, PA_CY = 0.78, 0.86, 0.30


def pine_body(c: Canvas) -> np.ndarray:
    f = (np.abs(c.X / PA_RX) ** 2.5 + np.abs((c.Y - PA_CY) / PA_RY) ** 2.7)
    return np.clip((1.0 - f) * PA_RX * c.S * 0.5 + 0.5, 0, 1)


def pine_crown(c: Canvas) -> np.ndarray:
    m = c.zeros()
    specs = ((0.0, -1.30, 0.13), (-0.34, -1.18, 0.12), (0.34, -1.18, 0.12),
             (-0.66, -0.96, 0.11), (0.66, -0.96, 0.11), (-0.92, -0.66, 0.10),
             (0.92, -0.66, 0.10), (-0.16, -1.24, 0.10), (0.18, -1.26, 0.10))
    for ux, uy, wd in specs:
        curve = 0.10 * np.sign(ux if ux != 0 else 1.0)
        m = np.maximum(m, leaf_blade(c, ux * 0.20, -0.50, ux, uy, wd, curve=curve))
    return np.clip(m, 0, 1)


def pine_mask(c: Canvas) -> np.ndarray:
    return np.clip(np.maximum(pine_body(c), pine_crown(c)), 0, 1)


def _pine_crown_paint(c: Canvas, rgb: np.ndarray) -> np.ndarray:
    crown = pine_crown(c)
    base = np.broadcast_to(_col((52.0, 148.0, 92.0)), (c.H, c.W, 3)).copy()
    base += (_smoothstep((-c.Y - 0.5) / 0.9) * 34.0)[..., None] * _col((0.3, 1.0, 0.6))
    base -= (_smoothstep((c.Y + 0.9) / 0.6) * 16.0)[..., None] * _col((0.2, 0.6, 0.4))
    sh = shade_body(c, crown, base, radius=0.12, ambient=0.66, kd=0.60, spec=0.28, shin=18.0,
                    sheen=None, rim=0.55, rim_w=0.035)
    return over(rgb, crown, sh)


def pine_whole(c: Canvas) -> tuple[np.ndarray, np.ndarray]:
    body = pine_body(c)
    lon = c.lon(PA_RX)
    v = (c.Y - PA_CY) / PA_RY

    base = np.broadcast_to(_col((36.0, 168.0, 232.0)), (c.H, c.W, 3)).copy()
    base += (np.clip(1.0 - np.abs(v), 0, 1) * 16.0)[..., None] * _col((0.2, 0.8, 1.0))

    # diamond scale lattice in longitude/height space
    a = (lon * 3.6 + v * 4.4)
    b = (lon * 3.6 - v * 4.4)
    fa, fb = np.abs(((a + 0.5) % 1.0) - 0.5), np.abs(((b + 0.5) % 1.0) - 0.5)
    groove = 1.0 - _smoothstep((np.minimum(fa, fb) - 0.045) / 0.075)
    cell = np.clip((fa + fb) * 1.9, 0, 1)
    base -= (groove * 66.0)[..., None] * _col((0.35, 0.9, 1.0))
    base += ((1.0 - cell) * 34.0)[..., None] * _col((0.25, 0.9, 1.0))
    # brown tuft in the middle of each scale
    tuft = np.clip(1.0 - (fa + fb) * 5.0, 0, 1) ** 1.5
    base = over(base, tuft * 0.55, (34.0, 96.0, 150.0))

    rgb = shade_body(c, body, base, radius=PA_RX, ambient=0.62, kd=0.58, spec=0.42, shin=20.0,
                     rim=0.60, rim_tint=(0.36, 0.32, 0.30),
                     sheen=(-0.38, -0.12, 0.26, 0.30, -0.4, 0.20))
    return _pine_crown_paint(c, rgb), pine_mask(c)


def pine_cross(c: Canvas) -> tuple[np.ndarray, np.ndarray]:
    body = pine_body(c)
    t = c.dist(body)
    f, ang = c.radial(PA_RX, PA_RY, 0.0, PA_CY)

    base = np.broadcast_to(_col((78.0, 202.0, 246.0)), (c.H, c.W, 3)).copy()
    base -= (f[..., None] ** 2.0) * _col((8.0, 24.0, 10.0))
    r = _rng("pafib")
    bins = 500
    n1 = _blur(r.standard_normal(bins).astype(np.float32)[None, :], 1.5)[0]
    fib = np.interp((ang + np.pi) / (2 * np.pi) * bins, np.arange(bins), n1, period=bins)
    base += (fib.astype(np.float32) * 14.0 * np.clip(f, 0.15, 1))[..., None] * _col((0.5, 1.0, 1.0))
    # concentric growth rings
    base -= (np.clip(np.sin(f * 22.0), 0, 1) * 5.0)[..., None]
    base += (c.grain("pax", 0.009, 3.0))[..., None]

    rgb = shade_flat(c, body, base, radius=PA_RX, ambient=0.97, kd=0.10, gloss=0.12)

    # paler fibrous core
    corem = 1.0 - _smoothstep((f - 0.17) / 0.13)
    rgb = over(rgb, corem * 0.60, (140.0, 226.0, 250.0))
    corering = np.clip((1.0 - _smoothstep((f - 0.20) / 0.05)) - corem, 0, 1)
    rgb = over(rgb, corering * 0.45, (58.0, 176.0, 226.0))
    # the "eyes": a ring of dimples where the scales bite into the flesh
    eyes = c.zeros()
    for i in range(16):
        a = i / 16 * 2 * np.pi
        c.dot(eyes, np.cos(a) * PA_RX * 0.74, PA_CY + np.sin(a) * PA_RY * 0.74,
              0.050, 0.038, np.degrees(a), 1.0)
    rgb = over(rgb, _blur(eyes, 0.6 * SS) * 0.42, (44.0, 164.0, 224.0))

    # rind: dark gold band with the scale grooves biting into it
    lon = c.lon(PA_RX)
    v = (c.Y - PA_CY) / PA_RY
    a2, b2 = (lon * 3.6 + v * 4.4), (lon * 3.6 - v * 4.4)
    fa, fb = np.abs(((a2 + 0.5) % 1.0) - 0.5), np.abs(((b2 + 0.5) % 1.0) - 0.5)
    groove = 1.0 - _smoothstep((np.minimum(fa, fb) - 0.045) / 0.070)
    band = 1.0 - _smoothstep((t - 0.055) / 0.055)
    outer = 1.0 - _smoothstep((t - 0.018) / 0.028)
    rgb = over(rgb, band * 0.95, (32.0, 150.0, 214.0))
    rgb = over(rgb, band * groove * 0.7, (22.0, 96.0, 150.0))
    rgb = over(rgb, outer * 0.9, (24.0, 108.0, 168.0))
    rgb += (rind_light(c, body, band, PA_RX, 0.30) * 60.0)[..., None]

    return _pine_crown_paint(c, rgb), pine_mask(c)


# ---------------------------------------------------------------------------
# coconut
# ---------------------------------------------------------------------------

CO_R = 0.94


def coco_mask(c: Canvas) -> np.ndarray:
    return c.ellipse(1.0, CO_R)


def coco_whole(c: Canvas) -> tuple[np.ndarray, np.ndarray]:
    mask = coco_mask(c)
    lon = c.lon(1.0)
    base = np.broadcast_to(_col((48.0, 78.0, 116.0)), (c.H, c.W, 3)).copy()
    base += (np.clip(-c.Y, -1, 1) * 12.0)[..., None] * _col((0.6, 0.9, 1.0))

    # husk fibres: many longitudinal hairs, crowded toward the silhouette
    fib = c.zeros()
    r = _rng("coco")
    for i in range(230):
        u0 = float(r.uniform(-1.0, 1.0))
        v0 = float(r.uniform(-1.0, 1.0))
        ln = float(r.uniform(0.25, 0.85))
        wob = float(r.uniform(-0.16, 0.16))
        pts = []
        for k in range(6):
            s = k / 5.0
            uu = u0 + wob * np.sin(s * np.pi)
            vv = v0 - ln * 0.5 + ln * s
            pts.append((np.sin(np.clip(uu, -1, 1) * np.pi / 2) * 1.0, vv * CO_R))
        c.stroke(fib, pts, float(r.uniform(0.008, 0.018)),
                 float(r.uniform(0.35, 1.0)) * (1 if r.random() > 0.45 else -1))
    fib = _blur(fib, 0.42 * SS)
    base += (fib * 26.0)[..., None] * _col((0.7, 1.0, 1.0))
    base += (c.grain("cocog", 0.008, 5.0))[..., None]

    rgb = shade_body(c, mask, base, radius=CO_R, ambient=0.60, kd=0.60, spec=0.34, shin=16.0,
                     rim=0.60, rim_tint=(0.36, 0.34, 0.32),
                     sheen=(-0.34, -0.40, 0.34, 0.20, -0.6, 0.20))

    # the three germination pores
    eyes = c.zeros()
    for u, v in ((-0.30, -0.42), (0.02, -0.52), (0.30, -0.40)):
        c.dot(eyes, u, v, 0.085, 0.070, 0.0, 1.0)
    rgb = over(rgb, eyes * 0.85, (24.0, 34.0, 52.0))
    rgb += (_blur(eyes, 0.9 * SS) * 12.0)[..., None]
    return rgb, mask


def coco_cross(c: Canvas) -> tuple[np.ndarray, np.ndarray]:
    mask = coco_mask(c)
    t = c.dist(mask)
    f, ang = c.radial(1.0, CO_R)

    # cavity first, then the flesh / shell / husk rings on top
    base = np.broadcast_to(_col((238.0, 246.0, 250.0)), (c.H, c.W, 3)).copy()
    base -= (f[..., None] ** 2.0) * _col((14.0, 14.0, 14.0))
    base += (c.grain("cox", 0.008, 3.0))[..., None]
    rgb = shade_flat(c, mask, base, radius=CO_R, ambient=0.97, kd=0.10, gloss=0.10)

    cav = 1.0 - _smoothstep((f - 0.40) / 0.045)
    cavcol = np.broadcast_to(_col((88.0, 116.0, 148.0)), (c.H, c.W, 3)).copy()
    cavcol += (c.blob(-0.06, 0.16, 0.34, 0.26) * 34.0)[..., None]
    cavcol -= (_smoothstep((-c.Y - 0.02) / 0.42) * 34.0)[..., None] * _col((0.7, 0.9, 1.0))
    rgb = over(rgb, cav * 0.95, cavcol)
    lip = np.clip((1.0 - _smoothstep((f - 0.46) / 0.06)) - cav, 0, 1)
    rgb = over(rgb, lip * 0.55, (168.0, 184.0, 198.0))

    r = _rng("cofib")
    bins = 420
    n1 = _blur(r.standard_normal(bins).astype(np.float32)[None, :], 1.4)[0]
    fibv = np.interp((ang + np.pi) / (2 * np.pi) * bins, np.arange(bins), n1, period=bins).astype(np.float32)

    husk = 1.0 - _smoothstep((t - 0.115) / 0.045)
    shell = 1.0 - _smoothstep((t - 0.052) / 0.030)
    skin = 1.0 - _smoothstep((t - 0.014) / 0.024)
    huskcol = np.broadcast_to(_col((60.0, 96.0, 140.0)), (c.H, c.W, 3)).copy()
    huskcol += (fibv * 16.0)[..., None]
    rgb = over(rgb, husk * 0.96, huskcol)
    rgb = over(rgb, shell * 0.95, (26.0, 40.0, 62.0))
    rgb = over(rgb, skin * 0.9, (44.0, 66.0, 96.0))
    # thin brown seed coat just inside the shell
    coat = np.clip((1.0 - _smoothstep((t - 0.140) / 0.022)) - (1.0 - _smoothstep((t - 0.118) / 0.022)), 0, 1)
    rgb = over(rgb, coat * 0.7, (66.0, 92.0, 126.0))
    rgb += (rind_light(c, mask, husk, CO_R, 0.30) * 55.0)[..., None]
    return rgb, mask


# ---------------------------------------------------------------------------
# bomb
# ---------------------------------------------------------------------------


def make_bomb(w: int = 120, h: int = 160, unit: float = 50.0) -> np.ndarray:
    c = Canvas(w, h, unit, oy=0.12)
    body_cy = 0.40
    body = c.ellipse(1.0, 1.0, 0.0, body_cy)

    cap = np.clip(np.maximum(
        c.poly([(-0.30, body_cy - 0.98), (0.30, body_cy - 0.98),
                (0.24, body_cy - 1.30), (-0.24, body_cy - 1.30)]),
        c.ellipse(0.25, 0.09, 0.0, body_cy - 1.29)), 0, 1)

    # fuse: a rope curving up and to the right
    fuse_pts = [(0.02, body_cy - 1.30), (0.09, body_cy - 1.46), (0.28, body_cy - 1.56),
                (0.47, body_cy - 1.40), (0.55, body_cy - 1.50)]
    fuse = c.zeros()
    c.stroke(fuse, fuse_pts, 0.10)
    fuse = np.clip(fuse, 0, 1)

    mask = np.clip(np.maximum(np.maximum(body, cap), fuse), 0, 1)

    # ---- body -----------------------------------------------------------
    base = np.broadcast_to(_col((30.0, 27.0, 26.0)), (c.H, c.W, 3)).copy()
    base += (c.grain("bombg", 0.010, 3.0))[..., None]
    rgb = shade_body(c, body, base, radius=1.0, ambient=0.55, kd=0.70, spec=0.95, shin=40.0,
                     rim=0.35, rim_w=0.08, rim_tint=(0.45, 0.45, 0.48),
                     sheen=(-0.36, body_cy - 0.42, 0.34, 0.24, -0.6, 0.30))
    # a hard little catch light plus the classic long streak
    rgb += (c.blob(-0.38, body_cy - 0.44, 0.17, 0.12, -0.6) * body * 210.0)[..., None]
    rgb += (c.blob(0.44, body_cy + 0.42, 0.30, 0.10, -0.7) * body * 42.0)[..., None]
    rgb += (c.blob(-0.10, body_cy + 0.70, 0.44, 0.12, 0.15) * body * 26.0)[..., None]

    # ---- cap ------------------------------------------------------------
    capbase = np.broadcast_to(_col((92.0, 96.0, 104.0)), (c.H, c.W, 3)).copy()
    capbase += (_smoothstep(((c.Y - body_cy) + 1.30) / 0.30) * -30.0)[..., None]
    caprgb = shade_body(c, cap, capbase, radius=0.26, ambient=0.55, kd=0.66, spec=0.75, shin=26.0,
                        rim=0.55, rim_w=0.05, rim_tint=(0.35, 0.35, 0.38),
                        sheen=(-0.14, body_cy - 1.16, 0.10, 0.14, 0.0, 0.35))
    band = np.clip(1.0 - np.abs(((c.Y - body_cy) + 1.14) / 0.045), 0, 1)
    caprgb -= (band * cap * 40.0)[..., None]
    rgb = over(rgb, cap, caprgb)

    # ---- fuse -----------------------------------------------------------
    fusebase = np.broadcast_to(_col((78.0, 104.0, 128.0)), (c.H, c.W, 3)).copy()
    twist = c.zeros()
    for i in range(14):
        s = i / 13.0
        k = int(s * (len(fuse_pts) - 1) * 8)
        # sample the polyline densely for the twist ticks
        tt = np.linspace(0, len(fuse_pts) - 1, 40)
        xs = np.interp(tt, np.arange(len(fuse_pts)), [p[0] for p in fuse_pts])
        ys = np.interp(tt, np.arange(len(fuse_pts)), [p[1] for p in fuse_pts])
        j = min(int(s * 39), 38)
        dx, dy = xs[j + 1] - xs[j], ys[j + 1] - ys[j]
        ln = np.hypot(dx, dy) + 1e-6
        nx, ny = -dy / ln, dx / ln
        c.stroke(twist, [(xs[j] - nx * 0.07 - dx, ys[j] - ny * 0.07 - dy),
                         (xs[j] + nx * 0.07 + dx, ys[j] + ny * 0.07 + dy)], 0.022)
    fusebase -= (np.clip(twist, 0, 1) * 34.0)[..., None]
    fusergb = shade_body(c, fuse, fusebase, radius=0.055, ambient=0.60, kd=0.62, spec=0.4,
                         shin=18.0, sheen=None, rim=0.55, rim_w=0.03,
                         rim_tint=(0.40, 0.40, 0.42))
    rgb = over(rgb, fuse, fusergb)

    # ---- spark ----------------------------------------------------------
    sx, sy = fuse_pts[-1]
    spark = c.zeros()
    r = _rng("spark")
    for i in range(11):
        a = i / 11 * 2 * np.pi + 0.2
        ln = 0.15 + 0.12 * float(r.random())
        c.stroke(spark, [(sx, sy), (sx + np.cos(a) * ln, sy + np.sin(a) * ln)], 0.030)
    spark = _blur(np.clip(spark, 0, 1), 0.5 * SS)
    glow_o = c.blob(sx, sy, 0.36, 0.36)
    glow_i = c.blob(sx, sy, 0.185, 0.185)
    core = c.blob(sx, sy, 0.095, 0.095)

    flame = np.zeros_like(rgb)
    flame = over(flame, np.clip(glow_o * 0.9, 0, 1), (30.0, 120.0, 255.0))
    flame = over(flame, np.clip(glow_i, 0, 1), (90.0, 210.0, 255.0))
    flame = over(flame, np.clip(spark * 0.9 + core, 0, 1), (235.0, 250.0, 255.0))
    a_flame = np.clip(glow_o ** 2.6 * 0.85 + glow_i ** 1.4 + spark * 0.9 + core * 1.2, 0, 1)

    alpha = np.clip(np.maximum(mask, a_flame), 0, 1)
    assert_inside(c, alpha, "bomb", margin=2)
    rgb = over(rgb, a_flame * np.clip(1.0 - mask * 0.0, 0, 1), flame)
    rgb = rim_key(c, mask, rgb, width=0.045, strength=0.30, tint=(0.55, 0.55, 0.60))
    # warm bounce from the spark onto the cap and the top of the body
    warm = _blur(glow_o, 1.2 * SS) * mask
    rgb += (warm * 34.0)[..., None] * _col((0.2, 0.6, 1.0))
    return finish(rgb, alpha, w, h, "bomb")


# ---------------------------------------------------------------------------
# fruit registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fruit:
    name: str
    w: int
    h: int
    unit: float
    whole: Callable[[Canvas], tuple[np.ndarray, np.ndarray]]
    cross: Callable[[Canvas], tuple[np.ndarray, np.ndarray]] | None
    radius: int
    juice: tuple[int, int, int]
    score: int = 1
    cut_v: float = 0.0  # cut height in unit coords
    oy: float = 0.0  # unit-space origin offset, to centre lopsided fruit
    ox: float = 0.0
    face: Callable | None = None  # solid-cut fruit: stamp an oval cut face


FRUITS: tuple[Fruit, ...] = (
    Fruit("watermelon", 140, 128, 60.0, watermelon_whole, watermelon_cross,
          radius=57, juice=(58, 42, 226), cut_v=0.0),
    Fruit("orange", 136, 130, 58.0, orange_whole, orange_cross,
          radius=56, juice=(30, 142, 250), cut_v=0.0),
    Fruit("apple", 136, 140, 56.0, apple_whole, apple_cross,
          radius=54, juice=(46, 46, 214), cut_v=0.06, oy=0.13),
    Fruit("lime", 134, 106, 52.0, lime_whole, lime_cross,
          radius=46, juice=(72, 212, 148), cut_v=0.0),
    Fruit("banana", 114, 160, 62.0, banana_whole, None,
          radius=44, juice=(112, 232, 250), cut_v=0.0, face=banana_face, ox=-0.355, oy=-0.067),
    Fruit("strawberry", 126, 140, 46.0, straw_whole, straw_cross,
          radius=47, juice=(66, 48, 232), cut_v=0.10, oy=0.08),
    Fruit("pineapple", 120, 152, 54.0, pine_whole, pine_cross,
          radius=45, juice=(86, 212, 248), score=2, cut_v=0.30, oy=0.07),
    Fruit("coconut", 136, 130, 58.0, coco_whole, coco_cross,
          radius=55, juice=(232, 240, 246), score=2, cut_v=0.0),
)


def render_fruit(f: Fruit) -> dict[str, np.ndarray]:
    c1 = Canvas(f.w, f.h, f.unit, f.oy, f.ox)
    rgb, mask = f.whole(c1)
    assert_inside(c1, mask, f.name)
    rgb = rim_key(c1, mask, rgb, width=0.035, strength=0.45)
    whole = finish(rgb * mask[..., None], mask, f.w, f.h, f.name + "w")

    c2 = Canvas(f.w, f.h, f.unit, f.oy, f.ox)
    if f.cross is not None:
        crgb, cmask = f.cross(c2)
    else:
        crgb, cmask = f.whole(c2)

    y_ss = int(round((c2.cy + f.cut_v * c2.S) / SS)) * SS
    y_px = y_ss // SS
    if not (8 < y_px < f.h - 8):
        raise RuntimeError(f"{f.name}: cut line out of range")

    out: dict[str, np.ndarray] = {"whole": whole}
    for key, above in (("half_a", True), ("half_b", False)):
        m = cmask.copy()
        if above:
            m[y_ss:] = 0.0
            rows = (0, y_px + CUT_PAD)
        else:
            m[:y_ss] = 0.0
            rows = (y_px - CUT_PAD, f.h)
        g = crgb
        if f.face is not None:
            g = f.face(c2, g.copy(), cmask, y_ss)
        g = cut_lip(c2, g, m, y_ss, above)
        g = rim_key(c2, m, g, width=0.035, strength=0.45)
        out[key] = finish(g * m[..., None], m, f.w, f.h, f.name + key, rows=rows)
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Fruit Ninja fruit sprites.")
    ap.add_argument("--out", type=Path, default=OUT_DIR, help="output directory")
    ap.add_argument("--only", default=None, help="render just one fruit (debug)")
    args = ap.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, str, float]] = []

    def write(name: str, img: np.ndarray) -> None:
        path = out_dir / name
        if not cv2.imwrite(str(path), img):
            raise RuntimeError(f"failed to write {path}")
        h, w = img.shape[:2]
        rows.append((name, f"{w}x{h}", path.stat().st_size / 1024.0))

    manifest = {"fruits": []}
    for f in FRUITS:
        if args.only and f.name != args.only:
            continue
        imgs = render_fruit(f)
        write(f"fruit_{f.name}_whole.png", imgs["whole"])
        write(f"fruit_{f.name}_half_a.png", imgs["half_a"])
        write(f"fruit_{f.name}_half_b.png", imgs["half_b"])
        manifest["fruits"].append({
            "name": f.name,
            "radius": int(f.radius),
            "juice": [int(v) for v in f.juice],
            "score": int(f.score),
        })

    write("bomb.png", make_bomb())

    if not args.only:
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    width = max(len(r[0]) for r in rows)
    print(f"\nwrote {len(rows)} sprites to {out_dir}\n")
    print(f"{'name'.ljust(width)}  {'size':>10}  {'KB':>7}")
    print("-" * (width + 22))
    for name, size, kb in rows:
        print(f"{name.ljust(width)}  {size:>10}  {kb:>7.1f}")
    print("-" * (width + 22))
    print(f"{'total'.ljust(width)}  {'':>10}  {sum(r[2] for r in rows):>7.1f}\n")


if __name__ == "__main__":
    main()
