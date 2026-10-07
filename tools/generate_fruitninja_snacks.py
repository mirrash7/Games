#!/usr/bin/env python3
"""Procedurally generate the snack, eaten-piece and trash-bag sprites for Snack Attack.

Run once (or whenever the look changes):

    uv run python tools/generate_fruitninja_snacks.py

Snack Attack is the Fruit Ninja engine re-themed: the player's hand is a
raccoon that swipes through flying snacks to gobble them and must avoid bags of
trash. The asset schema keeps the fruit names (see ``kpapp.game.fruitninja.art``):
a "fruit" is a snack, ``half_a`` is the snack with a bite taken out of it,
``half_b`` is the bitten-off chunk, ``juice`` is the crumb colour, and
``bomb.png`` is the trash bag.

Everything here is numpy + opencv only - no downloads, no binary sources, no
extra dependencies, no reference art of any kind. Output goes to
``assets/fruitninja/generated/`` together with ``manifest.json``.

Look target: bright, glossy, chunky arcade snacks that read at ~110 px from
two metres away. Every sprite is a shaded solid, not a flat vector: a diffuse
term from a fake surface normal, a Blinn specular, a soft sheen blob, a bounce
light on the shadow rim, a darker rim keyline so the sprite separates from a
busy background, and a small drop shadow.

Technique notes:

* Geometry is written in *unit* coordinates (1.0 = the snack's nominal radius)
  and rasterised through :class:`Canvas`, which supersamples by ``SS`` and box
  filters back down. Readable maths, clean edges.
* The fake normal comes from the distance transform of the silhouette
  (:func:`shade_body`): ``z = sqrt(t(2-t))`` with the horizontal component
  along the (blurred) distance gradient. That shades *any* silhouette like a
  rounded solid. A ``bump`` term tilts it further, from a height field (folds,
  cracks) or from the lobes of a metaball (cotton-candy puffs, a lumpy bag).
* Eating: every snack is drawn by one function ``draw(canvas, keep)``. ``keep``
  is None for the whole snack, or a coverage mask - the bite region for the
  bitten-off chunk, its complement for the bitten snack - so the two pieces
  are cut from the very same drawing and fit back together exactly. A bite is a
  jaw circle ringed with tooth circles: the snack keeps a row of scalloped tooth
  marks, and the exposed band along the cut is painted with the snack's inside
  (dough, potato, grape flesh). Cotton candy tears instead of biting.
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
# extra shading tools
# ---------------------------------------------------------------------------

LN2 = float(np.log(2.0))


def lit(base: np.ndarray, nx, ny, nz, *, ambient: float = 0.60, kd: float = 0.62,
        spec: float = 0.5, shin: float = 24.0) -> np.ndarray:
    """Lambert + Blinn for an explicit normal field."""
    lam = nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2]
    out = base * (ambient + kd * np.clip(lam, 0.0, 1.0))[..., None]
    s = np.clip(nx * HALFV[0] + ny * HALFV[1] + nz * HALFV[2], 0.0, 1.0) ** shin
    return out + (s * spec * 255.0)[..., None]


def sphere_normal(c: Canvas, u: float, v: float, ru: float, rv: float | None = None):
    rv = ru if rv is None else rv
    nx, ny = (c.X - u) / ru, (c.Y - v) / rv
    nz = np.sqrt(np.clip(1.0 - nx * nx - ny * ny, 0.0, 1.0))
    return nx, ny, nz


def bump(c: Canvas, height: np.ndarray, strength: float):
    """Normal tilt (x, y) from a height field given in units."""
    gx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3) / 8.0 * c.S
    gy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3) / 8.0 * c.S
    return -gx * strength, -gy * strength


def metaball(c: Canvas, lobes, sharp: float = 1.0):
    """Blob of gaussian lobes: AA mask plus the field and its gradient.

    Each lobe ``(u, v, r)`` on its own crosses the 0.5 threshold at radius
    ``r``. ``sharp`` = 1 sums the lobes (soft, merged creases); larger values
    approach a union of circles, keeping a bumpy, cloud-like outline.
    """
    acc, ax, ay = c.zeros(), c.zeros(), c.zeros()
    p = float(sharp)
    for u, v, r in lobes:
        dx, dy = c.X - u, c.Y - v
        k = LN2 / (r * r)
        g = np.exp(-(dx * dx + dy * dy) * k)
        gp1 = g ** (p - 1.0) if p != 1.0 else 1.0
        acc += g ** p if p != 1.0 else g
        ax -= 2.0 * k * dx * g * gp1
        ay -= 2.0 * k * dy * g * gp1
    f = np.maximum(acc, 1e-12) ** (1.0 / p)
    scale = f ** (1.0 - p) if p != 1.0 else 1.0
    fx, fy = ax * scale, ay * scale
    gn = np.sqrt(fx * fx + fy * fy) + 1e-6
    mask = np.clip((f - 0.5) / gn * c.S + 0.5, 0.0, 1.0)
    return mask, f, fx, fy


def lump_tilt(f: np.ndarray, fx: np.ndarray, fy: np.ndarray, amount: float):
    """Normal tilt that rounds every metaball lobe on its own."""
    ff = np.maximum(f, 1e-3)
    return (np.clip(-fx / ff * amount, -1.2, 1.2), np.clip(-fy / ff * amount, -1.2, 1.2))


def shade_bumped(c: Canvas, mask: np.ndarray, base: np.ndarray, tilt, *, radius: float,
                 ambient: float = 0.60, kd: float = 0.62, spec: float = 0.5, shin: float = 24.0,
                 bounce: float = 0.30) -> np.ndarray:
    """``shade_body`` lighting with an extra normal tilt (no sheen, no rim)."""
    (nx, ny, nz), t = surface_normal(c, mask, radius)
    nx, ny = nx + tilt[0], ny + tilt[1]
    n = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6
    nx, ny, nz = nx / n, ny / n, nz / n
    lam = nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2]
    out = base * (ambient + kd * np.clip(lam, 0.0, 1.0))[..., None]
    b = np.clip(-lam, 0.0, 1.0) * (1.0 - t) ** 2.4
    out += (b * bounce * 105.0)[..., None] * _col((1.05, 0.95, 0.88))
    s = np.clip(nx * HALFV[0] + ny * HALFV[1] + nz * HALFV[2], 0.0, 1.0) ** shin
    return out + (s * spec * 255.0)[..., None]


def field(c: Canvas, color) -> np.ndarray:
    return np.broadcast_to(_col(color), (c.H, c.W, 3)).copy()


def vgrad(c: Canvas, top, bottom, v0: float, v1: float) -> np.ndarray:
    """Vertical colour ramp from ``top`` at v0 to ``bottom`` at v1."""
    s = np.clip((c.Y - v0) / (v1 - v0), 0.0, 1.0)[..., None]
    return _col(top)[None, None, :] * (1.0 - s) + _col(bottom)[None, None, :] * s


# ---------------------------------------------------------------------------
# bites
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Bite:
    """A jaw circle at (u, v), radius r, ringed with tooth scallops."""

    u: float
    v: float
    r: float
    tooth: float = 0.12
    phase: float = 0.0  # rotates the tooth ring (radians)
    torn: bool = False  # cotton candy: a ragged tear instead of teeth


def bite_region(c: Canvas, b: Bite, tag: str) -> np.ndarray:
    if b.torn:
        d = np.hypot(c.X - b.u, c.Y - b.v) - b.r
        d = d + c.grain(tag + "tear", 0.05, 0.05) + c.grain(tag + "tear2", 0.02, 0.010)
        return np.clip(-d * c.S + 0.5, 0.0, 1.0)
    m = c.ellipse(b.r, b.r, b.u, b.v)
    # Centres closer than a tooth diameter overlap into sharp cusps: the food
    # keeps a row of concave scallops, the chunk a row of bumps.
    n = max(8, int(round(2.0 * np.pi * b.r / (b.tooth * 1.7))))
    for i in range(n):
        a = b.phase + i / n * 2.0 * np.pi
        m = np.maximum(m, c.ellipse(b.tooth, b.tooth, b.u + np.cos(a) * b.r, b.v + np.sin(a) * b.r))
    return m


def bitten(c: Canvas, rgb: np.ndarray, mask: np.ndarray, keep: np.ndarray | None,
           inner: np.ndarray, band: float = 0.10) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Clip a drawing to ``keep`` and paint the exposed inside along the cut.

    Returns (rgb, mask, d) where ``d`` is the distance from the cut in units
    (None for an unbitten snack), for snacks that add their own layers.
    """
    if keep is None:
        return rgb, mask, None
    m = mask * keep
    d = c.dist(keep)
    face = (1.0 - _smoothstep((d - band * 0.62) / (band * 0.38))) * m
    # the bitten face is a hollow: darker deep in the cut, lighter at its lip
    depth = np.clip(d / band, 0.0, 1.0)
    shade = (0.80 + 0.26 * depth)[..., None]
    rgb = over(rgb, face, inner * shade)
    lip = np.exp(-(((d - band * 0.80) / (band * 0.20)) ** 2)) * m
    rgb = rgb + (lip * 26.0)[..., None]
    edge = (1.0 - _smoothstep(d / 0.030)) * m
    rgb = rgb * (1.0 - (edge * 0.30)[..., None])
    return rgb, m, d


def crumb_field(c: Canvas, base, tag: str, holes=(0.0, 0.0, 0.0), amt: float = 0.5) -> np.ndarray:
    """Inside of a baked thing: base colour with air pockets and grain."""
    f = field(c, base)
    g = c.grain(tag + "crumb", 0.012, 1.0)
    f += (g * 9.0)[..., None]
    pockets = np.clip((c.grain(tag + "pock", 0.010, 1.0) - 1.1) * 1.6, 0.0, 1.0)
    return over(f, pockets * amt, holes)


# ---------------------------------------------------------------------------
# cotton candy
# ---------------------------------------------------------------------------

CONE_TOP, CONE_TIP, CONE_W = 0.30, 1.42, 0.40


@dataclass(frozen=True)
class Candy:
    tag: str
    lobes: tuple
    light: tuple  # BGR, lit side of the fluff
    deep: tuple  # BGR, shaded underside
    stripe: tuple  # BGR, paper-cone stripes
    swirl: tuple | None = None  # BGR, a second spun colour


def cone_mask(c: Canvas) -> np.ndarray:
    m = c.poly([(-CONE_W, CONE_TOP), (CONE_W, CONE_TOP), (0.045, CONE_TIP - 0.03), (-0.045, CONE_TIP - 0.03)])
    m = np.maximum(m, c.ellipse(0.055, 0.045, 0.0, CONE_TIP - 0.05))
    return np.clip(m, 0.0, 1.0)


def paint_cone(c: Canvas, cone: np.ndarray, stripe) -> np.ndarray:
    hw = CONE_W * np.clip((CONE_TIP - c.Y) / (CONE_TIP - CONE_TOP), 0.05, 1.2)
    lon = np.arcsin(np.clip(c.X / hw, -1.0, 1.0)) / (np.pi / 2.0)
    p = lon * 0.55 + (c.Y - CONE_TOP) * 2.1
    band = _smoothstep((np.sin(p * 2.0 * np.pi) - 0.05) / 0.25)
    base = over(field(c, (232.0, 240.0, 246.0)), band * 0.92, stripe)
    base += (c.grain("cone", 0.01, 2.5))[..., None]
    # paper seam spiralling down the cone
    seam = np.exp(-(((np.sin((p + 0.25) * np.pi) ) / 0.05) ** 2)) * (lon > -0.2)
    base *= (1.0 - 0.10 * seam)[..., None]
    rgb = shade_body(c, cone, base, radius=0.30, ambient=0.66, kd=0.55, spec=0.22, shin=14.0,
                     sheen=(-0.15, 0.60, 0.08, 0.30, 0.2, 0.18), rim=0.50, rim_w=0.07)
    return rgb


def fibres(c: Canvas, tag: str, region: np.ndarray, centre: tuple[float, float], n: int = 1100) -> np.ndarray:
    """Spun-sugar strands swirling round ``centre``; 0..1 coverage."""
    r = _rng(tag + "fib")
    layer = c.zeros()
    ys, xs = np.nonzero(region > 0.02)
    u_lo, u_hi = (xs.min() - c.cx) / c.S, (xs.max() - c.cx) / c.S
    v_lo, v_hi = (ys.min() - c.cy) / c.S, (ys.max() - c.cy) / c.S
    cu, cv = centre
    for _ in range(n):
        u0 = float(r.uniform(u_lo, u_hi))
        v0 = float(r.uniform(v_lo, v_hi))
        val = float(r.uniform(0.45, 1.0))
        ln = float(r.uniform(0.10, 0.28))
        wd = float(r.uniform(0.008, 0.020))
        jit = float(r.normal(0.0, 0.30))
        curv = float(r.normal(0.0, 0.25))
        px, py = c.ipx(u0, v0)
        if not (0 <= px < c.W and 0 <= py < c.H) or region[py, px] < 0.3:
            continue
        ox, oy = u0 - cu, v0 - cv
        rad = np.hypot(ox, oy) + 1e-6
        ang = np.arctan2(oy, ox) + np.pi / 2.0 - 0.12 + jit  # tangent, leaning outward a little
        pts = [(u0, v0)]
        uu, vv = u0, v0
        for _k in range(7):
            ang += curv * 0.3
            uu += np.cos(ang) * ln / 7.0
            vv += np.sin(ang) * ln / 7.0
            pts.append((uu, vv))
        c.stroke(layer, pts, wd, val)
    return np.clip(_blur(layer, 0.30 * SS), 0.0, 1.0)


def fluff_alpha(c: Canvas, solid: np.ndarray, fib: np.ndarray, reach: float = 0.07) -> np.ndarray:
    """Soft, wispy coverage for a fluffy silhouette."""
    din, dout = c.dist(solid), c.dist(1.0 - solid)
    sd = din - dout  # signed distance, + inside
    core = _smoothstep((sd + 0.02) / 0.06)
    edge_zone = 1.0 - _smoothstep(sd / 0.09)
    core = core * (1.0 - edge_zone * 0.45 * (1.0 - fib))
    fr = np.clip(1.0 - (-sd) / reach, 0.0, 1.0) * (sd < 0.03)
    wisp = fib * fr ** 1.4 * 0.80
    return np.clip(np.maximum(core, wisp), 0.0, 1.0)


def candy_draw(cd: Candy, c: Canvas, keep: np.ndarray | None):
    cone = cone_mask(c)
    shape, f, fx, fy = metaball(c, cd.lobes, sharp=4.0)
    ys = np.nonzero(shape.max(axis=1) > 0.5)[0]
    centre = (0.0, ((ys.min() + ys.max()) / 2.0 - c.cy) / c.S)
    fib = fibres(c, cd.tag, shape, centre)

    if keep is not None:
        cone = cone * keep
        shape = shape * keep
    cone_rgb = paint_cone(c, cone, cd.stripe)
    # the puff sits on the cone: soft contact shadow under it
    m = np.float32([[1, 0, 0], [0, 1, 0.06 * c.S]])
    occl = cv2.warpAffine(_blur(shape, 0.05 * c.S), m, (c.W, c.H))
    cone_rgb *= (1.0 - 0.45 * occl)[..., None]

    # colour: lit top, deeper underside, optional spun swirl
    base = vgrad(c, cd.light, cd.deep, centre[1] - 0.8, centre[1] + 0.9)
    if cd.swirl is not None:
        ang = np.arctan2(c.Y - centre[1], c.X - centre[0])
        rad = np.hypot(c.X - centre[0], c.Y - centre[1])
        sw = np.sin(ang * 2.0 + rad * 7.5 + 0.6)
        sw = _smoothstep((sw - 0.15) / 0.55) * np.clip(1.2 - rad * 0.4, 0, 1)
        base = over(base, _blur(sw, 0.03 * c.S) * 0.85, cd.swirl)

    tilt = lump_tilt(f, fx, fy, 0.30)
    rgb_f = shade_bumped(c, shape, base, tilt, radius=0.55, ambient=0.70, kd=0.46,
                         spec=0.06, shin=8.0, bounce=0.45)
    # soft sugar sheen high on the puff
    rgb_f += (c.blob(centre[0] - 0.30, centre[1] - 0.42, 0.40, 0.26, -0.5) * shape * 30.0)[..., None]
    # spun strands: light threads and darker gaps
    rgb_f += ((fib - 0.30) * 20.0)[..., None]
    # translucent, paler fringe (sugar catches the light at the edge)
    din = c.dist(shape)
    fringe = 1.0 - _smoothstep(din / 0.12)
    rgb_f = over(rgb_f, fringe * 0.35, tuple(min(255.0, v * 0.5 + 140.0) for v in cd.light))

    a_f = fluff_alpha(c, shape, fib)
    rgb = over(cone_rgb, a_f, rgb_f)
    alpha = np.clip(np.maximum(cone, a_f), 0.0, 1.0)
    return rgb, alpha


# ---------------------------------------------------------------------------
# frosted donut
# ---------------------------------------------------------------------------

DN_RY = 0.90
DN_HOLE = 0.30


def donut_coords(c: Canvas):
    f = np.sqrt(c.X ** 2 + (c.Y / DN_RY) ** 2)
    th = np.arctan2(c.Y / DN_RY, c.X)
    t = np.clip((f - DN_HOLE) / (1.0 - DN_HOLE), 0.0, 1.0)
    return f, th, t


def donut_draw(c: Canvas, keep: np.ndarray | None):
    outer = c.ellipse(1.0, DN_RY)
    hole = c.ellipse(DN_HOLE, DN_HOLE * DN_RY)
    mask = np.clip(outer - hole, 0.0, 1.0)
    f, th, t = donut_coords(c)
    # torus normal: inner side faces the hole, outer side faces out
    fsafe = np.maximum(f, 1e-3)
    dx, dy = c.X / fsafe, (c.Y / DN_RY) / fsafe
    nr = -np.cos(np.pi * t) * 0.85
    nx, ny, nz = nr * dx, nr * dy, np.sin(np.pi * t) + 0.12
    n = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / n, ny / n, nz / n

    dough = field(c, (62.0, 146.0, 218.0))
    dough = over(dough, np.clip(1.0 - np.abs(t - 0.5) / 0.5, 0, 1) ** 2 * 0.6, (78.0, 160.0, 226.0))
    dough = over(dough, _smoothstep((t - 0.80) / 0.15) * 0.75, (40.0, 104.0, 176.0))
    # the pale fried "equator" band
    eq = np.exp(-(((t - 0.93) / 0.035) ** 2))
    dough = over(dough, eq * 0.55, (150.0, 206.0, 240.0))
    dough += (c.grain("dough", 0.010, 4.0))[..., None]
    rgb = lit(dough, nx, ny, nz, ambient=0.55, kd=0.62, spec=0.15, shin=12.0)

    # icing with wobbly edges and drips
    r = _rng("icing")
    ti1 = 0.80 + 0.035 * np.sin(6.0 * th + 0.4) + 0.020 * np.sin(11.0 * th + 2.0)
    for _ in range(6):
        a0 = float(r.uniform(0.1, np.pi - 0.1))  # lower half (y down)
        w = float(r.uniform(0.10, 0.18))
        dth = np.angle(np.exp(1j * (th - a0)))
        ti1 = ti1 + float(r.uniform(0.07, 0.13)) * np.exp(-((dth / w) ** 2))
    ti0 = 0.11 + 0.025 * np.sin(5.0 * th + 1.0)
    icing = _smoothstep((t - ti0) / 0.035) * (1.0 - _smoothstep((t - ti1) / 0.030)) * mask
    # icing is a raised layer: drop shadow onto the dough + rounded edges
    sh = _blur(icing, 0.03 * c.S)
    m = np.float32([[1, 0, 0.025 * c.S], [0, 1, 0.035 * c.S]])
    sh = cv2.warpAffine(sh, m, (c.W, c.H))
    rgb *= (1.0 - 0.45 * sh * (1.0 - icing))[..., None]

    ice = field(c, (196.0, 136.0, 255.0))
    ice = over(ice, _smoothstep((t - 0.55) / 0.4) * 0.35, (170.0, 104.0, 240.0))
    height = _blur(icing, 0.045 * c.S) * 0.05
    bx, by = bump(c, height, 1.0)
    inx, iny = nx * 0.75 + bx, ny * 0.75 + by
    inz = nz
    nn = np.sqrt(inx * inx + iny * iny + inz * inz)
    ice_rgb = lit(ice, inx / nn, iny / nn, inz / nn, ambient=0.66, kd=0.45, spec=0.80, shin=42.0)
    ice_rgb += (c.blob(-0.46, -0.48, 0.30, 0.12, -0.7) * 70.0)[..., None]
    rgb = over(rgb, icing, ice_rgb)

    # sprinkles
    cols = [(60.0, 220.0, 255.0), (248.0, 248.0, 248.0), (240.0, 170.0, 60.0),
            (100.0, 215.0, 90.0), (40.0, 150.0, 255.0), (230.0, 120.0, 150.0)]
    layers = [c.zeros() for _ in cols]
    hi = c.zeros()
    rr = _rng("sprinkle")
    placed = 0
    for _ in range(400):
        if placed >= 34:
            break
        u = float(rr.uniform(-0.95, 0.95))
        v = float(rr.uniform(-0.88, 0.88))
        ff = np.hypot(u, v / DN_RY)
        tt = (ff - DN_HOLE) / (1.0 - DN_HOLE)
        if not (0.20 < tt < 0.72):
            continue
        a = float(rr.uniform(0, np.pi))
        ln = 0.075
        p0 = (u - np.cos(a) * ln, v - np.sin(a) * ln)
        p1 = (u + np.cos(a) * ln, v + np.sin(a) * ln)
        k = placed % len(cols)
        c.stroke(layers[k], [p0, p1], 0.060)
        c.stroke(hi, [(p0[0] - 0.012, p0[1] - 0.014), (u - 0.012, v - 0.014)], 0.020)
        placed += 1
    allsp = np.clip(sum(layers), 0, 1)
    rgb *= (1.0 - 0.35 * _blur(allsp, 0.02 * c.S) * icing)[..., None]  # tiny contact shadow
    for k, col in enumerate(cols):
        rgb = over(rgb, layers[k] * icing, (col[0] * 0.92, col[1] * 0.92, col[2] * 0.92))
    rgb += (hi * allsp * icing * 90.0)[..., None]

    rgb = rim_key(c, mask, rgb, width=0.05, strength=0.45)
    inner = crumb_field(c, (168.0, 214.0, 244.0), "donut", holes=(110.0, 165.0, 210.0), amt=0.6)
    rgb, mask, d = bitten(c, rgb, mask, keep, inner, band=0.11)
    return rgb, mask


# ---------------------------------------------------------------------------
# pizza slice
# ---------------------------------------------------------------------------

PZ_TIP = (0.0, 1.02)
PZ_X = 0.90
PZ_TOP = -0.52


def _pz_arc(x):
    return PZ_TOP - 0.24 * (1.0 - (x / PZ_X) ** 2)


def pizza_body(c: Canvas) -> np.ndarray:
    xs = np.linspace(-PZ_X, PZ_X, 40)
    pts = [(x, _pz_arc(x)) for x in xs] + [(0.06, PZ_TIP[1] - 0.02), (-0.06, PZ_TIP[1] - 0.02)]
    return c.poly(pts)


def pizza_crust(c: Canvas) -> np.ndarray:
    xs = np.linspace(-PZ_X - 0.02, PZ_X + 0.02, 40)
    m = c.zeros()
    c.stroke(m, [(x, _pz_arc(x) - 0.02) for x in xs], 0.27)
    return np.clip(m, 0, 1)


def pizza_draw(c: Canvas, keep: np.ndarray | None):
    body = pizza_body(c)
    crust = pizza_crust(c)
    t = c.dist(body)

    # bread edge -> sauce ring -> melted cheese on top
    rgb = shade_body(c, body, field(c, (110.0, 176.0, 226.0)), radius=0.12, ambient=0.62,
                     kd=0.55, spec=0.2, shin=14.0, sheen=None, rim=0.5, rim_w=0.05)
    sauce = 1.0 - _smoothstep((0.030 - t) / 0.012)
    rgb = over(rgb, sauce * body, (40.0, 52.0, 196.0))
    wob = c.grain("cheesewob", 0.05, 0.022)
    cheese = _smoothstep((t - 0.075 + wob) / 0.02) * body
    cbase = field(c, (80.0, 206.0, 255.0))
    brown = np.clip((c.grain("cheesebrown", 0.06, 1.0) - 0.7) * 1.2, 0, 1)
    cbase = over(cbase, brown * 0.55, (50.0, 150.0, 236.0))
    cbase += (c.grain("cheesefine", 0.015, 7.0))[..., None]
    bub = np.clip((c.grain("cheesebub", 0.025, 1.0) - 1.2) * 2.0, 0, 1)
    hb = bump(c, _blur(bub, 0.02 * c.S) * 0.03 + _blur(cheese, 0.03 * c.S) * 0.03, 1.0)
    crgb = shade_bumped(c, cheese, cbase, hb, radius=0.10, ambient=0.70, kd=0.40, spec=0.55,
                        shin=30.0, bounce=0.2)
    crgb += (c.blob(-0.35, -0.10, 0.32, 0.20, -0.8) * 34.0)[..., None]
    rgb = over(rgb, cheese, crgb)

    # pepperoni
    pep = c.zeros()
    rgb_p = rgb.copy()
    for u, v, rad in ((-0.38, -0.24, 0.19), (0.37, -0.20, 0.18), (0.0, 0.28, 0.17), (0.02, 0.76, 0.115)):
        disc = c.ellipse(rad, rad * 0.94, u, v)
        sh = cv2.warpAffine(_blur(disc, 0.03 * c.S), np.float32([[1, 0, 0.02 * c.S], [0, 1, 0.03 * c.S]]),
                            (c.W, c.H))
        rgb_p *= (1.0 - 0.40 * sh * (1 - disc))[..., None]
        nx, ny, nz = sphere_normal(c, u, v, rad * 1.9)
        pb = field(c, (44.0, 48.0, 186.0))
        fat = np.clip((c.grain("pepfat" + str(u), 0.018, 1.0) - 1.0) * 1.5, 0, 1)
        pb = over(pb, fat * 0.5, (70.0, 90.0, 220.0))
        pr = 1.0 - np.clip(np.hypot((c.X - u) / rad, (c.Y - v) / (rad * 0.94)), 0, 1)
        pb = over(pb, (1.0 - _smoothstep(pr / 0.25)) * 0.7, (30.0, 30.0, 120.0))
        prgb = lit(pb, nx, ny, nz, ambient=0.66, kd=0.45, spec=0.65, shin=36.0)
        rgb_p = over(rgb_p, disc, prgb)
        pep = np.maximum(pep, disc)
    rgb = rgb_p

    # herbs
    herb = c.zeros()
    r = _rng("herb")
    for _ in range(9):
        u, v = float(r.uniform(-0.55, 0.55)), float(r.uniform(-0.40, 0.55))
        if abs(u) > (0.85 - (v + 0.4) * 0.55):
            continue
        c.dot(herb, u, v, float(r.uniform(0.03, 0.05)), float(r.uniform(0.018, 0.03)), float(r.uniform(0, 180)))
    rgb = over(rgb, herb * cheese * 0.9, (50.0, 128.0, 52.0))

    # crust on top
    cb = field(c, (88.0, 160.0, 220.0))
    toast = np.clip((c.grain("toast", 0.05, 1.0) - 0.4) * 0.9, 0, 1)
    cb = over(cb, toast * 0.6, (50.0, 108.0, 176.0))
    cb += (c.grain("toastf", 0.012, 5.0))[..., None]
    crgb = shade_body(c, crust, cb, radius=0.13, ambient=0.60, kd=0.65, spec=0.30, shin=16.0,
                      sheen=(-0.3, -0.85, 0.35, 0.06, -0.15, 0.25), rim=0.55, rim_w=0.05)
    # crust casts a soft shadow onto the cheese below it
    sh = cv2.warpAffine(_blur(crust, 0.03 * c.S), np.float32([[1, 0, 0], [0, 1, 0.04 * c.S]]), (c.W, c.H))
    rgb *= (1.0 - 0.40 * sh * (1 - crust))[..., None]
    rgb = over(rgb, crust, crgb)
    mask = np.clip(np.maximum(body, crust), 0, 1)
    rgb = rim_key(c, mask, rgb, width=0.04, strength=0.45)

    inner = crumb_field(c, (150.0, 206.0, 240.0), "pizza", holes=(110.0, 170.0, 214.0), amt=0.5)
    rgb, mask, d = bitten(c, rgb, mask, keep, inner, band=0.10)
    if d is not None:
        # a sauce line and a cheese line across the bitten face
        sl = np.exp(-(((d - 0.055) / 0.012) ** 2)) * mask * (1.0 - crust)
        rgb = over(rgb, sl * 0.9, (40.0, 52.0, 196.0))
        cl = np.exp(-(((d - 0.082) / 0.010) ** 2)) * mask * (1.0 - crust)
        rgb = over(rgb, cl * 0.8, (80.0, 210.0, 255.0))
    return rgb, mask


# ---------------------------------------------------------------------------
# french fries
# ---------------------------------------------------------------------------

FR_TOP_SIDE, FR_TOP_MID, FR_BOT = -0.20, 0.08, 1.12
FR_W_TOP, FR_W_BOT = 0.74, 0.50

# (base u, top u, top v, width, brightness) - back row first
FRIES = (
    (-0.40, -0.52, -0.98, 0.15, 0.82), (-0.14, -0.20, -1.12, 0.15, 0.82),
    (0.12, 0.16, -1.06, 0.15, 0.82), (0.38, 0.50, -0.96, 0.15, 0.82),
    (-0.52, -0.70, -0.66, 0.16, 1.0), (-0.30, -0.38, -0.86, 0.16, 1.0),
    (-0.05, -0.02, -0.94, 0.16, 1.0), (0.20, 0.30, -0.84, 0.16, 1.0),
    (0.46, 0.66, -0.64, 0.16, 1.0),
)


def _carton_top(x):
    return FR_TOP_MID + (FR_TOP_SIDE - FR_TOP_MID) * (x / FR_W_TOP) ** 2


def carton_mask(c: Canvas) -> np.ndarray:
    xs = np.linspace(-FR_W_TOP, FR_W_TOP, 30)
    top = [(x, _carton_top(x)) for x in xs]
    pts = top + [(FR_W_BOT, FR_BOT), (-FR_W_BOT, FR_BOT)]
    m = c.poly(pts)
    # round the corners a touch
    return np.clip((_blur(m, 0.03 * c.S) - 0.5) * 3.0 + 0.5, 0, 1)


def fries_draw(c: Canvas, keep: np.ndarray | None):
    carton = carton_mask(c)
    # back wall of the carton and its dark inside
    back = c.poly([(-FR_W_TOP + 0.04, -0.30), (FR_W_TOP - 0.04, -0.30), (FR_W_TOP - 0.02, 0.4), (-FR_W_TOP + 0.02, 0.4)])
    back = np.clip((_blur(back, 0.03 * c.S) - 0.5) * 3.0 + 0.5, 0, 1)
    rgb = shade_body(c, back, field(c, (40.0, 36.0, 150.0)), radius=0.2, ambient=0.55, kd=0.4,
                     spec=0.1, sheen=None, rim=0.5, rim_w=0.04)
    mask = back.copy()

    fry_all = c.zeros()
    for bu, tu, tv, wd, br in FRIES:
        bv = 0.55
        dx, dy = tu - bu, tv - bv
        ln = np.hypot(dx, dy)
        ax, ay = dx / ln, dy / ln
        px, py = -ay, ax
        hw = wd / 2.0
        pts = [(bu + px * hw, bv + py * hw), (tu + px * hw - ax * 0.02, tv + py * hw - ay * 0.02),
               (tu + px * hw * 0.6, tv + py * hw * 0.6), (tu - px * hw * 0.6, tv - py * hw * 0.6),
               (tu - px * hw - ax * 0.02, tv - py * hw - ay * 0.02), (bu - px * hw, bv - py * hw)]
        fm = c.poly(pts)
        # shadow cast by this fry onto what is behind it
        sh = cv2.warpAffine(_blur(fm, 0.035 * c.S), np.float32([[1, 0, 0.03 * c.S], [0, 1, 0.02 * c.S]]),
                            (c.W, c.H))
        rgb *= (1.0 - 0.45 * sh * (1.0 - fm))[..., None]
        # local coords: across (-1..1) and along (0 base .. 1 tip)
        across = ((c.X - bu) * px + (c.Y - bv) * py) / hw
        along = ((c.X - bu) * ax + (c.Y - bv) * ay) / ln
        base = field(c, (64.0, 200.0, 252.0))
        base = over(base, _smoothstep((along - 0.78) / 0.25) * 0.55, (40.0, 150.0, 226.0))
        base += (c.grain("fry" + str(bu), 0.02, 5.0))[..., None]
        # a square fry: lit left face, darker right face, a bright edge highlight
        facef = np.where(across < 0.0, 1.10, 0.80)
        edge = np.exp(-(((across + 0.35) / 0.18) ** 2)) * 0.22
        frgb = base * (facef * br)[..., None] + (edge * 255.0 * br)[..., None]
        frgb = rim_key(c, fm, frgb, width=0.025, strength=0.40, tint=(0.45, 0.42, 0.40))
        rgb = over(rgb, fm, frgb)
        mask = np.maximum(mask, fm)
        fry_all = np.maximum(fry_all, fm)

    # front of the carton: red and white stripes, wrapped round a cylinder
    hw = FR_W_TOP + (FR_W_BOT - FR_W_TOP) * np.clip((c.Y - FR_TOP_SIDE) / (FR_BOT - FR_TOP_SIDE), 0, 1)
    lon = np.arcsin(np.clip(c.X / hw, -1.0, 1.0)) / (np.pi / 2.0)
    s = np.sin(lon * np.pi * 3.0)
    stripes = _smoothstep((s + 0.05) / 0.12)
    cb = over(field(c, (238.0, 240.0, 246.0)), stripes, (44.0, 40.0, 214.0))
    sh = cv2.warpAffine(_blur(fry_all, 0.03 * c.S), np.float32([[1, 0, 0], [0, 1, 0.03 * c.S]]), (c.W, c.H))
    crgb = shade_body(c, carton, cb, radius=0.55, ambient=0.66, kd=0.50, spec=0.30, shin=18.0,
                      sheen=(-0.40, 0.30, 0.16, 0.50, 0.0, 0.18), rim=0.55, rim_w=0.06)
    # rolled lip along the top edge
    xs = np.linspace(-FR_W_TOP + 0.05, FR_W_TOP - 0.05, 30)
    lipm = c.zeros()
    c.stroke(lipm, [(x, _carton_top(x) + 0.035) for x in xs], 0.06)
    crgb = over(crgb, lipm * carton * 0.85, (248.0, 248.0, 250.0))
    # a little emblem: gold star on a white disc
    disc = c.ellipse(0.22, 0.22, 0.0, 0.62)
    crgb = over(crgb, disc, (246.0, 246.0, 250.0))
    star = []
    for k in range(10):
        a = -np.pi / 2 + k * np.pi / 5
        rr = 0.17 if k % 2 == 0 else 0.075
        star.append((np.cos(a) * rr, 0.63 + np.sin(a) * rr))
    stm = c.poly(star)
    crgb = over(crgb, stm, (50.0, 196.0, 255.0))
    crgb = rim_key(c, disc, crgb, width=0.02, strength=0.25)
    rgb = over(rgb, carton, crgb)
    mask = np.clip(np.maximum(mask, carton), 0, 1)
    rgb = rim_key(c, mask, rgb, width=0.04, strength=0.45)

    inner = crumb_field(c, (200.0, 238.0, 254.0), "fries", holes=(170.0, 220.0, 245.0), amt=0.3)
    rgb, mask, _ = bitten(c, rgb, mask, keep, inner, band=0.10)
    return rgb, mask


# ---------------------------------------------------------------------------
# chocolate-chip cookie
# ---------------------------------------------------------------------------


def cookie_mask(c: Canvas) -> np.ndarray:
    th = np.arctan2(c.Y, c.X)
    rr = 1.0 + 0.035 * np.sin(5 * th + 0.3) + 0.025 * np.sin(9 * th + 1.7) + 0.015 * np.sin(14 * th)
    f = np.sqrt((c.X / 0.94) ** 2 + (c.Y / 0.90) ** 2) / rr
    return np.clip((1.0 - f) * 0.9 * c.S + 0.5, 0, 1)


def _chip_poly(r: np.random.Generator, u: float, v: float, size: float):
    k = 7
    a0 = float(r.uniform(0, 2 * np.pi))
    return [(u + np.cos(a0 + i / k * 2 * np.pi) * size * float(r.uniform(0.7, 1.15)),
             v + np.sin(a0 + i / k * 2 * np.pi) * size * float(r.uniform(0.7, 1.15)) * 0.85) for i in range(k)]


def cookie_draw(c: Canvas, keep: np.ndarray | None):
    mask = cookie_mask(c)
    t = c.dist(mask)
    base = field(c, (118.0, 182.0, 230.0))
    base = over(base, (1.0 - _smoothstep((t - 0.04) / 0.20)) * 0.75, (62.0, 120.0, 186.0))
    blot = np.clip((c.grain("ckblot", 0.07, 1.0) - 0.3) * 0.8, 0, 1)
    base = over(base, blot * 0.35, (88.0, 150.0, 210.0))
    base += (c.grain("ckfine", 0.012, 6.0))[..., None]

    # cracks and lumpy crumb texture as a height field
    h = c.grain("cklump", 0.05, 0.012) + c.grain("cklump2", 0.018, 0.005)
    cracks = c.zeros()
    r = _rng("crack")
    for _ in range(7):
        u, v = float(r.uniform(-0.6, 0.6)), float(r.uniform(-0.6, 0.6))
        a = float(r.uniform(0, 2 * np.pi))
        pts = [(u, v)]
        for _k in range(5):
            a += float(r.normal(0, 0.6))
            u += np.cos(a) * 0.09
            v += np.sin(a) * 0.09
            pts.append((u, v))
        c.stroke(cracks, pts, float(r.uniform(0.018, 0.03)))
    cracks = _blur(np.clip(cracks, 0, 1), 0.012 * c.S)
    h = h - cracks * 0.03
    base = over(base, cracks * 0.55, (60.0, 112.0, 170.0))
    rgb = shade_bumped(c, mask, base, bump(c, h, 1.0), radius=0.30, ambient=0.62, kd=0.55,
                       spec=0.18, shin=14.0)
    rgb += (c.blob(-0.38, -0.40, 0.35, 0.22, -0.6) * mask * 24.0)[..., None]

    # chocolate chips
    rc = _rng("chips")
    spots = ((-0.45, -0.38), (0.10, -0.55), (0.52, -0.18), (-0.62, 0.18), (-0.10, -0.02),
             (0.36, 0.32), (-0.30, 0.52), (0.12, 0.62), (0.66, -0.55), (-0.05, 0.28), (0.40, -0.66))
    chips = c.zeros()
    for u, v in spots:
        size = float(rc.uniform(0.10, 0.14))
        u += float(rc.normal(0, 0.03))
        v += float(rc.normal(0, 0.03))
        cm = c.poly(_chip_poly(rc, u, v, size))
        cm = np.clip((_blur(cm, 0.02 * c.S) - 0.5) * 2.5 + 0.5, 0, 1)
        ao = _blur(cm, 0.04 * c.S)
        rgb *= (1.0 - 0.35 * ao * (1 - cm))[..., None]
        nx, ny, nz = sphere_normal(c, u, v, size * 1.6)
        chip = lit(field(c, (30.0, 48.0, 86.0)), nx, ny, nz, ambient=0.60, kd=0.65, spec=0.55, shin=30.0)
        rgb = over(rgb, cm, chip)
        chips = np.maximum(chips, cm)
    rgb = rim_key(c, mask, rgb, width=0.045, strength=0.45)

    inner = crumb_field(c, (130.0, 196.0, 238.0), "cookie", holes=(84.0, 140.0, 198.0), amt=0.7)
    # broken chips show in the bitten face too
    rgb, mask, d = bitten(c, rgb, mask, keep, inner, band=0.11)
    if d is not None:
        face = (1.0 - _smoothstep((d - 0.07) / 0.04)) * mask
        bits = np.clip((c.grain("ckbits", 0.018, 1.0) - 1.3) * 3.0, 0, 1)
        rgb = over(rgb, bits * face * 0.9, (30.0, 46.0, 82.0))
    return rgb, mask


# ---------------------------------------------------------------------------
# green grapes
# ---------------------------------------------------------------------------

GR_R = 0.25
GRAPES_BACK = ((-0.44, -0.74), (0.0, -0.80), (0.44, -0.74), (-0.22, 0.12), (0.22, 0.12))
GRAPES_FRONT = ((-0.66, -0.42), (-0.22, -0.46), (0.22, -0.46), (0.66, -0.42),
                (-0.44, -0.08), (0.0, -0.10), (0.44, -0.08),
                (-0.23, 0.27), (0.23, 0.27), (0.0, 0.61))


def grapes_draw(c: Canvas, keep: np.ndarray | None):
    rgb = c.zeros()[..., None].repeat(3, axis=2)
    mask = c.zeros()

    # stem, tendril and leaf behind the top grapes
    stem = c.zeros()
    c.stroke(stem, [(0.06, -1.24), (0.03, -1.08), (0.0, -0.92), (0.0, -0.70)], 0.07)
    for u in (-0.44, 0.44, -0.22, 0.22):
        c.stroke(stem, [(0.0, -0.90), (u * 0.6, -0.84), (u, -0.70)], 0.035)
    stem = np.clip(stem, 0, 1)
    tendril = c.zeros()
    pts = []
    for k in range(40):
        s = k / 39.0
        a = s * 3.6 * np.pi
        rr = 0.13 * (1.0 - s * 0.7)
        pts.append((-0.10 - s * 0.32 + np.cos(a) * rr * 0.6, -1.04 + np.sin(a) * rr * 0.5))
    c.stroke(tendril, pts, 0.022)
    leaf = leaf_blade(c, 0.06, -1.00, 0.66, -1.12, 0.20, curve=-0.10)
    lbase = field(c, (60.0, 150.0, 70.0))
    lrgb = shade_body(c, leaf, lbase, radius=0.15, ambient=0.66, kd=0.6, spec=0.3, shin=20.0,
                      sheen=(0.30, -1.10, 0.20, 0.05, -0.2, 0.25), rim=0.55, rim_w=0.05)
    vein = c.zeros()
    c.stroke(vein, [(0.08, -1.0), (0.36, -1.08), (0.62, -1.12)], 0.016)
    lrgb = over(lrgb, vein * 0.5, (44.0, 110.0, 50.0))
    rgb = over(rgb, leaf, lrgb)
    mask = np.maximum(mask, leaf)
    stm = np.clip(np.maximum(stem, tendril), 0, 1)
    srgb = shade_body(c, stm, field(c, (50.0, 96.0, 128.0)), radius=0.04, ambient=0.62, kd=0.6,
                      spec=0.25, sheen=None, rim=0.5, rim_w=0.02)
    rgb = over(rgb, stm, srgb)
    mask = np.maximum(mask, stm)

    flesh_in = c.zeros()
    for layer, pts, br in (("b", GRAPES_BACK, 0.72), ("f", GRAPES_FRONT, 1.0)):
        for i, (u, v) in enumerate(pts):
            ru, rv = GR_R, GR_R * 1.08
            g = c.ellipse(ru, rv, u, v)
            # contact shadow on whatever is behind
            sh = cv2.warpAffine(_blur(g, 0.04 * c.S), np.float32([[1, 0, 0.03 * c.S], [0, 1, 0.04 * c.S]]),
                                (c.W, c.H))
            rgb *= (1.0 - 0.50 * sh * (1.0 - g))[..., None]
            nx, ny, nz = sphere_normal(c, u, v, ru * 1.02, rv * 1.02)
            base = field(c, (70.0, 214.0, 158.0))
            base += (c.grain("grape" + layer + str(i), 0.03, 4.0))[..., None]
            grgb = lit(base * br, nx, ny, nz, ambient=0.55, kd=0.62, spec=0.85, shin=48.0)
            # light passing through the grape glows on its shadow side
            glow = c.blob(u + 0.08, v + 0.10, 0.13, 0.13)
            grgb += (glow * 52.0 * br)[..., None] * _col((0.55, 1.0, 0.85))
            # second, window-shaped catch light
            grgb += (c.blob(u - 0.10, v - 0.12, 0.05, 0.03, -0.6) * 120.0)[..., None]
            grgb = rim_key(c, g, grgb, width=0.035, strength=0.45, tint=(0.40, 0.45, 0.38))
            rgb = over(rgb, g, grgb)
            mask = np.maximum(mask, g)
            flesh_in = np.maximum(flesh_in * (1 - g), g * br)
    rgb = rim_key(c, mask, rgb, width=0.035, strength=0.40)

    inner = field(c, (176.0, 244.0, 216.0))
    inner += (c.grain("grapein", 0.015, 5.0))[..., None]
    rgb, mask, _ = bitten(c, rgb, mask, keep, inner, band=0.10)
    return rgb, mask


# ---------------------------------------------------------------------------
# trash bag (the "bomb")
# ---------------------------------------------------------------------------

# pear-shaped sack: broad, slumped bottom, shoulders tapering into the neck
BAG_LOBES = ((0.0, 0.42, 0.74), (-0.56, 0.62, 0.44), (0.58, 0.60, 0.46), (-0.34, -0.08, 0.42),
             (0.34, -0.06, 0.44), (0.0, -0.32, 0.38), (0.02, 0.86, 0.42), (-0.66, 0.90, 0.28),
             (0.66, 0.88, 0.30), (0.0, 0.10, 0.50))


def make_bomb(w: int = 156, h: int = 184, unit: float = 50.0) -> np.ndarray:
    c = Canvas(w, h, unit, oy=0.10)
    body, f, fx, fy = metaball(c, BAG_LOBES, sharp=2.0)
    neck = c.poly([(-0.26, -0.50), (0.26, -0.50), (0.12, -0.94), (-0.12, -0.94)])
    knot = c.ellipse(0.17, 0.13, 0.0, -0.98)
    ear_l = leaf_blade(c, -0.02, -1.00, -0.56, -1.44, 0.24, curve=0.10)
    ear_r = leaf_blade(c, 0.02, -1.00, 0.52, -1.50, 0.25, curve=-0.10)
    top = np.clip(np.maximum.reduce([neck, knot, ear_l, ear_r]), 0, 1)
    bag = np.clip(np.maximum(body, top), 0, 1)

    # folds: pleats fanning out of the neck, a few slack creases low down
    folds = c.zeros()
    r = _rng("bagfold")
    for k in range(7):
        s = (k - 3) / 3.0
        end = (s * 0.70 + float(r.normal(0, 0.05)), -0.05 + abs(s) * 0.12 + float(r.normal(0, 0.05)))
        mid = (s * 0.30, -0.40)
        c.stroke(folds, [(s * 0.06, -0.62), mid, end], float(r.uniform(0.03, 0.05)),
                 float(r.uniform(0.6, 1.0)))
    for pts in (((-0.70, 0.55), (-0.35, 0.62), (-0.05, 0.85)), ((0.25, 0.80), (0.55, 0.55), (0.80, 0.50)),
                ((-0.25, 0.25), (0.05, 0.35), (0.30, 0.20))):
        c.stroke(folds, list(pts), 0.04, 0.7)
    for pts in (((0.0, -0.94), (-0.20, -1.12), (-0.38, -1.30)), ((0.0, -0.94), (0.20, -1.14), (0.38, -1.33))):
        c.stroke(folds, list(pts), 0.025, 0.8)
    folds = _blur(np.clip(folds, 0, 1), 0.035 * c.S)
    height = -folds * 0.05

    base = field(c, (30.0, 36.0, 28.0))
    base += (c.grain("bagg", 0.04, 1.0))[..., None]
    tb = lump_tilt(f, fx, fy, 0.20)
    fb = bump(c, height, 1.0)
    tilt = (tb[0] * body + fb[0], tb[1] * body + fb[1])
    rgb = shade_bumped(c, bag, base, tilt, radius=0.35, ambient=0.70, kd=0.70, spec=1.0, shin=55.0,
                       bounce=0.55)
    # broad plastic sheen + crisp streaks on the upper-left lumps
    rgb += (c.blob(-0.48, -0.22, 0.20, 0.08, -0.9) * body * 120.0)[..., None]
    rgb += (c.blob(-0.10, 0.10, 0.22, 0.06, -0.7) * body * 70.0)[..., None]
    rgb += (c.blob(0.55, 0.30, 0.10, 0.20, 0.3) * body * 40.0)[..., None]
    rgb += (c.blob(-0.20, -0.96, 0.05, 0.04) * 90.0)[..., None]
    # cool back-light along the rim so the dark bag separates from a dark scene
    t = c.dist(bag)
    rim = (1.0 - _smoothstep(t / 0.07)) * bag
    side = np.clip(0.6 + 0.6 * (c.X * 0.5 + c.Y * 0.6), 0, 1)
    rgb += (rim * side * 70.0)[..., None] * _col((1.0, 1.05, 0.95))

    # fish skeleton poking out on the right
    fish = c.zeros()
    sx0, sy0, sx1, sy1 = 0.52, -0.18, 1.10, -0.66
    ax, ay = sx1 - sx0, sy1 - sy0
    ln = np.hypot(ax, ay)
    ax, ay = ax / ln, ay / ln
    px, py = -ay, ax
    c.stroke(fish, [(sx0, sy0), (sx1 - ax * 0.12, sy1 - ay * 0.12)], 0.04)
    for k in range(4):
        s = 0.18 + k * 0.17
        cx_, cy_ = sx0 + ax * ln * s, sy0 + ay * ln * s
        rl = 0.17 - k * 0.012
        for sg in (-1, 1):
            c.stroke(fish, [(cx_, cy_), (cx_ + px * rl * sg * 0.6 - ax * 0.04, cy_ + py * rl * sg * 0.6 - ay * 0.04),
                            (cx_ + px * rl * sg - ax * 0.10, cy_ + py * rl * sg - ay * 0.10)], 0.032)
    hx, hy = sx1 - ax * 0.05, sy1 - ay * 0.05
    head = c.poly([(hx - ax * 0.10 + px * 0.14, hy - ay * 0.10 + py * 0.14),
                   (hx + ax * 0.14 + px * 0.03, hy + ay * 0.14 + py * 0.03),
                   (hx + ax * 0.14 - px * 0.03, hy + ay * 0.14 - py * 0.03),
                   (hx - ax * 0.10 - px * 0.14, hy - ay * 0.10 - py * 0.14)])
    head = np.maximum(head, c.ellipse(0.11, 0.11, hx - ax * 0.06, hy - ay * 0.06))
    fish = np.clip(np.maximum(fish, head), 0, 1)
    eye = c.ellipse(0.035, 0.035, hx - ax * 0.02 + px * 0.04, hy - ay * 0.02 + py * 0.04)
    # it pokes out of a torn hole: hide the part inside the bag beyond the hole
    hole = c.ellipse(0.10, 0.07, sx0 + ax * 0.06, sy0 + ay * 0.06)
    inside = body * (1.0 - _smoothstep(((c.X - sx0) * ax + (c.Y - sy0) * ay) / 0.02))
    fish_vis = np.clip(fish * (1.0 - inside), 0, 1)
    rgb = over(rgb, hole * 0.95, (8.0, 10.0, 8.0))
    fbase = field(c, (196.0, 214.0, 222.0))
    frgb = shade_body(c, fish, fbase, radius=0.05, ambient=0.70, kd=0.45, spec=0.3, shin=16.0,
                      sheen=None, rim=0.55, rim_w=0.02, rim_tint=(0.40, 0.42, 0.45))
    frgb = over(frgb, eye, (40.0, 44.0, 48.0))
    rgb = over(rgb, fish_vis, frgb)
    solid = np.clip(np.maximum(bag, fish_vis), 0, 1)
    rgb = rim_key(c, solid, rgb, width=0.03, strength=0.35, tint=(0.5, 0.5, 0.5))

    # stink lines, rising and fading
    stink = c.zeros()
    for x0, y0, y1, ph in ((-0.80, -0.55, -1.55, 0.0), (-1.12, 0.05, -0.85, 1.7), (0.86, -0.95, -1.70, 2.6)):
        pts = []
        for k in range(30):
            s = k / 29.0
            y = y0 + (y1 - y0) * s
            pts.append((x0 + 0.09 * np.sin(s * 3.0 * np.pi + ph), y))
        seg = c.zeros()
        c.stroke(seg, pts, 0.08)
        along = np.clip((c.Y - y1) / (y0 - y1), 0, 1)  # 1 at the bottom, 0 at the top
        fade = np.clip(along * 1.8, 0, 1) * np.clip((1 - along) * 4.0, 0, 1)
        stink = np.maximum(stink, seg * fade)
    glow = _blur(stink, 0.05 * c.S)
    a_st = np.clip(stink * 0.95 + glow * 0.55, 0, 1)
    st_rgb = over(field(c, (60.0, 190.0, 120.0)), stink, (110.0, 236.0, 170.0))

    rgb = over(rgb, a_st * (1 - solid), st_rgb)
    alpha = np.clip(np.maximum(solid, a_st), 0, 1)
    assert_inside(c, alpha, "bomb", margin=2)
    return finish(rgb, alpha, w, h, "bomb")


# ---------------------------------------------------------------------------
# snack registry
# ---------------------------------------------------------------------------


CANDY_PINK = Candy(
    "candy_pink",
    lobes=((0.0, -0.30, 0.52), (-0.46, -0.10, 0.40), (0.48, -0.08, 0.40), (-0.30, -0.68, 0.38),
           (0.30, -0.72, 0.40), (0.02, -0.92, 0.30), (-0.66, -0.44, 0.30), (0.68, -0.46, 0.30),
           (-0.22, 0.16, 0.34), (0.24, 0.16, 0.34)),
    light=(222.0, 176.0, 255.0), deep=(178.0, 96.0, 246.0), stripe=(178.0, 96.0, 240.0),
)
CANDY_BLUE = Candy(
    "candy_blue",
    lobes=((0.0, -0.36, 0.54), (-0.42, -0.06, 0.38), (0.42, -0.02, 0.38), (-0.40, -0.62, 0.40),
           (0.40, -0.64, 0.40), (0.0, -0.96, 0.34), (-0.70, -0.30, 0.28), (0.72, -0.34, 0.28),
           (0.0, 0.14, 0.36)),
    light=(255.0, 228.0, 170.0), deep=(246.0, 168.0, 82.0), stripe=(236.0, 150.0, 60.0),
)
CANDY_SWIRL = Candy(
    "candy_swirl",
    lobes=((0.0, -0.32, 0.50), (-0.50, -0.14, 0.38), (0.46, -0.06, 0.42), (-0.26, -0.74, 0.36),
           (0.34, -0.66, 0.40), (-0.06, -0.96, 0.30), (-0.70, -0.50, 0.30), (0.70, -0.40, 0.28),
           (-0.18, 0.18, 0.34), (0.28, 0.14, 0.32)),
    light=(255.0, 196.0, 216.0), deep=(232.0, 122.0, 168.0), stripe=(214.0, 110.0, 160.0),
    swirl=(214.0, 160.0, 255.0),
)


@dataclass(frozen=True)
class Snack:
    name: str
    w: int
    h: int
    unit: float
    draw: Callable[[Canvas, np.ndarray | None], tuple[np.ndarray, np.ndarray]]
    bite: Bite
    radius: int
    juice: tuple[int, int, int]  # BGR crumb colour
    score: int = 1
    oy: float = 0.0
    ox: float = 0.0


def _candy(cd: Candy):
    return lambda c, keep: candy_draw(cd, c, keep)


SNACKS: tuple[Snack, ...] = (
    Snack("cotton_candy_pink", 140, 170, 50.0, _candy(CANDY_PINK),
          Bite(0.62, -0.78, 0.50, torn=True), radius=50, juice=(214, 150, 255), oy=0.10),
    Snack("cotton_candy_blue", 140, 172, 50.0, _candy(CANDY_BLUE),
          Bite(-0.60, -0.80, 0.50, torn=True), radius=50, juice=(255, 214, 140), oy=0.10),
    Snack("cotton_candy_swirl", 140, 170, 50.0, _candy(CANDY_SWIRL),
          Bite(0.64, -0.70, 0.50, torn=True), radius=50, juice=(248, 170, 210), oy=0.10),
    Snack("donut", 134, 126, 56.0, donut_draw,
          Bite(1.02, -0.20, 0.50, 0.13, phase=0.2), radius=54, juice=(196, 140, 255)),
    Snack("pizza", 140, 140, 56.0, pizza_draw,
          Bite(0.0, 1.00, 0.54, 0.12, phase=0.1), radius=50, juice=(70, 196, 255)),
    Snack("fries", 120, 150, 50.0, fries_draw,
          Bite(-0.30, -1.10, 0.52, 0.11, phase=0.3), radius=44, juice=(80, 210, 255)),
    Snack("cookie", 116, 112, 48.0, cookie_draw,
          Bite(0.84, -0.54, 0.48, 0.12, phase=0.15), radius=44, juice=(100, 168, 222), score=2),
    Snack("grapes", 122, 140, 50.0, grapes_draw,
          Bite(0.10, 0.50, 0.42, 0.11, phase=0.4), radius=46, juice=(120, 230, 170), oy=0.25),
)


def crop_to_content(img: np.ndarray, margin: int = 3) -> np.ndarray:
    ys, xs = np.nonzero(img[..., 3] > 2)
    if ys.size == 0:
        raise RuntimeError("empty sprite")
    h, w = img.shape[:2]
    y0, y1 = max(0, int(ys.min()) - margin), min(h, int(ys.max()) + 1 + margin)
    x0, x1 = max(0, int(xs.min()) - margin), min(w, int(xs.max()) + 1 + margin)
    return img[y0:y1, x0:x1].copy()


def render_snack(s: Snack) -> dict[str, np.ndarray]:
    c = Canvas(s.w, s.h, s.unit, s.oy, s.ox)
    rgb, alpha = s.draw(c, None)
    assert_inside(c, alpha, s.name)
    out = {"whole": finish(rgb, alpha, s.w, s.h, s.name + "w")}

    bite = bite_region(c, s.bite, s.name)
    area = float(alpha.sum())
    for key, keep in (("half_a", 1.0 - bite), ("half_b", bite)):
        prgb, palpha = s.draw(c, keep)
        frac = float(palpha.sum()) / area
        if not 0.06 < frac < 0.94:
            raise RuntimeError(f"{s.name} {key}: piece is {frac:.0%} of the snack; move the bite")
        out[key] = crop_to_content(finish(prgb, palpha, s.w, s.h, s.name + key))
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Snack Attack snack and trash-bag sprites.")
    ap.add_argument("--out", type=Path, default=OUT_DIR, help="output directory")
    ap.add_argument("--only", default=None, help="render just one snack (debug)")
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
    for s in SNACKS:
        if args.only and s.name != args.only:
            continue
        imgs = render_snack(s)
        write(f"fruit_{s.name}_whole.png", imgs["whole"])
        write(f"fruit_{s.name}_half_a.png", imgs["half_a"])
        write(f"fruit_{s.name}_half_b.png", imgs["half_b"])
        manifest["fruits"].append({
            "name": s.name,
            "radius": int(s.radius),
            "juice": [int(v) for v in s.juice],
            "score": int(s.score),
        })

    if not args.only or args.only == "bomb":
        write("bomb.png", make_bomb())

    if not args.only:
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        # Retire sprites of snacks no longer in the manifest (old fruit theme etc.).
        # Only fruit_*.png is ours to prune: other generators share this folder.
        names = {f"fruit_{s.name}_{k}.png" for s in SNACKS for k in ("whole", "half_a", "half_b")}
        for stale in sorted(out_dir.glob("fruit_*.png")):
            if stale.name not in names:
                stale.unlink()
                print(f"removed stale {stale.name}")

    if not rows:
        return
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
