"""Drawing Raccoon Golf: a pinhole 3D view of the hole, Lenny, and the HUD.

The camera is a pinhole (focal FOCAL px at 1280x720, ~65 degrees across)
behind the ball. The ground is flat, so each surface of the hole (island,
fairway, mowing stripes, green, bunkers, water) is a polygon in metres that is
transformed, clipped at the near plane (a polygon that reaches behind the
camera would otherwise project inside out) and filled. Trees are billboards
sorted far to near. The sky is a panorama sliced by the camera's yaw; with a
fixed pitch per camera mode the horizon is a fixed row, so sky and sea are
slice copies.

The live camera image is not the backdrop: like Flappy Raccoon the game is a
full scene and the camera goes picture-in-picture (top right), where it shows
the player their tracked grip.

Budget (< 4 ms a frame on average): while the camera is still (aiming,
results, the scorecard) the whole 3D view is cached and each frame is one copy
plus the moving parts. While it moves (the hole intro, the ball in flight) the
world is drawn at half resolution and scaled up; the ball, Lenny and the HUD
are drawn at full resolution on top. Measured on an M5 Max: ~2.0 ms a frame
aiming (cached), ~1.9 ms in flight (2.9 ms at full resolution), 3.1 ms on the
green with the animated slope grid; p95 under 3.3 ms everywhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from ... import theme
from ...config import KP
from ...gfx import alpha_composite
from ...overlay import outlined_text
from . import physics as P
from .course import YARDS_PER_M, Hole
from .game import CardRow, GolfGame, to_par_text

NEAR = 0.4  # m: the near clipping plane
SHIFT = 2  # sub-pixel bits for polygon fills
ONE = 1 << SHIFT
STRIPE = 8.0  # m: fairway mowing stripes
GREEN_STRIPE = 2.5
STRIPE_FAR = 170.0  # m: stripes beyond this would only shimmer
PIN_HEIGHT = 2.2  # m (drawn no shorter than PIN_MIN_PX)
PIN_MIN_PX = 34
LENNY_WORLD = 0.0093  # Lenny's sprite scale per (focal / depth): 0.95 at the address view
LENNY_MAX = 1.15
THETA_TOP = 2.75  # radians of arm swing at a full backswing (swing angle 1)
THETA_FINISH = 2.6  # radians of follow-through (swing angle -1)
ARC_TOP = 118  # px: the dotted arc stops short of the top HUD row
LOW_RES = 2  # moving-camera frames draw the world at 1/LOW_RES and scale it up

PIP = (944, 16, 1264, 196)  # x0, y0, x1, y1: the live camera window (a quarter frame)
MAP = (1146, 208, 1264, 630)  # the course overview
GAUGE_AT = (22, 318)  # top-left of the power gauge (drawn at GAUGE_SCALE)
GAUGE_SCALE = 0.85
HOLE_PANEL = (16, 14, 330, 134)
CLUB_PANEL = (16, 142, 330, 212)
WIND_PANEL = (566, 12, 714, 106)

_GOLD = theme.ACCENT
_CYAN = theme.CYAN
_WHITE = theme.WHITE
_DIM = theme.DIM
_WARN = theme.WARN
_FONT = theme.TITLE_FONT
_BODY = theme.BODY_FONT


# --- geometry ---------------------------------------------------------------


class View:
    """A pinhole camera: world (x, y, z) metres -> screen pixels."""

    def __init__(self, cam, w: int, h: int, f: float, ox: float = 0.0, oy: float = 0.0) -> None:
        self.w, self.h, self.f = w, h, f
        self.cx, self.cy = w / 2 + ox, h / 2 + oy
        sy, cy = math.sin(cam.yaw), math.cos(cam.yaw)
        sp, cp = math.sin(cam.pitch), math.cos(cam.pitch)
        self.R = np.array([[cy, -sy, 0.0], [sy * sp, cy * sp, cp], [sy * cp, cy * cp, -sp]])
        self.C = np.array([cam.x, cam.y, cam.z])
        self.horizon = self.cy - f * math.tan(cam.pitch)
        self.yaw = cam.yaw

    def to_cam(self, pts: np.ndarray) -> np.ndarray:
        """(N, 3) world -> (N, 3) camera: x right, y up, z forward."""
        return (pts - self.C) @ self.R.T

    def project(self, q: np.ndarray) -> np.ndarray:
        z = q[:, 2]
        return np.stack([self.cx + self.f * q[:, 0] / z, self.cy - self.f * q[:, 1] / z], 1)

    def point(self, x: float, y: float, z: float = 0.0):
        """(sx, sy, depth) or None behind the near plane."""
        q = self.to_cam(np.array([[x, y, z]]))[0]
        if q[2] < NEAR:
            return None
        return (self.cx + self.f * q[0] / q[2], self.cy - self.f * q[1] / q[2], float(q[2]))


def _clip_near(q: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman against z >= NEAR for one polygon in camera space."""
    out = []
    n = len(q)
    for i in range(n):
        a, b = q[i], q[(i + 1) % n]
        ina, inb = a[2] >= NEAR, b[2] >= NEAR
        if ina:
            out.append(a)
        if ina != inb:
            t = (NEAR - a[2]) / (b[2] - a[2])
            out.append(a + (b - a) * t)
    return np.array(out) if out else np.zeros((0, 3))


def offset_poly(pts, d: float) -> list:
    """Grow (d > 0) or shrink a simple polygon by d metres (mitred, clamped)."""
    p = np.asarray(pts, np.float64)
    n = len(p)
    area = 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))
    sign = 1.0 if area < 0 else -1.0  # outward normal side for this winding
    out = []
    for i in range(n):
        a, b, c = p[i - 1], p[i], p[(i + 1) % n]
        e1, e2 = b - a, c - b
        n1 = np.array([e1[1], -e1[0]]) / (np.hypot(*e1) or 1.0)
        n2 = np.array([e2[1], -e2[0]]) / (np.hypot(*e2) or 1.0)
        m = n1 + n2
        ml = np.hypot(*m)
        if ml < 1e-9:
            m = n1
        else:
            m = m / ml
        k = min(3.0, 1.0 / max(0.2, float(np.dot(m, n1))))
        out.append(list(b + m * d * k * -sign))
    return out


def clip_halfplane(poly: list, nx: float, ny: float, c: float) -> list:
    """Keep the part of `poly` where nx x + ny y >= c."""
    out = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        da, db = nx * a[0] + ny * a[1] - c, nx * b[0] + ny * b[1] - c
        if da >= 0:
            out.append(a)
        if (da >= 0) != (db >= 0):
            t = da / (da - db)
            out.append([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t])
    return out


def stripes(poly: list, axis: tuple[float, float], width: float, origin: tuple[float, float]) -> list:
    """Every other `width`-wide band of `poly` across `axis` (mowing stripes)."""
    ax, ay = axis
    s = [(p[0] - origin[0]) * ax + (p[1] - origin[1]) * ay for p in poly]
    k0 = int(math.floor(min(s) / width))
    k1 = int(math.ceil(max(s) / width))
    out = []
    for k in range(k0, k1 + 1):
        if k % 2:
            continue
        lo = origin[0] * ax + origin[1] * ay + k * width
        band = clip_halfplane(poly, ax, ay, lo)
        band = clip_halfplane(band, -ax, -ay, -(lo + width))
        if len(band) >= 3:
            out.append(band)
    return out


@dataclass
class Layer:
    """Polygons of one colour, with their vertices stacked for one transform."""

    colour: tuple
    polys: list
    far_cut: float = math.inf  # skip polygons whose nearest point is further than this (m)
    edge: tuple | None = None  # an anti-aliased outline colour

    def __post_init__(self) -> None:
        self.verts = np.array([[p[0], p[1], 0.0] for poly in self.polys for p in poly], np.float64) \
            if self.polys else np.zeros((0, 3))
        self.splits = np.cumsum([len(p) for p in self.polys])[:-1] if self.polys else []


def _fixed(screen: np.ndarray) -> np.ndarray:
    return np.round(np.clip(screen, -60000, 60000) * ONE).astype(np.int32).reshape(-1, 1, 2)


def draw_layer(img, view: View, layer: Layer) -> None:
    if not layer.polys:
        return
    q = view.to_cam(layer.verts)
    polys = []
    for part in np.split(q, layer.splits):
        z = part[:, 2]
        if z.max() < NEAR or z.min() > layer.far_cut:
            continue
        if z.min() < NEAR:
            part = _clip_near(part)
            if len(part) < 3:
                continue
        s = view.project(part)
        if s[:, 0].max() < 0 or s[:, 0].min() > view.w or s[:, 1].max() < view.horizon - 2 or s[:, 1].min() > view.h:
            continue
        polys.append(_fixed(s))
    if polys:
        cv2.fillPoly(img, polys, layer.colour, cv2.LINE_AA, SHIFT)
        if layer.edge is not None:
            cv2.polylines(img, polys, True, layer.edge, 1, cv2.LINE_AA, SHIFT)


# --- per-hole scenery -------------------------------------------------------


class HoleScene:
    """Everything about a hole that is built once: ground layers, map, grid."""

    def __init__(self, hole: Hole, g: dict) -> None:
        self.hole = hole
        col = lambda k: tuple(int(v) for v in g[k])  # noqa: E731
        tx, ty = hole.tee
        px, py = hole.pin
        L = math.hypot(px - tx, py - ty) or 1.0
        self.axis = ((px - tx) / L, (py - ty) / L)
        fair = [a.pts for a in hole.fairway]
        green = hole.green.pts
        self.ground = [
            Layer(col("beach"), [offset_poly(hole.land.pts, 5.0)]),
            Layer(col("rough"), [hole.land.pts]),
            Layer(col("fairway"), fair, edge=col("edge")),
            Layer(col("fairway_stripe"), [s for f in fair for s in stripes(f, self.axis, STRIPE, hole.tee)],
                  far_cut=STRIPE_FAR),
            Layer(col("fringe"), [offset_poly(green, 1.6)]),
            Layer(col("green"), [green]),
            Layer(col("green_stripe"), stripes(green, self.axis, GREEN_STRIPE, hole.pin), far_cut=60.0),
            Layer(col("tee"), [hole.tee_box.pts], edge=col("edge")),
            Layer(col("bunker_rim"), [offset_poly(b.pts, 0.7) for b in hole.bunkers]),
            Layer(col("bunker"), [b.pts for b in hole.bunkers]),
            Layer(col("water_rim"), [offset_poly(w.pts, 1.0) for w in hole.water]),
            Layer(col("water"), [w.pts for w in hole.water]),
            Layer(col("water_hi"), [offset_poly(w.pts, -1.4) for w in hole.water]),
            Layer(col("water"), [offset_poly(w.pts, -2.2) for w in hole.water]),
        ]
        self.shadow = tuple(int(v * 0.72) for v in col("rough"))
        self.grid_lines, self.flow_pts = self._green_grid(hole)

    @staticmethod
    def _green_grid(hole: Hole):
        """World-space grid lines over the green, and the points for slope arrows."""
        a = hole.green
        lines = []
        step, sample = 1.5, 0.3
        for xs in np.arange(math.floor(a.x0), a.x1 + step, step):
            run = []
            for ys in np.arange(a.y0, a.y1 + sample, sample):
                if a.contains(float(xs), float(ys)):
                    run.append((float(xs), float(ys), 0.0))
                elif len(run) > 1:
                    lines.append(np.array(run))
                    run = []
                else:
                    run = []
            if len(run) > 1:
                lines.append(np.array(run))
        for ys in np.arange(math.floor(a.y0), a.y1 + step, step):
            run = []
            for xs in np.arange(a.x0, a.x1 + sample, sample):
                if a.contains(float(xs), float(ys)):
                    run.append((float(xs), float(ys), 0.0))
                elif len(run) > 1:
                    lines.append(np.array(run))
                    run = []
                else:
                    run = []
            if len(run) > 1:
                lines.append(np.array(run))
        pts = []
        for xs in np.arange(a.x0 + 0.75, a.x1, step):
            for ys in np.arange(a.y0 + 0.75, a.y1, step):
                if a.contains(float(xs), float(ys)):
                    gx, gy = hole.green_gradient(float(xs), float(ys))
                    pts.append((float(xs), float(ys), gx, gy))
        return lines, np.array(pts) if pts else np.zeros((0, 4))


def build_map(hole: Hole, g: dict, w: int, h: int):
    """A top-down map of the hole, tee at the bottom, pin at the top.

    Returns (BGR image, world -> map pixel function)."""
    col = lambda k: tuple(int(v) for v in g[k])  # noqa: E731
    tx, ty = hole.tee
    px, py = hole.pin
    ang = math.atan2(px - tx, py - ty)  # rotate this heading to straight up
    ca, sa = math.cos(ang), math.sin(ang)

    def rot(x, y):
        return (x - tx) * ca - (y - ty) * sa, (x - tx) * sa + (y - ty) * ca

    pts = [rot(*p) for p in hole.land.pts] + [rot(*hole.pin), rot(*hole.tee)]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    pad = 8
    s = min((w - 2 * pad) / (max(xs) - min(xs)), (h - 2 * pad) / (max(ys) - min(ys)))
    mx = (max(xs) + min(xs)) / 2
    my = (max(ys) + min(ys)) / 2

    def to_map(x, y):
        rx, ry = rot(x, y)
        return (w / 2 + (rx - mx) * s, h / 2 - (ry - my) * s)

    img = np.empty((h, w, 3), np.uint8)
    img[:] = col("sea")

    def fill(poly, colour):
        arr = np.array([to_map(*p) for p in poly])
        cv2.fillPoly(img, [_fixed(arr)], colour, cv2.LINE_AA, SHIFT)

    fill(offset_poly(hole.land.pts, 5.0), col("beach"))
    fill(hole.land.pts, col("rough"))
    for a in hole.fairway:
        fill(a.pts, col("fairway"))
    fill(offset_poly(hole.green.pts, 1.6), col("fringe"))
    fill(hole.green.pts, col("green"))
    fill(hole.tee_box.pts, col("tee"))
    for b in hole.bunkers:
        fill(b.pts, col("bunker"))
    for wt in hole.water:
        fill(wt.pts, col("water"))
    for t in hole.trees:
        x, y = to_map(t.x, t.y)
        r = max(2.0, t.radius * s)
        cv2.circle(img, (int(x * ONE), int(y * ONE)), int(r * ONE), (40, 100, 40), -1, cv2.LINE_AA, SHIFT)
    return img, to_map


# --- sprites ----------------------------------------------------------------


class SpriteCache:
    """A BGRA sprite resized to integer heights, built on first use."""

    def __init__(self, img: np.ndarray, limit: int = 160) -> None:
        self.img = img
        self.limit = limit
        self._cache: dict[int, np.ndarray] = {}

    def get(self, height: int) -> np.ndarray:
        height = max(2, int(height))
        spr = self._cache.get(height)
        if spr is None:
            h, w = self.img.shape[:2]
            k = height / h
            interp = cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR
            spr = cv2.resize(self.img, (max(1, round(w * k)), height), interpolation=interp)
            if len(self._cache) >= self.limit:
                self._cache.pop(next(iter(self._cache)))
            self._cache[height] = spr
        return spr


def _ease_out(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return 1.0 - (1.0 - t) ** 3


def _pop(age: float) -> float:
    """Overshoot then settle: 0.6 -> 1.25 -> 1.0 over the first 0.3 s."""
    if age < 0.12:
        return 0.6 + age / 0.12 * 0.65
    if age < 0.3:
        return 1.25 - (age - 0.12) / 0.18 * 0.25
    return 1.0


def _faded_text(img, text, cx, y, scale, colour, alpha, thick=2, font=_FONT):
    """Centred outlined text, faded by blending a patch."""
    if alpha <= 0.01:
        return
    if alpha >= 0.99:
        theme.centred_text(img, text, cx, int(y), scale, colour, thick, font)
        return
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    pad = 6
    x0, y0 = int(cx - tw / 2) - pad, int(y) - th - pad
    x1, y1 = int(cx + tw / 2) + pad, int(y) + base + pad
    H, W = img.shape[:2]
    x0c, y0c, x1c, y1c = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1c <= x0c or y1c <= y0c:
        return
    roi = img[y0c:y1c, x0c:x1c]
    patch = roi.copy()
    theme.centred_text(patch, text, cx - x0c, int(y) - y0c, scale, colour, thick, font)
    cv2.addWeighted(patch, alpha, roi, 1 - alpha, 0, roi)


def _capsule(img, a, b, r, colour):
    cv2.line(img, (int(a[0] * ONE), int(a[1] * ONE)), (int(b[0] * ONE), int(b[1] * ONE)), colour,
             max(1, int(round(2 * r))), cv2.LINE_AA, SHIFT)


def _triangle(img, cx, cy, size, direction, colour, outline=(20, 16, 24)):
    """A filled arrow head pointing left/right/up/down."""
    d = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}[direction]
    nx, ny = -d[1], d[0]
    tip = (cx + d[0] * size, cy + d[1] * size)
    b1 = (cx - d[0] * size * 0.6 + nx * size * 0.8, cy - d[1] * size * 0.6 + ny * size * 0.8)
    b2 = (cx - d[0] * size * 0.6 - nx * size * 0.8, cy - d[1] * size * 0.6 - ny * size * 0.8)
    pts = _fixed(np.array([tip, b1, b2]))
    cv2.polylines(img, [pts], True, outline, 5, cv2.LINE_AA, SHIFT)
    cv2.fillPoly(img, [pts], colour, cv2.LINE_AA, SHIFT)


def _border(img, rect, r: int, colour, thickness: int) -> None:
    """A rounded rectangle's outline only (theme.panel would also blend its inside)."""
    x0, y0, x1, y1 = (int(v) for v in rect)
    for a, b in (((x0 + r, y0), (x1 - r, y0)), ((x0 + r, y1), (x1 - r, y1)),
                 ((x0, y0 + r), (x0, y1 - r)), ((x1, y0 + r), (x1, y1 - r))):
        cv2.line(img, a, b, colour, thickness, cv2.LINE_AA)
    for (cx, cy), ang in (((x0 + r, y0 + r), 180), ((x1 - r, y0 + r), 270),
                          ((x1 - r, y1 - r), 0), ((x0 + r, y1 - r), 90)):
        cv2.ellipse(img, (cx, cy), (r, r), 0, ang, ang + 90, colour, thickness, cv2.LINE_AA)


def _rounded_mask(w: int, h: int, r: int) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    cv2.rectangle(m, (r, 0), (w - 1 - r, h - 1), 255, -1)
    cv2.rectangle(m, (0, r), (w - 1, h - 1 - r), 255, -1)
    for cx, cy in ((r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)):
        cv2.circle(m, (cx, cy), r, 255, -1, cv2.LINE_AA)
    return m > 127


# --- the renderer -----------------------------------------------------------


class GolfRenderer:
    def __init__(self, size: tuple[int, int], art) -> None:
        self.width, self.height = size
        self.art = art
        m = art.manifest
        self.m = m
        self.k = self.width / 1280.0
        self.f = float(m["focal"]) * self.k
        self.g = m["ground"]
        # Sky and sea, padded so a slice never wraps.
        W = self.width
        sky = art.sky
        if abs(self.k - 1.0) > 1e-6:
            sky = cv2.resize(sky, (int(sky.shape[1] * self.k), int(sky.shape[0] * self.k)))
        self.pano_w = sky.shape[1]
        self.sky = np.ascontiguousarray(np.hstack([sky, sky[:, :W + 2]]))
        self.sky_top = tuple(int(v) for v in sky[0, 0])
        sea = art.sea
        if sea.shape[0] < self.height + 40:  # tall enough for a horizon at the very top
            sea = np.vstack([sea, np.repeat(sea[-1:], self.height + 40 - sea.shape[0], axis=0)])
        self.sea = np.ascontiguousarray(np.hstack([sea] * (2 + W // sea.shape[1])))
        self.sea_w = sea.shape[1]
        self.haze = art.haze if art.haze.shape[1] == W else cv2.resize(art.haze, (W, art.haze.shape[0]))
        self.trees = {k: SpriteCache(v) for k, v in art.trees.items()}
        self.tree_spec = m["trees"]
        self.ball = SpriteCache(art.ball, limit=60)
        self.lenny_spec = m["lenny"]
        self.lenny = SpriteCache(art.lenny, limit=40)
        gs = GAUGE_SCALE
        gh, gw = art.gauge.shape[:2]
        size_g = (max(1, round(gw * gs)), max(1, round(gh * gs)))
        self.gauge = cv2.resize(art.gauge, size_g, interpolation=cv2.INTER_AREA)
        self.gauge_fill = cv2.resize(art.gauge_fill, size_g, interpolation=cv2.INTER_AREA)
        gi = m["gauge"]["inner"]
        self.gauge_inner = [v * gs for v in gi]
        self.gauge_full = float(m["gauge"]["full_frac"])

        self.scenes: dict[int, HoleScene] = {}
        self.maps: dict[int, tuple] = {}
        mw, mh = MAP[2] - MAP[0], MAP[3] - MAP[1]
        self._map_mask = np.repeat(_rounded_mask(mw, mh, 12)[:, :, None], 3, axis=2)
        pw, ph = PIP[2] - PIP[0], PIP[3] - PIP[1]
        self._pip_mask = np.repeat(_rounded_mask(pw, ph, 14)[:, :, None], 3, axis=2)
        self._cache_key = None
        self._cache_img: np.ndarray | None = None
        self._low: np.ndarray | None = None
        self._rng = np.random.default_rng()
        self.timings: dict[str, float] = {}

    # --- scene pieces ---

    def scene(self, game: GolfGame) -> HoleScene:
        sc = self.scenes.get(game.hole_i)
        if sc is None:
            sc = self.scenes[game.hole_i] = HoleScene(game.hole, self.g)
        return sc

    def hole_map(self, game: GolfGame):
        mp = self.maps.get(game.hole_i)
        if mp is None:
            mp = self.maps[game.hole_i] = build_map(game.hole, self.g, MAP[2] - MAP[0], MAP[3] - MAP[1])
        return mp

    def draw_backdrop(self, img, view: View) -> None:
        h, w = img.shape[:2]
        hy = int(round(view.horizon))
        k = w / self.width
        if hy > 0:
            rows = min(hy, h)
            sky = self.sky if k == 1.0 else self._small_sky(k)
            pw = sky.shape[1] - w - 2
            x0 = int(round((view.yaw / (2 * math.pi)) * pw)) % pw
            top = sky.shape[0] - rows
            if top >= 0:
                img[:rows] = sky[top:top + rows, x0:x0 + w]
            else:
                img[:-top] = self.sky_top
                img[-top:rows] = sky[:, x0:x0 + w]
        if hy < h:
            r0 = max(0, hy)
            sea = self.sea if k == 1.0 else self._small_sea(k)
            sw = self.sea_w if k == 1.0 else int(self.sea_w * k)
            x0 = int(round((view.yaw / (2 * math.pi)) * self.pano_w * k)) % sw
            img[r0:] = sea[r0 - hy:h - hy, x0:x0 + w]

    def _small_sky(self, k):
        if getattr(self, "_sky_k", None) != k:
            sky = self.sky[:, :self.pano_w]
            sm = cv2.resize(sky, (int(self.pano_w * k), int(sky.shape[0] * k)), interpolation=cv2.INTER_AREA)
            w = int(self.width * k)
            self._sky_small = np.ascontiguousarray(np.hstack([sm, sm[:, :w + 2]]))
            self._sky_k = k
        return self._sky_small

    def _small_sea(self, k):
        if getattr(self, "_sea_k", None) != k:
            sea = self.sea[:, :self.sea_w]
            sm = cv2.resize(sea, (int(self.sea_w * k), int(sea.shape[0] * k)), interpolation=cv2.INTER_AREA)
            w = int(self.width * k)
            self._sea_small = np.ascontiguousarray(np.hstack([sm] * (2 + w // sm.shape[1])))
            self._sea_k = k
        return self._sea_small

    def draw_ground(self, img, view: View, sc: HoleScene) -> None:
        for layer in sc.ground:
            draw_layer(img, view, layer)
        # Distance haze just under the horizon.
        hy = int(round(view.horizon))
        if 0 <= hy < img.shape[0]:
            haze = self.haze
            if img.shape[1] != haze.shape[1]:
                haze = cv2.resize(haze, (img.shape[1], max(1, int(haze.shape[0] * img.shape[1] / haze.shape[1]))))
            alpha_composite(img, haze, 0, hy)

    def draw_trees(self, img, view: View, game: GolfGame, sc: HoleScene) -> None:
        hole = game.hole
        if not hole.trees:
            return
        pts = np.array([[t.x, t.y, 0.0] for t in hole.trees])
        q = view.to_cam(pts)
        order = np.argsort(-q[:, 2])
        for i in order:
            z = q[i, 2]
            if z < 2.0:
                continue
            t = hole.trees[i]
            sx = view.cx + view.f * q[i, 0] / z
            sy = view.cy - view.f * q[i, 1] / z
            tall = view.f * t.height / z  # screen px from foot to canopy top
            if tall < 4 or sx < -tall or sx > img.shape[1] + tall or sy < view.horizon - 2:
                continue
            spec = self.tree_spec.get(t.kind, self.tree_spec["palm"])
            src = self.art.trees.get(t.kind) if t.kind in self.art.trees else self.art.trees["palm"]
            native_tall = max(1.0, spec["foot"][1] - spec["top"])
            scale = tall / native_tall
            spr = self.trees.get(t.kind, self.trees["palm"]).get(int(round(src.shape[0] * scale)))
            sk = spr.shape[0] / src.shape[0]
            # A soft shadow on the grass first, then the tree on its foot.
            rx, ry = max(2, int(view.f * t.radius * 0.9 / z)), max(1, int(view.f * t.radius * 0.9 / z * 0.28))
            if ry >= 1 and z > 0:
                cv2.ellipse(img, (int(sx * ONE), int(sy * ONE)), (rx * ONE, ry * ONE), 0, 0, 360, sc.shadow, -1,
                            cv2.LINE_AA, SHIFT)
            alpha_composite(img, spr, int(round(sx - spec["foot"][0] * sk)), int(round(sy - spec["foot"][1] * sk)))

    def draw_pin(self, img, view: View, game: GolfGame, t: float) -> None:
        px, py = game.hole.pin
        base = view.point(px, py, 0.0)
        if base is None:
            return
        sx, sy, z = base
        tall = max(PIN_MIN_PX * img.shape[1] / self.width, view.f * PIN_HEIGHT / z)
        k = tall / (view.f * PIN_HEIGHT / z)
        if z < 30:  # the cup, when near enough to see
            r = view.f * P.CUP_RADIUS / z
            cv2.ellipse(img, (int(sx * ONE), int(sy * ONE)), (max(1, int(r * ONE)), max(1, int(r * 0.35 * ONE))),
                        0, 0, 360, (30, 50, 30), -1, cv2.LINE_AA, SHIFT)
        top = (sx, sy - tall)
        w = max(1.0, tall * 0.045)
        _capsule(img, (sx, sy), top, w + 1, (40, 40, 50))
        _capsule(img, (sx, sy), top, w, (250, 250, 250))
        # The flag streams with the wind, rippling.
        wind_dir = game.wind_dir - view.yaw
        side = math.sin(wind_dir) if game.wind_mph > 0.5 else 1.0
        side = 1.0 if side >= 0 else -1.0
        fw, fh = tall * 0.55 * k / k, tall * 0.32
        pts = []
        n = 8
        for i in range(n + 1):
            u = i / n
            wave = math.sin(t * 7.0 + u * 5.0) * fh * 0.12 * u
            pts.append((sx + side * fw * u, top[1] + fh * 0.5 * u + wave))
        for i in range(n, -1, -1):
            u = i / n
            wave = math.sin(t * 7.0 + u * 5.0) * fh * 0.12 * u
            pts.append((sx + side * fw * u, top[1] + fh - fh * 0.5 * u + wave))
        arr = _fixed(np.array(pts))
        cv2.fillPoly(img, [arr], (60, 70, 235), cv2.LINE_AA, SHIFT)
        cv2.polylines(img, [arr], True, (30, 30, 120), 1, cv2.LINE_AA, SHIFT)

    def world(self, img, view: View, game: GolfGame, t: float) -> None:
        sc = self.scene(game)
        self.draw_backdrop(img, view)
        self.draw_ground(img, view, sc)
        self.draw_trees(img, view, game, sc)

    # --- frame ---

    def draw(self, frame: np.ndarray, game: GolfGame) -> None:
        W, H = self.width, self.height
        pip = cv2.resize(frame, (PIP[2] - PIP[0], PIP[3] - PIP[1]), interpolation=cv2.INTER_AREA)
        shake = game.fx.shake
        ox = oy = 0.0
        if shake > 0:
            ox, oy = (self._rng.uniform(-1, 1, 2) * 14.0 * shake).tolist()
        cam = game.camera
        view = View(cam, W, H, self.f, ox, oy)
        moving = game.phase.value in ("intro", "flight") or shake > 0 or (
            game.phase.value == "holed" and game.phase_t < 1.3)
        key = (game.hole_i, round(cam.x, 3), round(cam.y, 3), round(cam.z, 3), round(cam.yaw, 5),
               round(cam.pitch, 5))
        if not moving and key == self._cache_key and self._cache_img is not None:
            np.copyto(frame, self._cache_img)
        elif moving:
            lw, lh = W // LOW_RES, H // LOW_RES
            if self._low is None or self._low.shape[:2] != (lh, lw):
                self._low = np.empty((lh, lw, 3), np.uint8)
            low_view = View(cam, lw, lh, self.f / LOW_RES, ox / LOW_RES, oy / LOW_RES)
            self.world(self._low, low_view, game, game.clock)
            cv2.resize(self._low, (W, H), dst=frame, interpolation=cv2.INTER_LINEAR)
            self._cache_key = None
        else:
            self.world(frame, view, game, game.clock)
            self._cache_key = key
            self._cache_img = frame.copy()

        sc = self.scene(game)
        self.draw_pin(frame, view, game, game.clock)
        phase = game.phase.value
        if phase == "aim":
            if game.putting or game.lie == "green":
                self._draw_green_grid(frame, view, sc, game.clock)
            self._draw_target(frame, view, game)
        if phase in ("aim", "flight", "result") and game.flight is not None and game.flight.state == "air" \
                or phase == "flight":
            self._draw_trail(frame, view, game)
        self._draw_particles(frame, view, game)
        if phase in ("aim", "flight", "result", "intro"):
            self._draw_lenny(frame, view, game)
        self._draw_ball(frame, view, game)
        if game.fx.flash > 0:
            f = min(1.0, game.fx.flash) * 0.6
            cv2.addWeighted(frame, 1 - f, frame, 0, 255 * f, dst=frame)
        self._draw_hud(frame, game, view)
        self._draw_pip(frame, pip, game)

    # --- overlays in the world ---

    def _draw_green_grid(self, img, view: View, sc: HoleScene, t: float) -> None:
        lines = []
        for ln in sc.grid_lines:
            q = view.to_cam(ln)
            if q[:, 2].min() < NEAR:
                keep = q[:, 2] >= NEAR
                q = q[keep]
                if len(q) < 2:
                    continue
            lines.append(_fixed(view.project(q)))
        if lines:
            cv2.polylines(img, lines, False, (196, 236, 180), 1, cv2.LINE_AA, SHIFT)
        # Slope arrows: short chevrons drifting downhill, coloured by steepness
        # (white gentle, yellow medium, orange steep), like a green's slope grid.
        pts = sc.flow_pts
        if not len(pts):
            return
        g = np.hypot(pts[:, 2], pts[:, 3])
        d = -pts[:, 2:4] / np.maximum(g, 1e-9)[:, None]
        n = np.stack([-d[:, 1], d[:, 0]], 1)
        phase = (t * (0.4 + np.clip(g / 0.03, 0, 1) * 0.8) + (pts[:, 0] * 0.37 + pts[:, 1] * 0.61)) % 1.0
        tip = pts[:, :2] + d * (phase * 0.9 - 0.45)[:, None]
        back = tip - d * 0.32
        wing_a, wing_b = back + n * 0.16, back - n * 0.16
        stack = np.concatenate([wing_a, tip, wing_b])
        q = view.to_cam(np.c_[stack, np.zeros(len(stack))])
        m = len(pts)
        qa, qt, qb = q[:m], q[m:2 * m], q[2 * m:]
        ok = (qa[:, 2] >= NEAR) & (qt[:, 2] >= NEAR) & (qb[:, 2] >= NEAR) & (g > 0.002) & (qt[:, 2] < 40)
        if not ok.any():
            return
        sa, st, sb = view.project(qa[ok]), view.project(qt[ok]), view.project(qb[ok])
        steep = g[ok]
        for lo, hi, colour in ((0.0, 0.012, (245, 250, 255)), (0.012, 0.025, (90, 230, 255)),
                               (0.025, 9.0, (40, 140, 255))):
            sel = (steep >= lo) & (steep < hi)
            if not sel.any():
                continue
            segs = [_fixed(np.array([a_, t_, b_])) for a_, t_, b_ in zip(sa[sel], st[sel], sb[sel])]
            cv2.polylines(img, segs, False, colour, 2, cv2.LINE_AA, SHIFT)

    def _draw_target(self, img, view: View, game: GolfGame) -> None:
        pv = game.preview
        if pv is None:
            return
        if game.putting:
            # The line the putt starts on, to just past the hole.
            hx, hy = game.ball
            L = min(P.putt_range_m(), game.distance_to_pin + 1.5)
            n = max(2, int(L / 0.4))
            pts = np.array([[hx + math.sin(game.heading) * L * i / n, hy + math.cos(game.heading) * L * i / n, 0.0]
                            for i in range(1, n + 1)])
            q = view.to_cam(pts)
            q = q[q[:, 2] >= NEAR]
            for s in view.project(q):
                cv2.circle(img, (int(s[0] * ONE), int(s[1] * ONE)), 3 * ONE, (255, 255, 255), -1, cv2.LINE_AA, SHIFT)
            return
        # Dotted arc of a full-power straight shot, then the landing ring.
        path = np.array(pv.path)
        if len(path) > 2:
            q = view.to_cam(path[1::2])
            q = q[q[:, 2] >= NEAR]
            if len(q):
                s = view.project(q)
                s = s[s[:, 1] > ARC_TOP]  # the top of the arc would run through the HUD
                for i, p in enumerate(s):
                    r = 4 if i % 2 == 0 else 3
                    c = (int(p[0] * ONE), int(p[1] * ONE))
                    cv2.circle(img, c, (r + 2) * ONE, (40, 30, 30), -1, cv2.LINE_AA, SHIFT)
                    cv2.circle(img, c, r * ONE, (255, 255, 255), -1, cv2.LINE_AA, SHIFT)
        if pv.landing is None:
            return
        lx, ly = pv.landing
        ring = np.array([[lx + 5.0 * math.cos(a), ly + 5.0 * math.sin(a), 0.0]
                         for a in np.linspace(0, 2 * math.pi, 28, endpoint=False)])
        q = view.to_cam(ring)
        if q[:, 2].min() >= NEAR:
            s = view.project(q)
            # Keep the ring readable when it is far away and flattened.
            c = s.mean(axis=0)
            hgt = s[:, 1].max() - s[:, 1].min()
            if hgt < 10:
                s[:, 1] = c[1] + (s[:, 1] - c[1]) * (10 / max(hgt, 1e-3))
            arr = _fixed(s)
            cv2.polylines(img, [arr], True, (30, 30, 40), 6, cv2.LINE_AA, SHIFT)
            cv2.polylines(img, [arr], True, _GOLD, 3, cv2.LINE_AA, SHIFT)
            # A marker hovering over it, bobbing.
            bob = 4 * math.sin(game.clock * 4)
            top = s[:, 1].min() - 14 + bob
            _triangle(img, c[0], top - 8, 11, "down", _GOLD)

    def _draw_trail(self, img, view: View, game: GolfGame) -> None:
        f = game.flight
        if f is None or len(f.path) < 2:
            return
        pts = np.array(f.path[-10:])
        pts = pts[pts[:, 2] > 0.05] if f.state != "air" else pts
        if len(pts) < 2:
            return
        q = view.to_cam(pts)
        q = q[q[:, 2] >= NEAR]
        if len(q) < 2:
            return
        s = _fixed(view.project(q))
        cv2.polylines(img, [s], False, (255, 255, 255), 2, cv2.LINE_AA, SHIFT)

    def _draw_particles(self, img, view: View, game: GolfGame) -> None:
        for p in game.fx.particles:
            pt = view.point(p.x, p.y, p.z)
            if pt is None:
                continue
            sx, sy, z = pt
            r = max(1.5, view.f * (0.09 if p.kind == "water" else 0.07) / z)
            colour = (255, 240, 220) if p.kind == "water" else (170, 214, 240)
            cv2.circle(img, (int(sx * ONE), int(sy * ONE)), int(r * ONE), colour, -1, cv2.LINE_AA, SHIFT)

    def _ball_state(self, game: GolfGame):
        f = game.flight
        if game.phase.value == "flight" and f is not None:
            return (f.x, f.y, f.z)
        if game.phase.value == "holed":
            return None
        return (*game.ball, 0.0)

    def _draw_ball(self, img, view: View, game: GolfGame) -> None:
        b = self._ball_state(game)
        if b is None:
            return
        x, y, z = b
        if game.flight is not None and game.flight.state in ("water", "ob") and game.phase.value != "aim":
            return
        ground = view.point(x, y, 0.0)
        pt = view.point(x, y, z)
        if pt is None:
            return
        sx, sy, d = pt
        r = max(2.5, view.f * float(self.m.get("ball_visual_radius_m", 0.075)) / d)
        if ground is not None:
            gx, gy, _ = ground
            sr = r * (1.0 + min(1.5, z * 0.1))
            cv2.ellipse(img, (int(gx * ONE), int(gy * ONE)), (int(sr * ONE), max(ONE, int(sr * 0.35 * ONE))), 0, 0,
                        360, (40, 70, 40), -1, cv2.LINE_AA, SHIFT)
        spr = self.ball.get(int(round(2 * r * 48 / 42)))
        alpha_composite(img, spr, int(round(sx - spr.shape[1] / 2)), int(round(sy - spr.shape[0] / 2)))

    # --- Lenny ---

    def lenny_scale(self, depth: float) -> float:
        return min(LENNY_MAX, LENNY_WORLD * self.f / max(depth, 0.1))

    def _draw_lenny(self, img, view: View, game: GolfGame) -> None:
        """Lenny stands left of the ball where the shot was played from. His
        arms and club are drawn here every frame, at game.lenny_angle."""
        if game.phase.value == "intro" and game.phase_t < 1.6:
            return
        spot = game.flight.start if (game.flight is not None and game.phase.value != "aim") else game.ball
        pt = view.point(spot[0], spot[1], 0.0)
        if pt is None:
            return
        bx, by, depth = pt
        if depth < 2.0:
            return
        s = self.lenny_scale(depth)
        if s < 0.08:
            return
        spec = self.lenny_spec
        putt = game.putting if game.phase.value == "aim" else (game.flight is not None and game.flight.shot.club.putter)
        arm, club = spec["arm"] * s, spec["club"] * s * (0.62 if putt else 1.0)
        a0 = math.radians(spec["address_dir"] - (12.0 if putt else 0.0))
        ball = (bx - 9.0 * s, by - 2.0 * s)  # the club head rests just behind the ball
        pivot = (ball[0] - (arm + club) * math.sin(a0), ball[1] - (arm + club) * math.cos(a0))
        sb, sf = spec["shoulder_back"], spec["shoulder_front"]
        mid = ((sb[0] + sf[0]) / 2, (sb[1] + sf[1]) / 2)
        spr = self.lenny.get(int(round(self.art.lenny.shape[0] * s)))
        k = spr.shape[0] / self.art.lenny.shape[0]
        ox, oy = pivot[0] - mid[0] * k, pivot[1] - mid[1] * k
        alpha_composite(img, spr, int(round(ox)), int(round(oy)))

        ang = game.lenny_angle
        theta = ang * (THETA_TOP if ang >= 0 else THETA_FINISH)
        if putt:
            theta = ang * 0.9
        a = a0 - theta  # arm direction, from straight down, + toward the ball
        grip = (pivot[0] + arm * math.sin(a), pivot[1] + arm * math.cos(a))
        if theta >= 0:
            hinge = min(theta, 1.6) * 0.85
        else:
            hinge = max(theta, -1.4) * 0.6
        c = a - hinge if not putt else a
        head = (grip[0] + club * math.sin(c), grip[1] + club * math.cos(c))
        col = {k_: tuple(int(v) for v in vv) for k_, vv in spec["colours"].items()}
        ink = col["ink"]
        # Club first (behind the hands), then both arms to the grip, then the glove.
        _capsule(img, grip, head, max(1.5, 3.2 * s), ink)
        _capsule(img, grip, head, max(1.0, 1.8 * s), col["shaft"])
        hd = (math.sin(c), math.cos(c))
        if putt:
            n = (hd[1], -hd[0])
            p0 = (head[0] - n[0] * 9 * s, head[1] - n[1] * 9 * s)
            p1 = (head[0] + n[0] * 9 * s, head[1] + n[1] * 9 * s)
            _capsule(img, p0, p1, max(1.5, 5 * s), ink)
            _capsule(img, p0, p1, max(1.0, 3.4 * s), col["head"])
        else:
            big = 1.35 if (game.club.key in ("1W", "3W") if game.phase.value == "aim" else False) else 1.0
            cv2.ellipse(img, (int(head[0] * ONE), int(head[1] * ONE)),
                        (max(ONE, int(11 * s * big * ONE)), max(ONE, int(6 * s * big * ONE))),
                        math.degrees(math.atan2(hd[1], hd[0])) + 90, 0, 360, ink, -1, cv2.LINE_AA, SHIFT)
            cv2.ellipse(img, (int(head[0] * ONE), int(head[1] * ONE)),
                        (max(ONE, int(9 * s * big * ONE)), max(ONE, int(4.4 * s * big * ONE))),
                        math.degrees(math.atan2(hd[1], hd[0])) + 90, 0, 360, col["head"], -1, cv2.LINE_AA, SHIFT)
        for sh in (sb, sf):
            p = (ox + sh[0] * k, oy + sh[1] * k)
            _capsule(img, p, grip, 10.5 * s, ink)
        for sh in (sb, sf):
            p = (ox + sh[0] * k, oy + sh[1] * k)
            mid_arm = (p[0] + (grip[0] - p[0]) * 0.42, p[1] + (grip[1] - p[1]) * 0.42)
            _capsule(img, p, grip, 8.0 * s, col["fur"])
            _capsule(img, p, mid_arm, 9.0 * s, col["sleeve"])
        cv2.circle(img, (int(grip[0] * ONE), int(grip[1] * ONE)), int((10 * s + 2) * ONE), ink, -1, cv2.LINE_AA, SHIFT)
        cv2.circle(img, (int(grip[0] * ONE), int(grip[1] * ONE)), int(10 * s * ONE), col["glove"], -1, cv2.LINE_AA,
                   SHIFT)

    # --- HUD ---

    def _draw_hud(self, img, game: GolfGame, view: View) -> None:
        phase = game.phase.value
        if phase == "intro":
            self._draw_intro(img, game)
            self._draw_map(img, game)
            return
        if phase != "game_over":
            self._draw_hole_panel(img, game)
            self._draw_wind(img, game, view)
            self._draw_map(img, game)
        if phase == "aim":
            self._draw_club_panel(img, game)
            self._draw_gauge(img, game)
            self._draw_status(img, game)
            self._draw_buttons(img, game)
        elif phase in ("flight", "result"):
            self._draw_gauge(img, game)
        self._draw_message(img, game)
        self._draw_popups(img, game)
        if phase == "holed" and game.phase_t >= 2.2 or phase == "game_over":
            self._draw_scorecard(img, game, final=phase == "game_over")

    def _draw_hole_panel(self, img, game: GolfGame) -> None:
        x0, y0, x1, y1 = HOLE_PANEL
        theme.panel(img, HOLE_PANEL, alpha=0.72)
        h = game.hole
        outlined_text(img, f"HOLE {h.number}", (x0 + 16, y0 + 40), _FONT, 1.05, _GOLD, 2, width=2)
        outlined_text(img, f"PAR {h.par}", (x0 + 200, y0 + 40), _FONT, 0.85, _WHITE, 2, width=2)
        if game.phase.value == "holed":
            dist = "IN THE HOLE"
        else:
            dist = f"{game.distance_text()} to pin"
        outlined_text(img, dist, (x0 + 16, y0 + 76), _BODY, 0.75, _WHITE, 2, width=1)
        shot = game.strokes + (1 if game.phase.value == "aim" else 0)
        tp = game.to_par
        bump = game.fx.score_bump
        col = tuple(int(c + (w - c) * (1 - bump)) for c, w in zip(_GOLD, _WHITE))
        outlined_text(img, f"SHOT {max(1, shot)}", (x0 + 16, y0 + 108), _BODY, 0.66, _DIM, 2, width=1)
        outlined_text(img, f"TOTAL {to_par_text(tp)}", (x0 + 150, y0 + 108), _FONT, 0.72, col, 2, width=1)

    def _draw_club_panel(self, img, game: GolfGame) -> None:
        x0, y0, x1, y1 = CLUB_PANEL
        theme.panel(img, CLUB_PANEL, alpha=0.72)
        outlined_text(img, game.club.key, (x0 + 16, y0 + 46), _FONT, 1.15, _GOLD, 2, width=2)
        outlined_text(img, game.club.name, (x0 + 96, y0 + 30), _BODY, 0.62, _WHITE, 2, width=1)
        lie = {"tee": "TEE", "fairway": "FAIRWAY", "rough": "ROUGH 80%", "bunker": "BUNKER 55%",
               "green": "GREEN"}.get(game.lie, game.lie.upper())
        outlined_text(img, f"{game.club_range_text()}   {lie}", (x0 + 96, y0 + 58), _BODY, 0.55, _DIM, 1, width=1)

    def _draw_buttons(self, img, game: GolfGame) -> None:
        active = game.buttons_active
        for name, rect in game.buttons():
            x0, y0, x1, y1 = rect
            hot = active and game.hover == name
            aim = name.startswith("aim")
            if aim and not active:
                # Just a hint of them while the hands are gripping.
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                _triangle(img, cx, cy, 14, "left" if name == "aim_left" else "right", (200, 200, 210))
                continue
            theme.panel(img, rect, colour=(46, 38, 34) if hot else theme.PANEL, alpha=0.8 if active else 0.45,
                        border=_GOLD if hot else (110, 110, 120), thickness=3 if hot else 2, radius=14)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            arrow = {"club_up": "up", "club_down": "down", "aim_left": "left", "aim_right": "right"}[name]
            label = {"club_up": "CLUB", "club_down": "CLUB", "aim_left": "AIM", "aim_right": "AIM"}[name]
            colour = _GOLD if hot else (_WHITE if active else (150, 150, 160))
            if aim:
                ax = x0 + 26 if arrow == "left" else x1 - 26
                _triangle(img, ax, cy, 13, arrow, colour)
                theme.centred_text(img, label, (x0 + x1) / 2 + (12 if arrow == "left" else -12), int(cy + 10), 0.75,
                                   colour, 2)
            else:
                _triangle(img, x0 + 28, cy, 12, arrow, colour)
                theme.centred_text(img, label, (x0 + x1) / 2 + 14, int(cy + 10), 0.72, colour, 2)
            if hot and game.hover_t > 0:
                fill = min(1.0, game.hover_t / 0.6)
                cv2.rectangle(img, (x0 + 10, y1 - 9), (x0 + 10 + int((x1 - x0 - 20) * fill), y1 - 5), _GOLD, -1)
        if active and game.cursor is not None:
            c = (int(game.cursor[0]), int(game.cursor[1]))
            cv2.circle(img, c, 20, (20, 20, 20), 5, cv2.LINE_AA)
            cv2.circle(img, c, 20, _WHITE, 2, cv2.LINE_AA)
            cv2.circle(img, c, 4, _WHITE, -1, cv2.LINE_AA)

    def _draw_gauge(self, img, game: GolfGame) -> None:
        gx, gy = GAUGE_AT
        alpha_composite(img, self.gauge, gx, gy)
        d = game.detector
        if game.phase.value == "aim":
            power = d.power if (d.swinging or d.locked) else 0.0
            practice = d.swinging and not d.armed
        else:
            power = (game.last_shot or {}).get("power", 0.0)
            practice = False
        x0, y0, x1, y1 = self.gauge_inner
        top_frac = min(1.15, max(0.0, power)) / 1.15
        if top_frac > 0:
            yy = int(round(y1 - (y1 - y0) * top_frac))
            alpha_composite(img, self.gauge_fill[yy:], gx, gy + yy)
        label = "PUTT" if game.putting and game.phase.value == "aim" else "POWER"
        theme.centred_text(img, label, gx + self.gauge.shape[1] / 2, gy - 6, 0.5, _WHITE, 1)
        if power > 0:
            pct = f"{int(round(power * 100))}%"
            colour = theme.DANGER if power > 1.0 else _GOLD
            theme.centred_text(img, pct, gx + self.gauge.shape[1] + 40, int(gy + y1 - (y1 - y0) * top_frac + 10), 0.8,
                               colour, 2)
        if practice:
            outlined_text(img, "PRACTICE", (gx + 4, gy + self.gauge.shape[0] + 26), _BODY, 0.55, _DIM, 1, width=1)

    def _draw_status(self, img, game: GolfGame) -> None:
        d = game.detector
        cx = self.width / 2
        y = 150
        msg = d.guidance()
        if d.hint:
            theme.centred_text(img, d.hint, cx, y, 1.1, _WARN, 3)
            return
        if d.stage.value == "ready":
            theme.centred_text(img, "READY", cx, y, 1.2, _CYAN, 3)
            sub = "rock your hands back, then through" if game.putting else "swing back, then through"
            theme.centred_text(img, sub, cx, y + 34, 0.7, _WHITE, 2, _BODY)
        elif d.swinging and not d.armed:
            theme.centred_text(img, "PRACTICE SWING", cx, y, 1.0, _WHITE, 2)
            theme.centred_text(img, "hold your hands still at your waist first", cx, y + 32, 0.62, _DIM, 1, _BODY)
        elif d.swinging:
            pass
        elif d.stage.value == "addressing":
            theme.centred_text(img, "HOLD STILL", cx, y, 1.0, _WHITE, 2)
            p = d.hold_progress
            cv2.ellipse(img, (int(cx), y + 36), (18, 18), -90, 0, 360 * p, _GOLD, 4, cv2.LINE_AA)
        elif msg:
            colour = _WARN if msg.startswith(("STEP", "SHOW")) else _WHITE
            theme.centred_text(img, msg, cx, y, 0.95, colour, 2)
            if game.buttons_active:
                theme.centred_text(img, "or point at AIM / CLUB to adjust", cx, y + 32, 0.6, _DIM, 1, _BODY)
        elif game.practice_t >= 0:
            pass

    def _draw_wind(self, img, game: GolfGame, view: View) -> None:
        theme.panel(img, WIND_PANEL, alpha=0.72)
        x0, y0, x1, y1 = WIND_PANEL
        cx, cy = (x0 + x1) / 2, y0 + 38
        mph = game.wind_mph
        if mph < 0.5:
            theme.centred_text(img, "CALM", cx, int(cy + 10), 0.8, _WHITE, 2)
        else:
            # Arrow in the camera's frame: up the screen = away from the camera.
            a = game.wind_dir - view.yaw
            dx, dy = math.sin(a), -math.cos(a)
            dy *= 0.75  # a little perspective
            L = 26
            tail = (cx - dx * L, cy - dy * L)
            tip = (cx + dx * L, cy + dy * L)
            colour = _CYAN if mph < 6 else _GOLD if mph < 10 else theme.DANGER
            _capsule(img, tail, tip, 5, (20, 16, 24))
            _capsule(img, tail, tip, 3, colour)
            nx, ny = -dy, dx
            head = np.array([tip, (tip[0] - dx * 14 + nx * 10, tip[1] - dy * 14 + ny * 10),
                             (tip[0] - dx * 14 - nx * 10, tip[1] - dy * 14 - ny * 10)])
            cv2.fillPoly(img, [_fixed(head)], colour, cv2.LINE_AA, SHIFT)
        theme.centred_text(img, f"{mph:.0f} mph" if mph >= 0.5 else "no wind", cx, y1 - 12, 0.6, _WHITE, 1, _BODY)

    def _draw_map(self, img, game: GolfGame) -> None:
        base, to_map = self.hole_map(game)
        x0, y0, x1, y1 = MAP
        roi = img[y0:y1, x0:x1]
        mp = base.copy()
        # Aim line and landing target, the pin, the ball.
        bx, by = to_map(*game.ball) if game.phase.value != "flight" or game.flight is None else to_map(
            game.flight.x, game.flight.y)
        if game.phase.value == "aim" and game.preview is not None:
            tgt = game.preview.landing if not game.putting else game.hole.pin
            if tgt is not None:
                tx, ty = to_map(*tgt)
                cv2.line(mp, (int(bx), int(by)), (int(tx), int(ty)), (255, 255, 255), 1, cv2.LINE_AA)
                cv2.circle(mp, (int(tx), int(ty)), 5, _GOLD, 2, cv2.LINE_AA)
        if game.flight is not None and game.phase.value in ("flight", "result"):
            pts = np.array([to_map(p[0], p[1]) for p in game.flight.path[::3]], np.float64)
            if len(pts) > 1:
                cv2.polylines(mp, [_fixed(pts)], False, (255, 255, 255), 1, cv2.LINE_AA, SHIFT)
        px, py = to_map(*game.hole.pin)
        cv2.line(mp, (int(px), int(py)), (int(px), int(py) - 12), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.fillPoly(mp, [np.array([[int(px), int(py) - 12], [int(px) + 9, int(py) - 9], [int(px), int(py) - 6]])],
                     (60, 70, 235), cv2.LINE_AA)
        cv2.circle(mp, (int(bx), int(by)), 5, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(mp, (int(bx), int(by)), 3, (255, 255, 255), -1, cv2.LINE_AA)
        np.copyto(roi, mp, where=self._map_mask)
        _border(img, (x0 - 1, y0 - 1, x1, y1), 12, (20, 20, 20), 5)
        _border(img, (x0 - 1, y0 - 1, x1, y1), 12, (235, 235, 235), 2)

    def _draw_message(self, img, game: GolfGame) -> None:
        m = game.message
        if m is None:
            return
        age = m.life
        scale = m.size * _pop(age)
        alpha = 1.0 if age < m.duration - 0.35 else max(0.0, (m.duration - age) / 0.35)
        y = int(self.height * 0.40)
        thick = 5 if m.size >= 2 else 3
        _faded_text(img, m.text, self.width / 2, y, scale, m.colour, alpha, thick)
        if m.sub:
            _faded_text(img, m.sub, self.width / 2, y + 50, 0.75, _WHITE, alpha, 2, _BODY)

    def _draw_popups(self, img, game: GolfGame) -> None:
        for p in game.fx.popups:
            age = p.life / p.duration
            y = p.pos[1] - 60 * _ease_out(age)
            alpha = 1.0 if age < 0.6 else max(0.0, (1 - age) / 0.4)
            _faded_text(img, p.text, p.pos[0], y, 1.6 * _pop(p.life), p.colour, alpha, 3)

    def _draw_intro(self, img, game: GolfGame) -> None:
        t = game.phase_t
        h = game.hole
        a = min(1.0, t / 0.4) * (1.0 if t < 2.5 else max(0.0, (3.0 - t) / 0.5))
        cx = self.width / 2
        if a <= 0:
            return
        _faded_text(img, f"HOLE {h.number}", cx, 230, 2.6 * _pop(t), _GOLD, a, 5)
        _faded_text(img, h.name.upper(), cx, 285, 1.0, _WHITE, a, 2)
        yd = h.length * YARDS_PER_M
        _faded_text(img, f"PAR {h.par}   -   {yd:.0f} yd", cx, 335, 1.1, _WHITE, a, 2)
        wind = f"WIND {game.wind_mph:.0f} mph" if game.wind_mph >= 0.5 else "NO WIND"
        _faded_text(img, wind, cx, 375, 0.75, _CYAN, a, 2, _BODY)

    def _draw_scorecard(self, img, game: GolfGame, final: bool) -> None:
        W = self.width
        x0, x1 = W // 2 - 380, W // 2 + 380
        y0, y1 = (110, 540) if final else (150, 470)
        theme.panel(img, (x0, y0, x1, y1), alpha=0.9, border=_GOLD)
        title = "ROUND COMPLETE" if final else "SCORECARD"
        theme.centred_text(img, title, W / 2, y0 + 52, 1.2, _GOLD, 3)
        cols = len(game.card)
        cw = 120
        label_w = 190
        gx = x0 + 40
        gy = y0 + 104
        heads = ["HOLE"] + [str(i + 1) for i in range(cols)] + ["TOTAL"]
        par_total = sum(r.par for r in game.card)
        rows = [
            ("PAR", [str(r.par) for r in game.card], str(par_total)),
            ("STROKES", [str(r.strokes) if r.done else "-" for r in game.card],
             str(sum(r.strokes for r in game.card if r.done))),
            ("SCORE", [self._diff_text(r) for r in game.card], to_par_text(game.to_par)),
            ("POINTS", [str(r.points) if r.done else "-" for r in game.card], str(game.score)),
        ]
        for j, hd in enumerate(heads):
            x = gx + (0 if j == 0 else label_w + (j - 1) * cw + cw / 2)
            if j == 0:
                outlined_text(img, hd, (int(x), gy), _BODY, 0.7, _DIM, 2, width=1)
            else:
                theme.centred_text(img, hd, x, gy, 0.75, _DIM, 2)
        cv2.line(img, (gx, gy + 14), (x1 - 40, gy + 14), (110, 110, 120), 2, cv2.LINE_AA)
        for i, (name, vals, total) in enumerate(rows):
            y = gy + 56 + i * 48
            outlined_text(img, name, (gx, y), _BODY, 0.75, _WHITE, 2, width=1)
            for j, v in enumerate(vals + [total]):
                x = gx + label_w + j * cw + cw / 2
                current = j == game.hole_i and not final
                colour = _GOLD if j == len(vals) else (_CYAN if current else _WHITE)
                if name == "SCORE" and v not in ("-",):
                    if v.startswith("-"):
                        colour = _CYAN if j < len(vals) else _GOLD
                theme.centred_text(img, v, x, y, 0.95, colour, 2)
        if final:
            tp = game.to_par
            verdict = ("UNDER PAR - SUPERB!" if tp < 0 else "LEVEL PAR - NICE ROUND!" if tp == 0
                       else "WELL PLAYED!" if tp <= 4 else "KEEP SWINGING!")
            theme.centred_text(img, verdict, W / 2, y1 - 28, 0.9, _CYAN, 2)

    @staticmethod
    def _diff_text(r: CardRow) -> str:
        if not r.done:
            return "-"
        if r.picked_up:
            return "PU"
        return to_par_text(r.strokes - r.par)

    # --- the camera window ---

    def _draw_pip(self, img, pip: np.ndarray, game: GolfGame) -> None:
        x0, y0, x1, y1 = PIP
        s = (x1 - x0) / self.width
        d = game.detector
        pose = game.last_pose
        if pose is not None:
            pts = {}
            for n in ("left_shoulder", "right_shoulder", "left_wrist", "right_wrist", "left_elbow", "right_elbow"):
                i = KP[n]
                if pose.confidence[i] >= 0.4:
                    pts[n] = (int(pose.xy[i][0] * s), int(pose.xy[i][1] * s))
            for a, b in (("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),
                         ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"),
                         ("right_elbow", "right_wrist")):
                if a in pts and b in pts:
                    cv2.line(pip, pts[a], pts[b], (20, 20, 20), 4, cv2.LINE_AA)
                    cv2.line(pip, pts[a], pts[b], (230, 230, 230), 2, cv2.LINE_AA)
        g = d.grip
        if g is not None:
            colour = _CYAN if d.ready else _GOLD if d.swinging else (_WHITE if d.together else _WARN)
            c = (int(g[0] * s), int(g[1] * s))
            cv2.circle(pip, c, 9, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(pip, c, 7, colour, -1, cv2.LINE_AA)
        if d.address is not None and d.body_frame is not None and game.phase.value == "aim":
            fx, fy, sw = d.body_frame
            au, ah = d.address
            ax, ay = (fx + au * sw) * s, (fy - ah * sw) * s
            cv2.line(pip, (int(ax), int(ay - 30)), (int(ax), int(ay + 20)), _CYAN, 1, cv2.LINE_AA)
        if not d.visible and game.phase.value == "aim":
            cy = (y1 - y0) // 2
            cv2.rectangle(pip, (8, cy - 22), (x1 - x0 - 8, cy + 14), (20, 20, 20), -1)
            theme.centred_text(pip, "STEP INTO VIEW", (x1 - x0) / 2, cy + 6, 0.7, _WARN, 2)
        roi = img[y0:y1, x0:x1]
        np.copyto(roi, pip, where=self._pip_mask)
        border = _CYAN if d.ready else _GOLD if d.swinging else (235, 235, 235)
        _border(img, (x0, y0, x1 - 1, y1 - 1), 14, (20, 20, 20), 6)
        _border(img, (x0, y0, x1 - 1, y1 - 1), 14, border, 3)
