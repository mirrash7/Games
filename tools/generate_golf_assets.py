#!/usr/bin/env python3
"""Procedurally generate every sprite and backdrop for Raccoon Golf.

Run once (or whenever the look changes):

    uv run python tools/generate_golf_assets.py

numpy + opencv only, nothing downloaded or traced: an original look in the
spirit of a sunny resort-island golf game. Output goes to
``assets/golf/generated/`` with ``manifest.json``; the schema is defined by
``kpapp.game.golf.art``. The course itself (polygons, pins, trees) is data in
``assets/golf/course.json``; the ground is drawn from it at run time.

Look target: bright, clean and friendly. A deep blue sky with soft cumulus, a
turquoise sea on the horizon with hazy islands, chunky tropical trees with a
dark keyline, and Lenny - the arcade's purple raccoon - in a golfer's visor and
a sunny yellow polo, seen from behind as he addresses the ball. His arms and
club are not in the sprite: the game draws them every frame so they follow the
player's own swing (the manifest says where they attach).

Technique: the supersampled, premultiplied ``Canvas`` from
``generate_flappy_assets.py`` (imported, not copied), so every edge is clean
and outlines come from a distance transform. Determinism: one seed and
per-tag generators keyed by crc32; two runs give byte-identical files
(checked with md5 when this was written; re-check after changing it).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import zlib
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "assets" / "golf" / "generated"
SEED = 20261016


def _load_flappy_gen():
    spec = importlib.util.spec_from_file_location("_flappy_gen", ROOT / "tools" / "generate_flappy_assets.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_fg = _load_flappy_gen()
Canvas = _fg.Canvas
_blur = _fg._blur
_vramp = _fg._vramp
_dither_u8 = _fg._dither_u8


def rgb(r: float, g: float, b: float) -> tuple[float, float, float]:
    """Palette entries are written as RGB and stored as BGR."""
    return (float(b), float(g), float(r))


def _rng(tag: str) -> np.random.Generator:
    return np.random.default_rng(SEED + zlib.crc32(tag.encode()))


# ---------------------------------------------------------------------------
# palette
# ---------------------------------------------------------------------------

INK = rgb(46, 24, 52)  # the arcade's plum keyline
# Lenny (same purples as Snack Attack and Flappy Raccoon)
FUR = rgb(185, 165, 215)
FUR_HI = rgb(214, 200, 238)
FUR_DEEP = rgb(104, 63, 155)
FUR_WHITE = rgb(250, 248, 255)
NOSE = rgb(30, 22, 18)
POLO = rgb(255, 204, 60)
POLO_SH = rgb(232, 160, 36)
POLO_HI = rgb(255, 232, 140)
COLLAR = rgb(255, 255, 255)
SHORTS = rgb(46, 62, 120)
SHORTS_SH = rgb(32, 42, 88)
VISOR = rgb(255, 255, 255)
VISOR_UNDER = rgb(104, 63, 155)
SOCK = rgb(255, 255, 255)
SHOE = rgb(248, 248, 250)
SHOE_SOLE = rgb(60, 180, 190)
GLOVE = rgb(255, 255, 255)
SHAFT = rgb(205, 212, 222)
CLUB_HEAD = rgb(52, 56, 70)
CLUB_FACE = rgb(170, 178, 190)

# scenery
SKY_STOPS = [(0.00, rgb(52, 126, 214)), (0.45, rgb(98, 170, 236)), (0.82, rgb(168, 214, 246)),
             (1.00, rgb(214, 238, 252))]
CLOUD = rgb(255, 255, 255)
CLOUD_SHADE = rgb(214, 230, 248)
SUN = rgb(255, 250, 214)
ISLAND_FAR = rgb(128, 172, 196)
ISLAND_NEAR = rgb(96, 150, 170)
SEA_STOPS = [(0.00, rgb(150, 214, 230)), (0.10, rgb(64, 172, 210)), (0.40, rgb(28, 136, 196)),
             (1.00, rgb(18, 102, 168))]
TRUNK = rgb(150, 104, 66)
TRUNK_DK = rgb(110, 72, 44)
TRUNK_HI = rgb(196, 150, 100)
LEAF = rgb(64, 168, 72)
LEAF_DK = rgb(36, 118, 58)
LEAF_HI = rgb(140, 214, 96)
CANOPY = rgb(70, 160, 80)
CANOPY_DK = rgb(42, 116, 62)
CANOPY_HI = rgb(130, 204, 100)
COCONUT = rgb(120, 82, 48)

# The ground's flat colours (BGR in the manifest, drawn by the renderer).
GROUND = {
    "rough": rgb(84, 160, 64),
    "fairway": rgb(128, 204, 86),
    "fairway_stripe": rgb(112, 190, 74),
    "green": rgb(146, 222, 108),
    "green_stripe": rgb(134, 212, 98),
    "fringe": rgb(116, 196, 82),
    "tee": rgb(136, 210, 92),
    "bunker": rgb(242, 226, 176),
    "bunker_rim": rgb(206, 182, 128),
    "water": rgb(52, 150, 214),
    "water_hi": rgb(150, 210, 240),
    "water_rim": rgb(236, 230, 196),
    "beach": rgb(240, 224, 170),
    "edge": rgb(52, 112, 46),
    "haze": rgb(214, 236, 248),
    "sea": rgb(30, 140, 200),
}

# Geometry the renderer needs; all at 1280x720.
FOCAL = 1000.0  # px: the game's pinhole focal length (65 degrees across)
PANO_W = int(round(2 * math.pi * FOCAL))  # one full turn of yaw, at full resolution
PANO_H = 440  # px of sky above the horizon
SEA_H = 560


# ---------------------------------------------------------------------------
# sky panorama (one full turn; the game slices it by yaw)
# ---------------------------------------------------------------------------

CLOUDS = [  # (x fraction of the turn, height above horizon px, scale)
    (0.02, 300, 1.2), (0.09, 190, 0.8), (0.17, 330, 1.0), (0.24, 150, 0.7), (0.31, 270, 1.3),
    (0.40, 360, 0.9), (0.47, 200, 1.1), (0.55, 310, 0.8), (0.62, 140, 0.75), (0.70, 280, 1.25),
    (0.78, 360, 0.85), (0.85, 210, 1.0), (0.93, 320, 1.1),
]
CLOUD_PUFFS = [(-60, 6, 30), (-26, -16, 38), (14, -24, 34), (50, -4, 30), (6, 4, 36), (78, 10, 20), (-86, 12, 18)]
ISLANDS = [  # (x fraction, width px, height px, kind)
    (0.06, 420, 46, "hill"), (0.21, 260, 30, "flat"), (0.36, 560, 90, "volcano"), (0.52, 300, 34, "hill"),
    (0.66, 380, 54, "hill"), (0.80, 220, 24, "flat"), (0.90, 480, 64, "hill"),
]
SUN_AT = (0.12, 330)


def make_sky() -> np.ndarray:
    """BGR panorama at half resolution (the loader doubles it): PANO_W/2 x PANO_H/2."""
    w, h = PANO_W // 2, PANO_H // 2
    c = Canvas(w, h, ss=2, wrap=True)
    full = np.ones((c.H, c.W), np.float32)
    ramp = _vramp(c.H, c.W, SKY_STOPS)
    c.paint(full, ramp)
    k = 0.5  # half resolution

    sx, sy = SUN_AT[0] * w, h - SUN_AT[1] * k
    dx = (c.xs[None, :] - sx + w / 2) % w - w / 2
    dist = np.sqrt(dx ** 2 + (c.ys[:, None] - sy) ** 2)
    c.paint(np.exp(-(dist / 120.0) ** 2) * 0.45, SUN)
    c.paint(c.circle(sx, sy, 30), SUN, 0.25)
    c.paint(c.circle(sx, sy, 19), SUN)

    for fx, hy, s in CLOUDS:
        cx, cy, s = fx * w, h - hy * k, s * k
        m = c.zeros()
        for (px, py, r) in CLOUD_PUFFS:
            m = np.maximum(m, c.circle(cx + px * s, cy + py * s, r * s))
        m *= 1.0 - c.halfplane_y(cy + 24 * s)
        m = np.maximum(m, c.rrect(cx - 80 * s, cy + 6 * s, cx + 90 * s, cy + 24 * s, 10 * s))
        fade = 0.75 + 0.25 * min(1.0, hy / 300.0)  # lower clouds sit in the haze
        c.paint(m, CLOUD, fade)
        c.paint(c.edge_band(m, 0.0, -8.0 * s), CLOUD_SHADE, fade)

    # Hazy islands on the horizon, the far ones paler.
    for i, (fx, iw, ih, kind) in enumerate(ISLANDS):
        cx, iw, ih = fx * w, iw * k, ih * k
        pts = []
        n = 40
        for j in range(n + 1):
            u = j / n
            x = cx - iw / 2 + iw * u
            if kind == "volcano":
                prof = min(1.0, (1 - abs(u - 0.5) * 2) * 1.6) ** 1.3
                if abs(u - 0.5) < 0.07:
                    prof *= 0.93  # a crater notch
            elif kind == "flat":
                prof = math.sin(math.pi * u) ** 0.35
            else:
                prof = math.sin(math.pi * u) ** 1.4 * (0.85 + 0.15 * math.sin(u * 9 + i))
            pts.append((x, h - ih * prof))
        pts += [(cx + iw / 2, h + 2), (cx - iw / 2, h + 2)]
        m = c.poly(pts)
        col = ISLAND_FAR if ih < 30 * k * 1.5 else ISLAND_NEAR
        c.paint(m, col, 0.85)
        c.paint(c.edge_band(m, 0.0, -2.0), rgb(170, 204, 214), 0.5)
    # Horizon haze over the islands' feet.
    haze = np.clip(1.0 - (h - c.ys) / (14.0), 0, 1)[:, None] * np.ones((1, c.W), np.float32)
    c.paint(haze * 0.6, rgb(200, 230, 246))
    return c.to_bgr("sky", dither=0.8)


def make_sea() -> np.ndarray:
    """BGR, 1280 wide (tiles horizontally), SEA_H rows from the horizon down."""
    w, h = 1280, SEA_H
    c = Canvas(w, h, ss=1, wrap=True)
    full = np.ones((c.H, c.W), np.float32)
    c.paint(full, _vramp(c.H, c.W, SEA_STOPS))
    r = _rng("sea")
    glint = c.zeros()
    for _ in range(420):
        y = float(r.uniform(2, h)) ** 1.0
        y = 2 + (y / h) ** 2.2 * h  # denser near the horizon
        x = float(r.uniform(0, w))
        L = 6 + 30 * (y / h)
        th = 0.6 + 1.6 * (y / h)
        glint = np.maximum(glint, c.rrect(x - L, y - th / 2, x + L, y + th / 2, th / 2) * float(r.uniform(0.25, 0.7)))
    c.paint(glint, rgb(214, 244, 252), 0.55)
    return c.to_bgr("sea", dither=0.8)


def make_haze() -> np.ndarray:
    """BGRA strip blended over the ground just below the horizon: distance haze."""
    w, h = 1280, 64
    a = np.clip(1.0 - np.arange(h, dtype=np.float32) / h, 0, 1) ** 1.6 * 0.7
    out = np.zeros((h, w, 4), np.uint8)
    out[:, :, :3] = np.array(GROUND["haze"], np.uint8)
    out[:, :, 3] = (a[:, None] * 255 + 0.5).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# trees (anchored at the trunk's foot)
# ---------------------------------------------------------------------------

PALM_W, PALM_H = 300, 380
PALM_FOOT = (138, 374)
ROUND_W, ROUND_H = 280, 300
ROUND_FOOT = (140, 294)


def _frond(c: Canvas, x0, y0, ang, length, width, droop, dark=False):
    """A palm frond: a curved, tapering leaf with a dark midrib and notched edge."""
    pts_l, pts_r, spine = [], [], []
    n = 24
    for i in range(n + 1):
        t = i / n
        x = x0 + math.cos(math.radians(ang)) * length * t
        y = y0 + math.sin(math.radians(ang)) * length * t + droop * 60 * t * t
        spine.append((x, y))
    for i, (x, y) in enumerate(spine):
        j0, j1 = max(0, i - 1), min(n, i + 1)
        dx, dy = spine[j1][0] - spine[j0][0], spine[j1][1] - spine[j0][1]
        L = math.hypot(dx, dy) + 1e-6
        nx, ny = -dy / L, dx / L
        t = i / n
        wdt = width * math.sin(math.pi * min(1.0, t * 1.1)) ** 0.8
        wdt *= 0.82 + 0.18 * (1 if i % 3 else 0)  # notches
        pts_l.append((x + nx * wdt, y + ny * wdt))
        pts_r.append((x - nx * wdt * 0.8, y - ny * wdt * 0.8))
    leaf = c.poly(pts_l + pts_r[::-1])
    c.inked(leaf, LEAF_DK if dark else LEAF, INK, 2.2)
    if not dark:
        c.paint(leaf * (1.0 - c.shift(leaf, 0.0, -3.0)), LEAF_HI, 0.8)
    rib = c.zeros()
    for (ax, ay), (bx, by) in zip(spine[:-4], spine[1:-3]):
        rib = np.maximum(rib, c.poly([(ax, ay - 1.1), (bx, by - 1.1), (bx, by + 1.1), (ax, ay + 1.1)]))
    c.paint(rib * leaf, LEAF_DK if not dark else INK, 0.9)


def make_palm() -> np.ndarray:
    c = Canvas(PALM_W, PALM_H)
    fx, fy = PALM_FOOT
    # Trunk: a gentle S-curve of stacked rings.
    top = (fx + 26.0, 120.0)
    trunk = c.zeros()
    rings = c.zeros()
    n = 22
    pts = []
    for i in range(n + 1):
        t = i / n
        x = fx + (top[0] - fx) * t + 10 * math.sin(t * math.pi)
        y = fy + (top[1] - fy) * t
        pts.append((x, y, 11.5 - 4.0 * t))
    for i, (x, y, r) in enumerate(pts):
        seg = c.ellipse(x, y, r, 7.0)
        trunk = np.maximum(trunk, seg)
        if i % 2 == 0:
            rings = np.maximum(rings, c.ellipse(x, y + 3.0, r * 0.95, 2.0))
    c.inked(trunk, TRUNK, INK, 2.4)
    c.paint(trunk * (1.0 - c.shift(trunk, 4.0, 0.0)), TRUNK_HI, 0.7)
    c.paint(trunk * c.shift(trunk, 0.0, 0.0) * (1.0 - c.shift(trunk, -4.0, 0.0)), TRUNK_DK, 0.6)
    c.paint(rings * trunk, TRUNK_DK, 0.8)
    tx, ty = top
    # Back fronds (darker), coconuts, front fronds.
    for ang, L, droop in ((-160, 100, 0.9), (-20, 104, 0.8), (-112, 80, 0.4), (-68, 84, 0.4)):
        _frond(c, tx, ty, ang, L, 17, droop, dark=True)
    for dx, dy in ((-7, 10), (6, 12), (0, 4)):
        nut = c.circle(tx + dx, ty + dy, 6.5)
        c.inked(nut, COCONUT, INK, 1.8)
    for ang, L, droop in ((-178, 116, 1.1), (-2, 118, 1.0), (-140, 98, 0.6), (-40, 100, 0.6),
                          (160, 80, 1.4), (20, 82, 1.4), (-92, 72, 0.2)):
        _frond(c, tx, ty, ang, L, 19, droop)
    return c.to_bgra("palm")


def make_round_tree() -> np.ndarray:
    c = Canvas(ROUND_W, ROUND_H)
    fx, fy = ROUND_FOOT
    trunk = c.poly([(fx - 11, fy), (fx + 11, fy), (fx + 7, fy - 70), (fx + 30, fy - 120), (fx + 22, fy - 124),
                    (fx + 2, fy - 92), (fx - 18, fy - 126), (fx - 26, fy - 120), (fx - 6, fy - 70)])
    c.inked(trunk, TRUNK, INK, 2.4)
    c.paint(trunk * (1.0 - c.shift(trunk, 4.0, 0.0)), TRUNK_HI, 0.6)
    # A wide umbrella canopy of overlapping puffs, flat underneath.
    r = _rng("canopy")
    canopy = c.zeros()
    puffs = []
    for i in range(15):
        u = i / 14
        x = fx - 112 + 224 * u + float(r.uniform(-6, 6))
        y = fy - 150 - 52 * math.sin(math.pi * u) ** 0.8 + float(r.uniform(-6, 6))
        rad = 30 + 16 * math.sin(math.pi * u) + float(r.uniform(-4, 4))
        puffs.append((x, y, rad))
        canopy = np.maximum(canopy, c.circle(x, y, rad))
    canopy = np.maximum(canopy, c.ellipse(fx, fy - 150, 118, 30))
    canopy *= 1.0 - c.halfplane_y(fy - 128)
    canopy = np.maximum(canopy, c.ellipse(fx, fy - 134, 112, 12) * c.halfplane_y(fy - 140))
    c.inked(canopy, CANOPY, INK, 2.6)
    c.paint(canopy * c.halfplane_y(fy - 150) * (1.0 - c.halfplane_y(fy - 120)), CANOPY_DK, 0.75)
    hi = c.zeros()
    for x, y, rad in puffs:
        hi = np.maximum(hi, c.circle(x - rad * 0.2, y - rad * 0.35, rad * 0.55))
    c.paint(hi * canopy * (1.0 - c.halfplane_y(fy - 160)), CANOPY_HI, 0.7)
    return c.to_bgra("round")


# ---------------------------------------------------------------------------
# Lenny the golfer, seen from behind at address (no arms: the game draws them)
# ---------------------------------------------------------------------------

LENNY_W, LENNY_H = 240, 300
# Where the arms attach and how long they are, native px.
SHOULDER_BACK = (114.0, 124.0)
SHOULDER_FRONT = (156.0, 118.0)
ARM_LEN = 84.0
CLUB_LEN = 132.0
ADDRESS_DIR = 34.0  # degrees right of straight down: arms + club point at the ball


def make_lenny() -> np.ndarray:
    c = Canvas(LENNY_W, LENNY_H)

    # Tail, behind everything: big, fluffy, ringed, curling down to the left.
    tail, rings = c.zeros(), c.zeros()
    p0, p1, p2 = np.array([104.0, 196.0]), np.array([58.0, 196.0]), np.array([34.0, 246.0])
    n = 9
    for i in range(n):
        t = i / (n - 1)
        q = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t * t * p2
        rr = 15.0 - 5.0 * t
        seg = c.ellipse(float(q[0]), float(q[1]), rr, rr * 0.92)
        tail = np.maximum(tail, seg)
        if i in (3, 5, 7, 8):
            rings = np.maximum(rings, seg)
    c.inked(tail, FUR, INK, 2.6)
    c.paint(tail * rings, FUR_DEEP)
    c.paint(tail * (1.0 - c.shift(tail, 0.0, -4.0)), FUR_HI, 0.5)

    # Legs (fur), knees flexed toward the ball, white socks, golf shoes.
    legs = c.zeros()
    for (hx, hy), (kx, ky), (ax, ay) in (((112, 214), (108, 248), (104, 274)), ((140, 214), (152, 246), (148, 274))):
        for (x0, y0), (x1, y1) in (((hx, hy), (kx, ky)), ((kx, ky), (ax, ay))):
            L = math.hypot(x1 - x0, y1 - y0)
            ang = math.degrees(math.atan2(y1 - y0, x1 - x0))
            legs = np.maximum(legs, c.ellipse((x0 + x1) / 2, (y0 + y1) / 2, L / 2 + 6, 9.5, ang))
    c.inked(legs, FUR, INK, 2.4)
    socks = c.zeros()
    for ax in (104, 148):
        socks = np.maximum(socks, c.rrect(ax - 9.5, 264, ax + 9.5, 280, 4))
    c.inked(socks, SOCK, INK, 2.0)
    shoes = c.zeros()
    soles = c.zeros()
    for ax in (104, 148):
        s = c.ellipse(ax + 8, 284, 19, 10)
        shoes = np.maximum(shoes, s)
        soles = np.maximum(soles, s * c.halfplane_y(287))
    c.inked(shoes, SHOE, INK, 2.4)
    c.paint(soles, SHOE_SOLE)

    # Shorts.
    shorts = c.zeros()
    shorts = np.maximum(shorts, c.rrect(98, 194, 156, 222, 9))
    shorts = np.maximum(shorts, c.ellipse(110, 226, 14, 16, 10))
    shorts = np.maximum(shorts, c.ellipse(146, 224, 15, 15, -20))
    c.inked(shorts, SHORTS, INK, 2.4)
    c.paint(shorts * c.halfplane_y(214) * (1.0 - c.halfplane_y(240)), SHORTS_SH, 0.5)

    # Torso: the polo, bent forward from the hips toward the ball.
    torso = c.ellipse(132, 158, 36, 52, -24)
    torso = np.maximum(torso, c.rrect(100, 182, 158, 204, 10))
    c.inked(torso, POLO, INK, 2.6)
    c.paint(torso * (1.0 - c.shift(torso, -6.0, 0.0)), POLO_SH, 0.8)  # shade on the back's left edge
    c.paint(torso * (1.0 - c.shift(torso, 0.0, 5.0)) * (1.0 - c.halfplane_y(150)), POLO_HI, 0.6)
    # A purple band across the back, and the belt.
    band = c.poly([(100, 160), (168, 140), (170, 150), (102, 171)]) * torso
    c.paint(band, FUR_DEEP, 0.9)
    belt = c.rrect(98, 196, 158, 203, 3)
    c.inked(belt, rgb(255, 255, 255), INK, 1.6)

    # Neck fur and collar.
    neck = c.ellipse(150, 110, 16, 10, -20)
    c.inked(neck, FUR, INK, 2.2)
    collar = c.poly([(130, 116), (160, 102), (170, 108), (138, 124)])
    c.inked(collar, COLLAR, INK, 1.8)

    # Head from behind, turned toward the ball.
    hx, hy = 152.0, 74.0
    ears = c.zeros()
    ear_tips = c.zeros()
    for ex, ey, ang in ((128, 46, -28), (172, 40, 22)):
        ear = c.ellipse(ex, ey, 11, 15, ang)
        ears = np.maximum(ears, ear)
        ear_tips = np.maximum(ear_tips, ear * (1.0 - c.halfplane_y(ey - 2)))
    c.inked(ears, FUR, INK, 2.4)
    c.paint(ear_tips, FUR_DEEP)
    head = c.ellipse(hx, hy, 38, 34, 8)
    c.inked(head, FUR, INK, 2.6)
    c.paint(head * (1.0 - c.shift(head, 0.0, 6.0)), FUR_HI, 0.55)
    # Snout poking out on the ball side, nose at the tip.
    snout = c.ellipse(hx + 38, hy + 16, 15, 9, 18)
    c.inked(snout, FUR_WHITE, INK, 2.2)
    c.paint(c.circle(hx + 52, hy + 21, 5.5), NOSE)
    # The edge of the purple mask, round the eye on the ball side.
    mask = head * c.ellipse(hx + 31, hy + 6, 10, 8, 20)
    c.paint(mask, FUR_DEEP)
    c.paint(c.circle(hx + 33, hy + 5, 3.2) * head, NOSE)
    fluff = c.ellipse(hx + 20, hy + 29, 12, 7, 20)
    c.inked(fluff, FUR_WHITE, INK, 1.8)
    # The visor: a white band round the crown, brim out front over the eyes.
    band = head * c.ellipse(hx + 2, hy - 14, 44, 6.5, 10)
    c.paint(band, VISOR)
    c.paint(band * c.halfplane_y(hy - 13), rgb(214, 214, 232))
    brim = c.ellipse(hx + 42, hy - 8, 21, 6.5, 14)
    c.inked(brim, VISOR, INK, 2.0)
    c.paint(brim * c.halfplane_y(hy - 7), VISOR_UNDER, 0.9)
    return c.to_bgra("lenny")


# ---------------------------------------------------------------------------
# small things
# ---------------------------------------------------------------------------


def make_ball(size: int = 48) -> np.ndarray:
    c = Canvas(size, size)
    r = size / 2 - 3
    ball = c.circle(size / 2, size / 2, r)
    c.inked(ball, rgb(255, 255, 255), rgb(60, 60, 80), 1.6)
    c.paint(ball * (1.0 - c.circle(size / 2 - r * 0.3, size / 2 - r * 0.3, r * 1.05)), rgb(206, 212, 228))
    dimples = c.zeros()
    rr = _rng("ball")
    for _ in range(14):
        a, d = float(rr.uniform(0, 2 * math.pi)), float(rr.uniform(0.15, 0.8)) * r
        dimples = np.maximum(dimples, c.circle(size / 2 + math.cos(a) * d, size / 2 + math.sin(a) * d, 1.6))
    c.paint(dimples * ball, rgb(220, 224, 236), 0.7)
    c.paint(c.circle(size / 2 - r * 0.35, size / 2 - r * 0.38, r * 0.22), rgb(255, 255, 255))
    return c.to_bgra("ball")


GAUGE_W, GAUGE_H = 66, 400
GAUGE_INNER = (14, 18, 52, 382)  # x0, y0, x1, y1 of the fill well
GAUGE_FULL_FRAC = 1.0 / 1.15  # 100% sits this far up the well; above it is the overswing zone


def make_gauge() -> tuple[np.ndarray, np.ndarray]:
    """The power gauge: its frame, and the fill the game reveals from the bottom."""
    c = Canvas(GAUGE_W, GAUGE_H)
    x0, y0, x1, y1 = GAUGE_INNER
    outer = c.rrect(4, 4, GAUGE_W - 4, GAUGE_H - 4, 26)
    c.inked(outer, rgb(32, 26, 44), rgb(255, 255, 255), 3.0)
    well = c.rrect(x0, y0, x1, y1, 16)
    c.paint(well, rgb(18, 14, 26))
    full_y = y1 - (y1 - y0) * GAUGE_FULL_FRAC
    red = well * (1.0 - c.halfplane_y(full_y))
    c.paint(red, rgb(150, 30, 40), 0.75)
    ticks = c.zeros()
    for f in (0.25, 0.5, 0.75):
        y = y1 - (y1 - y0) * GAUGE_FULL_FRAC * f
        ticks = np.maximum(ticks, c.rrect(x0 - 8, y - 1.5, x0 + 6, y + 1.5, 1))
        ticks = np.maximum(ticks, c.rrect(x1 - 6, y - 1.5, x1 + 8, y + 1.5, 1))
    c.paint(ticks, rgb(255, 255, 255), 0.8)
    line = c.rrect(x0 - 10, full_y - 2.5, x1 + 10, full_y + 2.5, 2)
    c.inked(line, rgb(255, 255, 255), INK, 1.2)
    frame = c.to_bgra("gauge")

    f = Canvas(GAUGE_W, GAUGE_H, ss=2)
    rows = f.H
    ys = f.ys
    t = np.clip((y1 - ys) / (y1 - y0), 0, 1) / GAUGE_FULL_FRAC  # 0 bottom .. 1 at 100% .. 1.15 top
    stops = [(0.0, rgb(70, 214, 255)), (0.55, rgb(120, 236, 140)), (0.85, rgb(255, 226, 70)),
             (1.0, rgb(255, 160, 40)), (1.001, rgb(255, 70, 60)), (1.2, rgb(220, 30, 40))]
    ts = np.array([s[0] for s in stops], np.float32)
    cols = np.array([s[1] for s in stops], np.float32)
    col = np.stack([np.interp(t, ts, cols[:, k]) for k in range(3)], axis=1)
    field = np.repeat(col[:, None, :], f.W, axis=1).astype(np.float32)
    inner = f.rrect(x0 + 3, y0 + 3, x1 - 3, y1 - 3, 13)
    f.paint(inner, field)
    f.paint(inner * (1.0 - f.shift(inner, -6.0, 0.0)), rgb(255, 255, 255), 0.35)  # glossy left edge
    assert rows == GAUGE_H * 2
    return frame, f.to_bgra("gauge_fill")


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def build() -> dict[str, np.ndarray]:
    gauge, fill = make_gauge()
    return {
        "sky.png": make_sky(),
        "sea.png": make_sea(),
        "haze.png": make_haze(),
        "palm.png": make_palm(),
        "tree_round.png": make_round_tree(),
        "lenny.png": make_lenny(),
        "ball.png": make_ball(),
        "gauge.png": gauge,
        "gauge_fill.png": fill,
    }


def manifest() -> dict:
    bgr = lambda c: [int(round(v)) for v in c]  # noqa: E731
    return {
        "focal": FOCAL,
        "pano_w": PANO_W,
        "pano_h": PANO_H,
        "sun": list(SUN_AT),
        "trees": {
            "palm": {"file": "palm.png", "foot": list(PALM_FOOT), "top": 8},
            "round": {"file": "tree_round.png", "foot": list(ROUND_FOOT), "top": 40},
        },
        "lenny": {
            "file": "lenny.png",
            "shoulder_back": list(SHOULDER_BACK),
            "shoulder_front": list(SHOULDER_FRONT),
            "arm": ARM_LEN,
            "club": CLUB_LEN,
            "address_dir": ADDRESS_DIR,
            "colours": {"fur": bgr(FUR), "sleeve": bgr(POLO), "glove": bgr(GLOVE), "ink": bgr(INK),
                        "shaft": bgr(SHAFT), "head": bgr(CLUB_HEAD), "face": bgr(CLUB_FACE)},
        },
        "gauge": {"inner": list(GAUGE_INNER), "full_frac": GAUGE_FULL_FRAC},
        "ground": {k: bgr(v) for k, v in GROUND.items()},
        "ball_visual_radius_m": 0.075,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Raccoon Golf art.")
    ap.add_argument("--out", type=Path, default=OUT_DIR, help="output directory")
    args = ap.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    total = 0.0
    for name, img in build().items():
        path = out / name
        if not cv2.imwrite(str(path), img):
            raise RuntimeError(f"failed to write {path}")
        kb = path.stat().st_size / 1024
        total += kb
        print(f"{name:18s} {img.shape[1]:>5}x{img.shape[0]:<4} {kb:8.1f} KB")
    (out / "manifest.json").write_text(json.dumps(manifest(), indent=2) + "\n")
    print(f"{'total':18s} {'':>10} {total:8.1f} KB -> {out}")


if __name__ == "__main__":
    main()
