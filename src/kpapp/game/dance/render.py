"""Drawing Raccoon Dance: the neon stage, the coach, the pictogram lane, feedback, HUD.

The camera is not the backdrop here. The player copies a coach, so the coach and
the cards must own the screen, and the camera moves to a picture-in-picture
window (top-right) where it does one job: show the player what the judge sees,
a skeleton tinted by the last rating.

Layout (docs/DANCE.md section 10): the coach stands centre-left in a spotlight
(x ~520, feet at y 660, ~520 px tall); the pictogram lane runs along the bottom
right onto a target ring at x 812; the PiP is top-right (384 x 216); the score
is top-left above a vertical star bar, with the difficulty badge beside it;
the bottom-left footer and the PAUSE button (x 1090-1258, y 646-698) stay clear.

Budget: < 4 ms a frame. What makes that possible:
* The stage is one pre-rendered image, copied (not blended) each frame.
* Every light that changes with the beat (beams, the floor pool, the neon trims,
  the coach's halo, the gold edge glow) is summed into one quarter-resolution
  buffer, then upscaled and added once. Light is soft, so 1/4 is invisible.
* The coach is a few dozen anti-aliased convex polygons drawn straight onto the
  frame, inside a fixed ROI; its head is a pre-scaled sprite rotated in 2-degree
  buckets, cached.
* Sprites are pre-split into premultiplied colour and 255 - alpha, so a blend is
  two saturating OpenCV calls; fades and scales are cached in buckets.

The renderer reads the game's state (docs/DANCE.md section 8) and never
mutates it. Its own animation state (score pulses, newly lit stars, when the
outro began) lives here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from ... import theme
from ...overlay import outlined_text
from .art import DanceArt
from .chart import idle_groove
from .pose import REST, forward, lerp_pose, normalize_pose, smoothstep

# --- layout (1280 x 720) ----------------------------------------------------
COACH_X, COACH_FEET = 520, 660
COACH_SCALE = 114.0  # px per shoulder width
COACH_ROI = (110, 56, 930, 704)  # the coach never draws outside this
POPUP_Y_MIN = 78

PAUSE_RECT = (1090, 646, 1258, 698)  # the shell's PAUSE button: nothing of ours enters it
FOOTER_Y = 700  # the shell's footer (bottom-left) starts here

# The lane: cards ~20% bigger than at first (76 -> 108 px), re-laid out so two
# cards a beat apart still keep a gap, everything above PAUSE_RECT, and the
# label above the footer.
LANE_RECT = (740, 513, 1262, 631)  # the track (118 px tall)
TARGET_X, LANE_Y = 812, 572
PX_PER_BEAT = 128.0
CARD_SMALL, CARD_BIG = 92, 132  # card widths (multiples of 4: card_width's steps) (cards are 208/192 as tall as wide)
GROW_BEATS = 1.0  # a card grows from small to big over its last beat to the ring (cards come every beat)
RING = 146  # target ring diameter
LANE_CLIP_BOTTOM = PAUSE_RECT[1] - 1  # cards (and their sparkles) never draw below this
LABEL_Y = 670  # baseline of the move's name under the ring
LABEL_SCALE = 0.72

PIP_RECT = (880, 16, 1264, 232)  # 384 x 216, 16:9
PIP_RADIUS = 16
STAR_BAR = (30, 128, 50, 552)  # the fill track; star markers sit on it
SCORE_ORG = (78, 76)
HUD_BADGE_ORG = (262, 58)  # the difficulty badge's left edge and centre line, right of the score

INTRO_C = (958, 352)  # the title card and GET READY, centred under the PiP
COUNT_Y = 366  # the 3-2-1-GO words: their glow clears the PiP (y 232) once popped in
POPUP_X = (260, 660)  # rating words: centre x clamp (the widest word, gold, stays left of the PiP)

RESULTS_RECT = (700, 34, 1250, 528)  # leaves the bottom 25% to the shell

DIFFICULTY_LABELS = {"easy": "EASY", "normal": "NORMAL", "hard": "HARD"}

SHIFT = 4  # sub-pixel bits for cv2 drawing
ONE = 1 << SHIFT
FONT = theme.TITLE_FONT
WHITE = (255, 255, 255)

RATING_KEYS = ("perfect", "super", "good", "ok", "x")
RATING_FALLBACK = {"perfect": (70, 210, 255), "super": (255, 220, 110), "good": (140, 235, 150),
                   "ok": (255, 150, 200), "x": (70, 70, 255)}
SECTION_ENERGY = (("chorus", 1.0), ("verse", 0.72), ("bridge", 0.55), ("intro", 0.55), ("outro", 0.45))
BOW_POSE = normalize_pose({"la": [28, 18], "ra": [-28, -70], "head": 0, "lean": 0, "squat": 0.12})

POPUP_LIFE = 0.85
# An older popup clears this fast once a newer one appears. 0.12 s left both
# words readable for ~7 frames ("PERFECT" over "PERFECT", QA contact sheet);
# at 0.04 s (squared fade: 34% on the first frame, 3% on the second, then
# gone) the old word has left before the new one has popped to full size.
SUPERSEDE_FADE = 0.04


# --- small helpers ----------------------------------------------------------

def rating_key(rating) -> str | None:
    """'perfect' | 'super' | 'good' | 'ok' | 'x' from a string, enum or None."""
    if rating is None:
        return None
    r = str(getattr(rating, "value", getattr(rating, "name", rating))).lower()
    r = r.split(".")[-1]
    return r if r in RATING_KEYS else None


def card_width(d: float) -> int:
    """A lane card's width `d` beats before its beat: small until GROW_BEATS
    out, full size from the ring on (and while it slides through), in steps of
    4 px so the card cache stays small."""
    u = min(1.0, max(0.0, 1.0 - d / GROW_BEATS))
    return int(round((CARD_SMALL + (CARD_BIG - CARD_SMALL) * smoothstep(u)) / 4.0) * 4)


def _kick(beat: float, sharp: float = 5.0) -> float:
    """1 on each beat, decaying through it: the pulse every light shares."""
    return math.exp(-(beat - math.floor(beat)) * sharp)


def _ease_out(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return 1.0 - (1.0 - t) ** 3


def _pop(age: float, dur: float = 0.26, over: float = 1.28, start: float = 0.45) -> float:
    """Scale-in with an overshoot: start -> over -> 1.0."""
    if age < dur * 0.45:
        return start + (over - start) * (age / (dur * 0.45))
    if age < dur:
        return over - (over - 1.0) * ((age - dur * 0.45) / (dur * 0.55))
    return 1.0


def _resize_bgra(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Premultiplied resize, so soft edges keep their colour."""
    w, h = max(1, int(round(w))), max(1, int(round(h)))
    f = img.astype(np.float32)
    f[..., :3] *= f[..., 3:4] / 255.0
    interp = cv2.INTER_AREA if w < img.shape[1] else cv2.INTER_LINEAR
    f = cv2.resize(f, (w, h), interpolation=interp)
    a = np.maximum(f[..., 3:4], 1e-3)
    f[..., :3] = np.where(f[..., 3:4] > 0.01, f[..., :3] * 255.0 / a, 0)
    return np.clip(f + 0.5, 0, 255).astype(np.uint8)


def _tinted(white_bgra: np.ndarray, colour, gain: float = 1.0) -> np.ndarray:
    """Premultiplied BGR light from a white BGRA sprite, for additive blending."""
    a = white_bgra[..., 3:4].astype(np.float32) * (gain / 255.0)
    return np.clip(np.asarray(colour, np.float32)[None, None, :] * a + 0.5, 0, 255).astype(np.uint8)


def _clip_rect(x, y, w, h, bounds):
    bx0, by0, bx1, by1 = bounds
    x0, y0, x1, y1 = max(bx0, x), max(by0, y), min(bx1, x + w), min(by1, y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


class Sprite:
    """A BGRA sprite split for fast blending: premultiplied colour and 255 - alpha.

    `draw` composites (dst = dst * inv / 255 + pre), `add` adds light; both are
    two saturating OpenCV calls, clipped to the frame (or `clip`).
    """

    __slots__ = ("pre", "inv", "w", "h")

    def __init__(self, bgra: np.ndarray, tint=None) -> None:
        a = bgra[..., 3:4].astype(np.float32) / 255.0
        col = (np.asarray(tint, np.float32)[None, None, :] if tint is not None
               else bgra[..., :3].astype(np.float32))
        self.pre = np.ascontiguousarray(np.clip(col * a + 0.5, 0, 255).astype(np.uint8))
        self.inv = np.ascontiguousarray(np.repeat(np.clip(255.0 - a * 255.0 + 0.5, 0, 255)
                                                  .astype(np.uint8), 3, axis=2))
        self.h, self.w = bgra.shape[:2]

    def _views(self, dst, x, y, clip):
        h, w = dst.shape[:2]
        r = _clip_rect(int(x), int(y), self.w, self.h, clip or (0, 0, w, h))
        if r is None:
            return None
        x0, y0, x1, y1 = r
        sx, sy = x0 - int(x), y0 - int(y)
        return (dst[y0:y1, x0:x1], self.pre[sy:sy + y1 - y0, sx:sx + x1 - x0],
                self.inv[sy:sy + y1 - y0, sx:sx + x1 - x0])

    def draw(self, dst, x, y, fade: float = 1.0, clip=None) -> None:
        if fade <= 0.004:
            return
        v = self._views(dst, x, y, clip)
        if v is None:
            return
        roi, pre, inv = v
        if fade < 0.996:
            pre = cv2.convertScaleAbs(pre, alpha=fade)
            inv = cv2.convertScaleAbs(inv, alpha=fade, beta=255.0 * (1.0 - fade))
        cv2.multiply(roi, inv, dst=roi, scale=1.0 / 255.0)
        cv2.add(roi, pre, dst=roi)

    def draw_c(self, dst, cx, cy, fade: float = 1.0, clip=None) -> None:
        self.draw(dst, int(round(cx - self.w / 2)), int(round(cy - self.h / 2)), fade, clip)

    def add_c(self, dst, cx, cy, gain: float = 1.0, clip=None) -> None:
        if gain <= 0.004:
            return
        v = self._views(dst, int(round(cx - self.w / 2)), int(round(cy - self.h / 2)), clip)
        if v is None:
            return
        roi, pre, _ = v
        if gain >= 0.996:
            cv2.add(roi, pre, dst=roi)
        else:
            cv2.addWeighted(roi, 1.0, pre, gain, 0.0, dst=roi)


class ScaledSprites:
    """One BGRA image at many scales (buckets of `step`), built on first use."""

    def __init__(self, img: np.ndarray, step: float = 0.05, tint=None) -> None:
        self.img, self.step, self.tint = img, step, tint
        self._cache: dict[int, Sprite] = {}

    def get(self, scale: float) -> Sprite:
        k = max(1, int(round(scale / self.step)))
        spr = self._cache.get(k)
        if spr is None:
            s = k * self.step
            h, w = self.img.shape[:2]
            spr = self._cache[k] = Sprite(_resize_bgra(self.img, w * s, h * s), self.tint)
        return spr


def text_c(img, s: str, cx: float, y: float, scale: float, colour, thick: int = 2,
           font=FONT, outline: int | None = None) -> int:
    """Centred, outlined text (theme style; overlay.outlined_text keeps glyphs registered)."""
    (tw, _), _ = cv2.getTextSize(s, font, scale, thick)
    outlined_text(img, s, (int(round(cx - tw / 2)), int(round(y))), font, scale, colour, thick,
                  width=outline if outline is not None else (2 if scale >= 0.9 else 1))
    return tw


def _faded(img, rect, alpha: float, draw) -> None:
    """Run draw(patch, ox, oy) on a copy of `rect` and blend it back at `alpha`."""
    if alpha <= 0.004:
        return
    h, w = img.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in rect)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    if alpha >= 0.996:
        draw(roi, x0, y0)
        return
    patch = roi.copy()
    draw(patch, x0, y0)
    cv2.addWeighted(patch, alpha, roi, 1.0 - alpha, 0, dst=roi)


def make_glow_text(text: str, scale: float, colour, glow=None, thick: int = 3) -> np.ndarray:
    """A runtime word sprite in the generated words' style: fill, dark keyline, soft glow.

    For text only known at runtime (song titles). Built once and cached by the caller.
    """
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
    key, gw = max(3, int(th * 0.12)), max(6, int(th * 0.35))
    pad = gw * 2 + key
    h, w = th + base + 2 * pad, tw + 2 * pad
    m = np.zeros((h, w), np.uint8)
    cv2.putText(m, text, (pad, pad + th), FONT, scale, 255, thick, cv2.LINE_AA)
    mf = m.astype(np.float32) / 255.0
    dist = cv2.distanceTransform((m < 128).astype(np.uint8), cv2.DIST_L2, 3)
    keyline = np.clip(key - dist + 0.5, 0, 1)
    halo = cv2.GaussianBlur(np.clip(gw * 0.5 - dist + 0.5, 0, 1), (0, 0), gw * 0.45) * 0.8
    col = np.asarray(colour, np.float32)
    ys = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    top = col + (255 - col) * 0.6
    fill = top * (1 - ys) + col * ys
    out = np.empty((h, w, 4), np.float32)
    out[..., :3] = np.asarray(glow if glow is not None else colour, np.float32)
    ink = np.asarray((46, 24, 34), np.float32)
    k = keyline[..., None]
    out[..., :3] = out[..., :3] * (1 - k) + ink * k
    out[..., :3] = out[..., :3] * (1 - mf[..., None]) + fill * mf[..., None]
    out[..., 3] = np.maximum(np.maximum(mf, keyline), halo) * 255.0
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def make_badge(text: str, colour, scale: float = 0.62, thick: int = 2) -> np.ndarray:
    """A difficulty badge (BGRA): the label in `colour` on a dark, slightly
    translucent pill with a border of the same colour."""
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
    px, py = int(round(th * 0.85)), int(round(th * 0.5))
    w, h = tw + 2 * px, th + base + 2 * py
    ss = 2  # drawn at 2x and reduced, for clean edges

    def pill(width, height, inset):
        m = np.zeros((height, width), np.uint8)
        r = height // 2 - inset
        cv2.rectangle(m, (height // 2, inset), (width - 1 - height // 2, height - 1 - inset), 255, -1)
        for cx in (height // 2, width - 1 - height // 2):
            cv2.circle(m, (cx, height // 2), r, 255, -1, cv2.LINE_AA)
        return m.astype(np.float32) / 255.0

    W2, H2 = w * ss, h * ss
    outer = pill(W2, H2, 0)
    inner = pill(W2, H2, max(2, int(round(1.6 * ss))))
    border = np.clip(outer - inner, 0, 1)[..., None]
    out = np.empty((H2, W2, 4), np.float32)
    out[..., :3] = np.asarray((40, 18, 30), np.float32) * (1 - border) + np.asarray(colour, np.float32) * border
    out[..., 3] = 255.0 * (0.88 * inner + border[..., 0])
    bgr = np.ascontiguousarray(np.clip(out[..., :3], 0, 255).astype(np.uint8))
    ink = np.zeros((H2, W2), np.uint8)
    org = (px * ss, (py + th) * ss)
    cv2.putText(ink, text, org, FONT, scale * ss, 255, thick * ss, cv2.LINE_AA)
    k = ink.astype(np.float32)[..., None] / 255.0
    bgr = (bgr * (1 - k) + np.asarray(colour, np.float32) * k).astype(np.uint8)
    out[..., :3] = bgr
    return _resize_bgra(np.clip(out + 0.5, 0, 255).astype(np.uint8), w, h)


# --- the coach ------------------------------------------------------------------

# Radii in shoulder widths: [start, end] of each segment. Equal radii at shared
# joints (elbow, knee) so a limb reads as one smooth tapered shape.
R_UPPER = (0.135, 0.112)
R_FORE = (0.112, 0.092)
R_THIGH = (0.19, 0.145)
R_SHIN = (0.145, 0.112)
HEAD_FACE_SW = 1.2  # the face is drawn this many shoulder widths wide: a mascot, not a realistic head
FOOT_BELOW_ANKLE = 0.245  # shoe sole (with keyline and rim) below the ankle, in SW
_REST_ANKLE_Y = forward(REST)["l_ankle"][1]
_ARC_K = 10
_ARC_T = np.linspace(0.0, 1.0, _ARC_K, dtype=np.float64)


def _capsules(segs: np.ndarray, grow: float) -> list[np.ndarray]:
    """Tapered capsules as convex polygons (fixed point, SHIFT bits).

    segs: (N, 6) rows of x0, y0, x1, y1, r0, r1 in px. Each capsule is the convex
    hull of its two end circles: the front arc of the end circle and the back
    arc of the start circle, joined by the two outer tangents.
    """
    x0, y0, x1, y1 = segs[:, 0], segs[:, 1], segs[:, 2], segs[:, 3]
    r0, r1 = segs[:, 4] + grow, segs[:, 5] + grow
    dx, dy = x1 - x0, y1 - y0
    length = np.maximum(np.hypot(dx, dy), 1e-6)
    theta = np.arctan2(dy, dx)
    alpha = np.arccos(np.clip((r0 - r1) / length, -0.999, 0.999))  # tangent normal angle
    a1 = theta[:, None] + (-alpha[:, None] + 2 * alpha[:, None] * _ARC_T[None, :])
    a0 = theta[:, None] + (alpha[:, None] + (2 * np.pi - 2 * alpha[:, None]) * _ARC_T[None, :])
    px = np.concatenate([x1[:, None] + np.cos(a1) * r1[:, None], x0[:, None] + np.cos(a0) * r0[:, None]], 1)
    py = np.concatenate([y1[:, None] + np.sin(a1) * r1[:, None], y0[:, None] + np.sin(a0) * r0[:, None]], 1)
    pts = np.round(np.stack([px, py], axis=2) * ONE).astype(np.int32)
    return list(pts)


def _hull(points: np.ndarray, radii: np.ndarray, grow: float) -> np.ndarray:
    """Convex hull of several circles (fixed point)."""
    t = np.linspace(0, 2 * np.pi, 18, endpoint=False)
    r = radii[:, None] + grow
    px = points[:, 0:1] + np.cos(t)[None, :] * r
    py = points[:, 1:2] + np.sin(t)[None, :] * r
    pts = np.round(np.stack([px.ravel(), py.ravel()], 1) * ONE).astype(np.int32)
    return cv2.convexHull(pts)


def _fill(img, polys, colour) -> None:
    for p in polys:
        cv2.fillConvexPoly(img, p, colour, cv2.LINE_AA, SHIFT)


def _fxp(p) -> tuple[int, int]:
    return int(round(p[0] * ONE)), int(round(p[1] * ONE))


def _line(img, a, b, colour, width: float) -> None:
    cv2.line(img, _fxp(a), _fxp(b), colour, max(1, int(round(width))), cv2.LINE_AA, SHIFT)


def _ellipse(img, c, axes, angle: float, colour, thick: int = -1, a0: float = 0, a1: float = 360) -> None:
    cv2.ellipse(img, _fxp(c), (max(1, int(round(axes[0] * ONE))), max(1, int(round(axes[1] * ONE)))),
                angle, a0, a1, colour, thick, cv2.LINE_AA, SHIFT)


@dataclass
class CoachGeometry:
    """Everything needed to draw the coach this frame, in frame pixels."""

    j: dict[str, np.ndarray]
    silhouette: list[np.ndarray] = field(default_factory=list)  # fixed-point polys (grown by ink + rim)
    head_c: tuple[float, float] = (0.0, 0.0)
    head_angle: float = 0.0  # degrees, positive tips the head screen-right
    head_top: float = 0.0  # y of the ear tips
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)


class CoachPainter:
    """The coach: the arcade's purple raccoon as a bold flat-colour dancer.

    `layout(pose)` turns a pose (docs/DANCE.md section 3) into pixel geometry;
    `paint(img, geo)` draws it. The renderer uses the geometry first for the
    coach's soft halo in the light buffer, then paints the body on top.
    Parts are drawn back to front, each with a dark keyline, over one neon rim
    around the whole silhouette: tail, legs and pants (one shape, no seam),
    jacket, neck, head, then arms in front.
    """

    def __init__(self, art: DanceArt, outfit: str | None = None, scale: float = COACH_SCALE,
                 origin: tuple[float, float] | None = None) -> None:
        self.S = float(scale)
        o = art.outfit(outfit)
        self.c = o
        self.ink_w = max(1.2, 0.026 * self.S)
        self.rim_w = max(1.2, 0.032 * self.S)
        if origin is None:
            origin = self.origin_for(COACH_X, COACH_FEET)
        self.origin = origin
        # The head: rim (tinted) under the raccoon, scaled so the face is ~1.05 SW wide.
        hm = art.manifest.get("head", {})
        self._face = tuple(hm.get("centre", (0.5, 0.565)))
        self._ear_top = 0.105
        fw = float(hm.get("face_width", 0.67))
        side = max(8, int(round(HEAD_FACE_SW * self.S / fw)))
        head = _resize_bgra(art.head, side, side).astype(np.float32)
        rim = _resize_bgra(art.head_rim, side, side).astype(np.float32)
        ra = rim[..., 3:4] / 255.0
        ha = head[..., 3:4] / 255.0
        comb = np.empty_like(head)
        rim_col = np.asarray(o.get("rim", (255, 255, 255)), np.float32)
        comb[..., :3] = head[..., :3] * ha + rim_col * ra * (1 - ha)
        comb[..., 3:4] = ha + ra * (1 - ha)
        comb[..., :3] /= np.maximum(comb[..., 3:4], 1e-3)
        comb[..., 3:4] *= 255.0
        comb = np.clip(comb + 0.5, 0, 255).astype(np.uint8)
        # Pad so the face centre is the canvas centre: rotation turns about it.
        fx, fy = self._face[0] * side, self._face[1] * side
        reach = math.ceil(max(math.hypot(fx - cx, fy - cy) for cx in (0, side) for cy in (0, side))) + 2
        d = 2 * reach
        pad = np.zeros((d, d, 4), np.uint8)
        ox, oy = int(round(reach - fx)), int(round(reach - fy))
        pad[oy:oy + side, ox:ox + side] = comb
        self._head_pad = pad
        self._head_side = side
        self._head_cache: dict[int, Sprite] = {}

    def origin_for(self, x: float, feet_y: float) -> tuple[float, float]:
        """The rest hip centre that puts the coach's shoe soles at feet_y."""
        return (x, feet_y - (FOOT_BELOW_ANKLE + _REST_ANKLE_Y) * self.S)

    # --- geometry ---

    def layout(self, pose: dict | None, bow: float = 0.0, beat: float = 0.0,
               origin: tuple[float, float] | None = None) -> CoachGeometry:
        """Pixel geometry for `pose`. `origin` (the rest hip centre, px) defaults to the stage spot."""
        p = normalize_pose(pose or REST)
        if bow > 0.0:
            p = lerp_pose(p, BOW_POSE, min(1.0, bow))
        jj = forward(p)
        S, (ox, oy) = self.S, origin or self.origin
        j = {k: np.array((ox + v[0] * S, oy + v[1] * S)) for k, v in jj.items()}
        if bow > 0.0:
            # A front-view bow: the upper body foreshortens towards the camera.
            hip_y = j["hip"][1]
            k = 1.0 - 0.24 * min(1.0, bow)
            for name in ("neck", "head", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
                         "l_wrist", "r_wrist", "l_hand", "r_hand"):
                j[name] = np.array((j[name][0], hip_y + (j[name][1] - hip_y) * k))
        g = CoachGeometry(j=j)
        up = j["neck"] - j["hip"]
        up /= max(1e-6, float(np.hypot(*up)))
        g.head_angle = p["lean"] + p["head"]
        ha = math.radians(g.head_angle)
        hdir = np.array((math.sin(ha), -math.cos(ha)))
        g.head_c = tuple(j["neck"] + hdir * (0.50 - 0.22 * min(1.0, bow)) * S)
        g.head_top = g.head_c[1] - (self._face[1] - self._ear_top) * self._head_side
        self._beat = beat
        self._up = up
        g.silhouette = self._silhouette(g, self.ink_w + self.rim_w)
        xs = [v[0] for v in j.values()] + [g.head_c[0]]
        ys = [v[1] for v in j.values()] + [g.head_top]
        m = 0.5 * S
        g.bbox = (int(min(xs) - m), int(min(ys) - 0.1 * S), int(max(xs) + m), int(max(ys) + 0.4 * S))
        return g

    def _segs(self, j):
        S = self.S
        arms = np.array([[*j[f"{s}_shoulder"], *j[f"{s}_elbow"], R_UPPER[0] * S, R_UPPER[1] * S] for s in "lr"]
                        + [[*j[f"{s}_elbow"], *j[f"{s}_wrist"], R_FORE[0] * S, R_FORE[1] * S] for s in "lr"])
        legs = np.array([[*j[f"{s}_hip"], *j[f"{s}_knee"], R_THIGH[0] * S, R_THIGH[1] * S] for s in "lr"]
                        + [[*j[f"{s}_knee"], *j[f"{s}_ankle"], R_SHIN[0] * S, R_SHIN[1] * S] for s in "lr"])
        return arms, legs

    def _torso(self, j, grow: float):
        S = self.S
        up = self._up
        right = np.array((-up[1], up[0]))
        neck, hip = j["neck"], j["hip"]
        waist = neck + (hip - neck) * 0.66
        top = _hull(np.array([j["l_shoulder"], j["r_shoulder"], waist - right * 0.30 * S, waist + right * 0.30 * S]),
                    np.array([0.2, 0.2, 0.13, 0.13]) * S, grow)
        pants = _hull(np.array([waist - right * 0.29 * S, waist + right * 0.29 * S, j["l_hip"], j["r_hip"]]),
                      np.array([0.12, 0.12, 0.19, 0.19]) * S, grow)
        return top, pants, waist, right

    def _tail(self, j):
        """A ringed raccoon tail from behind the hips, swaying on the beat."""
        S = self.S
        sway = math.sin(self._beat * math.pi) * 14.0
        start = j["hip"] + np.array((-0.22 * S, 0.12 * S))
        pts, radii = [start], [0.12 * S]
        ang = 150.0 + sway * 0.5  # screen degrees, y down: left and a little down
        p = start
        for k, r in enumerate((0.16, 0.19, 0.19, 0.16, 0.11)):
            ang += 14.0 + sway * 0.25
            a = math.radians(ang)
            p = p + np.array((math.cos(a), math.sin(a))) * 0.2 * S
            pts.append(p)
            radii.append(r * S)
        return pts, radii

    def _shoe(self, j, s):
        side = -1.0 if s == "l" else 1.0
        S = self.S
        c = j[f"{s}_ankle"] + np.array((side * 0.07 * S, 0.075 * S))
        return c, (0.205 * S, 0.112 * S)

    def _glove(self, j, s):
        d = j[f"{s}_hand"] - j[f"{s}_wrist"]
        n = max(1e-6, float(np.hypot(*d)))
        d /= n
        c = j[f"{s}_wrist"] + d * 0.1 * self.S
        return c, (0.15 * self.S, 0.125 * self.S), math.degrees(math.atan2(d[1], d[0]))

    def _silhouette(self, g: CoachGeometry, grow: float) -> list[np.ndarray]:
        j = g.j
        arms, legs = self._segs(j)
        polys = _capsules(arms, grow) + _capsules(legs, grow)
        top, pants, _, _ = self._torso(j, grow)
        polys += [top.reshape(-1, 2), pants.reshape(-1, 2)]
        tail_pts, tail_r = self._tail(j)
        segs = np.array([[*a, *b, ra, rb] for a, b, ra, rb in zip(tail_pts, tail_pts[1:], tail_r, tail_r[1:])])
        polys += _capsules(segs, grow)
        neck = np.array([[*j["neck"], *(j["neck"] + self._up * 0.25 * self.S), 0.14 * self.S, 0.14 * self.S]])
        polys += _capsules(neck, grow)
        for s in "lr":
            c, ax = self._shoe(j, s)
            polys.append(_ellipse_poly(c, (ax[0] + grow, ax[1] + grow), 0.0))
            c, ax, ang = self._glove(j, s)
            polys.append(_ellipse_poly(c, (ax[0] + grow, ax[1] + grow), ang))
        return polys

    # --- drawing ---

    def head_sprite(self, angle: float) -> Sprite:
        b = int(round(angle / 2.0))
        spr = self._head_cache.get(b)
        if spr is None:
            d = self._head_pad.shape[0]
            if b:
                m = cv2.getRotationMatrix2D((d / 2, d / 2), -b * 2.0, 1.0)
                img = _rotate_bgra(self._head_pad, m)
            else:
                img = self._head_pad
            spr = self._head_cache[b] = Sprite(img)
        return spr

    def paint(self, img, g: CoachGeometry, clip=None) -> None:
        """Draw the coach. With `clip` (x0, y0, x1, y1) it draws into that view only."""
        if clip is not None:
            x0, y0, x1, y1 = clip
            view = img[y0:y1, x0:x1]
            off = np.array((x0, y0), np.float64)
            self._paint(view, g, off)
        else:
            self._paint(img, g, np.zeros(2))

    def _paint(self, img, g: CoachGeometry, off: np.ndarray) -> None:
        c, S = self.c, self.S
        ink, iw = c["ink"], self.ink_w
        j = {k: v - off for k, v in g.j.items()}
        o16 = np.round(off * ONE).astype(np.int32)

        def shifted(polys):
            return [p.reshape(-1, 2) - o16 for p in polys]

        # 1. the neon rim around the whole silhouette
        _fill(img, shifted(g.silhouette), c["rim"])

        # 2. tail
        tail_pts, tail_r = self._tail(g.j)
        tail_pts = [p - off for p in tail_pts]
        segs = np.array([[*a, *b, ra, rb] for a, b, ra, rb in zip(tail_pts, tail_pts[1:], tail_r, tail_r[1:])])
        _fill(img, _capsules(segs, iw), ink)
        _fill(img, _capsules(segs, 0.0), c["fur"])
        for k in (2, 4):  # dark rings across the tail
            a, b = tail_pts[k - 1], tail_pts[k + 1] if k + 1 < len(tail_pts) else tail_pts[k]
            d = b - a
            n = np.array((-d[1], d[0])) / max(1e-6, float(np.hypot(*d)))
            r = tail_r[k] - iw * 0.3
            _line(img, tail_pts[k] - n * r * 0.92, tail_pts[k] + n * r * 0.92, c["fur_dark"], 0.11 * S)
        tip = tail_pts[-1]
        cv2.circle(img, _fxp(tip), int(round(tail_r[-1] * ONE)), c["fur_dark"], -1, cv2.LINE_AA, SHIFT)

        # 3. legs and pants: one shape (ink for all, then fill for all: no seam)
        arms, legs = self._segs(j)
        top, pants, waist, right = self._torso(j, 0.0)
        top_i, pants_i, _, _ = self._torso(j, iw)
        _fill(img, _capsules(legs, iw), ink)
        _fill(img, [pants_i.reshape(-1, 2)], ink)
        _fill(img, _capsules(legs, 0.0), c["pants"])
        _fill(img, [pants.reshape(-1, 2)], c["pants"])
        for s in "lr":  # a cyan stripe down the outside of each leg
            side = -1.0 if s == "l" else 1.0
            pts = [j[f"{s}_hip"], j[f"{s}_knee"], j[f"{s}_ankle"]]
            off_pts = []
            for i, p in enumerate(pts):
                a, b = pts[max(0, i - 1)], pts[min(2, i + 1)]
                d = b - a
                n = np.array((-d[1], d[0])) / max(1e-6, float(np.hypot(*d)))
                if n[0] * side < 0:
                    n = -n
                r = (R_THIGH[0], R_THIGH[1], R_SHIN[1])[i] * S * 0.58
                off_pts.append(p + n * r)
            cv2.polylines(img, [np.round(np.array(off_pts) * ONE).astype(np.int32)], False, c["accent"],
                          max(1, int(round(0.045 * S))), cv2.LINE_AA, SHIFT)
        for s in "lr":  # sneakers
            sc, ax = self._shoe(j, s)
            _ellipse(img, sc, (ax[0] + iw, ax[1] + iw), 0, ink)
            _ellipse(img, sc, ax, 0, c["shoe"])
            _ellipse(img, sc, (ax[0] - 0.02 * S, ax[1] - 0.025 * S), 0, c["accent"],
                     max(1, int(round(0.05 * S))), 25, 155)

        # 4. jacket: magenta, a lavender V of chest fur, a cyan belt
        _fill(img, [top_i.reshape(-1, 2)], ink)
        _fill(img, [top.reshape(-1, 2)], c["top"])
        up = self._up
        neck = j["neck"]
        v = np.array([neck - right * 0.2 * S - up * 0.02 * S, neck + right * 0.2 * S - up * 0.02 * S,
                      neck - up * -0.52 * S])
        cv2.fillConvexPoly(img, np.round(v * ONE).astype(np.int32), c["fur"], cv2.LINE_AA, SHIFT)
        cv2.polylines(img, [np.round(v[[0, 2, 1]] * ONE).astype(np.int32)], False, c["accent"],
                      max(1, int(round(0.04 * S))), cv2.LINE_AA, SHIFT)
        wl, wr = waist - right * 0.36 * S, waist + right * 0.36 * S
        _line(img, wl, wr, ink, 0.11 * S + 2 * iw * 0.7)
        _line(img, wl, wr, c["accent"], 0.11 * S)

        # 5. neck, 6. head
        neck_seg = np.array([[*neck, *(neck + up * 0.25 * S), 0.14 * S, 0.14 * S]])
        _fill(img, _capsules(neck_seg, iw), ink)
        _fill(img, _capsules(neck_seg, 0.0), c["fur"])
        spr = self.head_sprite(g.head_angle)
        spr.draw_c(img, g.head_c[0] - off[0], g.head_c[1] - off[1])

        # 7. arms in front: fur forearm, magenta sleeve with a cyan cuff, cyan wristband, white glove
        for idx, s in enumerate("lr"):
            upper, fore = arms[idx:idx + 1], arms[idx + 2:idx + 3]
            gc, gax, gang = self._glove(j, s)
            _fill(img, _capsules(np.vstack([upper, fore]), iw), ink)
            _ellipse(img, gc, (gax[0] + iw, gax[1] + iw), gang, ink)
            _fill(img, _capsules(fore, 0.0), c["fur"])
            _fill(img, _capsules(upper, 0.0), c["top"])
            sh, el, wr = j[f"{s}_shoulder"], j[f"{s}_elbow"], j[f"{s}_wrist"]
            band = np.array([[*(el + (wr - el) * 0.78), *wr, R_FORE[1] * S * 1.08, R_FORE[1] * S * 1.08]])
            _fill(img, _capsules(band, 0.0), c["accent"])
            _ellipse(img, gc, gax, gang, c["glove"])
            # A thumb on the inner side, so the hand reads as a hand, not a ball.
            d = j[f"{s}_hand"] - wr
            d /= max(1e-6, float(np.hypot(*d)))
            n = np.array((-d[1], d[0]))
            side = -1.0 if s == "l" else 1.0
            if n[0] * side > 0:
                n = -n
            tc = gc + n * 0.12 * S - d * 0.02 * S
            cv2.circle(img, _fxp(tc), int(round((0.055 * S + iw) * ONE)), ink, -1, cv2.LINE_AA, SHIFT)
            cv2.circle(img, _fxp(tc), int(round(0.055 * S * ONE)), c["glove"], -1, cv2.LINE_AA, SHIFT)
            _ellipse(img, gc, (gax[0] * 0.92, gax[1] * 0.86), gang, c["glove"])


def _ellipse_poly(c, axes, angle_deg: float) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    a = math.radians(angle_deg)
    ex, ey = np.cos(t) * axes[0], np.sin(t) * axes[1]
    x = c[0] + ex * math.cos(a) - ey * math.sin(a)
    y = c[1] + ex * math.sin(a) + ey * math.cos(a)
    return np.round(np.stack([x, y], 1) * ONE).astype(np.int32)


def _rotate_bgra(img: np.ndarray, m: np.ndarray) -> np.ndarray:
    f = img.astype(np.float32)
    f[..., :3] *= f[..., 3:4] / 255.0
    d = img.shape[0]
    f = cv2.warpAffine(f, m, (img.shape[1], d), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    a = np.maximum(f[..., 3:4], 1e-3)
    f[..., :3] = np.where(f[..., 3:4] > 0.01, f[..., :3] * 255.0 / a, 0)
    return np.clip(f + 0.5, 0, 255).astype(np.uint8)


# --- pictograms -------------------------------------------------------------------

def _pose_key(pose) -> tuple:
    p = normalize_pose(pose)
    return tuple((k, tuple(np.round(v, 1)) if isinstance(v, list) else round(v, 1)) for k, v in sorted(p.items()))


def draw_pictogram_figure(img, pose, rect, gold: bool = False, alpha_plane=None) -> None:
    """A bold pictogram dancer for `pose`, fitted into rect (x0, y0, x1, y1).

    Body in light lavender, arms in white (gold-white on gold cards) over it with
    their own keyline, because the arms are what the judge mostly reads.
    """
    j = forward(pose)
    x0, y0, x1, y1 = rect
    head_r = 0.36
    pts = np.array(list(j.values()))
    lo = pts.min(0) - (0.25, 0.2)
    hi = pts.max(0) + (0.25, 0.25)
    lo[1] = min(lo[1], j["head"][1] - head_r - 0.12)
    bw, bh = hi - lo
    s = min((x1 - x0) / bw, (y1 - y0) / bh, (y1 - y0) / 4.3)
    cx, cy = (x0 + x1) / 2 - (lo[0] + bw / 2) * s, (y0 + y1) / 2 - (lo[1] + bh / 2) * s

    def P(name):
        return (cx + j[name][0] * s, cy + j[name][1] * s)

    ink = (46, 24, 34)
    body = (222, 205, 236)
    arm = (170, 238, 255) if gold else (255, 255, 255)
    lw, ow = 0.3 * s, 0.3 * s + 0.16 * s

    def limb(img_, names, width, colour):
        cv2.polylines(img_, [np.round(np.array([P(n) for n in names]) * ONE).astype(np.int32)], False,
                      colour, max(1, int(round(width))), cv2.LINE_AA, SHIFT)

    def torso(img_, grow, colour):
        quad = np.array([P("l_shoulder"), P("r_shoulder"), P("r_hip"), P("l_hip")])
        h = _hull(quad, np.full(4, 0.16 * s), grow)
        cv2.fillConvexPoly(img_, h, colour, cv2.LINE_AA, SHIFT)

    targets = [img] + ([alpha_plane] if alpha_plane is not None else [])
    for k, t in enumerate(targets):
        ci = 255 if k else None
        for s_ in "lr":
            limb(t, [f"{s_}_hip", f"{s_}_knee", f"{s_}_ankle"], ow, ci or ink)
        torso(t, 0.08 * s, ci or ink)
        cv2.circle(t, _fxp(P("head")), int(round((head_r * s + 0.08 * s) * ONE)), ci or ink, -1, cv2.LINE_AA, SHIFT)
        for s_ in "lr":
            limb(t, [f"{s_}_shoulder", f"{s_}_elbow", f"{s_}_hand"], ow, ci or ink)
        if k:
            continue
        for s_ in "lr":
            limb(img, [f"{s_}_hip", f"{s_}_knee", f"{s_}_ankle"], lw, body)
        torso(img, 0.0, body)
        cv2.circle(img, _fxp(P("head")), int(round(head_r * s * ONE)), body, -1, cv2.LINE_AA, SHIFT)
        for s_ in "lr":  # ears: it's the raccoon
            side = -1 if s_ == "l" else 1
            hx, hy = P("head")
            cv2.circle(img, _fxp((hx + side * 0.24 * s, hy - 0.3 * s)), int(round(0.12 * s * ONE)), body,
                       -1, cv2.LINE_AA, SHIFT)
        for s_ in "lr":
            limb(img, [f"{s_}_shoulder", f"{s_}_elbow", f"{s_}_hand"], ow, ink)
            limb(img, [f"{s_}_shoulder", f"{s_}_elbow", f"{s_}_hand"], lw, arm)


class PictogramCards:
    """Pictogram cards (frame + figure), built once per (pose, gold, width) and cached."""

    def __init__(self, art: DanceArt) -> None:
        self.frames = {False: art.card, True: art.card_gold}
        self._cache: dict[tuple, Sprite] = {}

    def get(self, pose, gold: bool, width: int) -> Sprite:
        width = int(width)
        key = (_pose_key(pose), bool(gold), width)
        spr = self._cache.get(key)
        if spr is None:
            frame = self.frames[bool(gold)]
            fh, fw = frame.shape[:2]
            ss = 2
            w2, h2 = width * ss, int(round(width * fh / fw)) * ss
            card = _resize_bgra(frame, w2, h2)
            bgr = np.ascontiguousarray(card[..., :3])
            alpha = np.ascontiguousarray(card[..., 3])
            inset = 0.135 * w2
            draw_pictogram_figure(bgr, pose, (inset, inset * 0.95, w2 - inset, h2 - inset * 0.95), gold, alpha)
            spr = self._cache[key] = Sprite(_resize_bgra(np.dstack([bgr, alpha]), w2 / ss, h2 / ss))
        return spr


# --- the renderer -------------------------------------------------------------------

class DanceRenderer:
    def __init__(self, size: tuple[int, int], art: DanceArt) -> None:
        self.width, self.height = size
        self.art = art
        W, H = size
        man = art.manifest
        stage = art.stage
        if stage.shape[:2] != (H, W):
            stage = cv2.resize(stage, (W, H), interpolation=cv2.INTER_AREA)
        self._stage = np.ascontiguousarray(stage[..., :3])
        self._dark_stage = None

        # The light buffer: everything that pulses, summed at 1/q resolution.
        q = int(man.get("light_scale", 4))
        self.q = q if q in (1, 2, 4, 8) else 4
        self._qbits = int(math.log2(self.q))
        lw, lh = W // self.q, H // self.q
        self._light = np.zeros((lh, lw, 3), np.uint8)
        self._light_up = np.empty((H, W, 3), np.uint8)
        self._mask_q = np.zeros((lh, lw), np.uint8)

        def quarter(img, colour=None):
            img = cv2.resize(img, (lw, lh), interpolation=cv2.INTER_AREA) if img.shape[:2] != (lh, lw) else img
            return np.ascontiguousarray(_tinted(img, colour) if colour is not None else img[..., :3])

        cols = man.get("colours", {})
        self.c_magenta = tuple(cols.get("magenta", (190, 50, 255)))
        self.c_cyan = tuple(cols.get("cyan", (255, 235, 70)))
        self.c_violet = tuple(cols.get("violet", (255, 90, 150)))
        self.c_gold = tuple(cols.get("gold", (70, 205, 255)))
        self._trims = quarter(art.glow_trims)
        self._floor = [quarter(art.glow_floor, c) for c in (self.c_violet, self.c_magenta, self.c_cyan, self.c_magenta)]
        self._edge_gold = quarter(art.glow_edge, self.c_gold)

        # Beams: the white beam at 1/q, rotated about its apex in 2-degree steps,
        # tinted per light, cropped to its pixels.
        lights = man.get("stage", {}).get("lights", [[170, 94], [345, 94], [520, 94], [695, 94], [870, 94]])
        self._lights = [(x / self.q, y / self.q) for x, y in lights]
        palette = [self.c_magenta, self.c_cyan, (235, 225, 255), self.c_cyan, self.c_magenta]
        self._light_cols = [palette[i % len(palette)] for i in range(len(lights))]
        self._beams: dict[tuple[int, int], tuple[np.ndarray, int, int]] = {}
        bh, bw = art.beam.shape[:2]
        self._beam_grey = cv2.resize(art.beam[..., 3], (max(4, bw // self.q), max(8, bh // self.q)),
                                     interpolation=cv2.INTER_AREA)

        # The coach (outfit set when the song is known).
        self._painter: CoachPainter | None = None
        self._painter_outfit = None
        self._cards = PictogramCards(art)
        self._ring = Sprite(_resize_bgra(art.ring, RING, RING), tint=self.c_cyan)
        self._ring_gold = Sprite(_resize_bgra(art.ring, RING, RING), tint=self.c_gold)
        self._ring_light = _tinted(_resize_bgra(art.ring, RING, RING), WHITE)
        # Difficulty colours: EASY green, NORMAL cyan, HARD magenta (the manifest's
        # palette, overridable there as "difficulty": {"easy": [b, g, r], ...}).
        dc = man.get("difficulty", {}) if isinstance(man.get("difficulty", {}), dict) else {}
        self.difficulty_colours = {
            "easy": tuple(dc.get("easy", theme.GOOD)),
            "normal": tuple(dc.get("normal", self.c_cyan)),
            "hard": tuple(dc.get("hard", self.c_magenta)),
        }
        self._badges: dict[tuple[str, float], Sprite] = {}
        self._lane_bg = self._make_lane_bg()

        # FX
        self._words = {k: ScaledSprites(v) for k, v in art.words.items()}
        self._sparkle = {k: Sprite(_resize_bgra(art.sparkle, s, s), tint=WHITE) for k, s in
                         ((0, 22), (1, 30), (2, 40), (3, 52))}
        self._sparkle_gold = {k: Sprite(_resize_bgra(art.sparkle, s, s), tint=(150, 230, 255)) for k, s in
                              ((0, 22), (1, 30), (2, 40), (3, 52))}
        self._burst = ScaledSprites(art.burst, step=0.1, tint=(120, 215, 255))
        self._glow_dot = Sprite(_resize_bgra(art.glow, 90, 90), tint=(150, 230, 255))
        self._star_full = ScaledSprites(art.star_full, step=0.05)
        self._star_empty = Sprite(_resize_bgra(art.star_empty, 40, 40))
        self._star_empty_big = Sprite(_resize_bgra(art.star_empty, 64, 64))
        self._star_bar = self._make_star_bar()
        self._pip_mask = self._make_pip_mask()
        self._title_cache: dict[str, tuple[Sprite, Sprite | None]] = {}
        self._count_words = {k: ScaledSprites(art.words[k], step=0.05) for k in
                             ("get_ready", "count_3", "count_2", "count_1", "count_go", "superstar") if k in art.words}

        # Animation state the game doesn't carry.
        self._score_seen = None
        self._score_t = -10.0
        self._stars_lit: dict[int, float] = {}
        self._outro_t0: float | None = None
        self._clock = 0.0
        self._combo_seen = 0
        self._combo_t = -10.0
        self.timings: dict[str, float] | None = None  # set to {} to collect per-part ms

    # --- pre-built pieces ---

    def _make_lane_bg(self) -> Sprite:
        x0, y0, x1, y1 = LANE_RECT
        w, h = x1 - x0, y1 - y0
        img = np.zeros((h, w, 4), np.uint8)
        r = 20
        body = np.zeros((h, w), np.uint8)
        cv2.rectangle(body, (r, 0), (w - 1 - r, h - 1), 255, -1)
        cv2.rectangle(body, (0, r), (w - 1, h - 1 - r), 255, -1)
        for cx, cy in ((r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)):
            cv2.circle(body, (cx, cy), r, 255, -1, cv2.LINE_AA)
        # Fade in from the right edge so cards emerge rather than pop.
        ramp = np.clip(np.linspace(1.6, 0.0, w) * 1.0, 0, 1) ** 0.6
        ramp = np.clip(0.55 + 0.45 * ramp, 0, 1)
        img[..., :3] = (34, 14, 22)
        img[..., 3] = (body.astype(np.float32) * 0.72 * ramp[None, :]).astype(np.uint8)
        edge = np.zeros((h, w), np.uint8)
        cv2.line(edge, (r, 1), (w - r, 1), 255, 2, cv2.LINE_AA)
        cv2.line(edge, (r, h - 2), (w - r, h - 2), 255, 2, cv2.LINE_AA)
        e = edge.astype(np.float32) / 255.0 * ramp[None, :]
        img[..., :3] = (img[..., :3] * (1 - e[..., None]) + np.array(self.c_violet) * e[..., None]).astype(np.uint8)
        img[..., 3] = np.maximum(img[..., 3], (e * 200).astype(np.uint8))
        return Sprite(img)

    def _make_star_bar(self) -> dict:
        x0, y0, x1, y1 = STAR_BAR
        w, h = x1 - x0 + 12, y1 - y0 + 12
        track = np.zeros((h, w, 4), np.uint8)
        cv2.rectangle(track, (6, 6), (w - 7, h - 7), (60, 30, 40, 220), -1, cv2.LINE_AA)
        cv2.rectangle(track, (4, 4), (w - 5, h - 5), (220, 150, 190, 160), 2, cv2.LINE_AA)
        # The fill: magenta at the bottom rising to gold.
        ys = np.linspace(0, 1, y1 - y0, dtype=np.float32)[:, None, None]
        bottom = np.array(self.c_magenta, np.float32)
        top = np.array(self.c_gold, np.float32)
        fill = (top * (1 - ys) + bottom * ys) * np.ones((1, x1 - x0, 1), np.float32)
        xs = np.linspace(-1, 1, x1 - x0, dtype=np.float32)[None, :, None]
        fill = fill + (255 - fill) * np.clip(0.35 - np.abs(xs + 0.3), 0, 1)  # a glossy stripe
        return {"track": Sprite(track), "fill": np.clip(fill, 0, 255).astype(np.uint8)}

    def _make_pip_mask(self) -> np.ndarray:
        x0, y0, x1, y1 = PIP_RECT
        w, h, r = x1 - x0, y1 - y0, PIP_RADIUS
        m = np.zeros((h, w), np.uint8)
        cv2.rectangle(m, (r, 0), (w - 1 - r, h - 1), 255, -1)
        cv2.rectangle(m, (0, r), (w - 1, h - 1 - r), 255, -1)
        for cx, cy in ((r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)):
            cv2.circle(m, (cx, cy), r, 255, -1, cv2.LINE_AA)
        return np.repeat((m > 127)[:, :, None], 3, axis=2)

    def _grab_pip(self, frame: np.ndarray) -> np.ndarray:
        """The camera frame shrunk into the PiP. OpenCV's INTER_AREA is only fast
        at integer ratios: 1280 -> 384 directly took 4.5 ms a frame. A 2x box
        filter (its fast path) then a bilinear step costs 0.08 ms and looks the same."""
        x0, y0, x1, y1 = PIP_RECT
        w, h = x1 - x0, y1 - y0
        fh, fw = frame.shape[:2]
        if fw >= 2 * w and fh >= 2 * h:
            half = getattr(self, "_pip_half", None)
            if half is None or half.shape[:2] != (fh // 2, fw // 2):
                half = self._pip_half = np.empty((fh // 2, fw // 2, 3), np.uint8)
            cv2.resize(frame, (fw // 2, fh // 2), dst=half, interpolation=cv2.INTER_AREA)
            return cv2.resize(half, (w, h), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

    def _beam(self, light: int, angle: float):
        b = int(round(angle / 2.0))
        key = (light, b)
        hit = self._beams.get(key)
        if hit is None:
            g = self._beam_grey
            bh, bw = g.shape
            d = int(math.ceil(math.hypot(bh, bw / 2))) + 2
            canvas = np.zeros((d + 4, 2 * d), np.uint8)
            canvas[4:4 + bh, d - bw // 2:d - bw // 2 + bw] = g
            m = cv2.getRotationMatrix2D((d, 4), b * 2.0, 1.0)
            rot = cv2.warpAffine(canvas, m, (2 * d, d + 4), flags=cv2.INTER_LINEAR)
            ys, xs = np.nonzero(rot > 2)
            if len(ys) == 0:
                hit = (np.zeros((1, 1, 3), np.uint8), 0, 0)
            else:
                ya, yb, xa, xb = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                crop = rot[ya:yb, xa:xb].astype(np.float32)[..., None] / 255.0
                col = np.asarray(self._light_cols[light], np.float32)[None, None, :]
                hit = (np.clip(crop * col + 0.5, 0, 255).astype(np.uint8), xa - d, ya - 4)
            self._beams[key] = hit
        return hit

    def _painter_for(self, song) -> CoachPainter:
        outfit = (getattr(song, "coach", None) or {}).get("outfit", "neon") if song is not None else "neon"
        if self._painter is None or self._painter_outfit != outfit:
            self._painter = CoachPainter(self.art, outfit)
            self._painter_outfit = outfit
        return self._painter

    # --- main ---

    def render(self, frame: np.ndarray, game) -> None:
        tm = self.timings
        t0 = _now() if tm is not None else 0.0

        def lap(name):
            nonlocal t0
            if tm is not None:
                t1 = _now()
                tm[name] = tm.get(name, 0.0) + (t1 - t0) * 1000
                t0 = t1

        phase = getattr(getattr(game, "phase", None), "value", str(getattr(game, "phase", "")))
        song = getattr(game, "song", None)
        beat = float(getattr(game, "beat", 0.0) or 0.0)
        song_time = float(getattr(game, "song_time", 0.0) or 0.0)
        over = phase == "game_over"
        self._clock = song_time if not over else self._clock
        pip = None
        if not over:  # the camera, as received, before anything is drawn over it
            pip = self._grab_pip(frame)
        lap("pip_grab")

        np.copyto(frame, self._stage)
        lap("stage")

        painter = self._painter_for(song)
        bow = self._bow(phase, song, beat, song_time)
        pose = getattr(game, "coach_pose", None)
        if over:  # the coach grooves in its spotlight while the results count up
            beat = float(getattr(game, "results_t", 0.0) or 0.0) * 2.0
            pose = idle_groove(beat)
        geo = painter.layout(pose, bow=bow, beat=beat)
        energy = self._energy(song, beat, phase)
        self._draw_lights(frame, game, geo, beat, energy, phase)
        lap("lights")

        painter.paint(frame, geo, clip=COACH_ROI)
        lap("coach")

        if over:
            self._draw_results(frame, game, song)
            lap("results")
            return
        if phase in ("intro", "dancing"):
            self._draw_lane(frame, game, beat)
        lap("lane")
        self._draw_feedback(frame, game, geo)
        lap("feedback")
        if phase == "intro":
            self._draw_intro(frame, song, beat, getattr(game, "difficulty", None))
        if phase == "outro":
            self._draw_outro_fade(frame, song, beat)
        lap("intro_outro")
        self._draw_hud(frame, game, song, phase, beat)
        lap("hud")
        self._draw_pip(frame, pip, game)
        lap("pip")

    # --- timing helpers ---

    def _energy(self, song, beat: float, phase: str) -> float:
        if phase == "game_over":
            return 0.35
        name = ""
        if song is not None and hasattr(song, "section_at"):
            try:
                name = str(song.section_at(beat)).lower()
            except Exception:  # noqa: BLE001 - a malformed section list must not stop the show
                name = ""
        for key, e in SECTION_ENERGY:
            if key in name:
                return e
        return 0.72

    def _bow(self, phase: str, song, beat: float, song_time: float) -> float:
        if phase not in ("outro", "game_over"):
            self._outro_t0 = None
            return 0.0
        if self._outro_t0 is None:
            self._outro_t0 = max(song_time, _ending_start(song))
        if phase == "game_over":
            return 0.0
        t = song_time - self._outro_t0
        bs = (song.beat_seconds if song is not None and hasattr(song, "beat_seconds") else 0.5)
        # Down over a beat, hold two, back up over a beat.
        u = t / bs
        if u < 0.4:
            return 0.0
        if u < 1.4:
            return smoothstep(u - 0.4)
        if u < 3.4:
            return 1.0
        return 1.0 - smoothstep((u - 3.4) / 1.2)

    # --- layers ---

    def _draw_lights(self, frame, game, geo: CoachGeometry, beat: float, energy: float, phase: str) -> None:
        L = self._light
        L.fill(0)
        kick = _kick(beat) if phase in ("intro", "dancing", "outro") else 0.0
        bar = _kick(beat / 4.0, 6.0) if phase in ("dancing",) else 0.0
        # Neon trims flare on each bar's downbeat; floor pool on every beat.
        cv2.addWeighted(L, 1.0, self._trims, 0.10 + 0.55 * bar * energy + 0.12 * kick, 0.0, dst=L)
        fl = self._floor[int(math.floor(beat / 4.0)) % len(self._floor)] if beat >= 0 else self._floor[0]
        cv2.addWeighted(L, 1.0, fl, (0.18 + 0.62 * kick) * (0.5 + 0.5 * energy), 0.0, dst=L)

        # Beams: outer lights sweep over two bars; the centre one breathes straight down.
        n = len(self._lights)
        for i, (lx, ly) in enumerate(self._lights):
            centre = abs(i - (n - 1) / 2) < 0.6
            if centre:
                ang = 0.0
                gain = 0.22 + 0.25 * kick * energy
            else:
                side = -1.0 if i < (n - 1) / 2 else 1.0
                swing = 24.0 if energy >= 0.9 else 16.0
                ang = -side * 14.0 + swing * math.sin(2 * math.pi * beat / 8.0 + i * 1.9)
                gain = (0.38 + 0.5 * kick) * energy
            crop, ox, oy = self._beam(i, ang)
            x, y = int(round(lx)) + ox, int(round(ly)) + oy
            h, w = crop.shape[:2]
            r = _clip_rect(x, y, w, h, (0, 0, L.shape[1], L.shape[0]))
            if r is None or gain <= 0.01:
                continue
            xa, ya, xb, yb = r
            roi = L[ya:yb, xa:xb]
            cv2.addWeighted(roi, 1.0, crop[ya - y:yb - y, xa - x:xb - x], gain, 0.0, dst=roi)

        # The coach's halo: its silhouette at 1/q, blurred, in the outfit's glow colour.
        painter = self._painter
        m = self._mask_q
        bx0, by0, bx1, by1 = geo.bbox
        q = self.q
        pad = 6
        qx0, qy0 = max(0, bx0 // q - pad), max(0, by0 // q - pad)
        qx1, qy1 = min(m.shape[1], bx1 // q + pad), min(m.shape[0], by1 // q + pad)
        if qx1 > qx0 and qy1 > qy0 and painter is not None:
            sub = m[qy0:qy1, qx0:qx1]
            sub.fill(0)
            o = np.array((qx0 * q * ONE, qy0 * q * ONE), np.int32)
            for p in geo.silhouette:
                cv2.fillConvexPoly(sub, p.reshape(-1, 2) - o, 255, cv2.LINE_AA, SHIFT + self._qbits)
            hx, hy = geo.head_c
            cv2.ellipse(sub, (int(hx / q - qx0), int(hy / q - qy0)),
                        (int(0.62 * painter.S / q), int(0.58 * painter.S / q)), 0, 0, 360, 255, -1, cv2.LINE_AA)
            blur = cv2.GaussianBlur(sub, (0, 0), 2.4)
            col = painter.c.get("glow", (255, 80, 200))
            g = 0.75 + 0.35 * kick
            glow = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)
            cv2.multiply(glow, (col[0] / 255 * g, col[1] / 255 * g, col[2] / 255 * g, 0), dst=glow)
            roi = L[qy0:qy1, qx0:qx1]
            cv2.add(roi, glow, dst=roi)

        gold = float(getattr(game, "gold_flash", 0.0) or 0.0) if phase != "game_over" else 0.0
        if gold > 0.01:
            cv2.addWeighted(L, 1.0, self._edge_gold, min(1.6, gold * 1.6), 0.0, dst=L)

        cv2.resize(L, (self.width, self.height), dst=self._light_up, interpolation=cv2.INTER_LINEAR)
        cv2.add(frame, self._light_up, dst=frame)

    def _draw_lane(self, frame, game, beat: float) -> None:
        items = list(getattr(game, "upcoming", None) or [])
        if not items:
            return
        self._lane_bg.draw(frame, LANE_RECT[0], LANE_RECT[1])
        x0, y0, x1, y1 = LANE_RECT
        clip = (x0 + 2, y0 - 16, x1 - 2, LANE_CLIP_BOTTOM)
        kick = _kick(beat)
        # Beat dots along the track, scrolling with the cards.
        b0 = math.ceil(beat - (TARGET_X - x0) / PX_PER_BEAT)
        for b in range(b0, int(beat + (x1 - TARGET_X) / PX_PER_BEAT) + 1):
            x = TARGET_X + (b - beat) * PX_PER_BEAT
            if x0 + 14 < x < x1 - 14 and abs(x - TARGET_X) > RING * 0.41:
                big = b % 4 == 0
                cv2.circle(frame, (int(x), y1 - 10), 4 if big else 2, self.c_violet if not big else (240, 200, 255),
                           -1, cv2.LINE_AA)
        # The next card: the first one not yet past the ring.
        nxt = None
        for it in items:
            if float(it.beat) - beat >= -0.3:
                nxt = it
                break
        ring = self._ring_gold if (nxt is not None and getattr(nxt, "gold", False)) else self._ring
        ring.draw_c(frame, TARGET_X, LANE_Y, fade=0.75 + 0.25 * kick)
        flash = 0.0
        if nxt is not None:
            d = float(nxt.beat) - beat
            if -0.3 <= d <= 0.12:
                flash = 1.0 - abs(d + 0.05) / 0.3
        for spr_gain in ((0.25 + 0.5 * kick) * 0.6 + flash * 0.9,):
            rh = RING // 2
            v = _clip_rect(TARGET_X - rh, LANE_Y - rh, RING, RING, (0, 0, self.width, self.height))
            if v is not None and spr_gain > 0.01:
                xa, ya, xb, yb = v
                roi = frame[ya:yb, xa:xb]
                src = self._ring_light[ya - (LANE_Y - rh):yb - (LANE_Y - rh), xa - (TARGET_X - rh):xb - (TARGET_X - rh)]
                cv2.addWeighted(roi, 1.0, src, min(1.0, spr_gain), 0.0, dst=roi)

        visible = []
        for it in items:
            d = float(it.beat) - beat
            if d < -0.65:
                continue
            x = TARGET_X + d * PX_PER_BEAT
            if x - CARD_SMALL / 2 > x1:
                continue
            visible.append((d, x, it))
        # Cards come as often as every beat (one per judge point). Size goes by
        # distance alone, so nothing pops when the "next" card changes: small
        # until GROW_BEATS out, full size on the ring, and it stays full size
        # while it slides through and fades. Passed cards are drawn first,
        # under the incoming ones; then the farthest first, the next on top.
        for d, x, it in sorted(visible, key=lambda v: (v[0] >= 0, -v[0])):
            w = card_width(d)
            fade = 1.0
            if d < 0:
                fade = max(0.0, 1.0 + d / 0.65)
                x = TARGET_X + d * PX_PER_BEAT * 0.6
            spr = self._cards.get(it.pose, bool(getattr(it, "gold", False)), w)
            spr.draw_c(frame, x, LANE_Y, fade=fade, clip=clip)
            if getattr(it, "gold", False) and fade > 0.3:  # twinkles on gold cards
                for k in range(2):
                    ph = beat * 1.3 + k * 0.5
                    tw = abs(math.sin(ph * math.pi))
                    sx = x + (-0.36 if k else 0.38) * spr.w
                    sy = LANE_Y + (-0.42 if k else 0.36) * spr.h
                    self._sparkle_gold[1].add_c(frame, sx, sy, gain=tw * fade, clip=clip)
        if nxt is not None:
            label = str(getattr(nxt, "label", None) or getattr(nxt, "move_label", "") or "").upper()
            d = float(nxt.beat) - beat
            if label and d < 4.0:
                col = self.c_gold if getattr(nxt, "gold", False) else WHITE
                a = min(1.0, (4.0 - d) / 0.6)
                text = label if not getattr(nxt, "gold", False) else f"{label}  x2"
                _faded(frame, (TARGET_X - 270, LABEL_Y - 26, TARGET_X + 270, LABEL_Y + 10), a,
                       lambda img, ox, oy: text_c(img, text, TARGET_X - ox, LABEL_Y - oy, LABEL_SCALE, col, 2))

    def _draw_feedback(self, frame, game, geo: CoachGeometry) -> None:
        fbs = list(getattr(game, "feedback", None) or [])
        if not fbs or getattr(game, "notice", None):
            # The notice ("STEP BACK...") sits where the words pop; a word
            # showing through its translucent panel read as a ghost (QA).
            return
        cx = float(np.clip(geo.head_c[0], *POPUP_X))
        base_y = max(POPUP_Y_MIN + 30, geo.head_top - 26)
        gold_flash = float(getattr(game, "gold_flash", 0.0) or 0.0)
        if gold_flash > 0.01:  # a burst of rays behind the popup
            s = 0.6 + 0.5 * (1.0 - gold_flash)
            self._burst.get(s).add_c(frame, cx, base_y - 10, gain=min(1.0, gold_flash * 1.1))
        shown = [fb for fb in fbs if rating_key(getattr(fb, "rating", None)) in self._words]
        for i, fb in enumerate(shown):
            key = rating_key(getattr(fb, "rating", None))
            age = float(getattr(fb, "age", 0.0) or 0.0)
            if age < 0 or age > POPUP_LIFE:
                continue
            # Judge points come every beat (0.5 s) but a popup lives 0.85 s: when a
            # newer rating arrives, the older word whisks up and out instead of
            # stacking under it ("PERFECT" over "PERFECT" on the full-song run).
            superseded = float(getattr(shown[i + 1], "age", 0.0) or 0.0) if i + 1 < len(shown) else None
            if superseded is not None and superseded >= SUPERSEDE_FADE:
                continue
            gold = bool(getattr(fb, "gold", False))
            s = _pop(age)
            x = cx
            if key == "x":  # a miss shakes instead of bouncing; one letter needs the extra size
                s = min(1.0, 0.7 + age * 3) * 1.35
                x += math.sin(age * 55) * 9 * max(0.0, 1 - age / 0.35)
            y = base_y - 34 * _ease_out(age / POPUP_LIFE)
            fade = 1.0 if age < 0.5 else max(0.0, 1.0 - (age - 0.5) / (POPUP_LIFE - 0.5))
            if superseded is not None:
                u = superseded / SUPERSEDE_FADE
                fade *= (1.0 - u) ** 2  # mostly gone on the new word's first frame
                y -= 46 * u
            if key == "perfect" or (gold and key in ("perfect", "super")):
                self._sparkles(frame, x, y, age, gold)
            self._words[key].get(s * (1.1 if gold and key != "x" else 1.0)).draw_c(frame, x, y, fade=fade)
        combo = int(getattr(game, "combo", 0) or 0)
        if combo != self._combo_seen:
            if combo > self._combo_seen:
                self._combo_t = self._clock
            self._combo_seen = combo

    def _sparkles(self, frame, x, y, age: float, gold: bool) -> None:
        u = min(1.0, age / 0.7)
        n = 8 if gold else 6
        sprites = self._sparkle_gold if gold else self._sparkle
        for k in range(n):
            a = (k / n) * 2 * math.pi + (0.4 if gold else 0.0)
            r = 60 + 120 * _ease_out(u) * (1.0 if k % 2 else 0.75)
            sx = x + math.cos(a) * r * 1.5
            sy = y + math.sin(a) * r * 0.7 - 20 * u
            size = 3 if (k % 2 == 0 and u < 0.5) else (2 if u < 0.6 else 1)
            sprites[size].add_c(frame, sx, sy, gain=max(0.0, 1.0 - u) * 1.2)
        if age < 0.25:
            self._glow_dot.add_c(frame, x, y, gain=(1 - age / 0.25) * (0.9 if gold else 0.6))

    def _draw_intro(self, frame, song, beat: float, difficulty: str | None = None) -> None:
        fm = float(getattr(song, "first_move_beat", 16.0) or 16.0) if song is not None else 16.0
        cx, cy = INTRO_C
        title = self._title(song)
        if title is not None and beat < fm - 7.5:
            t_spr, a_spr = title
            age = beat + 4.0  # the title is up from before beat 0
            s = 1.0
            fade = min(1.0, (fm - 7.5 - beat) / 0.75)
            dy = (1 - _ease_out(age / 1.2)) * 30
            _faded(frame, (cx - 160, cy - 100 + dy, cx + 160, cy - 64 + dy), fade,
                   lambda img, ox, oy: text_c(img, "NOW DANCING", cx - ox, cy - 74 + dy - oy, 0.62, theme.DIM, 1,
                                              theme.BODY_FONT))
            t_spr.draw_c(frame, cx, cy + dy, fade=fade * s)
            if a_spr is not None:
                a_spr.draw_c(frame, cx, cy + 66 + dy, fade=fade)
            badge = self.badge(difficulty, 0.8)
            if badge is not None:
                badge.draw_c(frame, cx, cy + 122 + dy, fade=fade)
        k = beat - (fm - 8)
        if 0 <= k < 4 and "get_ready" in self._count_words:
            s = _pop(k * 0.5, dur=0.4) * (1.0 + 0.05 * _kick(beat))
            fade = min(1.0, (4 - k) / 0.4)
            self._count_words["get_ready"].get(s).draw_c(frame, cx, cy, fade=fade)
        for i, name in enumerate(("count_3", "count_2", "count_1", "count_go")):
            b = fm - 4 + i
            age = (beat - b) * (getattr(song, "beat_seconds", 0.5) if song is not None else 0.5)
            if 0 <= beat - b < 1 and name in self._count_words:
                s = _pop(age, dur=0.22, over=1.18, start=1.35)  # 1.6 reached up into the PiP
                fade = 1.0 if beat - b < 0.6 else max(0.0, (1 - (beat - b)) / 0.4)
                self._count_words[name].get(s).draw_c(frame, cx, COUNT_Y, fade=fade)
        if beat >= fm - 7.5 and song is not None:  # small title, top-centre
            text_c(frame, f"{song.title}  -  {getattr(song, 'artist', '')}".strip(" -"), 640, 40, 0.55,
                   theme.DIM, 1, theme.BODY_FONT)

    def badge(self, difficulty, scale: float = 0.62) -> Sprite | None:
        """The difficulty's badge sprite (cached), or None for a game without difficulties."""
        key = str(difficulty).lower() if difficulty is not None else ""
        if key not in DIFFICULTY_LABELS:
            return None
        hit = self._badges.get((key, scale))
        if hit is None:
            hit = self._badges[(key, scale)] = Sprite(make_badge(DIFFICULTY_LABELS[key],
                                                                 self.difficulty_colours[key], scale))
        return hit

    def _title(self, song):
        if song is None:
            return None
        key = f"{getattr(song, 'id', '')}|{song.title}|{getattr(song, 'artist', '')}"
        hit = self._title_cache.get(key)
        if hit is None:
            t = make_glow_text(str(song.title), 2.4, self.c_gold, glow=self.c_magenta, thick=3)
            if t.shape[1] > 560:  # long titles shrink to fit the right side of the stage
                t = _resize_bgra(t, 560, t.shape[0] * 560 / t.shape[1])
            artist = str(getattr(song, "artist", "") or "")
            a = make_glow_text(artist, 0.95, self.c_cyan, thick=2) if artist else None
            hit = self._title_cache[key] = (Sprite(t), Sprite(a) if a is not None else None)
        return hit

    def _draw_outro_fade(self, frame, song, beat: float) -> None:
        if self._outro_t0 is None:
            return
        t = self._clock - self._outro_t0
        u = min(1.0, max(0.0, (t - 2.0) / 3.0))
        if u > 0.01:
            cv2.convertScaleAbs(frame, dst=frame, alpha=1.0 - 0.45 * u)

    def _draw_hud(self, frame, game, song, phase: str, beat: float) -> None:
        score = int(getattr(game, "score", 0) or 0)
        shown = int(round(float(getattr(game, "display_score", score) or 0)))
        if self._score_seen is None:
            self._score_seen = score
        if score != self._score_seen:
            self._score_seen = score
            self._score_t = self._clock
        pulse = max(0.0, 1.0 - (self._clock - self._score_t) / 0.35)
        colour = tuple(int(g + (255 - g) * pulse * 0.7) for g in theme.ACCENT)
        outlined_text(frame, f"{shown}", (SCORE_ORG[0], SCORE_ORG[1]), FONT, 1.5 * (1.0 + 0.18 * pulse),
                      colour, 3, width=2)
        combo = int(getattr(game, "combo", 0) or 0)
        if combo >= 3:
            cp = max(0.0, 1.0 - (self._clock - self._combo_t) / 0.3)
            text = f"x{combo} COMBO"
            outlined_text(frame, text, (SCORE_ORG[0], SCORE_ORG[1] + 44), FONT, 0.85 * (1 + 0.2 * cp),
                          theme.CYAN, 2, width=2)
        badge = self.badge(getattr(game, "difficulty", None), 0.6)
        if badge is not None:
            badge.draw(frame, HUD_BADGE_ORG[0], HUD_BADGE_ORG[1] - badge.h // 2)
        self._draw_star_bar(frame, float(getattr(game, "stars", 0.0) or 0.0))
        notice = getattr(game, "notice", None)
        if notice:
            text = str(notice)
            (tw, th), _ = cv2.getTextSize(text, FONT, 0.9, 2)
            cx, cy = 600, 108 if phase == "intro" else 96
            theme.panel(frame, (cx - tw / 2 - 22, cy - th - 16, cx + tw / 2 + 22, cy + 14), alpha=0.82,
                        border=theme.WARN, radius=14)
            text_c(frame, text, cx, cy, 0.9, theme.WARN, 2)

    def _draw_star_bar(self, frame, stars: float) -> None:
        x0, y0, x1, y1 = STAR_BAR
        sb = self._star_bar
        sb["track"].draw(frame, x0 - 6, y0 - 6)
        stars = min(5.0, max(0.0, stars))
        h = int(round((y1 - y0) * stars / 5.0))
        if h > 0:
            frame[y1 - h:y1, x0:x1] = sb["fill"][(y1 - y0) - h:]
            cv2.line(frame, (x0, y1 - h), (x1 - 1, y1 - h), WHITE, 2, cv2.LINE_AA)
        cx = (x0 + x1) // 2
        for k in range(1, 6):
            y = y1 - (y1 - y0) * k / 5.0
            if stars >= k - 1e-6:
                t_lit = self._stars_lit.setdefault(k, self._clock)
                age = self._clock - t_lit
                s = _pop(age, dur=0.35, over=1.5, start=0.6) * 0.64
                if age < 0.6:
                    self._sparkle_gold[2].add_c(frame, cx, y, gain=1.0 - age / 0.6)
                self._star_full.get(s).draw_c(frame, cx, y)
            else:
                self._stars_lit.pop(k, None)
                self._star_empty.draw_c(frame, cx, y)

    def _draw_pip(self, frame, pip, game) -> None:
        if pip is None:
            return
        x0, y0, x1, y1 = PIP_RECT
        key = rating_key(getattr(game, "last_rating", None))
        col = self.art.rating_colour(key) if key else (235, 235, 235)
        if key and col == (255, 255, 255):
            col = RATING_FALLBACK[key]
        m = getattr(game, "player", None)
        if m is not None:
            _draw_measured(pip, m, (x1 - x0) / self.width, col)
        roi = frame[y0:y1, x0:x1]
        np.copyto(roi, pip, where=self._pip_mask)
        r = PIP_RADIUS
        for (a, b) in (((x0 + r, y0), (x1 - r, y0)), ((x0 + r, y1), (x1 - r, y1)),
                       ((x0, y0 + r), (x0, y1 - r)), ((x1, y0 + r), (x1, y1 - r))):
            cv2.line(frame, a, b, col, 3, cv2.LINE_AA)
        for (cx, cy), ang in (((x0 + r, y0 + r), 180), ((x1 - r, y0 + r), 270),
                              ((x1 - r, y1 - r), 0), ((x0 + r, y1 - r), 90)):
            cv2.ellipse(frame, (cx, cy), (r, r), 0, ang, ang + 90, col, 3, cv2.LINE_AA)
        outlined_text(frame, "YOU", (x0 + 14, y1 - 14), theme.BODY_FONT, 0.6, WHITE, 1, width=1)

    def _draw_results(self, frame, game, song) -> None:
        # Everything has landed by ~1.25 s: the shell covers this screen with its
        # name entry after ENTRY_DELAY (1.6 s) and stops updating the game, so a
        # slower count-up froze half-way behind it (QA: "9419" next to a 9463 board).
        t = float(getattr(game, "results_t", 0.0) or 0.0)
        x0, y0, x1, y1 = RESULTS_RECT
        a = _ease_out(t / 0.4)
        theme.panel(frame, (x0, y0 + (1 - a) * 30, x1, y1 + (1 - a) * 30), alpha=0.86 * a,
                    border=theme.ACCENT if a > 0.6 else None, radius=22, thickness=3)
        if a < 0.6:
            return
        cx = (x0 + x1) // 2
        title = str(getattr(song, "title", "")) if song is not None else ""
        badge = self.badge(getattr(game, "difficulty", None), 0.65)
        if badge is None:
            text_c(frame, "RESULTS", cx, y0 + 40, 0.6, theme.DIM, 1, theme.BODY_FONT)
        else:  # "RESULTS [EASY]", centred as a pair
            (tw, _), _ = cv2.getTextSize("RESULTS", theme.BODY_FONT, 0.6, 1)
            left = cx - (tw + 12 + badge.w) / 2
            text_c(frame, "RESULTS", left + tw / 2, y0 + 40, 0.6, theme.DIM, 1, theme.BODY_FONT)
            badge.draw(frame, int(round(left + tw + 12)), int(round(y0 + 34 - badge.h / 2)))
        text_c(frame, title.upper(), cx, y0 + 80, 1.05, theme.ACCENT, 2)
        stars = int(math.floor(float(getattr(game, "stars", 0.0) or 0.0) + 1e-6))
        for k in range(5):
            sx = cx + (k - 2) * 84
            sy = y0 + 128
            appear = 0.35 + 0.18 * k
            if k < stars and t >= appear:
                age = t - appear
                self._star_full.get(_pop(age, dur=0.35, over=1.45, start=0.5) * 1.15).draw_c(frame, sx, sy)
                if age < 0.6:
                    self._sparkle_gold[3].add_c(frame, sx, sy, gain=1.0 - age / 0.6)
            else:
                self._star_empty_big.draw_c(frame, sx, sy)
        score = int(getattr(game, "score", 0) or 0)
        u = _ease_out((t - 0.2) / 1.0)
        shown = int(round(score * u))
        text_c(frame, f"{shown}", cx, y0 + 222, 2.1, theme.ACCENT, 3, outline=3)
        if score >= 9600 and t > 1.25 and "superstar" in self._count_words:
            self._count_words["superstar"].get(_pop(t - 1.25) * 0.85).draw_c(frame, cx, y0 + 266)
        counts = getattr(game, "counts", None) or {}
        row_y = y0 + 318
        for i, key in enumerate(RATING_KEYS):
            appear = 0.55 + 0.1 * i
            if t < appear:
                continue
            fa = min(1.0, (t - appear) / 0.25)
            dx = (1 - _ease_out((t - appear) / 0.3)) * 40
            y = row_y + i * 36
            word = self._words[key].get(0.5)
            word.draw(frame, int(x0 + 130 - word.w / 2 + dx), int(y - word.h / 2), fade=fa)
            n = int(counts.get(key, 0) or 0)
            txt = f"{n}"
            (tw, _), _ = cv2.getTextSize(txt, FONT, 0.85, 2)
            _faded(frame, (x1 - 200 + dx, y - 22, x1 - 40 + dx, y + 14), fa,
                   lambda img, ox, oy, txt=txt, tw=tw, y=y: outlined_text(
                       img, txt, (int(x1 - 60 - tw - ox + dx), int(y + 12 - oy)), FONT, 0.85, WHITE, 2, width=2))
            # A dotted leader between word and count.
            if fa > 0.5:
                for xx in range(int(x0 + 240 + dx), int(x1 - 90 - tw), 14):
                    cv2.circle(frame, (xx, int(y + 4)), 1, (150, 130, 160), -1, cv2.LINE_AA)


def _ending_start(song) -> float:
    """When the chart's own ending starts: the first of its trailing unjudged
    moves (Neon Nights ends finale -> "bow"). The renderer's bow waits for it,
    so the last judged pose is held on its beat and there is one bow, not two
    (found in QA: the auto bow cut the finale short, then the chart bowed again)."""
    chart = getattr(song, "chart", None) or []
    start = None
    for placed in reversed(chart):
        if getattr(placed.move, "judge", None):
            break
        start = placed.beat
    if start is None or not hasattr(song, "time_of"):
        return -math.inf
    return float(song.time_of(start))


def _now() -> float:
    import time

    return time.perf_counter()


def _draw_measured(img, m, k: float, colour) -> None:
    """The player as the judge measured them (pose.Measured), scaled by k into the PiP."""
    from .pose import FOREARM, HIP_W, SHIN, THIGH, TORSO, UPPER_ARM, limb_dir

    sw = float(m.shoulder_px)
    lean = math.radians(float(m.lean))
    c, s = math.cos(lean), math.sin(lean)

    def rot(v):
        return (v[0] * c - v[1] * s, v[0] * s + v[1] * c)

    cx, cy = m.centre_px

    def P(v):
        return (int(round((cx + v[0]) * k * ONE)), int(round((cy + v[1]) * k * ONE)))

    lines = []
    sh = {}
    for side in "lr":
        sgn = -1 if side == "l" else 1
        sh[side] = rot((sgn * 0.5 * sw, 0.0))
    lines.append((sh["l"], sh["r"]))
    head = rot((0.0, -0.55 * sw))
    for side in "lr":
        a = m.arms.get(side) if hasattr(m, "arms") else None
        if a is None:
            continue
        p0 = sh[side]
        if getattr(a, "upper", None) is not None and getattr(a, "fore", None) is not None:
            d1 = rot(limb_dir(side, a.upper))
            el = (p0[0] + d1[0] * UPPER_ARM * sw, p0[1] + d1[1] * UPPER_ARM * sw)
            d2 = rot(limb_dir(side, a.fore))
            wr = (el[0] + d2[0] * FOREARM * sw, el[1] + d2[1] * FOREARM * sw)
            lines += [(p0, el), (el, wr)]
        elif getattr(a, "reach", None) is not None:
            d = rot(limb_dir(side, a.reach))
            ln = (UPPER_ARM + FOREARM) * 0.92 * sw
            lines.append((p0, (p0[0] + d[0] * ln, p0[1] + d[1] * ln)))
    if getattr(m, "has_hips", False):
        hc = rot((0.0, TORSO * sw))
        hips = {side: (hc[0] + (-1 if side == "l" else 1) * HIP_W / 2 * sw, hc[1]) for side in "lr"}
        lines += [(sh["l"], hips["l"]), (sh["r"], hips["r"]), (hips["l"], hips["r"])]
        for side, leg in (getattr(m, "legs", {}) or {}).items():
            h0 = hips.get(side)
            if h0 is None or leg.thigh is None or leg.shin is None:
                continue
            d1 = limb_dir(side, leg.thigh)
            kn = (h0[0] + d1[0] * THIGH * sw, h0[1] + d1[1] * THIGH * sw)
            d2 = limb_dir(side, leg.shin)
            an = (kn[0] + d2[0] * SHIN * sw, kn[1] + d2[1] * SHIN * sw)
            lines += [(h0, kn), (kn, an)]
    for a, b in lines:
        cv2.line(img, P(a), P(b), (24, 16, 20), 6, cv2.LINE_AA, SHIFT)
    for a, b in lines:
        cv2.line(img, P(a), P(b), colour, 3, cv2.LINE_AA, SHIFT)
    hr = int(round(0.3 * sw * k * ONE))
    cv2.circle(img, P(head), hr, (24, 16, 20), 4, cv2.LINE_AA, SHIFT)
    cv2.circle(img, P(head), hr, colour, 2, cv2.LINE_AA, SHIFT)
