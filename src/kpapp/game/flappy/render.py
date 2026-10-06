"""Drawing Flappy Bird: a full game scene, plus the camera as picture-in-picture.

Unlike Fruit Ninja, the camera is not blended under the scene. There the hand
*is* the blade and the player must see it on the play field; here the input is
a whole-body gesture whose position does not matter, and a busy camera image
behind thin pipes made the gaps hard to read. The camera moves to a corner
window instead, where it does a different job: it teaches the gesture. It shows
the tracked shoulders and wrists, a live "wing" meter for how high the arms
are, and flashes FLAP! the instant a flap is recognised, so a player whose
flaps are not registering can see why.

Budget: <= 4 ms a frame at 1280x720. Every layer is either a slice copy of a
pre-built strip or a blend of premultiplied pixels with two SIMD OpenCV calls
(`dst * (255 - a) / 255 + premultiplied colour`). Nothing is resized, rotated
or tiled per frame; anything that would be is built once at init or cached.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ...config import KP
from ...overlay import outlined_text
from .art import FlappyArt
from .game import FEATHER_TINTS

_WHITE = (255, 255, 255)
_GOLD = (40, 200, 255)
_ORANGE = (40, 120, 235)
_BROWN = (40, 70, 110)
_RED = (60, 60, 235)
_FONT = cv2.FONT_HERSHEY_DUPLEX

SKYLINE_PARALLAX = 0.30  # the far layer scrolls at 30% of the ground: reads as distance
PIP_MARGIN = 16
PIP_RADIUS = 14
ANGLE_STEP = 3.0  # degrees per cached bird rotation; finer is invisible at this size
FEATHER_ANGLES = 12
FEATHER_FADES = 4
FEATHER_SCALE = 1.5  # the 24 px feather was lost against the sky at play distance


class Premul:
    """A BGRA sprite split for fast compositing: colour * alpha, and 255 - alpha.

    Blending is then `dst = dst * inv / 255 + pre`, two saturating OpenCV
    calls on uint8 that run several times faster than float or uint16 numpy.
    """

    __slots__ = ("pre", "inv", "w", "h")

    def __init__(self, bgra: np.ndarray, fade: float = 1.0, tint=None) -> None:
        a = bgra[:, :, 3:4].astype(np.float32) * (fade / 255.0)
        colour = (np.array(tint, np.float32)[None, None, :] if tint is not None
                  else bgra[:, :, :3].astype(np.float32))
        self.pre = np.ascontiguousarray((colour * a + 0.5).astype(np.uint8))
        self.inv = np.ascontiguousarray(np.repeat((255.0 - a * 255.0 + 0.5).astype(np.uint8), 3, axis=2))
        self.h, self.w = bgra.shape[:2]


def _blend(dst: np.ndarray, pre: np.ndarray, inv: np.ndarray) -> None:
    cv2.multiply(dst, inv, dst=dst, scale=1.0 / 255.0)
    cv2.add(dst, pre, dst=dst)


def _blit(frame: np.ndarray, spr: Premul, x: int, y: int,
          clip: tuple[int, int, int, int] | None = None) -> None:
    """Composite `spr` with its top-left at (x, y), clipped to the frame (and `clip`)."""
    fx0, fy0, fx1, fy1 = clip or (0, 0, frame.shape[1], frame.shape[0])
    x0, y0 = max(fx0, x), max(fy0, y)
    x1, y1 = min(fx1, x + spr.w), min(fy1, y + spr.h)
    if x1 <= x0 or y1 <= y0:
        return
    sy, sx = y0 - y, x0 - x
    _blend(frame[y0:y1, x0:x1], spr.pre[sy:sy + y1 - y0, sx:sx + x1 - x0],
           spr.inv[sy:sy + y1 - y0, sx:sx + x1 - x0])


def _tiles(offset: int, tile_w: int, out_w: int):
    """(dst_x0, dst_x1, src_x0) runs that fill [0, out_w) from a strip scrolled by offset."""
    x, src = 0, offset % tile_w
    while x < out_w:
        n = min(tile_w - src, out_w - x)
        yield x, x + n, src
        x += n
        src = 0


def _pad_for_rotation(sprite: np.ndarray) -> np.ndarray:
    """Grow the canvas to the diagonal so rotation never clips the corners."""
    h, w = sprite.shape[:2]
    d = int(math.ceil(math.hypot(h, w))) + 2
    out = np.zeros((d, d, sprite.shape[2]), sprite.dtype)
    y, x = (d - h) // 2, (d - w) // 2
    out[y:y + h, x:x + w] = sprite
    return out


def _rounded_mask(w: int, h: int, r: int) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    cv2.rectangle(m, (r, 0), (w - 1 - r, h - 1), 255, -1)
    cv2.rectangle(m, (0, r), (w - 1, h - 1 - r), 255, -1)
    for cx, cy in ((r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)):
        cv2.circle(m, (cx, cy), r, 255, -1, cv2.LINE_AA)
    return m


def _rounded_border(img, rect, r: int, colour, thickness: int) -> None:
    x0, y0, x1, y1 = rect
    for a, b in (((x0 + r, y0), (x1 - r, y0)), ((x0 + r, y1), (x1 - r, y1)),
                 ((x0, y0 + r), (x0, y1 - r)), ((x1, y0 + r), (x1, y1 - r))):
        cv2.line(img, a, b, colour, thickness, cv2.LINE_AA)
    for (cx, cy), ang in (((x0 + r, y0 + r), 180), ((x1 - r, y0 + r), 270),
                          ((x1 - r, y1 - r), 0), ((x0 + r, y1 - r), 90)):
        cv2.ellipse(img, (cx, cy), (r, r), 0, ang, ang + 90, colour, thickness, cv2.LINE_AA)


def _centred(img, text, cx, y, scale, colour, thick=2, outline=2, font=_FONT) -> None:
    (tw, _), _ = cv2.getTextSize(text, font, scale, thick)
    outlined_text(img, text, (int(cx - tw / 2), int(y)), font, scale, colour, thick, width=outline)


def _ease_out(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return 1.0 - (1.0 - t) ** 3


class FlappyRenderer:
    def __init__(self, size: tuple[int, int], art: FlappyArt, bird_scale: float = 1.0) -> None:
        self.width, self.height = size
        self.art = art
        W, H = size
        self.ground_y = H - art.ground_height

        # Sky: static. Only the rows the skyline does not fully cover are
        # copied each frame (see below).
        self._sky = cv2.resize(art.sky, size) if art.sky.shape[:2] != (H, W) else art.sky

        # Skyline: cropped to its visible rows and split into a translucent top
        # band (blended) and a fully opaque bottom band (plain slice copies).
        sk = art.skyline
        rows = np.nonzero(sk[:, :, 3].max(axis=1))[0]
        top = int(rows[0]) if len(rows) else sk.shape[0]
        opaque = (sk[:, :, 3] == 255).all(axis=1)
        split = sk.shape[0]
        while split > top and opaque[split - 1]:
            split -= 1
        self._sky_y = self.ground_y - sk.shape[0]  # skyline sits on the ground
        self._skyline_top = self._sky_y + top
        self._skyline_split = self._sky_y + split
        self._skyline_blend = Premul(sk[top:split]) if split > top else None
        self._skyline_opaque = np.ascontiguousarray(sk[split:, :, :3])
        self._skyline_w = sk.shape[1]
        # The arcade prints its key hints and footer over the bottom of the
        # ground in light grey, unreadable on sand. Shade the lower ground
        # toward dark, baked into the strip once so it costs nothing per frame.
        ground = art.ground[:, :, :3].astype(np.float32)
        gh = ground.shape[0]
        ramp = np.clip((np.arange(gh, dtype=np.float32) - gh * 0.25) / (gh * 0.75), 0, 1)
        ground *= (1.0 - 0.6 * ramp ** 1.1)[:, None, None]
        self._ground = np.ascontiguousarray(ground.astype(np.uint8))

        # Pipes: a body strip tiled once to the full play height (both
        # orientations), and the cap upright and flipped for the top pipe.
        body = art.pipe_body
        reps = int(math.ceil(self.ground_y / body.shape[0])) + 1
        tall = np.vstack([body] * reps)[: self.ground_y + body.shape[0]]
        self._body_opaque = bool((body[:, :, 3] == 255).all())
        self._body_down = np.ascontiguousarray(tall[:, :, :3])  # anchored at its top
        self._body_up = np.ascontiguousarray(tall[::-1, :, :3])  # anchored at its bottom
        self._body_pm_down = None if self._body_opaque else Premul(tall)
        self._body_pm_up = None if self._body_opaque else Premul(tall[::-1])
        self.pipe_w = body.shape[1]
        self._cap = Premul(art.pipe_cap)
        self._cap_up = Premul(np.ascontiguousarray(art.pipe_cap[::-1]))
        self.cap_h = art.pipe_cap.shape[0]

        # Bird: scaled and padded once; rotations cached on first use.
        self.bird_scale = bird_scale
        frames = art.bird_frames
        if abs(bird_scale - 1.0) > 1e-3:
            h, w = frames[0].shape[:2]
            frames = [cv2.resize(f, (round(w * bird_scale), round(h * bird_scale)),
                                 interpolation=cv2.INTER_CUBIC) for f in frames]
        self._bird_padded = [_pad_for_rotation(f) for f in frames]
        self._bird_cache: dict[tuple[int, int], Premul] = {}

        # Feathers: every (tint, angle, fade) combination built now, ~150 tiny
        # sprites, so a burst costs two OpenCV calls per particle.
        feather = cv2.resize(art.feather, None, fx=FEATHER_SCALE, fy=FEATHER_SCALE,
                             interpolation=cv2.INTER_CUBIC)
        fp = _pad_for_rotation(feather)
        fh, fw = fp.shape[:2]
        self._feathers: dict[tuple[int, int, int], Premul] = {}
        for k in range(FEATHER_ANGLES):
            m = cv2.getRotationMatrix2D((fw / 2, fh / 2), k * 360.0 / FEATHER_ANGLES, 1.0)
            rot = cv2.warpAffine(fp, m, (fw, fh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
            for ti, tint in enumerate(FEATHER_TINTS):
                # Tinting multiplies the white feather, keeping its shading.
                tinted = rot.copy()
                tinted[:, :, :3] = (rot[:, :, :3].astype(np.float32)
                                    * np.array(tint, np.float32) / 255.0).astype(np.uint8)
                for f in range(FEATHER_FADES):
                    self._feathers[(ti, k, f)] = Premul(tinted, fade=(f + 1) / FEATHER_FADES)

        # Game-over scoreboard pieces.
        self._panel = Premul(art.panel)
        self._medals = {m: Premul(cv2.resize(s, (104, 104), interpolation=cv2.INTER_CUBIC))
                        for m, s in art.medals.items()}

        # Picture-in-picture: exactly a quarter of the frame. An integer
        # downscale lets INTER_AREA take its fast path (0.05 ms; an arbitrary
        # size like 300x170 costs 7 ms).
        self.pip_w, self.pip_h = W // 4, H // 4
        x1 = W - PIP_MARGIN
        self.pip_rect = (x1 - self.pip_w, PIP_MARGIN, x1, PIP_MARGIN + self.pip_h)
        self._pip_mask = _rounded_mask(self.pip_w, self.pip_h, PIP_RADIUS) > 127
        self._pip_mask3 = np.repeat(self._pip_mask[:, :, None], 3, axis=2)

        self._shake = np.zeros(2, np.float32)
        self._rng = np.random.default_rng()

    # --- helpers ---

    def _bird_sprite(self, frame_i: int, angle_deg: float) -> Premul:
        step = int(round(angle_deg / ANGLE_STEP))
        key = (frame_i, step)
        spr = self._bird_cache.get(key)
        if spr is None:
            padded = self._bird_padded[frame_i]
            h, w = padded.shape[:2]
            # OpenCV angles are counter-clockwise; ours are + nose down (clockwise).
            m = cv2.getRotationMatrix2D((w / 2, h / 2), -step * ANGLE_STEP, 1.0)
            rot = cv2.warpAffine(padded, m, (w, h), flags=cv2.INTER_LINEAR,
                                 borderValue=(0, 0, 0, 0))
            spr = self._bird_cache[key] = Premul(rot)
        return spr

    # --- main ---

    def draw(self, frame: np.ndarray, game) -> None:
        # The camera, captured before the scene covers it.
        pip = cv2.resize(frame, (self.pip_w, self.pip_h), interpolation=cv2.INTER_AREA)

        if game.fx.shake > 0:
            self._shake = self._rng.uniform(-1, 1, 2).astype(np.float32) * 12.0 * game.fx.shake
        else:
            self._shake[:] = 0
        ox, oy = int(self._shake[0]), int(self._shake[1])

        self._draw_backdrop(frame, game)
        self._draw_pipes(frame, game, ox, oy)
        self._draw_ground(frame, game, ox)
        self._draw_feathers(frame, game, ox, oy)
        self._draw_bird(frame, game, ox, oy)
        self._draw_popups(frame, game)
        self._draw_score(frame, game)
        if game.fx.flash > 0:
            f = min(1.0, game.fx.flash) * 0.85
            cv2.addWeighted(frame, 1.0 - f, frame, 0.0, 255.0 * f, dst=frame)
        phase = game.phase.value
        if phase == "ready":
            self._draw_ready(frame, game)
        elif phase == "game_over":
            self._draw_game_over(frame, game)
        self._draw_pip(frame, pip, game)

    # --- world ---

    def _draw_backdrop(self, frame, game) -> None:
        W = self.width
        frame[: self._skyline_split] = self._sky[: self._skyline_split]
        off = int(game.distance * SKYLINE_PARALLAX)
        if self._skyline_blend is not None:
            y0, y1 = self._skyline_top, self._skyline_split
            sb = self._skyline_blend
            for d0, d1, s0 in _tiles(off, self._skyline_w, W):
                _blend(frame[y0:y1, d0:d1], sb.pre[:, s0:s0 + d1 - d0], sb.inv[:, s0:s0 + d1 - d0])
        y0, y1 = self._skyline_split, self.ground_y
        for d0, d1, s0 in _tiles(off, self._skyline_w, W):
            frame[y0:y1, d0:d1] = self._skyline_opaque[:, s0:s0 + d1 - d0]

    def _draw_ground(self, frame, game, ox: int) -> None:
        y0 = self.ground_y
        g = self._ground
        for d0, d1, s0 in _tiles(int(game.distance) - ox, g.shape[1], self.width):
            frame[y0:y0 + g.shape[0], d0:d1] = g[: self.height - y0, s0:s0 + d1 - d0]

    def _body(self, frame, x_left: int, y0: int, y1: int, down: bool) -> None:
        """Pipe body between y0 and y1, anchored at the cap so it never crawls."""
        W = self.width
        cx0, cx1 = max(0, x_left), min(W, x_left + self.pipe_w)
        y0c, y1c = max(0, y0), min(self.ground_y, y1)
        if cx1 <= cx0 or y1c <= y0c:
            return
        sx0, sx1 = cx0 - x_left, cx1 - x_left
        if down:  # bottom pipe: pattern starts under the cap
            s0 = y0c - y0
        else:  # top pipe: pattern starts above the cap, counted upward
            n = self._body_up.shape[0]
            s0 = n - (y1 - y0c)
            s0 = max(0, s0)
        n_rows = y1c - y0c
        if self._body_opaque:
            src = self._body_down if down else self._body_up
            frame[y0c:y1c, cx0:cx1] = src[s0:s0 + n_rows, sx0:sx1]
        else:
            pm = self._body_pm_down if down else self._body_pm_up
            _blend(frame[y0c:y1c, cx0:cx1], pm.pre[s0:s0 + n_rows, sx0:sx1],
                   pm.inv[s0:s0 + n_rows, sx0:sx1])

    def _draw_pipes(self, frame, game, ox: int, oy: int) -> None:
        clip = (0, 0, self.width, self.ground_y)
        cap_w = self._cap.w
        for p in game.pipes:
            cx = int(round(p.x)) + ox
            if cx + cap_w < 0 or cx - cap_w > self.width:
                continue
            bx = cx - self.pipe_w // 2
            top = int(round(p.gap_top)) + oy
            bottom = int(round(p.gap_bottom)) + oy
            # Top pipe: body down from the ceiling, flipped cap at the opening.
            self._body(frame, bx, min(0, top - self.cap_h), top - self.cap_h, down=False)
            _blit(frame, self._cap_up, cx - cap_w // 2, top - self.cap_h, clip)
            # Bottom pipe: cap at the opening, body down to the ground.
            self._body(frame, bx, bottom + self.cap_h, self.ground_y + 20, down=True)
            _blit(frame, self._cap, cx - cap_w // 2, bottom, clip)

    def _draw_bird(self, frame, game, ox: int, oy: int) -> None:
        b = game.bird
        n = len(self._bird_padded)
        # Wing cycle: down, ..., up, ... back (frames are ordered up -> down).
        cycle = [n - 1 - i for i in range(n)] + list(range(1, n - 1))
        if game.phase.value == "game_over":
            fi = min(1, n - 1)  # wings still, half-folded
        else:
            fi = cycle[int(b.wing) % len(cycle)]
        spr = self._bird_sprite(fi, b.angle)
        _blit(frame, spr, int(round(b.x - spr.w / 2)) + ox, int(round(b.y - spr.h / 2)) + oy)

    def _draw_feathers(self, frame, game, ox: int, oy: int) -> None:
        for f in game.fx.feathers:
            if f.life <= 0:
                continue
            k = int(round(f.angle / (360.0 / FEATHER_ANGLES))) % FEATHER_ANGLES
            fade = min(FEATHER_FADES - 1, int(min(1.0, f.life * 1.6) * FEATHER_FADES))
            ti = FEATHER_TINTS.index(f.colour) if f.colour in FEATHER_TINTS else 0
            spr = self._feathers[(ti, k, fade)]
            _blit(frame, spr, int(f.pos[0] - spr.w / 2) + ox, int(f.pos[1] - spr.h / 2) + oy)

    # --- HUD ---

    @staticmethod
    def _pop_scale(age: float) -> float:
        """Overshoot then settle: 0.6 -> 1.3 -> 1.0 over the first 30%."""
        if age < 0.12:
            return 0.6 + (age / 0.12) * 0.7
        if age < 0.30:
            return 1.3 - ((age - 0.12) / 0.18) * 0.3
        return 1.0

    def _draw_popups(self, frame, game) -> None:
        """Text has no alpha in OpenCV: draw on a copy of a small patch, blend back."""
        for pop in game.fx.popups:
            age = 1.0 - max(0.0, pop.life)
            scale = 1.6 * self._pop_scale(age)
            thick = 3
            (tw, th), base = cv2.getTextSize(pop.text, _FONT, scale, thick)
            x = int(pop.pos[0] - tw / 2)
            y = int(pop.pos[1])
            pad = 6
            x0, y0 = max(0, x - pad), max(0, y - th - pad)
            x1, y1 = min(self.width, x + tw + pad), min(self.height, y + base + pad)
            if x1 <= x0 or y1 <= y0:
                continue
            alpha = 1.0 if age < 0.6 else max(0.0, (1.0 - age) / 0.4)
            roi = frame[y0:y1, x0:x1]
            patch = roi.copy()
            outlined_text(patch, pop.text, (x - x0, y - y0), _FONT, scale, _GOLD, thick, width=2)
            cv2.addWeighted(patch, alpha, roi, 1.0 - alpha, 0, roi)

    def _draw_score(self, frame, game) -> None:
        if game.phase.value != "playing":
            return
        bump = game.fx.score_bump
        colour = tuple(int(c + (255 - c) * (1 - bump)) for c in _GOLD)  # flashes gold, settles white
        _centred(frame, str(game.score), self.width / 2, 104, 2.6 * (1.0 + 0.3 * bump), colour,
                 thick=5, outline=4)

    def _draw_ready(self, frame, game) -> None:
        # Not before the first update: during the arcade's 3-2-1 the game is
        # rendered but not updated, and the countdown has the centre stage.
        if game.clock <= 0.0:
            return
        W = self.width
        pulse = 1.0 + 0.06 * math.sin(game.clock * 2 * math.pi * 1.2)
        _centred(frame, "FLAP TO START", W / 2, 205, 2.1 * pulse, _GOLD, thick=4, outline=4)
        if getattr(game, "too_close", False):
            # Flaps leave the frame when standing this close; fix that first.
            _centred(frame, "STEP BACK", W / 2, 468, 1.4, (90, 200, 255), thick=3, outline=3)
            _centred(frame, "until your waist is in the camera view", W / 2, 512, 0.8,
                     _WHITE, thick=2, outline=2)
            return
        _centred(frame, "Raise both arms, then beat them down like wings", W / 2, 470, 0.9,
                 _WHITE, thick=2, outline=2)
        _centred(frame, "Each flap = one hop", W / 2, 512, 0.75, (225, 240, 255), thick=2, outline=2)

    def _draw_game_over(self, frame, game) -> None:
        W = self.width
        t = game.dead_time
        # Title drops in once the crash flash has faded.
        k = _ease_out((t - 0.25) / 0.35)
        if k > 0:
            _centred(frame, "GAME OVER", W / 2, 150 - (1 - k) * 60, 2.4, _ORANGE, thick=5, outline=4)

        # Scoreboard slides up from below.
        s = _ease_out((t - 0.55) / 0.45)
        if s <= 0:
            return
        panel = self._panel
        px = W // 2 - panel.w // 2
        py = int(190 + (1 - s) * (self.height - 190))
        _blit(frame, panel, px, py)

        # Left: the medal. Right: score (counting up) and best.
        medal = game.medal
        mx, my = px + panel.w * 0.29, py + panel.h * 0.56
        _centred(frame, "MEDAL", mx, py + 70, 0.85, _ORANGE, thick=2, outline=0)
        if medal is not None:
            spr = self._medals[medal]
            _blit(frame, spr, int(mx - spr.w / 2), int(my - spr.h / 2 + 6))
        else:
            cv2.circle(frame, (int(mx), int(my + 6)), 46, (150, 190, 215), 3, cv2.LINE_AA)

        count = _ease_out((t - 1.0) / 0.6) if s >= 1.0 else 0.0
        shown = int(round(game.score * count))
        rx = px + panel.w * 0.70
        _centred(frame, "SCORE", rx, py + 70, 0.85, _ORANGE, thick=2, outline=0)
        _centred(frame, str(shown), rx, py + 132, 1.8, _WHITE, thick=4, outline=3)
        _centred(frame, "BEST", rx, py + 182, 0.85, _ORANGE, thick=2, outline=0)
        _centred(frame, str(game.best), rx, py + 244, 1.8, _WHITE, thick=4, outline=3)

        if game.new_best and count >= 1.0:
            pop = 1.0 + 0.08 * math.sin(t * 2 * math.pi * 1.5)
            _centred(frame, "NEW BEST!", W / 2, py + panel.h + 50, 1.3 * pop, _GOLD,
                     thick=3, outline=3)

    # --- the camera window ---

    def _draw_pip(self, frame, pip: np.ndarray, game) -> None:
        x0, y0, x1, y1 = self.pip_rect
        det = game.detector
        s = self.pip_w / self.width
        visible = bool(getattr(det, "arms_visible", False))

        if not visible:
            cv2.addWeighted(pip, 0.45, pip, 0, 0, dst=pip)

        pose = getattr(game, "last_pose", None)
        if pose is not None:
            self._draw_arms(pip, pose, s, getattr(det, "wing", None))

        wing = getattr(det, "wing", None) if visible else None
        self._draw_wing_meter(pip, wing, game.fx.flap)

        if not visible and game.clock > 0.0:
            cy = self.pip_h // 2
            cv2.rectangle(pip, (8, cy - 22), (self.pip_w - 32, cy + 16), (20, 20, 20), -1)
            _centred(pip, "SHOW BOTH ARMS", self.pip_w / 2 - 12, cy + 8, 0.75,
                     (90, 200, 255), thick=2, outline=0)
        elif getattr(game, "too_close", False) and game.fx.flap <= 0:
            cy = self.pip_h - 22
            cv2.rectangle(pip, (8, cy - 22), (self.pip_w - 32, cy + 10), (20, 20, 20), -1)
            _centred(pip, "STEP BACK", self.pip_w / 2 - 12, cy + 2, 0.7,
                     (90, 200, 255), thick=2, outline=0)
        elif game.fx.flap > 0:
            age = 1.0 - game.fx.flap
            scale = 1.25 * self._pop_scale(age)
            _centred(pip, "FLAP!", self.pip_w / 2 - 10, self.pip_h / 2 + 14, scale, _GOLD,
                     thick=3, outline=3)

        roi = frame[y0:y1, x0:x1]
        np.copyto(roi, pip, where=self._pip_mask3)
        flash = game.fx.flap
        colour = tuple(int(w + (g - w) * flash) for w, g in zip((235, 235, 235), _GOLD))
        _rounded_border(frame, (x0, y0, x1 - 1, y1 - 1), PIP_RADIUS, (20, 20, 20), 6)
        _rounded_border(frame, (x0, y0, x1 - 1, y1 - 1), PIP_RADIUS, colour,
                        3 if flash < 0.3 else 4)

    def _draw_arms(self, pip, pose, s: float, wing: float | None) -> None:
        names = ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                 "left_wrist", "right_wrist")
        pts = {}
        for n in names:
            i = KP[n]
            if pose.confidence[i] >= 0.4:
                pts[n] = (int(pose.xy[i][0] * s), int(pose.xy[i][1] * s))
        k = 0.0 if wing is None else min(1.0, max(0.0, wing))
        arm = tuple(int(w + (g - w) * k) for w, g in zip((235, 235, 235), _GOLD))
        for a, b, colour in (("left_shoulder", "right_shoulder", (200, 200, 200)),
                             ("left_shoulder", "left_elbow", arm), ("left_elbow", "left_wrist", arm),
                             ("right_shoulder", "right_elbow", arm), ("right_elbow", "right_wrist", arm)):
            if a in pts and b in pts:
                cv2.line(pip, pts[a], pts[b], (20, 20, 20), 5, cv2.LINE_AA)
                cv2.line(pip, pts[a], pts[b], colour, 2, cv2.LINE_AA)
        # Shoulder to wrist directly when the elbow is missing (common up close).
        for sh, el, wr in (("left_shoulder", "left_elbow", "left_wrist"),
                           ("right_shoulder", "right_elbow", "right_wrist")):
            if sh in pts and wr in pts and el not in pts:
                cv2.line(pip, pts[sh], pts[wr], arm, 2, cv2.LINE_AA)
        for n in ("left_shoulder", "right_shoulder"):
            if n in pts:
                cv2.circle(pip, pts[n], 4, (230, 230, 230), -1, cv2.LINE_AA)
        for n in ("left_wrist", "right_wrist"):
            if n in pts:
                cv2.circle(pip, pts[n], 7, (20, 20, 20), -1, cv2.LINE_AA)
                cv2.circle(pip, pts[n], 5, arm, -1, cv2.LINE_AA)

    def _draw_wing_meter(self, pip, wing: float | None, flap: float) -> None:
        """Vertical gauge on the PiP's right edge: how high the arms are now."""
        h = self.pip_h
        x0, x1 = self.pip_w - 22, self.pip_w - 10
        y0, y1 = 14, h - 14
        cv2.rectangle(pip, (x0 - 2, y0 - 2), (x1 + 2, y1 + 2), (20, 20, 20), -1)
        if wing is not None:
            k = min(1.0, max(0.0, wing))
            top = int(y1 - (y1 - y0) * k)
            colour = _GOLD if flap > 0 else tuple(int(w + (g - w) * k) for w, g in
                                                   zip((210, 210, 210), _GOLD))
            cv2.rectangle(pip, (x0, top), (x1, y1), colour, -1)
        cv2.rectangle(pip, (x0 - 2, y0 - 2), (x1 + 2, y1 + 2), (235, 235, 235), 1)
