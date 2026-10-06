"""Drawing the Hole in the Wall scene over the live camera frame.

The compositing trick: the wall is drawn ON TOP of the camera image everywhere
except the opening, so the player sees themselves through the hole exactly the
way the TV show frames it. As the wall closes in it covers more of the picture,
until at impact only the shape itself is left.

Performance shapes most of the decisions here. A per-pixel float alpha blend
over a full 1280x720 frame costs ~10 ms, which would eat a third of the frame
budget on its own. Instead the hole is punched with a binary mask - a fast
uint8 select - and the resulting hard edge is hidden under an anti-aliased
glow rim, which is what sells it visually anyway.
"""

from __future__ import annotations

import cv2
import numpy as np

from ...config import KP
from ...overlay import outlined_text
from .assets import additive_composite, alpha_composite, load_assets
from .core import torso_frame
from .matching import BASE_CLEARANCE
from .silhouette import draw_silhouette

# Perspective: z runs 1 (far) -> 0 (impact). Larger k = more dramatic rush.
_PERSPECTIVE_K = 2.6
_WALL_OVERSCAN = 1.28  # wall is bigger than the frame at impact, so no edges show

_YELLOW = (60, 220, 250)
_CYAN = (255, 220, 90)
_WHITE = (255, 255, 255)
_RED = (80, 80, 255)


def _perspective(approach: float) -> float:
    """Screen scale of the wall at a given approach progress."""
    z = 1.0 - float(np.clip(approach, 0.0, 1.0))
    near = 1.0 / (1.0 + _PERSPECTIVE_K * z)
    far = 1.0 / (1.0 + _PERSPECTIVE_K)
    # Renormalise so approach=0 starts small and approach=1 lands exactly at 1.
    return (near - far) / (1.0 - far) * (1.0 - far) + far


class HoleInWallRenderer:
    _last_pose = None

    def __init__(self, size: tuple[int, int], cache) -> None:
        self.width, self.height = size
        self.cache = cache
        self.assets = load_assets()

        # Pre-scale everything that would otherwise be resized every frame.
        self._bg = cv2.resize(self.assets.arena_bg, size, interpolation=cv2.INTER_LINEAR)
        src = self.assets.wall_panel
        big = (int(self.width * _WALL_OVERSCAN), int(self.height * _WALL_OVERSCAN))
        self._wall_src = cv2.resize(src, big, interpolation=cv2.INTER_LINEAR)

        self._particles: list[list[float]] = []
        self._rng = np.random.default_rng(7)

    # --- geometry ---

    def _bbox(self, pts: np.ndarray, margin: float) -> tuple[int, int, int, int]:
        """Clipped pixel bounds of a point set, so blends touch only what they must."""
        lo = pts.min(axis=0) - margin
        hi = pts.max(axis=0) + margin
        x0 = int(max(0, np.floor(lo[0]))); y0 = int(max(0, np.floor(lo[1])))
        x1 = int(min(self.width, np.ceil(hi[0]))); y1 = int(min(self.height, np.ceil(hi[1])))
        return x0, y0, x1, y1

    def _hole_mask(self, game, scale: float, centre: np.ndarray) -> np.ndarray:
        """Full-frame binary mask of the opening at the current perspective."""
        target = game.fb.target
        if target is None or scale <= 2.0:
            return np.zeros((self.height, self.width), np.uint8)
        pts = self.cache.torso(target) * scale + centre
        pad = BASE_CLEARANCE * target.clearance * scale
        return draw_silhouette(pts, (self.width, self.height), thickness=1.0, pad=pad)

    # --- main entry ---

    def draw(self, frame: np.ndarray, game) -> None:
        fb = game.fb
        approach = fb.approach if fb.phase.value in ("approach", "result") else 0.0
        p = _perspective(approach)

        centre, torso_px = self._anchor_from_game(game)
        hole_scale = torso_px * p
        # Before impact the opening is further away, so it sits nearer the
        # vanishing point and slides onto the player as the wall arrives.
        screen_mid = np.array([self.width * 0.5, self.height * 0.5], np.float32)
        hole_centre = screen_mid + (centre - screen_mid) * p

        shake = self._shake_offset(fb.shake)

        if fb.phase.value in ("approach", "result"):
            self._draw_wall(frame, game, p, hole_scale, hole_centre + shake)

        if fb.phase.value in ("prep", "approach"):
            self._draw_ghost(frame, game, centre, torso_px)

        self._update_particles(frame, fb)
        self._draw_hud(frame, game)
        self._draw_verdict(frame, game)
        self._draw_game_over(frame, game)

    def set_pose(self, pose) -> None:
        """The game loop hands the renderer the pose it drew this frame."""
        self._last_pose = pose

    def _anchor_from_game(self, game) -> tuple[np.ndarray, float]:
        pose = getattr(self, "_last_pose", None)
        if pose is not None:
            frame = torso_frame(pose.xy, pose.confidence)
            if frame is not None:
                return frame
        return np.array([self.width * 0.5, self.height * 0.52], np.float32), self.height * 0.16

    def _shake_offset(self, shake: float) -> np.ndarray:
        if shake <= 0.0:
            return np.zeros(2, np.float32)
        amp = shake * 18.0
        return (self._rng.random(2).astype(np.float32) - 0.5) * 2.0 * amp

    # --- wall ---

    def _draw_wall(self, frame, game, p: float, hole_scale: float, hole_centre) -> None:
        ww = max(2, int(self.width * _WALL_OVERSCAN * p))
        wh = max(2, int(self.height * _WALL_OVERSCAN * p))
        cx, cy = self.width * 0.5, self.height * 0.5
        x0, y0 = int(cx - ww * 0.5), int(cy - wh * 0.5)
        x1, y1 = x0 + ww, y0 + wh

        # Clip to the frame; the wall overscans deliberately at impact.
        vx0, vy0 = max(0, x0), max(0, y0)
        vx1, vy1 = min(self.width, x1), min(self.height, y1)
        if vx1 <= vx0 or vy1 <= vy0:
            return

        wall = cv2.resize(self._wall_src, (ww, wh), interpolation=cv2.INTER_LINEAR)
        wall_vis = wall[vy0 - y0 : vy1 - y0, vx0 - x0 : vx1 - x0]

        hole = self._hole_mask(game, hole_scale, hole_centre)
        hole_vis = hole[vy0:vy1, vx0:vx1]

        region = frame[vy0:vy1, vx0:vx1]
        # Binary select: keep the camera inside the opening, wall everywhere else.
        np.copyto(region, wall_vis, where=(hole_vis == 0)[:, :, None])

        target = game.fb.target
        pts = self.cache.torso(target) * hole_scale + hole_centre
        margin = BASE_CLEARANCE * target.clearance * hole_scale + 8
        self._draw_rim(frame, hole, self._bbox(pts, margin))
        self._draw_wall_border(frame, (x0, y0, x1, y1))

    def _draw_rim(self, frame, hole, box) -> None:
        """Glowing edge around the opening - also hides the binary mask jaggies.

        Restricted to the opening's own bounding box: a float blend across the
        whole frame costs ~7 ms, and the rim only ever touches a fraction of it.
        """
        x0, y0, x1, y1 = box
        if x1 <= x0 or y1 <= y0:
            return
        sub = hole[y0:y1, x0:x1]
        if not sub.any():
            return
        edge = cv2.subtract(cv2.dilate(sub, np.ones((5, 5), np.uint8)), sub)
        edge = cv2.GaussianBlur(edge, (0, 0), 2.0)
        region = frame[y0:y1, x0:x1]
        glow = (edge.astype(np.float32) / 255.0)[:, :, None]
        np.copyto(region, np.clip(region + glow * np.array(_CYAN, np.float32), 0, 255).astype(np.uint8))

    def _draw_wall_border(self, frame, rect) -> None:
        x0, y0, x1, y1 = rect
        cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), (40, 40, 45), 6, cv2.LINE_AA)
        cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), (190, 190, 200), 2, cv2.LINE_AA)

    # --- guides and HUD ---

    def _draw_ghost(self, frame, game, centre, torso_px) -> None:
        """Outline of the shape to make, drawn at the player's own size."""
        target = game.fb.target
        if target is None:
            return
        pts = self.cache.torso(target) * torso_px + centre
        body = draw_silhouette(pts, (self.width, self.height), thickness=1.0)
        x0, y0, x1, y1 = self._bbox(pts, torso_px * 0.6 + 8)
        if x1 <= x0 or y1 <= y0:
            return
        sub = body[y0:y1, x0:x1]
        edge = cv2.subtract(cv2.dilate(sub, np.ones((7, 7), np.uint8)), sub)
        colour = np.array(_CYAN if game.fb.fit.passed else _WHITE, np.float32)
        alpha = (edge.astype(np.float32) / 255.0 * 0.85)[:, :, None]
        region = frame[y0:y1, x0:x1]
        np.copyto(region, np.clip(region * (1 - alpha) + alpha * colour, 0, 255).astype(np.uint8))

    def _draw_hud(self, frame, game) -> None:
        fb = game.fb
        panel = self.assets.hud_panel
        alpha_composite(frame, panel, 12, 12)

        # Panel is 448x120 at (12, 12): score left, lives right, streak beneath.
        cv2.putText(frame, "SCORE", (36, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 185), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{game.score}", (34, 92), cv2.FONT_HERSHEY_DUPLEX, 1.5, _WHITE, 2, cv2.LINE_AA)
        cv2.putText(frame, f"STREAK {game.streak}", (34, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.46, _CYAN, 1, cv2.LINE_AA)
        for i in range(game.rules.lives):
            colour = _RED if i < game.lives else (66, 66, 72)
            cv2.circle(frame, (410 - i * 30, 52), 10, colour, -1, cv2.LINE_AA)
            cv2.circle(frame, (410 - i * 30, 52), 10, (25, 25, 30), 1, cv2.LINE_AA)

        if fb.target is not None:
            label = fb.target.name
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
            outlined_text(frame, label, (self.width // 2 - tw // 2, 54),
                          cv2.FONT_HERSHEY_DUPLEX, 1.0, _YELLOW, 2, width=2)

        # Live fit meter: the player can see themselves getting warmer.
        if fb.phase.value in ("prep", "approach"):
            if fb.fit.tracked:
                self._draw_meter(frame, fb.fit.score, fb.fit.passed)
            else:
                self._banner(frame, fb.fit.problem or "NO PLAYER DETECTED",
                             self.height - 54, (70, 200, 255), 0.8)

        if fb.phase.value == "prep":
            n = str(fb.countdown)
            ring = self.assets.countdown_ring
            rx = self.width // 2 - ring.shape[1] // 2
            ry = self.height // 2 - ring.shape[0] // 2
            alpha_composite(frame, ring, rx, ry)
            (tw, th), _ = cv2.getTextSize(n, cv2.FONT_HERSHEY_DUPLEX, 3.0, 6)
            cv2.putText(frame, n, (self.width // 2 - tw // 2, self.height // 2 + th // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 3.0, _WHITE, 6, cv2.LINE_AA)

    def _draw_meter(self, frame, score: float, passed: bool) -> None:
        w, h = 260, 14
        x, y = self.width // 2 - w // 2, self.height - 46
        cv2.rectangle(frame, (x, y), (x + w, y + h), (30, 30, 34), -1, cv2.LINE_AA)
        fill = int(w * float(np.clip(score, 0.0, 1.0)))
        colour = (120, 255, 140) if passed else (80, 170, 255)
        cv2.rectangle(frame, (x, y), (x + fill, y + h), colour, -1, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 200, 210), 1, cv2.LINE_AA)

    def _banner(self, frame, text: str, y: int, colour, scale: float = 1.0) -> None:
        """Centred text with a dark backing, legible over any wall colour."""
        font = cv2.FONT_HERSHEY_DUPLEX
        (tw, th), _ = cv2.getTextSize(text, font, scale, 2)
        x = self.width // 2 - tw // 2
        cv2.rectangle(frame, (x - 16, y - th - 12), (x + tw + 16, y + 12), (18, 18, 22), -1)
        cv2.rectangle(frame, (x - 16, y - th - 12), (x + tw + 16, y + 12), colour, 2)
        cv2.putText(frame, text, (x, y), font, scale, colour, 2, cv2.LINE_AA)

    def _draw_verdict(self, frame, game) -> None:
        fb = game.fb
        if fb.phase.value != "result" or fb.result is None:
            return

        # A tracking failure is not a miss - say so instead of showing SPLASH,
        # which would blame the player for a framing problem.
        if not fb.result.tracked:
            self._banner(frame, fb.result.problem or "NO PLAYER DETECTED",
                         int(self.height * 0.42), (70, 200, 255), 1.0)
            self._banner(frame, "NO PENALTY - WALL REPLAYS",
                         int(self.height * 0.52), (200, 200, 210), 0.6)
            return

        badge = self.assets.badge_pass if fb.result.passed else self.assets.badge_fail
        bx = self.width // 2 - badge.shape[1] // 2
        by = int(self.height * 0.30)
        alpha_composite(frame, badge, bx, by)
        if fb.result.passed and not self._particles:
            self._spawn_particles(np.array([self.width * 0.5, self.height * 0.45]))

    def _draw_game_over(self, frame, game) -> None:
        if game.fb.phase.value != "game_over":
            return
        frame //= 2  # dim the scene so the summary reads
        self._banner(frame, "GAME OVER", int(self.height * 0.34), _YELLOW, 1.6)
        self._banner(frame, f"SCORE {game.score}", int(self.height * 0.46), _WHITE, 1.0)
        self._banner(frame, f"WALLS CLEARED {game.cleared}   BEST STREAK {game.best_streak}",
                     int(self.height * 0.55), (190, 190, 200), 0.6)

    # --- particles ---

    def _spawn_particles(self, origin, count: int = 28) -> None:
        ang = self._rng.random(count) * 2 * np.pi
        speed = 160 + self._rng.random(count) * 420
        for i in range(count):
            self._particles.append(
                [float(origin[0]), float(origin[1]),
                 float(np.cos(ang[i]) * speed[i]), float(np.sin(ang[i]) * speed[i]), 1.0]
            )

    def _update_particles(self, frame, fb) -> None:
        if not self._particles:
            return
        sprite = self.assets.particle
        alive = []
        for px, py, vx, vy, life in self._particles:
            life -= 0.035
            if life <= 0:
                continue
            px += vx * 0.016
            py += vy * 0.016 + 140 * 0.016  # gravity
            additive_composite(frame, sprite,
                               int(px - sprite.shape[1] // 2), int(py - sprite.shape[0] // 2),
                               gain=float(life))
            alive.append([px, py, vx, vy, life])
        self._particles = alive
