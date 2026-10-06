"""Drawing the Fruit Ninja scene over the live camera frame.

The camera is blended with the backdrop rather than hidden. In the arcade game
you aim with a finger you can see; here the blade is your hand, and if the
player cannot see where their hand is they are aiming blind. Dimming the feed
under the backdrop keeps the fruit readable while leaving the player visible.
"""

from __future__ import annotations

import cv2
import numpy as np

from ...overlay import outlined_text
from ...gfx import additive_composite, blit
from .art import Art

_WHITE = (255, 255, 255)
_CYAN = (255, 225, 120)
_GOLD = (70, 210, 255)
_RED = (70, 70, 255)

CAMERA_WEIGHT = 0.45  # how much of the live feed survives the backdrop blend


def _pad_for_rotation(sprite: np.ndarray) -> np.ndarray:
    """Grow the canvas to the diagonal so rotation never clips the corners."""
    h, w = sprite.shape[:2]
    d = int(np.ceil(np.hypot(h, w)))
    out = np.zeros((d, d, sprite.shape[2]), sprite.dtype)
    y, x = (d - h) // 2, (d - w) // 2
    out[y : y + h, x : x + w] = sprite
    return out


class FruitNinjaRenderer:
    def __init__(self, size: tuple[int, int], art: Art) -> None:
        self.width, self.height = size
        self.art = art
        self._bg = cv2.resize(art.background, size, interpolation=cv2.INTER_LINEAR)

        # Splats are soft blobs with no fine detail, and they are the single
        # most expensive sprite on a busy board purely because of their area.
        # Halving the side quarters the blend cost and is not visible in play.
        self._splat = cv2.resize(art.splat, (140, 140), interpolation=cv2.INTER_AREA)

        # Pad once at startup; rotation happens every frame on these.
        self._padded: dict[int, np.ndarray] = {}
        for fruit in art.fruits:
            for spr in (fruit.whole, fruit.half_a, fruit.half_b):
                self._padded[id(spr)] = _pad_for_rotation(spr)
        self._padded[id(art.bomb)] = _pad_for_rotation(art.bomb)
        self._rot_cache: dict[tuple[int, int], np.ndarray] = {}

    # --- helpers ---

    def _rotated(self, sprite: np.ndarray, angle: float) -> np.ndarray:
        """Rotate a padded sprite, quantised so the cache actually hits."""
        padded = self._padded.get(id(sprite))
        if padded is None:
            padded = _pad_for_rotation(sprite)
            self._padded[id(sprite)] = padded
        step = int(round(np.degrees(angle) / 6.0)) % 60  # 6-degree buckets
        key = (id(sprite), step)
        cached = self._rot_cache.get(key)
        if cached is None:
            h, w = padded.shape[:2]
            m = cv2.getRotationMatrix2D((w / 2, h / 2), -step * 6.0, 1.0)
            cached = cv2.warpAffine(padded, m, (w, h), flags=cv2.INTER_LINEAR,
                                    borderValue=(0, 0, 0, 0))
            self._rot_cache[key] = cached
        return cached

    def _blit(self, frame, sprite, centre, alpha: float = 1.0, additive: bool = False,
              tint: tuple[int, int, int] | None = None) -> None:
        if additive:
            h, w = sprite.shape[:2]
            additive_composite(frame, sprite, int(centre[0] - w / 2), int(centre[1] - h / 2),
                               gain=alpha)
        else:
            blit(frame, sprite, centre, alpha=alpha, tint=tint)

    # --- main ---

    def draw(self, frame: np.ndarray, game) -> None:
        cv2.addWeighted(frame, CAMERA_WEIGHT, self._bg, 1.0 - CAMERA_WEIGHT, 0, frame)

        self._draw_splats(frame, game)
        self._draw_halves(frame, game)
        self._draw_fruits(frame, game)
        self._draw_sparks(frame, game)
        self._draw_popups(frame, game)
        self._draw_blade(frame, game)
        self._draw_hud(frame, game)
        self._draw_banner(frame, game)
        self._draw_game_over(frame, game)

    def _draw_splats(self, frame, game) -> None:
        for s in game.fx.splats:
            self._blit(frame, self._splat, s.pos,
                       alpha=float(np.clip(s.life, 0, 1)) * 0.75, tint=s.colour)

    def _draw_halves(self, frame, game) -> None:
        for h in game.fx.halves:
            if h.sprite is None:
                continue
            self._blit(frame, self._rotated(h.sprite, h.angle), h.pos,
                       alpha=float(np.clip(h.life, 0, 1)))

    def _draw_fruits(self, frame, game) -> None:
        for f in game.fruits:
            if f.sliced:
                continue
            sprite = self.art.bomb if f.is_bomb else (f.art.whole if f.art else None)
            if sprite is None:
                continue
            self._blit(frame, self._rotated(sprite, f.angle), f.pos)

    def _draw_sparks(self, frame, game) -> None:
        for s in game.fx.sparks:
            if s.life <= 0:
                continue
            r = max(1, int(4 * s.life))
            cv2.circle(frame, (int(s.pos[0]), int(s.pos[1])), r, s.colour, -1, cv2.LINE_AA)

    def _blade_polygon(self, pts: list[np.ndarray], max_width: float) -> np.ndarray | None:
        """A single tapered ribbon along the hand path.

        Stroking each segment separately with round caps produced a lumpy
        white worm - every join bulged. Offsetting the path perpendicular into
        one polygon gives a clean blade that tapers from a point at the tail to
        full width at the hand, and fills in one call.
        """
        n = len(pts)
        if n < 2:
            return None
        left, right = [], []
        for i, p in enumerate(pts):
            nxt = pts[min(i + 1, n - 1)]
            prv = pts[max(i - 1, 0)]
            d = nxt - prv
            norm = float(np.linalg.norm(d))
            if norm < 1e-3:
                d = np.array([1.0, 0.0], np.float32)
                norm = 1.0
            perp = np.array([-d[1], d[0]], np.float32) / norm
            # Width grows toward the newest sample; the tail comes to a point.
            w = max_width * (i / (n - 1)) ** 0.8
            left.append(p + perp * w)
            right.append(p - perp * w)
        return np.array(left + right[::-1], np.int32)

    @staticmethod
    def _chaikin(pts: list[np.ndarray], rounds: int = 2) -> list[np.ndarray]:
        """Corner-cutting subdivision: turns a polyline into a smooth curve.

        Visual only - cuts are still tested on the real samples. Endpoints are
        kept so the streak still starts and ends where the hand was.
        """
        for _ in range(rounds):
            if len(pts) < 3:
                return pts
            out = [pts[0]]
            for a, b in zip(pts, pts[1:]):
                out.append(a * 0.75 + b * 0.25)
                out.append(a * 0.25 + b * 0.75)
            out.append(pts[-1])
            pts = out
        return pts

    @staticmethod
    def _pop_scale(age: float) -> float:
        """Overshoot then settle: 0.6 -> 1.35 -> 1.0 over the first 30%."""
        if age < 0.12:
            return 0.6 + (age / 0.12) * 0.75
        if age < 0.30:
            return 1.35 - ((age - 0.12) / 0.18) * 0.35
        return 1.0

    def _draw_popups(self, frame, game) -> None:
        """Floating points at each cut, in the fruit's juice colour.

        Text has no alpha in OpenCV, so each popup is drawn onto a copy of its
        own small patch and blended back - fading costs a few hundred pixels,
        not a full frame.
        """
        font = cv2.FONT_HERSHEY_DUPLEX
        for pop in game.fx.popups:
            age = 1.0 - max(0.0, pop.life)
            # Sized to read from a couple of metres back, where the player stands.
            scale = (1.9 if pop.big else 1.45) * self._pop_scale(age)
            thick = 3
            # Lighten the juice colour so dark fruit still reads on the dark board.
            colour = tuple(int(c * 0.55 + 255 * 0.45) for c in pop.colour)
            (tw, th), base = cv2.getTextSize(pop.text, font, scale, thick)
            x = int(pop.pos[0] - tw / 2)
            y = int(pop.pos[1] - 40)
            pad = 8
            x0, y0 = max(0, x - pad), max(0, y - th - pad)
            x1, y1 = min(self.width, x + tw + pad), min(self.height, y + base + pad)
            if x1 <= x0 or y1 <= y0:
                continue
            alpha = 1.0 if age < 0.6 else max(0.0, (1.0 - age) / 0.4)
            roi = frame[y0:y1, x0:x1]
            patch = roi.copy()
            outlined_text(patch, pop.text, (x - x0, y - y0), font, scale, colour, thick, width=2)
            cv2.addWeighted(patch, alpha, roi, 1.0 - alpha, 0, roi)

    def _draw_blade(self, frame, game) -> None:
        """Tapered streak along the recent hand path."""
        pts = self._chaikin(game.blade.trail(game.clock))
        if len(pts) == 1:
            self._draw_cursor(frame, game, pts[0])  # hand visible but still
            return
        if len(pts) < 2:
            return

        glow = self._blade_polygon(pts, 13.0)
        core = self._blade_polygon(pts, 7.0)
        if glow is not None:
            cv2.fillPoly(frame, [glow], (150, 140, 130), cv2.LINE_AA)
        if core is not None:
            cv2.fillPoly(frame, [core], _WHITE, cv2.LINE_AA)

        tip = pts[-1]
        if game.blade.slicing:
            self._blit(frame, self.art.flash, tip, alpha=0.5, additive=True)
        self._draw_cursor(frame, game, tip)

    def _draw_cursor(self, frame, game, tip) -> None:
        """Ring showing the blade's real hit area, so the player can aim with it."""
        r = int(game.blade.hit_radius)
        colour = _CYAN if game.blade.slicing else (200, 200, 210)
        cv2.circle(frame, tuple(tip.astype(int)), r, colour, 2, cv2.LINE_AA)
        cv2.circle(frame, tuple(tip.astype(int)), 4, colour, -1, cv2.LINE_AA)

    def _draw_hud(self, frame, game) -> None:
        # The score pulses when it changes: bigger, and flashing toward white.
        bump = game.fx.score_bump
        colour = tuple(int(g + (255 - g) * bump * 0.7) for g in _GOLD)
        outlined_text(frame, f"{game.score}", (28, 66), cv2.FONT_HERSHEY_DUPLEX,
                      1.5 * (1.0 + 0.35 * bump), colour, 2, width=2)

        icon = self.art.life_full
        step = icon.shape[1] + 8
        for i in range(game.rules.lives):
            spr = self.art.life_full if i < game.lives else self.art.life_lost
            self._blit(frame, spr, (self.width - 30 - i * step, 40))

        # Only once play has run a moment: before the first update (e.g. during
        # the arcade countdown) the game has not looked for the hand yet, and
        # "no blade tracked" would be false.
        if not game.blade.active and game.phase.value != "game_over" and game.clock > 0.5:
            self._hint(frame, "RAISE YOUR RIGHT HAND - NO BLADE TRACKED"
                       if game.hand == "right" else "RAISE YOUR LEFT HAND - NO BLADE TRACKED",
                       self.height - 40)

    def _hint(self, frame, text: str, y: int, colour=(70, 200, 255)) -> None:
        font = cv2.FONT_HERSHEY_DUPLEX
        (tw, th), _ = cv2.getTextSize(text, font, 0.7, 2)
        x = self.width // 2 - tw // 2
        cv2.rectangle(frame, (x - 14, y - th - 10), (x + tw + 14, y + 10), (18, 18, 22), -1)
        cv2.putText(frame, text, (x, y), font, 0.7, colour, 2, cv2.LINE_AA)

    def _draw_banner(self, frame, game) -> None:
        b = game.fx.banner
        if b.life <= 0 or b.count < game.rules.combo_min:
            return
        text = f"{b.count} COMBO  +{b.bonus}"
        scale = 1.2 + 0.5 * b.life
        font = cv2.FONT_HERSHEY_DUPLEX
        (tw, th), _ = cv2.getTextSize(text, font, scale, 3)
        x, y = self.width // 2 - tw // 2, int(self.height * 0.28)
        outlined_text(frame, text, (x, y), font, scale, _GOLD, 3, width=3)

    def _draw_game_over(self, frame, game) -> None:
        if game.phase.value != "game_over":
            return
        frame //= 2
        font = cv2.FONT_HERSHEY_DUPLEX
        for text, dy, scale, colour in (
            ("GAME OVER", 0.32, 1.8, _RED),
            (game.death_reason, 0.42, 0.9, _WHITE),
            (f"SCORE {game.score}", 0.53, 1.3, _GOLD),
            (f"BEST COMBO {game.best_combo}", 0.61, 0.7, (190, 190, 200)),
        ):
            (tw, _), _ = cv2.getTextSize(text, font, scale, 2)
            cv2.putText(frame, text, (self.width // 2 - tw // 2, int(self.height * dy)),
                        font, scale, colour, 2, cv2.LINE_AA)
