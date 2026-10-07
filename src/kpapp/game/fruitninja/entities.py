"""Tossed objects and their physics."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from .art import FruitArt

GRAVITY = 1500.0  # px/s^2


@dataclass
class Body:
    pos: np.ndarray
    vel: np.ndarray
    angle: float = 0.0
    spin: float = 0.0  # radians/s
    radius: float = 50.0

    def step(self, dt: float, gravity: float = GRAVITY) -> None:
        self.vel[1] += gravity * dt
        self.pos += self.vel * dt
        self.angle += self.spin * dt


@dataclass
class Fruit(Body):
    art: FruitArt | None = None
    is_bomb: bool = False
    sliced: bool = False

    @property
    def score(self) -> int:
        return self.art.score if self.art else 1


@dataclass
class Half(Body):
    sprite: np.ndarray | None = None
    life: float = 1.0

    def step(self, dt: float, gravity: float = GRAVITY) -> None:
        super().step(dt, gravity)
        # Halves fade rather than vanish, so a slice reads as a follow-through.
        self.life = max(0.0, self.life - dt * 0.55)


@dataclass
class Splat:
    pos: np.ndarray
    colour: tuple[int, int, int]
    scale: float
    life: float = 1.0


@dataclass
class Popup:
    """Floating "+1" at the point of a cut: pops in, rises, fades."""

    pos: np.ndarray
    text: str
    colour: tuple[int, int, int]
    big: bool = False
    life: float = 1.0  # 1 -> 0 over POPUP_SECONDS


POPUP_SECONDS = 0.9


@dataclass
class Spark:
    pos: np.ndarray
    vel: np.ndarray
    colour: tuple[int, int, int]
    life: float = 1.0


class Spawner:
    """Tosses waves of fruit up from below the bottom edge.

    Launch speed is solved backwards from a target apex so fruit always peaks
    inside the play area: picking a velocity directly makes objects either
    barely clear the edge or fly off the top, and the difference is invisible
    until you watch it.

    Every game is different: the spawner used to run on a fixed seed, so each
    round threw the same fruit to the same places. A seed is now only used when
    one is passed (tests do, for reproducibility).

    Bombs are only thrown where they can be avoided. Fruit and bombs fall under
    the same gravity, so whole flight paths can be projected at launch; a bomb
    is placed only if it stays at least `separation` from every fruit for as
    long as both are on screen. If no safe path is found it is not thrown -
    a missing bomb is better than a fruit that cannot be cut without one.
    """

    TRIES = 24
    HORIZON = 3.2  # seconds of flight to check
    # On-screen bounds, generous at the edges where objects enter and leave.
    X_MIN, Y_MIN, Y_PAD, X_PAD = -40.0, -60.0, 10.0, 40.0

    def __init__(self, size: tuple[int, int], seed: int | None = None, clearance: float = 34.0) -> None:
        self.width, self.height = size
        self.rng = random.Random(seed)  # None -> fresh entropy each game
        self.clearance = clearance  # the blade's half-width
        self.timer = 1.6  # a beat after the start before anything flies
        self.wave = 0
        self.bombs_dropped = 0  # bombs skipped because no safe path existed

    def separation(self, a: Fruit, b: Fruit) -> float:
        """Centre distance at which a fruit can be cut without touching a bomb.

        A swing through the fruit's centre, at right angles to the bomb, clears
        it when the distance exceeds bomb radius + blade half-width. The fruit
        radius and a margin on top keep that comfortable rather than surgical.
        """
        return a.radius + b.radius + self.clearance + 30.0

    def _on_screen(self, body: Fruit) -> list[tuple[float, float]]:
        """Time intervals (from now) during which a body is on screen.

        Solved, not sampled: x(t) is linear and y(t) a parabola, so each bound
        is a root. Sampling had a real gap - at 1/30 s it missed a fruit rising
        past a falling bomb at the bottom edge (closing at ~2700 px/s); fine
        enough sampling to be safe cost up to 58 ms per wave.
        """
        x0, y0 = float(body.pos[0]), float(body.pos[1])
        vx, vy = float(body.vel[0]), float(body.vel[1])
        g = GRAVITY

        def quad_below(limit: float) -> tuple[float, float] | None:
            """Interval where y(t) < limit (y opens upward under gravity)."""
            c = y0 - limit
            disc = vy * vy - 2.0 * g * c
            if disc <= 0.0:
                return None
            r = np.sqrt(disc)
            return (-vy - r) / g, (-vy + r) / g

        lo, hi = 0.0, self.HORIZON
        below_bottom = quad_below(self.height + self.Y_PAD)
        if below_bottom is None:
            return []
        lo, hi = max(lo, below_bottom[0]), min(hi, below_bottom[1])

        if abs(vx) > 1e-9:
            ta = (self.X_MIN - x0) / vx
            tb = (self.width + self.X_PAD - x0) / vx
            lo, hi = max(lo, min(ta, tb)), min(hi, max(ta, tb))
        elif not (self.X_MIN < x0 < self.width + self.X_PAD):
            return []
        if lo >= hi:
            return []

        # Above the top edge: invisible between these roots, if it gets that high.
        above_top = quad_below(self.Y_MIN)
        if above_top is None:
            return [(lo, hi)]
        out = []
        if lo < above_top[0]:
            out.append((lo, min(hi, above_top[0])))
        if hi > above_top[1]:
            out.append((max(lo, above_top[1]), hi))
        return [iv for iv in out if iv[0] < iv[1]]

    def min_gap(self, a: Fruit, b: Fruit) -> float:
        """Exact closest approach while both are on screen (inf if never).

        Both fall under the same gravity, so it cancels: their separation moves
        in a straight line, and its closest point over each shared on-screen
        interval is a clamp.
        """
        r0 = (a.pos - b.pos).astype(np.float64)
        v = (a.vel - b.vel).astype(np.float64)
        vv = float(v @ v)
        best = float("inf")
        for a0, a1 in self._on_screen(a):
            for b0, b1 in self._on_screen(b):
                t0, t1 = max(a0, b0), min(a1, b1)
                if t0 >= t1:
                    continue
                t = 0.0 if vv < 1e-12 else -float(r0 @ v) / vv
                t = min(max(t, t0), t1)
                best = min(best, float(np.linalg.norm(r0 + v * t)))
        return best

    def _clearance_margin(self, body: Fruit, others: list[Fruit]) -> float:
        """How far the tightest pairing is from being unsafe (negative = unsafe)."""
        return min((self.min_gap(body, o) - self.separation(body, o) for o in others),
                   default=float("inf"))

    def _launch(self, art: FruitArt | None, is_bomb: bool, delay_rows: int = 0) -> Fruit:
        w, h = self.width, self.height
        x = self.rng.uniform(w * 0.12, w * 0.88)
        apex = self.rng.uniform(h * 0.10, h * 0.42)
        rise = (h + 60) - apex
        vy = -float(np.sqrt(2.0 * GRAVITY * rise))
        # Drift back toward the middle so nothing exits sideways immediately.
        vx = (w * 0.5 - x) * self.rng.uniform(0.25, 0.75) + self.rng.uniform(-90, 90)
        radius = (art.radius if art else 52)
        return Fruit(
            # Stagger a wave slightly so they do not leave as one clump.
            pos=np.array([x, h + 60.0 + delay_rows * 26.0], dtype=np.float32),
            vel=np.array([vx, vy], dtype=np.float32),
            angle=self.rng.uniform(0, 6.28),
            spin=self.rng.uniform(-3.0, 3.0),
            radius=float(radius),
            art=art,
            is_bomb=is_bomb,
        )

    def update(self, dt: float, fruits: list[Fruit], art_list: list[FruitArt], level: int) -> None:
        self.timer -= dt
        if self.timer > 0.0:
            return

        if not art_list:
            raise RuntimeError(
                "no fruit in the manifest - run tools/generate_fruitninja_snacks.py"
            )
        self.wave += 1
        # Recorded play ended three rounds in about three seconds each: the
        # opening waves of two to five fruit were too much while a player is
        # still finding the blade. Start with one or two, ramp up, and keep
        # bombs out of the first few waves entirely.
        if level < 2:
            count = 1 + self.rng.randint(0, 1)
        else:
            count = min(5, 2 + self.rng.randint(0, min(3, level // 2)))
        bomb_chance = 0.0 if self.wave <= 4 else min(0.3, 0.05 + level * 0.025)

        kinds = [self.rng.random() < bomb_chance for _ in range(count)]
        if all(kinds):
            kinds[0] = False  # a wave is never all bombs
        airborne = [f for f in fruits if not f.sliced]
        live_fruit = [f for f in airborne if not f.is_bomb]
        live_bombs = [f for f in airborne if f.is_bomb]

        # Fruit first, kept clear of any bomb already in the air. A fruit with
        # no safe path sits this wave out rather than spawn beside a bomb.
        new_fruit: list[Fruit] = []
        for row, is_bomb in enumerate(kinds):
            if is_bomb:
                continue
            art = self.rng.choice(art_list)
            for _ in range(self.TRIES if live_bombs else 1):
                cand = self._launch(art, False, row)
                if self._clearance_margin(cand, live_bombs) >= 0.0:
                    new_fruit.append(cand)
                    break
        fruits.extend(new_fruit)

        # Then bombs, which are optional: only thrown on a provably safe path.
        targets = live_fruit + new_fruit
        for row, is_bomb in enumerate(kinds):
            if not is_bomb:
                continue
            for _ in range(self.TRIES):
                cand = self._launch(None, True, row)
                if self._clearance_margin(cand, targets) >= 0.0:
                    fruits.append(cand)
                    break
            else:
                self.bombs_dropped += 1

        gap = max(0.8, 2.2 - level * 0.09)
        self.timer = self.rng.uniform(gap, gap + 0.7)
