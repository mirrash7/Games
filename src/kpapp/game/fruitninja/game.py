"""Snack Attack - Fruit Ninja rules, with a raccoon for a hand.

The player's palm is a raccoon that gobbles flying snacks (cotton candy and
other raccoon favourites) and must stay away from bags of trash. The package
and identifiers keep their Fruit Ninja names: a "fruit" is a snack, a "bomb"
is the bag of trash, "slicing" is the raccoon taking a bite.

Scoring, lives and combos follow the arcade original: a point per fruit, a
bonus for cutting several in one swing, three misses and you are out, and a
bomb ends the run immediately.

Everything the player does comes through `Blade`, which is fed the wrist
position each frame. The game itself never looks at keypoints.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ...controls import ControlState
from ..base import Game
from .art import Art, FruitArt, load_art
from ...hand import HandTracker
from .blade import CUT_WINDOW, HIT_RADIUS, MIN_SLICE_SPEED, Blade
from .entities import POPUP_SECONDS, Fruit, Half, Popup, Spark, Splat, Spawner
from .tutorial import SNACK_TUTORIAL

COMBO_WINDOW = 0.40  # slices this close together count as one swing
COMBO_MIN = 3


class Phase(str, Enum):
    PLAYING = "playing"
    GAME_OVER = "game_over"


@dataclass
class Rules:
    lives: int = 3
    combo_window: float = COMBO_WINDOW
    combo_min: int = COMBO_MIN
    max_swing: float = 1.2  # a swing this long is settled up even if unbroken
    # How forgiving cutting is. These are the knobs if it feels too hard.
    # Blade half-width, px. Wider than the plain blade's 34: the raccoon head
    # drawn on the palm is ~100 px across, and players judge a bite by the
    # head they see, so the hit area matches it.
    hit_radius: float = 44.0
    cut_window: float = CUT_WINDOW  # seconds the trail stays sharp
    min_speed: float = MIN_SLICE_SPEED  # px/s a swing needs to cut


@dataclass
class ComboBanner:
    count: int = 0
    bonus: int = 0
    life: float = 0.0


@dataclass
class Effects:
    halves: list[Half] = field(default_factory=list)
    splats: list[Splat] = field(default_factory=list)
    sparks: list[Spark] = field(default_factory=list)
    popups: list[Popup] = field(default_factory=list)
    chomp: float = 0.0  # 1 -> 0 after a bite: the raccoon shows its open mouth
    score_bump: float = 0.0  # 1 -> 0 after the score changes, drives the HUD pulse
    flash: float = 0.0
    shake: float = 0.0
    banner: ComboBanner = field(default_factory=ComboBanner)


class FruitNinjaGame(Game):
    name = "fruitninja"
    title = "SNACK ATTACK"
    board_id = "snack"  # its name on the shared leaderboard and in the browser build
    blurb = "Your hand is a raccoon. Gobble the snacks, dodge the trash bags."
    tip = "Stand back so your upper body is in frame"
    tutorial = SNACK_TUTORIAL  # the shell's how-to-play cards

    def __init__(
        self,
        size: tuple[int, int],
        rules: Rules | None = None,
        hand: str = "right",
        mirrored: bool = True,
        art: Art | None = None,
        seed: int | None = None,
        **options: object,
    ) -> None:
        super().__init__(size, **options)
        self.rules = rules or Rules()
        # Injectable so the simulation can be exercised without generated art.
        self.art: Art = art if art is not None else load_art()
        self.seed = seed  # None: a different game every time
        self.hand = hand
        self.mirrored = mirrored
        self.tracker = HandTracker(hand, mirrored)
        self.blade = Blade(
            hit_radius=self.rules.hit_radius,
            cut_window=self.rules.cut_window,
            min_speed=self.rules.min_speed,
            max_jump=self.width * 0.35,
        )
        self.reset()

    # --- lifecycle ---

    def reset(self) -> None:
        self.phase = Phase.PLAYING
        self.score = 0
        self.lives = self.rules.lives
        self.best_combo = 0
        self.sliced_total = 0
        self.fruits: list[Fruit] = []
        self.fx = Effects()
        self.spawner = Spawner((self.width, self.height), seed=self.seed,
                               clearance=self.rules.hit_radius)
        self.blade.lose_track()
        self.tracker.reset()
        self._clock = 0.0
        self._swing: list[float] = []  # timestamps of recent slices
        self._death_reason = ""

    def on_resume(self) -> None:
        """Coming back from pause: the hand moved while the clock stood still,
        so the old trail would read as one impossibly fast swing."""
        self.blade.lose_track()
        self.tracker.reset()

    def swap_hand(self) -> None:
        """Flip which hand drives the blade, for when the mirror guess is wrong."""
        self.hand = "left" if self.hand == "right" else "right"
        self.tracker = HandTracker(self.hand, self.mirrored)
        self.blade.lose_track()

    @property
    def clock(self) -> float:
        """Seconds of gameplay elapsed; the timebase for blade samples."""
        return self._clock

    @property
    def level(self) -> int:
        return self.sliced_total // 8

    # --- simulation ---

    def update(self, controls: ControlState, dt: float) -> None:
        dt = min(dt, 0.05)  # a stall must not fling everything off-screen
        self._clock += dt

        self._track_hand(controls)
        self._decay_effects(dt)

        if self.phase is Phase.GAME_OVER:
            self._step_bodies(dt)
            return

        self.spawner.update(dt, self.fruits, self.art.fruits, self.level)
        self._slice_pass()
        self._step_bodies(dt)
        self._cull()
        self._resolve_combo()

    def _track_hand(self, controls: ControlState) -> None:
        pose = controls.pose
        if pose is None:
            self.blade.lose_track()
            self.tracker.reset()
            return
        self.blade.update(self.tracker.update(pose, self._clock), self._clock)

    def _slice_pass(self) -> None:
        """Test this frame's swept blade segment against everything airborne."""
        if not self.blade.slicing:
            return
        for fruit in self.fruits:
            if fruit.sliced:
                continue
            direction = self.blade.hits(fruit.pos, fruit.radius)
            if direction is None:
                continue
            fruit.sliced = True
            if fruit.is_bomb:
                self._explode(fruit)
                return
            self._slice_fruit(fruit, direction)

    def _slice_fruit(self, fruit: Fruit, direction: np.ndarray) -> None:
        art: FruitArt = fruit.art
        self.score += fruit.score
        self.sliced_total += 1
        self._swing.append(self._clock)

        # Halves separate along the cut normal, so the split follows the swing.
        normal = np.array([-direction[1], direction[0]], dtype=np.float32)
        for sprite, sign in ((art.half_a, 1.0), (art.half_b, -1.0)):
            self.fx.halves.append(
                Half(
                    pos=fruit.pos.copy(),
                    vel=fruit.vel + normal * sign * 190.0,
                    angle=fruit.angle,
                    spin=fruit.spin + sign * 2.6,
                    radius=fruit.radius,
                    sprite=sprite,
                )
            )

        self.fx.splats.append(Splat(fruit.pos.copy(), art.juice, scale=1.0))
        self.fx.chomp = 1.0
        self.fx.popups.append(Popup(fruit.pos.copy(), f"+{fruit.score}", art.juice,
                                    big=fruit.score > 1))
        self.fx.score_bump = 1.0
        for _ in range(8):
            v = np.random.default_rng().normal(0, 240, 2).astype(np.float32)
            self.fx.sparks.append(Spark(fruit.pos.copy(), v, art.juice))
        self.fx.flash = 1.0

    def _explode(self, bomb: Fruit) -> None:
        self.phase = Phase.GAME_OVER
        self._death_reason = "THE RACCOON GRABBED THE TRASH"
        self.fx.shake = 1.0
        self.fx.flash = 1.0
        for _ in range(60):
            v = np.random.default_rng().normal(0, 520, 2).astype(np.float32)
            self.fx.sparks.append(Spark(bomb.pos.copy(), v, (60, 60, 255)))

    def _step_bodies(self, dt: float) -> None:
        for f in self.fruits:
            f.step(dt)
        for h in self.fx.halves:
            h.step(dt)
        for s in self.fx.sparks:
            s.vel[1] += 900.0 * dt
            s.pos += s.vel * dt
            s.life -= dt * 1.5

    def _cull(self) -> None:
        floor = self.height + 110
        kept: list[Fruit] = []
        for f in self.fruits:
            if f.pos[1] <= floor:
                kept.append(f)
                continue
            # Dropped fruit costs a life; a dropped bomb is a relief, not a loss.
            if not f.sliced and not f.is_bomb:
                self.lives -= 1
                if self.lives <= 0:
                    self.phase = Phase.GAME_OVER
                    self._death_reason = "TOO MANY SNACKS GOT AWAY"
        self.fruits = kept
        self.fx.halves = [h for h in self.fx.halves if h.life > 0 and h.pos[1] < floor]
        self.fx.sparks = [s for s in self.fx.sparks if s.life > 0]

    def _resolve_combo(self) -> None:
        """A swing that cut several fruits pays a bonus once it has ended."""
        if not self._swing:
            return
        still_swinging = self._clock - self._swing[-1] < self.rules.combo_window
        # A player slicing continuously never leaves a gap, so without this the
        # swing would never close and the bonus would never be paid.
        overrun = self._clock - self._swing[0] > self.rules.max_swing
        if still_swinging and not overrun:
            return
        count = len(self._swing)
        if count >= self.rules.combo_min:
            bonus = count * 2
            self.score += bonus
            self.best_combo = max(self.best_combo, count)
            self.fx.banner = ComboBanner(count=count, bonus=bonus, life=1.0)
            self.fx.score_bump = 1.0
        self._swing.clear()

    def _decay_effects(self, dt: float) -> None:
        fx = self.fx
        fx.flash = max(0.0, fx.flash - dt * 4.0)
        fx.shake = max(0.0, fx.shake - dt * 1.6)
        fx.banner.life = max(0.0, fx.banner.life - dt * 0.7)
        fx.score_bump = max(0.0, fx.score_bump - dt * 3.5)
        fx.chomp = max(0.0, fx.chomp - dt / 0.22)  # mouth stays open ~0.2 s
        for pop in fx.popups:
            pop.life -= dt / POPUP_SECONDS
            pop.pos[1] -= 85.0 * dt * max(pop.life, 0.0)  # rise, easing to a stop
        fx.popups = [pop for pop in fx.popups if pop.life > 0][-12:]
        for s in fx.splats:
            s.life -= dt * 0.5
        # Debris is cosmetic, and on a busy board it is also the bulk of the
        # frame time, so it is capped. Oldest goes first, which is the least
        # noticeable thing to drop.
        fx.splats = [s for s in fx.splats if s.life > 0][-10:]
        fx.sparks = fx.sparks[-110:]
        fx.halves = fx.halves[-18:]

    @property
    def death_reason(self) -> str:
        return self._death_reason

    # --- rendering ---

    def render(self, frame: np.ndarray) -> None:
        from .render import FruitNinjaRenderer

        if not hasattr(self, "_renderer"):
            self._renderer = FruitNinjaRenderer((self.width, self.height), self.art)
        self._renderer.draw(frame, self)
