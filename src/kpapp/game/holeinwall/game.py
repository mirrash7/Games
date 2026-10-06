"""Hole in the Wall - match the shape or get knocked down.

A wall advances on the player with a body-shaped opening cut through it. They
have the length of the approach to contort into that shape.

One deliberate design choice runs through this: the opening is drawn locked to
the player's own torso - their position and their apparent size - rather than
pinned to the middle of the screen. Scoring is already invariant to where the
player stands and how far from the camera they are, so pinning the hole to the
screen would let a perfectly-matched pose look like a miss. Locking the hole to
the body keeps the picture and the score telling the same story, and it means
players do not have to hunt for a sweet spot in frame before they can play.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ...controls import ControlState
from ..base import Game
from .core import FitResult, TargetPose
from .matching import HoleCache, evaluate
from .poses import TARGET_POSES


class Phase(str, Enum):
    PREP = "prep"  # showing the next shape, player gets set
    APPROACH = "approach"  # wall closing in
    RESULT = "result"  # verdict on screen
    GAME_OVER = "game_over"


@dataclass
class Rules:
    prep_time: float = 1.6
    approach_time: float = 4.0  # at level 1; shrinks as levels climb
    min_approach_time: float = 1.9
    result_time: float = 1.5
    lives: int = 3
    # Speed-up per cleared wall. Small, so difficulty creeps rather than spikes.
    ramp: float = 0.14


@dataclass
class Feedback:
    """What the renderer should be showing right now."""

    phase: Phase = Phase.PREP
    target: TargetPose | None = None
    approach: float = 0.0  # 0 = far away, 1 = at the player
    fit: FitResult = field(default_factory=FitResult)
    result: FitResult | None = None  # frozen verdict during RESULT
    shake: float = 0.0  # 0..1 impact shake, decays
    flash: float = 0.0  # 0..1 pass flash, decays
    time_left: float = 0.0
    countdown: int = 0


class HoleInWallGame(Game):
    name = "holeinwall"
    title = "HOLE IN THE WALL"
    blurb = "Match the shape before the wall reaches you."
    menu = False  # replaced on the welcome screen by Flappy Bird; `--game holeinwall` still runs it

    def __init__(self, size: tuple[int, int], rules: Rules | None = None,
                 seed: int | None = None, **options: object) -> None:
        super().__init__(size, **options)
        self.rules = rules or Rules()
        self.cache = HoleCache()
        # Unseeded by default: a fixed seed meant the same shapes in the same
        # order every game.
        self._rng = random.Random(seed)
        self.reset()

    # --- lifecycle ---

    def reset(self) -> None:
        self.score = 0
        self.streak = 0
        self.best_streak = 0
        self.cleared = 0
        self.lives = self.rules.lives
        self.fb = Feedback()
        self._last_pose = None
        # Set on every PREP -> APPROACH transition, but initialised here so the
        # object is never in a state where update() can divide by a missing field.
        self._duration = self.rules.approach_time
        self._recent: list[str] = []
        self._begin_prep()

    def _pick_target(self) -> TargetPose:
        """Pick a shape, avoiding immediate repeats and ramping difficulty."""
        max_difficulty = 1 if self.cleared < 3 else (2 if self.cleared < 8 else 3)
        pool = [p for p in TARGET_POSES if p.difficulty <= max_difficulty]
        fresh = [p for p in pool if p.name not in self._recent[-3:]]
        choice = self._rng.choice(fresh or pool)
        self._recent.append(choice.name)
        return choice

    def _begin_prep(self) -> None:
        self.fb = Feedback(phase=Phase.PREP, target=self._pick_target())
        self._timer = self.rules.prep_time

    def _approach_duration(self) -> float:
        t = self.rules.approach_time - self.cleared * self.rules.ramp
        return max(self.rules.min_approach_time, t)

    # --- simulation ---

    def update(self, controls: ControlState, dt: float) -> None:
        dt = min(dt, 0.1)  # a stall must not teleport the wall into the player
        self.fb.shake = max(0.0, self.fb.shake - dt * 2.2)
        self.fb.flash = max(0.0, self.fb.flash - dt * 2.0)

        pose = controls.pose
        self._last_pose = pose
        target = self.fb.target

        # Live fit runs during PREP and APPROACH so the player can correct.
        if target is not None and self.fb.phase in (Phase.PREP, Phase.APPROACH):
            self.fb.fit = evaluate(
                pose.xy if pose else None,
                pose.confidence if pose else None,
                target,
                self.cache,
            )

        self._timer -= dt
        self.fb.time_left = max(0.0, self._timer)

        if self.fb.phase is Phase.PREP:
            self.fb.countdown = max(1, int(np.ceil(self._timer)))
            if self._timer <= 0.0:
                self.fb.phase = Phase.APPROACH
                self._timer = self._approach_duration()
                self._duration = self._timer

        elif self.fb.phase is Phase.APPROACH:
            self.fb.approach = float(np.clip(1.0 - self._timer / self._duration, 0.0, 1.0))
            if self._timer <= 0.0:
                self._impact()

        elif self.fb.phase is Phase.RESULT:
            if self._timer <= 0.0:
                if self.lives <= 0:
                    self.fb.phase = Phase.GAME_OVER
                else:
                    self._begin_prep()

        elif self.fb.phase is Phase.GAME_OVER:
            # Terminal until the player restarts; app.py binds this to "r".
            pass

    def _impact(self) -> None:
        """The wall arrives: freeze the verdict and pay out."""
        result = self.fb.fit
        self.fb.result = result
        self.fb.approach = 1.0
        self.fb.phase = Phase.RESULT
        self._timer = self.rules.result_time

        if not result.tracked:
            # The camera could not see enough of the player to judge anything.
            # Charging a life for a framing problem is punishing them for
            # something they did not do, so the wall is simply replayed.
            return

        if result.passed:
            self.streak += 1
            self.best_streak = max(self.best_streak, self.streak)
            self.cleared += 1
            base = 250 if result.grade == "PERFECT" else 100
            # Streak bonus caps so a long run cannot run away with the score.
            self.score += int(base * (1.0 + min(self.streak - 1, 9) * 0.1))
            self.fb.flash = 1.0
        else:
            self.streak = 0
            self.lives -= 1
            self.fb.shake = 1.0

    # --- rendering is delegated so the sim stays testable headlessly ---

    def render(self, frame: np.ndarray) -> None:
        from .render import HoleInWallRenderer

        if not hasattr(self, "_renderer"):
            self._renderer = HoleInWallRenderer((self.width, self.height), self.cache)
        self._renderer.set_pose(self._last_pose)
        self._renderer.draw(frame, self)


