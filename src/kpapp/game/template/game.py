"""Touch Targets - the reference starting point for a new game.

Targets appear; hold your palm on one to score it. Small on purpose: it shows
every contract a game needs (menu metadata, phase/game_over, reset, on_resume,
dt clamping, unseeded RNG, palm input, drawing over the camera frame) and
nothing else. To start a new game, copy this package (see AGENTS.md).
It is registered but kept off the welcome screen; run it with
`uv run kp game --game template`. tests/test_template.py keeps it working.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from ...controls import ControlState
from ...hand import HandTracker
from ... import theme
from ..base import Game


class Phase(str, Enum):
    PLAYING = "playing"
    GAME_OVER = "game_over"  # the shell watches for exactly this value


@dataclass
class Target:
    pos: np.ndarray
    radius: float = 46.0
    held: float = 0.0  # seconds the palm has been on it


class TouchTargetsGame(Game):
    name = "touchtargets"
    title = "TOUCH TARGETS"  # shown on the welcome screen
    blurb = "Hold your palm on each target before time runs out."
    menu = False  # a reference implementation, not part of the arcade line-up

    ROUND_SECONDS = 30.0
    HOLD_SECONDS = 0.35  # a brief hold, so a hand sweeping past doesn't score

    def __init__(self, size, mirrored: bool = True, seed: int | None = None, **options):
        super().__init__(size, **options)
        self.mirrored = mirrored
        self.rng = random.Random(seed)  # unseeded by default: every game differs
        self.tracker = HandTracker("right", mirrored)
        self.reset()

    def reset(self) -> None:
        self.phase = Phase.PLAYING
        self.score = 0
        self.clock = 0.0
        self.time_left = self.ROUND_SECONDS
        self.palm: np.ndarray | None = None
        self.target = self._new_target()
        self.tracker.reset()

    def on_resume(self) -> None:
        self.tracker.reset()  # the hand moved while paused; drop smoothing history

    def _new_target(self) -> Target:
        m = 120  # keep clear of the edges, the HUD and the shell's PAUSE button
        return Target(np.array([self.rng.uniform(m, self.width - m),
                                self.rng.uniform(m, self.height - 2 * m)], np.float32))

    def update(self, controls: ControlState, dt: float) -> None:
        dt = min(dt, 0.1)  # a stalled frame must not skip the game forward
        self.clock += dt
        self.palm = self.tracker.update(controls.pose, self.clock)
        if self.phase is Phase.GAME_OVER:
            return
        self.time_left -= dt
        if self.time_left <= 0:
            self.phase = Phase.GAME_OVER
            return
        t = self.target
        on = self.palm is not None and np.linalg.norm(self.palm - t.pos) <= t.radius + 20
        t.held = t.held + dt if on else 0.0
        if t.held >= self.HOLD_SECONDS:
            self.score += 1
            self.target = self._new_target()

    def render(self, frame: np.ndarray) -> None:
        # `frame` is the live, mirrored camera image. Draw the game on top of it.
        cv2.addWeighted(frame, 0.55, np.zeros_like(frame), 0.45, 0, frame)
        t = self.target
        c = (int(t.pos[0]), int(t.pos[1]))
        cv2.circle(frame, c, int(t.radius), theme.ACCENT, -1, cv2.LINE_AA)
        if t.held > 0:  # fill-up ring shows the hold progress
            cv2.ellipse(frame, c, (int(t.radius) + 8,) * 2, -90, 0,
                        360 * min(1.0, t.held / self.HOLD_SECONDS), (255, 255, 255), 4, cv2.LINE_AA)
        if self.palm is not None:
            cv2.circle(frame, (int(self.palm[0]), int(self.palm[1])), 14, (255, 255, 255), 2, cv2.LINE_AA)
        # House style (kpapp.theme): gold outlined score top-left, white timer.
        theme.centred_text(frame, f"{self.score}", 60, 66, 1.5, theme.ACCENT, 2, theme.TITLE_FONT)
        theme.centred_text(frame, f"{max(0.0, self.time_left):4.1f}s", self.width - 110, 66, 1.2,
                           theme.WHITE, 2, theme.TITLE_FONT)
        if self.phase is Phase.GAME_OVER:
            theme.panel(frame, (self.width // 2 - 300, self.height // 2 - 70,
                                self.width // 2 + 300, self.height // 2 + 40), border=theme.ACCENT)
            theme.centred_text(frame, f"TIME!  SCORE {self.score}", self.width // 2,
                               self.height // 2, 1.6, theme.ACCENT, 3, theme.TITLE_FONT)
