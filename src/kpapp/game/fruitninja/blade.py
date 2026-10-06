"""Turning a tracked hand into a blade, and deciding what it cut.

Three things make cutting feel fair rather than fussy:

**Swept segments.** Poses arrive at ~15-30 Hz, so a fast hand jumps a long way
between samples. Testing only where the blade is *now* would miss most cuts,
and miss hardest on the fastest swings. Every segment swept between samples is
tested instead, so nothing can tunnel through a fruit.

**A blade with width, and a trail that still cuts.** The blade is a thick
stroke rather than a line (`hit_radius`), and the recent part of the trail
stays sharp for `cut_window` seconds, so a fruit that flies into the streak a
moment after the swing is still cut. Both are the knobs to turn if hitting
feels too hard or too easy.

**Speed measured over a window.** Inference updates the hand in steps while
the game runs at 60 Hz, so frame-to-frame speed flickers between near-zero and
huge. Gating cuts on that would randomly ignore real swings; a short window
average does not.

A slice still requires real speed, otherwise resting a hand on a fruit would
quietly dissolve the board. And an impossible jump is treated as the tracker
losing the hand, not as a swing - on recorded play a glitch drew a stroke the
full height of the frame, which would otherwise have cut everything on it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from ...hand import hand_keypoint

# Re-exported: tests and older callers know it by this name.
blade_keypoint = hand_keypoint

MIN_SLICE_SPEED = 260.0  # px/s, over SPEED_WINDOW
TRAIL_SECONDS = 0.22  # how long the drawn streak lingers
CUT_WINDOW = 0.15  # how long the streak stays sharp enough to cut
HIT_RADIUS = 34.0  # blade half-width in px, added to each fruit's radius
SPEED_WINDOW = 0.08
MAX_JUMP = 450.0  # px between consecutive samples; beyond this it's a glitch


@dataclass
class Sample:
    pos: np.ndarray
    t: float
    speed: float = 0.0  # windowed hand speed when this sample arrived


class Blade:
    """Recent hand path, and which of it is sharp."""

    def __init__(
        self,
        trail_seconds: float = TRAIL_SECONDS,
        min_speed: float = MIN_SLICE_SPEED,
        hit_radius: float = HIT_RADIUS,
        cut_window: float = CUT_WINDOW,
        max_jump: float = MAX_JUMP,
    ) -> None:
        self.trail_seconds = trail_seconds
        self.min_speed = min_speed
        self.hit_radius = hit_radius
        self.cut_window = cut_window
        self.max_jump = max_jump
        self._samples: deque[Sample] = deque(maxlen=64)
        self.speed = 0.0
        self.active = False  # is the hand visible at all
        self.slicing = False  # visible AND moving fast enough to cut

    @property
    def pos(self) -> np.ndarray | None:
        return self._samples[-1].pos if self._samples else None

    @property
    def segment(self) -> tuple[np.ndarray, np.ndarray] | None:
        """The path swept since the previous sample, or None."""
        if len(self._samples) < 2:
            return None
        return self._samples[-2].pos, self._samples[-1].pos

    def trail(self, now: float) -> list[np.ndarray]:
        """Recent points, newest last, for drawing the streak."""
        cutoff = now - self.trail_seconds
        return [s.pos for s in self._samples if s.t >= cutoff]

    def lose_track(self) -> None:
        """Hand went missing: break the trail so it does not draw a long jump."""
        self._samples.clear()
        self.speed = 0.0
        self.active = False
        self.slicing = False

    def _window_speed(self, now: float) -> float:
        """Path length over the last SPEED_WINDOW, divided by the time it spans."""
        samples = list(self._samples)
        length, start = 0.0, None
        for a, b in zip(samples, samples[1:]):
            if b.t < now - SPEED_WINDOW:
                continue
            length += float(np.linalg.norm(b.pos - a.pos))
            start = a.t if start is None else start
        if start is None or now <= start:
            return 0.0
        return length / (now - start)

    def update(self, pos: np.ndarray | None, t: float) -> None:
        if pos is None:
            self.lose_track()
            return

        pos = np.asarray(pos, dtype=np.float32)
        if self._samples:
            prev = self._samples[-1]
            if t - prev.t <= 0.0:
                return
            if float(np.linalg.norm(pos - prev.pos)) > self.max_jump:
                # The tracker lost the hand and found it (or a phantom) far
                # away. Start a fresh stroke instead of drawing - and cutting
                # along - a line across the screen.
                self._samples.clear()

        self._samples.append(Sample(pos, t))
        while len(self._samples) > 2 and self._samples[0].t < t - self.trail_seconds:
            self._samples.popleft()

        self.speed = self._window_speed(t)
        self._samples[-1].speed = self.speed
        self.active = True
        self.slicing = self.speed >= self.min_speed

    # --- collision ---

    def hits(self, centre: np.ndarray, radius: float) -> np.ndarray | None:
        """Cut direction if any sharp part of the recent trail crossed the circle.

        Tests every segment within `cut_window`, newest first, each widened by
        `hit_radius`. Returns the unit direction of the cutting segment (used
        to split the halves), or None.
        """
        samples = list(self._samples)
        if len(samples) < 2:
            return None
        now = samples[-1].t
        reach = radius + self.hit_radius
        for a, b in zip(reversed(samples[:-1]), reversed(samples[1:])):
            if b.t < now - self.cut_window:
                break
            if b.speed < self.min_speed:
                continue
            d = b.pos - a.pos
            length_sq = float(d @ d)
            if length_sq < 1e-6:
                continue
            u = float(np.clip((centre - a.pos) @ d / length_sq, 0.0, 1.0))
            if float(np.linalg.norm(a.pos + d * u - centre)) <= reach:
                return (d / np.sqrt(length_sq)).astype(np.float32)
        return None
