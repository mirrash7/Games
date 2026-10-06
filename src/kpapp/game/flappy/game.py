"""Flappy Bird, flown by flapping your arms.

The rules are the classic ones: one input (a flap) that sets the bird's
vertical speed, gravity the rest of the time, a point for every pipe passed,
and any touch of a pipe or the ground ends the run. The first flap starts play.

What is *not* classic is the tuning. The original is tuned for a thumb on a
touchscreen: ~20 ms of input lag and taps as fast as you like. Here a flap is a
whole-arm motion seen by a camera and a pose model running at 15-30 Hz, which
adds ~100-150 ms before the game even hears about it, and nobody can flap their
arms much faster than about twice a second. With the original numbers the bird
falls a third of the gap in that lag alone. Every number below is chosen for
that input, and `tests/test_flappy.py` flies a bot with a 150 ms handicap and a
human flap rate through thousands of pipes to prove no level is impossible.

The game never looks at keypoints: `FlapDetector` turns poses into flap events,
and the game only reacts to those.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ...controls import ControlState
from ..base import Game
from .art import FlappyArt, load_art


class Phase(str, Enum):
    READY = "ready"
    PLAYING = "playing"
    GAME_OVER = "game_over"


@dataclass
class Rules:
    """Every tuning knob, in px and seconds at 1280x720 and 60 FPS.

    Vertical numbers are written for a 720 px tall frame and scaled by the real
    height at construction, so the feel is the same at other resolutions.
    """

    # --- the bird ---
    # Gravity about half the original's (scaled to 720p it is ~2300 px/s^2).
    # Halving it doubles the time the bird takes to fall any distance, which is
    # exactly what absorbs the input lag: in 150 ms the bird now drops at most
    # ~80 px instead of ~150, against a gap of 175-230.
    gravity: float = 1150.0
    # A flap *sets* vertical speed (classic), it does not add to it, so a flap
    # always feels the same however the bird was moving. 410 px/s gives an
    # apex 73 px above the flap point (v^2 / 2g) reached in 0.36 s, and a hover
    # cycle (flap, rise, fall back to the same height) of 2v/g = 0.71 s, i.e.
    # ~1.4 flaps a second to hold altitude: a comfortable arm rhythm. A single
    # flap rising 73 px is well under half the smallest gap, so one flap from
    # the bottom of a gap can never carry the bird into the top pipe.
    flap_velocity: float = 410.0
    # Capped fall speed. Without it a long dive reaches 900+ px/s and the lag
    # alone costs 135 px. 540 px/s is reached ~0.47 s after the apex, so short
    # hops feel purely ballistic and only real dives hit the cap.
    terminal_velocity: float = 540.0
    # Nose angle (degrees, + is nose down) follows vertical speed: pitched up
    # while climbing, diving once the bird falls faster than `dive_after`.
    nose_up: float = -24.0
    nose_down: float = 75.0  # not the original's 90: a bird pointing straight down reads as dead
    dive_after: float = 180.0  # px/s of fall before the nose starts to drop
    nose_drop_rate: float = 320.0  # deg/s; the nose snaps up on a flap, sinks smoothly
    # Fraction of width: ~970 px of look-ahead (4.8 s at the start). Also keeps
    # the bird, once it has fallen dead to the ground, clear of the arcade's
    # PLAY AGAIN button, which starts at 28% of the width.
    bird_x: float = 0.24
    # The generated bird is 72x56; at 1.25x (90x70, hit radius 19) it still
    # reads from two metres back. The hit circle scales with it and stays
    # well inside the sprite, so grazes forgive.
    bird_scale: float = 1.25

    # --- the pipes ---
    # The gap starts at 230 px (32% of 720) and loses 2 px a point down to
    # 175 px (24%) at score ~28. The original's gap is ~25% of play height
    # from the first pipe; we only get there as the player warms up.
    gap_start: float = 230.0
    gap_min: float = 175.0
    gap_shrink: float = 2.0  # px per point
    # Horizontal speed: 200 px/s, +2 px/s per point up to 260 at score 30.
    # Slow enough that a pipe is on screen for ~4.6 s before it reaches the bird.
    speed_start: float = 200.0
    speed_max: float = 260.0
    speed_ramp: float = 2.0
    # Time between consecutive pipes reaching the bird: 2.0 s at the start
    # (400 px apart), shortening 0.015 s a point to 1.6 s (416 px at 260 px/s).
    # The original is ~1.3 s; a player needs time to see the next gap, decide,
    # and get a full-body motion through the camera.
    interval_start: float = 2.0
    interval_min: float = 1.6
    interval_ramp: float = 0.015
    # Gaps stay away from the ceiling and ground: a gap hugging either needs
    # pinpoint flaps, and one at the ground punishes the lag the hardest.
    ceiling_margin: float = 80.0
    ground_margin: float = 70.0
    # How fast a human can flap their arms, sustained. Consecutive gaps are
    # never further apart vertically than this rate can climb (with a safety
    # factor) in the time between leaving one pipe and entering the next.
    flap_period: float = 0.40
    reach_safety: float = 0.80
    max_drop: float = 230.0  # cap on a downward step; diving is easy, but a cliff reads as unfair
    first_pipe_delay: float = 4.0  # seconds of open sky after the first flap

    # --- feel ---
    max_dt: float = 0.05  # a stalled frame advances at most this much
    substep: float = 1.0 / 120.0  # physics step; 4 px a step at worst, so nothing tunnels
    ready_bob: float = 10.0  # px of idle bobbing on the READY screen

    def scaled(self, height: int) -> "Rules":
        """The same rules for a frame `height` px tall."""
        k = height / 720.0
        if abs(k - 1.0) < 1e-9:
            return self
        out = Rules(**self.__dict__)
        for name in ("gravity", "flap_velocity", "terminal_velocity", "dive_after", "gap_start",
                     "gap_min", "gap_shrink", "ceiling_margin", "ground_margin", "max_drop",
                     "ready_bob"):
            setattr(out, name, getattr(self, name) * k)
        return out


MEDALS = (("platinum", 40), ("gold", 30), ("silver", 20), ("bronze", 10))


def medal_for(score: int) -> str | None:
    for name, need in MEDALS:
        if score >= need:
            return name
    return None


@dataclass
class Bird:
    x: float
    y: float
    vy: float = 0.0
    angle: float = 0.0  # degrees, + is nose down
    wing: float = 0.0  # animation phase, in frames of the flap cycle
    wing_boost: float = 0.0  # 1 -> 0 after a flap: wings beat faster


@dataclass
class Pipe:
    x: float  # centre
    gap_y: float  # centre of the opening
    gap: float  # height of the opening
    scored: bool = False

    @property
    def gap_top(self) -> float:
        return self.gap_y - self.gap / 2

    @property
    def gap_bottom(self) -> float:
        return self.gap_y + self.gap / 2


@dataclass
class Feather:
    pos: np.ndarray
    vel: np.ndarray
    angle: float
    spin: float
    colour: tuple[int, int, int]
    life: float = 1.0


@dataclass
class Popup:
    """Floating "+1" where a pipe was passed: pops in, rises, fades."""

    pos: np.ndarray
    text: str
    life: float = 1.0  # 1 -> 0 over POPUP_SECONDS


POPUP_SECONDS = 0.8
FEATHER_SECONDS = 0.75
# BGR, the bird's own colours: white, pink, coral. Cream and gold vanished
# against the clouds and the sun.
FEATHER_TINTS = ((255, 255, 255), (150, 110, 255), (80, 95, 240))


@dataclass
class Effects:
    feathers: list[Feather] = field(default_factory=list)
    popups: list[Popup] = field(default_factory=list)
    score_bump: float = 0.0  # 1 -> 0 after the score changes, drives the HUD pulse
    flash: float = 0.0  # white crash flash
    shake: float = 0.0
    flap: float = 0.0  # 1 -> 0 after any recognised flap: the PiP "FLAP!" flash
    flap_strength: float = 0.0


def _circle_hits_rect(cx: float, cy: float, r: float,
                      x0: float, y0: float, x1: float, y1: float) -> bool:
    """Exact circle-rectangle overlap: distance to the closest point of the rect."""
    nx = min(max(cx, x0), x1)
    ny = min(max(cy, y0), y1)
    return (cx - nx) ** 2 + (cy - ny) ** 2 < r * r


class FlappyGame(Game):
    name = "flappy"
    title = "FLAPPY BIRD"
    blurb = "Flap your arms to fly. Don't touch the pipes."

    # Session best. The arcade builds a fresh game for every PLAY AGAIN, so a
    # best kept on the instance would be forgotten on every restart; on the
    # class it lasts for the whole session, as the player expects.
    session_best: int = 0

    def __init__(
        self,
        size: tuple[int, int],
        rules: Rules | None = None,
        mirrored: bool = True,
        art: FlappyArt | None = None,
        seed: int | None = None,
        detector=None,
        **options: object,
    ) -> None:
        super().__init__(size, **options)
        self.rules = (rules or Rules()).scaled(self.height)
        # Injectable so the simulation can be exercised without generated art
        # or a camera.
        self.art: FlappyArt = art if art is not None else load_art()
        if detector is None:
            from .gesture import FlapDetector

            detector = FlapDetector(mirrored)
        self.detector = detector
        self.mirrored = mirrored
        self.seed = seed  # None: a different course every game
        self.pipe_width = float(self.art.pipe_body.shape[1])
        self.hit_radius = float(self.art.bird_hit_radius) * self.rules.bird_scale
        self.ground_y = float(self.height - self.art.ground_height)
        self.reset()

    # --- lifecycle ---

    def reset(self) -> None:
        r = self.rules
        self.phase = Phase.READY
        self.too_close = False  # player too near the camera for flaps to be seen
        self.room_below: float | None = None
        self.score = 0
        self.new_best = False
        self.rng = random.Random(self.seed)
        self.bird = Bird(x=self.width * r.bird_x, y=self._ready_y())
        self.pipes: list[Pipe] = []
        self.fx = Effects()
        self.distance = 0.0  # world scroll in px; drives ground and skyline parallax
        self._clock = 0.0
        self.dead_time = 0.0  # seconds since the crash
        self.flaps = 0
        self.last_pose = None  # for the camera window's arm overlay
        self.detector.reset()

    def on_resume(self) -> None:
        """The arms kept moving while the clock stood still; without a reset
        the first frame back would compare against a pose from before the
        pause and could read as a flap."""
        self.detector.reset()

    @property
    def best(self) -> int:
        return max(type(self).session_best, self.score)

    @property
    def medal(self) -> str | None:
        return medal_for(self.score)

    @property
    def clock(self) -> float:
        return self._clock

    def _ready_y(self) -> float:
        return self.ground_y * 0.46

    # --- difficulty ---

    def gap_size(self, score: int | None = None) -> float:
        r = self.rules
        s = self.score if score is None else score
        return max(r.gap_min, r.gap_start - r.gap_shrink * s)

    def pipe_speed(self, score: int | None = None) -> float:
        r = self.rules
        s = self.score if score is None else score
        return min(r.speed_max, r.speed_start + r.speed_ramp * s)

    def pipe_interval(self, score: int | None = None) -> float:
        r = self.rules
        s = self.score if score is None else score
        return max(r.interval_min, r.interval_start - r.interval_ramp * s)

    @property
    def speed(self) -> float:
        return self.pipe_speed() if self.phase is Phase.PLAYING else 0.0

    def climb_budget(self, spacing: float, speed: float) -> float:
        """Highest the next gap may sit above the last one.

        Between leaving one pipe and entering the next the bird covers the
        spacing minus a pipe width minus its own diameter. Flapping at the
        human rate `flap_period` climbs v*P - g*P^2/2 per flap; the budget is
        that rate over the transit time, times a safety factor. The reach
        test verifies it with lag on top.
        """
        r = self.rules
        transit = max(0.0, spacing - self.pipe_width - 2 * self.hit_radius) / max(speed, 1e-6)
        p = r.flap_period
        per_flap = r.flap_velocity * p - 0.5 * r.gravity * p * p
        return max(0.0, r.reach_safety * per_flap * transit / p)

    def _gap_bounds(self, gap: float) -> tuple[float, float]:
        r = self.rules
        lo = r.ceiling_margin + gap / 2
        hi = self.ground_y - r.ground_margin - gap / 2
        return lo, max(lo, hi)

    def _spawn_pipe(self, x: float, prev: Pipe | None, spacing: float, speed: float) -> Pipe:
        """A pipe at x whose gap can be reached from `prev`'s."""
        gap = self.gap_size()
        lo, hi = self._gap_bounds(gap)
        if prev is None:
            centre = (lo + hi) / 2  # the first gap is dead centre: a gentle opener
            centre += self.rng.uniform(-0.25, 0.25) * (hi - lo)
        else:
            up = self.climb_budget(spacing, speed)
            lo = max(lo, prev.gap_y - up)
            hi = min(hi, prev.gap_y + self.rules.max_drop)
            centre = self.rng.uniform(lo, hi) if hi > lo else min(max(prev.gap_y, lo), hi)
        return Pipe(x=x, gap_y=centre, gap=gap)

    def _fill_pipes(self) -> None:
        """Keep one pipe queued past the right edge, spaced by time-to-bird."""
        speed = self.pipe_speed()
        if not self.pipes:
            x = self.bird.x + self.rules.first_pipe_delay * speed
            # Never appear inside the frame; it would pop in under the PiP.
            x = max(x, self.width + self.pipe_width)
            self.pipes.append(self._spawn_pipe(x, None, 0.0, speed))
        while self.pipes[-1].x < self.width + self.pipe_width:
            spacing = speed * self.pipe_interval()
            last = self.pipes[-1]
            self.pipes.append(self._spawn_pipe(last.x + spacing, last, spacing, speed))

    # --- framing ---

    # Room below the shoulder line, in shoulder widths, needed to see a full
    # downstroke. Standing close (as in recorded play, 0.1-0.4 widths) sends
    # the wrists out of the bottom of the frame on every flap; at a 15-20 Hz
    # model rate the gesture detector then misses 6-31% of normal-pace flaps.
    # With the wrists in view it catches them all. Separate on/off levels so
    # the prompt does not flicker at the boundary.
    TOO_CLOSE_ON = 0.8
    TOO_CLOSE_OFF = 1.0

    def _update_framing(self, pose) -> None:
        from ...config import KP

        if pose is None:
            return
        ls = pose.point(KP["left_shoulder"], 0.4)
        rs = pose.point(KP["right_shoulder"], 0.4)
        if ls is None or rs is None:
            return
        width = float(np.linalg.norm(ls - rs))
        if width < 1.0:
            return
        room = (self.height - float((ls[1] + rs[1]) * 0.5)) / width
        self.room_below = room
        if self.too_close:
            self.too_close = room < self.TOO_CLOSE_OFF
        else:
            self.too_close = room < self.TOO_CLOSE_ON

    # --- simulation ---

    def update(self, controls: ControlState, dt: float) -> None:
        dt = min(max(dt, 0.0), self.rules.max_dt)  # a stall must not teleport the bird
        self._clock += dt

        self.last_pose = controls.pose
        self._update_framing(controls.pose)
        event = self.detector.update(controls.pose, self._clock)
        if event is not None:
            self.fx.flap = 1.0
            self.fx.flap_strength = float(getattr(event, "strength", 1.0))
        self._decay_effects(dt)

        if self.phase is Phase.READY:
            self._update_ready(dt)
            if event is not None:
                self.phase = Phase.PLAYING
                self._fill_pipes()
                self._flap(self.fx.flap_strength)
            return

        if self.phase is Phase.PLAYING:
            if event is not None:
                self._flap(self.fx.flap_strength)
            self._step_world(dt)
        else:
            self.dead_time += dt
            self._step_corpse(dt)
        self._step_feathers(dt)

    def _update_ready(self, dt: float) -> None:
        b = self.bird
        b.y = self._ready_y() + self.rules.ready_bob * math.sin(self._clock * 2 * math.pi * 0.9)
        b.vy, b.angle = 0.0, 0.0
        self._animate_wings(dt)
        # The ground keeps rolling on the READY screen, as in the original:
        # the bird reads as flying in place rather than frozen.
        self.distance += self.rules.speed_start * dt
        self._step_feathers(dt)

    def _flap(self, strength: float) -> None:
        b = self.bird
        b.vy = -self.rules.flap_velocity
        b.angle = self.rules.nose_up  # the nose snaps up at once
        b.wing, b.wing_boost = 0.0, 1.0  # wings snap down, then beat fast
        self.flaps += 1
        rng = np.random.default_rng()
        n = 5 + int(round(4 * min(1.0, max(0.0, strength))))
        speed = self.speed
        for _ in range(n):
            # Shed downward and backward, against the beat, and left behind by
            # the scrolling world so they hang in the air the bird flew through.
            ang = rng.uniform(0.35, 2.8)  # radians, below the horizon
            v = rng.uniform(60, 190)
            vel = np.array([math.cos(ang) * v - speed * 0.6, math.sin(ang) * v], np.float32)
            self.fx.feathers.append(Feather(
                pos=np.array([b.x - 8 + rng.uniform(-6, 6), b.y + rng.uniform(-4, 10)], np.float32),
                vel=vel, angle=rng.uniform(0, 360), spin=rng.uniform(-360, 360),
                colour=FEATHER_TINTS[int(rng.integers(len(FEATHER_TINTS)))],
            ))

    def _animate_wings(self, dt: float) -> None:
        b = self.bird
        b.wing += dt * (7.0 + 13.0 * b.wing_boost)  # frames/s: lazy glide -> hard beat
        b.wing_boost = max(0.0, b.wing_boost - dt * 2.5)

    def _pitch(self, dt: float) -> None:
        r, b = self.rules, self.bird
        if b.vy <= r.dive_after:
            target = r.nose_up
        else:
            k = (b.vy - r.dive_after) / max(1.0, r.terminal_velocity - r.dive_after)
            target = r.nose_up + (r.nose_down - r.nose_up) * min(1.0, k)
        if target < b.angle:
            b.angle = max(target, b.angle - r.nose_drop_rate * 2 * dt)
        else:
            b.angle = min(target, b.angle + r.nose_drop_rate * dt)

    def _step_world(self, dt: float) -> None:
        r, b = self.rules, self.bird
        n = max(1, int(math.ceil(dt / r.substep)))
        h = dt / n
        for _ in range(n):
            speed = self.pipe_speed()
            b.vy = min(b.vy + r.gravity * h, r.terminal_velocity)
            b.y += b.vy * h
            if b.y < self.hit_radius:  # the ceiling is a soft cap, not a death
                b.y, b.vy = self.hit_radius, max(b.vy, 0.0)
            for p in self.pipes:
                p.x -= speed * h
            self.distance += speed * h
            self._score_pass()
            if self._collides():
                self._crash()
                break
        self.pipes = [p for p in self.pipes if p.x > -self.pipe_width]
        if self.phase is Phase.PLAYING:
            self._fill_pipes()
        self._pitch(dt)
        self._animate_wings(dt)

    def _score_pass(self) -> None:
        b = self.bird
        for p in self.pipes:
            if not p.scored and p.x <= b.x:
                p.scored = True
                self.score += 1
                self.fx.score_bump = 1.0
                self.fx.popups.append(Popup(np.array([b.x, b.y - 36], np.float32), "+1"))

    def _collides(self) -> bool:
        b, r = self.bird, self.hit_radius
        if b.y + r >= self.ground_y:
            return True
        half = self.pipe_width / 2
        for p in self.pipes:
            if abs(p.x - b.x) > half + r:
                continue
            x0, x1 = p.x - half, p.x + half
            if (_circle_hits_rect(b.x, b.y, r, x0, -1e6, x1, p.gap_top)
                    or _circle_hits_rect(b.x, b.y, r, x0, p.gap_bottom, x1, 1e6)):
                return True
        return False

    def _crash(self) -> None:
        self.phase = Phase.GAME_OVER
        self.dead_time = 0.0
        self.fx.flash = 1.0
        self.fx.shake = 1.0
        self.bird.vy = max(self.bird.vy, 0.0)  # no more lift: it drops from where it hit
        self.bird.y = min(self.bird.y, self.ground_y - self.hit_radius)  # never sunk into the ground
        if self.score > type(self).session_best:
            self.new_best = True
            type(self).session_best = self.score

    def _step_corpse(self, dt: float) -> None:
        """After a crash the bird falls nose-first to the ground and stays."""
        r, b = self.rules, self.bird
        floor = self.ground_y - self.hit_radius
        if b.y < floor:
            b.vy = min(b.vy + r.gravity * 1.4 * dt, r.terminal_velocity * 1.6)
            b.y = min(floor, b.y + b.vy * dt)
        else:
            b.vy = 0.0
        b.angle = min(90.0, b.angle + 480.0 * dt)

    @property
    def landed(self) -> bool:
        return self.phase is Phase.GAME_OVER and self.bird.y >= self.ground_y - self.hit_radius - 0.5

    def _step_feathers(self, dt: float) -> None:
        for f in self.fx.feathers:
            f.vel *= max(0.0, 1.0 - 2.2 * dt)  # air drag: they float, not fall
            f.vel[1] += 260.0 * dt
            f.pos += f.vel * dt
            f.angle += f.spin * dt
            f.life -= dt / FEATHER_SECONDS
        self.fx.feathers = [f for f in self.fx.feathers if f.life > 0][-40:]

    def _decay_effects(self, dt: float) -> None:
        fx = self.fx
        fx.flash = max(0.0, fx.flash - dt * 3.0)
        fx.shake = max(0.0, fx.shake - dt * 2.5)
        fx.score_bump = max(0.0, fx.score_bump - dt * 3.5)
        fx.flap = max(0.0, fx.flap - dt * 2.5)
        for pop in fx.popups:
            pop.life -= dt / POPUP_SECONDS
            pop.pos[1] -= 70.0 * dt * max(pop.life, 0.0)  # rise, easing to a stop
            pop.pos[0] -= self.speed * dt * 0.5  # drift back with the pipe it was for
        fx.popups = [p for p in fx.popups if p.life > 0][-6:]

    # --- rendering ---

    def render(self, frame: np.ndarray) -> None:
        from .render import FlappyRenderer

        if not hasattr(self, "_renderer"):
            self._renderer = FlappyRenderer((self.width, self.height), self.art,
                                            bird_scale=self.rules.bird_scale)
        self._renderer.draw(frame, self)
