"""Flappy Bird: physics, pipes, scoring, fairness under input lag, rendering."""

from __future__ import annotations

import math
import random
import time
from collections import deque

import numpy as np
import pytest

from kpapp.config import KP
from kpapp.controls import ControlState
from kpapp.game.flappy.art import FlappyArt
from kpapp.game.flappy.game import FlappyGame, Phase, Pipe, Rules, medal_for
from kpapp.inference import Pose

W, H = 1280, 720
DT = 1 / 60
GROUND = 96


def _shipped_bird() -> tuple[int, tuple[int, int]]:
    """Hit radius and sprite size of the art that actually ships.

    Read from the generated manifest rather than hard-coded, so the fairness
    proof below always flies the real collision circle: when the bird became a
    raccoon, a hard-coded radius silently kept testing the old, smaller one.
    """
    import json
    from pathlib import Path

    gen = Path(__file__).resolve().parents[1] / "assets" / "flappy" / "generated"
    radius = json.loads((gen / "manifest.json").read_text())["bird_hit_radius"]
    import cv2
    h, w = cv2.imread(str(gen / "bird_0.png"), cv2.IMREAD_UNCHANGED).shape[:2]
    return int(radius), (h, w)


SHIPPED_RADIUS, SHIPPED_SIZE = _shipped_bird()


def stub_art() -> FlappyArt:
    """Bird size and hit radius come from the shipped art, so fairness carries over."""
    def sprite(h, w, bgr=(200, 200, 200)):
        s = np.zeros((h, w, 4), np.uint8)
        s[:, :, :3] = bgr
        s[:, :, 3] = 255
        return s

    skyline = sprite(220, W, (120, 140, 90))
    skyline[:120, :, 3] = 0  # transparent sky above the rooftops
    skyline[120:130, :, 3] = 128
    return FlappyArt(
        bird_frames=[sprite(*SHIPPED_SIZE, (40, 200, 250)) for _ in range(3)],
        bird_hit_radius=SHIPPED_RADIUS,
        pipe_body=sprite(64, 110, (60, 180, 60)),
        pipe_cap=sprite(44, 130, (50, 200, 50)),
        sky=np.full((H, W, 3), (230, 200, 140), np.uint8),
        skyline=skyline,
        ground=np.full((GROUND, W, 3), (80, 160, 200), np.uint8),
        ground_height=GROUND,
        feather=sprite(24, 24, (255, 255, 255)),
        medals={m: sprite(88, 88) for m in ("bronze", "silver", "gold", "platinum")},
        panel=sprite(300, 520, (120, 200, 230)),
    )


class FakeFlapEvent:
    def __init__(self, t: float, strength: float = 0.8) -> None:
        self.t, self.strength = t, strength


class FakeDetector:
    """Stands in for FlapDetector: emits a flap on the next update when asked."""

    def __init__(self) -> None:
        self.queued = 0
        self.resets = 0
        self.arms_visible = True
        self.wing: float | None = 0.4
        self.seen: list[float] = []

    def flap(self) -> None:
        self.queued += 1

    def reset(self) -> None:
        self.resets += 1
        self.queued = 0

    def update(self, pose, t):
        self.seen.append(t)
        if self.queued:
            self.queued -= 1
            return FakeFlapEvent(t)
        return None


@pytest.fixture(autouse=True)
def _fresh_session_best():
    FlappyGame.session_best = 0
    yield
    FlappyGame.session_best = 0


def new_game(**kw) -> tuple[FlappyGame, FakeDetector]:
    det = kw.pop("detector", None) or FakeDetector()
    return FlappyGame((W, H), art=stub_art(), detector=det, **kw), det


def idle(game: FlappyGame, seconds: float, dt: float = DT) -> None:
    for _ in range(int(round(seconds / dt))):
        game.update(ControlState(), dt)


def start(game: FlappyGame, det: FakeDetector) -> None:
    det.flap()
    game.update(ControlState(), DT)
    assert game.phase is Phase.PLAYING


# --- the READY screen ---


def test_starts_ready_and_waits_for_a_flap():
    """Classic: the bird bobs in place and nothing happens until the first flap."""
    game, _ = new_game(seed=1)
    assert game.phase is Phase.READY
    assert game.phase.value == "ready"
    y0 = game.bird.y
    idle(game, 5.0)
    assert game.phase is Phase.READY
    assert abs(game.bird.y - y0) <= game.rules.ready_bob + 1
    assert game.score == 0 and not game.pipes


def test_first_flap_starts_play_and_lifts_the_bird():
    game, det = new_game(seed=1)
    idle(game, 0.5)
    start(game, det)
    assert game.bird.vy < 0
    assert game.pipes, "pipes should be queued once play starts"
    assert min(p.x for p in game.pipes) >= W, "the first pipe must not pop in on screen"


def test_ready_feeds_the_detector_a_clock():
    game, det = new_game()
    idle(game, 0.2)
    assert len(det.seen) == 12 and det.seen == sorted(det.seen) and det.seen[0] > 0


# --- physics ---


def test_without_flaps_the_bird_falls_to_the_ground():
    game, det = new_game(seed=2)
    start(game, det)
    idle(game, 3.0)
    assert game.phase is Phase.GAME_OVER
    assert game.phase.value == "game_over"
    idle(game, 1.5)
    assert game.landed
    assert game.bird.y + game.hit_radius == pytest.approx(game.ground_y, abs=1.0)


def test_flap_sets_rather_than_adds_velocity():
    """Classic: a flap always feels the same, whatever the bird was doing."""
    game, det = new_game(seed=2)
    start(game, det)
    idle(game, 0.6)  # falling now
    det.flap()
    game.update(ControlState(), DT)
    falling_flap = game.bird.vy
    det.flap()
    game.update(ControlState(), DT)  # flap again while already rising
    assert falling_flap == pytest.approx(game.bird.vy, abs=1.0)
    assert falling_flap < -game.rules.flap_velocity * 0.9


def test_fall_speed_is_capped():
    game, det = new_game(seed=2)
    start(game, det)
    game.bird.y = -5000  # a long way to fall, so it reaches the cap
    game.pipes.clear()
    game.rules.first_pipe_delay = 1e9
    for _ in range(90):
        game.update(ControlState(), DT)
        assert game.bird.vy <= game.rules.terminal_velocity + 1e-6


def test_ceiling_is_a_soft_cap():
    """Flapping into the top of the screen pins the bird there; it does not die."""
    game, det = new_game(seed=3)
    start(game, det)
    game.pipes = [Pipe(x=1e6, gap_y=300, gap=230)]  # nothing nearby
    for _ in range(240):
        det.flap()
        game.update(ControlState(), DT)
        assert game.bird.y >= game.hit_radius - 1e-6
    assert game.phase is Phase.PLAYING


def test_nose_follows_vertical_speed():
    game, det = new_game(seed=4)
    start(game, det)
    game.update(ControlState(), DT)
    assert game.bird.angle < 0, "nose up while climbing"
    game.pipes = [Pipe(x=1e6, gap_y=300, gap=230)]
    idle(game, 0.9)
    assert game.bird.angle >= game.rules.nose_down - 0.5, "nose down in a dive"
    assert game.bird.angle <= game.rules.nose_down + 1e-6, "but no further than the cap"


def test_stall_does_not_teleport_the_bird():
    game, det = new_game(seed=5)
    start(game, det)
    y0 = game.bird.y
    game.update(ControlState(), 5.0)  # absurd stall
    r = game.rules
    worst = r.terminal_velocity * r.max_dt
    assert abs(game.bird.y - y0) <= worst + 1


def test_stall_cannot_carry_the_bird_through_a_pipe():
    """Every stalled frame is clamped and substepped, so a pipe right ahead
    is hit rather than skipped over."""
    game, det = new_game(seed=5)
    start(game, det)
    b = game.bird
    game.pipes = [Pipe(x=b.x + game.pipe_width / 2 + game.hit_radius + 2, gap_y=b.y - 400, gap=200)]
    for _ in range(30):
        game.update(ControlState(), 2.0)
    assert game.phase is Phase.GAME_OVER
    assert game.score == 0


# --- collision ---


def _bird_at(game, x, y):
    game.bird.x, game.bird.y = x, y


def test_circle_collision_against_pipes():
    game, det = new_game(seed=6)
    start(game, det)
    r, half = game.hit_radius, game.pipe_width / 2
    game.pipes = [Pipe(x=600, gap_y=360, gap=200)]  # opening 260..460
    _bird_at(game, 600, 360)
    assert not game._collides(), "dead centre of the gap"
    _bird_at(game, 600, 260 + r - 2)
    assert game._collides(), "grazing the top pipe"
    _bird_at(game, 600, 460 - r + 2)
    assert game._collides(), "grazing the bottom pipe"
    _bird_at(game, 600 - half - r - 1, 100)
    assert not game._collides(), "level with the top pipe but just in front of it"
    # Near the pipe's corner but outside the circle: a box test would kill here.
    _bird_at(game, 600 - half - r * 0.75, 260 + r * 0.75)
    assert not game._collides(), "corner miss must not count"


def test_ground_kills():
    game, det = new_game(seed=6)
    start(game, det)
    game.pipes.clear()
    _bird_at(game, game.bird.x, game.ground_y - game.hit_radius + 1)
    assert game._collides()


# --- scoring ---


def test_passing_a_pipe_scores_with_popup_and_pulse():
    game, det = new_game(seed=7)
    start(game, det)
    b = game.bird
    game.pipes = [Pipe(x=b.x + 10, gap_y=b.y, gap=300)]
    game.bird.vy = 0
    idle(game, 0.1)
    assert game.score == 1
    assert [p.text for p in game.fx.popups] == ["+1"]
    assert game.fx.score_bump > 0.5
    idle(game, 0.05)
    assert game.score == 1, "a pipe scores once"


def test_popups_rise_and_expire():
    game, det = new_game(seed=7)
    start(game, det)
    b = game.bird
    game.pipes = [Pipe(x=b.x + 2, gap_y=b.y, gap=400)]
    game.update(ControlState(), DT)
    pop = game.fx.popups[0]
    y0 = float(pop.pos[1])
    idle(game, 0.2)
    assert pop.pos[1] < y0
    idle(game, 1.0)
    assert not game.fx.popups


def test_flap_sheds_feathers_that_fade():
    game, det = new_game(seed=8)
    start(game, det)
    assert len(game.fx.feathers) >= 5
    game.pipes = [Pipe(x=1e6, gap_y=300, gap=230)]
    for _ in range(50):
        det.flap() if _ % 30 == 0 else None
        game.update(ControlState(), DT)
    idle(game, 1.0)
    assert not game.fx.feathers or all(f.life > 0 for f in game.fx.feathers)
    assert len(game.fx.feathers) <= 40


def test_medals():
    assert medal_for(9) is None
    assert medal_for(10) == "bronze"
    assert medal_for(19) == "bronze"
    assert medal_for(20) == "silver"
    assert medal_for(30) == "gold"
    assert medal_for(40) == "platinum"
    assert medal_for(400) == "platinum"


# --- lifecycle ---


def test_best_survives_reset_and_new_games():
    """The arcade builds a new game for every PLAY AGAIN: best must outlive it."""
    game, det = new_game(seed=9)
    start(game, det)
    game.score = 12
    idle(game, 3.0)
    assert game.phase is Phase.GAME_OVER
    assert game.best == 12 and game.new_best
    game.reset()
    assert game.phase is Phase.READY and game.score == 0
    assert game.best == 12
    again, det2 = new_game(seed=9)
    assert again.best == 12
    start(again, det2)
    again.score = 5
    idle(again, 3.0)
    assert again.best == 12 and not again.new_best


def test_game_over_is_terminal_and_world_stops():
    game, det = new_game(seed=10)
    start(game, det)
    idle(game, 3.0)
    assert game.phase is Phase.GAME_OVER
    xs = [p.x for p in game.pipes]
    d = game.distance
    for _ in range(60):
        det.flap()
        game.update(ControlState(), DT)
    assert game.phase is Phase.GAME_OVER
    assert [p.x for p in game.pipes] == xs and game.distance == d


def test_resume_resets_the_detector():
    """The arms moved during the pause; old history must not read as a flap."""
    game, det = new_game()
    before = det.resets
    game.on_resume()
    assert det.resets == before + 1


def test_every_game_is_different_but_a_seed_reproduces():
    def course(seed):
        game, det = new_game(seed=seed)
        start(game, det)
        game._fill_pipes()
        while len(game.pipes) < 6:
            game.pipes.append(game._spawn_pipe(game.pipes[-1].x + 400, game.pipes[-1], 400, 200))
        return tuple(round(p.gap_y, 3) for p in game.pipes)

    assert course(42) == course(42)
    assert len({course(None) for _ in range(5)}) == 5


# --- difficulty curve ---


def test_difficulty_starts_generous_and_ramps_to_a_floor():
    game, _ = new_game()
    assert game.gap_size(0) == pytest.approx(230)
    assert game.gap_size(0) / game.ground_y > 0.32  # of the play height
    assert game.gap_size(100) == pytest.approx(175)
    assert game.pipe_speed(0) == pytest.approx(200)
    assert game.pipe_speed(100) == pytest.approx(260)
    assert game.pipe_interval(0) >= 1.6
    assert game.pipe_interval(100) >= 1.6
    gaps = [game.gap_size(s) for s in range(60)]
    assert all(a >= b for a, b in zip(gaps, gaps[1:])), "never gets easier"


def test_gaps_stay_in_bounds_and_reachable():
    """Generated courses respect the margins and the climb budget."""
    for seed in range(40):
        game, det = new_game(seed=seed)
        start(game, det)
        r = game.rules
        for score in range(0, 60, 3):
            game.score = score
            speed = game.pipe_speed()
            spacing = speed * game.pipe_interval()
            prev = game.pipes[-1]
            p = game._spawn_pipe(prev.x + spacing, prev, spacing, speed)
            assert p.gap_top >= r.ceiling_margin - 1e-6
            assert p.gap_bottom <= game.ground_y - r.ground_margin + 1e-6
            rise = prev.gap_y - p.gap_y
            assert rise <= game.climb_budget(spacing, speed) + 1e-6
            assert -rise <= r.max_drop + 1e-6
            game.pipes.append(p)


# --- fairness: a lagged bot must be able to fly every course ---


class LaggedBot:
    """A skilled player with camera lag and human arms.

    Decides from what is on screen, but its flaps land `delay` seconds later
    (camera + model latency), optionally with random jitter it cannot predict.
    It flaps no faster than the game's assumed human rate, `flap_period`.

    Policy: hold the bird on a line inside the next gap, chosen so a flap from
    the line neither reaches the top pipe at its apex nor sinks onto the bottom
    pipe. Being skilled, it predicts where the bird will be when the flap
    actually lands (using its *nominal* delay, not the jittered real one).
    """

    def __init__(self, game: FlappyGame, det: FakeDetector, delay: float,
                 jitter: float = 0.0, seed: int = 0) -> None:
        self.game, self.det = game, det
        self.delay, self.jitter = delay, jitter
        self.rng = random.Random(seed)
        self.pending: deque[float] = deque()
        self.last = -1e9

    def _predict(self, t: float) -> tuple[float, float]:
        g, r = self.game, self.game.rules
        y, vy = g.bird.y, g.bird.vy
        flaps = sorted(self.pending)
        tt, end = t, t + self.delay
        while tt < end - 1e-9:
            while flaps and flaps[0] <= tt + 1e-9:
                vy = -r.flap_velocity
                flaps.pop(0)
            vy = min(vy + r.gravity * DT, r.terminal_velocity)
            y += vy * DT
            tt += DT
        return y, vy

    def _target(self, ahead: float) -> Pipe | None:
        g = self.game
        reach = g.pipe_width / 2 + g.hit_radius
        for p in g.pipes:
            if p.x - ahead + reach > g.bird.x:
                return p
        return None

    def step(self, t: float) -> None:
        g, r = self.game, self.game.rules
        if t - self.last >= r.flap_period:
            y, _ = self._predict(t)
            p = self._target(g.pipe_speed() * self.delay)
            if p is not None:
                rise = r.flap_velocity ** 2 / (2 * r.gravity)
                lo = p.gap_top + g.hit_radius + rise
                hi = p.gap_bottom - g.hit_radius - r.terminal_velocity * DT - 2
                line = (lo + hi) / 2
            else:
                line = g.ground_y * 0.5
            if y > line:
                lag = self.delay + (self.rng.uniform(-self.jitter, self.jitter) if self.jitter else 0)
                self.pending.append(t + max(0.0, lag))
                self.last = t
        while self.pending and self.pending[0] <= t + 1e-9:
            self.pending.popleft()
            self.det.flap()


def fly(seed: int, delay: float, jitter: float = 0.0, pipes: int = 60) -> tuple[int, str]:
    game, det = new_game(seed=seed)
    start(game, det)
    bot = LaggedBot(game, det, delay, jitter, seed)
    t = 0.0
    limit = pipes * 2.2 + 10
    while game.phase is Phase.PLAYING and game.score < pipes and t < limit:
        bot.step(t)
        game.update(ControlState(), DT)
        t += DT
    return game.score, game.phase.value


@pytest.mark.parametrize("delay,jitter", [(0.0, 0.0), (0.150, 0.0), (0.150, 0.05)])
def test_every_course_is_flyable_with_lag(delay, jitter):
    """No course may be impossible for a player with ~150 ms of input lag.

    60 pipes per run takes the difficulty past its floor (gap 175 px, 260
    px/s from score ~30), across 30 random courses (1800 pipes). The jittered run has the
    lag vary 100-200 ms unpredictably, as real camera + model latency does.
    """
    failures = []
    for seed in range(30):
        score, phase = fly(seed, delay, jitter)
        if score < 60:
            failures.append((seed, score, phase))
    assert not failures, f"bot crashed (seed, score, phase): {failures[:5]}"


# --- rendering ---


def _render(game, frame=None):
    frame = np.full((H, W, 3), 90, np.uint8) if frame is None else frame
    game.render(frame)
    return frame


def _pose() -> Pose:
    xy = np.zeros((17, 2), np.float32)
    c = np.zeros(17, np.float32)
    for name, p in (("left_shoulder", (560, 300)), ("right_shoulder", (720, 300)),
                    ("left_elbow", (470, 300)), ("right_elbow", (810, 300)),
                    ("left_wrist", (380, 290)), ("right_wrist", (900, 290))):
        xy[KP[name]] = p
        c[KP[name]] = 0.9
    return Pose(xy=xy, confidence=c, score=0.9)


def test_renders_every_phase():
    game, det = new_game(seed=11)
    _render(game)  # before any update, as during the arcade countdown
    idle(game, 0.3)
    _render(game)
    start(game, det)
    for i in range(400):
        if i % 25 == 0:
            det.flap()
        game.update(ControlState(pose=_pose()), DT)
        if i % 40 == 0:
            _render(game)
    game.bird.y = game.ground_y
    idle(game, 0.05)
    assert game.phase is Phase.GAME_OVER
    for _ in range(4):
        _render(game)
        idle(game, 0.4)
    det.arms_visible, det.wing = False, None
    _render(game)


def test_picture_in_picture_shows_the_camera():
    """The PiP is the camera frame as it arrived, before the scene covers it."""
    game, _ = new_game(seed=12)
    idle(game, 0.1)
    frame = np.zeros((H, W, 3), np.uint8)
    frame[:, :, 2] = 255  # a pure red camera image
    _render(game, frame)
    r = game._renderer
    x0, y0, x1, y1 = r.pip_rect
    centre = frame[(y0 + y1) // 2 + 30, (x0 + x1) // 2 - 60]
    assert centre[2] > 200 and centre[0] < 60, "camera feed should show in the PiP"
    assert frame[H // 2, 100, 2] < 200, "the scene should cover the rest of the camera"
    assert x1 <= W and y0 >= 0 and x0 > W // 2, "PiP sits in the top-right corner"


def test_bird_sprite_is_not_clipped_when_rotated():
    game, _ = new_game()
    game._render_init = _render(game)
    r = game._renderer
    for frame in r._bird_padded:
        h, w = frame.shape[:2]
        d = math.hypot(56, 72) * game.rules.bird_scale
        assert h >= d - 1 and w >= d - 1


def test_render_budget():
    """Render must stay well inside a 60 FPS frame on a busy scene."""
    game, det = new_game(seed=13)
    start(game, det)
    for i in range(200):
        if i % 20 == 0:
            det.flap()
        game.update(ControlState(pose=_pose()), DT)
    game.fx.score_bump = 1.0
    frame = np.full((H, W, 3), 90, np.uint8)
    _render(game, frame.copy())
    samples = []
    for _ in range(40):
        f = frame.copy()
        t0 = time.perf_counter()
        game.render(f)
        samples.append(time.perf_counter() - t0)
    median_ms = sorted(samples)[len(samples) // 2] * 1000
    assert median_ms < 12.0, f"render took {median_ms:.1f} ms"


def test_the_fairness_bot_can_fail():
    """The reach test has teeth: on courses that really are too hard, the
    same bot crashes. Otherwise a broken bot would pass everything."""
    for rules in (Rules(gap_start=95, gap_min=95),  # barely wider than the bird
                  # 3x the speed, no climb limit, gaps allowed at the edges:
                  # climbs no human flap rate can make in the time available.
                  Rules(max_drop=1e9, reach_safety=4.0, speed_start=600, speed_max=600,
                        interval_start=0.8, interval_min=0.8, ceiling_margin=20,
                        ground_margin=20)):
        crashed = 0
        for seed in range(8):
            game, det = new_game(seed=seed, rules=rules)
            start(game, det)
            bot = LaggedBot(game, det, 0.15)
            t = 0.0
            while game.phase is Phase.PLAYING and game.score < 30 and t < 80:
                bot.step(t)
                game.update(ControlState(), DT)
                t += DT
            crashed += game.phase is Phase.GAME_OVER
        assert crashed >= 4, f"bot survived impossible courses ({crashed}/8 crashed)"


def test_default_detector_is_the_real_one():
    """Without injection the game builds FlapDetector(mirrored) itself."""
    try:
        game = FlappyGame((W, H), art=stub_art(), mirrored=False)
    except NotImplementedError:
        pytest.skip("FlapDetector not implemented yet")
    from kpapp.game.flappy.gesture import FlapDetector

    assert isinstance(game.detector, FlapDetector)
    game.update(ControlState(), DT)  # no pose: no flap, no crash
    assert game.phase is Phase.READY
    _render(game)


# --- framing guidance ---


def _shoulders_at(y: float, width: float = 300.0) -> Pose:
    xy = np.zeros((17, 2), np.float32)
    c = np.zeros(17, np.float32)
    xy[KP["left_shoulder"]] = (640 + width / 2, y)
    xy[KP["right_shoulder"]] = (640 - width / 2, y)
    c[[KP["left_shoulder"], KP["right_shoulder"]]] = 0.9
    return Pose(xy=xy, confidence=c, score=0.9)


def test_framing_too_close_is_flagged_with_hysteresis():
    """Standing close sends the wrists out of frame on every flap, and at
    15-20 Hz the detector then misses flaps - so the game tells the player."""
    game, _ = new_game()

    def room(widths: float) -> bool:
        game.update(ControlState(present=True, pose=_shoulders_at(720 - widths * 300)), 1 / 60)
        return game.too_close

    assert room(0.3), "0.3 shoulder widths of room (as recorded) is too close"
    assert room(0.9), "between the on/off levels it stays flagged"
    assert not room(1.2)
    assert not room(0.9), "and stays clear between the levels"


def test_framing_is_left_alone_without_shoulders():
    game, _ = new_game()
    game.update(ControlState(present=False, pose=None), 1 / 60)
    assert not game.too_close



def test_fairness_flies_the_shipped_collision_circle():
    """Guard for the proof above: it must use the real raccoon's radius."""
    game, _ = new_game()
    assert game.hit_radius == pytest.approx(SHIPPED_RADIUS * game.rules.bird_scale)
    assert game.hit_radius > 20, "the raccoon's circle is ~22 px at play size"
