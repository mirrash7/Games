"""Raccoon Golf: course data, physics, scoring, the game flow, the arcade shell,
and fairness bots that play the whole course through the real swing detector.

No camera and no model: poses are synthetic (golf_poses.py), delivered the
way the pipeline does - stepped at 20 Hz, 150 ms late, with keypoint noise.
"""

from __future__ import annotations

import math
import random
import time

import numpy as np
import pytest

from golf_poses import CENTRE, SW, arc, grip_pose, model_clock, top_angle_for
from kpapp.controls import ControlState
from kpapp.game import REGISTRY
from kpapp.game.golf import physics as P
from kpapp.game.golf import render as R
from kpapp.game.golf.art import stub_art
from kpapp.game.golf.course import YARDS_PER_M, load_course
from kpapp.game.golf.game import (KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_UP, MAX_OVER_PAR, GolfGame, Phase,
                                  hole_points, result_name, to_par_text)
from kpapp.game.golf.swing import PUTT_FULL, PUTT_START, SwingEvent
from kpapp.shell import Screen, Shell

W, H = 1280, 720
DT = 1 / 60
COURSE = load_course()


def new_game(**kw) -> GolfGame:
    return GolfGame((W, H), art=stub_art(), **kw)


def to_aim(game: GolfGame) -> None:
    """Skip the hole's intro."""
    while game.phase is Phase.INTRO:
        game.update(ControlState(), DT)


def finish_flight(game: GolfGame, limit: float = 60.0) -> None:
    t = 0.0
    while game.phase is Phase.FLIGHT and t < limit:
        game.update(ControlState(), DT)
        t += DT
    assert game.phase is not Phase.FLIGHT


def hit(game: GolfGame, power: float, quality: float = 1.0) -> None:
    game._hit(SwingEvent(t=game.clock, power=power, quality=quality, side=1, speed=9.0, practice=False))


# --- the course -------------------------------------------------------------


def test_course_is_three_holes_par_12():
    assert [h.par for h in COURSE.holes] == [4, 3, 5]
    yards = [h.length * YARDS_PER_M for h in COURSE.holes]
    assert 330 < yards[0] < 420 and 130 < yards[1] < 190 and 470 < yards[2] < 560
    for h in COURSE.holes:
        assert h.lie_at(*h.tee) == "tee"
        assert h.lie_at(*h.pin) == "green"
        for t in h.trees:
            assert h.lie_at(t.x, t.y) not in ("ob", "water", "green", "bunker"), (h.number, t)
        lo, hi = h.wind_range
        assert 0 <= lo <= hi <= 12


def test_water_wins_over_the_fairway_it_crosses():
    """Hole 3's creek runs across the fairway: a ball in it is in the water."""
    h = COURSE.holes[2]
    assert any(f.contains(-17.0, 306.0) for f in h.fairway)
    assert h.lie_at(-17.0, 306.0) == "water"


def test_auto_aim_follows_the_dogleg():
    """From the tee of the par 4, a driver aims down the fairway, not at the pin
    over the trees; on the par 3 it aims straight at the pin."""
    h1 = COURSE.holes[0]
    driver = P.CLUBS[0]
    tx, ty = h1.aim_point(*h1.tee, driver.carry_m)
    assert (tx, ty) != h1.pin
    assert h1.lie_at(tx, ty) == "fairway"
    h2 = COURSE.holes[1]
    assert h2.aim_point(*h2.tee, P.club_by_key("5I").carry_m) == h2.pin


# --- physics ----------------------------------------------------------------


def flat_hole(lie: str = "fairway"):
    from kpapp.game.golf.course import hole_from_dict

    big = [[-800, -800], [800, -800], [800, 900], [-800, 900]]
    return hole_from_dict({
        "par": 4, "tee": [0, 0], "pin": [0, 700], "path": [[0, 0], [0, 700]], "land": big,
        "tee_box": [[-1, -1], [1, -1], [1, 1], [-1, 1]],
        "fairway": [big] if lie == "fairway" else [],
        "green": big if lie == "green" else [[0, 690], [1, 690], [1, 691]],
        "bunkers": [big] if lie == "bunker" else [], "water": [], "trees": [],
    }, 1)


FLAT = flat_hole()


def carry(club, power=1.0, **kw):
    return P.preview(FLAT, (0.0, 0.0), club, 0.0, kw.pop("lie", "fairway"), power) if not kw else \
        _air(P.Flight(FLAT, (0.0, 0.0), P.Shot(club, power, 0.0, "fairway", **kw)))


def _air(f):
    while f.state == "air" and f.landing is None:
        f._substep(P.SUBSTEP)
    return f


@pytest.mark.parametrize("club", P.CLUBS[:-1], ids=lambda c: c.key)
def test_clubs_carry_their_yardage(club):
    yd = carry(club).carry * YARDS_PER_M
    assert yd == pytest.approx(club.carry_yd, rel=0.03)


def test_a_full_driver_flies_like_a_drive():
    f = P.simulate(FLAT, (0, 0), P.Shot(P.CLUBS[0], 1.0, 0.0, "tee"))
    total = math.hypot(f.x, f.y) * YARDS_PER_M
    assert 235 <= total <= 265  # ~230 carry plus a fairway roll
    assert 20 <= f.apex <= 35


def test_carry_is_roughly_proportional_to_power():
    """POWER_EXP makes the gauge read as a share of the club's distance."""
    full = carry(P.CLUBS[3]).carry
    for p in (0.25, 0.5, 0.75):
        assert carry(P.CLUBS[3], p).carry / full == pytest.approx(p, abs=0.06)


def test_physics_is_deterministic_whatever_the_frame_rate():
    shot = P.Shot(P.CLUBS[2], 0.93, 0.05, "fairway", wind=(1.5, -2.0), curve=0.3)
    a = P.simulate(COURSE.holes[0], (0.0, 0.0), shot)
    b = P.Flight(COURSE.holes[0], (0.0, 0.0), shot)
    for dt in [1 / 30, 1 / 144, 0.05] * 10000:
        if b.done:
            break
        b.step(dt)
    assert (a.state, a.x, a.y) == (b.state, b.x, b.y)


def test_wind_pushes_the_ball():
    calm = carry(P.CLUBS[0]).carry
    head = carry(P.CLUBS[0], wind=(0.0, -4.5)).carry
    tail = carry(P.CLUBS[0], wind=(0.0, 4.5)).carry
    assert head < calm - 8 and tail > calm + 8
    cross = carry(P.CLUBS[0], wind=(4.5, 0.0))
    assert cross.x > 8.0


def test_overswing_curve_bends_both_ways():
    slice_ = carry(P.CLUBS[0], 1.15, curve=1.0)
    hook = carry(P.CLUBS[0], 1.15, curve=-1.0)
    assert slice_.x > 20 and hook.x < -20


def test_lies_and_strikes_cost_distance():
    fair = carry(P.CLUBS[3]).carry
    rough = _air(P.Flight(FLAT, (0, 0), P.Shot(P.CLUBS[3], 1.0, 0.0, "rough"))).carry
    sand = _air(P.Flight(FLAT, (0, 0), P.Shot(P.CLUBS[3], 1.0, 0.0, "bunker"))).carry
    thin = _air(P.Flight(FLAT, (0, 0), P.Shot(P.CLUBS[3], 1.0, 0.0, "fairway", quality=0.0)))
    assert 0.72 < rough / fair < 0.88
    assert 0.45 < sand / fair < 0.65
    assert thin.carry / fair < 0.75
    assert thin.apex < carry(P.CLUBS[3]).apex * 0.6


def test_a_wedge_stops_on_the_green_and_a_driver_runs():
    green = flat_hole("green")
    pw = P.simulate(green, (0, 0), P.Shot(P.club_by_key("PW"), 1.0, 0.0, "fairway"))
    dr = P.simulate(green, (0, 0), P.Shot(P.CLUBS[0], 1.0, 0.0, "fairway"))
    assert math.hypot(pw.x, pw.y) - pw.carry < 4.0
    assert math.hypot(dr.x, dr.y) - dr.carry > 12.0


def test_putts_roll_in_proportion_to_power():
    green = flat_hole("green")
    d = [P.simulate(green, (0, 0), P.Shot(P.PUTTER, p, 0.0, "green")).y for p in (0.25, 0.5, 1.0)]
    assert d[1] / d[2] == pytest.approx(0.5, abs=0.03)
    assert d[0] / d[2] == pytest.approx(0.25, abs=0.03)
    assert d[2] == pytest.approx(P.putt_range_m(), rel=0.03)


def test_the_green_slope_breaks_a_putt():
    """Hole 1's green falls to the right (+x): a putt aimed straight drifts right."""
    h = COURSE.holes[0]
    px, py = h.pin
    gx, _ = h.green_gradient(px - 4, py - 4)
    assert gx < 0  # downhill toward +x
    f = P.simulate(h, (px, py - 8.0), P.Shot(P.PUTTER, 0.55, 0.0, "green"))
    assert f.x > px + 0.3


def test_cup_takes_a_slow_ball_and_rejects_a_fast_one():
    green = flat_hole("green")
    pin = green.pin  # (0, 700)
    slow = P.simulate(green, (0.0, pin[1] - 3.0), P.Shot(P.PUTTER, 0.27, 0.0, "green"))
    fast = P.simulate(green, (0.0, pin[1] - 3.0), P.Shot(P.PUTTER, 1.0, 0.0, "green"))
    assert slow.state == "holed"
    assert fast.state == "rest" and fast.y > pin[1] + 5


def test_water_is_a_hazard_with_a_dry_drop_short_of_it():
    h = COURSE.holes[1]  # the carry over water
    f = P.simulate(h, h.tee, P.Shot(P.club_by_key("9I"), 0.55, math.atan2(-6, 148), "tee"))
    assert f.state == "water"
    drop = P.drop_point(h, f)
    assert h.lie_at(*drop) not in ("water", "ob")
    assert h.distance_to_pin(*drop) > h.distance_to_pin(*f.water_entry)


def test_leaving_the_island_is_out_of_bounds():
    h = COURSE.holes[0]
    f = P.simulate(h, h.tee, P.Shot(P.CLUBS[0], 1.0, math.radians(-60), "tee"))
    assert f.state == "ob"


def test_trees_knock_the_ball_down():
    h = COURSE.holes[0]
    tr = h.trees[1]
    head = math.atan2(tr.x, tr.y)
    # Find a power whose flight passes through that canopy.
    hits = [p / 20 for p in range(4, 21)
            if P.simulate(h, h.tee, P.Shot(P.club_by_key("9I"), p / 20, head, "tee")).hit_trees]
    assert hits


# --- scoring ----------------------------------------------------------------


def test_points_by_result():
    assert hole_points(1, 3) == 1000
    assert hole_points(2, 5) == 800 and hole_points(3, 5) == 500
    assert [hole_points(s, 4) for s in (3, 4, 5, 6, 7, 9)] == [300, 200, 100, 50, 10, 10]
    assert hole_points(4, 4, picked_up=True) == 10
    assert result_name(1, 3) == "HOLE IN ONE!" and result_name(3, 4) == "BIRDIE!"
    assert result_name(3, 5) == "EAGLE!" and result_name(5, 4) == "BOGEY"
    assert [to_par_text(d) for d in (0, 2, -1)] == ["E", "+2", "-1"]


# --- the game ---------------------------------------------------------------


def test_contract_basics():
    g = new_game(seed=1)
    assert REGISTRY["golf"] is GolfGame
    assert GolfGame.title and GolfGame.blurb and GolfGame.tip and len(GolfGame.tutorial) == 3
    assert GolfGame.board_id == "golf" and GolfGame.menu
    assert g.phase is Phase.INTRO and g.score == 0 and g.hole.number == 1
    g.update(ControlState(), 5.0)  # a stall advances at most 0.1 s
    assert g.clock == pytest.approx(0.1)


def test_wind_is_random_by_default_and_seedable():
    a = [new_game(seed=7)._wind_dir for _ in range(2)]
    assert a[0] == a[1]
    winds = {round(new_game()._wind_dir, 6) for _ in range(6)}
    assert len(winds) > 1


def test_auto_club_and_putter_on_the_green():
    g = new_game(seed=1)
    to_aim(g)
    assert g.club.key == "1W" and not g.putting
    px, py = g.hole.pin
    g.ball, g.lie = (px, py - 5), "green"
    g._setup_shot()
    assert g.putting and g.detector.mode == "putt"
    g.ball, g.lie = (px, py - 110), "fairway"
    g._setup_shot()
    assert g.club.key in ("9I", "PW")


def test_arrow_keys_aim_and_change_club():
    g = new_game(seed=1)
    to_aim(g)
    h0, c0 = g.heading, g.club_i
    assert g.handle_key(KEY_RIGHT) and g.heading > h0
    assert g.handle_key(KEY_LEFT) and g.handle_key(KEY_LEFT) and g.heading < h0
    assert g.handle_key(KEY_DOWN) and g.club_i == c0 + 1
    assert g.handle_key(KEY_UP) and g.club_i == c0
    assert not g.handle_key(ord("x"))
    # The driver is for the tee only.
    g.ball, g.lie = (0.0, 150.0), "fairway"
    g._setup_shot()
    for _ in range(12):
        g.change_club(-1)
        assert g.club.key != "1W"


def test_water_penalty_and_drop():
    g = new_game(seed=1)
    g._start_hole(1)
    to_aim(g)
    hit(g, 0.55)
    finish_flight(g)
    assert g.phase is Phase.RESULT and g.message.text == "WATER HAZARD"
    assert g.strokes == 2
    assert g.lie not in ("water", "ob")
    while g.phase is Phase.RESULT:
        g.update(ControlState(), DT)
    assert g.phase is Phase.AIM


def test_out_of_bounds_replays_from_the_same_spot():
    g = new_game(seed=1)
    to_aim(g)
    g.aim_offset = math.radians(-40)
    g.auto_heading = math.radians(-30)
    hit(g, 1.0)
    finish_flight(g)
    assert g.message.text == "OUT OF BOUNDS"
    assert g.strokes == 2 and g.ball == g.hole.tee


def test_pick_up_at_par_plus_five():
    g = new_game(seed=1)
    g._start_hole(1)  # par 3
    to_aim(g)
    for _ in range(20):
        if g.phase is Phase.HOLED:
            break
        hit(g, 0.08)  # dribbles
        finish_flight(g)
        while g.phase is Phase.RESULT:
            g.update(ControlState(), DT)
    assert g.phase is Phase.HOLED
    row = g.card[1]
    assert row.picked_up and row.strokes == 3 + MAX_OVER_PAR and row.points == 10


def test_holing_out_scores_and_the_round_ends_after_hole_three():
    g = new_game(seed=2)
    for i in range(3):
        to_aim(g)
        assert g.hole.number == i + 1
        g.ball, g.lie = (g.hole.pin[0], g.hole.pin[1] - 2.0), "green"
        g._setup_shot()
        g.auto_heading, g.aim_offset = 0.0, 0.0
        g.strokes = g.hole.par - 2  # this putt makes a birdie
        best = next(p / 100 for p in range(5, 100) if
                    P.simulate(g.hole, g.ball, P.Shot(P.PUTTER, p / 100, 0.0, "green")).state == "holed")
        hit(g, best)
        finish_flight(g)
        assert g.phase is Phase.HOLED and g.message.text == "BIRDIE!"
        while g.phase is Phase.HOLED:
            g.update(ControlState(), DT)
    assert g.phase.value == "game_over"
    assert g.score == 900 and g.to_par == -3
    assert all(r.done for r in g.card)


def test_on_resume_forgets_the_swing():
    g = new_game(seed=1)
    to_aim(g)
    g.detector.stage = g.detector.stage.__class__.BACKSWING
    g.on_resume()
    assert g.detector.stage.value == "no_grip"


def test_aim_buttons_are_dead_while_the_hands_grip():
    """A backswing sweeps the hands past the AIM buttons: they must not fire."""
    g = new_game(seed=1)
    to_aim(g)
    heading = g.heading
    x0, y0, x1, y1 = dict(g.buttons())["aim_right"]
    u, h = ((x0 + x1) / 2 - CENTRE[0]) / SW, (CENTRE[1] - (y0 + y1) / 2) / SW
    pose = grip_pose(u, h)  # both hands together, held over the button
    for _ in range(120):
        g.update(ControlState(present=True, pose=pose, step_pose=pose), DT)
    assert g.heading == heading


def test_tracking_loss_never_costs_a_stroke():
    g = new_game(seed=1)
    to_aim(g)
    for _ in range(600):
        g.update(ControlState(), DT)
    assert g.strokes == 0 and g.phase is Phase.AIM
    assert g.detector.guidance()


# --- through the real arcade shell -------------------------------------------


class _StubGolf(GolfGame):
    def __init__(self, size, **kw):
        super().__init__(size, art=stub_art(), **kw)


def test_plays_through_the_real_shell():
    shell = Shell((W, H), {"golf": _StubGolf}, selected="golf")
    shell.handle_key(ord("1"))
    assert shell.screen is Screen.TUTORIAL
    shell.handle_key(13)
    for _ in range(200):
        shell.update(ControlState(), DT)
    assert shell.screen is Screen.PLAYING
    for _ in range(int(3.2 / DT)):
        shell.update(ControlState(), DT)
    g = shell.game
    assert g.phase is Phase.AIM
    h0 = g.heading
    shell.handle_key(KEY_RIGHT)  # arrows reach the game
    assert g.heading > h0 and shell.screen is Screen.PLAYING
    shell.handle_key(32)  # space still pauses
    assert shell.screen is Screen.PAUSED


def test_welcome_screen_fits_four_tiles():
    from kpapp.leaderboard import LocalBoard  # noqa: F401 - the menu works with or without a board
    shell = Shell((W, H), REGISTRY)
    assert len(shell.menu_games) == 4 and "golf" in shell.menu_games
    tiles = [b.rect for b in shell._layout()]
    assert tiles[0][0] >= 0 and tiles[-1][2] <= W
    for a, b in zip(tiles, tiles[1:]):
        assert a[2] < b[0]
    frame = np.zeros((H, W, 3), np.uint8)
    shell.render(frame)


def test_hud_keeps_clear_of_the_pause_button_and_footer():
    pause = (1090, 646, 1258, 698)

    def overlap(a, b):
        return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]

    g = new_game(seed=1)
    rects = [R.PIP, R.MAP, R.HOLE_PANEL, R.CLUB_PANEL, R.WIND_PANEL] + [r for _, r in g.buttons()]
    gx, gy = R.GAUGE_AT
    rects.append((gx, gy, gx + 66 * R.GAUGE_SCALE, gy + 400 * R.GAUGE_SCALE))
    for r in rects:
        assert not overlap(r, pause), r
        assert r[3] <= 700, r  # the footer runs along y 700+


# --- rendering ---------------------------------------------------------------


def _real_art_or_skip():
    from kpapp.game.golf.art import AssetsMissingError, load_art

    try:
        return load_art()
    except AssetsMissingError as exc:
        pytest.skip(str(exc))


def test_renders_every_phase_within_budget():
    art = _real_art_or_skip()
    g = GolfGame((W, H), art=art, seed=4)
    frame = np.zeros((H, W, 3), np.uint8)
    times = []

    def go(n):
        for _ in range(n):
            g.update(ControlState(), DT)
            f = frame.copy()
            t0 = time.perf_counter()
            g.render(f)
            times.append(time.perf_counter() - t0)

    g.render(frame.copy())  # build caches
    go(60)  # intro
    to_aim(g)
    go(30)
    hit(g, 1.0)
    go(240)  # flight
    mean_ms = 1000 * float(np.mean(times))
    assert mean_ms < 6.0, f"{mean_ms:.2f} ms per frame"  # ~2 ms measured; slack for a loaded machine


def test_projection_survives_the_camera_inside_polygons():
    """Polygons that surround or pass behind the camera clip at the near plane
    instead of turning inside out (the island always contains the camera)."""
    from kpapp.game.golf.game import Camera

    view = R.View(Camera(0.0, 0.0, 4.0, 0.0, 0.17), W, H, 1000.0)
    img = np.zeros((H, W, 3), np.uint8)
    square = R.Layer((255, 255, 255), [[[-3000, -3000], [3000, -3000], [3000, 3000], [-3000, 3000]]])
    R.draw_layer(img, view, square)
    # Ground everywhere below the horizon is covered, nothing above it.
    assert img[int(view.horizon) + 40:, :].min() == 255
    assert img[: int(view.horizon) - 2, :].max() == 0


def test_hole_out_view_frames_the_flag_beside_the_text_and_lenny_cheering():
    """After holing out the camera pulls back and turns so the flag stands right
    of the result text (centred at y ~288) and Lenny cheers left of it, both on
    dry land, below the text and clear of the top HUD row and the map. The old
    close-up put the pole straight through "BIRDIE!"."""
    from kpapp.game.golf.game import holed_camera

    for hole in COURSE.holes:
        for hd in (-0.3, 0.0, 0.25, math.atan2(hole.pin[0] - hole.path[-2][0], hole.pin[1] - hole.path[-2][1])):
            cam, spot = holed_camera(hole.pin, hd)
            view = R.View(cam, W, H, 1000.0)
            px, py, _ = view.point(*hole.pin)
            lx, ly, _ = view.point(*spot)
            assert 760 <= px <= 1000 and 480 <= py <= 680, (hole.number, hd, px, py)
            assert 300 <= lx <= 560 and 420 <= ly <= 640, (hole.number, hd, lx, ly)
            assert hole.lie_at(*spot) not in ("water", "ob")


def test_tutorial_cards_draw_only_inside_their_rects():
    from kpapp.game.golf.tutorial import GOLF_TUTORIAL

    rects = [(69 + i * 400, 134, 411 + i * 400, 334) for i in range(3)]
    for t in (0.0, 0.7, 1.4, 2.2, 3.1):
        noise = np.random.default_rng(1).integers(0, 256, (H, W, 3), dtype=np.uint8)
        frame = noise.copy()
        for step, rect in zip(GOLF_TUTORIAL, rects):
            step.draw(frame, rect, t)
        changed = np.any(frame != noise, axis=2)
        mask = np.zeros((H, W), bool)
        for x0, y0, x1, y1 in rects:
            mask[y0:y1 + 1, x0:x1 + 1] = True
            assert changed[y0:y1, x0:x1].any()
        assert not (changed & ~mask).any()


# --- fairness bots: whole rounds through the real detector --------------------


class Player:
    """Feeds a game poses the way the camera pipeline would: a model at 20 Hz,
    each pose showing the player 150 ms in the past, ~5 px of noise."""

    def __init__(self, game, model_hz=20.0, lag=0.15, noise=5.0, seed=0):
        self.game, self.lag, self.noise = game, lag, noise
        self.rng = np.random.default_rng(seed)
        self.clock = model_clock(model_hz, 3600.0, self.rng)
        self.t = 0.0
        self.motion = lambda t: None  # (u, h, gap) or None at time t
        self._held_key, self._held = None, None

    def step(self):
        t = self.t
        ts = float(self.clock[np.searchsorted(self.clock, t + 1e-9) - 1])
        if ts != self._held_key:
            self._held_key = ts
            m = self.motion(ts - self.lag)
            if m is None:
                self._held = None
            else:
                u, h, gap = m
                self._held = grip_pose(u, h, noise=self.noise, rng=self.rng, gap=gap)
        pose = self._held
        self.game.update(ControlState(present=pose is not None, pose=pose, step_pose=pose), DT)
        self.t += DT


def full_swing(t0, power, side=1):
    top = top_angle_for(power)

    def m(t):
        t -= t0
        if t < 0:
            return None
        if t < 0.8:
            return (*arc(0.0, side=side), 0.15)
        if t < 1.5:
            e = (1 - math.cos(math.pi * (t - 0.8) / 0.7)) / 2
            return (*arc(top * e, side=side), 0.15)
        if t < 1.65:
            return (*arc(top, side=side), 0.15)
        if t < 2.1:
            k = (t - 1.65) / 0.45
            return (*arc(top - (top + 0.9 * top) * k * k * (3 - 2 * k), side=side), 0.15)
        return None
    return m, t0 + 2.3


def putt_stroke(t0, power, side=1):
    amp = power * PUTT_FULL

    def m(t):
        t -= t0
        if t < 0:
            return None
        if t < 0.8:
            return (0.0, -1.3, 0.15)
        if t < 1.4:
            return (side * amp * (1 - math.cos(math.pi * (t - 0.8) / 0.6)) / 2, -1.28, 0.15)
        if t < 1.55:
            return (side * amp, -1.28, 0.15)
        if t < 2.0:
            k = (t - 1.55) / 0.45
            return (side * amp * (1 - 1.8 * k * k * (3 - 2 * k)), -1.28, 0.15)
        return None
    return m, t0 + 2.2


def plan(game: GolfGame):
    """The skilled bot's choice: aim (in button steps) and power, by trying shots
    in the game's own simulator - wind, slope and all - and avoiding trouble."""
    hole, ball, lie = game.hole, game.ball, game.lie
    step = math.radians(1.0) if game.putting else math.radians(2.0)
    penalty = {"water": 1e4, "ob": 1e4, "bunker": 12.0, "rough": 6.0}

    def cost(off, power):
        f = P.simulate(hole, ball, P.Shot(game.club, power, game.auto_heading + off * step, lie, wind=game.wind))
        if f.state == "holed":
            return -1.0
        d = hole.distance_to_pin(f.x, f.y)
        return d + penalty.get(f.state if f.state in ("water", "ob") else f.lie, 0.0)

    lo = PUTT_START / PUTT_FULL + 0.01 if game.putting else 0.1  # the gentlest stroke the detector sees
    offs = range(-12, 13, 3) if game.putting else range(-6, 7, 2)
    powers = np.arange(lo, 1.0001, 0.05 if game.putting else 0.1)
    best = min(((cost(o, p), o, p) for o in offs for p in powers))
    _, o0, p0 = best
    fine = [(cost(o, p), o, p) for o in range(o0 - (1 if not game.putting else 2), o0 + (2 if not game.putting else 3))
            for p in np.clip(np.arange(p0 - 0.05, p0 + 0.051, 0.025 if not game.putting else 0.01), lo, 1.0)]
    _, o, p = min(fine + [best])
    return o, float(p)


def play_round(game: GolfGame, brain, limit=900.0):
    pl = Player(game)
    busy_until = -1.0
    while game.phase.value != "game_over" and pl.t < limit:
        if game.phase is Phase.AIM and pl.t > busy_until:
            busy_until = brain(game, pl)
        pl.step()
    return pl.t


def skilled(game, pl):
    o, p = plan(game)
    while round(game.aim_offset / (math.radians(1.0) if game.putting else math.radians(2.0))) != o:
        game.adjust_aim(1 if o > round(game.aim_offset / (math.radians(1.0) if game.putting else math.radians(2.0)))
                        else -1)
    side = 1 if random.random() < 0.5 else -1
    m, end = (putt_stroke if game.putting else full_swing)(pl.t, p, side)
    pl.motion = m
    return end + 1.0


@pytest.mark.parametrize("seed", [11, 13])
def test_skilled_bot_with_lag_plays_near_par(seed):
    """150 ms late, 20 Hz, 5 px of noise, both swing sides: still about par.
    Proves the course and the input are fair, not only possible."""
    random.seed(seed)
    g = new_game(seed=seed)
    play_round(g, skilled)
    assert g.phase.value == "game_over"
    strokes = sum(r.strokes for r in g.card)
    # It reads wind and slope exactly (it plans in the game's own simulator),
    # so the lag and noise are what it fights: about par, a little under.
    assert COURSE.par - 4 <= strokes <= COURSE.par + 3, [r.strokes for r in g.card]
    assert not any(r.picked_up for r in g.card)
    assert g.score >= 300


def test_idle_player_scores_nothing_and_nothing_breaks():
    g = new_game(seed=3)
    pl = Player(g)
    for _ in range(int(120 / DT)):
        pl.step()
    assert g.score == 0 and g.strokes == 0 and g.phase is Phase.AIM


def test_random_flailer_scores_poorly_and_never_crashes():
    rng = random.Random(5)
    g = new_game(seed=5)
    pl = Player(g, seed=5)
    keys = [(0.0, 0.0, -1.0, 0.15)]
    t = 0.0
    while t < 600:
        t += rng.uniform(0.15, 0.7)
        keys.append((t, rng.uniform(-1.8, 1.8), rng.uniform(-1.7, 1.3), rng.choice((0.15, 0.15, 0.6, 1.4))))

    def motion(tt):
        i = max(0, min(len(keys) - 2, int(np.searchsorted([k[0] for k in keys], tt)) - 1))
        (t0, u0, h0, g0), (t1, u1, h1, _) = keys[i], keys[i + 1]
        e = min(1.0, max(0.0, (tt - t0) / (t1 - t0)))
        return (u0 + (u1 - u0) * e, h0 + (h1 - h0) * e, g0)

    pl.motion = motion
    while pl.t < 600 and g.phase.value != "game_over":
        pl.step()
    done = [r for r in g.card if r.done]
    assert g.score <= 100 * max(1, len(done))  # bogey or worse on average, if anything
