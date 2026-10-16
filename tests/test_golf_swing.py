"""Golf swing detection (game/golf/swing.py) on synthetic, realistically awkward input.

Poses arrive the way the pipeline delivers `controls.step_pose`: from a model
at 15-30 Hz, held between updates, 150 ms late, with ~5 px of keypoint noise
and the elbows mostly missing. The player stands ~2 m back (a 200 px shoulder
width). See golf_poses.py for the synthetic golfer.
"""

from __future__ import annotations

import math

import pytest

from golf_poses import arc, path, run_detector, stream, swing_path, top_angle_for
from kpapp.config import KP
from kpapp.game.golf.swing import HIT_SPEED, OVERSWING_MAX, PUTT_FULL, Stage, SwingDetector, SwingEvent

REAL = dict(model_hz=20.0, lag=0.15, noise=5.0, elbows_every=3)


def one(events) -> SwingEvent:
    assert len(events) == 1, events
    return events[0]


# --- address ----------------------------------------------------------------


def test_hands_together_low_and_still_is_ready():
    det = SwingDetector()
    seen = []
    for t, pose in stream(lambda t: arc(0.0), 1.5, **REAL):
        det.update(pose, t)
        seen.append((t, det.stage))
    ready_at = next(t for t, s in seen if s is Stage.READY)
    # The hold is 0.4 s, plus the 150 ms lag and a model period.
    assert 0.4 <= ready_at <= 0.8
    assert det.address is not None and det.address[1] == pytest.approx(-1.35, abs=0.06)


@pytest.mark.parametrize("why,motion,kw", [
    ("hands apart", lambda t: arc(0.0), dict(gap=1.2)),
    ("hands at the chest", lambda t: (0.0, -0.4), {}),
    ("one hand hidden", lambda t: arc(0.0), dict(hands=lambda t: 1)),
])
def test_no_address_without_a_low_two_handed_grip(why, motion, kw):
    det = SwingDetector()
    for t, pose in stream(motion, 2.0, **{**REAL, **kw}):
        det.update(pose, t)
        assert det.stage is not Stage.READY, why
    assert det.guidance() is not None  # and the HUD says what to fix


def test_restless_hands_never_address():
    """Hands together and low but drifting: READY needs them still."""
    det = SwingDetector()
    wander = lambda t: (0.3 * math.sin(t * 5.0), -1.2 + 0.15 * math.sin(t * 7.0))  # noqa: E731
    for t, pose in stream(wander, 3.0, **REAL):
        det.update(pose, t)
        assert det.stage is not Stage.READY


def test_too_close_asks_to_step_back():
    det = SwingDetector()
    # Shoulders 120 px above the bottom of the frame with a 200 px shoulder width.
    for t, pose in stream(lambda t: (0.0, -0.5), 1.0, centre=(640.0, 600.0), **REAL):
        det.update(pose, t)
    assert det.too_close
    assert det.guidance().startswith("STEP BACK")


# --- full swings ------------------------------------------------------------


@pytest.mark.parametrize("hz", [15.0, 20.0, 30.0])
@pytest.mark.parametrize("side", [1, -1])
def test_a_full_swing_hits_on_either_side(hz, side):
    """Right- and left-handed backswings both work, at any model rate."""
    motion, t_impact = swing_path(power=0.9, side=side)
    ev = one(run_detector(SwingDetector(), motion, 4.0, **{**REAL, "model_hz": hz}))
    assert not ev.practice
    assert ev.side == side
    assert ev.power == pytest.approx(0.9, abs=0.06)
    assert ev.quality > 0.9
    # Late by the 150 ms lag plus up to about a model period, never early.
    assert t_impact <= ev.t <= t_impact + 0.15 + 2.5 / hz


def test_power_rises_with_backswing_height():
    powers = []
    for p in (0.25, 0.45, 0.65, 0.85, 1.0, 1.1):
        motion, _ = swing_path(power=p)
        powers.append(one(run_detector(SwingDetector(), motion, 4.0, **REAL)).power)
    assert powers == sorted(powers)
    assert powers[0] < 0.35 and powers[-1] > 1.0


def test_hands_at_head_height_are_full_power():
    """The design: power 1 = the grip back up at about head height."""
    motion, _ = swing_path(power=1.0)
    ev = one(run_detector(SwingDetector(), motion, 4.0, **REAL))
    assert ev.power == pytest.approx(1.0, abs=0.05)
    assert ev.overswing < 0.4


def test_overswing_is_reported_and_capped():
    motion, _ = swing_path(power=1.3)  # hands well above the head
    ev = one(run_detector(SwingDetector(), motion, 4.0, **REAL))
    assert ev.power == pytest.approx(OVERSWING_MAX)
    assert ev.overswing == 1.0


def test_practice_swing_without_addressing_does_not_count():
    """Swinging before the hands were held still at address: a practice swing."""
    top = top_angle_for(0.8)

    def motion(t):
        if t < 0.6:  # hands come down from the chest and go straight back up
            return 0.0, -0.4 - 0.9 * (t / 0.6)
        if t < 1.3:
            return arc(top * (t - 0.6) / 0.7)
        if t < 1.7:
            return arc(top - (top + 1.8) * (t - 1.3) / 0.4)
        return arc(-1.8)

    ev = one(run_detector(SwingDetector(), motion, 3.0, **REAL))
    assert ev.practice


def test_slow_lowering_is_not_a_hit():
    """Backswing, then bringing the hands back down to address over 1.5 s."""
    top = top_angle_for(1.0)

    def motion(t):
        if t < 1.6:
            return arc(0.0)
        if t < 2.4:
            return arc(top * (t - 1.6) / 0.8)
        if t < 2.6:
            return arc(top)
        if t < 4.1:
            return arc(top * (1 - (1 - math.cos(math.pi * (t - 2.6) / 1.5)) / 2))
        return arc(0.0)

    det = SwingDetector()
    hints = set()
    events = []
    for t, pose in stream(motion, 5.0, **REAL):
        ev = det.update(pose, t)
        if ev:
            events.append(ev)
        if det.hint:
            hints.add(det.hint)
    assert events == []
    assert "SWING THROUGH!" in hints


def test_hit_speed_threshold_sits_between_lowering_and_swinging():
    """Pin the reason for HIT_SPEED: lowering the hands over 1.5 s peaks under
    it, a brisk 0.3 s downswing crosses the ball well over it."""
    radius, top = 1.35, top_angle_for(1.0)
    lowering_peak = radius * top * math.pi / (2 * 1.5)  # cosine-eased over 1.5 s
    assert lowering_peak < HIT_SPEED
    motion, _ = swing_path(power=1.0, down=0.3)
    ev = one(run_detector(SwingDetector(), motion, 4.0, **REAL))
    assert ev.speed > 1.5 * HIT_SPEED


def test_thin_strike_when_the_hands_come_through_high():
    clean = one(run_detector(SwingDetector(), swing_path(power=0.9)[0], 4.0, **REAL))
    thin = one(run_detector(SwingDetector(), swing_path(power=0.9, low_cross=1.0)[0], 4.0, **REAL))
    assert clean.quality > 0.9
    assert thin.quality < 0.3
    assert thin.power == pytest.approx(clean.power, abs=0.05)  # power still comes from the backswing


def test_shoulders_turning_away_do_not_inflate_power():
    """From the front the shoulders narrow as the body turns into a backswing
    (to about half at the top). The frame is frozen at address, so the hands'
    height is not overstated."""
    top = top_angle_for(0.7)
    motion, _ = swing_path(power=0.7)
    det = SwingDetector()
    events, done = [], set()
    for t, pose in stream(motion, 4.0, **REAL):
        if pose is not None and id(pose) not in done:
            done.add(id(pose))
            u, h = motion(t - REAL["lag"])
            turn = 0.5 * min(1.0, max(0.0, math.acos(max(-1.0, min(1.0, -h / 1.35))) / top))
            ls, rs = KP["left_shoulder"], KP["right_shoulder"]
            mid = (pose.xy[ls] + pose.xy[rs]) / 2
            pose.xy[ls] = mid + (pose.xy[ls] - mid) * (1 - turn)
            pose.xy[rs] = mid + (pose.xy[rs] - mid) * (1 - turn)
        ev = det.update(pose, t)
        if ev:
            events.append(ev)
    assert one(events).power == pytest.approx(0.7, abs=0.08)


# --- tracking ---------------------------------------------------------------


def test_teleport_is_lost_then_found_not_a_swing():
    """A grip that jumps across the frame for one model frame is a glitch."""
    def motion(t):
        if 1.5 <= t < 1.55:
            return (3.0, 0.5)  # 600 px away for one sample
        return arc(0.0)

    events = run_detector(SwingDetector(), motion, 3.0, model_hz=20.0, noise=5.0)
    assert events == []


def test_losing_the_player_mid_swing_never_hits():
    motion, t_impact = swing_path(power=0.9)
    lost = lambda t: t > t_impact - 0.25  # noqa: E731 - gone just before the ball
    assert run_detector(SwingDetector(), motion, 4.0, lost=lost, **REAL) == []


def test_a_short_wrist_dropout_mid_swing_still_hits():
    motion, t_impact = swing_path(power=0.8)
    hands = lambda t: 1 if t_impact - 0.35 < t < t_impact - 0.25 else 2  # noqa: E731
    ev = one(run_detector(SwingDetector(), motion, 4.0, hands=hands, **REAL))
    assert ev.power == pytest.approx(0.8, abs=0.08)


def test_elbows_seen_or_not_give_the_same_power():
    """The grip is the wrists, so elbows flickering in and out (as they do in
    about half of real frames) cannot shift it."""
    motion, _ = swing_path(power=0.75)
    a = one(run_detector(SwingDetector(), motion, 4.0, **{**REAL, "elbows_every": 0}))
    b = one(run_detector(SwingDetector(), motion, 4.0, **{**REAL, "elbows_every": 2}))
    assert a.power == pytest.approx(b.power, abs=0.02)


def test_reset_forgets_a_swing_in_progress():
    motion, t_impact = swing_path(power=0.9)
    det = SwingDetector()
    events = []
    for t, pose in stream(motion, 4.0, **REAL):
        if abs(t - (t_impact - 0.3)) < 1e-9 or (t_impact - 0.31 < t < t_impact - 0.29):
            det.reset()  # the game was paused right here
        ev = det.update(pose, t)
        if ev:
            events.append(ev)
    assert events == []


# --- putting ----------------------------------------------------------------


def putt_motion(power, back=0.7, through=0.45, side=1):
    amp = power * PUTT_FULL
    return path((0.0, 0.0, -1.3), (1.4, 0.0, -1.3), (1.4 + back, side * amp, -1.27),
                (1.6 + back, side * amp, -1.27), (1.6 + back + through, -side * amp * 0.8, -1.27),
                (3.5 + back, -side * amp * 0.8, -1.27))


@pytest.mark.parametrize("side", [1, -1])
def test_putting_pendulum_power_follows_the_backstroke(side):
    powers = []
    for p in (0.15, 0.35, 0.6, 0.9):
        ev = one(run_detector(SwingDetector("putt"), putt_motion(p, side=side), 4.5, **REAL))
        assert not ev.practice and ev.quality == 1.0
        powers.append(ev.power)
    assert powers == sorted(powers)
    for got, want in zip(powers, (0.15, 0.35, 0.6, 0.9)):
        assert got == pytest.approx(want, abs=0.06)


def test_putts_never_overswing():
    ev = one(run_detector(SwingDetector("putt"), putt_motion(1.6), 4.5, **REAL))
    assert ev.power == 1.0


def test_easing_the_putter_back_is_not_a_putt():
    """Backstroke, then drifting back to the ball far too slowly."""
    amp = 0.5 * PUTT_FULL
    slow = path((0.0, 0.0, -1.3), (1.4, 0.0, -1.3), (2.0, amp, -1.27), (2.2, amp, -1.27),
                (6.2, -0.3, -1.27))
    assert run_detector(SwingDetector("putt"), slow, 7.0, **REAL) == []


@pytest.mark.parametrize("mode", ["full", "putt"])
def test_standing_at_address_never_hits_by_itself(mode):
    """Jitter during a long address (6 px, a little worse than the 4-5 px
    measured) must not fire. A putt's backstroke starts at only PUTT_START,
    so the putter's grip is smoothed and must stay out for two samples."""
    for seed in range(6):
        events = run_detector(SwingDetector(mode), lambda t: arc(0.0), 30.0,
                              **{**REAL, "noise": 6.0, "seed": seed})
        assert events == [], (mode, seed)
