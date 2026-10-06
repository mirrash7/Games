"""Arm-flap detection on synthetic, realistically awkward pose streams.

Heights are in shoulder widths (sw) above the shoulder line: 0 = shoulder
height, ~-1.3 = wrists hanging at the sides. Poses are fed at a 60 Hz game
rate; the model is simulated at 15-30 Hz by holding each pose until the next
model update, which is how motion actually arrives (steps, not a ramp).
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest

from kpapp.config import KP
from kpapp.game.flappy.gesture import FlapDetector, FlapEvent
from kpapp.inference import Pose

GAME_HZ = 60.0
TOP, BOTTOM = 0.7, -1.3  # a full flap: wrists above the shoulders -> at the sides


# --- synthetic input ------------------------------------------------------


def make_pose(hl, hr, sw=420.0, center=(640.0, 560.0), noise=0.0, rng=None,
              shoulders=True) -> Pose:
    """Shoulders a shoulder-width `sw` apart; each wrist `h` sw above its
    shoulder (None = wrist not visible). Elbows and hips are never visible."""
    rng = rng if rng is not None else np.random.default_rng(0)
    xy = np.zeros((17, 2), np.float32)
    c = np.zeros(17, np.float32)
    cx, cy = center
    pts = {}
    if shoulders:
        pts["left_shoulder"] = (cx + sw / 2, cy)
        pts["right_shoulder"] = (cx - sw / 2, cy)
    for name, side, h in (("left_wrist", 1, hl), ("right_wrist", -1, hr)):
        if h is not None:
            pts[name] = (cx + side * 0.9 * sw, cy - h * sw)
    pts["nose"] = (cx, cy - 0.8 * sw)
    for name, p in pts.items():
        xy[KP[name]] = np.asarray(p) + (rng.normal(0, noise, 2) if noise else 0.0)
        c[KP[name]] = 0.9
    c[KP["left_elbow"]] = c[KP["right_elbow"]] = 0.1
    return Pose(xy=xy, confidence=c, score=0.9)


def keyframes(*kf):
    """h(t) through (t, h) keyframes with cosine easing between them."""

    def h(t):
        if t <= kf[0][0]:
            return kf[0][1]
        for (t0, h0), (t1, h1) in pairwise(kf):
            if t <= t1:
                u = (t - t0) / (t1 - t0)
                return h0 + (h1 - h0) * (1 - math.cos(math.pi * u)) / 2
        return kf[-1][1]

    return h


def wingbeats(start, n, hz, top=TOP, bottom=BOTTOM, lead=0.6):
    """Arms raised and held from t=0, then `n` sinusoidal flaps from `start`.

    Returns (h(t), downstroke start times)."""
    period = 1.0 / hz
    mid, amp = (top + bottom) / 2, (top - bottom) / 2
    end = start + n * period

    def h(t):
        if t < start or t >= end:
            return top
        return mid + amp * math.cos(2 * math.pi * (t - start) / period)

    return h, [start + k * period for k in range(n)]


def model_clock(model_hz, phase, jitter, duration, rng):
    """Times at which the model publishes a pose. jitter=0.3 spreads each
    interval over +-30% of the nominal period, as real inference does."""
    times, t = [], phase - 1.0 / model_hz
    while t < duration + 1.0:
        times.append(t)
        t += (1.0 / model_hz) * (1.0 + rng.uniform(-jitter, jitter))
    return np.asarray(times)


def run(left, right=None, duration=3.0, model_hz=None, phase=0.0, sw=420.0,
        center=(640.0, 560.0), noise=0.0, seed=0, hidden=None, lost=None,
        frame_bottom=None, both=True, det=None, jitter=0.0):
    """Feed a detector and collect events.

    left/right: h(t) per arm (right defaults to left; both=False hides it).
    model_hz: hold each pose for 1/model_hz s (stepped input); None = fresh
      pose every game frame. jitter: randomise the model's update interval.
    hidden(arm, t) -> True hides that wrist; lost(t) -> True feeds pose=None.
    frame_bottom: wrists lower than this (in sw) are out of frame.
    """
    rng = np.random.default_rng(seed)
    det = det or FlapDetector()
    right = right or left
    events: list[FlapEvent] = []
    wings = []
    held_key, held_pose = None, None
    clock = None if model_hz is None else model_clock(model_hz, phase, jitter, duration, rng)
    for i in range(int(duration * GAME_HZ)):
        t = i / GAME_HZ
        ts = t if clock is None else float(clock[np.searchsorted(clock, t + 1e-9) - 1])
        if lost is not None and lost(ts):
            pose = None
        else:
            if ts != held_key:
                hl, hr = left(ts), (right(ts) if both else None)
                if frame_bottom is not None:
                    hl = None if hl is not None and hl < frame_bottom else hl
                    hr = None if hr is not None and hr < frame_bottom else hr
                if hidden is not None:
                    hl = None if hidden(0, ts) else hl
                    hr = None if hidden(1, ts) else hr
                held_key = ts
                held_pose = make_pose(hl, hr, sw=sw, center=center, noise=noise, rng=rng)
            pose = held_pose
        ev = det.update(pose, t)
        wings.append((t, det.wing, det.arms_visible))
        if ev is not None:
            events.append(ev)
    return events, det, wings


def match(events, starts, max_lat=0.2):
    """Latency of each downstroke's event; None where a flap was missed.
    Asserts every event belongs to some downstroke (no phantoms)."""
    lats = []
    used = set()
    for s in starts:
        hit = [e for e in events if s <= e.t <= s + max_lat]
        lats.append(hit[0].t - s if hit else None)
        used.update(id(e) for e in hit[:1])
        assert len(hit) <= 1, f"double fire on the stroke at {s:.3f}: {[e.t for e in hit]}"
    stray = [e.t for e in events if id(e) not in used]
    assert not stray, f"phantom flaps at {stray}"
    return lats


def single_flap(t_down=1.2, down=0.18, top=TOP, bottom=BOTTOM):
    """Arms rise from the sides, pause up, one brisk downstroke at t_down."""
    return keyframes((0.0, bottom), (0.3, bottom), (0.8, top), (t_down, top),
                     (t_down + down, bottom))


# --- tests ----------------------------------------------------------------


def test_one_clean_flap_fires_exactly_once_and_fast():
    events, _, _ = run(single_flap(), duration=2.5)
    assert len(events) == 1
    lat = events[0].t - 1.2
    print(f"\nclean flap latency (60 Hz input): {lat * 1000:.0f} ms")
    assert 0.0 < lat <= 0.06
    assert 0.0 <= events[0].strength <= 1.0


@pytest.mark.parametrize("model_hz", [15.0, 20.0, 30.0])
@pytest.mark.parametrize("phase", [0.0, 0.013, 0.031, 0.049])
def test_stepped_input_still_detects_the_flap(model_hz, phase):
    events, _, _ = run(single_flap(), duration=2.5, model_hz=model_hz, phase=phase, noise=2.0)
    lats = match(events, [1.2])
    assert lats[0] is not None
    # Cannot beat the model: allow one model period on top of the 60 Hz budget.
    assert lats[0] <= 0.06 + 1.0 / model_hz


def test_latency_report():
    """Measure latency across stroke speeds and input rates, and report it."""
    rows = []
    for down in (0.15, 0.18, 0.25):
        for hz in (None, 30.0, 15.0):
            lats = []
            for phase in np.linspace(0, 1 / (hz or 60.0), 6, endpoint=False):
                ev, _, _ = run(single_flap(down=down), duration=2.0, model_hz=hz, phase=phase)
                (lat,) = match(ev, [1.2])
                assert lat is not None
                lats.append(lat)
            rows.append((down, hz, np.mean(lats), np.max(lats)))
    print("\n2 sw downstroke  input   mean    max")
    for down, hz, mean, mx in rows:
        print(f"  {down * 1000:4.0f} ms      {('60 Hz' if hz is None else f'{hz:.0f} Hz'):6s} "
              f"{mean * 1000:5.0f}ms {mx * 1000:5.0f}ms")
    for down, hz, mean, mx in rows:
        if hz is None and down <= 0.18:
            assert mx <= 0.06


@pytest.mark.parametrize("seconds", [1.2, 2.0, 3.0])
@pytest.mark.parametrize("model_hz", [None, 15.0])
def test_slowly_lowering_the_arms_does_not_flap(seconds, model_hz):
    h = keyframes((0.0, TOP), (0.5, TOP), (0.5 + seconds, BOTTOM))
    events, _, _ = run(h, duration=seconds + 1.5, model_hz=model_hz, noise=3.0)
    assert events == []


@pytest.mark.parametrize("level", [TOP, 0.0, -0.6, BOTTOM])
@pytest.mark.parametrize("sw", [420.0, 150.0])
def test_still_arms_with_noise_never_flap(level, sw):
    events, _, _ = run(lambda t: level, duration=4.0, model_hz=20.0, sw=sw, noise=4.0, seed=3)
    assert events == []


@pytest.mark.parametrize("around", [-0.6, 0.0, 0.5])
@pytest.mark.parametrize("hz", [2.0, 4.0, 7.0])
def test_jitter_and_small_wobbles_around_a_level_do_not_flap(around, hz):
    """Arms wobbling +-0.12 sw (0.24 sw peak to peak) plus pixel noise."""
    h = lambda t: around + 0.12 * math.sin(2 * math.pi * hz * t)
    events, _, _ = run(h, duration=4.0, model_hz=20.0, noise=3.0, seed=7)
    assert events == []


@pytest.mark.parametrize("hz", [3.0, 3.5, 4.0])
@pytest.mark.parametrize("model_hz", [None, 30.0, 15.0])
@pytest.mark.parametrize("amp", [(TOP, BOTTOM), (0.5, -0.5)])
def test_rapid_flapping_counts_every_flap(hz, model_hz, amp):
    n = int(3.0 * hz)
    h, starts = wingbeats(0.6, n, hz, top=amp[0], bottom=amp[1])
    events, _, _ = run(h, duration=0.6 + 3.0 + 0.5, model_hz=model_hz, noise=2.0, seed=11)
    lats = match(events, starts)
    assert all(lat is not None for lat in lats), f"missed flaps: {lats}"
    assert len(events) == n


@pytest.mark.parametrize("model_hz", [None, 15.0])
def test_same_flap_reads_the_same_near_and_far(model_hz):
    near, _, _ = run(single_flap(), model_hz=model_hz, sw=420.0, center=(640.0, 560.0))
    far, _, _ = run(single_flap(), model_hz=model_hz, sw=140.0, center=(600.0, 250.0))
    assert len(near) == len(far) == 1
    assert abs(near[0].t - far[0].t) <= 1.0 / GAME_HZ + 1e-9
    assert abs(near[0].strength - far[0].strength) < 0.05


def test_moving_toward_the_camera_does_not_flap():
    """Scale changing (player steps in) with arms held up is not a flap."""
    det = FlapDetector()
    events = []
    for i in range(240):
        t = i / GAME_HZ
        sw = 200.0 + 220.0 * min(t / 2.0, 1.0)
        ev = det.update(make_pose(0.4, 0.4, sw=sw, center=(640, 300 + 0.6 * sw)), t)
        events += [ev] if ev else []
    assert events == []


@pytest.mark.parametrize("model_hz", [None, 15.0])
def test_one_arm_visible_still_works(model_hz):
    h, starts = wingbeats(0.6, 9, 3.0)
    events, _, _ = run(h, duration=4.0, model_hz=model_hz, both=False, noise=2.0)
    lats = match(events, starts)
    assert all(lat is not None for lat in lats), lats


def test_one_arm_flapping_while_the_other_is_held_still_works():
    h, starts = wingbeats(0.6, 6, 2.0)
    events, _, _ = run(h, right=lambda t: 0.3, duration=4.0, model_hz=20.0, noise=2.0)
    lats = match(events, starts)
    assert all(lat is not None for lat in lats), lats


@pytest.mark.parametrize("gap_at", [1.24, 1.27, 1.30])
def test_wrist_dropout_mid_stroke_fires_once_not_twice(gap_at):
    """One wrist vanishes for ~60 ms in the middle of the downstroke."""
    hidden = lambda arm, t: arm == 0 and gap_at <= t < gap_at + 0.06
    events, _, _ = run(single_flap(), duration=2.5, model_hz=20.0, hidden=hidden, noise=2.0)
    assert len(events) == 1


@pytest.mark.parametrize("arm", [0, 1])
def test_wrist_dropouts_while_still_do_not_flap(arm):
    """Brief repeated dropouts of one wrist with arms up, then arms down."""
    hidden = lambda a, t: a == arm and (t % 0.5) < 0.1
    for level in (TOP, -0.3, BOTTOM):
        events, _, _ = run(lambda t, level=level: level, duration=3.0, model_hz=20.0, hidden=hidden,
                           noise=3.0)
        assert events == []


@pytest.mark.parametrize("jump", [-0.5, 3.0])
def test_a_glitched_wrist_jump_is_not_a_flap(jump):
    """One model output misplaces one wrist (0.5 sw: within the averaging;
    3 sw: an impossible speed), then it snaps back."""
    left = lambda t: TOP - abs(jump) if 1.50 <= t < 1.55 else TOP
    events, _, _ = run(left, right=lambda t: TOP, duration=3.0, model_hz=20.0, noise=2.0)
    assert events == []


@pytest.mark.parametrize("gap", [0.1, 0.4, 1.0])
@pytest.mark.parametrize("before_after", [(TOP, BOTTOM), (TOP, -0.2), (BOTTOM, TOP), (0.0, -0.6)])
def test_tracking_lost_then_reacquired_elsewhere_is_not_a_flap(gap, before_after):
    before, after = before_after
    h = lambda t: before if t < 1.0 else after
    lost = lambda t: 1.0 - gap <= t < 1.0
    events, _, _ = run(h, duration=3.0, model_hz=20.0, lost=lost, noise=2.0)
    assert events == []


def test_shoulders_lost_then_reacquired_is_not_a_flap():
    det = FlapDetector()
    events = []
    for i in range(180):
        t = i / GAME_HZ
        if t < 1.0:
            pose = make_pose(TOP, TOP)
        elif t < 1.3:
            pose = make_pose(TOP, TOP, shoulders=False)
        else:
            pose = make_pose(-0.3, -0.3)
        ev = det.update(pose, t)
        events += [ev] if ev else []
    assert events == []


def test_flapping_works_again_after_tracking_comes_back():
    lost = lambda t: 1.0 <= t < 1.6
    h, starts = wingbeats(2.2, 4, 2.0)
    events, _, _ = run(h, duration=4.5, model_hz=20.0, lost=lost)
    lats = match(events, starts)
    assert all(lat is not None for lat in lats), lats


@pytest.mark.parametrize(("top", "model_hz"),
                         [(0.8, None), (0.8, 30.0), (0.8, 20.0), (0.3, None), (0.3, 30.0)])
def test_close_framing_wrists_leave_the_frame_every_stroke(model_hz, top):
    """As recorded: shoulders ~0.25 sw above the bottom edge, so each
    downstroke carries both wrists out of frame part-way down."""
    h, starts = wingbeats(0.6, 10, 3.0, top=top, bottom=-1.3)
    events, _, wings = run(h, duration=4.2, model_hz=model_hz, frame_bottom=-0.25, noise=2.0)
    lats = match(events, starts)
    assert all(lat is not None for lat in lats), f"missed: {lats}"
    # Still "visible" while flapping, even though the wrists keep leaving.
    assert all(vis for t, _, vis in wings if 0.7 <= t <= 3.8)


def test_close_framing_hit_rate_report():
    """Hit rate when the wrists leave the frame on every stroke, with
    realistic model timing jitter. Never a phantom; at 30 Hz ~every flap. At
    15-20 Hz a 3-4 Hz flap leaves only 1-3 model frames per stroke in view,
    often all on the way up, and some strokes are simply not observable."""
    print("\nclose framing (edge 0.25 sw below shoulders), jitter +-30%: hit rate")
    print("  top   model  3 Hz   4 Hz")
    for top in (0.5, 0.8, 1.1):
        for hz in (30.0, 20.0, 15.0):
            rates = []
            for flap_hz in (3.0, 4.0):
                hits = total = 0
                for seed in range(6):
                    h, starts = wingbeats(0.6, int(3 * flap_hz), flap_hz, top=top)
                    ev, _, _ = run(h, duration=4.2, model_hz=hz, phase=seed * 0.011,
                                   frame_bottom=-0.25, noise=2.0, seed=seed, jitter=0.3)
                    lats = match(ev, starts)  # asserts: no phantoms, no doubles
                    hits += sum(lat is not None for lat in lats)
                    total += len(lats)
                rates.append(hits / total)
            print(f"  {top:.1f}  {hz:4.0f}Hz  {rates[0]:4.0%}  {rates[1]:4.0%}")
            if hz >= 30.0 and top >= 0.8:
                assert min(rates) >= 0.97


@pytest.mark.parametrize("model_hz", [20.0, 15.0])
@pytest.mark.parametrize("flap_hz", [3.0, 4.0])
def test_rapid_flapping_with_model_timing_jitter(model_hz, flap_hz):
    """Wrists in view throughout; model intervals vary +-30%."""
    for seed in range(5):
        n = int(3.0 * flap_hz)
        h, starts = wingbeats(0.6, n, flap_hz, top=1.1, bottom=-1.3)
        events, _, _ = run(h, duration=4.2, model_hz=model_hz, phase=seed * 0.013,
                           noise=2.0, seed=seed, jitter=0.3)
        lats = match(events, starts)
        assert all(lat is not None for lat in lats), (seed, lats)


def test_both_wrists_dropping_out_while_held_up_is_not_a_flap():
    """Raise the arms (re-arming on a seen upstroke), hold, lose both wrists."""
    h = keyframes((0.0, -0.2), (0.5, -0.2), (0.8, 0.8))
    for gap_at in (0.85, 1.0, 1.2):
        hidden = lambda arm, t, g=gap_at: g <= t < g + 0.2
        events, _, _ = run(h, duration=2.0, model_hz=20.0, hidden=hidden, noise=3.0,
                           frame_bottom=-0.25)
        assert events == [], gap_at


@pytest.mark.parametrize("model_hz", [None, 15.0])
def test_close_framing_slow_lowering_out_of_frame_does_not_flap(model_hz):
    h = keyframes((0.0, TOP), (0.6, TOP), (2.0, BOTTOM))
    events, _, _ = run(h, duration=3.0, model_hz=model_hz, frame_bottom=-0.25, noise=3.0)
    assert events == []


def test_one_arm_leaving_frame_while_other_held_up_is_not_a_flap():
    """A single wrist dropping out of view while the other stays up."""
    left = keyframes((0.0, 0.2), (1.0, 0.2), (1.25, -1.3))
    events, _, _ = run(left, right=lambda t: 0.6, duration=2.5, model_hz=20.0,
                       frame_bottom=-0.25, noise=2.0)
    assert events == []


def test_needs_to_rise_again_before_the_next_flap():
    """Two drops without lifting the arms in between: only the first fires."""
    h = keyframes((0.0, TOP), (0.6, TOP), (0.75, 0.1), (1.2, 0.1), (1.35, -0.5))
    events, _, _ = run(h, duration=2.0)
    assert len(events) == 1


def test_strength_tracks_vigour():
    brisk, _, _ = run(single_flap(down=0.12, top=1.0))
    lazy, _, _ = run(single_flap(down=0.40, top=0.0))
    assert len(brisk) == len(lazy) == 1
    assert 0.0 <= lazy[0].strength < brisk[0].strength <= 1.0


def test_wing_and_visibility():
    det = FlapDetector()
    assert det.wing is None and not det.arms_visible
    det.update(None, 0.0)
    assert det.wing is None and not det.arms_visible
    t = 0.0
    for _ in range(30):
        t += 1 / GAME_HZ
        det.update(make_pose(0.5, 0.5), t)
    assert det.arms_visible and det.wing is not None and det.wing > 0.95
    for _ in range(30):
        t += 1 / GAME_HZ
        det.update(make_pose(-1.3, -1.3), t)
    assert det.wing is not None and det.wing < 0.1
    det.update(make_pose(None, None, shoulders=False), t + 0.02)
    assert det.wing is None and not det.arms_visible
    # Shoulders only, for longer than the grace period: not visible.
    t += 0.02
    for _ in range(60):
        t += 1 / GAME_HZ
        det.update(make_pose(None, None), t)
    assert not det.arms_visible and det.wing is None


def test_reset_clears_everything():
    det = FlapDetector()
    t = 0.0
    for _ in range(60):
        t += 1 / GAME_HZ
        det.update(make_pose(TOP, TOP), t)
    det.reset()
    assert det.wing is None and not det.arms_visible
    # Arms suddenly much lower after the reset: not a flap.
    for _ in range(60):
        t += 1 / GAME_HZ
        assert det.update(make_pose(-0.5, -0.5), t) is None


def test_time_not_advancing_is_ignored():
    det = FlapDetector()
    det.update(make_pose(TOP, TOP), 1.0)
    assert det.update(make_pose(BOTTOM, BOTTOM), 1.0) is None
    assert det.update(make_pose(BOTTOM, BOTTOM), 0.5) is None
