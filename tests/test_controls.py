"""Control-mapping tests driven by synthetic skeletons.

These let the gesture logic be validated without a camera or a GPU.
"""

from __future__ import annotations

import numpy as np

from kpapp.config import KP, KEYPOINT_NAMES
from kpapp.controls import ControlMapper
from kpapp.inference import Pose, PoseSmoother

SIZE = (1280, 720)


def make_pose(**points: tuple[float, float]) -> Pose:
    """Build a pose from named keypoints; unnamed ones are marked invisible."""
    xy = np.zeros((len(KEYPOINT_NAMES), 2), dtype=np.float32)
    conf = np.zeros(len(KEYPOINT_NAMES), dtype=np.float32)
    for name, (x, y) in points.items():
        xy[KP[name]] = (x, y)
        conf[KP[name]] = 0.9
    return Pose(xy=xy, confidence=conf, score=0.9)


def standing(**overrides: tuple[float, float]) -> Pose:
    base = {
        "left_shoulder": (540.0, 260.0),
        "right_shoulder": (740.0, 260.0),
        "left_hip": (560.0, 500.0),
        "right_hip": (720.0, 500.0),
        "left_wrist": (540.0, 460.0),
        "right_wrist": (740.0, 460.0),
    }
    base.update(overrides)
    return make_pose(**base)


def test_no_pose_is_not_present():
    state = ControlMapper(SIZE).map(None)
    assert not state.present
    assert state.steer == 0.0


def test_missing_shoulders_is_not_present():
    pose = make_pose(left_wrist=(100.0, 100.0))
    assert not ControlMapper(SIZE).map(pose).present


def test_neutral_pose_is_centred():
    state = ControlMapper(SIZE).map(standing())
    assert state.present
    assert abs(state.steer) < 0.1, "hands at torso centre should not steer"
    assert state.throttle == 0.0, "hands below shoulders means no throttle"
    assert not state.hands_up


def test_hands_right_steers_positive():
    # Shoulder width is 200px; shift both hands 150px right of centre.
    pose = standing(left_wrist=(690.0, 460.0), right_wrist=(890.0, 460.0))
    state = ControlMapper(SIZE).map(pose)
    assert state.steer > 0.5, f"expected strong right steer, got {state.steer}"


def test_hands_left_steers_negative():
    pose = standing(left_wrist=(390.0, 460.0), right_wrist=(590.0, 460.0))
    assert ControlMapper(SIZE).map(pose).steer < -0.5


def test_deadzone_suppresses_small_offsets():
    # 10px off a 200px shoulder width is well inside the deadzone.
    pose = standing(left_wrist=(550.0, 460.0), right_wrist=(750.0, 460.0))
    assert ControlMapper(SIZE).map(pose).steer == 0.0


def test_raised_hands_give_throttle_and_latch():
    # Wrists a full shoulder-width above the shoulder line.
    pose = standing(left_wrist=(540.0, 60.0), right_wrist=(740.0, 60.0))
    state = ControlMapper(SIZE).map(pose)
    assert state.throttle > 0.8, state.throttle
    assert state.hands_up
    assert state.pressed("hands_up")


def test_hands_up_has_hysteresis():
    mapper = ControlMapper(SIZE)
    high = standing(left_wrist=(540.0, 60.0), right_wrist=(740.0, 60.0))
    # Throttle ~0.46: below the 0.55 on-threshold, above the 0.40 off-threshold.
    mid = standing(left_wrist=(540.0, 150.0), right_wrist=(740.0, 150.0))

    assert not mapper.map(mid).hands_up, "should not turn on below on-threshold"
    assert mapper.map(high).hands_up
    assert mapper.map(mid).hands_up, "should stay on until it drops below off-threshold"


def test_steer_is_scale_invariant():
    """The same gesture farther from the camera gives the same signal."""

    def shifted(scale: float) -> float:
        cx, cy = 640.0, 360.0
        pts = {
            "left_shoulder": (cx - 100 * scale, cy - 100 * scale),
            "right_shoulder": (cx + 100 * scale, cy - 100 * scale),
            "left_hip": (cx - 80 * scale, cy + 140 * scale),
            "right_hip": (cx + 80 * scale, cy + 140 * scale),
            "left_wrist": (cx + 50 * scale, cy + 100 * scale),
            "right_wrist": (cx + 250 * scale, cy + 100 * scale),
        }
        return ControlMapper(SIZE).map(make_pose(**pts)).steer

    near, far = shifted(1.0), shifted(0.5)
    assert abs(near - far) < 0.02, f"{near} vs {far} should match across distance"


def test_falls_back_to_lean_when_hands_hidden():
    pose = make_pose(
        left_shoulder=(640.0, 260.0),
        right_shoulder=(840.0, 260.0),
        left_hip=(560.0, 500.0),
        right_hip=(720.0, 500.0),
    )
    state = ControlMapper(SIZE).map(pose)
    assert state.present
    assert state.steer > 0.1, "torso leaning right should still steer"


def test_smoother_blends_toward_previous():
    smoother = PoseSmoother(alpha=0.5)
    first = standing()
    smoother.apply([first])
    moved = standing(left_wrist=(340.0, 460.0))
    out = smoother.apply([moved])[0]
    x = out.xy[KP["left_wrist"]][0]
    assert 340.0 < x < 540.0, f"smoothed value {x} should sit between frames"


def test_smoother_disabled_passes_through():
    smoother = PoseSmoother(alpha=0.0)
    smoother.apply([standing()])
    moved = standing(left_wrist=(340.0, 460.0))
    assert smoother.apply([moved])[0].xy[KP["left_wrist"]][0] == 340.0


# --- velocity extrapolation ---


def moving_result(x: float, t: float) -> "Result":
    from kpapp.inference import Result

    pose = standing(left_wrist=(x, 460.0), right_wrist=(x + 200.0, 460.0))
    return Result(poses=[pose], frame_seq=int(t * 1000), timestamp=t)


def test_extrapolates_linear_motion():
    """A wrist moving 1000 px/s should be projected ahead by gain * lead."""
    from kpapp.inference import PoseExtrapolator

    ex = PoseExtrapolator(gain=1.0, velocity_smoothing=0.0)
    ex.update(moving_result(400.0, 0.00))
    ex.update(moving_result(445.0, 0.045))  # +45px in 45ms == 1000 px/s

    poses = ex.poses_at(0.045 + 0.040)  # 40 ms after the last result
    x = poses[0].xy[KP["left_wrist"]][0]
    assert abs(x - 485.0) < 1.0, f"expected ~485 (445 + 1000*0.04), got {x}"


def test_gain_scales_the_projection():
    from kpapp.inference import PoseExtrapolator

    def project(gain: float) -> float:
        ex = PoseExtrapolator(gain=gain, velocity_smoothing=0.0)
        ex.update(moving_result(400.0, 0.00))
        ex.update(moving_result(445.0, 0.045))
        return ex.poses_at(0.045 + 0.040)[0].xy[KP["left_wrist"]][0]

    full, half = project(1.0) - 445.0, project(0.5) - 445.0
    assert abs(half - full / 2) < 0.5, f"gain 0.5 should halve the lead: {half} vs {full}"


def test_lead_is_clamped():
    """A stalled worker must not fling the skeleton off-screen."""
    from kpapp.inference import PoseExtrapolator

    ex = PoseExtrapolator(gain=1.0, max_lead=0.12, velocity_smoothing=0.0)
    ex.update(moving_result(400.0, 0.00))
    ex.update(moving_result(445.0, 0.045))

    far = ex.poses_at(0.045 + 5.0)[0].xy[KP["left_wrist"]][0]
    assert abs(far - (445.0 + 1000.0 * 0.12)) < 1.0, f"lead not clamped: {far}"


def test_long_gap_resets_velocity():
    from kpapp.inference import PoseExtrapolator

    ex = PoseExtrapolator(gain=1.0, max_gap=0.25, velocity_smoothing=0.0)
    ex.update(moving_result(400.0, 0.0))
    ex.update(moving_result(445.0, 2.0))  # 2s gap: velocity is meaningless
    x = ex.poses_at(2.02)[0].xy[KP["left_wrist"]][0]
    assert x == 445.0, f"stale velocity should be dropped, got {x}"


def test_disabled_extrapolation_passes_through():
    from kpapp.inference import PoseExtrapolator

    ex = PoseExtrapolator(gain=0.0)
    ex.update(moving_result(400.0, 0.00))
    ex.update(moving_result(445.0, 0.045))
    assert ex.poses_at(0.10)[0].xy[KP["left_wrist"]][0] == 445.0


def test_does_not_mutate_published_poses():
    """Worker poses are shared across threads; projection must copy."""
    from kpapp.inference import PoseExtrapolator

    ex = PoseExtrapolator(gain=1.0, velocity_smoothing=0.0)
    ex.update(moving_result(400.0, 0.00))
    last = moving_result(445.0, 0.045)
    ex.update(last)
    before = last.poses[0].xy[KP["left_wrist"]][0]
    ex.poses_at(0.045 + 0.040)
    assert last.poses[0].xy[KP["left_wrist"]][0] == before


def test_reappearing_keypoint_is_not_flung():
    """A keypoint that was hidden has no history, so it must not get velocity."""
    from kpapp.inference import PoseExtrapolator, Result

    hidden = standing()
    hidden.confidence[KP["left_wrist"]] = 0.0
    ex = PoseExtrapolator(gain=1.0, velocity_smoothing=0.0)
    ex.update(Result(poses=[hidden], timestamp=0.0))

    visible = standing(left_wrist=(900.0, 460.0))  # snaps in far away
    ex.update(Result(poses=[visible], timestamp=0.045))
    x = ex.poses_at(0.045 + 0.040)[0].xy[KP["left_wrist"]][0]
    assert x == 900.0, f"reappearing keypoint should not extrapolate, got {x}"
