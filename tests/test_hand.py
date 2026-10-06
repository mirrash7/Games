"""The hand cursor: on the palm, and steady."""

from __future__ import annotations

import numpy as np

from kpapp.config import KP
from kpapp.hand import PALM_REACH, HandTracker, OneEuroFilter, hand_point
from kpapp.inference import Pose

WRIST = np.array([860.0, 320.0])
ELBOW = np.array([820.0, 520.0])


def arm(elbow: bool = True, shoulders: bool = True, noise: float = 0.0, rng=None) -> Pose:
    """Player's raised right hand (left_wrist on a mirrored feed)."""
    rng = rng or np.random.default_rng(0)
    xy = np.zeros((17, 2), np.float32)
    c = np.zeros(17, np.float32)
    pts = {"left_wrist": WRIST, "left_elbow": ELBOW}
    if shoulders:
        pts |= {"left_shoulder": np.array([700.0, 420.0]), "right_shoulder": np.array([520.0, 420.0])}
    for name, p in pts.items():
        xy[KP[name]] = p + rng.normal(0, noise, 2)
        c[KP[name]] = 0.9
    if not elbow:
        c[KP["left_elbow"]] = 0.1
    return Pose(xy=xy, confidence=c, score=0.9)


def test_dot_sits_past_the_wrist_toward_the_fingers():
    p = hand_point(arm())
    expected = WRIST + (WRIST - ELBOW) * PALM_REACH
    assert np.allclose(p, expected, atol=1e-3)
    assert p[1] < WRIST[1], "the palm is beyond the wrist, away from the elbow"


def test_missing_elbow_does_not_snap_to_the_wrist():
    """Without an elbow the old cursor fell back to the wrist - the base of the hand."""
    p = hand_point(arm(elbow=False))
    assert np.linalg.norm(p - WRIST) > 30, "fell back to the bare wrist"


def test_only_wrist_visible_still_returns_something():
    p = hand_point(arm(elbow=False, shoulders=False))
    assert np.allclose(p, WRIST)


def test_tracker_remembers_the_forearm_through_dropouts():
    tr = HandTracker(smooth=False)
    with_elbow = tr.update(arm(), 0.0)
    without = tr.update(arm(elbow=False), 0.1)
    assert np.allclose(with_elbow, without, atol=1e-3), "dot moved when only the elbow vanished"


def test_tracker_forgets_a_stale_forearm():
    tr = HandTracker(smooth=False)
    tr.update(arm(), 0.0)
    later = tr.update(arm(elbow=False), 5.0)
    assert np.allclose(later, hand_point(arm(elbow=False))), "used a forearm from 5 s ago"


def test_flickering_elbow_does_not_shake_the_dot():
    """The measured cause of the jittery path.

    On recorded play the elbow was missing in 54% of frames with the wrist
    visible; the old cursor jumped ~39 px per frame between palm and wrist.
    """
    rng = np.random.default_rng(1)
    tr = HandTracker()
    pts = [tr.update(arm(elbow=bool(rng.random() > 0.5), noise=3.0, rng=rng), i / 60)
           for i in range(120)]
    jumps = np.linalg.norm(np.diff(np.array(pts[10:]), axis=0), axis=1)
    assert jumps.mean() < 4.0, f"dot shakes {jumps.mean():.1f} px/frame"
    assert jumps.max() < 12.0


def test_filter_steadies_a_still_hand_without_dragging_a_swipe():
    rng = np.random.default_rng(0)
    f = OneEuroFilter()
    still = [f(np.array([640.0, 360.0]) + rng.normal(0, 6, 2), i / 60) for i in range(120)]
    jitter = np.std(np.diff(np.array(still[30:]), axis=0), axis=0).mean()
    assert jitter < 3.0, f"still hand shimmers {jitter:.1f} px"

    f = OneEuroFilter()
    lags = []
    for i in range(60):
        t = i / 60
        truth = np.array([200 + 2000 * t, 360.0])
        lags.append(abs(f(truth, t)[0] - truth[0]))
    assert np.mean(lags[20:]) < 25.0, f"a 2000 px/s swipe lags {np.mean(lags[20:]):.0f} px"


def test_losing_the_hand_resets_the_tracker():
    tr = HandTracker()
    tr.update(arm(), 0.0)
    assert tr.update(None, 0.1) is None
    # Reacquiring far away must not glide in from the old position.
    xy = arm().xy.copy()
    xy[KP["left_wrist"]] += 400
    xy[KP["left_elbow"]] += 400
    p = arm(); p.xy = xy
    out = tr.update(p, 0.2)
    assert np.linalg.norm(out - (xy[KP["left_wrist"]] + (xy[KP["left_wrist"]] - xy[KP["left_elbow"]]) * PALM_REACH)) < 1e-3


def test_missing_elbow_assumes_a_raised_hand():
    """With the elbow below the frame, the palm is directly above the wrist."""
    p = hand_point(arm(elbow=False))
    assert abs(p[0] - WRIST[0]) < 1e-3, "fallback should not drift sideways"
    assert p[1] < WRIST[1] - 30, "palm should sit above the wrist"
