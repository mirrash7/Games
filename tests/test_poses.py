"""Sanity checks on the hole-shape library.

These are anatomy tests, not rendering tests: a hole the player physically
cannot make is a bug, and so is a hole whose left arm is longer than its right.
Because every pose is normalised to fill the unit box, absolute limb lengths
differ between poses - so cross-pose checks compare each limb to that pose's
own torso length, which is the one thing a real body keeps constant.
"""

from __future__ import annotations

import numpy as np
import pytest

from kpapp.config import KEYPOINT_NAMES, KP
from kpapp.game.holeinwall.core import TargetPose
from kpapp.game.holeinwall.poses import TARGET_POSES, poses_for_difficulty


def seg(xy: np.ndarray, a: str, b: str) -> float:
    return float(np.linalg.norm(xy[KP[a]] - xy[KP[b]]))


def torso(xy: np.ndarray) -> float:
    shoulder = (xy[KP["left_shoulder"]] + xy[KP["right_shoulder"]]) * 0.5
    hip = (xy[KP["left_hip"]] + xy[KP["right_hip"]]) * 0.5
    return float(np.linalg.norm(shoulder - hip))


# (name, endpoint a, endpoint b) for each bone that should be mirrored L/R.
PAIRED = (
    ("upper_arm", "shoulder", "elbow"),
    ("forearm", "elbow", "wrist"),
    ("thigh", "hip", "knee"),
    ("shin", "knee", "ankle"),
)

ALL = pytest.mark.parametrize("pose", TARGET_POSES, ids=[p.name for p in TARGET_POSES])


def test_library_is_populated() -> None:
    assert 12 <= len(TARGET_POSES) <= 16
    assert all(isinstance(p, TargetPose) for p in TARGET_POSES)


def test_names_are_unique_and_punchy() -> None:
    names = [p.name for p in TARGET_POSES]
    assert len(set(names)) == len(names)
    for name in names:
        assert name == name.upper()
        assert 0 < len(name) <= 12


def test_every_difficulty_is_represented() -> None:
    for level in (1, 2, 3):
        assert poses_for_difficulty(level), f"no poses at difficulty {level}"


@ALL
def test_array_contract(pose: TargetPose) -> None:
    kp = pose.keypoints
    assert kp.shape == (len(KEYPOINT_NAMES), 2) == (17, 2)
    assert kp.dtype == np.float32
    assert np.isfinite(kp).all()


@ALL
def test_inside_the_unit_box(pose: TargetPose) -> None:
    kp = pose.keypoints
    assert kp.min() >= 0.0
    assert kp.max() <= 1.0


@ALL
def test_figure_fills_the_box(pose: TargetPose) -> None:
    """A hole that occupies a corner of the wall would be no fun."""
    extent = pose.keypoints.max(axis=0) - pose.keypoints.min(axis=0)
    assert extent.max() > 0.7
    assert extent.min() > 0.2


@ALL
def test_difficulty_and_clearance(pose: TargetPose) -> None:
    assert pose.difficulty in (1, 2, 3)
    assert 1.0 <= pose.clearance <= 1.5
    if pose.difficulty == 1:
        assert pose.clearance == 1.0


def test_hard_poses_are_forgiving() -> None:
    hard = poses_for_difficulty(3)
    assert all(p.clearance > 1.0 for p in hard), "difficulty 3 needs extra clearance"


@ALL
def test_left_right_limbs_match(pose: TargetPose) -> None:
    """Nobody has one arm 25% longer than the other."""
    kp = pose.keypoints
    for label, a, b in PAIRED:
        left = seg(kp, f"left_{a}", f"left_{b}")
        right = seg(kp, f"right_{a}", f"right_{b}")
        assert left == pytest.approx(right, rel=0.25), f"{pose.name}: {label} L/R mismatch"


@ALL
def test_proportions_are_human(pose: TargetPose) -> None:
    """Upper arm ~ forearm, thigh ~ shin, and the arm shorter than the leg."""
    kp = pose.keypoints
    t = torso(kp)
    upper = seg(kp, "left_shoulder", "left_elbow")
    fore = seg(kp, "left_elbow", "left_wrist")
    thigh = seg(kp, "left_hip", "left_knee")
    shin = seg(kp, "left_knee", "left_ankle")

    assert 0.8 <= upper / fore <= 1.3
    assert 0.85 <= thigh / shin <= 1.2
    assert (upper + fore) < (thigh + shin)
    assert 0.5 <= upper / t <= 0.8
    assert 0.7 <= thigh / t <= 1.1


@pytest.mark.parametrize("label,a,b", PAIRED, ids=[p[0] for p in PAIRED])
def test_limb_lengths_are_constant_across_the_library(label: str, a: str, b: str) -> None:
    """The same person strikes every pose, so bone/torso ratios must not drift."""
    ratios = [
        seg(p.keypoints, f"{side}_{a}", f"{side}_{b}") / torso(p.keypoints)
        for p in TARGET_POSES
        for side in ("left", "right")
    ]
    lo, hi = min(ratios), max(ratios)
    assert hi / lo < 1.05, f"{label} ratio varies {lo:.3f}..{hi:.3f} across poses"


@ALL
def test_head_sits_above_the_shoulders(pose: TargetPose) -> None:
    kp = pose.keypoints
    shoulder_mid = (kp[KP["left_shoulder"]] + kp[KP["right_shoulder"]]) * 0.5
    for part in ("nose", "left_eye", "right_eye", "left_ear", "right_ear"):
        assert kp[KP[part]][1] < shoulder_mid[1], f"{pose.name}: {part} below shoulders"


@ALL
def test_mirrored_view_sides_are_consistent(pose: TargetPose) -> None:
    """Selfie view: the player's left is on the right of the image."""
    kp = pose.keypoints
    assert kp[KP["left_shoulder"]][0] > kp[KP["right_shoulder"]][0]
    assert kp[KP["left_hip"]][0] > kp[KP["right_hip"]][0]
    assert kp[KP["left_eye"]][0] > kp[KP["right_eye"]][0]
    assert kp[KP["left_ear"]][0] > kp[KP["right_ear"]][0]


@ALL
def test_torso_is_upright_enough_to_stand_in(pose: TargetPose) -> None:
    """Hips below shoulders, i.e. the player is on their feet."""
    kp = pose.keypoints
    shoulder_mid = (kp[KP["left_shoulder"]] + kp[KP["right_shoulder"]]) * 0.5
    hip_mid = (kp[KP["left_hip"]] + kp[KP["right_hip"]]) * 0.5
    delta = hip_mid - shoulder_mid
    assert delta[1] > 0
    lean = abs(np.degrees(np.arctan2(delta[0], delta[1])))
    assert lean <= 30.0, f"{pose.name} leans {lean:.0f} degrees"


LIMB_ENDS = [
    KP[k]
    for k in (
        "left_elbow", "right_elbow", "left_wrist", "right_wrist",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    )
]


def test_poses_are_visually_distinct() -> None:
    """No two holes may be near-duplicates, or the player cannot tell them apart.

    Compared on the limb endpoints only, in torso units: heads and torsos are
    near-identical everywhere, so including them would drown out exactly the
    difference the player has to read.
    """
    normed = []
    for p in TARGET_POSES:
        kp = p.keypoints.astype(np.float64)
        hip_mid = (kp[KP["left_hip"]] + kp[KP["right_hip"]]) * 0.5
        normed.append((kp - hip_mid) / torso(kp))
    for i, a in enumerate(normed):
        for j, b in enumerate(normed[i + 1 :], start=i + 1):
            dist = float(np.linalg.norm(a[LIMB_ENDS] - b[LIMB_ENDS], axis=1).mean())
            assert dist > 0.2, (
                f"{TARGET_POSES[i].name} and {TARGET_POSES[j].name} are too alike "
                f"({dist:.3f})"
            )


@ALL
def test_scaled_stays_inside_its_box(pose: TargetPose) -> None:
    pts = pose.scaled(192, 144, margin=0.05)
    assert pts.dtype == np.float32
    assert pts[:, 0].min() >= 0 and pts[:, 0].max() <= 192
    assert pts[:, 1].min() >= 0 and pts[:, 1].max() <= 144
