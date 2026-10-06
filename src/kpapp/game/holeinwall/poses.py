"""The library of hole shapes the wall can carry.

Every hole is a full-body human pose, so the poses are *built* rather than
typed out: a single set of body proportions (``_UPPER_ARM``, ``_THIGH``, ...)
feeds a small forward-kinematics helper, and each entry in ``_SPECS`` only says
where the joints point. That keeps limb lengths identical across the whole
library - the same person contorting - which hand-written coordinate tables
never manage.

Angles are in degrees in screen terms: 90 is up, 0 is +x, -90 is down. Note
that the game runs on a mirrored (selfie) camera, so a keypoint named "left"
belongs to the player's left hand and therefore appears on the *right* of the
image, at larger x. All the angle tables below are written in image terms with
that already accounted for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ...config import KP
from .core import NUM_KEYPOINTS, TargetPose

# --- body proportions, in units of torso length (shoulder mid -> hip mid) ---
_SHOULDER_HALF = 0.40
_HIP_HALF = 0.23
_UPPER_ARM = 0.62
_FOREARM = 0.56
_THIGH = 0.88
_SHIN = 0.84

# Head geometry mirrors what silhouette.py draws, so the canonical box can be
# padded to fit the rendered shape rather than just the keypoints.
_HEAD_R = 0.42
_HEAD_UP = _HEAD_R * 0.95

# Half-widths used only for padding the bounding box before normalisation.
_ARM_PAD = 0.11
_LEG_PAD = 0.14


def _dir(deg: float) -> np.ndarray:
    """Unit vector for a screen-space angle: 90 up, 0 right, -90 down."""
    r = math.radians(deg)
    return np.array([math.cos(r), -math.sin(r)], dtype=np.float64)


def _chain(root: np.ndarray, a1: float, l1: float, a2: float, l2: float):
    """Two-segment limb from absolute angles. Returns (joint, end)."""
    joint = root + _dir(a1) * l1
    end = joint + _dir(a2) * l2
    return joint, end


def _reach(root: np.ndarray, target: np.ndarray, l1: float, l2: float, bend: float):
    """Two-bone IK: bend the limb so its end lands on ``target``.

    ``bend`` is +1/-1 and picks which way the elbow (or knee) bulges. Used for
    poses defined by where the hands go - on the hips, clasped overhead - where
    guessing angles would leave the hands floating off the body.
    """
    delta = target - root
    d = float(np.linalg.norm(delta))
    d = min(max(d, abs(l1 - l2) + 1e-3), l1 + l2 - 1e-3)
    n = delta / max(float(np.linalg.norm(delta)), 1e-9)
    a = (d * d + l1 * l1 - l2 * l2) / (2.0 * d)
    h = math.sqrt(max(l1 * l1 - a * a, 0.0))
    perp = np.array([-n[1], n[0]], dtype=np.float64)
    joint = root + n * a + perp * (h * bend)
    end = root + n * d
    return joint, end


# A limb spec is either absolute angles ("ang", a1, a2) or an IK target
# ("ik", tx, ty, bend) given in torso units relative to the mid-hip, x right,
# y *up* (so positive y is toward the head - easier to reason about).
Limb = tuple


@dataclass(frozen=True)
class _Spec:
    name: str
    difficulty: int
    left_arm: Limb
    right_arm: Limb
    left_leg: Limb
    right_leg: Limb
    lean: float = 0.0  # torso tilt, degrees; positive leans the head toward +x
    clearance: float = 1.0
    note: str = ""


def _limb_points(root: np.ndarray, spec: Limb, l1: float, l2: float):
    kind = spec[0]
    if kind == "ang":
        return _chain(root, spec[1], l1, spec[2], l2)
    if kind == "ik":
        target = np.array([spec[1], -spec[2]], dtype=np.float64)
        return _reach(root, target, l1, l2, spec[3])
    raise ValueError(f"unknown limb spec {kind!r}")


def _build(spec: _Spec) -> np.ndarray:
    """Spec -> (17, 2) keypoints in canonical 0..1 space."""
    xy = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float64)

    # Torso. Mid-hip at the origin, mid-shoulder one torso length "up" along
    # the lean direction; everything else hangs off those two.
    hip_mid = np.zeros(2, dtype=np.float64)
    up = _dir(90.0 + spec.lean)
    shoulder_mid = hip_mid + up
    side = np.array([-up[1], up[0]], dtype=np.float64)  # points toward +x

    l_sh = shoulder_mid + side * _SHOULDER_HALF
    r_sh = shoulder_mid - side * _SHOULDER_HALF
    l_hip = hip_mid + side * _HIP_HALF
    r_hip = hip_mid - side * _HIP_HALF

    l_el, l_wr = _limb_points(l_sh, spec.left_arm, _UPPER_ARM, _FOREARM)
    r_el, r_wr = _limb_points(r_sh, spec.right_arm, _UPPER_ARM, _FOREARM)
    l_kn, l_an = _limb_points(l_hip, spec.left_leg, _THIGH, _SHIN)
    r_kn, r_an = _limb_points(r_hip, spec.right_leg, _THIGH, _SHIN)

    head_c = shoulder_mid + up * _HEAD_UP
    xy[KP["nose"]] = head_c + up * 0.04
    xy[KP["left_eye"]] = head_c + up * 0.12 + side * 0.11
    xy[KP["right_eye"]] = head_c + up * 0.12 - side * 0.11
    xy[KP["left_ear"]] = head_c + up * 0.05 + side * 0.20
    xy[KP["right_ear"]] = head_c + up * 0.05 - side * 0.20
    xy[KP["left_shoulder"]] = l_sh
    xy[KP["right_shoulder"]] = r_sh
    xy[KP["left_elbow"]] = l_el
    xy[KP["right_elbow"]] = r_el
    xy[KP["left_wrist"]] = l_wr
    xy[KP["right_wrist"]] = r_wr
    xy[KP["left_hip"]] = l_hip
    xy[KP["right_hip"]] = r_hip
    xy[KP["left_knee"]] = l_kn
    xy[KP["right_knee"]] = r_kn
    xy[KP["left_ankle"]] = l_an
    xy[KP["right_ankle"]] = r_an

    return _normalise(xy, head_c)


def _normalise(xy: np.ndarray, head_c: np.ndarray) -> np.ndarray:
    """Fit the *rendered* shape - not just the joints - into the unit square.

    Limbs are drawn with thickness and the head as a disc, so padding each
    joint by its limb half-width keeps the silhouette from clipping while still
    letting the figure fill the box.
    """
    pads = np.full(NUM_KEYPOINTS, 0.06)
    for k in ("left_elbow", "right_elbow", "left_wrist", "right_wrist",
              "left_shoulder", "right_shoulder"):
        pads[KP[k]] = _ARM_PAD
    for k in ("left_knee", "right_knee", "left_ankle", "right_ankle",
              "left_hip", "right_hip"):
        pads[KP[k]] = _LEG_PAD

    lo = np.min(xy - pads[:, None], axis=0)
    hi = np.max(xy + pads[:, None], axis=0)
    lo = np.minimum(lo, head_c - _HEAD_R)
    hi = np.maximum(hi, head_c + _HEAD_R)

    extent = np.maximum(hi - lo, 1e-6)
    scale = 1.0 / float(np.max(extent))
    out = (xy - lo) * scale
    # Centre whichever axis has slack left over.
    out += (1.0 - extent * scale) * 0.5
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# --- the library ------------------------------------------------------------
# Arms first, then legs; every entry is written looking at the mirrored image,
# so "left_*" values live on the right-hand side of the picture.

_SPECS: tuple[_Spec, ...] = (
    _Spec(
        name="I-POSE",
        difficulty=1,
        left_arm=("ang", -62.0, -70.0),
        right_arm=("ang", -118.0, -110.0),
        left_leg=("ang", -83.0, -86.0),
        right_leg=("ang", -97.0, -94.0),
        note="stand tall, arms just clear of the body",
    ),
    _Spec(
        name="T-POSE",
        difficulty=1,
        left_arm=("ang", 4.0, 0.0),
        right_arm=("ang", 176.0, 180.0),
        left_leg=("ang", -81.0, -85.0),
        right_leg=("ang", -99.0, -95.0),
    ),
    _Spec(
        name="Y-POSE",
        difficulty=1,
        left_arm=("ang", 56.0, 60.0),
        right_arm=("ang", 124.0, 120.0),
        left_leg=("ang", -82.0, -86.0),
        right_leg=("ang", -98.0, -94.0),
    ),
    _Spec(
        name="STAR",
        difficulty=1,
        left_arm=("ang", 38.0, 40.0),
        right_arm=("ang", 142.0, 140.0),
        left_leg=("ang", -58.0, -60.0),
        right_leg=("ang", -122.0, -120.0),
    ),
    _Spec(
        name="CACTUS",
        difficulty=1,
        left_arm=("ang", 10.0, 88.0),
        right_arm=("ang", 170.0, 92.0),
        left_leg=("ang", -72.0, -82.0),
        right_leg=("ang", -108.0, -98.0),
        note="goalpost arms: elbows out at shoulder height, hands straight up",
    ),
    _Spec(
        name="AKIMBO",
        difficulty=2,
        left_arm=("ik", 0.30, 0.12, -1.0),
        right_arm=("ik", -0.30, 0.12, 1.0),
        left_leg=("ang", -74.0, -83.0),
        right_leg=("ang", -106.0, -97.0),
        note="hands on hips - the triangles under the arms are the tell",
    ),
    _Spec(
        name="TEAPOT",
        difficulty=2,
        left_arm=("ang", 62.0, 78.0),
        right_arm=("ang", 178.0, 176.0),
        left_leg=("ang", -76.0, -84.0),
        right_leg=("ang", -104.0, -96.0),
        note="one arm straight up, the other straight out",
    ),
    _Spec(
        name="DIAGONAL",
        difficulty=2,
        left_arm=("ang", 48.0, 52.0),
        right_arm=("ang", -128.0, -132.0),
        left_leg=("ang", -66.0, -68.0),
        right_leg=("ang", -104.0, -100.0),
        note="a single slash from raised hand to trailing foot",
    ),
    _Spec(
        name="LEAN",
        difficulty=2,
        lean=-24.0,
        left_arm=("ang", -40.0, -62.0),
        right_arm=("ang", 108.0, 72.0),
        left_leg=("ang", -84.0, -88.0),
        right_leg=("ang", -98.0, -94.0),
        note="crescent side bend, top arm reaching over the head",
    ),
    _Spec(
        name="SUMO",
        difficulty=2,
        left_arm=("ang", 4.0, 84.0),
        right_arm=("ang", 176.0, 96.0),
        left_leg=("ang", -32.0, -84.0),
        right_leg=("ang", -148.0, -96.0),
        note="wide squat, knees out over the toes",
    ),
    _Spec(
        name="ANCHOR",
        difficulty=2,
        left_arm=("ang", 14.0, -74.0),
        right_arm=("ang", 166.0, -106.0),
        left_leg=("ang", -86.0, -88.0),
        right_leg=("ang", -94.0, -92.0),
        note="elbows out at shoulder height, forearms swept down and out",
    ),
    _Spec(
        name="FLAMINGO",
        difficulty=2,
        left_arm=("ang", 12.0, 8.0),
        right_arm=("ang", 168.0, 172.0),
        left_leg=("ang", -36.0, -38.0),
        right_leg=("ang", -92.0, -90.0),
        clearance=1.1,
        note="one leg lifted out to the side, arms out to balance",
    ),
    _Spec(
        name="RUNNER",
        difficulty=3,
        left_arm=("ang", -78.0, -18.0),
        right_arm=("ang", 128.0, 58.0),
        left_leg=("ang", -26.0, -92.0),
        right_leg=("ang", -114.0, -120.0),
        clearance=1.2,
        note="mid-stride: front knee driven up, opposite arm forward",
    ),
    _Spec(
        name="ARCHER",
        difficulty=3,
        left_arm=("ang", 6.0, 2.0),
        right_arm=("ang", 140.0, -30.0),
        left_leg=("ang", -40.0, -88.0),
        right_leg=("ang", -132.0, -104.0),
        clearance=1.15,
        note="bow arm straight out, draw hand pulled back past the chin",
    ),
    _Spec(
        name="KARATE",
        difficulty=3,
        left_arm=("ang", 46.0, 20.0),
        right_arm=("ang", -128.0, -140.0),
        left_leg=("ang", -14.0, -10.0),
        right_leg=("ang", -94.0, -90.0),
        clearance=1.25,
        note="side kick, leading arm tracking the foot",
    ),
    _Spec(
        name="ZIGZAG",
        difficulty=3,
        left_arm=("ang", -6.0, -94.0),
        right_arm=("ang", 176.0, 98.0),
        left_leg=("ang", -46.0, -104.0),
        right_leg=("ang", -118.0, -80.0),
        clearance=1.2,
        note="right angles everywhere, each one folded the opposite way",
    ),
)


TARGET_POSES: list[TargetPose] = [
    TargetPose(
        name=s.name,
        keypoints=_build(s),
        difficulty=s.difficulty,
        clearance=s.clearance,
    )
    for s in _SPECS
]

POSES_BY_NAME: dict[str, TargetPose] = {p.name: p for p in TARGET_POSES}


def poses_for_difficulty(level: int) -> list[TargetPose]:
    """Every hole at a given difficulty, in library order."""
    return [p for p in TARGET_POSES if p.difficulty == level]


__all__ = ["TARGET_POSES", "POSES_BY_NAME", "poses_for_difficulty"]
