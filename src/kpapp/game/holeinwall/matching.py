"""Does the player fit through the hole?

The test is literal rather than geometric: render the player's silhouette and
the hole into the same small raster and ask what fraction of the player lies
inside the hole. That mirrors what the player sees, so the score never
disagrees with the picture — a limb sticking out of the wall is exactly the
thing that costs points.

Both silhouettes are drawn at a FIXED pixels-per-torso scale rather than being
fitted to the frame. Fitting each pose to the box would silently rescale a
narrow pose up to the width of a wide one and call them a match.
"""

from __future__ import annotations

import numpy as np

from ...config import KP
from .core import FitResult, TargetPose, to_torso_space
from .core import MIN_CONF
from .silhouette import draw_silhouette

# Square working raster, sized so a spread-eagle pose fits with room to spare.
MATCH_RASTER = (160, 160)
PX_PER_TORSO = 32.0
# Mid-hip sits at torso-space origin, but the body extends further below it
# than above, so the raster is centred slightly below the hips.
Y_OFFSET = 0.3
# Hole padding in torso units. Generous enough that a good-faith attempt gets
# through; the game reads as unfair well before it reads as too easy.
BASE_CLEARANCE = 0.30

# Blend weights and thresholds, calibrated against deliberately wrong poses:
# an identical pose scores 1.00, a small error 0.95, one wrong arm 0.85,
# arms-down-instead-of-out 0.71.
JOINT_WEIGHT = 0.6
JOINT_TOLERANCE = 0.55  # torso units at which a joint scores zero
PASS_THRESHOLD = 0.82
PERFECT_THRESHOLD = 0.93

# Extremities carry the shape; a misplaced wrist should cost more than a
# misplaced hip, which barely moves under any pose.
_JOINTS = [
    KP["left_shoulder"], KP["right_shoulder"], KP["left_elbow"], KP["right_elbow"],
    KP["left_wrist"], KP["right_wrist"], KP["left_hip"], KP["right_hip"],
    KP["left_knee"], KP["right_knee"], KP["left_ankle"], KP["right_ankle"],
]
_JOINT_WEIGHTS = np.array(
    [0.6, 0.6, 1.0, 1.0, 1.4, 1.4, 0.6, 0.6, 1.0, 1.0, 1.2, 1.2], dtype=np.float32
)


def to_raster_pixels(
    torso_xy: np.ndarray,
    size: tuple[int, int] = MATCH_RASTER,
    px_per_torso: float = PX_PER_TORSO,
) -> np.ndarray:
    """Torso space -> raster pixels at a fixed, pose-independent scale."""
    out = torso_xy * px_per_torso
    out[:, 0] += size[0] * 0.5
    out[:, 1] += size[1] * 0.5 - Y_OFFSET * px_per_torso
    return out.astype(np.float32)


def build_hole_mask(
    target: TargetPose,
    size: tuple[int, int] = MATCH_RASTER,
    px_per_torso: float = PX_PER_TORSO,
    clearance: float = BASE_CLEARANCE,
) -> np.ndarray:
    """The opening for a target pose: its silhouette, grown by clearance."""
    torso = to_torso_space(target.keypoints)
    if torso is None:
        raise ValueError(f"target pose {target.name!r} has a degenerate torso")
    px = to_raster_pixels(torso, size, px_per_torso)
    pad = clearance * target.clearance * px_per_torso
    return draw_silhouette(px, size, thickness=1.0, pad=pad)


def build_target_mask(target: TargetPose) -> np.ndarray:
    """The bare target silhouette, with no clearance - the shape to match."""
    torso = to_torso_space(target.keypoints)
    if torso is None:
        raise ValueError(f"target pose {target.name!r} has a degenerate torso")
    return draw_silhouette(to_raster_pixels(torso), MATCH_RASTER, thickness=1.0)


class HoleCache:
    """Masks are static per target; build each one once."""

    def __init__(self, clearance: float = BASE_CLEARANCE) -> None:
        self.clearance = clearance
        self._hole: dict[str, np.ndarray] = {}
        self._target: dict[str, np.ndarray] = {}
        self._torso: dict[str, np.ndarray] = {}

    def hole(self, target: TargetPose) -> np.ndarray:
        if target.name not in self._hole:
            self._hole[target.name] = build_hole_mask(target, clearance=self.clearance)
        return self._hole[target.name]

    def target_mask(self, target: TargetPose, conf: np.ndarray | None = None) -> np.ndarray:
        """Target silhouette, drawn with the SAME visibility as the player.

        Otherwise the target keeps limbs the camera cannot see while the player
        loses them, and the overlap collapses for reasons the player cannot do
        anything about: a flawless pose with both knees out of frame scored
        0.63 IoU purely because the target still had legs.
        """
        key = target.name if conf is None else (
            target.name + (conf >= MIN_CONF).tobytes().hex()
        )
        if key not in self._target:
            torso = self.torso(target)
            self._target[key] = draw_silhouette(
                to_raster_pixels(torso), MATCH_RASTER, conf=conf, thickness=1.0
            )
        return self._target[key]

    def torso(self, target: TargetPose) -> np.ndarray:
        if target.name not in self._torso:
            self._torso[target.name] = to_torso_space(target.keypoints)
        return self._torso[target.name]


def player_mask(
    xy: np.ndarray,
    conf: np.ndarray,
    size: tuple[int, int] = MATCH_RASTER,
    px_per_torso: float = PX_PER_TORSO,
) -> np.ndarray | None:
    """Player silhouette in the same raster as the hole, or None if untracked."""
    torso = to_torso_space(xy, conf)
    if torso is None:
        return None
    return draw_silhouette(
        to_raster_pixels(torso, size, px_per_torso), size, conf=conf, thickness=1.0
    )


def _diagnose(conf: np.ndarray | None) -> str:
    """Turn a tracking failure into something the player can act on."""
    if conf is None:
        return "NO PLAYER DETECTED"
    missing = [n for n in ("left_shoulder", "right_shoulder") if conf[KP[n]] < MIN_CONF]
    if missing:
        return "NO PLAYER DETECTED"
    if conf[KP["left_hip"]] < MIN_CONF or conf[KP["right_hip"]] < MIN_CONF:
        return "STEP BACK - HIPS MUST BE IN FRAME"
    return "STEP BACK - FULL BODY NEEDED"


def joint_agreement(
    player_torso: np.ndarray,
    target_torso: np.ndarray,
    conf: np.ndarray,
    tolerance: float = JOINT_TOLERANCE,
) -> float | None:
    """Per-joint match, weighted equally per limb rather than by pixel mass.

    Silhouette overlap alone is dominated by the torso, so a single wrong arm
    barely registers. This metric is what actually notices a wrong limb.
    Joints the model cannot see are dropped rather than counted as errors.
    """
    visible = conf[_JOINTS] >= MIN_CONF
    if visible.sum() < 6:
        return None
    d = np.linalg.norm(player_torso[_JOINTS] - target_torso[_JOINTS], axis=1)
    per_joint = np.clip(1.0 - d / tolerance, 0.0, 1.0)
    w = _JOINT_WEIGHTS * visible
    total = float(w.sum())
    return float((w * per_joint).sum() / total) if total > 0 else None


def evaluate(
    xy: np.ndarray | None,
    conf: np.ndarray | None,
    target: TargetPose,
    cache: HoleCache,
    threshold: float = PASS_THRESHOLD,
) -> FitResult:
    """Score a player pose against a target pose."""
    if xy is None or conf is None:
        return FitResult(tracked=False, problem="NO PLAYER DETECTED")

    player_torso = to_torso_space(xy, conf)
    if player_torso is None:
        return FitResult(tracked=False, problem=_diagnose(conf))

    joint = joint_agreement(player_torso, cache.torso(target), conf)
    if joint is None:
        return FitResult(tracked=False, problem="STEP BACK - ARMS AND LEGS NEEDED")

    body = draw_silhouette(to_raster_pixels(player_torso), MATCH_RASTER, conf=conf)
    body_px = float(np.count_nonzero(body))
    if body_px < 1.0:
        return FitResult(tracked=False, problem="NO PLAYER DETECTED")

    target_mask = cache.target_mask(target, conf)
    hole = cache.hole(target)
    union = float(np.count_nonzero(body | target_mask)) or 1.0
    iou = float(np.count_nonzero(body & target_mask)) / union

    score = JOINT_WEIGHT * joint + (1.0 - JOINT_WEIGHT) * iou
    passed = score >= threshold
    return FitResult(
        score=score,
        passed=passed,
        grade="PERFECT" if score >= PERFECT_THRESHOLD else ("GOOD" if passed else "MISS"),
        joint=joint,
        iou=iou,
        containment=float(np.count_nonzero(body & hole)) / body_px,
        tracked=True,
    )
