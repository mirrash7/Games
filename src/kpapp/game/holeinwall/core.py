"""Shared types and the canonical pose space for Hole in the Wall.

Two coordinate spaces matter here:

**Canonical space** - how target poses are authored and stored. x and y both
run 0..1 over a unit square, y downward like screen coords, with the figure
filling most of the box. Easy to hand-author and to draw.

**Torso space** - how poses are compared. The mid-hip sits at the origin and
the shoulder-to-hip distance is 1.0. Normalising this way makes matching
independent of how far the player stands from the camera and where they stand
in frame, which is what lets the same target pose work for any body.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...config import KP

NUM_KEYPOINTS = 17
MIN_CONF = 0.5


@dataclass(frozen=True)
class TargetPose:
    """One hole shape, authored in canonical space."""

    name: str
    keypoints: np.ndarray  # (17, 2) float32 in canonical 0..1 space
    difficulty: int = 1  # 1 easy .. 3 hard
    clearance: float = 1.0  # multiplier on hole padding; >1 is more forgiving

    def scaled(self, width: int, height: int, margin: float = 0.0) -> np.ndarray:
        """Canonical space -> pixel coords inside a width x height box."""
        box = 1.0 - 2.0 * margin
        pts = self.keypoints * box + margin
        return np.stack([pts[:, 0] * width, pts[:, 1] * height], axis=1).astype(np.float32)


@dataclass
class FitResult:
    """Outcome of testing a player pose against a hole."""

    score: float = 0.0  # 0..1 blended match quality; drives pass/fail
    passed: bool = False
    grade: str = "MISS"  # PERFECT | GOOD | MISS
    joint: float = 0.0  # per-joint agreement, every limb weighted equally
    iou: float = 0.0  # silhouette overlap with the target shape
    containment: float = 0.0  # fraction of the body inside the opening
    tracked: bool = False  # was there a usable pose at all
    # Why tracking failed, for on-screen guidance. Empty when tracked.
    problem: str = ""


def torso_frame(xy: np.ndarray, conf: np.ndarray | None = None) -> tuple[np.ndarray, float] | None:
    """Return (origin, scale) defining the torso space for a pose.

    Origin is the mid-hip; scale is the shoulder-to-hip distance. Returns None
    when the four torso keypoints are not confidently visible, since every
    downstream normalisation would otherwise divide by noise.
    """
    idx = [KP["left_shoulder"], KP["right_shoulder"], KP["left_hip"], KP["right_hip"]]
    if conf is not None and float(np.min(conf[idx])) < MIN_CONF:
        return None

    shoulder_mid = (xy[KP["left_shoulder"]] + xy[KP["right_shoulder"]]) * 0.5
    hip_mid = (xy[KP["left_hip"]] + xy[KP["right_hip"]]) * 0.5
    scale = float(np.linalg.norm(shoulder_mid - hip_mid))
    if scale < 1e-3:
        return None
    return hip_mid.astype(np.float32), scale


def to_torso_space(xy: np.ndarray, conf: np.ndarray | None = None) -> np.ndarray | None:
    """Normalise a pose so matching ignores position, distance and body size."""
    frame = torso_frame(xy, conf)
    if frame is None:
        return None
    origin, scale = frame
    return ((xy - origin) / scale).astype(np.float32)


def place_in_box(
    torso_xy: np.ndarray, width: int, height: int, fill: float = 0.82
) -> np.ndarray:
    """Torso space -> pixels, centred and scaled to fill a box.

    Used to draw a player and a target at the same size so their silhouettes
    can be compared directly.
    """
    lo = torso_xy.min(axis=0)
    hi = torso_xy.max(axis=0)
    extent = np.maximum(hi - lo, 1e-3)
    scale = min(width * fill / extent[0], height * fill / extent[1])
    centre = (lo + hi) * 0.5
    out = (torso_xy - centre) * scale
    out[:, 0] += width * 0.5
    out[:, 1] += height * 0.5
    return out.astype(np.float32)
