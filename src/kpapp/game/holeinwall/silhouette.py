"""Turn a set of keypoints into a filled body silhouette mask.

This is the geometric heart of the game: the hole in the wall is a dilated
silhouette of the target pose, and the player "fits" when their own silhouette
lies inside it. Deriving both from the same function means the shape you see is
exactly the shape you are scored against.

Masks are deliberately rendered small (see MATCH_SIZE). Fit testing is a
whole-mask numpy comparison every frame, and at 192x144 that costs microseconds
while still resolving limbs cleanly.
"""

from __future__ import annotations

import cv2
import numpy as np

from ...config import KP

# Working resolution for fit testing. Big enough to resolve an arm, small
# enough that per-frame mask arithmetic is free.
MATCH_SIZE = (192, 144)

# Limb widths as a fraction of torso length, roughly human proportions.
_LIMB_W = 0.26
_ARM_W = 0.20
_HEAD_R = 0.42

_LIMBS: tuple[tuple[int, int, float], ...] = (
    (KP["left_shoulder"], KP["left_elbow"], _ARM_W),
    (KP["left_elbow"], KP["left_wrist"], _ARM_W),
    (KP["right_shoulder"], KP["right_elbow"], _ARM_W),
    (KP["right_elbow"], KP["right_wrist"], _ARM_W),
    (KP["left_hip"], KP["left_knee"], _LIMB_W),
    (KP["left_knee"], KP["left_ankle"], _LIMB_W),
    (KP["right_hip"], KP["right_knee"], _LIMB_W),
    (KP["right_knee"], KP["right_ankle"], _LIMB_W),
)


def _torso_length(xy: np.ndarray) -> float:
    shoulder = (xy[KP["left_shoulder"]] + xy[KP["right_shoulder"]]) * 0.5
    hip = (xy[KP["left_hip"]] + xy[KP["right_hip"]]) * 0.5
    return max(float(np.linalg.norm(shoulder - hip)), 1.0)


def draw_silhouette(
    xy: np.ndarray,
    size: tuple[int, int],
    conf: np.ndarray | None = None,
    min_conf: float = 0.5,
    thickness: float = 1.0,
    pad: float = 0.0,
) -> np.ndarray:
    """Render a filled body silhouette. Returns a uint8 mask, 255 = body.

    `pad` grows the shape outward by that many pixels, giving the same result
    as a morphological dilation but for free. Doing it with cv2.dilate instead
    is ruinous: an ellipse kernel wide enough for the hole clearance costs
    22 ms at 65px and 363 ms at 259px on a 720p frame, because a non-separable
    kernel scales with its own area. Baking the padding into the line widths
    and stroking the torso outline is geometrically equivalent and ~0.1 ms.
    """
    w, h = size
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = xy.astype(np.int32)
    vis = np.ones(len(xy), dtype=bool) if conf is None else conf >= min_conf

    unit = _torso_length(xy) * thickness
    pad = max(0.0, float(pad))

    # Torso as a filled quad. Drawn first so limbs blend into it. The stroked
    # outline is what applies `pad` to the torso.
    torso_idx = [KP["left_shoulder"], KP["right_shoulder"], KP["right_hip"], KP["left_hip"]]
    if vis[torso_idx].all():
        cv2.fillConvexPoly(mask, pts[torso_idx], 255, cv2.LINE_AA)
        if pad > 0.5:
            cv2.polylines(mask, [pts[torso_idx]], True, 255, int(pad * 2), cv2.LINE_AA)

    for a, b, width in _LIMBS:
        if vis[a] and vis[b]:
            px = max(2, int(unit * width + pad * 2))
            cv2.line(mask, tuple(pts[a]), tuple(pts[b]), 255, px, cv2.LINE_AA)
            # Round the joints so elbows and knees do not pinch to a point.
            cv2.circle(mask, tuple(pts[b]), px // 2, 255, -1, cv2.LINE_AA)

    # Head: centred above the shoulders, sized from the torso rather than from
    # the face keypoints, which are noisy and vanish when the player turns.
    if vis[KP["left_shoulder"]] and vis[KP["right_shoulder"]]:
        shoulder_mid = (xy[KP["left_shoulder"]] + xy[KP["right_shoulder"]]) * 0.5
        hip_mid = (xy[KP["left_hip"]] + xy[KP["right_hip"]]) * 0.5
        up = shoulder_mid - hip_mid
        norm = np.linalg.norm(up)
        up = up / norm if norm > 1e-3 else np.array([0.0, -1.0], dtype=np.float32)
        radius = max(3, int(unit * _HEAD_R + pad))
        centre = shoulder_mid + up * radius * 0.95
        cv2.circle(mask, (int(centre[0]), int(centre[1])), radius, 255, -1, cv2.LINE_AA)
        # Neck, so the head is never a floating disc.
        cv2.line(
            mask,
            (int(shoulder_mid[0]), int(shoulder_mid[1])),
            (int(centre[0]), int(centre[1])),
            255,
            max(2, int(unit * 0.18 + pad * 2)),
            cv2.LINE_AA,
        )

    return mask


def dilate(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Grow a mask outward - used to give the hole clearance around the pose."""
    if pixels <= 0:
        return mask
    k = 2 * int(pixels) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask, kernel)


def outline(mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    """Edge band of a mask, for drawing a glowing rim around the hole."""
    eroded = cv2.erode(mask, np.ones((thickness * 2 + 1,) * 2, np.uint8))
    return cv2.subtract(mask, eroded)
