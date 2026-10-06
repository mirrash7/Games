"""Where the player's hand is - shared by the blade and the menu cursor."""

from __future__ import annotations

import numpy as np

from .config import KP
from .inference import Pose

# How far past the wrist, as a fraction of the forearm, the palm sits. The
# model tracks the wrist joint, but players aim with their palm. On recorded
# play 0.30 landed on the lower palm; 0.38 is the middle of it, below the
# fingers.
PALM_REACH = 0.38

# Forearm length relative to shoulder width, measured on recorded play
# (median 0.76). Used to size the palm offset when no elbow has been seen.
FOREARM_PER_SHOULDER = 0.76

# How long a remembered forearm stays usable after the elbow drops out.
FOREARM_MEMORY = 0.6  # seconds


def hand_keypoint(hand: str = "right", mirrored: bool = True) -> int:
    """COCO wrist keypoint for the player's real hand.

    On a mirrored (selfie) feed the model labels limbs by how they appear, and a
    flipped person looks like someone with swapped anatomy, so the player's real
    right hand comes back as `left_wrist`. Confirmed on recorded frames: the
    raised right hand was tracked as left_wrist 70% of the time.
    """
    right = hand == "right"
    if mirrored:
        return KP["left_wrist"] if right else KP["right_wrist"]
    return KP["right_wrist"] if right else KP["left_wrist"]


def _elbow_for(wrist: int) -> int:
    return KP["left_elbow"] if wrist == KP["left_wrist"] else KP["right_elbow"]


def _upright_forearm(pose: Pose, min_conf: float) -> np.ndarray | None:
    """Forearm estimate when the elbow is not visible: pointing straight up.

    On recorded play the elbow went missing in 54% of frames with the wrist
    visible, almost always because it had dropped below the bottom of the
    frame while the hand was raised. Then the palm is directly above the wrist:
    "straight up" landed mid-palm in 8 of 9 such frames, while a
    shoulder-to-wrist direction missed the hand in most of them. Length comes
    from shoulder width, the steadiest body scale available.
    """
    ls, rs = pose.point(KP["left_shoulder"], min_conf), pose.point(KP["right_shoulder"], min_conf)
    if ls is None or rs is None:
        return None
    width = float(np.linalg.norm(ls - rs))
    if width < 1e-3:
        return None
    return np.array([0.0, -width * FOREARM_PER_SHOULDER], np.float32)


def hand_point(
    pose: Pose | None,
    hand: str = "right",
    mirrored: bool = True,
    reach: float = PALM_REACH,
    min_conf: float = 0.4,
) -> np.ndarray | None:
    """Palm position from a single pose: the wrist, pushed out along the forearm.

    Stateless. Without an elbow it assumes the forearm points straight up
    (the raised-hand case); only without shoulders to size that does it fall
    back to the bare wrist. `HandTracker` adds
    memory and smoothing on top, and is what the games use.
    """
    if pose is None:
        return None
    w = hand_keypoint(hand, mirrored)
    wrist = pose.point(w, min_conf)
    if wrist is None:
        return None
    wrist = np.asarray(wrist, np.float32)
    if reach <= 0.0:
        return wrist
    elbow = pose.point(_elbow_for(w), min_conf)
    fore = (wrist - elbow) if elbow is not None else _upright_forearm(pose, min_conf)
    if fore is None:
        return wrist
    return (wrist + fore * reach).astype(np.float32)


class OneEuroFilter:
    """Adaptive low-pass filter for a 2D pointer (Casiez et al., CHI 2012).

    Smooths hard when the hand is slow, so a held hand does not shimmer, and
    barely at all when it is fast, so a swipe is not dragged behind. A fixed
    smoothing amount can only trade one of those against the other.

    Defaults tuned with poses arriving at 20 Hz and read at 60 Hz: still-hand
    jitter 4.6 -> 2.0 px, for +7.6 px lag on a 2000 px/s swipe (~4 ms).
    """

    def __init__(self, min_cutoff: float = 3.0, beta: float = 0.02, d_cutoff: float = 1.5) -> None:
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self) -> None:
        self._x: np.ndarray | None = None
        self._dx = np.zeros(2, np.float32)
        self._t = 0.0

    @staticmethod
    def _alpha(dt: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, np.float32)
        if self._x is None:
            self._x, self._t = x.copy(), t
            return x.copy()
        dt = t - self._t
        if dt <= 0.0:
            return self._x.copy()
        dx = (x - self._x) / dt
        a_d = self._alpha(dt, self.d_cutoff)
        self._dx = a_d * dx + (1.0 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self._dx))
        a = self._alpha(dt, cutoff)
        self._x = (a * x + (1.0 - a) * self._x).astype(np.float32)
        self._t = t
        return self._x.copy()


class HandTracker:
    """Stable palm position over time: remembered forearm, then smoothing.

    On recorded play the elbow was missing in 54% of frames where the wrist
    was visible. Snapping to the bare wrist each time made the dot jump
    between palm and wrist - both "the dot sits at the base of my hand" and a
    shaky path. Remembering the last forearm across short drop-outs removes
    those jumps; the One Euro filter removes what tracking noise is left.
    """

    def __init__(self, hand: str = "right", mirrored: bool = True, reach: float = PALM_REACH,
                 min_conf: float = 0.4, smooth: bool = True) -> None:
        self.hand, self.mirrored, self.reach, self.min_conf = hand, mirrored, reach, min_conf
        self.filter = OneEuroFilter() if smooth else None
        self._fore: np.ndarray | None = None
        self._fore_t = -1e9

    def reset(self) -> None:
        self._fore, self._fore_t = None, -1e9
        if self.filter is not None:
            self.filter.reset()

    def update(self, pose: Pose | None, t: float) -> np.ndarray | None:
        if pose is None:
            self.reset()
            return None
        w = hand_keypoint(self.hand, self.mirrored)
        wrist = pose.point(w, self.min_conf)
        if wrist is None:
            self.reset()
            return None
        wrist = np.asarray(wrist, np.float32)

        elbow = pose.point(_elbow_for(w), self.min_conf)
        if elbow is not None:
            fore = (wrist - elbow).astype(np.float32)
            self._fore, self._fore_t = fore, t
        elif self._fore is not None and t - self._fore_t <= FOREARM_MEMORY:
            fore = self._fore
        else:
            fore = _upright_forearm(pose, self.min_conf)

        palm = wrist if fore is None else (wrist + fore * self.reach).astype(np.float32)
        return self.filter(palm, t) if self.filter is not None else palm
