"""Turn raw keypoints into normalised, game-ready control signals.

Everything is normalised against the player's own shoulder width and torso
height, so the same gesture produces the same signal whether the player is
close to the camera or across the room.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import KP
from .inference import Pose

MIN_CONF = 0.5


@dataclass
class ControlState:
    """One frame of player input. This is the contract games code against."""

    present: bool = False  # is a player being tracked at all
    steer: float = 0.0  # -1 (left) .. +1 (right)
    throttle: float = 0.0  # 0 (hands down) .. 1 (hands up)
    left_hand: tuple[float, float] | None = None  # normalised 0..1 screen space
    right_hand: tuple[float, float] | None = None
    hands_up: bool = False
    arms_out: bool = False
    crouching: bool = False
    buttons: dict[str, bool] = field(default_factory=dict)
    # The raw pose behind these signals. Most games should use the derived
    # fields above, but pose-matching games need the keypoints themselves.
    pose: Pose | None = None
    # The same player, but updated only when the model produces a new result
    # (projected to when it arrived), so it moves in steps. Gesture detectors
    # that time motion between model updates want this one; see
    # PoseExtrapolator.update.
    step_pose: Pose | None = None

    def pressed(self, name: str) -> bool:
        return self.buttons.get(name, False)


def _mid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) * 0.5


class ControlMapper:
    """Pose -> ControlState, with hysteresis on the boolean gestures.

    Booleans use separate on/off thresholds so a hand hovering at the boundary
    doesn't machine-gun the game with on/off events.
    """

    def __init__(self, frame_size: tuple[int, int], deadzone: float = 0.08) -> None:
        self.width, self.height = frame_size
        self.deadzone = deadzone
        self._latched: dict[str, bool] = {}

    def _latch(self, name: str, value: float, on: float, off: float) -> bool:
        state = self._latched.get(name, False)
        state = value > on if not state else value > off
        self._latched[name] = state
        return state

    def _norm(self, p: np.ndarray) -> tuple[float, float]:
        return (float(p[0] / self.width), float(p[1] / self.height))

    def _apply_deadzone(self, value: float) -> float:
        if abs(value) < self.deadzone:
            return 0.0
        scaled = (abs(value) - self.deadzone) / (1.0 - self.deadzone)
        return float(np.clip(scaled, 0.0, 1.0)) * np.sign(value)

    def map(self, pose: Pose | None) -> ControlState:
        if pose is None:
            self._latched.clear()
            return ControlState(present=False)

        xy, conf = pose.xy, pose.confidence
        ls = pose.point(KP["left_shoulder"], MIN_CONF)
        rs = pose.point(KP["right_shoulder"], MIN_CONF)
        if ls is None or rs is None:
            # Without shoulders there is no stable body frame to normalise to.
            # The pose still rides along: a pose-matching game can decide for
            # itself whether it has enough keypoints to work with.
            return ControlState(present=False, pose=pose)

        shoulder_mid = _mid(ls, rs)
        shoulder_w = max(float(np.linalg.norm(ls - rs)), 1.0)

        lh = pose.point(KP["left_wrist"], MIN_CONF)
        rh = pose.point(KP["right_wrist"], MIN_CONF)
        lhip = pose.point(KP["left_hip"], MIN_CONF)
        rhip = pose.point(KP["right_hip"], MIN_CONF)

        state = ControlState(present=True, pose=pose)
        state.left_hand = self._norm(lh) if lh is not None else None
        state.right_hand = self._norm(rh) if rh is not None else None

        # Steering: horizontal offset of the hand midpoint from the torso
        # centre, in shoulder-widths. Falls back to torso lean if hands are
        # hidden, so control degrades gracefully instead of dropping out.
        if lh is not None and rh is not None:
            offset = (_mid(lh, rh)[0] - shoulder_mid[0]) / shoulder_w
            state.steer = self._apply_deadzone(float(np.clip(offset / 0.9, -1.0, 1.0)))
        elif lhip is not None and rhip is not None:
            lean = (shoulder_mid[0] - _mid(lhip, rhip)[0]) / shoulder_w
            state.steer = self._apply_deadzone(float(np.clip(lean / 0.5, -1.0, 1.0)))

        # Throttle: how far above the shoulder line the higher hand is.
        # Screen y grows downward, hence the negation.
        heights = [
            -(h[1] - shoulder_mid[1]) / shoulder_w
            for h in (lh, rh)
            if h is not None
        ]
        if heights:
            state.throttle = float(np.clip(max(heights) / 1.2, 0.0, 1.0))

        state.hands_up = self._latch("hands_up", state.throttle, on=0.55, off=0.40)

        # Arms out: both wrists far from the torso centre horizontally.
        if lh is not None and rh is not None:
            spread = (abs(lh[0] - shoulder_mid[0]) + abs(rh[0] - shoulder_mid[0])) / shoulder_w
            state.arms_out = self._latch("arms_out", spread, on=1.7, off=1.4)

        # Crouch: torso vertically compressed relative to shoulder width.
        if lhip is not None and rhip is not None:
            torso = (_mid(lhip, rhip)[1] - shoulder_mid[1]) / shoulder_w
            state.crouching = self._latch("crouch", -torso, on=-0.85, off=-1.0)

        state.buttons = {
            "hands_up": state.hands_up,
            "arms_out": state.arms_out,
            "crouch": state.crouching,
        }
        return state
