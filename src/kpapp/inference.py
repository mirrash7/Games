"""RF-DETR keypoint model wrapper.

Normalises the supervision `KeyPoints` return into a small, allocation-free-ish
`Pose` record so the rest of the app never imports rfdetr or supervision.
"""

from __future__ import annotations

import contextlib
import logging
import time
import warnings
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import Config


@contextlib.contextmanager
def _quiet(enabled: bool = True):
    """Silence library chatter while the model loads and compiles.

    rf-detr logs a dozen INFO/WARNING lines about checkpoint plumbing, and
    graph tracing emits ~20 TracerWarnings - none actionable, and enough to
    bury the app's own messages. Scoped to model setup only; --verbose skips it.
    """
    if not enabled:
        yield
        return
    loggers = [logging.getLogger(n) for n in ("rf-detr", "transformers")]
    levels = [lg.level for lg in loggers]
    for lg in loggers:
        lg.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        for lg, lvl in zip(loggers, levels):
            lg.setLevel(lvl)


def pick_device(requested: str | None = None) -> str:
    import torch

    if requested:
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class Pose:
    """One detected person, in pixel coordinates of the source frame."""

    xy: np.ndarray  # (K, 2) float32
    confidence: np.ndarray  # (K,) float32, per-keypoint
    score: float  # whole-person detection confidence
    box: np.ndarray | None = None  # (4,) xyxy

    def visible(self, min_conf: float = 0.5) -> np.ndarray:
        return self.confidence >= min_conf

    def point(self, index: int, min_conf: float = 0.5) -> np.ndarray | None:
        """Keypoint `index` if it is confidently visible, else None."""
        if self.confidence[index] < min_conf:
            return None
        return self.xy[index]


@dataclass
class Result:
    poses: list[Pose] = field(default_factory=list)
    frame_seq: int = 0
    infer_ms: float = 0.0
    # perf_counter at the moment the frame entered the model. Used as the
    # reference time for extrapolation, so the model's own latency counts
    # toward how stale the pose is.
    timestamp: float = 0.0


class KeypointDetector:
    def __init__(self, cfg: Config) -> None:
        from rfdetr import RFDETRKeypointPreview

        self.cfg = cfg
        self.device = pick_device(cfg.device)
        with _quiet(not cfg.verbose):
            self.model = RFDETRKeypointPreview(
                device=self.device,
                resolution=cfg.resolution,
            )
        self._optimized = False

    def optimize(self) -> None:
        """Fuse/compile the graph. Costs ~10-60s once, then every call is faster."""
        if self._optimized:
            return
        import torch

        dtype = torch.float16 if self.cfg.half else torch.float32
        try:
            # `inference()` replaced `optimize_for_inference()` in rfdetr 1.9.
            optimize = getattr(self.model, "inference", None)
            if optimize is None:
                optimize = self.model.optimize_for_inference
            with _quiet(not self.cfg.verbose):
                optimize(compile=self.cfg.compile_model, batch_size=1, dtype=dtype)
            self._optimized = True
        except Exception as exc:  # pragma: no cover - backend dependent
            print(f"[kp] graph optimization unavailable ({exc}); running eager.")

    def warmup(self, size: tuple[int, int], rounds: int = 3) -> None:
        """Force lazy kernel compilation before the first real frame."""
        w, h = size
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        with _quiet(not self.cfg.verbose):
            for _ in range(rounds):
                self.infer(dummy)

    def infer(self, frame: np.ndarray, frame_seq: int = 0) -> Result:
        t0 = time.perf_counter()
        # rfdetr documents RGB input; OpenCV frames are BGR. Same size, so the
        # returned keypoint coordinates need no adjustment.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # include_source_image=False matters: the default attaches a copy of the
        # frame to every detection, which is pure allocation churn in a video loop.
        kps = self.model.predict(
            rgb,
            threshold=self.cfg.threshold,
            include_source_image=False,
        )
        infer_ms = (time.perf_counter() - t0) * 1000.0
        return Result(
            poses=self._to_poses(kps),
            frame_seq=frame_seq,
            infer_ms=infer_ms,
            timestamp=t0,
        )

    def _to_poses(self, kps) -> list[Pose]:
        xy = getattr(kps, "xy", None)
        if xy is None or len(xy) == 0:
            return []

        kp_conf = getattr(kps, "keypoint_confidence", None)
        if kp_conf is None:
            kp_conf = np.ones(xy.shape[:2], dtype=np.float32)
        det_conf = getattr(kps, "detection_confidence", None)
        boxes = kps.data.get("xyxy") if getattr(kps, "data", None) else None

        order = (
            np.argsort(-np.asarray(det_conf))
            if det_conf is not None
            else np.arange(len(xy))
        )
        poses = [
            Pose(
                xy=np.asarray(xy[i], dtype=np.float32),
                confidence=np.asarray(kp_conf[i], dtype=np.float32),
                score=float(det_conf[i]) if det_conf is not None else 1.0,
                box=np.asarray(boxes[i], dtype=np.float32) if boxes is not None else None,
            )
            for i in order[: self.cfg.max_people]
        ]
        return poses


class PoseSmoother:
    """Exponential moving average over keypoints, keyed by pose slot.

    Raw per-frame keypoints jitter by a few pixels, which reads as twitchy
    control input. Smoothing trades a little latency for steadiness.
    """

    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha
        self._prev: dict[int, np.ndarray] = {}
        self._prev_conf: dict[int, np.ndarray] = {}

    def apply(self, poses: list[Pose]) -> list[Pose]:
        if self.alpha <= 0.0:
            return poses
        a = 1.0 - self.alpha
        for slot, pose in enumerate(poses):
            prev = self._prev.get(slot)
            prev_conf = self._prev_conf.get(slot)
            if (
                prev is not None
                and prev_conf is not None
                and prev.shape == pose.xy.shape
            ):
                # Blend only keypoints confident in BOTH frames. Checking the
                # current frame alone would drag a reappearing limb in from
                # wherever it was last seen instead of letting it snap in.
                usable = (pose.confidence >= 0.5) & (prev_conf >= 0.5)
                blended = np.where(
                    usable[:, None],
                    a * pose.xy + self.alpha * prev,
                    pose.xy,
                )
                pose.xy = blended.astype(np.float32)
            self._prev[slot] = pose.xy.copy()
            self._prev_conf[slot] = pose.confidence.copy()
        for slot in list(self._prev):
            if slot >= len(poses):
                del self._prev[slot]
                del self._prev_conf[slot]
        return poses


class PoseExtrapolator:
    """Projects keypoints forward in time using per-keypoint velocity.

    Inference lands at ~22 Hz and each result already describes a frame that is
    one model-latency old, so a render loop that draws the newest result is
    always 45-90 ms behind the player. Estimating velocity between consecutive
    results and projecting to render time removes most of that lag, which is
    what actually makes the controls feel responsive.

    The projection is deliberately conservative: `gain` below 1.0 under-shoots
    rather than over-shoots, because overshoot at a direction reversal reads as
    a visible snap-back, which feels worse than a little lag.
    """

    def __init__(
        self,
        gain: float = 0.7,
        max_lead: float = 0.12,
        velocity_smoothing: float = 0.5,
        max_gap: float = 0.25,
    ) -> None:
        self.gain = gain
        self.max_lead = max_lead
        self.velocity_smoothing = velocity_smoothing
        self.max_gap = max_gap
        self._result = Result()
        self._vel: dict[int, np.ndarray] = {}
        self._prev_xy: dict[int, np.ndarray] = {}
        self._prev_conf: dict[int, np.ndarray] = {}
        # None, not 0.0: "no result yet" is a distinct state from a timestamp
        # that happens to be zero.
        self._prev_t: float | None = None
        self.last_lead_ms = 0.0
        # The newest result projected to the moment it arrived, then held
        # until the next one: see `held`.
        self.held: list[Pose] = []

    def update(self, result: Result, arrived: float | None = None) -> None:
        """Feed a freshly published inference result. Call once per result.

        `arrived` is when the render loop picked it up. `held` is then the
        result projected to that moment and held until the next result.

        Use `held` for gesture detectors that measure speed between model
        updates (FlapDetector). `poses_at()` moves every render frame, so
        each new result shows up as a jump between two frames 8-16 ms apart.
        The detector's teleport guard reads that jump as a tracking glitch
        and drops its history. Simulated at 120 Hz render with 15-30 Hz
        poses, it then caught 0 of 30 flaps; with `held`, 30 of 30.
        """
        if self.gain > 0.0 and self._prev_t is not None:
            dt = result.timestamp - self._prev_t
            if 0.0 < dt <= self.max_gap:
                for slot, pose in enumerate(result.poses):
                    prev = self._prev_xy.get(slot)
                    prev_conf = self._prev_conf.get(slot)
                    if prev is None or prev_conf is None or prev.shape != pose.xy.shape:
                        continue
                    raw = (pose.xy - prev) / dt
                    # Require confidence in BOTH frames. A keypoint that just
                    # reappeared has no usable history, and a velocity measured
                    # from its stale position would fling it across the frame.
                    usable = (pose.confidence >= 0.5) & (prev_conf >= 0.5)
                    raw = np.where(usable[:, None], raw, 0.0)
                    old = self._vel.get(slot)
                    if old is not None and old.shape == raw.shape:
                        a = self.velocity_smoothing
                        raw = a * old + (1.0 - a) * raw
                    self._vel[slot] = raw.astype(np.float32)
            else:
                # Stalled worker or first result: stale velocities would be worse
                # than none at all.
                self._vel.clear()

        self._prev_xy = {i: p.xy.copy() for i, p in enumerate(result.poses)}
        self._prev_conf = {i: p.confidence.copy() for i, p in enumerate(result.poses)}
        self._prev_t = result.timestamp
        self._result = result
        for slot in list(self._vel):
            if slot >= len(result.poses):
                del self._vel[slot]
        self.held = self.poses_at(arrived) if arrived is not None else list(result.poses)

    def poses_at(self, now: float) -> list[Pose]:
        """Poses projected to wall-clock time `now`."""
        result = self._result
        if self.gain <= 0.0 or not result.poses or self._prev_t is None:
            self.last_lead_ms = 0.0
            return result.poses

        # Clamped so a hung worker cannot launch the skeleton off-screen.
        lead = min(max(now - self._prev_t, 0.0), self.max_lead) * self.gain
        self.last_lead_ms = lead * 1000.0
        if lead <= 0.0:
            return result.poses

        projected: list[Pose] = []
        for slot, pose in enumerate(result.poses):
            vel = self._vel.get(slot)
            if vel is None:
                projected.append(pose)
                continue
            # New Pose objects: the worker's published poses are shared across
            # threads and must not be mutated here.
            projected.append(
                Pose(
                    xy=(pose.xy + vel * lead).astype(np.float32),
                    confidence=pose.confidence,
                    score=pose.score,
                    box=pose.box,
                )
            )
        return projected
