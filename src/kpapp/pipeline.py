"""Decoupled capture -> inference -> render pipeline.

Inference runs ~18-25 FPS while the camera delivers 30-60 FPS. Running them in
lockstep would cap the display at model speed and make the whole app feel
sluggish. Instead the inference worker always picks up the *newest* frame and
publishes its result; the render loop draws every camera frame using the most
recent pose it has. The overlay lags the video by one inference period, but
motion stays fluid.
"""

from __future__ import annotations

import threading
import time
from collections import deque

from .camera import CameraStream
from .config import Config
from .inference import KeypointDetector, PoseSmoother, Result


class RateMeter:
    """Rolling FPS/latency over a short window."""

    def __init__(self, window: int = 60) -> None:
        self._times: deque[float] = deque(maxlen=window)
        self._values: deque[float] = deque(maxlen=window)

    def tick(self, value: float = 0.0) -> None:
        self._times.append(time.perf_counter())
        self._values.append(value)

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / span if span > 0 else 0.0

    @property
    def mean(self) -> float:
        return sum(self._values) / len(self._values) if self._values else 0.0


class InferenceWorker:
    """Runs the model on the latest camera frame in a background thread."""

    def __init__(self, camera: CameraStream, detector: KeypointDetector, cfg: Config) -> None:
        self.camera = camera
        self.detector = detector
        self.cfg = cfg
        self.smoother = PoseSmoother(cfg.smoothing)
        self.meter = RateMeter()

        self._lock = threading.Lock()
        self._result = Result()
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="inference", daemon=True)
        self.skipped = 0

    def start(self) -> "InferenceWorker":
        self._thread.start()
        return self

    def _loop(self) -> None:
        last_seq = -1
        while not self._stopped.is_set():
            frame, seq = self.camera.wait_for_frame(last_seq, timeout=0.5)
            if frame is None or seq == last_seq:
                # No new frame. Never spin here: a camera that returns early
                # without one would turn this into a busy loop that hogs the
                # GIL and starves the render thread.
                time.sleep(0.005)
                continue
            self.skipped += max(0, seq - last_seq - 1)
            last_seq = seq

            # Keyed to the camera sequence, not to frames this worker happened
            # to observe, so N means "every Nth camera frame" regardless of how
            # busy the worker is.
            if self.cfg.infer_every > 1 and seq % self.cfg.infer_every:
                continue

            try:
                result = self.detector.infer(frame, frame_seq=seq)
            except Exception as exc:  # keep the app alive on a bad frame
                print(f"[kp] inference error: {exc}")
                continue

            result.poses = self.smoother.apply(result.poses)
            self.meter.tick(result.infer_ms)
            with self._lock:
                self._result = result

    @property
    def result(self) -> Result:
        with self._lock:
            return self._result

    def stop(self) -> None:
        self._stopped.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
