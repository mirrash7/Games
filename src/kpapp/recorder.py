"""Optional capture of a live session for review: frames plus timings.

Exists because the interesting failures in this app only show up with a real
person in a real room - synthetic frames contain no one, so they skip all the
detection post-processing and hide the actual cost. Recording a short live run
gives something concrete to look at and measure afterwards.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np


class Recorder:
    def __init__(
        self,
        directory: str | Path,
        every: float = 1.0,
        max_event_frames: int = 40,
        raw_every: float = 3.0,
    ) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.every = every
        self.max_event_frames = max_event_frames
        self._last_periodic = 0.0
        self.raw_every = raw_every
        self._last_raw = 0.0
        self._events = 0
        self._n = 0
        self._log = open(self.dir / "metrics.jsonl", "w")
        self._t0 = time.perf_counter()

    def frame(
        self,
        canvas: np.ndarray,
        metrics: dict,
        event: str | None = None,
        raw: np.ndarray | None = None,
    ) -> None:
        now = time.perf_counter()
        # Unannotated camera frames, kept so model behaviour can be re-tested
        # offline on exactly what the camera saw.
        if raw is not None and now - self._last_raw >= self.raw_every:
            self._last_raw = now
            cv2.imwrite(str(self.dir / f"raw_{self._n:04d}.png"), raw)
        metrics = {"t": round(now - self._t0, 4), **metrics}
        if event:
            metrics["event"] = event
        self._log.write(json.dumps(metrics) + "\n")

        save = None
        if now - self._last_periodic >= self.every:
            self._last_periodic = now
            save = f"periodic_{self._n:04d}.jpg"
        elif event and self._events < self.max_event_frames:
            self._events += 1
            save = f"event_{self._n:04d}_{event}.jpg"
        if save:
            cv2.imwrite(str(self.dir / save), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
        self._n += 1

    def close(self) -> None:
        self._log.close()
