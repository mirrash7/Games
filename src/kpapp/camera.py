"""Threaded camera capture that always hands back the newest frame.

A plain `cap.read()` in the main loop hands you whatever is next in the driver
queue, so any hitch downstream shows up as growing latency. This thread drains
continuously and keeps only the latest frame, trading dropped frames for a
constant, small delay between the world and the screen.
"""

from __future__ import annotations

import sys
import threading
import time

import cv2
import numpy as np

# Pin OpenCV to the native macOS camera backend. Left on auto, OpenCV falls
# back to its FFmpeg path when AVFoundation fails - and by then the model's
# first prediction has loaded PyAV (via supervision), which bundles a second
# copy of FFmpeg's camera classes. macOS warns those duplicates "may cause
# spurious casting failures and mysterious crashes"; the native backend never
# touches them.
_BACKEND = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY


def open_capture(index: int) -> cv2.VideoCapture:
    return cv2.VideoCapture(index, _BACKEND)


class CameraStream:
    def __init__(
        self,
        index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 60,
        mirror: bool = True,
    ) -> None:
        self.index = index
        self.mirror = mirror
        self._cap = open_capture(index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open camera {index}. On macOS, grant camera access to "
                "your terminal in System Settings > Privacy & Security > Camera."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        # A 1-frame driver buffer keeps the newest frame closest to the sensor.
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._stopped = threading.Event()
        self._new_frame = threading.Condition(self._lock)
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)

        self.dropped = 0
        self._last_seq_read = 0
        # Health timestamps (monotonic). A stall or a driver handing over
        # placeholder frames is otherwise invisible: readers just keep getting
        # the last frame they already had.
        self.last_frame_time = 0.0
        self.last_real_time = 0.0
        self._width, self._height, self._fps = width, height, fps

    @property
    def frames_seen(self) -> int:
        """Frames delivered so far, real or blank."""
        return self._seq

    def is_blank_stream(self, min_frames: int = 75) -> bool:
        """Frames are arriving, but not one has been real.

        Distinct from a cold start, which delivers nothing at all: a device
        that streams solid blank frames at full rate (seen on this Mac's
        camera 0, 30 fps of exact black) will not come good by waiting.
        """
        return self.last_real_time == 0.0 and self._seq >= min_frames

    @property
    def size(self) -> tuple[int, int]:
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def start(self, wait: float = 5.0) -> "CameraStream":
        """Begin capturing. With wait=0, return at once and let it warm up
        in the background (check readiness later with wait_until_real)."""
        self._thread.start()
        if wait <= 0:
            return self
        # Block briefly so callers get a valid frame on their first read.
        deadline = time.monotonic() + wait
        while self.read()[0] is None and time.monotonic() < deadline:
            time.sleep(0.01)
        if self._frame is None:
            # Release before raising. Otherwise the capture thread keeps the
            # device open, and the fallback that probes for a working camera
            # ends up fighting this abandoned handle for the same device.
            self.stop()
            raise RuntimeError(f"Camera opened but delivered no frames within {wait:.0f}s.")
        return self

    def _loop(self) -> None:
        while not self._stopped.is_set():
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            if self.mirror:
                frame = cv2.flip(frame, 1)
            now = time.monotonic()
            self.last_frame_time = now
            if not frame_is_placeholder(frame):
                self.last_real_time = now
            with self._new_frame:
                if self._frame is not None and self._seq > self._last_seq_read:
                    self.dropped += 1
                self._frame = frame
                self._seq += 1
                self._new_frame.notify_all()

    def read(self) -> tuple[np.ndarray | None, int]:
        """Return (latest_frame, sequence_number) without blocking."""
        with self._lock:
            self._last_seq_read = self._seq
            return self._frame, self._seq

    def wait_for_frame(self, last_seq: int, timeout: float = 1.0) -> tuple[np.ndarray | None, int]:
        """Block until a frame newer than `last_seq` arrives."""
        with self._new_frame:
            if self._seq <= last_seq:
                self._new_frame.wait(timeout)
            self._last_seq_read = self._seq
            return self._frame, self._seq

    def wait_until_real(self, seconds: float) -> bool:
        """Block until a genuine (non-placeholder) frame arrives, or give up."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self.last_real_time > 0.0:
                return True
            time.sleep(0.05)
        return self.last_real_time > 0.0

    def reopen(self, index: int | None = None) -> bool:
        """Release and reacquire the device, optionally switching index.

        The last good frame is kept so readers never see None mid-recovery.
        """
        self._stopped.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._cap.release()
        if index is not None:
            self.index = index
        self._cap = open_capture(self.index)
        if not self._cap.isOpened():
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stopped.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._cap.release()

    def __enter__(self) -> "CameraStream":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


class SyntheticCamera:
    """Drop-in stand-in for CameraStream that generates frames.

    Lets the full pipeline be timed without camera permission or a webcam,
    which is the difference between "the model is fast" and "the app is fast".
    """

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30) -> None:
        self.width, self.height = width, height
        self.period = 1.0 / fps
        self._seq = 0
        self._t0 = time.monotonic()
        self._frame_seq = -1
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self.last_frame_time = time.monotonic()
        self.last_real_time = time.monotonic()
        self._base = np.random.default_rng(0).integers(
            0, 255, (height, width, 3), dtype=np.uint8
        )
        self.dropped = 0

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)

    def start(self, wait: float = 0.0) -> "SyntheticCamera":
        return self

    def _render(self) -> np.ndarray:
        frame = self._base.copy()
        x = int((self._seq * 7) % max(1, self.width - 60))
        cv2.rectangle(frame, (x, 100), (x + 60, 260), (255, 255, 255), -1)
        return frame

    def _current(self) -> tuple[np.ndarray, int]:
        """The frame for 'now': sequence advances with wall time, like a sensor."""
        seq = int((time.monotonic() - self._t0) / self.period)
        with self._lock:
            if seq != self._frame_seq:
                self._seq = seq
                self._frame = self._render()
                self._frame_seq = seq
                self.last_frame_time = self.last_real_time = time.monotonic()
            return self._frame, self._frame_seq

    def read(self) -> tuple[np.ndarray, int]:
        return self._current()

    def wait_for_frame(self, last_seq: int, timeout: float = 1.0) -> tuple[np.ndarray, int]:
        target = self._t0 + (last_seq + 1) * self.period
        delay = target - time.monotonic()
        if delay > 0:
            time.sleep(min(delay, timeout))
        return self._current()

    def wait_until_real(self, seconds: float) -> bool:
        return True

    def reopen(self, index: int | None = None) -> bool:
        return True

    def stop(self) -> None:
        pass


# --- camera health ---------------------------------------------------------

def frame_is_placeholder(frame: np.ndarray | None) -> bool:
    """True for frames no real sensor produces.

    A covered lens or a dark room still has sensor noise. Frames that are
    exactly uniform come from a driver that is not actually streaming - most
    often a Continuity Camera iPhone that is locked or out of reach, which
    macOS may have slotted in as camera 0.
    """
    if frame is None:
        return True
    small = frame[::16, ::16]
    return float(small.std()) < 0.5


def probe(index: int, seconds: float = 2.5, width: int = 1280, height: int = 720) -> dict:
    """Open a camera briefly and report whether it delivers real images.

    Reads happen on a helper thread with a deadline, because a stalled device
    can block `cap.read()` indefinitely and must not hang the probe.
    """
    cap = open_capture(index)
    if not cap.isOpened():
        return {"index": index, "opened": False}
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    stats = {"frames": 0, "real": 0, "shape": None}
    stop = threading.Event()

    def reader() -> None:
        while not stop.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            stats["frames"] += 1
            stats["shape"] = frame.shape[:2]
            if not frame_is_placeholder(frame):
                stats["real"] += 1

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and stats["real"] < 15:
        time.sleep(0.05)  # a healthy camera proves itself in about half a second
    elapsed = seconds - max(0.0, deadline - time.monotonic())
    stop.set()
    t.join(timeout=1.0)
    cap.release()

    fps = stats["frames"] / max(elapsed, 1e-3)
    healthy = stats["real"] >= 15 or stats["real"] >= max(3, int(stats["frames"] * 0.3))
    return {
        "index": index,
        "opened": True,
        "fps": round(fps, 1),
        "real_frames": stats["real"],
        "frames": stats["frames"],
        "size": None if stats["shape"] is None else (stats["shape"][1], stats["shape"][0]),
        "healthy": healthy,
        "stalled": not t.is_alive() and stats["frames"] == 0 or t.is_alive(),
    }


def find_working_camera(
    preferred: int = 0, candidates: int = 4, seconds: float = 2.0, skip: tuple[int, ...] = ()
) -> int | None:
    """First camera index that streams real frames, trying `preferred` first.

    `skip` holds indices already known to be dead, so they are not re-probed.
    """
    order = [preferred] + [i for i in range(candidates) if i != preferred]
    for idx in (i for i in order if i not in skip):
        if probe(idx, seconds=seconds).get("healthy"):
            return idx
    return None


# --- choosing a camera by name ----------------------------------------------

from dataclasses import dataclass  # noqa: E402


@dataclass(frozen=True)
class CameraInfo:
    index: int  # the index OpenCV opens this device with
    name: str
    builtin: bool
    phone: bool  # Continuity Camera iPhone/iPad


@dataclass(frozen=True)
class CameraChoice:
    index: int
    label: str
    explicit: bool  # the player named it; respect it, don't second-guess
    avoid: tuple[int, ...] = ()  # never switch to these automatically (phones)


def list_cameras() -> list[CameraInfo]:
    """Cameras in the order OpenCV numbers them, with their names.

    On macOS camera numbers are not stable: with Continuity Camera the built-in
    camera and an iPhone swap between 0 and 1. OpenCV's AVFoundation backend
    takes AVFoundation's video devices followed by its muxed devices, then
    **sorts them by uniqueID**, so that is the order to reproduce. An iPhone's
    uniqueID changes between connections, so it can sort before or after the
    built-in camera from one day to the next. Using AVFoundation's own order
    instead once opened the iPhone while reporting "MacBook Pro Camera (camera 0)".
    """
    if sys.platform != "darwin":
        return []
    try:
        import AVFoundation as AV  # pyobjc, macOS only
    except ImportError:
        return []
    devices = list(AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeVideo)) + list(
        AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeMuxed)
    )
    return describe_devices(
        (str(d.uniqueID()), str(d.localizedName()), str(d.deviceType())) for d in devices)


def describe_devices(devices) -> list[CameraInfo]:
    """(uniqueID, name, deviceType) tuples -> CameraInfo numbered as OpenCV does."""
    out = []
    # NSString compare: on these ASCII IDs is a plain ordinal comparison.
    for i, (_uid, name, kind) in enumerate(sorted(devices, key=lambda d: d[0])):
        phone = "Continuity" in kind or "iphone" in name.lower() or "ipad" in name.lower()
        out.append(CameraInfo(i, name, builtin="BuiltIn" in kind, phone=phone))
    return out


def resolve_camera(choice: str | None, cameras: list[CameraInfo]) -> CameraChoice:
    """Turn --camera (an index, a name fragment, or nothing) into a device.

    Nothing means the built-in camera, wherever it is numbered today, never
    the phone. A name like "iphone" or "macbook" matches case-insensitively.
    """
    phones = tuple(c.index for c in cameras if c.phone)
    by_index = {c.index: c for c in cameras}

    def label(i: int) -> str:
        return f"{by_index[i].name} (camera {i})" if i in by_index else f"camera {i}"

    if choice is None or str(choice).strip() == "":
        pick = next((c for c in cameras if c.builtin), None) or next(
            (c for c in cameras if not c.phone), None)
        idx = pick.index if pick else 0
        return CameraChoice(idx, label(idx), explicit=False, avoid=phones)

    text = str(choice).strip()
    if text.isdigit():
        idx = int(text)
        return CameraChoice(idx, label(idx), explicit=True, avoid=phones)

    matches = [c for c in cameras if text.lower() in c.name.lower()]
    if not matches:
        names = ", ".join(f"{c.index}: {c.name}" for c in cameras) or "none found"
        raise ValueError(f"no camera matches {text!r} (available: {names})")
    return CameraChoice(matches[0].index, label(matches[0].index), explicit=True, avoid=phones)
