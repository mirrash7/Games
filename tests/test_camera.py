"""Camera health: telling a dead camera from a dark room, and reacting to it."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np

from kpapp.app import _camera_problem
from kpapp.camera import SyntheticCamera, frame_is_placeholder
from kpapp.config import Config
from kpapp.inference import Result
from kpapp.pipeline import InferenceWorker


def test_exact_black_is_a_placeholder():
    """What the live1 run produced: frames of exact zeros, no sensor noise."""
    assert frame_is_placeholder(np.zeros((720, 1280, 3), np.uint8))


def test_uniform_grey_is_a_placeholder():
    assert frame_is_placeholder(np.full((720, 1280, 3), 128, np.uint8))


def test_dark_room_is_not_a_placeholder():
    """A dim scene is dark but noisy; it must not be mistaken for a dead camera."""
    rng = np.random.default_rng(0)
    dark = np.clip(rng.normal(12, 4, (720, 1280, 3)), 0, 255).astype(np.uint8)
    assert not frame_is_placeholder(dark)


def test_none_is_a_placeholder():
    assert frame_is_placeholder(None)


def _cam(frame_age: float, real_age: float):
    now = time.monotonic()
    return SimpleNamespace(last_frame_time=now - frame_age, last_real_time=now - real_age), now


def test_healthy_camera_reports_no_problem():
    cam, now = _cam(0.03, 0.03)
    assert _camera_problem(cam, now) is None


def test_stalled_camera_is_reported():
    cam, now = _cam(5.0, 5.0)
    assert "STALLED" in _camera_problem(cam, now)


def test_blank_frames_are_reported():
    """Frames keep arriving, but none are real - the Continuity Camera case."""
    cam, now = _cam(0.03, 5.0)
    assert "BLANK" in _camera_problem(cam, now)


def test_synthetic_camera_advances_with_wall_time():
    """A free-running render loop must not mint a new frame on every read."""
    cam = SyntheticCamera(fps=30)
    seqs = set()
    t0 = time.monotonic()
    while time.monotonic() - t0 < 0.5:
        seqs.add(cam.read()[1])
        time.sleep(0.002)
    assert 12 <= len(seqs) <= 18, f"expected ~15 frames in 0.5s at 30fps, got {len(seqs)}"


class _FrozenCamera:
    """Returns the same frame immediately, forever - the spin hazard."""

    def __init__(self):
        self.calls = 0
        self.frame = np.zeros((8, 8, 3), np.uint8)

    def wait_for_frame(self, last_seq, timeout=0.5):
        self.calls += 1
        return self.frame, 1


class _NoopDetector:
    def infer(self, frame, frame_seq=0):
        return Result(frame_seq=frame_seq, timestamp=time.perf_counter())


def test_worker_does_not_busy_spin_without_new_frames():
    """A worker spinning on a frozen camera hogs the GIL and starves rendering."""
    cam = _FrozenCamera()
    worker = InferenceWorker(cam, _NoopDetector(), Config()).start()
    time.sleep(0.3)
    worker.stop()
    # A spinning loop would make tens of thousands of calls in 0.3s.
    assert cam.calls < 200, f"worker polled {cam.calls} times in 0.3s - it is spinning"


def test_detector_feeds_the_model_rgb():
    """rfdetr documents RGB input; OpenCV frames arrive BGR."""
    from kpapp.inference import KeypointDetector

    seen = {}

    class FakeModel:
        def predict(self, img, **kw):
            seen["img"] = img
            return SimpleNamespace(xy=np.zeros((0, 17, 2)), data={})

    det = KeypointDetector.__new__(KeypointDetector)
    det.cfg = Config()
    det.model = FakeModel()
    bgr = np.zeros((4, 4, 3), np.uint8)
    bgr[..., 0] = 255  # pure blue in OpenCV's BGR order
    det.infer(bgr)
    assert seen["img"][0, 0].tolist() == [0, 0, 255], "model must receive RGB"


def test_failed_start_releases_the_device(monkeypatch):
    """A camera that never delivers must be released, or the fallback that
    probes for another camera fights this abandoned handle for the device."""
    import cv2
    from kpapp import camera as cam_mod

    released = []

    class SilentCapture:
        def __init__(self, *a): pass
        def isOpened(self): return True
        def set(self, *a): return True
        def get(self, *a): return 0
        def read(self):
            time.sleep(0.01)
            return False, None
        def release(self): released.append(True)

    monkeypatch.setattr(cam_mod.cv2, "VideoCapture", SilentCapture)
    monkeypatch.setattr(cam_mod.time, "monotonic",
                        (lambda base=[0.0]: (base.__setitem__(0, base[0] + 1.0), base[0])[1]))
    stream = cam_mod.CameraStream(index=0)
    try:
        stream.start()
    except RuntimeError:
        pass
    assert released, "device left open after a failed start"


class _WarmingCamera:
    """Starts streaming real frames only after `delay` seconds - a cold start."""

    def __init__(self, delay: float):
        self.t0 = time.monotonic()
        self.delay = delay
        self.stopped = False
        self.size = (1280, 720)

    @property
    def last_real_time(self) -> float:
        return time.monotonic() if time.monotonic() - self.t0 >= self.delay else 0.0

    def wait_until_real(self, seconds: float) -> bool:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self.last_real_time:
                return True
            time.sleep(0.01)
        return bool(self.last_real_time)

    def stop(self):
        self.stopped = True


from kpapp.camera import CameraChoice, CameraInfo, resolve_camera  # noqa: E402

MAC = CameraInfo(0, "MacBook Pro Camera", builtin=True, phone=False)
PHONE = CameraInfo(1, "Alex's iPhone Camera", builtin=False, phone=True)
DEFAULT = CameraChoice(0, "MacBook Pro Camera (camera 0)", explicit=False, avoid=(1,))


def _no_reopen(idx):
    raise RuntimeError("no device in tests")


def test_slow_cold_start_is_waited_for():
    """A camera that takes a few seconds to wake must not be written off."""
    from kpapp.app import _ensure_streaming

    cam = _WarmingCamera(delay=0.6)
    assert _ensure_streaming(cam, DEFAULT, Config(), timeout=2.0, opener=_no_reopen) is cam
    assert not cam.stopped


def test_dead_default_camera_is_released_before_searching(monkeypatch):
    import pytest
    from kpapp import app, camera as cam_mod

    monkeypatch.setattr(cam_mod, "find_working_camera", lambda **kw: None)
    cam = _WarmingCamera(delay=1e9)
    with pytest.raises(RuntimeError, match="not streaming"):
        app._ensure_streaming(cam, DEFAULT, Config(), timeout=0.2, opener=_no_reopen)
    assert cam.stopped, "the dead camera must be released before probing others"


def test_explicit_camera_is_kept_even_if_slow():
    from kpapp.app import _ensure_streaming

    cam = _WarmingCamera(delay=1e9)
    named = CameraChoice(1, "Alex's iPhone Camera (camera 1)", explicit=True, avoid=(1,))
    assert _ensure_streaming(cam, named, Config(), timeout=0.1, opener=_no_reopen) is cam
    assert not cam.stopped


def test_blank_streaming_camera_is_abandoned_quickly(monkeypatch):
    """A camera sending solid black at full rate will not recover by waiting."""
    import pytest
    from kpapp import app, camera as cam_mod

    probed = []
    monkeypatch.setattr(cam_mod, "find_working_camera",
                        lambda **kw: probed.append(kw.get("skip")) or None)

    class BlankCamera(_WarmingCamera):
        def is_blank_stream(self):
            return True

    t0 = time.monotonic()
    with pytest.raises(RuntimeError):
        app._ensure_streaming(BlankCamera(delay=1e9), DEFAULT, Config(), timeout=5.0,
                              opener=lambda idx: BlankCamera(delay=1e9))
    assert time.monotonic() - t0 < 1.5, "waited out the timeout on a blank camera"
    assert probed and 0 in probed[0], "the known-blank camera must not be re-probed"


# --- choosing the camera ---


def test_default_is_the_built_in_camera_wherever_it_is_numbered():
    """Camera numbers swap on this Mac; the built-in camera is found by type."""
    assert resolve_camera(None, [MAC, PHONE]).index == 0
    swapped = [CameraInfo(0, PHONE.name, False, True), CameraInfo(1, MAC.name, True, False)]
    choice = resolve_camera(None, swapped)
    assert choice.index == 1 and "MacBook" in choice.label
    assert not choice.explicit


def test_phone_is_never_an_automatic_fallback(monkeypatch):
    """With the Mac camera dead, the app must not quietly switch to the phone."""
    import pytest
    from kpapp import app, camera as cam_mod

    seen = {}
    def search(**kw):
        seen.update(kw)
        return None
    monkeypatch.setattr(cam_mod, "find_working_camera", search)
    with pytest.raises(RuntimeError, match="--camera iphone"):
        app._ensure_streaming(_WarmingCamera(delay=1e9), DEFAULT, Config(), timeout=0.1,
                              opener=_no_reopen)
    assert 1 in seen["skip"], "the phone must be excluded from the search"


def test_dead_built_in_camera_is_revived_before_anything_else():
    """A reopen brought this Mac's camera back once; try that first."""
    from kpapp.app import _ensure_streaming

    opened = []
    def opener(idx):
        opened.append(idx)
        return _WarmingCamera(delay=0.0)  # streams once reopened
    cam = _ensure_streaming(_WarmingCamera(delay=1e9), DEFAULT, Config(), timeout=0.1,
                            opener=opener)
    assert opened == [0], "should reopen the built-in camera, not switch"
    assert cam.last_real_time > 0


def test_camera_by_name_or_number():
    import pytest

    cams = [MAC, PHONE]
    assert resolve_camera("iphone", cams).index == 1
    assert resolve_camera("MacBook", cams).index == 0
    assert resolve_camera("1", cams).index == 1
    assert resolve_camera("iphone", cams).explicit
    with pytest.raises(ValueError, match="no camera matches"):
        resolve_camera("webcam", cams)



def _fake_pmset(monkeypatch, batt: str, settings: str):
    import subprocess

    def run(cmd, **kw):
        out = batt if cmd[-1] == "batt" else settings
        return SimpleNamespace(stdout=out)
    monkeypatch.setattr(subprocess, "run", run)


def test_power_tip_flags_low_power_mode(monkeypatch):
    """Low Power Mode costs ~30 ms of pose lag; only the player can fix it."""
    import sys
    from kpapp.app import _power_tip

    monkeypatch.setattr(sys, "platform", "darwin")
    _fake_pmset(monkeypatch, "Now drawing from 'Battery Power'", " powermode            1\n")
    assert "Low Power Mode" in _power_tip()
    _fake_pmset(monkeypatch, "Now drawing from 'Battery Power'", " powermode            0\n")
    assert "battery" in _power_tip().lower()
    _fake_pmset(monkeypatch, "Now drawing from 'AC Power'", " powermode            0\n")
    assert _power_tip() is None
