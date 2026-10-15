"""The landing page's showcase clips (tools/render_showcase.py).

Checks the committed media against the page's limits by probing them with
PyAV, and that the generator's frames are deterministic. Nothing is encoded
here: regenerate with `uv run python tools/render_showcase.py`.
"""

from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "web" / "media"
CLIPS = ("snack", "flappy", "dance")


def _load_tool():
    if "render_showcase" in sys.modules:
        return sys.modules["render_showcase"]
    spec = importlib.util.spec_from_file_location("render_showcase", ROOT / "tools" / "render_showcase.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["render_showcase"] = mod  # dataclasses look their module up here
    spec.loader.exec_module(mod)
    return mod


def _probe(path: Path) -> tuple[float, int, int, int, str]:
    av = pytest.importorskip("av")
    with av.open(str(path)) as box:
        st = box.streams.video[0]
        assert not box.streams.audio, f"{path.name} has an audio track"
        frames = sum(1 for _ in box.decode(st))
        fps = float(st.average_rate)
        return frames / fps, st.codec_context.width, st.codec_context.height, frames, st.codec_context.name


@pytest.mark.parametrize("clip", (*CLIPS, "hero"))
def test_media_exist_within_the_page_limits(clip):
    for ext in ("mp4", "webm", "jpg"):
        assert (MEDIA / f"{clip}.{ext}").exists(), f"missing web/media/{clip}.{ext}: run tools/render_showcase.py"
    assert (MEDIA / f"{clip}.jpg").stat().st_size <= 120 * 1024
    lo, hi = (8.5, 11.0) if clip == "hero" else (6.0, 8.0)
    for ext, codec in (("mp4", "h264"), ("webm", "vp9")):
        path = MEDIA / f"{clip}.{ext}"
        assert path.stat().st_size <= 1_500_000, f"{path.name} is {path.stat().st_size} bytes"
        seconds, w, h, frames, name = _probe(path)
        assert name == codec
        assert (w, h) == (960, 540)
        assert lo <= seconds <= hi, f"{path.name}: {seconds:.2f} s"
        assert frames >= 30 * lo


def test_mp4_is_faststart():
    """The moov atom comes before the media data, so the browser can start playing at once."""
    for clip in (*CLIPS, "hero"):
        data = (MEDIA / f"{clip}.mp4").read_bytes()[:1 << 16]
        moov, mdat = data.find(b"moov"), data.find(b"mdat")
        assert moov != -1 and (mdat == -1 or moov < mdat), clip


def test_poster_is_a_720p_or_540p_jpeg():
    import cv2

    for clip in (*CLIPS, "hero"):
        img = cv2.imread(str(MEDIA / f"{clip}.jpg"))
        assert img is not None and img.shape[1] * 9 == img.shape[0] * 16, clip
        assert img.shape[1] in (960, 1280)


@pytest.mark.parametrize("clip", CLIPS)
def test_frames_are_deterministic(clip):
    tool = _load_tool()

    def first(n=2):
        return list(itertools.islice(tool.clip_frames(tool.make(clip)), n))

    a, b = first(), first()
    assert a[0].shape == (720, 1280, 3)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
    assert not np.array_equal(a[0], a[1]), "the clip moves"
