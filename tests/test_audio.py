"""Music playback clocks (src/kpapp/audio.py, audio_device.py; docs/DANCE.md section 8).

The dance game's song time *is* the player's position(), so the clock must be
monotonic, freeze on pause, continue from the same place on resume, and (for
the real device) advance at real-time rate with output latency compensated.
"""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

from kpapp.audio import SilentPlayer, file_duration, open_player

SONG = Path(__file__).resolve().parents[1] / "assets" / "dance" / "songs" / "neon_nights" / "audio.m4a"


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


# ---------------------------------------------------------------- SilentPlayer

def test_silent_player_clock():
    clock = Clock()
    p = SilentPlayer(10.0, clock=clock)
    assert p.position() == 0.0 and not p.playing and not p.finished
    p.play()
    clock.t += 1.5
    assert p.position() == pytest.approx(1.5) and p.playing
    p.pause()
    clock.t += 5.0
    assert p.position() == pytest.approx(1.5) and not p.playing, "paused: the clock stands still"
    p.pause()
    assert p.position() == pytest.approx(1.5), "a second pause changes nothing"
    p.resume()
    assert p.position() == pytest.approx(1.5), "resume continues from the pause"
    clock.t += 2.0
    assert p.position() == pytest.approx(3.5)
    p.resume()
    assert p.position() == pytest.approx(3.5), "resume while playing changes nothing"
    p.play(8.0)
    clock.t += 1.0
    assert p.position() == pytest.approx(9.0)
    clock.t += 5.0
    assert p.position() == 10.0 and p.finished, "clamped at the end"
    p.stop()
    assert p.position() == 0.0 and not p.playing and not p.finished


def test_silent_player_never_goes_backwards_while_playing():
    clock = Clock()
    p = SilentPlayer(30.0, clock=clock)
    rng = np.random.default_rng(1)
    p.play()
    last = 0.0
    for _ in range(300):
        clock.t += rng.uniform(0, 0.05)
        action = rng.integers(0, 6)
        if action == 0:
            p.pause()
        elif action == 1:
            p.resume()
        pos = p.position()
        assert pos >= last
        last = pos


def test_open_player_falls_back_to_silence_with_the_files_duration(monkeypatch):
    p = open_player(SONG, silent=True)
    assert isinstance(p, SilentPlayer) and p.duration == pytest.approx(106.0, abs=1e-3)
    # No sounddevice / no PortAudio: still a working clock.
    monkeypatch.setitem(sys.modules, "kpapp.audio_device", None)
    p = open_player(SONG)
    assert isinstance(p, SilentPlayer) and p.duration == pytest.approx(106.0, abs=1e-3)
    assert file_duration(None) == 0.0 and file_duration("/nonexistent.m4a") == 0.0


def test_headless_runs_are_silent(monkeypatch):
    """`--headless` sets audio.SILENT: benchmarks keep the music clock but make no sound."""
    import kpapp.audio as audio

    monkeypatch.setattr(audio, "SILENT", True)
    p = open_player(SONG)
    assert isinstance(p, SilentPlayer) and p.duration == pytest.approx(106.0, abs=1e-3)


# ---------------------------------------------------------------- AudioPlayer decode

def write_tone(path: Path, seconds: float, rate: int = 44100, amp: float = 0.0) -> None:
    av = pytest.importorskip("av")
    n = int(seconds * rate)
    t = np.arange(n) / rate
    pcm = np.stack([amp * np.sin(2 * np.pi * 440 * t)] * 2).astype(np.float32)
    with av.open(str(path), "w", format="mp4") as out:
        stream = out.add_stream("aac", rate=rate, layout="stereo")
        for i in range(0, n, 1024):
            frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(pcm[:, i:i + 1024]), format="fltp",
                                               layout="stereo")
            frame.sample_rate = rate
            frame.pts = i
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)


def test_decode_length_matches_duration(tmp_path):
    from kpapp.audio_device import decode

    path = tmp_path / "tone.m4a"
    write_tone(path, 1.3, amp=0.3)
    for rate in (44100, 48000):
        data = decode(path, rate)
        assert data.dtype == np.float32 and data.shape[1] == 2
        assert abs(len(data) / rate - 1.3) < 0.02
    song = decode(SONG, 44100)
    assert len(song) == 106 * 44100, "the song decodes to exactly 106.0 s (priming and padding trimmed)"


# ---------------------------------------------------------------- AudioPlayer clock (fake device)

class FakeStream:
    """Stands in for sounddevice.RawOutputStream; the test drives its clock and callback."""

    def __init__(self, samplerate, channels, dtype, device, latency, callback) -> None:
        self.callback = callback
        self.time = 0.0
        self.latency = 0.03
        self.active = False

    def start(self) -> None:
        self.active = True

    def stop(self) -> None:
        self.active = False

    def close(self) -> None:
        self.active = False


@pytest.fixture
def fake_player(monkeypatch, tmp_path):
    fake = types.SimpleNamespace(
        query_devices=lambda device=None, kind=None: {"default_samplerate": 48000.0},
        RawOutputStream=FakeStream,
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    from kpapp.audio_device import AudioPlayer

    path = tmp_path / "tone.m4a"
    write_tone(path, 2.0, amp=0.3)
    return AudioPlayer(path)


def run_device(player, seconds: float, block: int = 512, latency: float = 0.03) -> None:
    """Advance the fake device: one callback per block, DAC time `latency` ahead."""
    stream = player._stream
    for _ in range(max(1, round(seconds * player.rate / block))):
        out = np.zeros((block, 2), np.float32)
        info = types.SimpleNamespace(outputBufferDacTime=stream.time + latency, currentTime=stream.time)
        stream.callback(out, block, info, None)
        stream.time += block / player.rate


def test_audio_player_position_is_the_audible_position(fake_player):
    p = fake_player
    assert p.rate == 48000 and p.duration == pytest.approx(2.0, abs=0.02)
    p.play()
    assert p.position() == 0.0
    run_device(p, 0.5)
    # The device clock has run ~0.5 s, but the speaker is 30 ms behind the callback.
    played = p._stream.time
    assert played == pytest.approx(0.5, abs=0.01)
    assert p.position() == pytest.approx(played - 0.03, abs=1e-6)
    p._stream.time += 0.004  # between callbacks the position keeps moving
    assert p.position() == pytest.approx(played - 0.03 + 0.004, abs=1e-6)


def test_audio_player_pause_resume_never_jumps(fake_player):
    p = fake_player
    p.play(0.25)
    run_device(p, 0.4)
    heard = p.position()
    p.pause()
    assert not p.playing and p.position() == heard
    run_device(p, 0.3)
    assert p.position() == heard, "paused: frozen"
    p.resume()
    assert p.position() == heard
    run_device(p, 0.02)
    assert p.position() == heard, "holds until the resumed audio reaches the speaker"
    last = heard
    for _ in range(40):
        run_device(p, 0.01)
        pos = p.position()
        assert last <= pos <= last + 0.02
        last = pos
    assert last > heard + 0.3


def test_audio_player_finishes_at_the_end(fake_player):
    p = fake_player
    p.play(p.duration - 0.2)
    run_device(p, 0.5)
    assert p.finished and p.position() == pytest.approx(p.duration)
    p.stop()
    assert p.position() == 0.0 and not p.playing and not p._stream.active


# ---------------------------------------------------------------- the real device

def real_player(path):
    try:
        import sounddevice as sd  # noqa: PLC0415

        sd.query_devices(kind="output")
        from kpapp.audio_device import AudioPlayer  # noqa: PLC0415

        return AudioPlayer(path)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no audio output device ({exc})")


def test_real_device_clock_runs_at_real_time(tmp_path):
    path = tmp_path / "silence.m4a"
    write_tone(path, 3.0, amp=0.0)  # silent: tests must not make noise
    p = real_player(path)
    try:
        p.play()
        samples = []
        t_end = time.perf_counter() + 1.5
        while time.perf_counter() < t_end:
            samples.append((time.perf_counter(), p.position()))
            time.sleep(0.004)
        assert p.playing
    finally:
        p.close()
    t = np.array([s[0] for s in samples])
    pos = np.array([s[1] for s in samples])
    assert np.all(np.diff(pos) >= 0), "monotonic"
    moving = np.nonzero(pos > 0)[0]
    assert len(moving), "the position never advanced"
    a = moving[0] + 10  # skip the start-up
    assert len(t) - a > 20, "too few samples to judge the clock"
    # Under load this thread can be descheduled for tens of ms, so the gap
    # between two samples says nothing about the clock. Judge the clock
    # against the wall clock instead: a least-squares rate (not two
    # endpoints, which one late sample skews), and steps measured *relative*
    # to the wall time that passed between the samples.
    rate = float(np.polyfit(t[a:] - t[a], pos[a:], 1)[0])
    assert rate == pytest.approx(1.0, abs=0.02), f"clock rate {rate:.4f}"
    ahead = np.diff(pos[a:]) - np.diff(t[a:])
    assert ahead.max() < 0.03, "near-continuous: never jumps ahead of real time"
    # The audible position trails the wall clock by the output latency, not more.
    lag = (t[-1] - t[moving[0]]) - (pos[-1] - pos[moving[0]])
    assert -0.03 < lag < 0.25
    print(f"device clock rate {rate:.4f}, largest step ahead {ahead.max() * 1000:.1f} ms, "
          f"latency {p._stream.latency * 1000:.0f} ms, lag {lag * 1000:.0f} ms")
