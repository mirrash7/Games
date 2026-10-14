"""Music playback with a clock games can sync to (docs/DANCE.md section 8).

Surface (the browser's web/js/core/audio.js matches it):

    player = open_player(path)        # AudioPlayer, or SilentPlayer if there is no audio device
    player.play(start=0.0)            # seconds into the track
    player.pause(); player.resume(); player.stop()
    player.position()                 # seconds of audio the player has *heard* (output latency compensated)
    player.duration, player.playing, player.finished
    player.close()                    # stop and release the device

SilentPlayer keeps the same clock without sound (tests, headless runs, no
device). AudioPlayer is the real thing (sounddevice); it lives here so any game
can use music.
"""

from __future__ import annotations

import time
from pathlib import Path


class SilentPlayer:
    """The clock of a player, without sound. `clock` is injectable for tests."""

    def __init__(self, duration: float = 0.0, clock=time.perf_counter) -> None:
        self.duration = float(duration)
        self._clock = clock
        self._start: float | None = None  # clock time at which position 0 played
        self._paused_at: float | None = 0.0
        self.playing = False

    def play(self, start: float = 0.0) -> None:
        self._start = self._clock() - start
        self._paused_at = None
        self.playing = True

    def pause(self) -> None:
        if self.playing:
            self._paused_at = self.position()
            self.playing = False

    def resume(self) -> None:
        if not self.playing and self._paused_at is not None:
            self.play(self._paused_at)

    def stop(self) -> None:
        self.playing = False
        self._paused_at = 0.0

    def position(self) -> float:
        if not self.playing:
            return self._paused_at or 0.0
        pos = self._clock() - (self._start or 0.0)
        return min(pos, self.duration) if self.duration else pos

    @property
    def finished(self) -> bool:
        return bool(self.duration) and self.position() >= self.duration

    def close(self) -> None:
        self.stop()


def file_duration(path: Path | str | None) -> float:
    """The duration a container declares, in seconds (0.0 if unknown)."""
    if not path:
        return 0.0
    try:
        import av  # noqa: PLC0415

        with av.open(str(path)) as container:
            stream = container.streams.audio[0]
            if stream.duration is not None:
                return float(stream.duration * stream.time_base)
            return (container.duration or 0) / 1e6
    except Exception:  # noqa: BLE001 - missing or unreadable file
        return 0.0


# The app sets this for headless runs (benchmarks, CI): the same clock, no sound.
SILENT = False


def open_player(path: Path | str | None, duration: float = 0.0, silent: bool | None = None):
    """A real player for `path` if audio works here, else a SilentPlayer.

    The SilentPlayer's duration is `duration`, or the file's own if not given.
    `silent=True` skips the device (tests); the default follows `SILENT`
    (headless runs).
    """
    if silent is None:
        silent = SILENT
    if silent or path is None:
        return SilentPlayer(duration or file_duration(path))
    try:
        from .audio_device import AudioPlayer  # noqa: PLC0415 - optional dependency
    except (ImportError, OSError):  # OSError: sounddevice without a PortAudio library
        return SilentPlayer(duration or file_duration(path))
    try:
        return AudioPlayer(path)
    except Exception as exc:  # noqa: BLE001 - no device, bad file: play on silently
        print(f"[kp] audio unavailable ({exc}); playing silently")
        return SilentPlayer(duration or file_duration(path))
