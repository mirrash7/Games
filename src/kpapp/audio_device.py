"""AudioPlayer: plays a music file through the default output device (sounddevice).

The surface matches audio.SilentPlayer (see audio.py). The file is decoded up
front with PyAV to float32 stereo at the device's rate; a PortAudio callback
copies it out block by block.

**position() is what the player has heard.** Each callback is told when its
first frame will reach the speaker (`outputBufferDacTime`, on the stream's
clock). position() extends that line to the stream clock's *now*, so it moves
smoothly between callbacks, already compensated for output latency, and never
runs past audio the callback has delivered. It never goes backwards: pause()
freezes it at the audible position (and rewinds the read head there, so
nothing is skipped), and after resume() it holds until new audio is audible.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np


def decode(path: Path | str, rate: int) -> np.ndarray:
    """The whole file as float32 (n, 2) at `rate`, trimmed to the container's duration."""
    import av  # noqa: PLC0415

    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="flt", layout="stereo", rate=rate)
        chunks = []
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1, 2))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1, 2))
        declared = None
        if stream.duration is not None and stream.time_base is not None:
            declared = float(stream.duration * stream.time_base)
    data = np.concatenate(chunks) if chunks else np.zeros((0, 2), np.float32)
    # AAC pads the last frame; the container's duration says where the music ends.
    if declared:
        data = data[: int(round(declared * rate))]
    return np.ascontiguousarray(data, dtype=np.float32)


class AudioPlayer:
    """Plays `path` on the default output device. Raises if there is no device."""

    def __init__(self, path: Path | str, device=None, latency="high") -> None:
        import sounddevice as sd  # noqa: PLC0415 - optional at import time

        info = sd.query_devices(device, kind="output")
        self.rate = int(info["default_samplerate"])
        self._data = decode(path, self.rate)
        self.duration = len(self._data) / self.rate
        self._lock = threading.Lock()
        self._cursor = 0              # next frame the callback copies
        self._anchor = None           # (frame, dac time) of the last delivered block
        self._floor = 0.0             # position() never returns less than this
        self._frozen = 0.0            # position while not playing
        self.playing = False
        # Raw stream: the callback writes into PortAudio's buffer without sounddevice's numpy wrapper.
        self._stream = sd.RawOutputStream(samplerate=self.rate, channels=2, dtype="float32",
                                          device=device, latency=latency, callback=self._callback)

    # -- the PortAudio thread ------------------------------------------------
    def _callback(self, buffer, frames, time_info, status) -> None:  # noqa: ARG002
        outdata = np.frombuffer(buffer, dtype=np.float32).reshape(frames, 2)
        with self._lock:
            if not self.playing:
                outdata.fill(0)
                return
            n = max(0, min(frames, len(self._data) - self._cursor))
            outdata[:n] = self._data[self._cursor:self._cursor + n]
            outdata[n:] = 0
            if n == 0:  # past the end: keep the last anchor so position() runs on to the end
                return
            dac = time_info.outputBufferDacTime
            if not dac:  # some host APIs leave it empty
                dac = (time_info.currentTime or self._stream.time) + self._stream.latency
            self._anchor = (self._cursor, dac)
            self._cursor += n

    # -- the game's side -----------------------------------------------------
    def play(self, start: float = 0.0) -> None:
        start = min(max(0.0, float(start)), self.duration)
        with self._lock:
            self._cursor = int(round(start * self.rate))
            self._anchor = None
            self._floor = start
            self.playing = True
        if not self._stream.active:
            self._stream.start()

    def pause(self) -> None:
        if not self.playing:
            return
        pos = self.position()
        with self._lock:
            self.playing = False
            self._frozen = pos
            self._cursor = int(round(pos * self.rate))  # resume from what was heard

    def resume(self) -> None:
        if not self.playing:
            self.play(self._frozen)

    def stop(self) -> None:
        with self._lock:
            self.playing = False
            self._frozen = 0.0
            self._cursor = 0
            self._anchor = None
        if self._stream.active:
            self._stream.stop()

    def close(self) -> None:
        """Stop and release the device."""
        self.stop()
        self._stream.close()

    def position(self) -> float:
        with self._lock:
            if not self.playing:
                return self._frozen
            if self._anchor is not None:
                frame, dac = self._anchor
                now = self._stream.time
                heard = frame / self.rate + (now - dac)
                delivered = self._cursor / self.rate
                self._floor = max(self._floor, min(heard, delivered, self.duration))
            return self._floor

    @property
    def finished(self) -> bool:
        return self.duration > 0 and self.position() >= self.duration - 1e-3

    def __del__(self) -> None:
        try:
            self._stream.close()
        except Exception:  # noqa: BLE001 - interpreter shutdown
            pass

