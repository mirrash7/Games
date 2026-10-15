"""The synth and mixing engine behind Raccoon Dance's songs (tools/generate_dance_song.py).

numpy oscillators, noise and convolution; no samples, nothing downloaded. A song
is a module in tools/dance_songs/ that provides its timing, an `arrange(mix)`
that places notes and hits on a `Mix`, and its mix settings (`SongSpec`).
Everything here is shared, so two songs differ only in their own module.

Timing is per song: `mix.sample(beat)` is the sample index of a beat,
`mix.seconds(beats)` a length, and `mix.rng(*key)` a noise source seeded by
(song seed, event), so an excerpt renders the same events as the full song and
two renders are byte-identical.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Callable

import numpy as np

SR = 44100
TABLE_N = 4096


def midi_hz(note: float) -> float:
    return 440.0 * 2.0 ** ((note - 69) / 12)


def seeded_rng(seed: int, *key) -> np.random.Generator:
    """A noise source seeded by the event, so excerpts match the full render."""
    return np.random.default_rng([seed, zlib.crc32(repr(key).encode())])


# ---------------------------------------------------------------- the song's settings

@dataclass(frozen=True)
class SongSpec:
    """Timing and mix settings of one song. The defaults are Neon Nights'."""

    id: str
    bpm: float
    seed: int
    music_beats: int
    duration: float                     # seconds, the audio file's exact length
    bus_levels: dict
    reverb_sends: dict
    delay_sends: dict
    ducked: dict                        # bus -> how much it follows the sidechain
    peak_dbfs: float = -1.8             # PCM ceiling; AAC overshoots ~0.8 dB, so decoded peaks land near -1 dBFS
    bitrate: int = 160_000
    sidechain_depth: float = 0.65
    sidechain_tau: float = 0.085        # seconds
    delay_beats: float = 0.75           # ping-pong delay time
    delay_feedback: float = 0.4
    delay_band: tuple = (300.0, 4500.0)
    reverb_seconds: float = 2.2
    reverb_rt60: float = 1.7
    reverb_band: tuple = (250.0, 5500.0)
    wet: float = 0.32
    drive: float = 1.5                  # into the limiter at the loudest peak (~3.5 dB)
    fade: float = 0.4                   # the last of the ring-out goes to silence

    @property
    def spb(self) -> float:
        """Samples per beat (22050 at 120 BPM; fractional at most tempos)."""
        return SR * 60 / self.bpm

    @property
    def n_total(self) -> int:
        return int(round(self.duration * SR))

    def sample(self, beat: float) -> int:
        return int(round(beat * self.spb))

    def seconds(self, beats: float) -> float:
        return beats * 60 / self.bpm


# ---------------------------------------------------------------- building blocks

_TABLES: dict = {}


def wavetable(shape: str, f0: float, cutoff_hz: float) -> np.ndarray:
    """One band-limited cycle of `shape`, low-passed at cutoff_hz (2-pole response)."""
    nh = max(1, int(15000 / f0))
    ch = round(float(np.clip(cutoff_hz / f0, 0.5, nh)), 1)
    key = (shape, nh, ch)
    if key not in _TABLES:
        k = np.arange(1, nh + 1, dtype=np.float64)
        if shape == "saw":
            amp = 1 / k
        elif shape == "square":
            amp = np.where(k % 2 == 1, 1 / k, 0.0)
        else:  # "sine"
            amp = (k == 1).astype(np.float64)
        amp = amp / np.sqrt(1 + (k / ch) ** 4)
        x = np.arange(TABLE_N) / TABLE_N
        tab = np.zeros(TABLE_N)
        for kk, a in zip(k, amp):
            if a > 1e-4:
                tab += a * np.sin(2 * np.pi * kk * x)
        _TABLES[key] = tab / 1.85  # the unfiltered saw peaks near 1.85
    return _TABLES[key]


def osc(tab: np.ndarray, freq, n: int, phase0: float = 0.0) -> np.ndarray:
    """Wavetable oscillator; `freq` is a scalar or a per-sample array (Hz)."""
    if np.ndim(freq) == 0:
        ph = phase0 + freq / SR * np.arange(n)
    else:
        ph = phase0 + np.cumsum(freq) / SR
    ph = (ph % 1.0) * TABLE_N
    i = ph.astype(np.int64)
    frac = ph - i
    a = tab[i]
    return a + (tab[(i + 1) & (TABLE_N - 1)] - a) * frac


def envelope(gate: float, attack: float, decay: float, sustain: float, release: float) -> np.ndarray:
    """ADSR in seconds; starts and ends at exactly 0, so notes never click."""
    n_gate = max(1, int(gate * SR))
    n_rel = max(1, int(release * SR))
    t = np.arange(n_gate) / SR
    env = sustain + (1 - sustain) * np.exp(-np.maximum(t - attack, 0) / max(decay, 1e-4))
    na = max(1, int(attack * SR))
    ramp = np.sin(0.5 * np.pi * np.minimum(np.arange(n_gate) / na, 1.0))
    env = env * ramp
    level = env[-1]
    rel = level * np.cos(0.5 * np.pi * np.arange(1, n_rel + 1) / n_rel)
    return np.concatenate([env, rel])


def spectral_filter(x: np.ndarray, lo: float = 0.0, hi: float = 0.0, order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth-magnitude band filter along axis 0 via FFT."""
    n = x.shape[0]
    size = 1 << int(np.ceil(np.log2(max(n, 2))))
    spec = np.fft.rfft(x, size, axis=0)
    f = np.fft.rfftfreq(size, 1 / SR)
    g = np.ones_like(f)
    if hi:
        g /= np.sqrt(1 + (f / hi) ** (2 * order))
    if lo:
        g /= np.sqrt(1 + (lo / np.maximum(f, 1e-6)) ** (2 * order))
    if x.ndim == 2:
        g = g[:, None]
    return np.fft.irfft(spec * g, size, axis=0)[:n]


def pan(mono: np.ndarray, p: float) -> np.ndarray:
    """Equal-power pan; p = -1 left .. +1 right. Returns (n, 2)."""
    a = (p + 1) * np.pi / 4
    return np.stack([mono * np.cos(a), mono * np.sin(a)], axis=1)


# ---------------------------------------------------------------- the mix buffer

@dataclass
class Mix:
    """Bus buffers for samples [s0, s1) of the song; events outside are skipped."""

    spec: SongSpec
    s0: int
    s1: int
    shots: Callable[["Mix", str], np.ndarray] | None = None   # a song's own drum kit
    buses: dict = field(default_factory=dict)
    hits: list = field(default_factory=list)   # (bus, shot name, beat) of every drum hit placed, for tests

    def sample(self, beat: float) -> int:
        return self.spec.sample(beat)

    def seconds(self, beats: float) -> float:
        return self.spec.seconds(beats)

    def rng(self, *key) -> np.random.Generator:
        return seeded_rng(self.spec.seed, *key)

    def bus(self, name: str) -> np.ndarray:
        if name not in self.buses:
            self.buses[name] = np.zeros((self.s1 - self.s0, 2))
        return self.buses[name]

    def wants(self, start: int, length: int) -> bool:
        return start < self.s1 and start + length > self.s0

    def add(self, name: str, start: int, audio: np.ndarray) -> None:
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=1)
        a, b = max(start, self.s0), min(start + len(audio), self.s1)
        if a < b:
            self.bus(name)[a - self.s0:b - self.s0] += audio[a - start:b - start]


# ---------------------------------------------------------------- instruments (one-shots)

_SHOTS: dict = {}


def classic_shot(rng: np.random.Generator, name: str) -> np.ndarray:
    """The original electro kit (Neon Nights): kick, clap, snare, hat, openhat, crash (stereo)."""
    if name == "kick":
        n = int(0.45 * SR)
        t = np.arange(n) / SR
        f = 48 + 112 * np.exp(-t / 0.028) + 40 * np.exp(-t / 0.004)
        body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.26)
        click = spectral_filter(rng.standard_normal(n), lo=1500, hi=6000) * np.exp(-t / 0.0025) * 0.25
        x = np.tanh(1.6 * (body + click)) / np.tanh(1.6)
        x *= np.minimum(1, (n - np.arange(n)) / (0.02 * SR))
        return np.stack([x, x], axis=1)
    if name == "clap":
        n = int(0.32 * SR)
        t = np.arange(n) / SR
        noise = spectral_filter(rng.standard_normal((n, 2)), lo=900, hi=3200, order=2)
        env = np.zeros(n)
        for k, d in enumerate((0.0, 0.011, 0.022)):
            tt = np.maximum(t - d, 0) * (t >= d)
            env += (t >= d) * np.exp(-tt / (0.006 if k < 2 else 0.11)) * (0.8 if k < 2 else 1.0)
        out = noise * env[:, None] * 0.9
        out[:, 1] = 0.85 * out[:, 1] + 0.15 * out[:, 0]
        return out
    if name == "snare":
        n = int(0.3 * SR)
        t = np.arange(n) / SR
        tone = (np.sin(2 * np.pi * 190 * t) + 0.5 * np.sin(2 * np.pi * 330 * t)) * np.exp(-t / 0.05)
        noise = spectral_filter(rng.standard_normal(n), lo=1800, hi=9000) * np.exp(-t / 0.09)
        x = 0.5 * tone + 0.9 * noise
        x *= np.minimum(1, np.arange(n) / 40)
        return np.stack([x, x], axis=1)
    if name in ("hat", "openhat"):
        n = int((0.09 if name == "hat" else 0.35) * SR)
        t = np.arange(n) / SR
        noise = spectral_filter(rng.standard_normal((n, 2)), lo=7500, hi=16000, order=3)
        # a metallic edge: a few inharmonic square partials
        metal = sum(np.sign(np.sin(2 * np.pi * fr * t)) for fr in (3150, 4270, 5820, 7330)) * 0.04
        x = noise + metal[:, None] * 0.6
        out = x * np.exp(-t / (0.022 if name == "hat" else 0.12))[:, None]
        out *= np.minimum(1, np.arange(n) / 20)[:, None]
        return out
    if name == "crash":
        n = int(2.6 * SR)
        t = np.arange(n) / SR
        noise = spectral_filter(rng.standard_normal((n, 2)), lo=3500, hi=15000, order=2)
        out = noise * (np.exp(-t / 0.75) * np.minimum(1, np.arange(n) / 30))[:, None]
        out *= np.minimum(1, (n - np.arange(n)) / (0.2 * SR))[:, None]
        return out
    raise KeyError(name)


def shot(mix: Mix, name: str) -> np.ndarray:
    """A drum one-shot, synthesised once per (song seed, kit, name)."""
    design = mix.shots or (lambda m, nm: classic_shot(m.rng("shot", nm), nm))
    key = (mix.spec.seed, mix.shots is None, name)
    if key not in _SHOTS:
        _SHOTS[key] = design(mix, name)
    return _SHOTS[key]


def hit(mix: Mix, bus: str, name: str, beat: float, gain: float, p: float = 0.0) -> None:
    s = mix.sample(beat)
    x = shot(mix, name)
    mix.hits.append((bus, name, beat))
    if not mix.wants(s, len(x)):
        return
    if p:
        a = (p + 1) * np.pi / 4
        x = x * np.array([np.cos(a), np.sin(a)]) * np.sqrt(2)
    mix.add(bus, s, x * gain)


# ---------------------------------------------------------------- instruments (pitched)

def bass_note(mix: Mix, beat: float, length: float, note: int, gain: float, bright: float = 900,
              legato: float = 0.9) -> None:
    """Saturated saw + sine bass with a plucky filter envelope."""
    s = mix.sample(beat)
    gate = mix.seconds(length) * legato
    env = envelope(gate, 0.004, 0.18, 0.7, 0.025)
    if not mix.wants(s, len(env)):
        return
    f = midi_hz(note)
    n = len(env)
    t = np.arange(n) / SR
    open_ = np.exp(-t / 0.07)  # filter envelope: a bright pluck settling dark
    saw = osc(wavetable("saw", f, bright), f, n) * open_ + osc(wavetable("saw", f, 220), f, n) * (1 - open_)
    sub = osc(wavetable("sine", f, f), f, n)
    x = np.tanh(1.8 * (0.8 * saw + 0.6 * sub)) * env
    mix.add("bass", s, x * gain)


def pad_chord(mix: Mix, beat: float, length: float, notes, gain: float,
              cutoff=(1400.0, 1400.0), attack: float = 0.12, release: float = 0.6,
              transpose: int = 0, key: str = "") -> None:
    """Detuned saw pad, spread in stereo; `cutoff` sweeps from start to end. `key` names the chord (its seed)."""
    s = mix.sample(beat)
    gate = mix.seconds(length)
    env = envelope(gate, attack, 0.5, 0.85, release)
    if not mix.wants(s, len(env)):
        return
    n = len(env)
    rng = mix.rng("pad", beat, key, transpose)
    sweep = np.linspace(0, 1, n) ** 1.5 if cutoff[0] != cutoff[1] else None
    out = np.zeros((n, 2))
    for note in notes:
        f = midi_hz(note + transpose)
        dark, bright = wavetable("saw", f, cutoff[0]), wavetable("saw", f, cutoff[1])
        for ch, detunes in enumerate(((-11, 3), (-3, 11))):
            for cents in detunes:
                ff = f * 2 ** (cents / 1200)
                ph = rng.random()
                if sweep is None:
                    v = osc(bright, ff, n, ph)
                else:
                    v = osc(dark, ff, n, ph) * (1 - sweep) + osc(bright, ff, n, ph) * sweep
                out[:, ch] += v
    mix.add("pad", s, out * env[:, None] * (gain / 4))


def pluck(mix: Mix, beat: float, note: int, gain: float, p: float, length: float = 0.5,
          bus: str = "arp") -> None:
    """Square + saw pluck with a fast filter decay."""
    s = mix.sample(beat)
    env = envelope(mix.seconds(length), 0.003, 0.16, 0.25, 0.08)
    if not mix.wants(s, len(env)):
        return
    f = midi_hz(note)
    n = len(env)
    t = np.arange(n) / SR
    open_ = np.exp(-t / 0.05)
    tone = (osc(wavetable("square", f, 4500), f, n) * open_
            + osc(wavetable("square", f, 900), f, n) * (1 - open_))
    tone += 0.5 * osc(wavetable("saw", f * 1.004, 1500), f * 1.004, n)
    mix.add(bus, s, pan(tone * env * gain, p))


def lead_note(mix: Mix, beat: float, length: float, note: int, gain: float, bus: str = "lead") -> None:
    """Detuned supersaw-ish lead with delayed vibrato."""
    s = mix.sample(beat)
    gate = mix.seconds(length) * 0.92
    env = envelope(gate, 0.008, 0.25, 0.75, 0.07)
    if not mix.wants(s, len(env)):
        return
    f = midi_hz(note)
    n = len(env)
    t = np.arange(n) / SR
    vib = 1 + (2 ** (14 / 1200) - 1) * np.sin(2 * np.pi * 5.5 * t) * np.clip((t - 0.18) / 0.2, 0, 1)
    a = osc(wavetable("saw", f, 4200), f * vib, n)
    b = osc(wavetable("saw", f, 4200), f * vib * 2 ** (9 / 1200), n, 0.37)
    c = osc(wavetable("square", f / 2, 2500), f / 2 * vib, n, 0.11)
    stereo = np.stack([0.55 * a + 0.45 * b + 0.3 * c, 0.45 * a + 0.55 * b + 0.3 * c], axis=1)
    mix.add(bus, s, stereo * (env * gain)[:, None])


def bell(mix: Mix, beat: float, length: float, note: int, gain: float) -> None:
    """Additive bell: four sine partials, each with its own decay."""
    s = mix.sample(beat)
    n = int((mix.seconds(length) + 1.2) * SR)
    if not mix.wants(s, n):
        return
    f = midi_hz(note)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for ratio, amp, tau in ((1, 1.0, 1.1), (2.0, 0.35, 0.6), (2.76, 0.25, 0.35), (5.4, 0.12, 0.15)):
        x += amp * np.sin(2 * np.pi * f * ratio * t) * np.exp(-t / tau)
    x *= np.minimum(1, np.arange(n) / 60) * np.minimum(1, (n - np.arange(n)) / (0.05 * SR))
    mix.add("bell", s, x * gain)


def riser(mix: Mix, beat: float, beats: float = 4, gain: float = 0.35, lo: float = 400, hi: float = 9000,
          bus: str = "fx") -> None:
    """A noise sweep over `beats`, band rising from `lo` to `hi` Hz, swelling."""
    s = mix.sample(beat)
    n = mix.sample(beats)
    if not mix.wants(s, n):
        return
    rng = mix.rng("riser", beat)
    noise = rng.standard_normal((n + 2048, 2))
    hop, win = 512, np.hanning(2048)
    out = np.zeros((n + 2048, 2))
    freqs = np.fft.rfftfreq(2048, 1 / SR)
    for start in range(0, n, hop):
        prog = start / n
        centre = lo * (hi / lo) ** prog
        band = np.exp(-0.5 * (np.log2(np.maximum(freqs, 1) / centre) / 1.1) ** 2)
        seg = noise[start:start + 2048] * win[:, None]
        out[start:start + 2048] += np.fft.irfft(np.fft.rfft(seg, axis=0) * band[:, None], 2048, axis=0)
    out = out[:n] / 1.5
    t = np.arange(n) / n
    out *= (t ** 2.2 * np.minimum(1, (n - np.arange(n)) / 200))[:, None]
    mix.add(bus, s, out * gain)


# ---------------------------------------------------------------- effects and mastering

def sidechain(spec: SongSpec, kicks, s0: int, s1: int) -> np.ndarray:
    """Gain that dips after every kick (the pumping of dance music)."""
    g = np.ones(s1 - s0)
    n = int(0.3 * SR)
    t = np.arange(n) / SR
    dip = 1 - spec.sidechain_depth * np.exp(-t / spec.sidechain_tau) * np.minimum(1, t / 0.004 + 0.6)
    for b in kicks:
        s = spec.sample(b)
        a, e = max(s, s0), min(s + n, s1)
        if a < e:
            g[a - s0:e - s0] = np.minimum(g[a - s0:e - s0], dip[a - s:e - s])
    return g[:, None]


def ping_pong(x: np.ndarray, d: int, feedback: float = 0.4, taps: int = 6,
              band: tuple = (300, 4500)) -> np.ndarray:
    """Ping-pong delay of `d` samples: repeats alternate right and left, getting darker."""
    mono = x.mean(axis=1)
    out = np.zeros_like(x)
    for k in range(1, taps + 1):
        sh = d * k
        if sh >= len(mono):
            break
        out[sh:, k % 2] += mono[:-sh] * feedback ** k  # right, left, right...
    return spectral_filter(out, lo=band[0], hi=band[1])


def reverb_ir(rng: np.random.Generator, seconds: float = 2.2, rt60: float = 1.7,
              band: tuple = (250, 5500)) -> np.ndarray:
    n = int(seconds * SR)
    t = np.arange(n) / SR
    ir = rng.standard_normal((n, 2)) * np.exp(-t * 6.91 / rt60)[:, None]
    ir = spectral_filter(ir, lo=band[0], hi=band[1])
    ir[: int(0.018 * SR)] = 0                                   # pre-delay
    ir *= np.minimum(1, (n - np.arange(n)) / (0.1 * SR))[:, None]
    return ir / np.sqrt((ir ** 2).sum(axis=0).mean())


def convolve(x: np.ndarray, ir: np.ndarray) -> np.ndarray:
    n = len(x)
    size = 1 << int(np.ceil(np.log2(n + len(ir))))
    out = np.fft.irfft(np.fft.rfft(x, size, axis=0) * np.fft.rfft(ir, size, axis=0), size, axis=0)
    return out[:n]


def limit(x: np.ndarray, ceiling: float, block: int = 256, release: float = 0.12) -> np.ndarray:
    """Look-ahead block limiter: the gain is down before a peak arrives, so no sample exceeds `ceiling`."""
    n = len(x)
    nb = -(-n // block)
    peak = np.pad(np.abs(x), ((0, nb * block - n), (0, 0))).reshape(nb, block, 2).max(axis=(1, 2))
    need = np.minimum(1.0, ceiling / np.maximum(peak, 1e-9))
    need = np.minimum(need, np.minimum(np.r_[1.0, need[:-1]], np.r_[need[1:], 1.0]))
    rel = block / (release * SR)
    g = np.empty(nb)
    level = 1.0
    for i in range(nb):
        level = min(need[i], level + (1 - level) * rel)
        g[i] = level
    centres = np.arange(nb) * block + block / 2
    gain = np.interp(np.arange(n), centres, g)
    return x * gain[:, None]


def render(song: ModuleType, start_beat: float = 0.0, end_beat: float | None = None,
           master: bool = True, buses: bool = False):
    """Render a song module (or an excerpt of it) to float32 stereo (n, 2) at SR.

    The excerpt contains exactly the full song's events in that range (seeded
    per event); tails from earlier notes and the mastering gain differ.
    `buses=True` returns the dry, levelled bus signals instead (for balancing).
    """
    spec: SongSpec = song.SPEC
    s0 = spec.sample(start_beat)
    s1 = spec.n_total if end_beat is None else min(spec.n_total, spec.sample(end_beat))
    mix = Mix(spec, s0, s1, shots=getattr(song, "make_shot", None))
    song.arrange(mix)
    duck = sidechain(spec, song.kick_beats(), s0, s1)
    dry = np.zeros((s1 - s0, 2))
    rev_send = np.zeros_like(dry)
    dly_send = np.zeros_like(dry)
    levelled = {}
    for name, buf in mix.buses.items():
        if name in spec.ducked:
            buf = buf * (1 - spec.ducked[name] * (1 - duck))
        if name == "bass":
            buf = spectral_filter(buf, lo=30, hi=2200)
        buf = buf * spec.bus_levels[name]
        levelled[name] = buf
        dry += buf
        rev_send += buf * spec.reverb_sends.get(name, 0.0)
        dly_send += buf * spec.delay_sends.get(name, 0.0)
    if buses:
        return levelled
    delayed = ping_pong(dly_send, spec.sample(spec.delay_beats), spec.delay_feedback, band=spec.delay_band)
    ir = reverb_ir(mix.rng("reverb"), spec.reverb_seconds, spec.reverb_rt60, spec.reverb_band)
    wet = convolve(rev_send + 0.5 * delayed, ir)
    out = dry + delayed + spec.wet * wet
    if not master:
        return out.astype(np.float32)
    ceiling = 10 ** (spec.peak_dbfs / 20)
    out = out / max(np.abs(out).max(), 1e-9) * ceiling * spec.drive
    out = limit(np.tanh(out * 0.9) / 0.9, ceiling)
    fade = int(spec.fade * SR)
    if s1 == spec.n_total and len(out) > fade:
        out[-fade:] *= np.cos(0.5 * np.pi * np.arange(fade) / fade)[:, None] ** 2
    out *= ceiling / max(np.abs(out).max(), 1e-9)
    return out.astype(np.float32)


def to_int16(pcm: np.ndarray) -> np.ndarray:
    return np.round(np.clip(pcm, -1, 1) * 32767).astype(np.int16)


def encode_m4a(pcm: np.ndarray, path: Path, bitrate: int = 160_000) -> None:
    """AAC-LC in an MP4 container (44.1 kHz stereo). FFmpeg writes an edit list, so
    decoders drop the encoder's priming samples and beat 0 stays at 0.0 s."""
    import av  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w", format="mp4", options={"movflags": "+faststart"}) as out:
        stream = out.add_stream("aac", rate=SR, layout="stereo")
        stream.bit_rate = bitrate
        planar = np.ascontiguousarray(pcm.T.astype(np.float32))
        for i in range(0, planar.shape[1], 1024):
            frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(planar[:, i:i + 1024]),
                                               format="fltp", layout="stereo")
            frame.sample_rate = SR
            frame.pts = i
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)
