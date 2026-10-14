"""Synthesise "Neon Nights", Raccoon Dance's test song (docs/DANCE.md section 6).

An original synth-pop / electro track, built from numpy oscillators and noise.
No samples, no downloads, nothing borrowed. 120 BPM, 4/4, A minor. Beat 0 is
the very first sample (offset 0.0). 52 bars = 208 beats = 104.0 s of music,
then 2.0 s of ring-out: 106.0 s in all.

Arrangement (beats; one chord per bar):

    intro    0-15   Am F C G      kick from beat 0; the pad opens up from dark to
                                  bright; hats join at bar 2, offbeat bass at bar 3,
                                  and snare count-in hits on beats 12, 13, 14 (the
                                  game's "3, 2, 1") under a short riser
    verse 1  16-47  Am F C G x2   four-on-the-floor, claps on 2 and 4, 8th hats,
                                  offbeat bass, plucky arpeggio, soft pad;
                                  riser + snare roll in the last bar
    chorus 1 48-79  F G Am C      crash, rolling octave bass, bright pad, open hats,
                    F G E Am      the lead hook, claps + snare
    verse 2  80-111 as verse 1    16th hats, busier arpeggio, a lead answer every 4
                                  bars; riser + roll into the chorus
    chorus 2 112-143 as chorus 1
    bridge   144-159 Dm Am F E    the kick drops: warm pads, claps, held bass, a
                                  slow bell melody; riser + roll in the last bar
    final    160-191 as chorus    the biggest: everything plus the hook an octave up
    outro    192-207 Am . F Am    a final hit on beat 192 (kick, crash, clap, chord
                                  stab), then the chord held for the bow, a fading
                                  echo of the hook, and a plagal F -> Am to close

Mixing: bass / pad / arpeggio duck under the kick (sidechain), the bass is
low-passed and saturated, pads and hats are spread in stereo, the lead and
arpeggio feed a ping-pong delay, everything shares a convolution reverb (a
seeded noise impulse response), and a look-ahead limiter sets peaks to -1 dBFS.

Deterministic: every noise source is seeded from (SEED, event), so a render is
byte-identical run to run, and an excerpt renders the same events as the full
song (`render(start_beat, end_beat)`; tests use that).

    uv run python tools/generate_dance_song.py          # -> assets/dance/songs/neon_nights/audio.m4a
"""

from __future__ import annotations

import argparse
import hashlib
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "assets" / "dance" / "songs" / "neon_nights" / "audio.m4a"

SEED = 20261014
SR = 44100
BPM = 120
SPB = SR * 60 // BPM          # samples per beat: 22050, exact
OFFSET = 0.0                  # seconds into the audio of beat 0
MUSIC_BEATS = 208             # 52 bars
TAIL = 2.0                    # seconds of ring-out after the music
DURATION = MUSIC_BEATS * 60 / BPM + TAIL  # 106.0
N_TOTAL = int(round(DURATION * SR))       # 4_674_600
PEAK_DBFS = -1.8              # PCM ceiling; AAC overshoots ~0.8 dB, so decoded peaks land near -1 dBFS
BITRATE = 160_000

SECTIONS = [  # name, first beat, last beat + 1  (docs/DANCE.md section 6)
    ("intro", 0, 16), ("verse", 16, 48), ("chorus", 48, 80), ("verse", 80, 112),
    ("chorus", 112, 144), ("bridge", 144, 160), ("chorus", 160, 192), ("outro", 192, 208),
]
CRASHES = (48, 80, 112, 144, 160, 176, 192)
RISERS = (12, 44, 108, 156)          # the bar starting here builds into the next section

# Chords: pad voicing (MIDI) and bass root (MIDI).
CHORDS = {
    "Am": ([57, 60, 64, 69], 45),
    "F": ([57, 60, 65, 69], 41),
    "C": ([55, 60, 64, 67], 36),
    "G": ([55, 59, 62, 67], 43),
    "E": ([56, 59, 64, 68], 40),
    "Dm": ([57, 62, 65, 69], 38),
}
VERSE_CHORDS = ["Am", "F", "C", "G"] * 2
CHORUS_CHORDS = ["F", "G", "Am", "C", "F", "G", "E", "Am"]
PROGRESSION = (  # one chord per bar, 52 bars
    ["Am", "F", "C", "G"] + VERSE_CHORDS + CHORUS_CHORDS + VERSE_CHORDS + CHORUS_CHORDS
    + ["Dm", "Am", "F", "E"] + CHORUS_CHORDS + ["Am", "Am", "F", "Am"]
)
assert len(PROGRESSION) == MUSIC_BEATS // 4

# The chorus hook: (beat within the 8-bar chorus, length in beats, MIDI note).
HOOK = [
    (0, .5, 72), (.5, .5, 72), (1, .5, 69), (1.5, 1, 72), (3, .5, 74), (3.5, .5, 76),      # F
    (4, 1, 74), (5, .5, 71), (5.5, 1.5, 67), (7, .5, 71), (7.5, .5, 72),                  # G
    (8, .5, 72), (8.5, .5, 72), (9, .5, 76), (9.5, 1, 76), (10.5, .5, 74), (11, 1, 72),   # Am
    (12, 2, 67), (14.5, .5, 72), (15, .5, 74), (15.5, .5, 76),                             # C
    (16, .5, 72), (16.5, .5, 72), (17, .5, 69), (17.5, 1, 72), (19, .5, 74), (19.5, .5, 76),  # F
    (20, 1, 74), (21, .5, 76), (21.5, 1.5, 79), (23, .5, 76), (23.5, .5, 74),             # G
    (24, 1, 76), (25, .5, 74), (25.5, .5, 71), (26, 1, 68), (27, .5, 71), (27.5, .5, 74), # E
    (28, 3, 69),                                                                            # Am
]
# The verse-2 answer phrase, in the last bar of each 4-bar phrase (over G).
ANSWER = [(0, .5, 74), (.5, .5, 76), (1, 1, 79), (2, .5, 76), (2.5, 1.5, 74)]
# The bridge bell: (beat from 144, length, note) over Dm Am F E.
BELL = [(0, 2, 77), (2, 2, 76), (4, 2, 72), (6, 2, 76), (8, 2, 77), (10, 2, 81),
        (12, 2, 80), (14, 2, 83)]

TABLE_N = 4096


def beat_sample(beat: float) -> int:
    return int(round(beat * SPB))


def midi_hz(note: float) -> float:
    return 440.0 * 2.0 ** ((note - 69) / 12)


def rng_for(*key) -> np.random.Generator:
    """A noise source seeded by the event, so excerpts match the full render."""
    return np.random.default_rng([SEED, zlib.crc32(repr(key).encode())])


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

    s0: int
    s1: int
    buses: dict = field(default_factory=dict)

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


def shot(name: str) -> np.ndarray:
    """Drum one-shots, synthesised once (stereo)."""
    if name in _SHOTS:
        return _SHOTS[name]
    rng = rng_for("shot", name)
    if name == "kick":
        n = int(0.45 * SR)
        t = np.arange(n) / SR
        f = 48 + 112 * np.exp(-t / 0.028) + 40 * np.exp(-t / 0.004)
        body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.26)
        click = spectral_filter(rng.standard_normal(n), lo=1500, hi=6000) * np.exp(-t / 0.0025) * 0.25
        x = np.tanh(1.6 * (body + click)) / np.tanh(1.6)
        x *= np.minimum(1, (n - np.arange(n)) / (0.02 * SR))
        out = np.stack([x, x], axis=1)
    elif name == "clap":
        n = int(0.32 * SR)
        t = np.arange(n) / SR
        noise = spectral_filter(rng.standard_normal((n, 2)), lo=900, hi=3200, order=2)
        env = np.zeros(n)
        for k, d in enumerate((0.0, 0.011, 0.022)):
            tt = np.maximum(t - d, 0) * (t >= d)
            env += (t >= d) * np.exp(-tt / (0.006 if k < 2 else 0.11)) * (0.8 if k < 2 else 1.0)
        out = noise * env[:, None] * 0.9
        out[:, 1] = 0.85 * out[:, 1] + 0.15 * out[:, 0]
    elif name == "snare":
        n = int(0.3 * SR)
        t = np.arange(n) / SR
        tone = (np.sin(2 * np.pi * 190 * t) + 0.5 * np.sin(2 * np.pi * 330 * t)) * np.exp(-t / 0.05)
        noise = spectral_filter(rng.standard_normal(n), lo=1800, hi=9000) * np.exp(-t / 0.09)
        x = 0.5 * tone + 0.9 * noise
        x *= np.minimum(1, np.arange(n) / 40)
        out = np.stack([x, x], axis=1)
    elif name in ("hat", "openhat"):
        n = int((0.09 if name == "hat" else 0.35) * SR)
        t = np.arange(n) / SR
        noise = spectral_filter(rng.standard_normal((n, 2)), lo=7500, hi=16000, order=3)
        # a metallic edge: a few inharmonic square partials
        metal = sum(np.sign(np.sin(2 * np.pi * fr * t)) for fr in (3150, 4270, 5820, 7330)) * 0.04
        x = noise + metal[:, None] * 0.6
        out = x * np.exp(-t / (0.022 if name == "hat" else 0.12))[:, None]
        out *= np.minimum(1, np.arange(n) / 20)[:, None]
    elif name == "crash":
        n = int(2.6 * SR)
        t = np.arange(n) / SR
        noise = spectral_filter(rng.standard_normal((n, 2)), lo=3500, hi=15000, order=2)
        out = noise * (np.exp(-t / 0.75) * np.minimum(1, np.arange(n) / 30))[:, None]
        out *= np.minimum(1, (n - np.arange(n)) / (0.2 * SR))[:, None]
    else:
        raise KeyError(name)
    _SHOTS[name] = out
    return out


def hit(mix: Mix, bus: str, name: str, beat: float, gain: float, p: float = 0.0) -> None:
    s = beat_sample(beat)
    x = shot(name)
    if not mix.wants(s, len(x)):
        return
    if p:
        a = (p + 1) * np.pi / 4
        x = x * np.array([np.cos(a), np.sin(a)]) * np.sqrt(2)
    mix.add(bus, s, x * gain)


# ---------------------------------------------------------------- instruments (pitched)

def bass_note(mix: Mix, beat: float, length: float, note: int, gain: float, bright: float = 900,
              legato: float = 0.9) -> None:
    s = beat_sample(beat)
    gate = length * 60 / BPM * legato
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


def pad_chord(mix: Mix, beat: float, length: float, chord: str, gain: float,
              cutoff=(1400.0, 1400.0), attack: float = 0.12, release: float = 0.6,
              transpose: int = 0) -> None:
    """Detuned saw pad, spread in stereo; `cutoff` sweeps from start to end."""
    s = beat_sample(beat)
    gate = length * 60 / BPM
    env = envelope(gate, attack, 0.5, 0.85, release)
    if not mix.wants(s, len(env)):
        return
    n = len(env)
    rng = rng_for("pad", beat, chord, transpose)
    sweep = np.linspace(0, 1, n) ** 1.5 if cutoff[0] != cutoff[1] else None
    out = np.zeros((n, 2))
    for note in CHORDS[chord][0]:
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
    s = beat_sample(beat)
    env = envelope(length * 60 / BPM, 0.003, 0.16, 0.25, 0.08)
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
    s = beat_sample(beat)
    gate = length * 60 / BPM * 0.92
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
    s = beat_sample(beat)
    n = int((length * 60 / BPM + 1.2) * SR)
    if not mix.wants(s, n):
        return
    f = midi_hz(note)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for ratio, amp, tau in ((1, 1.0, 1.1), (2.0, 0.35, 0.6), (2.76, 0.25, 0.35), (5.4, 0.12, 0.15)):
        x += amp * np.sin(2 * np.pi * f * ratio * t) * np.exp(-t / tau)
    x *= np.minimum(1, np.arange(n) / 60) * np.minimum(1, (n - np.arange(n)) / (0.05 * SR))
    mix.add("bell", s, x * gain)


def riser(mix: Mix, beat: float, beats: float = 4, gain: float = 0.35) -> None:
    """A noise sweep over `beats`, band rising from 400 Hz to 9 kHz, swelling."""
    s = beat_sample(beat)
    n = beat_sample(beats)
    if not mix.wants(s, n):
        return
    rng = rng_for("riser", beat)
    noise = rng.standard_normal((n + 2048, 2))
    hop, win = 512, np.hanning(2048)
    out = np.zeros((n + 2048, 2))
    freqs = np.fft.rfftfreq(2048, 1 / SR)
    for start in range(0, n, hop):
        prog = start / n
        centre = 400 * (9000 / 400) ** prog
        band = np.exp(-0.5 * (np.log2(np.maximum(freqs, 1) / centre) / 1.1) ** 2)
        seg = noise[start:start + 2048] * win[:, None]
        out[start:start + 2048] += np.fft.irfft(np.fft.rfft(seg, axis=0) * band[:, None], 2048, axis=0)
    out = out[:n] / 1.5
    t = np.arange(n) / n
    out *= (t ** 2.2 * np.minimum(1, (n - np.arange(n)) / 200))[:, None]
    mix.add("fx", s, out * gain)


# ---------------------------------------------------------------- the arrangement

def section_of(beat: float) -> tuple[str, int, int]:
    for name, a, b in SECTIONS:
        if a <= beat < b:
            return name, a, b
    return "end", MUSIC_BEATS, MUSIC_BEATS


def kick_beats() -> list[int]:
    """Every beat with a kick: four-on-the-floor except the bridge; 192 is the final hit."""
    return [b for b in range(193) if not 144 <= b < 160]


def arrange(mix: Mix) -> None:
    final = lambda b: 160 <= b < 192  # noqa: E731

    # Drums.
    for b in kick_beats():
        hit(mix, "kick", "kick", b, 1.15 if b == 192 else (0.95 if b < 16 else 1.0))
    for b in CRASHES:
        hit(mix, "drums", "crash", b, 0.30 if b in (80, 144) else 0.42)
    for b in range(12, 15):                                  # count-in: 3, 2, 1
        hit(mix, "drums", "snare", b, 0.30 + 0.06 * (b - 12))
    for b in range(16, 192):
        name, a, _ = section_of(b)
        if b % 4 in (1, 3):
            hit(mix, "drums", "clap", b, 0.55 if name != "bridge" else 0.5)
            if name == "chorus":
                hit(mix, "drums", "snare", b, 0.35 if final(b) else 0.28)
    hit(mix, "drums", "clap", 192, 0.6)
    hit(mix, "drums", "snare", 192, 0.4)
    for i in range(4 * 192):                                 # 16th-note grid for hats
        b = i / 4
        name, a, _ = section_of(b)
        sub = i % 4
        if name == "intro" and b >= 4 and sub == 2:
            hit(mix, "drums", "hat", b, 0.16 + 0.05 * (b >= 8), 0.3)
        elif name == "verse":
            if a == 16 and sub in (0, 2):                    # verse 1: 8ths, offbeat accent
                hit(mix, "drums", "hat", b, 0.22 if sub == 2 else 0.1, 0.3)
            elif a == 80:                                    # verse 2: 16ths
                hit(mix, "drums", "hat", b, (0.10, 0.06, 0.22, 0.07)[sub], 0.3)
        elif name == "chorus":
            if sub == 2:
                hit(mix, "drums", "openhat", b, 0.20 if final(b) else 0.17, -0.25)
            else:
                hit(mix, "drums", "hat", b, (0.12, 0.07, 0, 0.08)[sub], 0.35)
        elif name == "bridge" and sub == 2:
            hit(mix, "drums", "hat", b, 0.07, 0.4)

    # Risers and snare rolls into the next section.
    for b in RISERS:
        riser(mix, b, 4, 0.22 if b == 12 else 0.32)
        if b != 12:
            for i in range(16):
                hit(mix, "drums", "snare", b + i / 4, 0.05 + 0.3 * (i / 15) ** 2)

    # Harmony and bass, bar by bar.
    for bar, chord in enumerate(PROGRESSION):
        b0 = 4 * bar
        name, a, _ = section_of(b0)
        root = CHORDS[chord][1]
        tones = sorted(CHORDS[chord][0])
        if name == "intro":
            lo = 500 * (2600 / 500) ** (bar / 4)
            hi = 500 * (2600 / 500) ** ((bar + 1) / 4)
            pad_chord(mix, b0, 4, chord, 0.32, (lo, hi), attack=0.4 if bar == 0 else 0.15)
            if bar >= 2:
                for k in range(4):
                    bass_note(mix, b0 + k + 0.5, 0.5, root, 0.42)
            if bar == 3:
                for k in range(8):
                    pluck(mix, b0 + k / 2, tones[(0, 1, 2, 3, 2, 1, 2, 3)[k]] + 12, 0.10 * (k + 1) / 8, (-0.4, 0.4)[k % 2])
        elif name == "verse":
            pad_chord(mix, b0, 4, chord, 0.22, (1500, 1500))
            for k in range(4):
                n = root + (12 if (k == 3 and bar % 2) else 0)
                bass_note(mix, b0 + k + 0.5, 0.5, n, 0.48)
            if a == 80 and bar % 2 == 1:                       # verse 2: a syncopated pickup
                bass_note(mix, b0 + 3.75, 0.25, root + 7, 0.32)
            pattern = (0, 1, 2, 3, 2, 1, 2, 3) if a == 16 else (0, 2, 1, 3, 2, 3, 1, 2)
            steps = 8 if a == 16 else 16
            for k in range(steps):
                tone = tones[pattern[k % 8]] + 12 + (12 if a == 80 and k % 8 == 7 else 0)
                pluck(mix, b0 + k * 4 / steps, tone, 0.12 if steps == 8 else 0.09, (-0.45, 0.45)[k % 2],
                      length=0.5 if steps == 8 else 0.3)
        elif name == "chorus":
            big = final(b0)
            pad_chord(mix, b0, 4, chord, 0.36 if big else 0.32, (3200, 3200) if big else (2600, 2600),
                      attack=0.03, release=0.3)
            if big:                                            # the final chorus: an airy octave on top
                pad_chord(mix, b0, 4, chord, 0.14, (5000, 5000), attack=0.03, release=0.3, transpose=12)
            for k in range(8):                                 # rolling octave bass
                bass_note(mix, b0 + k / 2, 0.5, root + (12 if k % 2 else 0), 0.48, bright=1400)
            for k in range(16):
                pluck(mix, b0 + k / 4, tones[(0, 2, 3, 1)[k % 4]] + 24, 0.08 if big else 0.06,
                      (-0.6, 0.6)[k % 2], length=0.25)
        elif name == "bridge":
            pad_chord(mix, b0, 4, chord, 0.75, (1100, 1900), attack=0.5, release=1.0)
            bass_note(mix, b0, 4, root, 0.5, bright=500, legato=1.0)
        elif name == "outro":
            if bar == 48:                                      # the final hit and the held chord
                pad_chord(mix, b0, 8, chord, 0.6, (3000, 1000), attack=0.01, release=1.5)
                bass_note(mix, b0, 3, root - 12 if root - 12 >= 33 else root, 0.55, bright=700)
                for note in tones:
                    lead_note(mix, b0, 1, note + 12, 0.10, bus="stab")
            elif bar == 50:
                pad_chord(mix, b0, 4, "F", 0.45, (1200, 900), attack=0.5, release=1.0)
            elif bar == 51:
                pad_chord(mix, b0, 4, "Am", 0.45, (1000, 700), attack=0.5, release=1.9)

    # Melody.
    for start in (48, 112, 160):
        for beat, length, note in HOOK:
            lead_note(mix, start + beat, length, note, 0.30)
            if start == 160:
                lead_note(mix, start + beat, length, note + 12, 0.12)
    for phrase in (92, 108):                                   # verse 2 answers (over G)
        for beat, length, note in ANSWER:
            lead_note(mix, phrase + beat, length, note, 0.20)
    for beat, length, note in BELL:
        bell(mix, 144 + beat, length, note, 0.32)
    for i, (beat, length, note) in enumerate(HOOK[:11]):       # the outro echo of the hook
        if beat < 8:
            pluck(mix, 194 + beat, note, 0.13 * (1 - beat / 9), (-0.35, 0.35)[i % 2], length=length)


# ---------------------------------------------------------------- effects and mastering

def sidechain(s0: int, s1: int) -> np.ndarray:
    """Gain that dips after every kick (the pumping of dance music)."""
    g = np.ones(s1 - s0)
    n = int(0.3 * SR)
    t = np.arange(n) / SR
    dip = 1 - 0.65 * np.exp(-t / 0.085) * np.minimum(1, t / 0.004 + 0.6)
    for b in kick_beats():
        s = beat_sample(b)
        a, e = max(s, s0), min(s + n, s1)
        if a < e:
            g[a - s0:e - s0] = np.minimum(g[a - s0:e - s0], dip[a - s:e - s])
    return g[:, None]


def ping_pong(x: np.ndarray, delay_beats: float = 0.75, feedback: float = 0.4, taps: int = 6) -> np.ndarray:
    """Dotted-8th ping-pong delay: repeats alternate right and left, getting darker."""
    mono = x.mean(axis=1)
    d = beat_sample(delay_beats)
    out = np.zeros_like(x)
    for k in range(1, taps + 1):
        sh = d * k
        if sh >= len(mono):
            break
        out[sh:, k % 2] += mono[:-sh] * feedback ** k  # right, left, right...
    return spectral_filter(out, lo=300, hi=4500)


def reverb_ir(seconds: float = 2.2, rt60: float = 1.7) -> np.ndarray:
    rng = rng_for("reverb")
    n = int(seconds * SR)
    t = np.arange(n) / SR
    ir = rng.standard_normal((n, 2)) * np.exp(-t * 6.91 / rt60)[:, None]
    ir = spectral_filter(ir, lo=250, hi=5500)
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


BUS_LEVELS = {"kick": 0.5, "drums": 1.0, "bass": 0.55, "pad": 0.8, "arp": 1.2,
              "lead": 1.0, "stab": 1.2, "bell": 0.6, "fx": 0.9}
REVERB_SENDS = {"drums": 0.08, "pad": 0.22, "arp": 0.25, "lead": 0.22, "stab": 0.4, "bell": 0.45, "fx": 0.2}
DELAY_SENDS = {"arp": 0.25, "lead": 0.3, "bell": 0.2}
DUCKED = {"bass": 1.0, "pad": 0.7, "arp": 0.5}


def render(start_beat: float = 0.0, end_beat: float | None = None, master: bool = True) -> np.ndarray:
    """Render the song, or an excerpt of it, to float32 stereo (n, 2) at SR.

    The excerpt contains exactly the full song's events in that range (seeded
    per event); tails from earlier notes and the mastering gain differ.
    """
    s0 = beat_sample(start_beat)
    s1 = N_TOTAL if end_beat is None else min(N_TOTAL, beat_sample(end_beat))
    mix = Mix(s0, s1)
    arrange(mix)
    duck = sidechain(s0, s1)
    dry = np.zeros((s1 - s0, 2))
    rev_send = np.zeros_like(dry)
    dly_send = np.zeros_like(dry)
    for name, buf in mix.buses.items():
        if name in DUCKED:
            buf = buf * (1 - DUCKED[name] * (1 - duck))
        if name == "bass":
            buf = spectral_filter(buf, lo=30, hi=2200)
        buf = buf * BUS_LEVELS[name]
        dry += buf
        rev_send += buf * REVERB_SENDS.get(name, 0.0)
        dly_send += buf * DELAY_SENDS.get(name, 0.0)
    delayed = ping_pong(dly_send)
    wet = convolve(rev_send + 0.5 * delayed, reverb_ir())
    out = dry + delayed + 0.32 * wet
    if not master:
        return out.astype(np.float32)
    ceiling = 10 ** (PEAK_DBFS / 20)
    out = out / max(np.abs(out).max(), 1e-9) * ceiling * 1.5   # ~3.5 dB into the limiter at the loudest peak
    out = limit(np.tanh(out * 0.9) / 0.9, ceiling)
    fade = int(0.4 * SR)                                        # the last of the ring-out goes to silence
    if s1 == N_TOTAL and len(out) > fade:
        out[-fade:] *= np.cos(0.5 * np.pi * np.arange(fade) / fade)[:, None] ** 2
    out *= ceiling / max(np.abs(out).max(), 1e-9)
    return out.astype(np.float32)


def to_int16(pcm: np.ndarray) -> np.ndarray:
    return np.round(np.clip(pcm, -1, 1) * 32767).astype(np.int16)


def encode_m4a(pcm: np.ndarray, path: Path) -> None:
    """AAC-LC in an MP4 container (44.1 kHz stereo). FFmpeg writes an edit list, so
    decoders drop the encoder's priming samples and beat 0 stays at 0.0 s."""
    import av  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w", format="mp4", options={"movflags": "+faststart"}) as out:
        stream = out.add_stream("aac", rate=SR, layout="stereo")
        stream.bit_rate = BITRATE
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    t0 = time.perf_counter()
    pcm = render()
    t1 = time.perf_counter()
    encode_m4a(pcm, args.out)
    t2 = time.perf_counter()
    peak = 20 * np.log10(np.abs(pcm).max())
    print(f"rendered {len(pcm) / SR:.3f} s ({len(pcm)} samples) in {t1 - t0:.1f} s; peak {peak:.2f} dBFS")
    print(f"pcm md5 float32 {hashlib.md5(pcm.tobytes()).hexdigest()}  "
          f"int16 {hashlib.md5(to_int16(pcm).tobytes()).hexdigest()}")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB) in {t2 - t1:.1f} s; "
          f"md5 {hashlib.md5(args.out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
