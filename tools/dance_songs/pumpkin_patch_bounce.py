"""Pumpkin Patch Bounce: Raccoon Dance's second song, a cheerful-spooky October bop.

Original, in the style of a short viral dance audio: a bouncy 808, snaps and
claps, a lightly shuffled shaker, and one whistled hook built for an easy,
repeatable 8-count routine. Halloween flavour from a haunted-house organ
skank, a music-box celesta, xylophone "bones", a theremin, tolling bells and
a ghostly pad. Nothing borrowed or imitated: numpy synthesis only (the engine
is tools/dance_synth.py).

116 BPM, 4/4, D minor. Beat 0 is the very first sample (offset 0.0).
44 bars = 176 beats = 91.034 s of music, then a ring-out: 93.000 s in all.

Arrangement (beats; one chord per bar; the routine repeats on 8-beat phrases):

    intro    0-15   Dm Gm Bb A    the music box plays the hook alone; kick from
                                  beat 0, rising; snaps at bar 1, shaker, 808 and
                                  organ at bar 2; snare count-in on 12, 13, 14 (the
                                  game's "3, 2, 1") under a riser; beat 15 stops dead
                                  but for the hook's pickup note
    hook 1   16-47  Dm Gm Bb A x2 crash; four-on-the-floor, claps + snaps on 2 and 4,
                                  shuffled shaker, the 808 bounce, organ skank on the
                                  offbeats, the whistled hook
    verse    48-79  Dm C Bb A x2  lighter: steady kick, snaps only, music-box arpeggio,
                                  a sparse 808, theremin "oo-ee-oo" calls at the end of
                                  each phrase
    hook 2   80-111 as hook 1     fuller: celesta doubles the hook, xylophone bones,
                                  open hats
    break    112-127 Dm Dm Gm A   spooky half-time: no kick, a clap on 3, soft snaps,
                                  ghost pad, tolling bell, the hook at half speed on
                                  the music box, a low theremin
    final    128-159 as hook      the biggest: everything, the lead doubled an octave
                                  up, ghost pad underneath
    outro    160-175 Dm Dm Gm Dm  a big hit on 160 (kick, crash, clap, organ, 808, bell),
                                  then the music box winds down under a falling
                                  theremin and a last bell on 172

Phrase marks: a quiet fill (two xylophone taps and a short swell) closes every
8-beat phrase; a riser, snare roll and swell close the bar before each section,
and a crash opens it (16, 48, 80, 112, 128, 160).

    uv run python tools/generate_dance_song.py --song pumpkin_patch_bounce
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from dance_synth import (SR, SongSpec, classic_shot, envelope, hit, midi_hz, riser, spectral_filter)

ID = "pumpkin_patch_bounce"
TITLE = "Pumpkin Patch Bounce"
OUT = Path(__file__).resolve().parents[2] / "assets" / "dance" / "songs" / ID / "audio.m4a"

SEED = 20261031
BPM = 116
OFFSET = 0.0                  # seconds into the audio of beat 0
MUSIC_BEATS = 176             # 44 bars
DURATION = 93.0               # the file's exact length: 91.034 s of music + ring-out
TAIL = DURATION - MUSIC_BEATS * 60 / BPM

SECTIONS = [  # name, first beat, last beat + 1
    ("intro", 0, 16), ("hook", 16, 48), ("verse", 48, 80), ("hook", 80, 112),
    ("break", 112, 128), ("hook", 128, 160), ("outro", 160, 176),
]
SECTION_STARTS = tuple(a for _, a, _ in SECTIONS[1:])
CRASHES = (16, 48, 80, 112, 128, 160)
COUNT_IN = (12, 13, 14)
RISERS = (12, 44, 76, 108, 124, 156)    # the bar starting here builds into the next section
SWING = 0.035                           # beats: the 2nd and 4th 16th of each beat land late

# Chords: voicing (MIDI) and 808 root (MIDI).
CHORDS = {
    "Dm": ([62, 65, 69], 38),
    "Gm": ([62, 67, 70], 43),
    "Bb": ([62, 65, 70], 34),
    "A": ([61, 64, 69], 33),
    "C": ([60, 64, 67], 36),
}
HOOK_CHORDS = ["Dm", "Gm", "Bb", "A"] * 2
PROGRESSION = (  # one chord per bar, 44 bars
    ["Dm", "Gm", "Bb", "A"] + HOOK_CHORDS + ["Dm", "C", "Bb", "A"] * 2 + HOOK_CHORDS
    + ["Dm", "Dm", "Gm", "A"] + HOOK_CHORDS + ["Dm", "Dm", "Gm", "Dm"]
)
assert len(PROGRESSION) == MUSIC_BEATS // 4

# The hook: (beat within the 16-beat loop, length in beats, MIDI note). Each 4-beat
# group opens with the same "da-da-da DUM" cell; beats 7.5 and 15.5 are left
# free for the phrase fill.
HOOK = [
    (0, .5, 74), (.5, .25, 69), (.75, .25, 74), (1, .75, 77), (2, .5, 76), (2.5, .5, 74), (3, .5, 73), (3.5, .5, 74),  # Dm
    (4, .5, 70), (4.5, .25, 67), (4.75, .25, 70), (5, .75, 74), (6, .5, 72), (6.5, .5, 70), (7, .5, 69),               # Gm
    (8, .5, 74), (8.5, .25, 69), (8.75, .25, 74), (9, .75, 77), (10, .5, 79), (10.5, .5, 77), (11, .5, 74), (11.5, .5, 77),  # Bb
    (12, .5, 76), (12.5, .25, 73), (12.75, .25, 76), (13, .75, 79), (14, .5, 77), (14.5, .5, 76), (15, .5, 73),       # A
]
# Verse theremin calls, in the second half of each 8-beat phrase: (start, [(beat, note)...], length).
CALL_C = [(0, 67), (0.5, 76), (1.5, 72)]
CALL_A = [(0, 69), (0.5, 77), (1.5, 73)]


# ---------------------------------------------------------------- the mix

SPEC = SongSpec(
    id=ID, bpm=BPM, seed=SEED, music_beats=MUSIC_BEATS, duration=DURATION,
    bus_levels={"kick": 0.55, "drums": 1.4, "bass": 0.5, "organ": 1.6, "celesta": 1.0, "lead": 0.75,
                "theremin": 1.0, "pad": 1.0, "bell": 0.8, "perc": 1.8, "fx": 0.8},
    reverb_sends={"drums": 0.06, "organ": 0.18, "celesta": 0.35, "lead": 0.2, "theremin": 0.45,
                  "pad": 0.4, "bell": 0.5, "perc": 0.2, "fx": 0.25},
    delay_sends={"celesta": 0.22, "lead": 0.16, "perc": 0.15, "theremin": 0.12},
    ducked={"bass": 0.8, "organ": 0.4, "pad": 0.45, "celesta": 0.2},
    sidechain_depth=0.6, sidechain_tau=0.07,
    delay_beats=0.5, delay_feedback=0.35, delay_band=(400.0, 6000.0),
    reverb_seconds=2.8, reverb_rt60=2.3, reverb_band=(220.0, 6500.0), wet=0.3,
)


def beat_sample(beat: float) -> int:
    return SPEC.sample(beat)


def section_of(beat: float) -> tuple[str, int, int]:
    for name, a, b in SECTIONS:
        if a <= beat < b:
            return name, a, b
    return "end", MUSIC_BEATS, MUSIC_BEATS


def kick_beats() -> list[int]:
    """Every beat with a kick: from beat 0 except the stop on 15 and the break; 160 is the final hit."""
    return [b for b in range(161) if b != 15 and not 112 <= b < 128]


# ---------------------------------------------------------------- the kit

def make_shot(mix, name: str) -> np.ndarray:
    """This song's drums (stereo). Snare and crash come from the shared kit."""
    rng = mix.rng("shot", name)
    if name in ("snare", "crash"):
        return classic_shot(rng, name)
    if name == "kick":                       # short and punchy: the 808 carries the low end
        n = int(0.34 * SR)
        t = np.arange(n) / SR
        f = 54 + 150 * np.exp(-t / 0.022) + 70 * np.exp(-t / 0.003)
        body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.17)
        click = spectral_filter(rng.standard_normal(n), lo=2000, hi=8000) * np.exp(-t / 0.002) * 0.3
        x = np.tanh(2.0 * (body + click)) / np.tanh(2.0)
        x *= np.minimum(1, (n - np.arange(n)) / (0.02 * SR))
        return np.stack([x, x], axis=1)
    if name == "snap":                       # finger snap: a bright crack and a tiny tone
        n = int(0.12 * SR)
        t = np.arange(n) / SR
        noise = spectral_filter(rng.standard_normal((n, 2)), lo=1800, hi=6500, order=2)
        tone = np.sin(2 * np.pi * 2150 * t) * np.exp(-t / 0.006) * 0.4
        x = (noise * 0.9 + tone[:, None]) * np.exp(-t / 0.016)[:, None]
        return x * np.minimum(1, np.arange(n) / 12)[:, None]
    if name == "clap":                       # four quick bursts and a short tail
        n = int(0.28 * SR)
        t = np.arange(n) / SR
        noise = spectral_filter(rng.standard_normal((n, 2)), lo=1000, hi=4200, order=2)
        env = np.zeros(n)
        for k, d in enumerate((0.0, 0.008, 0.016, 0.026)):
            tt = np.maximum(t - d, 0)
            env += (t >= d) * np.exp(-tt / (0.005 if k < 3 else 0.08)) * (0.75 if k < 3 else 1.0)
        out = noise * env[:, None] * 0.9
        out[:, 0] = 0.85 * out[:, 0] + 0.15 * out[:, 1]
        return out * np.minimum(1, np.arange(n) / 10)[:, None]
    if name == "shaker":                     # soft attack, so it shuffles rather than ticks
        n = int(0.11 * SR)
        t = np.arange(n) / SR
        noise = spectral_filter(rng.standard_normal((n, 2)), lo=5000, hi=13000, order=3)
        att = np.sin(0.5 * np.pi * np.minimum(t / 0.008, 1))
        return noise * (att * np.exp(-np.maximum(t - 0.008, 0) / 0.028))[:, None]
    if name in ("tick", "openhat"):
        n = int((0.07 if name == "tick" else 0.32) * SR)
        t = np.arange(n) / SR
        noise = spectral_filter(rng.standard_normal((n, 2)), lo=8000, hi=16000, order=3)
        metal = sum(np.sign(np.sin(2 * np.pi * fr * t)) for fr in (3520, 4890, 6150, 8210)) * 0.035
        x = noise + metal[:, None] * 0.5
        out = x * np.exp(-t / (0.013 if name == "tick" else 0.085))[:, None]
        return out * np.minimum(1, np.arange(n) / 16)[:, None]
    raise KeyError(name)


# ---------------------------------------------------------------- instruments

def _pan2(p: float) -> np.ndarray:
    a = (p + 1) * np.pi / 4
    return np.array([np.cos(a), np.sin(a)]) * np.sqrt(2)


def bass808(mix, beat: float, length: float, note: int, gain: float, glide_to: int | None = None) -> None:
    """Sine 808 with a punchy pitch drop, saturated so it reads on small speakers; can glide.

    The note (release included) ends 50 ms before `beat + length`: an 808 ringing
    into a kick would smear the kick's attack, so no note crosses a kicked beat.
    """
    s = mix.sample(beat)
    gate = mix.seconds(length) - 0.08
    env = envelope(gate, 0.003, 0.4, 0.55, 0.03)
    if not mix.wants(s, len(env)):
        return
    n = len(env)
    t = np.arange(n) / SR
    semis = 7 * np.exp(-t / 0.011)                       # the punch
    if glide_to is not None:
        g0 = 0.55 * gate
        semis = semis + (glide_to - note) * np.clip((t - g0) / (gate - g0), 0, 1) ** 2
    f = midi_hz(note) * 2 ** (semis / 12)
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.sin(ph) + 0.18 * np.sin(2 * ph)
    x = np.tanh(2.4 * x) / np.tanh(2.4)
    mix.add("bass", s, x * env * gain)


def organ(mix, beat: float, notes, gain: float, length: float = 0.3, p: float = 0.0) -> None:
    """Haunted-house drawbar organ: a staccato stab with key click and a little chorus."""
    s = mix.sample(beat)
    env = envelope(mix.seconds(length), 0.003, 0.09, 0.6, 0.04)
    if not mix.wants(s, len(env)):
        return
    n = len(env)
    t = np.arange(n) / SR
    out = np.zeros((n, 2))
    for note in notes:
        f = midi_hz(note)
        for ch, cents in enumerate((-4, 4)):
            ff = f * 2 ** (cents / 1200)
            for h, a in ((1, 1.0), (2, 0.55), (3, 0.4), (4, 0.22), (6, 0.1), (8, 0.07)):
                out[:, ch] += a * np.sin(2 * np.pi * ff * h * t + 0.3 * h + ch)
    click = spectral_filter(mix.rng("click", beat).standard_normal(n), lo=1500, hi=7000) * np.exp(-t / 0.0015)
    out += 0.4 * click[:, None]
    mix.add("organ", s, out * (env * gain / len(notes))[:, None] * _pan2(p))


def celesta(mix, beat: float, length: float, note: int, gain: float, p: float = 0.0) -> None:
    """Music-box celesta: a softly FM-coloured tone with an inharmonic ping, damped at note-off."""
    s = mix.sample(beat)
    gate = mix.seconds(length)
    n = int((gate + 0.35) * SR)
    if not mix.wants(s, n):
        return
    t = np.arange(n) / SR
    f = midi_hz(note)
    index = 1.0 * np.exp(-t / 0.06) + 0.15
    x = np.sin(2 * np.pi * f * t + index * np.sin(2 * np.pi * f * t)) * np.exp(-t / 0.6)
    x += 0.22 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t / 0.22)
    x += 0.25 * np.sin(2 * np.pi * 4.07 * f * t) * np.exp(-t / 0.03)    # the tine's ping
    damp = np.exp(-np.maximum(t - gate, 0) / 0.08)
    x *= damp * np.minimum(1, np.arange(n) / 40) * np.minimum(1, (n - np.arange(n)) / 200)
    mix.add("celesta", s, x[:, None] * gain * _pan2(p))


def bones(mix, beat: float, note: int, gain: float, p: float = 0.0) -> None:
    """Xylophone 'bones': a hard, woody, quickly damped bar."""
    s = mix.sample(beat)
    n = int(0.3 * SR)
    if not mix.wants(s, n):
        return
    t = np.arange(n) / SR
    f = midi_hz(note)
    x = np.zeros(n)
    for ratio, amp, tau in ((1, 1.0, 0.11), (3.93, 0.45, 0.03), (10.1, 0.18, 0.01)):
        x += amp * np.sin(2 * np.pi * f * ratio * t) * np.exp(-t / tau)
    x += spectral_filter(mix.rng("bones", beat).standard_normal(n), lo=2500, hi=9000) * np.exp(-t / 0.0015) * 0.3
    x *= np.minimum(1, np.arange(n) / 8) * np.minimum(1, (n - np.arange(n)) / 300)
    mix.add("perc", s, x[:, None] * gain * _pan2(p))


def whistle(mix, beat: float, length: float, note: int, gain: float, prev: int | None = None,
            p: float = 0.0) -> None:
    """A whistled note: near-sine with breath, a slide up from the previous note, late vibrato."""
    s = mix.sample(beat)
    gate = mix.seconds(length) * 0.88
    env = envelope(gate, 0.012, 0.2, 0.85, 0.05)
    if not mix.wants(s, len(env)):
        return
    n = len(env)
    t = np.arange(n) / SR
    semis = np.zeros(n) if prev is None else (prev - note) * np.exp(-t / 0.014)
    vib = 0.10 * np.sin(2 * np.pi * 5.8 * t) * np.clip((t - 0.12) / 0.15, 0, 1)
    f = midi_hz(note) * 2 ** ((semis + vib) / 12)
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.sin(ph) + 0.07 * np.sin(2 * ph) + 0.03 * np.sin(3 * ph)
    rng = mix.rng("whistle", beat, note)
    f0 = midi_hz(note)
    breath = spectral_filter(rng.standard_normal(n), lo=f0 * 0.9, hi=f0 * 1.5, order=2) * 0.12
    chiff = spectral_filter(rng.standard_normal(n), lo=2500, hi=9000) * np.exp(-t / 0.012) * 0.12
    x = (x + breath + chiff) * env
    mix.add("lead", s, x[:, None] * gain * _pan2(p))


def theremin(mix, beat: float, points, length: float, gain: float, glide: float = 0.3) -> None:
    """A wavering theremin line through `points` [(beat offset, note)], gliding between them."""
    s = mix.sample(beat)
    gate = mix.seconds(length)
    env = envelope(gate, 0.09, 0.4, 0.9, 0.35)
    if not mix.wants(s, len(env)):
        return
    n = len(env)
    t = np.arange(n) / SR
    tb = t * BPM / 60                                     # beats since the start
    xs, ys = [points[0][0]], [points[0][1]]
    for (_, prev), (b, note) in zip(points, points[1:]):
        xs += [b, b + glide]
        ys += [prev, note]
    semis = np.interp(tb, xs, ys)
    vib = 0.28 * np.sin(2 * np.pi * 5.2 * t) * np.clip((t - 0.05) / 0.3, 0, 1)
    f = 440.0 * 2 ** ((semis + vib - 69) / 12)
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = (np.sin(ph) + 0.16 * np.sin(2 * ph) + 0.06 * np.sin(3 * ph)) * env
    mix.add("theremin", s, x[:, None] * gain * _pan2(0.15))


def ghost_pad(mix, beat: float, length: float, notes, gain: float, attack: float = 0.6,
              release: float = 1.2) -> None:
    """An 'ooh' of slightly detuned hollow voices with slow vibrato, spread wide."""
    s = mix.sample(beat)
    env = envelope(mix.seconds(length), attack, 1.0, 0.9, release)
    if not mix.wants(s, len(env)):
        return
    n = len(env)
    t = np.arange(n) / SR
    rng = mix.rng("ghost", beat, tuple(notes))
    out = np.zeros((n, 2))
    for note in notes:
        for ch, cents in enumerate((-9, 9)):
            rate, ph0 = 4.2 + rng.random(), 2 * np.pi * rng.random()
            vib = 0.08 * np.sin(2 * np.pi * rate * t + ph0)
            f = midi_hz(note + cents / 100 + vib)
            ph = 2 * np.pi * np.cumsum(f) / SR + 2 * np.pi * rng.random()
            out[:, ch] += np.sin(ph) + 0.12 * np.sin(3 * ph) + 0.04 * np.sin(5 * ph)
    trem = 1 + 0.12 * np.sin(2 * np.pi * 0.4 * t)
    breath = spectral_filter(rng.standard_normal((n, 2)), lo=700, hi=2600) * 0.06
    out = (out / len(notes) + breath) * (env * trem)[:, None]
    mix.add("pad", s, out * gain)


def toll(mix, beat: float, note: int, gain: float, damp_at: float | None = None) -> None:
    """A tolling church bell: hum, prime, a minor-third tierce, quint, nominal and up.

    `damp_at` stops it 20 ms before that beat (its low hum would blur a kick there).
    """
    s = mix.sample(beat)
    n = int(4.0 * SR)
    if damp_at is not None:
        n = min(n, mix.sample(damp_at) - s - int(0.02 * SR))
    if not mix.wants(s, n):
        return
    t = np.arange(n) / SR
    f = midi_hz(note)
    x = np.zeros(n)
    for ratio, amp, tau in ((0.5, 0.5, 3.0), (1.0, 1.0, 2.2), (1.2, 0.55, 1.8), (1.5, 0.3, 1.4),
                            (2.0, 0.6, 1.2), (2.5, 0.22, 0.7), (3.0, 0.18, 0.5), (4.2, 0.1, 0.25)):
        x += amp * np.sin(2 * np.pi * f * ratio * t + ratio) * np.exp(-t / tau)
    x *= np.minimum(1, np.arange(n) / 30) * np.minimum(1, (n - np.arange(n)) / (0.3 * SR))
    mix.add("bell", s, x * gain / 2)


def swell(mix, end_beat: float, beats: float, gain: float) -> None:
    """A reversed-cymbal swell that cuts off on `end_beat`."""
    n = mix.sample(beats)
    s = mix.sample(end_beat) - n
    if not mix.wants(s, n):
        return
    t = np.arange(n) / n
    noise = spectral_filter(mix.rng("swell", end_beat).standard_normal((n, 2)), lo=2500, hi=12000)
    out = noise * (t ** 3 * np.minimum(1, (n - np.arange(n)) / 64))[:, None]
    mix.add("fx", s, out * gain)


# ---------------------------------------------------------------- the arrangement

def melody(mix, start: float, notes, fn, gain: float, transpose: int = 0, stretch: float = 1.0,
           until: float = 1e9, **kw) -> None:
    """Play [(beat, length, note)] from `start` (lengths and beats scaled by `stretch`)."""
    prev, prev_end = None, None
    for beat, length, note in notes:
        if beat >= until:
            break
        b = start + beat * stretch
        legato = prev_end is not None and abs(prev_end - b) < 1e-6
        if fn is whistle:
            whistle(mix, b, length * stretch, note + transpose, gain, prev + transpose if legato and prev else None, **kw)
        else:
            fn(mix, b, length * stretch, note + transpose, gain, **kw)
        prev, prev_end = note, b + length * stretch


def fill(mix, boundary: int, big: bool) -> None:
    """Mark a phrase boundary: two bone taps and a swell; before a section, a riser and snare roll too."""
    if big:
        if boundary != 16:                                # the intro has the count-in instead
            riser(mix, boundary - 4, 4, 0.26)
            for i in range(16):
                hit(mix, "drums", "snare", boundary - 4 + i / 4, 0.04 + 0.26 * (i / 15) ** 2)
        else:
            riser(mix, 12, 4, 0.2)
        swell(mix, boundary, 2, 0.16)
    else:
        swell(mix, boundary, 1, 0.07 if boundary % 16 else 0.1)
        if boundary % 16 == 0:
            hit(mix, "drums", "clap", boundary - 0.25, 0.22, -0.3)
    bones(mix, boundary - 0.5, 81, 0.16, 0.4)
    bones(mix, boundary - 0.25, 76, 0.13, -0.4)


def arrange(mix) -> None:
    final = lambda b: 128 <= b < 160  # noqa: E731

    # Drums: kick, crashes, count-in, backbeat.
    for b in kick_beats():
        if b == 160:
            g = 1.15
        elif b < 16:
            g = 0.62 + 0.33 * min(b, 8) / 8
        else:
            g = 0.85 if section_of(b)[0] == "verse" else 1.0
        hit(mix, "kick", "kick", b, g)
    for b in CRASHES:
        hit(mix, "drums", "crash", b, 0.36 if b in (16, 128, 160) else 0.28)
    for b in COUNT_IN:                                     # count-in: 3, 2, 1
        hit(mix, "drums", "snare", b, 0.5 + 0.08 * (b - 12))
    for b in range(4, 161):
        name, a, _ = section_of(b)
        if name == "intro" and b < 15 and b % 4 in (1, 3):
            hit(mix, "drums", "snap", b, 0.3, 0.2)
        elif name == "hook" and b % 4 in (1, 3):
            hit(mix, "drums", "clap", b, 0.55 if final(b) else 0.5)
            hit(mix, "drums", "snap", b, 0.3, 0.25)
        elif name == "verse" and b % 4 in (1, 3):
            hit(mix, "drums", "snap", b, 0.45, 0.2)
        elif name == "break":
            if b % 4 == 2:                                 # half time: the backbeat on 3
                hit(mix, "drums", "clap", b, 0.55)
                hit(mix, "drums", "snap", b, 0.3, 0.25)
            elif b % 2 == 1:
                hit(mix, "drums", "snap", b, 0.22, -0.3)
    hit(mix, "drums", "clap", 160, 0.6)
    hit(mix, "drums", "snap", 160, 0.35)
    for i in range(4 * 160):                               # the 16th grid, lightly shuffled
        b0, sub = i // 4, i % 4
        b = b0 + sub / 4 + (SWING if sub % 2 else 0)
        name, a, _ = section_of(b0)
        if name == "intro" and 8 <= b0 < 15:
            hit(mix, "drums", "shaker", b, (0.08, 0.04, 0.1, 0.05)[sub], -0.3)
        elif name == "hook":
            hit(mix, "drums", "shaker", b, (0.1, 0.06, 0.13, 0.07)[sub] * (1.15 if final(b0) else 1), -0.35)
            if sub == 2:
                if a == 16:
                    hit(mix, "drums", "tick", b, 0.16, 0.35)
                else:
                    hit(mix, "drums", "openhat", b, 0.12 if final(b0) else 0.1, 0.35)
        elif name == "verse":
            if sub == 2:
                hit(mix, "drums", "tick", b, 0.13, 0.35)
            elif sub == 3:
                hit(mix, "drums", "tick", b, 0.05, 0.35)
        elif name == "break" and sub in (0, 2):
            hit(mix, "drums", "shaker", b, 0.05, -0.3)

    # Phrase marks.
    for boundary in range(16, 161, 8):
        fill(mix, boundary, boundary in SECTION_STARTS)

    # Bass, chords and colour, bar by bar.
    for bar, chord in enumerate(PROGRESSION):
        b0 = 4 * bar
        name, a, _ = section_of(b0)
        tones, root = CHORDS[chord]
        nxt = CHORDS[PROGRESSION[bar + 1]][1] if bar + 1 < len(PROGRESSION) else root
        if name == "intro":
            if bar < 2:
                ghost_pad(mix, b0, 4, tones, 0.1 + 0.05 * bar, attack=1.0)
            else:
                bass808(mix, b0, 1, root, 0.5)
                if bar == 2:
                    bass808(mix, b0 + 2, 1, root, 0.42)
                else:
                    bass808(mix, b0 + 2.5, 0.5, root + 12, 0.42)
                for k in range(4 if bar == 2 else 3):      # bar 3 stops before beat 15
                    organ(mix, b0 + k + 0.5, tones, 0.1 + 0.03 * bar, p=0.2)
        elif name == "hook":
            big = final(b0)
            pattern = [(0, 1, root, None), (1.5, 0.5, root + 12, None), (2.5, 0.5, root, None),
                       (3, 0.5, root, None), (3.5, 0.5, root + 7, nxt)]
            for k, length, note, glide in pattern:
                bass808(mix, b0 + k, length, note, 0.55, glide)
            for k in range(4):                             # the organ skank on every offbeat
                organ(mix, b0 + k + 0.5, tones, 0.2 if a == 16 else 0.22, p=0.2)
            if a != 16:                                    # bones on the 'a' of 1 and 3
                for k in (0, 2):
                    bones(mix, b0 + k + 0.75 + SWING, tones[(bar + k) % 3] + 12, 0.09, (-0.5, 0.5)[k // 2])
            if big:
                ghost_pad(mix, b0, 4, tones, 0.1, attack=0.3, release=0.6)
        elif name == "verse":
            bass808(mix, b0, 1, root, 0.5)
            bass808(mix, b0 + 2.5, 0.5, root, 0.42)
            four = list(tones) + [tones[0] + 12]
            for k in range(8):                             # the music-box arpeggio
                celesta(mix, b0 + k / 2, 0.5, four[(0, 2, 1, 3, 2, 1, 3, 2)[k]] + 12, 0.11, (-0.4, 0.4)[k % 2])
        elif name == "break":
            bass808(mix, b0, 4, root, 0.45)
            ghost_pad(mix, b0, 4, tones, 0.42 if bar < 31 else 0.36, attack=0.8, release=1.0)
            toll(mix, b0, (62, 62, 67, 61)[bar - 28], 0.24, damp_at=128)
        elif name == "outro":
            if bar == 40:                                  # the final hit
                bass808(mix, b0, 3.5, root, 0.62)
                organ(mix, b0, tones + [tones[0] + 12], 0.34, length=1.0)
                toll(mix, b0, 62, 0.3)
                ghost_pad(mix, b0, 8, tones, 0.3, attack=0.05, release=2.0)
                for k, note in enumerate((74, 77, 81, 86)):
                    celesta(mix, b0 + k * 0.25, 1.5, note, 0.1, (-0.4, 0.4)[k % 2])
            elif bar == 42:
                ghost_pad(mix, b0, 4, tones, 0.24, attack=0.6, release=1.2)
            elif bar == 43:
                bass808(mix, b0, 3, root, 0.35)
                ghost_pad(mix, b0, 4, tones, 0.22, attack=0.4, release=1.8)
                toll(mix, b0, 50, 0.32)
                for k, note in enumerate((74, 77, 81, 86)):
                    celesta(mix, b0 + k * 0.5, 3 - k * 0.5, note, 0.12 - 0.015 * k, (-0.4, 0.4)[k % 2])

    # Melody.
    melody(mix, 0, HOOK, celesta, 0.2, transpose=12)       # the intro's music box (C#6 pickup on 15)
    for start in (16, 32, 80, 96, 128, 144):
        melody(mix, start, HOOK, whistle, 0.3)
        if start >= 80:
            melody(mix, start, HOOK, celesta, 0.09 if start < 128 else 0.1, transpose=12, p=0.3)
        if start >= 128:                                   # the final hook: the lead doubled an octave up
            melody(mix, start, HOOK, whistle, 0.11, transpose=12, p=-0.3)
    for phrase in range(48, 80, 8):                         # verse calls
        call = CALL_C if (phrase // 8) % 2 == 0 else CALL_A
        theremin(mix, phrase + 4.5, call, 2.5, 0.14)
    melody(mix, 112, HOOK, celesta, 0.17, transpose=12, stretch=2, until=8)   # the hook at half speed
    theremin(mix, 112, [(0, 62), (3, 65), (4, 64), (6, 62)], 7.5, 0.1)
    theremin(mix, 120, [(0, 67), (3, 70), (4, 69), (6, 73)], 7.5, 0.1)
    melody(mix, 164, HOOK, celesta, 0.13, transpose=12, stretch=2, until=4)   # winding down
    theremin(mix, 166, [(0, 74), (1.5, 77), (3, 74), (5, 62)], 6, 0.12, glide=1.2)
