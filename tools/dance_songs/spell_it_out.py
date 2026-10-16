"""Spell It Out: Raccoon Dance's letters song, a disco floor-filler (docs/DANCE.md section 6c).

Original, in the *cadence* of late-70s party disco: four-on-the-floor at 127
BPM, an octave-bouncing bass, claps on 2 and 4, an open hat on every off-beat,
string swoops and brass stabs. The chorus leaves the room the crowd needs to
spell: four tutti brass-and-string stabs, one every two beats, each a step
higher than the last, and nothing else on top. That is where the dance puts its
four letters (Y, M, C, A). Melody, harmony, bass line and arrangement are our
own; nothing is sampled or transcribed, and there are no lyrics. numpy
synthesis only (the engine is tools/dance_synth.py).

127 BPM, 4/4, B-flat major. Beat 0 is the very first sample (offset 0.0).
52 bars = 208 beats = 98.268 s of music, then a ring-out: 100.5 s in all.

Arrangement (beats; one chord per bar; the routine repeats on 8-beat phrases):

    intro    0-15    Bb Gm Eb F    kick from beat 0; strings swell in; off-beat hats
                                   at bar 1, octave bass and claps at bar 2; snare
                                   count-in on 12, 13, 14 (the game's "3, 2, 1")
                                   under a riser and a rising string swoop
    verse 1  16-47   Bb Dm Eb F    four-on-the-floor, claps on 2 and 4, off-beat open
                     Bb Dm Cm F    hats, the octave bass, clavinet chicks, soft strings
                                   and a synth melody; a swoop and a snare roll into
                                   the chorus
    chorus 1 48-79   Eb F Dm Gm    the letters: four rising stabs on 48, 50, 52, 54
                     Eb F Cm F     (and again on 64-70), then the brass hook answers
                                   over the next two bars; 16th hats, bright strings
    verse 2  80-111  as verse 1    busier: a tambourine, a brass answer every 8 bars
    chorus 2 112-143 as chorus 1
    bridge   144-159 Gm Eb Bb F    the kick drops: claps, held bass, a string melody
                                   in half time; riser, swoop and roll into the final
    final    160-191 as chorus     the biggest: everything, the hook an octave up
    outro    192-207 Eb F Bb Bb    a last set of letters (192-198), the final hit on
                                   200, then the chord held for the bow and rung out

Mixing: bass, strings and keys duck under the kick (sidechain); the brass and
lead feed a ping-pong delay; a seeded convolution reverb; the shared limiter
sets peaks near -1 dBFS. Deterministic: every noise source is seeded from
(SEED, event), so a render is byte-identical run to run.

    uv run python tools/generate_dance_song.py --song spell_it_out
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from dance_synth import (SR, SongSpec, bass_note, classic_shot, envelope, hit, lead_note, midi_hz, osc,
                         pad_chord, pluck, riser, spectral_filter, wavetable)

ID = "spell_it_out"
TITLE = "Spell It Out"
OUT = Path(__file__).resolve().parents[2] / "assets" / "dance" / "songs" / ID / "audio.m4a"

SEED = 20261016
BPM = 127
OFFSET = 0.0                  # seconds into the audio of beat 0
MUSIC_BEATS = 208             # 52 bars
DURATION = 100.5              # the file's exact length: 98.268 s of music + ring-out
TAIL = DURATION - MUSIC_BEATS * 60 / BPM

SECTIONS = [  # name, first beat, last beat + 1  (docs/DANCE.md section 6c)
    ("intro", 0, 16), ("verse", 16, 48), ("chorus", 48, 80), ("verse", 80, 112),
    ("chorus", 112, 144), ("bridge", 144, 160), ("chorus", 160, 192), ("outro", 192, 208),
]
CRASHES = (48, 80, 112, 144, 160, 176, 192, 200)
COUNT_IN = (12, 13, 14)
RISERS = (12, 44, 108, 156)          # the bar starting here builds into the next section
# The letters: four stabs, one every two beats, from each of these beats (the chart's
# `letters` move: Y, M, C, A on +0, +2, +4, +6).
LETTERS = (48, 64, 112, 128, 160, 176, 192)
LETTER_BARS = frozenset(b // 4 for start in LETTERS for b in (start, start + 4))

# Chords: string voicing (MIDI) and bass root (MIDI).
CHORDS = {
    "Bb": ([58, 62, 65, 70], 34),
    "Gm": ([58, 62, 67, 70], 31),
    "Eb": ([58, 63, 67, 70], 39),
    "F": ([57, 60, 65, 69], 41),
    "Dm": ([57, 62, 65, 69], 38),
    "Cm": ([55, 60, 63, 67], 36),
}
VERSE_CHORDS = ["Bb", "Dm", "Eb", "F", "Bb", "Dm", "Cm", "F"]
CHORUS_CHORDS = ["Eb", "F", "Dm", "Gm", "Eb", "F", "Cm", "F"]
PROGRESSION = (  # one chord per bar, 52 bars
    ["Bb", "Gm", "Eb", "F"] + VERSE_CHORDS + CHORUS_CHORDS + VERSE_CHORDS + CHORUS_CHORDS
    + ["Gm", "Eb", "Bb", "F"] + CHORUS_CHORDS + ["Eb", "F", "Bb", "Bb"]
)
assert len(PROGRESSION) == MUSIC_BEATS // 4

# The letter stabs (beat offset, voicing): over Eb, then F, each top note a step higher.
STABS = [(0, [63, 67, 70]), (2, [63, 67, 72]), (4, [65, 69, 74]), (6, [69, 72, 77])]
# The brass hook that answers the letters: (beat within the 8-beat answer, length, MIDI note).
HOOK_A = [  # over Dm Gm
    (0, .5, 69), (.5, .5, 69), (1, .5, 65), (1.5, 1, 69), (2.5, .5, 72), (3, 1, 74),
    (4, .5, 74), (4.5, .5, 72), (5, .5, 70), (5.5, 1, 67), (6.5, .5, 69), (7, 1, 70),
]
HOOK_B = [  # over Cm F, turning home to Bb
    (0, .5, 67), (.5, .5, 67), (1, .5, 63), (1.5, 1, 67), (2.5, .5, 70), (3, 1, 72),
    (4, .5, 72), (4.5, .5, 70), (5, .5, 69), (5.5, 1.5, 65), (7, .5, 69), (7.5, .5, 70),
]
# The verse melody, over the 8-bar verse: (beat within the verse, length, note).
VERSE_TUNE = [
    (0, .5, 70), (.5, .5, 72), (1, 1, 74), (2, .5, 72), (2.5, 1, 70), (4, .5, 69), (4.5, .5, 70), (5, 1.5, 74),
    (8, .5, 70), (8.5, .5, 72), (9, 1, 75), (10, .5, 74), (10.5, 1, 72), (12, .5, 72), (12.5, .5, 74), (13, 1.5, 77),
    (16, .5, 70), (16.5, .5, 72), (17, 1, 74), (18, .5, 72), (18.5, 1, 70), (20, .5, 69), (20.5, .5, 70), (21, 1.5, 74),
    (24, .5, 67), (24.5, .5, 70), (25, 1, 72), (26, .5, 70), (26.5, 1, 67), (28, .5, 69), (28.5, .5, 72), (29, 2, 77),
]
# Verse 2's brass answer, in the last bar of each 4-bar phrase.
ANSWER = [(0, .5, 77), (.5, .5, 74), (1, .5, 77), (1.5, 1, 79)]
# The bridge's string melody in half time, over Gm Eb Bb F: (beat from 144, length, note).
BRIDGE_TUNE = [(0, 2, 74), (2, 2, 70), (4, 2, 67), (6, 2, 70), (8, 2, 74), (10, 2, 77), (12, 2, 76), (14, 2, 72)]


# ---------------------------------------------------------------- the mix

SPEC = SongSpec(
    id=ID, bpm=BPM, seed=SEED, music_beats=MUSIC_BEATS, duration=DURATION,
    bus_levels={"kick": 0.55, "drums": 1.15, "perc": 1.2, "bass": 0.55, "pad": 0.75, "brass": 1.0,
                "keys": 1.1, "lead": 0.9, "fx": 0.85},
    reverb_sends={"drums": 0.07, "perc": 0.12, "pad": 0.3, "brass": 0.28, "keys": 0.15, "lead": 0.22, "fx": 0.25},
    delay_sends={"brass": 0.14, "lead": 0.24, "keys": 0.1},
    ducked={"bass": 1.0, "pad": 0.6, "keys": 0.45},
    sidechain_depth=0.6, sidechain_tau=0.08,
    delay_beats=0.75, delay_feedback=0.35, delay_band=(350.0, 5000.0),
    reverb_seconds=2.4, reverb_rt60=1.9, reverb_band=(240.0, 6000.0), wet=0.3,
    peak_dbfs=-2.3,                   # the brass stabs make AAC overshoot ~1.4 dB: decoded peaks land near -1 dBFS
)


def beat_sample(beat: float) -> int:
    return SPEC.sample(beat)


def section_of(beat: float) -> tuple[str, int, int]:
    for name, a, b in SECTIONS:
        if a <= beat < b:
            return name, a, b
    return "end", MUSIC_BEATS, MUSIC_BEATS


def kick_beats() -> list[int]:
    """Every beat with a kick: four-on-the-floor except the bridge; 200 is the final hit."""
    return [b for b in range(201) if not 144 <= b < 160]


# ---------------------------------------------------------------- the kit

def make_shot(mix, name: str) -> np.ndarray:
    """The shared electro kit with a shorter kick, plus a tambourine."""
    rng = mix.rng("shot", name)
    if name == "kick":                       # the shared kick, cut to 0.36 s: at 127 BPM a beat is only
        n = int(0.36 * SR)                   # 0.472 s, and a longer tail would ring into the next kick
        t = np.arange(n) / SR
        f = 48 + 112 * np.exp(-t / 0.028) + 40 * np.exp(-t / 0.004)
        body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.2)
        click = spectral_filter(rng.standard_normal(n), lo=1500, hi=6000) * np.exp(-t / 0.0025) * 0.25
        x = np.tanh(1.6 * (body + click)) / np.tanh(1.6)
        x *= np.minimum(1, (n - np.arange(n)) / (0.05 * SR))
        return np.stack([x, x], axis=1)
    if name == "tamb":                       # jingles: bright noise with a few ringing partials
        n = int(0.16 * SR)
        t = np.arange(n) / SR
        noise = spectral_filter(rng.standard_normal((n, 2)), lo=6000, hi=14000, order=2)
        ring = sum(np.sin(2 * np.pi * fr * t + k) for k, fr in enumerate((5300, 6850, 8900))) * 0.12
        x = noise + ring[:, None]
        att = np.sin(0.5 * np.pi * np.minimum(t / 0.004, 1))
        return x * (att * np.exp(-np.maximum(t - 0.004, 0) / 0.045))[:, None]
    return classic_shot(rng, name)


# ---------------------------------------------------------------- instruments

def _pan2(p: float) -> np.ndarray:
    a = (p + 1) * np.pi / 4
    return np.array([np.cos(a), np.sin(a)]) * np.sqrt(2)


def strings(mix, beat: float, length: float, chord, gain: float, bright: float = 2400.0,
            attack: float = 0.25, release: float = 0.7, transpose: int = 0) -> None:
    """The string section: the shared detuned saw pad, a little brighter as it opens."""
    notes = CHORDS[chord][0] if isinstance(chord, str) else chord
    pad_chord(mix, beat, length, notes, gain, (bright * 0.7, bright), attack=attack, release=release,
              transpose=transpose, key=str(chord))


def brass(mix, beat: float, length: float, notes, gain: float, p: float = 0.0, bus: str = "brass") -> None:
    """A synth brass section: stacked saws that blare open and settle, with a scoop up to pitch."""
    s = mix.sample(beat)
    gate = mix.seconds(length) * 0.9
    env = envelope(gate, 0.012, 0.16, 0.7, 0.07)
    if not mix.wants(s, len(env)):
        return
    n = len(env)
    t = np.arange(n) / SR
    open_ = np.exp(-t / 0.09)                            # the filter blares, then closes down
    scoop = 2 ** (-0.35 * np.exp(-t / 0.025) / 12)       # a third of a semitone flat, rising
    out = np.zeros((n, 2))
    for note in notes:
        f = midi_hz(note)
        for ch, cents in enumerate((-7, 7)):
            ff = f * 2 ** (cents / 1200)
            bright = osc(wavetable("saw", ff, 5200), ff * scoop, n, 0.13 * ch)
            dark = osc(wavetable("saw", ff, 1300), ff * scoop, n, 0.13 * ch)
            out[:, ch] += bright * open_ + dark * (1 - open_)
    out *= (env * gain / len(notes))[:, None]
    mix.add(bus, s, out * _pan2(p))


def swoop(mix, beat: float, beats: float, lo: int, hi: int, gain: float) -> None:
    """A disco string run: an octave-wide glissando that swells into the downbeat."""
    s = mix.sample(beat)
    n = mix.sample(beats)
    if not mix.wants(s, n):
        return
    t = np.arange(n) / n
    semis = lo + (hi - lo) * t ** 1.6
    out = np.zeros((n, 2))
    for ch, cents in enumerate((-9, 9)):
        f = 440.0 * 2 ** ((semis + cents / 100 - 69) / 12)
        for octave in (0, 12):
            ff = f * 2 ** (octave / 12)
            out[:, ch] += osc(wavetable("saw", 440.0 * 2 ** ((hi - 69) / 12), 3000), ff, n, 0.2 * ch + octave)
    out *= (t ** 1.8 * np.minimum(1, (n - np.arange(n)) / 300))[:, None]
    mix.add("pad", s, out * gain / 2)


def clav(mix, beat: float, notes, gain: float, p: float) -> None:
    """A clavinet 'chick' on the off-beat: short square plucks."""
    for k, note in enumerate(notes):
        pluck(mix, beat, note, gain / len(notes) * (1.0 - 0.15 * k), p, length=0.22, bus="keys")


def melody(mix, start: float, notes, fn, gain: float, transpose: int = 0, stretch: float = 1.0, **kw) -> None:
    """Play [(beat, length, note)] from `start`."""
    for beat, length, note in notes:
        fn(mix, start + beat * stretch, length * stretch, note + transpose, gain, **kw)


def _brass_note(mix, beat, length, note, gain, **kw):
    brass(mix, beat, length, [note, note - 12], gain, **kw)


# ---------------------------------------------------------------- the arrangement

def arrange(mix) -> None:
    final = lambda b: 160 <= b < 192  # noqa: E731

    # Drums: kick, crashes, count-in, claps on 2 and 4.
    for b in kick_beats():
        if b == 200:
            g = 1.15
        elif b < 16:
            g = 0.7 + 0.25 * min(b, 8) / 8
        else:
            g = 1.0
        hit(mix, "kick", "kick", b, g)
    for b in CRASHES:
        hit(mix, "drums", "crash", b, 0.3 if b in (80, 144) else 0.4)
    for b in COUNT_IN:                                       # count-in: 3, 2, 1
        hit(mix, "drums", "snare", b, 0.3 + 0.06 * (b - 12))
    for b in range(8, 200):                                  # from bar 2 on
        name, a, _ = section_of(b)
        if b % 2 == 1:
            hit(mix, "drums", "clap", b, 0.5 if name in ("intro", "bridge") else 0.6)
            if name == "chorus":
                hit(mix, "drums", "snare", b, 0.3 if final(b) else 0.24)
    hit(mix, "drums", "clap", 200, 0.65)
    hit(mix, "drums", "snare", 200, 0.4)

    # Hats: the disco off-beat open hat, closed 16ths around it from the verses on.
    for i in range(4 * 200):
        b = i / 4
        name, a, _ = section_of(b)
        sub = i % 4
        if name == "intro":
            if b >= 4 and sub == 2:
                hit(mix, "drums", "openhat", b, 0.12 + 0.05 * (b >= 8), 0.25)
        elif name in ("verse", "chorus", "outro"):
            if sub == 2:
                hit(mix, "drums", "openhat", b, 0.2 if name == "chorus" else 0.17, 0.25)
            elif name == "chorus" or (name == "verse" and a == 80) or sub == 0:
                hit(mix, "drums", "hat", b, (0.1, 0.06, 0, 0.07)[sub], -0.3)
        elif name == "bridge" and sub == 2:
            hit(mix, "drums", "hat", b, 0.08, 0.35)
    for i in range(4 * 80, 4 * 112):                         # verse 2's tambourine, on the 16ths
        if i % 2:
            hit(mix, "perc", "tamb", i / 4, 0.12 if i % 4 == 3 else 0.07, 0.45)
    for i in range(4 * 160, 4 * 192):                        # and the final chorus
        if i % 2:
            hit(mix, "perc", "tamb", i / 4, 0.13 if i % 4 == 3 else 0.08, 0.45)

    # Builds into each section: a riser, a snare roll and a string swoop.
    for b in RISERS:
        riser(mix, b, 4, 0.2 if b == 12 else 0.3)
        swoop(mix, b + 2, 2, 58, 82, 0.1 if b == 12 else 0.13)
        if b != 12:
            for i in range(16):
                hit(mix, "drums", "snare", b + i / 4, 0.05 + 0.28 * (i / 15) ** 2)

    # Harmony and bass, bar by bar.
    for bar, chord in enumerate(PROGRESSION):
        b0 = 4 * bar
        name, a, _ = section_of(b0)
        voicing, root = CHORDS[chord]
        tones = sorted(voicing)
        if name == "intro":
            strings(mix, b0, 4, chord, 0.2 + 0.05 * bar, bright=900 + 500 * bar, attack=0.6 if bar == 0 else 0.2)
            if bar >= 2:
                for k in range(8):                         # the octave bounce comes in
                    bass_note(mix, b0 + k / 2, 0.5, root + (12 if k % 2 else 0), 0.4, bright=1100)
        elif name == "verse":
            strings(mix, b0, 4, chord, 0.17, bright=1800)
            for k in range(8):
                bass_note(mix, b0 + k / 2, 0.5, root + (12 if k % 2 else 0), 0.46, bright=1200)
            for k in range(4):                             # clav chicks on the off-beats
                clav(mix, b0 + k + 0.5, [tones[1] + 12, tones[3] + 12], 0.16, (-0.45, 0.45)[k % 2])
            if a == 80:
                clav(mix, b0 + 3.75, [tones[2] + 12], 0.08, 0.3)
        elif name == "chorus":
            big = final(b0)
            for k in range(8):
                bass_note(mix, b0 + k / 2, 0.5, root + (12 if k % 2 else 0), 0.48, bright=1500)
            if bar in LETTER_BARS:                         # the letters get the room: stabs only (below)
                continue
            strings(mix, b0, 4, chord, 0.3 if big else 0.25, bright=3400 if big else 2900, attack=0.05, release=0.35)
            if big:
                strings(mix, b0, 4, chord, 0.1, bright=4500, attack=0.05, release=0.35, transpose=12)
            for k in range(4):
                clav(mix, b0 + k + 0.5, [tones[1] + 12, tones[3] + 12], 0.12, (-0.5, 0.5)[k % 2])
        elif name == "bridge":
            strings(mix, b0, 4, chord, 0.55, bright=1600, attack=0.5, release=1.0)
            bass_note(mix, b0, 2, root, 0.46, bright=600, legato=1.0)
            bass_note(mix, b0 + 2, 2, root + 12, 0.36, bright=600, legato=1.0)
        elif name == "outro":
            if bar == 48 or bar == 49:                      # the last letters, over Eb and F
                for k in range(8):
                    bass_note(mix, b0 + k / 2, 0.5, root + (12 if k % 2 else 0), 0.48, bright=1500)
            elif bar == 50:                                  # the final hit, held for the bow
                strings(mix, b0, 8, chord, 0.5, bright=3000, attack=0.01, release=2.2)
                strings(mix, b0, 8, chord, 0.16, bright=4000, attack=0.01, release=2.2, transpose=12)
                bass_note(mix, b0, 6, root, 0.55, bright=700, legato=1.0)
                brass(mix, b0, 1.5, [70, 74, 77, 82], 0.5)

    # The letters: four rising tutti stabs, two beats apart.
    for start in LETTERS:
        big = final(start) or start == 192
        for off, voicing in STABS:
            brass(mix, start + off, 0.8, voicing + [voicing[-1] - 12], 0.8 if big else 0.72)
            strings(mix, start + off, 0.9, voicing, 0.34, bright=3600, attack=0.005, release=0.3)
            strings(mix, start + off, 0.6, [v + 12 for v in voicing], 0.16, bright=4200, attack=0.005, release=0.25)
            hit(mix, "drums", "snare", start + off, 0.26 if big else 0.2)

    # The brass hook answering the letters, over the next two bars.
    for c in (48, 112, 160):
        for half, tune in ((8, HOOK_A), (24, HOOK_B)):
            melody(mix, c + half, tune, _brass_note, 0.34)
            if final(c):
                melody(mix, c + half, tune, lead_note, 0.12, transpose=12)

    # The verse melody, and verse 2's brass answers.
    for v in (16, 80):
        melody(mix, v, VERSE_TUNE, lead_note, 0.24 if v == 16 else 0.27)
    for phrase in (92, 108):
        melody(mix, phrase, ANSWER, _brass_note, 0.22)

    # The bridge's string melody, in half time.
    for beat, length, note in BRIDGE_TUNE:
        strings(mix, 144 + beat, length, [note, note + 12], 0.22, bright=2600, attack=0.15, release=0.6)
