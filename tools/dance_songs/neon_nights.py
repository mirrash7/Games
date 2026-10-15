"""Neon Nights: Raccoon Dance's first song (docs/DANCE.md section 6).

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
song (`dance_synth.render(song, start_beat, end_beat)`; tests use that).

    uv run python tools/generate_dance_song.py --song neon_nights   # -> assets/dance/songs/neon_nights/audio.m4a
"""

from __future__ import annotations

from pathlib import Path

from dance_synth import SR, SongSpec, bass_note, bell, hit, lead_note, pluck, riser
from dance_synth import pad_chord as _pad_chord

ID = "neon_nights"
TITLE = "Neon Nights"
OUT = Path(__file__).resolve().parents[2] / "assets" / "dance" / "songs" / ID / "audio.m4a"

SEED = 20261014
BPM = 120
SPB = SR * 60 // BPM          # samples per beat: 22050, exact
OFFSET = 0.0                  # seconds into the audio of beat 0
MUSIC_BEATS = 208             # 52 bars
TAIL = 2.0                    # seconds of ring-out after the music
DURATION = MUSIC_BEATS * 60 / BPM + TAIL  # 106.0
N_TOTAL = int(round(DURATION * SR))       # 4_674_600

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

SPEC = SongSpec(
    id=ID, bpm=BPM, seed=SEED, music_beats=MUSIC_BEATS, duration=DURATION,
    bus_levels={"kick": 0.5, "drums": 1.0, "bass": 0.55, "pad": 0.8, "arp": 1.2,
                "lead": 1.0, "stab": 1.2, "bell": 0.6, "fx": 0.9},
    reverb_sends={"drums": 0.08, "pad": 0.22, "arp": 0.25, "lead": 0.22, "stab": 0.4, "bell": 0.45, "fx": 0.2},
    delay_sends={"arp": 0.25, "lead": 0.3, "bell": 0.2},
    ducked={"bass": 1.0, "pad": 0.7, "arp": 0.5},
)


def beat_sample(beat: float) -> int:
    return SPEC.sample(beat)


def pad_chord(mix, beat, length, chord, gain, *args, **kw) -> None:
    _pad_chord(mix, beat, length, CHORDS[chord][0], gain, *args, key=chord, **kw)


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
