"""Raccoon Dance's synthesised songs (tools/generate_dance_song.py, docs/DANCE.md section 6).

The charts are authored against beats, so the audio must put a kick exactly on
them: beat b at b * 60 / bpm seconds, beat 0 at the very first sample. These
tests check the generator's own output and the shipped AAC files, whose
encoder adds priming samples that the container's edit list must hide.

Songs: "Neon Nights" (120 BPM, A minor) and "Pumpkin Patch Bounce" (116 BPM,
D minor). Each is a module in tools/dance_songs/ on the shared engine
tools/dance_synth.py.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import dance_synth  # noqa: E402
import generate_dance_song as gen  # noqa: E402
from dance_songs import neon_nights as neon  # noqa: E402
from dance_songs import pumpkin_patch_bounce as pumpkin  # noqa: E402

TOLERANCE_MS = 5.0
SR = dance_synth.SR
SONGS = [neon, pumpkin]


def kick_onsets_ms(song, pcm: np.ndarray, beats, first_sample: int = 0) -> np.ndarray:
    """Onset error (ms) of the kick nearest each beat.

    Low-pass at 150 Hz, take a 2 ms peak envelope, and find where it rises
    fastest: the first sample whose 3 ms rise reaches half the largest rise
    within 40 ms of the beat. Rise, not level, so a bass note ringing across
    the beat cannot pass for the kick.
    """
    ms = SR // 1000
    pad = 50 * ms
    mono = np.r_[np.zeros(pad), pcm.astype(np.float64).mean(axis=1), np.zeros(pad)]
    low = np.abs(dance_synth.spectral_filter(mono, hi=150, order=2))
    errors = []
    for b in beats:
        s = song.SPEC.sample(b) - first_sample + pad
        a0, a1 = s - 40 * ms, s + 40 * ms
        win = np.lib.stride_tricks.sliding_window_view(low[a0 - 3 * ms:a1 + 2 * ms], 2 * ms).max(axis=1)
        rise = win[3 * ms:3 * ms + (a1 - a0)] - win[:a1 - a0]
        onset = a0 + int(np.argmax(rise >= 0.5 * rise.max())) + ms
        errors.append((onset - s) / SR * 1000)
    return np.array(errors)


def decode(path: Path) -> tuple[np.ndarray, float]:
    """(pcm (n, 2) float32, declared duration in s) of an audio file, via PyAV."""
    av = pytest.importorskip("av")
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        assert stream.rate == 44100 and stream.channels == 2
        frames = [f.to_ndarray() for f in container.decode(stream)]
        declared = float(stream.duration * stream.time_base)
    return np.concatenate(frames, axis=1).T, declared


def test_every_song_is_registered():
    assert gen.SONG_IDS == ("neon_nights", "pumpkin_patch_bounce")
    for song in SONGS:
        assert gen.load(song.ID) is song and song.SPEC.id == song.ID
        assert song.OUT == REPO / "assets" / "dance" / "songs" / song.ID / "audio.m4a"
        assert song.OFFSET == 0.0
        assert song.SPEC.n_total == round(song.DURATION * SR)
        assert len(song.PROGRESSION) * 4 == song.MUSIC_BEATS
        assert song.SECTIONS[0][1] == 0 and song.SECTIONS[-1][2] == song.MUSIC_BEATS
        assert all(a[2] == b[1] for a, b in zip(song.SECTIONS, song.SECTIONS[1:])), "sections tile the song"


def test_neon_nights_timing_matches_the_spec():
    song = neon
    assert song.SPB == 22050 and song.SPEC.spb == 22050 and song.BPM == 120
    assert song.MUSIC_BEATS == 208 and song.DURATION == 106.0
    assert song.N_TOTAL == song.SPEC.n_total == 106 * 44100
    assert [(a, b) for _, a, b in song.SECTIONS] == [
        (0, 16), (16, 48), (48, 80), (80, 112), (112, 144), (144, 160), (160, 192), (192, 208)]
    kicks = set(song.kick_beats())
    assert all(b in kicks for b in range(0, 144)), "four-on-the-floor from beat 0 to the bridge"
    assert not kicks & set(range(144, 160)), "the bridge drops the kick"
    assert all(b in kicks for b in range(160, 193)) and max(kicks) == 192, "final hit on beat 192"
    assert {44, 108, 156} <= set(song.RISERS), "a riser in the bar before each chorus"


def test_pumpkin_patch_bounce_timing_matches_the_spec():
    song = pumpkin
    assert song.BPM == 116 and song.SPEC.spb == pytest.approx(44100 * 60 / 116)
    assert song.MUSIC_BEATS == 176 and song.DURATION == 93.0
    assert song.MUSIC_BEATS * 60 / song.BPM == pytest.approx(91.034, abs=1e-3)
    assert song.SPEC.n_total == 93 * 44100
    assert [(n, a, b) for n, a, b in song.SECTIONS] == [
        ("intro", 0, 16), ("hook", 16, 48), ("verse", 48, 80), ("hook", 80, 112),
        ("break", 112, 128), ("hook", 128, 160), ("outro", 160, 176)]
    assert song.CRASHES == (16, 48, 80, 112, 128, 160)
    kicks = set(song.kick_beats())
    assert all(b in kicks for b in range(16, 112)) and all(b in kicks for b in range(128, 161)), \
        "a kick on every beat of the hooks and the verse"
    assert set(range(0, 15)) <= kicks and 15 not in kicks, "kick through the intro, then the stop on 15"
    assert not kicks & set(range(112, 128)), "the break drops the kick"
    assert max(kicks) == 160, "the final hit on beat 160"
    assert set(song.RISERS) == {12, 44, 76, 108, 124, 156}, "a riser in the bar before each section"
    assert song.PROGRESSION[4:12] == song.PROGRESSION[20:28] == song.PROGRESSION[32:40], "the hook repeats"


def test_pumpkin_count_in_and_phrase_marks():
    """The game's 3-2-1 is a snare on beats 12, 13, 14; crashes open every section;
    every 8-beat phrase of the routine ends in an audible fill."""
    mix = dance_synth.Mix(pumpkin.SPEC, 0, pumpkin.SPEC.n_total, shots=pumpkin.make_shot)
    pumpkin.arrange(mix)
    snares = {b for bus, name, b in mix.hits if name == "snare" and b < 16}
    assert snares == {12, 13, 14}
    assert sorted(b for _, name, b in mix.hits if name == "crash") == [16, 48, 80, 112, 128, 160]
    claps = {b for _, name, b in mix.hits if name == "clap"}
    for b in range(17, 112, 2):
        if pumpkin.section_of(b)[0] == "hook":
            assert b in claps, f"claps on 2 and 4 (beat {b})"
    assert {114, 118, 122, 126} <= claps, "the break's half-time clap on 3"
    snaps = {b for _, name, b in mix.hits if name == "snap"}
    assert all(b in snaps for b in range(49, 80, 2)), "snaps on 2 and 4 through the verse"


def test_pumpkin_excerpt_is_deterministic_and_on_the_beat():
    a = gen.render(pumpkin, 40, 50, master=False)   # hook 1's riser into the verse
    b = gen.render("pumpkin_patch_bounce", 40, 50, master=False)
    assert hashlib.md5(a.tobytes()).hexdigest() == hashlib.md5(b.tobytes()).hexdigest()
    assert len(a) == pumpkin.SPEC.sample(50) - pumpkin.SPEC.sample(40)
    errors = kick_onsets_ms(pumpkin, a, range(40, 50), first_sample=pumpkin.SPEC.sample(40))
    assert np.abs(errors).max() < TOLERANCE_MS, errors


def test_neon_excerpt_is_deterministic_and_on_the_beat():
    a = gen.render(neon, 42, 50, master=False)  # verse 1 riser into chorus 1
    b = gen.render(neon, 42, 50, master=False)
    assert hashlib.md5(a.tobytes()).hexdigest() == hashlib.md5(b.tobytes()).hexdigest()
    assert len(a) == 8 * neon.SPB
    errors = kick_onsets_ms(neon, a, range(42, 50), first_sample=neon.beat_sample(42))
    assert np.abs(errors).max() < TOLERANCE_MS, errors


def test_neon_excerpt_is_unchanged_by_the_engine_refactor():
    """Neon Nights' bytes are pinned: the shared engine must render it exactly as the
    original single-song generator did (that render's md5; the full song then
    encodes to the shipped audio.m4a, md5 b8f44bbefc53508309de513046f80441).
    A numpy upgrade could legitimately change float rounding: re-render, listen, re-pin."""
    pcm = gen.render(neon, 44, 52, master=False)
    assert hashlib.md5(pcm.tobytes()).hexdigest() == "93c92a18d07e8611659b42be7ed1c6a4"


@pytest.mark.parametrize("song", SONGS, ids=lambda s: s.ID)
def test_shipped_audio_has_kicks_on_the_beat_grid(song):
    path = song.OUT
    assert path.exists(), f"run: uv run python tools/generate_dance_song.py --song {song.ID}"
    pcm, declared = decode(path)
    # FFmpeg drops the encoder priming (edit list) but not the end padding.
    assert declared == pytest.approx(song.DURATION, abs=1e-3)
    assert abs(len(pcm) / SR - song.DURATION) < 0.03
    pcm = pcm[:song.SPEC.n_total]
    peak = np.abs(pcm).max()
    assert 0.7 < peak < 1.0, f"peak {20 * np.log10(peak):.2f} dBFS"
    errors = kick_onsets_ms(song, pcm, song.kick_beats())
    assert np.abs(errors).max() < TOLERANCE_MS, f"worst kick {np.abs(errors).max():.2f} ms off its beat"
    # And the music fills the song: no silent stretch before the ring-out.
    music = pcm[:song.SPEC.sample(song.MUSIC_BEATS)].mean(axis=1)
    win = SR // 10
    rms = np.sqrt((music[:len(music) // win * win].reshape(-1, win) ** 2).mean(axis=1))
    assert 20 * np.log10(rms.min()) > -40


def test_shipped_pumpkin_audio_counts_in_and_ends_cleanly():
    path = pumpkin.OUT
    assert path.exists(), "run: uv run python tools/generate_dance_song.py --song pumpkin_patch_bounce"
    pcm, _ = decode(path)
    pcm = pcm[:pumpkin.SPEC.n_total]
    peak = np.abs(pcm).max()
    assert 20 * np.log10(peak) == pytest.approx(-1.0, abs=0.5), "peaks about -1 dBFS, never clipped"
    mono = pcm.astype(np.float64).mean(axis=1)
    high = dance_synth.spectral_filter(mono, lo=1500, hi=9000)

    def burst(beat):  # high-band energy in the first 60 ms after a beat
        s = pumpkin.SPEC.sample(beat)
        return np.sqrt((high[s:s + int(0.06 * SR)] ** 2).mean())
    count_in = min(burst(b) for b in pumpkin.COUNT_IN)
    plain = max(burst(b) for b in (8, 9, 10))   # the same intro, no snare
    assert 20 * np.log10(count_in / plain) > 4.5, "the 3-2-1 snares stand out"
    # beat 15 stops: the low end drops away before the hook lands on 16
    low = np.abs(dance_synth.spectral_filter(mono, hi=150))
    s15, s16 = pumpkin.SPEC.sample(15.25), pumpkin.SPEC.sample(16)
    assert low[s15:s16 - SR // 100].max() < 0.25 * low[s16:s16 + SR // 10].max()
    # the ring-out fades to silence, no click at the end
    assert np.abs(pcm[-SR // 100:]).max() < 1e-3
