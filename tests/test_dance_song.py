"""The test song "Neon Nights" (tools/generate_dance_song.py, docs/DANCE.md section 6).

The chart is authored against beats, so the audio must put a kick exactly on
them: beat b at b * 0.5 s, beat 0 at the very first sample. These tests check
the generator's own output and the shipped AAC file, whose encoder adds
priming samples that the container's edit list must hide.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import generate_dance_song as song  # noqa: E402

TOLERANCE_MS = 5.0


def kick_onsets_ms(pcm: np.ndarray, beats, first_sample: int = 0) -> np.ndarray:
    """Onset error (ms) of the kick nearest each beat.

    Low-pass at 150 Hz, take a 2 ms peak envelope, and find where it rises
    fastest: the first sample whose 3 ms rise reaches half the largest rise
    within 40 ms of the beat. Rise, not level, so a bass note ringing across
    the beat cannot pass for the kick.
    """
    sr, ms = song.SR, song.SR // 1000
    pad = 50 * ms
    mono = np.r_[np.zeros(pad), pcm.astype(np.float64).mean(axis=1), np.zeros(pad)]
    low = np.abs(song.spectral_filter(mono, hi=150, order=2))
    errors = []
    for b in beats:
        s = song.beat_sample(b) - first_sample + pad
        a0, a1 = s - 40 * ms, s + 40 * ms
        win = np.lib.stride_tricks.sliding_window_view(low[a0 - 3 * ms:a1 + 2 * ms], 2 * ms).max(axis=1)
        rise = win[3 * ms:3 * ms + (a1 - a0)] - win[:a1 - a0]
        onset = a0 + int(np.argmax(rise >= 0.5 * rise.max())) + ms
        errors.append((onset - s) / sr * 1000)
    return np.array(errors)


def test_timing_matches_the_spec():
    assert song.SPB == 22050 and song.SR == 44100 and song.BPM == 120
    assert song.OFFSET == 0.0
    assert song.MUSIC_BEATS == 208 and song.DURATION == 106.0
    assert song.N_TOTAL == 106 * 44100
    assert [(a, b) for _, a, b in song.SECTIONS] == [
        (0, 16), (16, 48), (48, 80), (80, 112), (112, 144), (144, 160), (160, 192), (192, 208)]
    assert len(song.PROGRESSION) * 4 == song.MUSIC_BEATS
    kicks = set(song.kick_beats())
    assert all(b in kicks for b in range(0, 144)), "four-on-the-floor from beat 0 to the bridge"
    assert not kicks & set(range(144, 160)), "the bridge drops the kick"
    assert all(b in kicks for b in range(160, 193)) and max(kicks) == 192, "final hit on beat 192"
    assert {44, 108, 156} <= set(song.RISERS), "a riser in the bar before each chorus"


def test_excerpt_is_deterministic_and_on_the_beat():
    a = song.render(42, 50, master=False)  # verse 1 riser into chorus 1
    b = song.render(42, 50, master=False)
    assert hashlib.md5(a.tobytes()).hexdigest() == hashlib.md5(b.tobytes()).hexdigest()
    assert len(a) == 8 * song.SPB
    errors = kick_onsets_ms(a, range(42, 50), first_sample=song.beat_sample(42))
    assert np.abs(errors).max() < TOLERANCE_MS, errors


def test_shipped_audio_has_kicks_on_the_beat_grid():
    av = pytest.importorskip("av")
    path = song.OUT
    assert path.exists(), "run: uv run python tools/generate_dance_song.py"
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        assert stream.rate == 44100 and stream.channels == 2
        frames = [f.to_ndarray() for f in container.decode(stream)]
        declared = stream.duration * stream.time_base
    pcm = np.concatenate(frames, axis=1).T
    # FFmpeg drops the encoder priming (edit list) but not the end padding.
    assert float(declared) == pytest.approx(song.DURATION, abs=1e-3)
    assert abs(len(pcm) / song.SR - song.DURATION) < 0.03
    pcm = pcm[:song.N_TOTAL]
    peak = np.abs(pcm).max()
    assert 0.7 < peak < 1.0, f"peak {20 * np.log10(peak):.2f} dBFS"
    errors = kick_onsets_ms(pcm, song.kick_beats())
    assert np.abs(errors).max() < TOLERANCE_MS, f"worst kick {np.abs(errors).max():.2f} ms off its beat"
    # And the music fills the song: no silent stretch before the ring-out.
    music = pcm[:song.beat_sample(song.MUSIC_BEATS)].mean(axis=1)
    win = song.SR // 10
    rms = np.sqrt((music[:len(music) // win * win].reshape(-1, win) ** 2).mean(axis=1))
    assert 20 * np.log10(rms.min()) > -40
