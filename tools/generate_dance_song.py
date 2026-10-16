"""Synthesise Raccoon Dance's songs (docs/DANCE.md sections 6 and 11).

Every song is original: numpy oscillators and noise, no samples, nothing
borrowed. Each song is a module in tools/dance_songs/ (its tempo, sections,
chords, melodies and arrangement); tools/dance_synth.py is the shared engine
(instruments, mixing, mastering, AAC encoding). Renders are deterministic:
byte-identical run to run.

    uv run python tools/generate_dance_song.py                        # Neon Nights (the default)
    uv run python tools/generate_dance_song.py --song pumpkin_patch_bounce
    uv run python tools/generate_dance_song.py --song spell_it_out
    uv run python tools/generate_dance_song.py --all                  # every song

Each writes assets/dance/songs/<id>/audio.m4a (AAC 44.1 kHz stereo). The
chart (song.json) is hand-written against the song's beat grid.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import sys
import time
from pathlib import Path
from types import ModuleType

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dance_synth  # noqa: E402
from dance_synth import SR, to_int16  # noqa: E402

SONG_IDS = ("neon_nights", "pumpkin_patch_bounce", "spell_it_out")


def load(song_id: str) -> ModuleType:
    if song_id not in SONG_IDS:
        raise SystemExit(f"unknown song {song_id!r}; choose from {', '.join(SONG_IDS)}")
    return importlib.import_module(f"dance_songs.{song_id}")


def render(song: ModuleType | str, start_beat: float = 0.0, end_beat: float | None = None,
           master: bool = True) -> np.ndarray:
    """Render a song (module or id), or an excerpt of it, to float32 stereo at SR."""
    if isinstance(song, str):
        song = load(song)
    return dance_synth.render(song, start_beat, end_beat, master)


def build(song: ModuleType, out: Path | None = None) -> Path:
    out = out or song.OUT
    t0 = time.perf_counter()
    pcm = render(song)
    t1 = time.perf_counter()
    dance_synth.encode_m4a(pcm, out, song.SPEC.bitrate)
    t2 = time.perf_counter()
    peak = 20 * np.log10(np.abs(pcm).max())
    print(f"[{song.ID}] rendered {len(pcm) / SR:.3f} s ({len(pcm)} samples) in {t1 - t0:.1f} s; "
          f"peak {peak:.2f} dBFS")
    print(f"[{song.ID}] pcm md5 float32 {hashlib.md5(pcm.tobytes()).hexdigest()}  "
          f"int16 {hashlib.md5(to_int16(pcm).tobytes()).hexdigest()}")
    print(f"[{song.ID}] wrote {out} ({out.stat().st_size / 1e6:.2f} MB) in {t2 - t1:.1f} s; "
          f"md5 {hashlib.md5(out.read_bytes()).hexdigest()}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    which = ap.add_mutually_exclusive_group()
    which.add_argument("--song", default="neon_nights", choices=SONG_IDS)
    which.add_argument("--all", action="store_true", help="render every song")
    ap.add_argument("--out", type=Path, help="output path (one song only; default assets/dance/songs/<id>/audio.m4a)")
    args = ap.parse_args()
    if args.all and args.out:
        ap.error("--out needs a single --song")
    for song_id in (SONG_IDS if args.all else (args.song,)):
        build(load(song_id), args.out)


if __name__ == "__main__":
    main()
