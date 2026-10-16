"""Write web/tests/fixtures/dance_motion.json: the Python coach motion
(src/kpapp/game/dance/motion.py) and the painter's 3D layout at sampled beats
of every song, for the browser port to reproduce (web/tests/dance_motion.test.mjs).

Motion is a pure function of (song keyframes, beat), so both sides compute it
from the same song.json and moves.json and must agree to within 1e-6. Every
fifth sample also carries the painter's joints (stub art, the stage scale), so
the twist, sway, squash, foreshortening and draw order are pinned too.

    uv run python tools/make_dance_motion_fixture.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from kpapp.game.dance import chart  # noqa: E402
from kpapp.game.dance.art import stub_art  # noqa: E402
from kpapp.game.dance.motion import coach_motion  # noqa: E402
from kpapp.game.dance.render import CoachPainter  # noqa: E402

OUT = REPO / "web" / "tests" / "fixtures" / "dance_motion.json"
STEP = 0.93  # beats between samples: not a divisor of a beat, so every phase of the groove shows up
GEOMETRY_EVERY = 5


def r(v):
    if isinstance(v, float):
        return round(v, 7)
    if isinstance(v, (list, tuple)):
        return [r(x) for x in v]
    if isinstance(v, dict):
        return {k: r(x) for k, x in v.items()}
    return v


def geometry(painter, motion) -> dict:
    g = painter.layout(None, motion=motion)
    return {"j": {k: [float(v[0]), float(v[1])] for k, v in g.j.items()}, "head_c": list(g.head_c),
            "head_angle": g.head_angle, "behind": list(g.behind), "front_first": list(g.front_first)}


def main() -> None:
    lib = chart.load_library(chart.LIBRARY)
    painter = CoachPainter(stub_art())
    songs = []
    for sid in json.loads((chart.SONGS / "index.json").read_text()):
        song = chart.load_song(chart.SONGS / sid, lib)
        end = song.chart[-1].end + 2.0
        samples = []
        b, i = -2.0, 0
        while b < end:
            m = coach_motion(song, b)
            s = {"beat": b, "motion": m}
            if i % GEOMETRY_EVERY == 0:
                s["geometry"] = geometry(painter, m)
            samples.append(r(s))
            b = round(b + STEP, 6)
            i += 1
        songs.append({"id": sid, "samples": samples})
    idle = [r({"beat": b, "motion": coach_motion(None, b, energy=0.5)}) for b in (0.0, 0.37, 1.5, 2.81, 7.2)]
    OUT.write_text(json.dumps({"step": STEP, "songs": songs, "idle": idle}, separators=(",", ":")) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB, {sum(len(s['samples']) for s in songs)} samples)")


if __name__ == "__main__":
    main()
