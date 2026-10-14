"""Write web/tests/fixtures/dance_parity.json: the Python judge's verdicts on a
recorded input stream on every difficulty, and the EASY charts derive_easy
makes, for the browser port to replay (web/tests/dance_parity.test.mjs).

Two streams: a short test chart (its library and song.json travel in the
fixture) and a 30 s slice of Neon Nights (read from assets/ by both sides).
Each stream is a list of pose updates (COCO keypoints at 20 Hz, from bots that
switch behaviour every second: on time, late, a few degrees off, far off,
mirror-wrong, arms down, elbows hidden, wrists hidden, hips out of frame, out of
view) and a list of 60 Hz frames (song time + the newest pose), plus a resume
after a pause and a stall that skips whole windows. The judge runs on exactly the stored
numbers: keypoints are stored in tenths of a pixel and read back through
float32 on both sides, so the inputs are bit-identical.

    uv run python tools/make_dance_parity_fixture.py
"""

from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from kpapp.game.dance import chart  # noqa: E402
from kpapp.game.dance.judge import Judge  # noqa: E402
from kpapp.game.dance.pose import REST, keypoints_for, mirror_pose, normalize_pose  # noqa: E402
from kpapp.inference import Pose  # noqa: E402

OUT = REPO / "web" / "tests" / "fixtures" / "dance_parity.json"
FPS, RATE = 60, 20

POSES = {
    "point_r": {"ra": [150, 165], "la": [20, 10], "lean": 6},
    "t_up_r": {"la": [90, 90], "ra": [170, 175]},
    "wave_r": {"ra": [90, 170], "la": [30, -20]},
    "diag": {"la": [135, 135], "ra": [45, 45]},
    "flex_l": {"la": [100, 180], "ra": [45, 30]},
    "cross": {"la": [-30, -60], "ra": [120, 150]},
}
MOVES = {
    "m1": {"label": "Point Up", "beats": 2,
           "keyframes": [{"beat": 0, "pose": POSES["point_r"]}, {"beat": 1, "pose": mirror_pose(POSES["t_up_r"])}]},
    "m2": {"label": "Wave", "beats": 2,
           "keyframes": [{"beat": 0, "pose": POSES["wave_r"]}, {"beat": 1, "pose": POSES["diag"]}]},
    "m3": {"label": "Flex", "beats": 2,
           "keyframes": [{"beat": 0, "pose": POSES["flex_l"]}, {"beat": 1, "pose": POSES["cross"]}]},
    "hold": {"label": "Diagonal", "beats": 4, "keyframes": [{"beat": 0, "pose": POSES["diag"]}]},
    "m3_m": {"mirror_of": "m3", "label": "Flex Back"},
}
ORDER = ["m1", "m2", "m3", "hold", "m3_m", "m2", "m1"]


BEATS = {"m1": 2, "m2": 2, "m3": 2, "hold": 4, "m3_m": 2}


def groove_json() -> dict:
    moves, beat, i = [], 4.0, 0
    while beat + BEATS[ORDER[i % len(ORDER)]] <= 36:
        name = ORDER[i % len(ORDER)]
        moves.append({"move": name, "beat": beat, "gold": i == 3})
        beat += BEATS[name]
        i += 1
    return {"id": "parity", "title": "Parity Groove", "bpm": 120, "offset": 0.0, "duration": 20.0, "moves": moves}


MODES = ["perfect", "late", "near", "far", "mirror", "rest", "no_elbows", "no_hips", "gone", "perfect", "near",
         "no_wrists"]


def bot_pose(song, st: float, mode: str, rng: random.Random):
    """The pose a bot in `mode` shows at song time `st` (None: not in view)."""
    coach = normalize_pose(song.coach_pose(song.beat_of(st - (0.3 if mode == "late" else 0.1))))
    if mode == "gone":
        return None
    if mode == "rest":
        return REST
    if mode == "mirror":
        return mirror_pose(coach)
    if mode in ("near", "far"):
        lo, hi = (12, 28) if mode == "near" else (30, 70)
        for k in ("la", "ra"):
            coach[k] = [a + rng.choice((-1, 1)) * rng.uniform(lo, hi) for a in coach[k]]
    return coach


def record(song, t0: float, t1: float, seed: int, resume_at: float, stall: tuple[float, float]):
    rng = random.Random(seed)
    noise = np.random.default_rng(seed)
    poses: list[dict | None] = []
    frames: list[list] = []
    st, last_capture, current = t0, -1e9, -1
    while st < t1:
        if stall[0] <= st < stall[1]:
            st = stall[1]  # a frame that jumped over whole windows
        if st - last_capture >= 1.0 / RATE - 1e-9:
            last_capture = st
            mode = MODES[random.Random(seed * 1000 + int(st)).randrange(len(MODES))]
            target = bot_pose(song, st, mode, rng)
            if target is None:
                poses.append(None)
            else:
                visible = {"no_elbows": {"shoulder", "wrist", "hip"},
                           "no_hips": {"shoulder", "elbow", "wrist"},
                           "no_wrists": {"shoulder", "elbow", "hip"}}.get(mode)
                kp = keypoints_for(target, visible=visible)
                xy = kp.xy.astype(np.float64) + noise.uniform(-4, 4, kp.xy.shape)
                tenths = np.round(xy * 10).astype(int)
                tenths[kp.confidence <= 0] = 0
                poses.append({"xy": tenths.reshape(-1).tolist(),
                              "v": [int(i) for i in np.flatnonzero(kp.confidence > 0)]})
            current = len(poses) - 1
        frames.append([round(st, 5), current])
        st += 1.0 / FPS
    resume = min(range(len(frames)), key=lambda i: abs(frames[i][0] - resume_at))
    return poses, frames, [resume]


def to_pose(p: dict | None):
    if p is None:
        return None
    xy = (np.array(p["xy"], np.float64) / 10.0).astype(np.float32).reshape(17, 2)
    conf = np.zeros(17, np.float32)
    conf[p["v"]] = 0.9
    return Pose(xy=xy, confidence=conf, score=0.9)


def judge_stream(song, poses, frames, resumes, level: str) -> dict:
    """The judge's verdicts on one stream, for one difficulty (its chart and rules)."""
    from kpapp.game.dance.pose import measure

    song = song.for_difficulty(level)
    j = Judge(song, level)
    measured = [None if p is None else measure(to_pose(p)) for p in poses]
    ratings = []
    for i, (st, idx) in enumerate(frames):
        if i in resumes:
            j.on_resume(st)
        for r in j.update(st, measured[idx] if idx >= 0 else None):
            ratings.append([i, r.point.index, r.name, r.similarity])
    # Close every window still open.
    end = frames[-1][0] + 5.0
    for r in j.update(end, None):
        ratings.append([len(frames), r.point.index, r.name, r.similarity])
    return {"points": [[p.index, p.time, p.weight] for p in song.judge_points()],
            "ratings": ratings, "end_time": end, "score": j.score, "counts": j.counts, "unseen": j.unseen,
            "best_combo": j.best_combo, "earned": j.earned}


def easy_chart(song) -> dict:
    """What derive_easy made of a song (chart.derive_easy), and the freeze scores
    it decided with: the browser must derive exactly the same chart."""
    normal = song.judge_points()
    easy = song.for_difficulty("easy")
    return {
        "chart": [[p.beat, p.move.name, p.move.beats, p.gold, list(p.move.judge)] for p in easy.chart],
        "judge_beats": [p.beat for p in easy.judge_points()],
        "freeze": [[a.index, b.index, chart.freeze_score(b.pose, a.pose)] for a, b in zip(normal, normal[1:])],
    }


LEVELS = ("easy", "normal", "hard")


def main() -> int:
    lib = chart.parse_moves(MOVES, "parity library")
    data = groove_json()
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "parity"
        folder.mkdir()
        (folder / "song.json").write_text(json.dumps(data))
        groove = chart.load_song(folder, lib)
    neon = chart.load_song(chart.SONGS / "neon_nights", chart.load_library())

    streams = []
    for name, song, (t0, t1), seed, resume_at, stall in (
        ("groove", groove, (-0.5, 18.5), 11, 9.0, (13.0, 14.2)),
        ("neon_nights", neon, (22.0, 52.0), 23, 37.0, (45.0, 46.3)),
    ):
        poses, frames, resumes = record(song, t0, t1, seed, resume_at, stall)
        out = {"name": name, "poses": poses, "frames": frames, "resumes": resumes,
               "levels": {level: judge_stream(song, poses, frames, resumes, level) for level in LEVELS}}
        streams.append(out)
        for level, res in out["levels"].items():
            names = {}
            for r in res["ratings"]:
                names[r[2]] = names.get(r[2], 0) + 1
            print(f"{name} {level}: {len(poses)} poses, {len(frames)} frames, score {res['score']}, {names}")

    easy = {"groove": easy_chart(groove), "neon_nights": easy_chart(neon)}
    print(f"easy charts: groove {len(easy['groove']['judge_beats'])} points, "
          f"neon_nights {len(easy['neon_nights']['judge_beats'])} points")
    fixture = {"about": "made by tools/make_dance_parity_fixture.py; replayed by web/tests/dance_parity.test.mjs",
               "groove": {"library": MOVES, "song": data}, "streams": streams, "easy": easy}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fixture, separators=(",", ":")))
    print(f"wrote {OUT.relative_to(REPO)} ({OUT.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
