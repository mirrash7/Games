"""Write web/tests/fixtures/player_parity.json: what the desktop PlayerLock
(src/kpapp/player.py) decides on fixed crowds, for the browser port
(web/js/core/player.js) to replay in web/tests/player.test.mjs.
tests/test_player_parity.py fails if the committed fixture no longer matches
the Python, so regenerate it after changing player.py:

    uv run python tools/make_player_parity_fixture.py

The crowds are tests/crowd_poses.py's (the same ones tests/test_player.py
checks): passers-by in front and behind, short and long occlusions, someone
standing in front, fast dance steps, a golf swing, bystanders at the edges, a
crowd at the start, the player leaving and a new one arriving, a far
bystander handing over, two random dense crowds (one with a known wrong
hand-over, so the port has to reproduce the mistake too), a teleport, a
repick, poses without boxes, poses without shoulders and duplicate detections.
Visible keypoints and boxes
are stored in tenths of a pixel and read back through float32 on both sides,
so the inputs are bit-identical. Per result: the state, the locked track id,
which input pose was handed out, the switch count, `changed`, how long the
player has been held, and which inputs went to `others`.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))

import crowd_poses as C  # noqa: E402
from kpapp.inference import Pose  # noqa: E402
from kpapp.player import PlayerLock, body_of, likeness, most_likely_player  # noqa: E402

OUT = REPO / "web" / "tests" / "fixtures" / "player_parity.json"
STATE = {"searching": "S", "locked": "L", "holding": "H"}


def _store(pose: Pose, box=True):
    vis = [int(i) for i in np.flatnonzero(pose.confidence >= 0.5)]
    xy = np.round(pose.xy[vis].astype(np.float64) * 10).astype(np.int64).reshape(-1).tolist()
    b = None
    if box and pose.box is not None:
        b = np.round(pose.box.astype(np.float64) * 10).astype(np.int64).tolist()
    return {"xy": xy, "v": vis, "b": b}


def _load(p) -> Pose:
    xy = np.zeros((17, 2), np.float32)
    xy[p["v"]] = (np.array(p["xy"], np.float64) / 10.0).astype(np.float32).reshape(-1, 2)
    conf = np.zeros(17, np.float32)
    conf[p["v"]] = 0.9
    box = None if p["b"] is None else (np.array(p["b"], np.float64) / 10.0).astype(np.float32)
    return Pose(xy=xy, confidence=conf, score=0.9, box=box)


def _dense(seed: int):
    r = random.Random(seed)
    psw = r.uniform(110, 170)
    people = {"player": (C.dancer if seed % 2 else C.golfer)(x=r.uniform(560, 720), sw=psw)}
    for k in range(6):
        sw = r.choice([r.uniform(60, psw * 0.78), r.uniform(psw * 1.3, 220), r.uniform(60, 220)])
        d = r.choice([-1, 1])
        people[f"w{k}"] = C.walker(-200 if d > 0 else 1480, d * r.uniform(2.6, 4.5), sw, t0=r.uniform(0.5, 8),
                                   score=r.uniform(0.85, 0.98))
    return people, r.choice([15, 20, 30])


def scenarios():
    """(name, frames [(t, [Pose])], repick frame indices)."""
    out = []

    def add(name, people, duration, rate, seed, repick=(), boxes=True, **kw):
        frames = [(t, [p for _, p in dets]) for t, dets in C.stream(people, duration, rate, seed, **kw)]
        if not boxes:
            for _, poses in frames:
                for p in poses:
                    p.box = None
        out.append((name, frames, list(repick)))

    passers = {
        "player": C.dancer(),
        "front": C.walker(-150, 3.7, 190, t0=1.0, score=0.95),
        "behind": C.walker(1400, -3.7, 80, t0=2.0, score=0.9),
        "same": C.walker(-100, 3.7, 120, t0=4.0, score=0.93),
        "front2": C.walker(1450, -3.0, 165, t0=5.0, score=0.97),
    }
    add("passers 15 Hz", passers, 8.0, 15, 0)
    add("passers 30 Hz", passers, 7.0, 30, 3)
    add("occluded 0.5 s", {"player": C.hidden(C.dancer(), 2.0, 2.5),
                           "front": C.walker(-150, 3.7, 190, t0=1.2, score=0.95)}, 4.0, 20, 1)
    add("occluded 2 s", {"player": C.hidden(C.dancer(), 2.0, 4.0),
                         "front": C.walker(-150, 2.5, 190, t0=1.0, score=0.95)}, 6.0, 20, 3)
    add("standing in front", {"player": C.hidden(C.dancer(step=0.0), 2.0, 4.0),
                              "s": C.stander(640, 190, t0=1.9, t1=4.0)}, 7.0, 20, 0)
    add("big dance steps", {"player": C.dancer(step=1.5, bpm=130), "w": C.walker(-150, 3.7, 160, t0=1.0),
                            "b": C.walker(1400, -3.7, 100, t0=2.5)}, 6.0, 30, 2)
    add("golf swing", {"player": C.golfer(), "w": C.walker(-150, 3.7, 170, t0=1.0),
                       "w2": C.walker(1400, -3.7, 140, t0=2.5)}, 6.0, 20, 1)
    add("bystanders at the edges", {"player": C.dancer(), "l": C.stander(70, 170, 0.97),
                                    "r": C.stander(1210, 170, 0.97)}, 4.0, 20, 2)
    add("crowd at the start", {"player": C.dancer(), "a": C.walker(100, 3.7, 150, score=0.95),
                               "b": C.walker(1100, -3.7, 110, score=0.95)}, 4.0, 15, 0)
    add("leaving and arriving", {"player": C.leaving(C.dancer(), 3.0),
                                 "new": C.walker(-150, 3.0, 140, t0=4.0, stop_x=640)}, 8.0, 20, 1)
    add("far bystander hands over", {"bg": C.stander(300, 60, 0.9),
                                     "player": C.walker(1400, -3.0, 150, t0=2.0, stop_x=640)}, 7.0, 20, 0)
    for seed in (13, 21):
        people, rate = _dense(seed)
        add(f"dense crowd {seed}", people, 10.0, rate, seed)
    add("repick", {"a": C.stander(560, 140), "b": C.stander(860, 120), "c": C.stander(250, 100)}, 3.0, 20, 4,
        repick=(20, 35, 50))
    add("no boxes", {"player": C.dancer(), "w": C.walker(-150, 3.7, 170, t0=0.5)}, 4.0, 20, 5, boxes=False)
    add("no shoulders", {"player": C.dancer(), "w": C.walker(1400, -3.7, 150, t0=0.5)}, 4.0, 20, 6)
    # Hide both shoulders now and then, report someone twice, and put a teleport glitch in.
    name, frames, _ = out[-1]
    for i, (_, poses) in enumerate(frames):
        if i % 7 == 3 and poses:
            poses[0].confidence[[5, 6]] = 0.0
        if i % 5 == 1 and poses:
            twin = Pose(xy=poses[-1].xy + 3.0, confidence=poses[-1].confidence.copy(), score=0.5,
                        box=None if poses[-1].box is None else poses[-1].box + 3.0)
            poses.append(twin)
        if i == 30 and poses:
            poses[0].xy[:, 0] += 0.4 * C.W
            if poses[0].box is not None:
                poses[0].box[[0, 2]] += 0.4 * C.W
    return out


def record(name, frames, repick):
    stored = [{"t": t, "poses": [_store(p) for p in poses]} for t, poses in frames]
    lock = PlayerLock((C.W, C.H))
    expect, bodies = [], []
    for i, f in enumerate(stored):
        poses = [_load(p) for p in f["poses"]]
        if i in repick:
            lock.repick()
        got = lock.update(poses, f["t"])
        idx = next((k for k, p in enumerate(poses) if p is got), -1)
        others = [k for k, p in enumerate(poses) if any(p is o for o in lock.others)]
        expect.append([STATE[lock.state], lock.player_id or 0, idx, lock.switches, int(lock.changed),
                       lock.held_for, others])
        if i % 10 == 0:
            for p in poses:
                b = body_of(p)
                bodies.append([i, b.ax, b.ay, b.scale, int(b.sure), b.quality, likeness(b, C.W, None),
                               list(b.box)])
    pick = []
    for f in stored[::10]:
        loaded = [_load(q) for q in f["poses"]]
        best = most_likely_player(loaded, C.W)
        pick.append(next((k for k, q in enumerate(loaded) if q is best), -1))
    return {"name": name, "frames": stored, "repick": repick, "expect": expect, "bodies": bodies, "pick": pick}


def build() -> dict:
    return {
        "about": "made by tools/make_player_parity_fixture.py; replayed by web/tests/player.test.mjs",
        "size": [C.W, C.H],
        "scenarios": [record(*s) for s in scenarios()],
    }


def main() -> int:
    data = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    for s in data["scenarios"]:
        states = "".join(e[0] for e in s["expect"])
        print(f"{s['name']:26s} {len(s['frames']):4d} results  switches {s['expect'][-1][3]}  "
              f"locked {states.count('L')}  held {states.count('H')}  searching {states.count('S')}")
    print(f"wrote {OUT.relative_to(REPO)} ({OUT.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
