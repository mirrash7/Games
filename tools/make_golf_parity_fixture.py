"""Write web/tests/fixtures/golf_parity.json: what the desktop Raccoon Golf
computes on fixed inputs, for the browser port to replay
(web/tests/golf_parity.test.mjs). tests/test_golf_parity.py fails if the
committed fixture no longer matches the Python, so regenerate it after changing
swing.py, physics.py, course.py or the game's rules:

    uv run python tools/make_golf_parity_fixture.py

What it holds:

- swing: SwingDetector on recorded pose streams (synthetic golfers from
  tests/golf_poses.py: full swings at 15-30 Hz on both sides, power from
  gentle to overswing, thin strikes, practice swings, a slow lowering, a
  teleport glitch, losing the player, a wrist dropout, the shoulders turning
  away, a reset mid-swing, framing problems, putts, and long jittery
  addresses). Keypoints are stored in tenths of a pixel and read back through
  float32 on both sides, so the inputs are bit-identical. Per frame: the
  stage, the live power and Lenny's swing angle; every event; every change of
  the guidance line.
- physics: shots on each hole (club, power, heading, lie, quality, curve,
  wind) -> final state, resting point, landing, apex, bounces, trees hit, the
  path, and the water drop point; previews; a shot stepped with uneven dts.
- course: lie_at on a grid over each hole, green_gradient over each green,
  aim_point from many spots and reaches, each hole's length.
- game: auto-club and auto-aim from many spots, the scoring and naming
  tables, and scripted rounds with explicit wind (Python's random draws can't
  be reproduced in the browser, so the fixture carries them).
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))

import golf_poses as GP  # noqa: E402
from kpapp.config import KP  # noqa: E402
from kpapp.controls import ControlState  # noqa: E402
from kpapp.game.golf import physics as P  # noqa: E402
from kpapp.game.golf.art import stub_art  # noqa: E402
from kpapp.game.golf.course import load_course  # noqa: E402
from kpapp.game.golf.game import GolfGame, Phase, hole_points, holed_camera, result_name, to_par_text  # noqa: E402
from kpapp.game.golf.swing import PUTT_FULL, SwingDetector, SwingEvent  # noqa: E402
from kpapp.inference import Pose  # noqa: E402

OUT = REPO / "web" / "tests" / "fixtures" / "golf_parity.json"
STAGE_CODE = {"no_grip": "N", "addressing": "A", "ready": "R", "backswing": "B", "downswing": "D"}
REAL = dict(model_hz=20.0, lag=0.15, noise=5.0, elbows_every=3)


# --- swing streams ------------------------------------------------------------


def _store(pose: Pose | None):
    if pose is None:
        return None
    vis = [int(i) for i in np.flatnonzero(pose.confidence > 0)]
    xy = np.zeros((17, 2), np.int64)
    xy[vis] = np.round(pose.xy[vis].astype(np.float64) * 10).astype(np.int64)
    return {"xy": xy.reshape(-1).tolist(), "v": vis}


def _load(p) -> Pose | None:
    if p is None:
        return None
    xy = (np.array(p["xy"], np.float64) / 10.0).astype(np.float32).reshape(17, 2)
    conf = np.zeros(17, np.float32)
    conf[p["v"]] = 0.9
    return Pose(xy=xy, confidence=conf, score=0.9)


def _turn_shoulders(power: float):
    """The shoulders narrow (to half at the top) as the body turns into the backswing."""
    top = GP.top_angle_for(power)

    def tweak(pose, t, motion, lag):
        u, h = motion(t - lag)
        turn = 0.5 * min(1.0, max(0.0, math.acos(max(-1.0, min(1.0, -h / 1.35))) / top))
        ls, rs = KP["left_shoulder"], KP["right_shoulder"]
        mid = (pose.xy[ls] + pose.xy[rs]) / 2
        pose.xy[ls] = mid + (pose.xy[ls] - mid) * (1 - turn)
        pose.xy[rs] = mid + (pose.xy[rs] - mid) * (1 - turn)
    return tweak


def _practice():
    top = GP.top_angle_for(0.8)

    def motion(t):
        if t < 0.6:
            return 0.0, -0.4 - 0.9 * (t / 0.6)
        if t < 1.3:
            return GP.arc(top * (t - 0.6) / 0.7)
        if t < 1.7:
            return GP.arc(top - (top + 1.8) * (t - 1.3) / 0.4)
        return GP.arc(-1.8)
    return motion


def _lowering():
    top = GP.top_angle_for(1.0)

    def motion(t):
        if t < 1.6:
            return GP.arc(0.0)
        if t < 2.4:
            return GP.arc(top * (t - 1.6) / 0.8)
        if t < 2.6:
            return GP.arc(top)
        if t < 4.1:
            return GP.arc(top * (1 - (1 - math.cos(math.pi * (t - 2.6) / 1.5)) / 2))
        return GP.arc(0.0)
    return motion


def _teleport(t):
    return (3.0, 0.5) if 1.5 <= t < 1.55 else GP.arc(0.0)


def _putt(power, back=0.7, through=0.45, side=1):
    amp = power * PUTT_FULL
    return GP.path((0.0, 0.0, -1.3), (1.4, 0.0, -1.3), (1.4 + back, side * amp, -1.27),
                   (1.6 + back, side * amp, -1.27), (1.6 + back + through, -side * amp * 0.8, -1.27),
                   (3.5 + back, -side * amp * 0.8, -1.27))


def swing_scenarios():
    """(name, mode, motion, duration, stream kwargs, pose tweak or None, reset frames)."""
    sw = GP.swing_path
    out = [
        ("address", "full", lambda t: GP.arc(0.0), 1.5, REAL, None),
        ("swing_p90_r15", "full", sw(power=0.9, side=1)[0], 4.0, {**REAL, "model_hz": 15.0}, None),
        ("swing_p90_l20", "full", sw(power=0.9, side=-1)[0], 4.0, REAL, None),
        ("swing_p90_r30", "full", sw(power=0.9, side=1)[0], 4.0, {**REAL, "model_hz": 30.0}, None),
        ("swing_p90_l30", "full", sw(power=0.9, side=-1)[0], 4.0, {**REAL, "model_hz": 30.0}, None),
        ("swing_clean", "full", sw(power=0.75)[0], 4.0, dict(model_hz=20.0), None),
        ("swing_p25", "full", sw(power=0.25)[0], 4.0, REAL, None),
        ("swing_p65", "full", sw(power=0.65)[0], 4.0, REAL, None),
        ("swing_p100", "full", sw(power=1.0)[0], 4.0, REAL, None),
        ("swing_p110", "full", sw(power=1.1)[0], 4.0, REAL, None),
        ("overswing", "full", sw(power=1.3)[0], 4.0, REAL, None),
        ("thin", "full", sw(power=0.9, low_cross=1.0)[0], 4.0, REAL, None),
        ("elbows", "full", sw(power=0.75)[0], 4.0, {**REAL, "elbows_every": 2}, None),
        ("practice", "full", _practice(), 3.0, REAL, None),
        ("lowering", "full", _lowering(), 5.0, REAL, None),
        ("teleport", "full", _teleport, 3.0, dict(model_hz=20.0, noise=5.0), None),
        ("turning", "full", sw(power=0.7)[0], 4.0, REAL, _turn_shoulders(0.7)),
        ("hands_apart", "full", lambda t: GP.arc(0.0), 2.0, {**REAL, "gap": 1.2}, None),
        ("chest", "full", lambda t: (0.0, -0.4), 2.0, REAL, None),
        ("one_hand", "full", lambda t: GP.arc(0.0), 2.0, {**REAL, "hands": lambda t: 1}, None),
        ("restless", "full", lambda t: (0.3 * math.sin(t * 5.0), -1.2 + 0.15 * math.sin(t * 7.0)), 3.0, REAL, None),
        ("too_close", "full", lambda t: (0.0, -0.5), 1.0, {**REAL, "centre": (640.0, 600.0)}, None),
        ("still_full", "full", lambda t: GP.arc(0.0), 10.0, {**REAL, "noise": 6.0, "seed": 3}, None),
        ("putt_15", "putt", _putt(0.15), 4.5, REAL, None),
        ("putt_60_l", "putt", _putt(0.6, side=-1), 4.5, REAL, None),
        ("putt_90", "putt", _putt(0.9), 4.5, REAL, None),
        ("putt_160", "putt", _putt(1.6), 4.5, REAL, None),
        ("putt_slow", "putt", GP.path((0.0, 0.0, -1.3), (1.4, 0.0, -1.3), (2.0, 0.6, -1.27), (2.2, 0.6, -1.27),
                                      (6.2, -0.3, -1.27)), 7.0, REAL, None),
        ("still_putt", "putt", lambda t: GP.arc(0.0), 10.0, {**REAL, "noise": 6.0, "seed": 4}, None),
    ]
    motion, t_imp = sw(power=0.9)
    out.append(("lost", "full", motion, 4.0, {**REAL, "lost": lambda t, ti=t_imp: t > ti - 0.25}, None))
    motion, t_imp = sw(power=0.8)
    out.append(("dropout", "full", motion, 4.0,
                {**REAL, "hands": lambda t, ti=t_imp: 1 if ti - 0.35 < t < ti - 0.25 else 2}, None))
    motion, t_imp = sw(power=0.9)
    out.append(("reset", "full", motion, 4.0, {**REAL, "reset_at": t_imp - 0.3}, None))
    return out


def record_swing(name, mode, motion, duration, kw, tweak):
    kw = dict(kw)
    reset_at = kw.pop("reset_at", None)
    poses, frames, index, keep = [], [], {}, []
    for t, pose in GP.stream(motion, duration, **kw):
        if pose is None:
            frames.append([t, -1])
            continue
        if id(pose) not in index:
            if tweak is not None:
                tweak(pose, t, motion, kw.get("lag", 0.0))
            index[id(pose)] = len(poses)
            keep.append(pose)  # alive, so its id is never reused by a later pose
            poses.append(_store(pose))
        frames.append([t, index[id(pose)]])
    loaded = [_load(p) for p in poses]
    resets = [i for i, (t, _) in enumerate(frames) if reset_at is not None and abs(t - reset_at) < 0.5 / 60]
    det = SwingDetector(mode)
    stages, power, angle, events, guide = [], [], [], [], []
    last_guide = object()
    for i, (t, idx) in enumerate(frames):
        if i in resets:
            det.reset()
        ev = det.update(loaded[idx] if idx >= 0 else None, t)
        if ev is not None:
            events.append([i, ev.t, ev.power, ev.quality, ev.side, ev.speed, ev.practice])
        stages.append(STAGE_CODE[det.stage.value])
        power.append(det.power)
        angle.append(det.swing_angle)
        g = det.guidance()
        if g != last_guide:
            guide.append([i, g])
            last_guide = g
    return {"name": name, "mode": mode, "poses": poses, "frames": frames, "resets": resets,
            "stages": "".join(stages), "power": power, "angle": angle, "events": events, "guide": guide,
            "end": {"address": list(det.address) if det.address else None, "too_close": det.too_close,
                    "hold": det.hold_progress, "body_frame": list(det.body_frame) if det.body_frame else None}}


# --- physics -------------------------------------------------------------------


def flight_record(f: P.Flight, hole) -> dict:
    out = {"state": f.state, "x": f.x, "y": f.y, "z": f.z, "t": f.t, "bounces": f.bounces,
           "landing": list(f.landing) if f.landing else None, "apex": f.apex, "carry": f.carry,
           "trees": sorted(f.hit_trees), "n_path": len(f.path), "path": [list(p) for p in f.path[::15]],
           "lie": f.lie}
    if f.state == "water":
        out["water_entry"] = list(f.water_entry)
        out["drop"] = list(P.drop_point(hole, f))
    return out


def physics_cases(course):
    rng = random.Random(2024)
    cases = []
    clubs = [c for c in P.CLUBS if not c.putter]
    for hi, hole in enumerate(course.holes):
        tx, ty = hole.tee
        to_pin = math.atan2(hole.pin[0] - tx, hole.pin[1] - ty)
        for k in range(16):
            club = clubs[k % len(clubs)] if k < len(clubs) else rng.choice(clubs)
            cases.append({"hole": hi, "start": [tx, ty], "club": club.key,
                          "power": round(rng.uniform(0.2, 1.15), 3),
                          "heading": to_pin + rng.uniform(-0.35, 0.35), "lie": "tee",
                          "quality": rng.choice((1.0, 1.0, 0.6, 0.0)), "curve": rng.choice((0.0, 0.0, 0.7, -0.45)),
                          "wind": [rng.uniform(-4, 4), rng.uniform(-4, 4)]})
        # From along the hole's centre line, in whatever lie that is.
        for k in range(8):
            a, b = hole.path[0], hole.path[-1]
            u = rng.uniform(0.3, 0.85)
            sx, sy = a[0] + (b[0] - a[0]) * u + rng.uniform(-15, 15), a[1] + (b[1] - a[1]) * u
            lie = hole.lie_at(sx, sy)
            if lie in ("water", "ob"):
                continue
            club = rng.choice(clubs[1:])
            cases.append({"hole": hi, "start": [sx, sy], "club": club.key, "power": round(rng.uniform(0.3, 1.0), 3),
                          "heading": math.atan2(hole.pin[0] - sx, hole.pin[1] - sy) + rng.uniform(-0.1, 0.1),
                          "lie": lie, "quality": 1.0, "curve": 0.0, "wind": [rng.uniform(-3, 3), rng.uniform(-3, 3)]})
        # Putts round the green, slopes and all.
        px, py = hole.pin
        for k in range(8):
            ang = rng.uniform(0, 2 * math.pi)
            d = rng.uniform(1.0, 9.0)
            sx, sy = px + d * math.sin(ang), py + d * math.cos(ang)
            if hole.lie_at(sx, sy) != "green":
                continue
            head = math.atan2(px - sx, py - sy) + rng.uniform(-0.06, 0.06)
            cases.append({"hole": hi, "start": [sx, sy], "club": "PT", "power": round(rng.uniform(0.08, 0.7), 3),
                          "heading": head, "lie": "green", "quality": 1.0, "curve": 0.0, "wind": [0.0, 0.0]})
    # Hand-picked: the water carry on hole 2, OB off hole 1's tee, a tree, a holed putt.
    h2 = course.holes[1]
    cases.append({"hole": 1, "start": list(h2.tee), "club": "9I", "power": 0.55, "heading": math.atan2(-6, 148),
                  "lie": "tee", "quality": 1.0, "curve": 0.0, "wind": [0.0, 0.0]})
    cases.append({"hole": 0, "start": list(course.holes[0].tee), "club": "1W", "power": 1.0,
                  "heading": math.radians(-60), "lie": "tee", "quality": 1.0, "curve": 0.0, "wind": [0.0, 0.0]})
    h1 = course.holes[0]
    tr = h1.trees[1]
    head = math.atan2(tr.x, tr.y)
    for p in range(4, 21, 4):
        cases.append({"hole": 0, "start": list(h1.tee), "club": "9I", "power": p / 20, "heading": head, "lie": "tee",
                      "quality": 1.0, "curve": 0.0, "wind": [0.0, 0.0]})
    for p in range(10, 60, 7):
        cases.append({"hole": 0, "start": [h1.pin[0], h1.pin[1] - 3.0], "club": "PT", "power": p / 100,
                      "heading": 0.0, "lie": "green", "quality": 1.0, "curve": 0.0, "wind": [0.0, 0.0]})
    return cases


def run_physics(course):
    out = []
    for c in physics_cases(course):
        hole = course.holes[c["hole"]]
        shot = P.Shot(P.club_by_key(c["club"]), c["power"], c["heading"], c["lie"], quality=c["quality"],
                      curve=c["curve"], wind=tuple(c["wind"]))
        f = P.simulate(hole, tuple(c["start"]), shot)
        out.append({**c, "result": flight_record(f, hole)})
    # Previews from each tee with every club.
    previews = []
    for hi, hole in enumerate(course.holes):
        for club in P.CLUBS:
            head = math.atan2(hole.pin[0] - hole.tee[0], hole.pin[1] - hole.tee[1])
            f = P.preview(hole, hole.tee, club, head, "tee")
            previews.append({"hole": hi, "club": club.key, "heading": head, "landing": list(f.landing) if f.landing else None,
                             "apex": f.apex, "n_path": len(f.path), "state": f.state, "x": f.x, "y": f.y})
    # The same shot stepped with uneven frame times must end where simulate() does.
    hole = course.holes[0]
    shot = P.Shot(P.CLUBS[2], 0.93, 0.05, "fairway", wind=(1.5, -2.0), curve=0.3)
    f = P.Flight(hole, (0.0, 0.0), shot)
    dts = [1 / 30, 1 / 144, 0.05, 0.0123, 1 / 60]
    i = 0
    while not f.done:
        f.step(dts[i % len(dts)])
        i += 1
    stepped = {"dts": dts, "club": "5I", "power": 0.93, "heading": 0.05, "lie": "fairway", "wind": [1.5, -2.0],
               "curve": 0.3, "steps": i, "result": flight_record(f, hole)}
    return out, previews, stepped


# --- course ----------------------------------------------------------------------


def course_tables(course):
    rng = random.Random(7)
    holes = []
    for hole in course.holes:
        a = hole.land
        step = 3.0
        xs = np.arange(a.x0 - 12, a.x1 + 12, step)
        ys = np.arange(a.y0 - 12, a.y1 + 12, step)
        code = {"ob": "o", "water": "w", "bunker": "b", "green": "g", "tee": "t", "fairway": "f", "rough": "r"}
        grid = ["".join(code[hole.lie_at(float(x) + 0.37, float(y) + 0.21)] for x in xs) for y in ys]
        g = hole.green
        grads = []
        for _ in range(40):
            x, y = rng.uniform(g.x0, g.x1), rng.uniform(g.y0, g.y1)
            grads.append([x, y, *hole.green_gradient(x, y), hole.green_height(x, y)])
        aims = []
        for _ in range(30):
            x, y = rng.uniform(a.x0, a.x1), rng.uniform(a.y0, a.y1)
            reach = rng.choice([c.carry_m for c in P.CLUBS[:-1]]) * rng.choice((1.0, 0.8, 0.55))
            aims.append([x, y, reach, *hole.aim_point(x, y, reach)])
        holes.append({"grid": {"x0": float(xs[0]) + 0.37, "y0": float(ys[0]) + 0.21, "step": step, "rows": grid},
                      "gradient": grads, "aim": aims, "length": hole.length, "par": hole.par})
    return holes


# --- the game --------------------------------------------------------------------


def _game(seed=1):
    return GolfGame((1280, 720), art=stub_art(), seed=seed)


def aim_tables(course):
    rng = random.Random(11)
    rows = []
    for hi, hole in enumerate(course.holes):
        g = _game()
        g._start_hole(hi)
        spots = [hole.tee]
        for _ in range(14):
            a, b = hole.path[0], hole.path[-1]
            u = rng.uniform(0.1, 1.0)
            spots.append((a[0] + (b[0] - a[0]) * u + rng.uniform(-20, 20), a[1] + (b[1] - a[1]) * u + rng.uniform(-8, 8)))
        for x, y in spots:
            lie = hole.lie_at(x, y)
            if lie in ("water", "ob"):
                continue
            g.ball, g.lie = (x, y), lie
            g._setup_shot()
            row = {"hole": hi, "ball": [x, y], "lie": lie, "club": g.club.key, "heading": g.auto_heading,
                   "landing": list(g.preview.landing) if g.preview.landing else None,
                   "distance": g.distance_text(), "range": g.club_range_text(), "putt": g.putting}
            # Clubs up and down the bag from here (skipping the driver off the tee, the putter in sand).
            seq = []
            for step in (1, 1, 1, -1, -1, -1, -1, -1):
                g.change_club(step)
                seq.append(g.club.key)
            row["changes"] = seq
            rows.append(row)
    return rows


def scoring_tables():
    pts = [[s, par, pu, hole_points(s, par, pu)] for par in (3, 4, 5) for s in range(1, 11) for pu in (False, True)]
    names = [[s, par, result_name(s, par)] for par in (3, 4, 5) for s in range(1, 11)]
    return {"points": pts, "names": names, "to_par": [[d, to_par_text(d)] for d in range(-5, 8)]}


def _ev(power, quality=1.0):
    return SwingEvent(t=0.0, power=power, quality=quality, side=1, speed=9.0, practice=False)


def _plan(g):
    """A simple planner (deterministic): the aim step and power whose shot ends
    nearest the pin, from the game's own simulator, avoiding water and OB."""
    best = None
    step = math.radians(1.0) if g.putting else math.radians(2.0)
    powers = [p / 100 for p in range(8, 101, 4)]
    for off in (-2, -1, 0, 1, 2):
        for power in powers:
            f = P.simulate(g.hole, g.ball, P.Shot(g.club, power, g.auto_heading + off * step, g.lie, wind=g.wind))
            cost = -1.0 if f.state == "holed" else g.hole.distance_to_pin(f.x, f.y) + (
                1e4 if f.state in ("water", "ob") else 0.0)
            if best is None or cost < best[0]:
                best = (cost, off, power)
    return best[1], best[2]


def scripted_rounds():
    """Whole holes played shot by shot with each hole's wind given (Python's
    random draws can't be reproduced in the browser). Some shots are scripted
    (power, quality, aim steps, club steps) to visit hazards, thin strikes and
    club changes; the rest come from _plan. The browser replays the recorded
    shots and must reach the same state after every one."""
    scripts = [
        # (hole, wind mph, wind dir, scripted opening shots)
        (0, 4.0, 2.1, [(1.0, 1.0, 0, 0)]),
        (0, 0.0, 0.0, [(1.0, 1.0, -20, 0), (0.9, 0.2, 0, 0), (0.7, 1.0, 3, 1)]),
        (1, 6.0, 4.0, [(0.55, 1.0, 0, 0)]),
        (1, 2.0, 0.5, [(0.08, 1.0, 0, 0)] * 9),
        (2, 9.0, 3.3, [(1.0, 1.0, 0, 0), (1.0, 1.0, 0, 0), (0.8, 0.5, 0, 0)]),
        (2, 3.0, 5.5, []),
    ]
    out = []
    for hole, mph, wdir, opening in scripts:
        g = _game(seed=5)
        g._start_hole(hole)
        g._wind_mph, g._wind_dir = mph, wdir
        while g.phase is Phase.INTRO:
            g.update(ControlState(), 1 / 60)
        log, shots = [], []
        while g.phase is Phase.AIM and len(shots) < 12:
            if len(shots) < len(opening):
                power, quality, aim, club = opening[len(shots)]
            else:
                quality, club = 1.0, 0
                aim, power = _plan(g)
            shots.append([power, quality, aim, club])
            for _ in range(abs(aim)):
                g.adjust_aim(1 if aim > 0 else -1)
            for _ in range(abs(club)):
                g.change_club(1 if club > 0 else -1)
            g._hit(_ev(power, quality))
            while g.phase is Phase.FLIGHT:
                g.update(ControlState(), 1 / 60)
            entry = {"club": g.flight.shot.club.key, "heading": g.flight.shot.heading,
                     "phase": g.phase.value, "strokes": g.strokes, "ball": list(g.ball), "lie": g.lie,
                     "message": g.message.text if g.message else None, "flight_state": g.flight.state,
                     "flight_t": g.flight.t}
            if g.phase is Phase.HOLED:
                entry["card"] = [[r.strokes, r.points, r.picked_up, r.done] for r in g.card]
                entry["holed_to"] = [g.holed_to.x, g.holed_to.y, g.holed_to.z, g.holed_to.yaw, g.holed_to.pitch]
                entry["celebrate"] = list(g.celebrate_spot)
            log.append(entry)
            while g.phase is Phase.RESULT:
                g.update(ControlState(), 1 / 60)
        out.append({"hole": hole, "wind": [mph, wdir], "shots": shots, "log": log, "score": g.score,
                    "to_par": g.to_par, "phase": g.phase.value})
    return out


def holed_views(course):
    rows = []
    for hole in course.holes:
        for hd in (-0.4, 0.0, 0.3, 2.0):
            cam, spot = holed_camera(hole.pin, hd)
            rows.append([list(hole.pin), hd, [cam.x, cam.y, cam.z, cam.yaw, cam.pitch], list(spot)])
    return rows


def build() -> dict:
    course = load_course()
    shots, previews, stepped = run_physics(course)
    return {
        "about": "made by tools/make_golf_parity_fixture.py; replayed by web/tests/golf_parity.test.mjs",
        "swing": [record_swing(*s) for s in swing_scenarios()],
        "physics": {"shots": shots, "previews": previews, "stepped": stepped},
        "course": course_tables(course),
        "game": {"aim": aim_tables(course), "scoring": scoring_tables(), "rounds": scripted_rounds(),
                 "holed": holed_views(course)},
    }


def main() -> int:
    data = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    sw = data["swing"]
    print(f"swing: {len(sw)} streams, {sum(len(s['frames']) for s in sw)} frames, "
          f"{sum(len(s['events']) for s in sw)} events")
    print(f"physics: {len(data['physics']['shots'])} shots, "
          f"{sorted({s['result']['state'] for s in data['physics']['shots']})}")
    print(f"game: {len(data['game']['aim'])} aim rows, {len(data['game']['rounds'])} rounds "
          f"{[r['score'] for r in data['game']['rounds']]}")
    print(f"wrote {OUT.relative_to(REPO)} ({OUT.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
