"""The coach's motion (motion.py) and its 3D layout: livelier between hits, and
never a different pose on a judged beat.

The player copies the drawn coach, and the judge scores them against the
chart's pose. So on every judged beat, a camera looking at the drawn coach
must read the chart's pose: these tests build COCO keypoints from the painter's
own joints (all secondary motion, twist, sway, squash and foreshortening
applied), run them through the judge's `measure`, and compare the angles with
the chart's, on every judge point of every song and difficulty.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from kpapp.config import KP
from kpapp.game.dance import chart
from kpapp.game.dance import motion as M
from kpapp.game.dance.art import stub_art
from kpapp.game.dance.pose import (LEAN_MIN, angle_diff, leg_target, measure, normalize_pose,
                                   side_keypoints)
from kpapp.game.dance.render import CoachPainter
from kpapp.inference import Pose

TOLERANCE = 6.0  # degrees: the drawn coach's arms and lean vs the chart's pose on a judged beat
BEND_TOLERANCE = 8.0  # degrees of knee bend (the groove's dip bends the knees ~4 on the beat)
SW_TOLERANCE = 0.08  # shoulder widths of stance, weight and lift (the groove's sway moves the hips 0.035)
SONG_IDS = json.loads((chart.SONGS / "index.json").read_text())


def _songs():
    lib = chart.load_library(chart.LIBRARY)
    for sid in SONG_IDS:
        base = chart.load_song(chart.SONGS / sid, lib)
        for level in chart.DIFFICULTIES:
            yield sid, level, base.for_difficulty(level)


SONGS = list(_songs())


def drawn_keypoints(geo) -> Pose:
    """What a camera sees of the drawn coach: its joints as COCO keypoints on a
    mirrored feed (screen-right limbs are left_*), everything visible."""
    sides = side_keypoints(True)
    xy = np.zeros((17, 2), np.float32)
    conf = np.zeros(17, np.float32)
    for name in ("nose", "left_eye", "right_eye"):
        xy[KP[name]] = geo.head_c
        conf[KP[name]] = 0.9
    for s in "lr":
        for part in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle"):
            xy[KP[f"{sides[s]}_{part}"]] = geo.j[f"{s}_{part}"]
            conf[KP[f"{sides[s]}_{part}"]] = 0.9
    return Pose(xy=xy, confidence=conf, score=0.9)


def drawn_errors(painter, song, beat: float, target: dict) -> dict[str, float]:
    """How far the drawn coach (as the judge measures it) is from `target`, as a
    fraction of each part's tolerance, for every part the judge would score:
    arm angles and lean, and the lower-body features the target asks for
    (stance, weight, lift, knee bends)."""
    geo = painter.layout(None, beat=beat, motion=M.coach_motion(song, beat))
    m = measure(drawn_keypoints(geo))
    t = normalize_pose(target)
    err = {}
    for s in "lr":
        a = m.arms[s]
        err[f"{s}a_upper"] = angle_diff(t[f"{s}a"][0], a.upper) / TOLERANCE
        err[f"{s}a_fore"] = angle_diff(t[f"{s}a"][1], a.fore) / TOLERANCE
    lt = leg_target(t)
    want, seen = lt.lower, m.lower
    if lt.step:
        err["stance"] = abs(want.stance - seen.stance) / SW_TOLERANCE
        err["weight"] = abs(want.weight - seen.weight) / SW_TOLERANCE
    if lt.lift:
        err["lift"] = abs(want.lift - seen.lift) / SW_TOLERANCE
    for s in lt.bend:
        err[f"{s}_bend"] = abs(want.bend[s] - seen.bend[s]) / BEND_TOLERANCE
    if abs(t["lean"]) > LEAN_MIN:
        err["lean"] = abs(t["lean"] - m.lean) / TOLERANCE
    return err


@pytest.mark.parametrize("sid,level,song", SONGS, ids=[f"{s}-{lv}" for s, lv, _ in SONGS])
def test_the_drawn_coach_shows_the_judged_pose_on_its_beat(sid, level, song):
    """Secondary motion only adds life between hits and in parts the judge
    doesn't read: on every judged beat every judged part of the *drawn* coach
    (arms, lean, and the legs' stance, weight, lift and knee bends) is within
    its tolerance of the chart's, however far the coach has stepped."""
    painter = CoachPainter(stub_art())
    worst = (0.0, None)
    for jp in song.judge_points():
        for part, e in drawn_errors(painter, song, jp.beat, jp.pose).items():
            if e > worst[0]:
                worst = (e, f"{jp.label} at beat {jp.beat:g}: {part}")
    assert worst[0] <= 1.0, f"{sid} {level}: {worst[0]:.2f} of the tolerance off ({worst[1]})"


def test_limbs_hit_every_keyframe_exactly():
    """The performed limbs (before the groove) are the keyframe itself on its
    beat: anticipation and overshoot are zero there, and the hold is still."""
    for _, level, song in SONGS:
        if level != "normal":
            continue
        for b, pose in song._frames:
            want = normalize_pose(pose)
            for at in (b, b + 0.05):
                got = M.performed_pose(song._frames, at)
                for k in ("la", "ra", "lf", "rf"):
                    assert got[k] == pytest.approx(want[k], abs=1e-9), (b, k)


def test_between_hits_the_coach_anticipates_overshoots_and_settles():
    """A single move, 90 -> 0 degrees: it first winds up past 90, then swings
    past 0 before settling on 0 exactly on the beat; the forearm trails the upper arm."""
    frames = [(0.0, normalize_pose({"ra": [90, 90]})), (4.0, normalize_pose({"ra": [0, 0]}))]
    start = 4.0 - chart.EASE_BEATS
    ups = [M.performed_pose(frames, start + u * chart.EASE_BEATS)["ra"] for u in np.linspace(0, 1, 101)]
    assert max(a for a, _ in ups[:40]) > 90.5, "anticipation: a wind-up the other way"
    assert min(a for a, _ in ups) < -3.0, "overshoot past the target"
    assert ups[-1] == [0.0, 0.0], "settled on the beat"
    assert any(f > a + 5 for a, f in ups[40:80]), "the forearm follows the upper arm"
    assert all(abs(M.performed_pose(frames, b)["ra"][0] - 90) < 1e-9 for b in (0.0, 1.0, 3.2)), "the hold is still"


def test_the_groove_keeps_moving_while_a_pose_is_held():
    """Held poses still breathe: the dip, squash, sway, shoulders, head and tail move on the beat."""
    frames = [(0.0, normalize_pose({"la": [90, 90], "ra": [90, 90]})), (8.0, normalize_pose({}))]
    ms = [M.coach_motion(None, b, energy=1.0, frames=frames) for b in np.arange(0.0, 2.0, 0.125)]
    for key in ("squash", "sway", "head_bob", "tail", "twist", "head_tilt"):
        vals = [m[key] for m in ms]
        assert max(vals) - min(vals) > 1e-3, key
    assert max(m["pose"]["squat"] for m in ms) > 0.03, "a dip on every beat"
    assert all(m["pose"]["la"] == [90.0, 90.0] for m in ms), "but the arms hold the pose"
    assert all(m["lift"] == 0.0 for m in ms)


def test_a_jump_hops_and_lands_before_its_beat():
    frames = [(0.0, normalize_pose({})), (2.0, normalize_pose({"lf": [0.9, 0], "rf": [0.9, 0]}))]
    lifts = [M.coach_motion(None, b, frames=frames)["lift"] for b in np.arange(1.0, 2.01, 0.02)]
    assert max(lifts) > 0.1
    assert abs(lifts[-1]) < 1e-9 and abs(lifts[-3]) < 1e-9, "on the floor at the hit"
    travel = [M.coach_motion(None, b, frames=frames)["travel"] for b in np.arange(1.0, 2.01, 0.05)]
    assert max(abs(t) for t in travel) < 1e-9, "a jump spreads the feet about the spot"
    one = [(0.0, normalize_pose({})), (2.0, normalize_pose({"rf": [0.2, 0.62], "shift": -0.17}))]
    assert all(M.coach_motion(None, b, frames=one)["lift"] == 0.0 for b in np.arange(1.0, 2.0, 0.05)), \
        "a knee lift doesn't hop"
    step = [(0.0, normalize_pose({})), (2.0, normalize_pose({"lf": [0.9, 0], "rf": [0.9, 0], "shift": 0.3}))]
    assert all(M.coach_motion(None, b, frames=step)["lift"] == 0.0 for b in np.arange(1.0, 2.0, 0.05)), \
        "spreading the feet with the weight going to one side is a step, not a jump"


def test_a_step_keeps_the_supporting_foot_planted_and_lifts_the_stepping_one():
    """Feet are relative to the body, so the coach travels: stepping right, the
    left foot stays where it is on the stage, the right one lifts on its way
    out, and the body moves over it. On the beat the pose is the chart's."""
    painter = CoachPainter(stub_art())
    a = normalize_pose({"lf": [0.22, 0], "rf": [0.22, 0], "shift": -0.1})  # feet together, weight left
    b = normalize_pose({"lf": [0.7, 0], "rf": [0.7, 0], "shift": 0.32})  # a step right, weight over it
    frames = [(0.0, a), (2.0, b), (4.0, a)]
    assert M.support(a, b) == "l" and M.support(b, a) == "r"
    lefts, rights, ups = [], [], []
    for beat in np.arange(1.0, 2.001, 0.05):
        m = M.coach_motion(None, beat, energy=0.0, frames=frames)
        g = painter.layout(None, motion=m)
        lefts.append(g.j["l_ankle"][0])
        rights.append(g.j["r_ankle"][0])
        ups.append(m["pose"]["rf"][1])
    # It may slide a hair: each transition drifts the coach back towards its spot (RECENTRE).
    assert max(lefts) - min(lefts) <= M.RECENTRE_MAX * painter.S + 0.5, "the supporting foot stays put"
    assert rights[-1] - rights[0] > 0.8 * painter.S, "the stepping foot travels about a step"
    assert max(ups) > 0.1 and ups[-1] == 0.0, "it lifts on the way and lands on the beat"
    assert M.coach_motion(None, 2.0, energy=0.0, frames=frames)["travel"] > 0.4, "the body went with it"


def test_the_coach_stays_near_its_spot_on_every_song():
    """Steps travel, but phrases balance and every transition drifts back a
    little, so the coach never wanders off the stage (the lint's WANDER_MAX)."""
    for sid, level, song in SONGS:
        offs, _, _ = M.keyframe_travel(song._frames)
        assert max(abs(o) for o in offs) <= chart.WANDER_MAX, (sid, level)


def test_a_twist_turns_towards_an_arm_reaching_across_and_hides_the_far_arm():
    frames = [(0.0, normalize_pose({"la": [-30, -60], "ra": [150, 155]}))]
    m = M.coach_motion(None, 0.0, energy=0.0, frames=frames)
    assert m["twist"] > 15, "the screen-left arm reaches across: the chest turns to screen-right"
    assert m["depth"]["l"][1] > 0.5 and m["depth"]["r"][0] < 0, "the crossing arm is near, the other far"
    geo = CoachPainter(stub_art()).layout(None, motion=m)
    assert geo.behind == ("r",), "the far arm goes behind the jacket"
    flat = CoachPainter(stub_art()).layout(frames[0][1])
    assert flat.behind == ()
    span = abs(geo.j["r_shoulder"][0] - geo.j["l_shoulder"][0])
    assert span < abs(flat.j["r_shoulder"][0] - flat.j["l_shoulder"][0]) - 4, "the shoulders narrow as it turns"


def test_motion_is_a_pure_function_of_the_beat():
    song = SONGS[0][2]
    beats = [3.3, 48.0, 61.71, 100.2, 17.5]
    a = [M.coach_motion(song, b) for b in beats]
    b = [M.coach_motion(song, x) for x in reversed(beats)][::-1]
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_feet_stay_on_the_floor():
    """The lower foot is planted wherever the pose puts the hips (a squat bends
    the knees instead of sinking the feet), unless the coach hops."""
    painter = CoachPainter(stub_art())
    rest = painter.layout({})
    floor = max(rest.j["l_ankle"][1], rest.j["r_ankle"][1])
    for pose in ({"squat": 0.9, "lf": [0.7, 0], "rf": [0.7, 0]}, {"rf": [0.2, 0.62]}, {"lf": [0.9, 0], "rf": [0.9, 0]}):
        g = painter.layout(pose)
        assert max(g.j["l_ankle"][1], g.j["r_ankle"][1]) == pytest.approx(floor, abs=0.5), pose
    for b in np.arange(16.0, 40.0, 0.27):
        g = painter.layout(None, motion=M.coach_motion(SONGS[0][2], b))
        assert max(g.j["l_ankle"][1], g.j["r_ankle"][1]) <= floor + 0.5
        assert math.isfinite(g.head_c[0])


def test_the_browser_fixture_is_current(tmp_path, monkeypatch):
    """web/tests/fixtures/dance_motion.json pins motion.js to this module: it
    must be what tools/make_dance_motion_fixture.py writes today."""
    import importlib.util

    root = chart.SONGS.parents[2]
    spec = importlib.util.spec_from_file_location("_motion_fixture", root / "tools" / "make_dance_motion_fixture.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    committed = tool.OUT.read_text()
    monkeypatch.setattr(tool, "OUT", tmp_path / "dance_motion.json")
    tool.main()
    assert (tmp_path / "dance_motion.json").read_text() == committed, \
        "run: uv run python tools/make_dance_motion_fixture.py"
