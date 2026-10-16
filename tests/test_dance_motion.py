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
from kpapp.game.dance.pose import (LEAN_MIN, LEG_REST_MARGIN, REST, angle_diff, measure, normalize_pose,
                                   side_keypoints)
from kpapp.game.dance.render import CoachPainter
from kpapp.inference import Pose

TOLERANCE = 6.0  # degrees: the drawn coach vs the chart's pose on a judged beat
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
    """Degrees between the drawn coach (as the judge measures it) and `target`,
    for every part the judge would score."""
    geo = painter.layout(None, beat=beat, motion=M.coach_motion(song, beat))
    m = measure(drawn_keypoints(geo))
    t = normalize_pose(target)
    err = {}
    for s in "lr":
        a = m.arms[s]
        err[f"{s}a_upper"] = angle_diff(t[f"{s}a"][0], a.upper)
        err[f"{s}a_fore"] = angle_diff(t[f"{s}a"][1], a.fore)
        tt, ts = t[f"{s}l"]
        rt, rs = REST[f"{s}l"]
        if abs(tt - rt) > LEG_REST_MARGIN or abs(ts - rs) > LEG_REST_MARGIN:  # a judged leg
            err[f"{s}l_thigh"] = angle_diff(tt, m.legs[s].thigh)
            err[f"{s}l_shin"] = angle_diff(ts, m.legs[s].shin)
    if abs(t["lean"]) > LEAN_MIN:
        err["lean"] = abs(t["lean"] - m.lean)
    return err


@pytest.mark.parametrize("sid,level,song", SONGS, ids=[f"{s}-{lv}" for s, lv, _ in SONGS])
def test_the_drawn_coach_shows_the_judged_pose_on_its_beat(sid, level, song):
    """Secondary motion only adds life between hits and in parts the judge
    doesn't read: on every judged beat every judged angle of the *drawn* coach
    is within TOLERANCE of the chart's."""
    painter = CoachPainter(stub_art())
    worst = (0.0, None)
    for jp in song.judge_points():
        for part, e in drawn_errors(painter, song, jp.beat, jp.pose).items():
            if e > worst[0]:
                worst = (e, f"{jp.label} at beat {jp.beat:g}: {part}")
    assert worst[0] <= TOLERANCE, f"{sid} {level}: {worst[0]:.2f} degrees off ({worst[1]})"


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
                for k in ("la", "ra", "ll", "rl"):
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


def test_a_two_footed_move_hops_and_lands_before_its_beat():
    frames = [(0.0, normalize_pose({})), (2.0, normalize_pose({"ll": [28, 25], "rl": [28, 25]}))]
    lifts = [M.coach_motion(None, b, frames=frames)["lift"] for b in np.arange(1.0, 2.01, 0.02)]
    assert max(lifts) > 0.1
    assert abs(lifts[-1]) < 1e-9 and abs(lifts[-3]) < 1e-9, "on the floor at the hit"
    one = [(0.0, normalize_pose({})), (2.0, normalize_pose({"rl": [70, 0]}))]
    assert all(M.coach_motion(None, b, frames=one)["lift"] == 0.0 for b in np.arange(1.0, 2.0, 0.05)), \
        "a knee lift doesn't hop"


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
    for pose in ({"squat": 0.9, "ll": [30, 5], "rl": [30, 5]}, {"rl": [70, 0]}, {"ll": [28, 25], "rl": [28, 25]}):
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
