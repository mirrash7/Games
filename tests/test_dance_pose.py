"""Dance core: pose angles, forward kinematics, measuring players, similarity, charts."""

from __future__ import annotations

import json

import pytest

from kpapp.config import KP
from kpapp.game.dance.chart import ChartError, idle_groove, load_song, parse_moves
from kpapp.game.dance.pose import (
    angle_diff, forward, keypoints_for, measure, mirror_pose, normalize_pose, similarity,
)

ARMS_UP = {"la": [170, 175], "ra": [170, 175]}
T_POSE = {"la": [90, 90], "ra": [90, 90]}
DISCO = {"ra": [150, 165], "la": [20, 10], "lean": 10}
POSES = [{}, ARMS_UP, T_POSE, DISCO, {"ra": [-40, -70], "la": [120, 200]}, {"lean": -14, "la": [60, 150]}]


def test_angle_convention():
    j = forward(ARMS_UP)
    assert j["r_wrist"][1] < j["r_shoulder"][1] - 1.0, "180 is straight up"
    j = forward(T_POSE)
    assert j["r_wrist"][0] > j["r_shoulder"][0] + 1.0, "90 on the right arm points screen-right"
    assert j["l_wrist"][0] < j["l_shoulder"][0] - 1.0, "and screen-left on the left arm"
    j = forward({})
    assert j["r_wrist"][1] > j["r_shoulder"][1] + 0.9, "rest hangs down"
    assert j["head"][1] < j["neck"][1] < 0, "head on top, origin at the hips"


@pytest.mark.parametrize("pose", POSES)
@pytest.mark.parametrize("mirrored", [True, False])
def test_a_perfect_player_measures_back_to_the_target(pose, mirrored):
    m = measure(keypoints_for(pose, mirrored=mirrored), mirrored=mirrored)
    full = normalize_pose(pose)
    for s in "lr":
        assert angle_diff(m.arms[s].upper, full[f"{s}a"][0]) < 1.0
        assert angle_diff(m.arms[s].fore, full[f"{s}a"][1]) < 1.0
    assert abs(m.lean - full["lean"]) < 1.0
    score, _ = similarity(pose, m)
    assert score > 0.99


def test_screen_right_is_the_players_real_right_arm_on_a_mirrored_feed():
    """Mirroring: the coach's screen-right arm is copied with the player's real
    right arm, which a mirrored feed labels left_* (AGENTS.md section 6)."""
    kp = keypoints_for({"ra": [170, 175]}, mirrored=True)
    assert kp.xy[KP["left_wrist"]][1] < kp.xy[KP["left_shoulder"]][1], "left_* is the raised one"
    assert kp.xy[KP["left_wrist"]][0] > kp.xy[KP["right_wrist"]][0], "and it is on the screen's right"


def test_without_elbows_the_reach_direction_is_judged():
    kp = keypoints_for(DISCO, visible={"shoulder", "wrist", "hip"})
    m = measure(kp)
    assert m.arms["r"].upper is None and m.arms["r"].reach is not None
    assert similarity(DISCO, m)[0] > 0.97
    assert similarity(ARMS_UP, m)[0] < 0.6


def test_with_the_wrist_out_of_the_top_of_the_frame_the_upper_arm_is_judged():
    """A player standing close raises their hands out of the top of the frame
    on every arms-up move: elbows seen, wrists not. measure() reads the upper
    arm; similarity must use it, or the window keeps only the arm on its way
    up and a perfect Clap Up is an X (found in QA, neon_nights at 300 px SW)."""
    kp = keypoints_for(ARMS_UP, centre_px=(640, 300), shoulder_px=300)
    m = measure(kp)
    assert all(m.arms[s].reach is None and m.arms[s].upper is not None for s in "lr"), "wrists above the frame"
    assert m.sees_arms
    assert similarity(ARMS_UP, m)[0] > 0.97
    assert similarity(T_POSE, m)[0] < 0.2, "the upper arm still discriminates"
    assert similarity({}, m)[0] == 0.0


def test_similarity_discriminates():
    up = measure(keypoints_for(ARMS_UP))
    assert similarity(ARMS_UP, up)[0] > 0.99
    assert similarity({}, up)[0] == 0.0, "arms down vs arms up shares nothing"
    assert similarity(T_POSE, up)[0] < 0.2
    assert similarity(mirror_pose(DISCO), measure(keypoints_for(DISCO)))[0] < 0.5, "the wrong side is wrong"


def test_no_shoulders_or_no_arms_is_unseen_not_a_miss():
    assert measure(None) is None
    m = measure(keypoints_for(ARMS_UP, visible={"shoulder"}))
    assert similarity(ARMS_UP, m) == (None, {})


def test_hips_out_of_frame_still_measure_arms():
    kp = keypoints_for(DISCO, centre_px=(640, 520), shoulder_px=260)  # close: hips below the frame
    m = measure(kp)
    assert not m.has_hips
    assert similarity(DISCO, m)[0] > 0.9


def test_mirror_pose_round_trips():
    assert normalize_pose(mirror_pose(mirror_pose(DISCO))) == normalize_pose(DISCO)


def test_normalize_rejects_bad_keys():
    with pytest.raises(ValueError):
        normalize_pose({"arm": [1, 2]})
    with pytest.raises(ValueError):
        normalize_pose({"la": [1]})


def test_idle_groove_is_a_valid_pose():
    normalize_pose(idle_groove(3.3))


# --- charts ---

LIB = parse_moves({
    "point_r": {"label": "Point", "beats": 2, "keyframes": [{"beat": 0, "pose": DISCO}, {"beat": 1, "pose": {}}],
                "judge": [0]},
    "point_l": {"mirror_of": "point_r"},
    "hold": {"beats": 4, "keyframes": [{"beat": 0, "pose": ARMS_UP}]},
}, "test")


def write_song(tmp_path, moves, **extra):
    folder = tmp_path / "s"
    folder.mkdir(exist_ok=True)
    (folder / "song.json").write_text(json.dumps({"title": "T", "bpm": 120, "duration": 20, "moves": moves, **extra}))
    return folder


def test_song_timeline(tmp_path):
    song = load_song(write_song(tmp_path, [{"move": "point_r", "beat": 4}, {"move": "point_l", "beat": 6, "gold": True},
                                           {"move": "hold", "beat": 8}]), LIB)
    jp = song.judge_points()
    assert [p.beat for p in jp] == [4, 6, 8]
    assert jp[1].gold and jp[1].weight == 2.0
    assert jp[0].time == pytest.approx(2.0)
    assert normalize_pose(jp[1].pose) == normalize_pose(mirror_pose(DISCO))
    assert song.coach_pose(4.0) == jp[0].pose, "the coach hits the keyframe on its beat"
    assert song.coach_pose(4.1) == jp[0].pose, "and holds it"
    assert [p.beat for p in song.pictograms()] == [4, 6, 8]


def test_every_judged_keyframe_has_its_own_card_with_that_pose(tmp_path):
    """A move with two judged poses shows two cards, each on its own beat with
    the pose judged there, under the move's label (QA: "Point Up" was labelled
    over the move's first, horizontal pose and the raised arm had no card)."""
    lib = parse_moves({"two": {"label": "Point Up", "beats": 2, "pictogram": 1, "judge": [0, 1],
                               "keyframes": [{"beat": 0, "pose": T_POSE}, {"beat": 1, "pose": ARMS_UP}]}}, "t")
    song = load_song(write_song(tmp_path, [{"move": "two", "beat": 4, "gold": True}]), lib)
    cards = song.pictograms()
    assert [c.beat for c in cards] == [p.beat for p in song.judge_points()] == [4, 5]
    assert [normalize_pose(c.pose) for c in cards] == [normalize_pose(T_POSE), normalize_pose(ARMS_UP)]
    assert all(c.label == "Point Up" and c.gold for c in cards)


@pytest.mark.parametrize("moves, message", [
    ([{"move": "nope", "beat": 0}], "unknown move"),
    ([{"move": "hold", "beat": 0}, {"move": "hold", "beat": 2}], "overlaps"),
    ([{"move": "hold", "beat": 38}], "past the song's end"),
    ([{"beat": 1}], "needs 'move'"),
])
def test_bad_charts_fail_with_a_clear_message(tmp_path, moves, message):
    with pytest.raises(ChartError, match=message):
        load_song(write_song(tmp_path, moves), LIB)


def test_bad_custom_moves_fail(tmp_path):
    folder = write_song(tmp_path, [], custom_moves={"x": {"beats": 2, "keyframes": [{"beat": 3, "pose": {}}]}})
    with pytest.raises(ChartError, match="outside"):
        load_song(folder, LIB)
    folder = write_song(tmp_path, [], custom_moves={"x": {"beats": 2, "keyframes": [{"beat": 0, "pose": {"bad": 1}}]}})
    with pytest.raises(ChartError, match="unknown pose key"):
        load_song(folder, LIB)
