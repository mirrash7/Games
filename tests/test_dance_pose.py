"""Dance core: pose angles, forward kinematics, measuring players, similarity, charts."""

from __future__ import annotations

import json

import pytest

from kpapp.config import KP
from kpapp.game.dance.chart import ChartError, idle_groove, load_song, parse_moves
from kpapp.game.dance.pose import (
    FULL_BODY, LEGS_WEIGHT, REST, SQUAT_BEND, angle_diff, forward, keypoints_for, leg_target, measure, mirror_pose,
    UNSEEN_LEGS, normalize_pose, pelvis, similarity,
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


# --- the lower body (docs/DANCE.md section 3a) ---

STEP_R = {"lf": [0.7, 0], "rf": [0.7, 0], "shift": 0.32, "squat": 0.12, "ra": [95, 100], "la": [20, -88]}
KNEE_R = {"lf": [0.37, 0], "rf": [0.2, 0.62], "shift": -0.235, "la": [138, 145], "ra": [40, -45]}
PLIE = {"lf": [0.72, 0], "rf": [0.72, 0], "squat": 0.6, "la": [90, 90], "ra": [90, 90]}
CROSS = {"lf": [-0.06, 0], "rf": [-0.06, 0], "shift": 0.04, "squat": 0.25, "la": [168, 175], "ra": [168, 175]}
LEG_POSES = [STEP_R, mirror_pose(STEP_R), KNEE_R, PLIE, CROSS, {"lf": [0.37, 0], "rf": [1.25, 0.5], "shift": 0.06}]


@pytest.mark.parametrize("pose", LEG_POSES)
@pytest.mark.parametrize("mirrored", [True, False])
def test_the_camera_reads_the_coachs_lower_body_back(pose, mirrored):
    """The judge compares features both sides compute with one function
    (read_lower): from the coach's forward() joints and from a player's
    keypoints. A perfect player, whole body in view, reads back exactly."""
    want = leg_target(normalize_pose(pose)).lower
    m = measure(keypoints_for(pose, mirrored=mirrored, **FULL_BODY), mirrored=mirrored)
    got = m.lower
    assert m.sees_feet
    for k in ("stance", "weight", "lift"):
        assert getattr(got, k) == pytest.approx(getattr(want, k), abs=0.01), k
    for s in "lr":
        assert got.bend[s] == pytest.approx(want.bend[s], abs=0.5)
    assert similarity(pose, m)[0] > 0.99


def test_shift_and_squat_mean_what_the_camera_reads():
    """`shift` is the hips over the feet: the weight the camera reads is
    shift minus the feet's midpoint. `squat` bends the supporting knee
    SQUAT_BEND * squat degrees. Feet stay on the floor."""
    for shift in (-0.4, 0.0, 0.25):
        p = normalize_pose({"lf": [0.5, 0], "rf": [0.9, 0], "shift": shift})
        assert leg_target(p).lower.weight == pytest.approx(shift - (0.9 - 0.5) / 2, abs=1e-9)
        assert pelvis(p)[0] == pytest.approx(shift)
    for squat in (0.2, 0.5, 0.8):
        lt = leg_target(normalize_pose({"squat": squat}))
        assert lt.lower.bend["l"] == pytest.approx(SQUAT_BEND * squat, abs=0.5)
        j = forward({"squat": squat})
        assert j["l_ankle"][1] == pytest.approx(forward({})["l_ankle"][1]), "the feet stay planted"
    lifted = forward(KNEE_R)
    assert lifted["r_ankle"][1] < lifted["l_ankle"][1] - 0.6, "a lifted foot is off the floor"


def test_legs_weigh_half_of_a_leg_led_move_and_standing_still_misses_them():
    m_step = measure(keypoints_for(STEP_R, **FULL_BODY))
    score, parts = similarity(STEP_R, m_step)
    assert score > 0.99 and parts["legs"] > 0.99
    assert LEGS_WEIGHT == 2.0, "as much as both arms together"
    arms_only = dict(STEP_R, lf=REST["lf"], rf=REST["rf"], shift=0.0, squat=0.0)
    score, parts = similarity(STEP_R, measure(keypoints_for(arms_only, **FULL_BODY)))
    assert parts["legs"] < 0.15 and score < 0.6, "perfect arms, legs standing still: an OK at best"
    for pose in LEG_POSES:
        assert leg_target(normalize_pose(pose)).active
        still = dict(normalize_pose(pose), lf=REST["lf"], rf=REST["rf"], shift=0.0, squat=0.0)
        assert similarity(pose, measure(keypoints_for(still, **FULL_BODY)))[1]["legs"] <= 0.5, pose


def test_an_arms_move_doesnt_judge_the_legs():
    """A target whose lower body is just standing (or nearly) leaves the legs
    free: a player bouncing or stepping to the music isn't marked down."""
    up = {"la": [170, 175], "ra": [170, 175]}
    assert not leg_target(normalize_pose(up)).active
    score, parts = similarity(up, measure(keypoints_for(dict(up, **{k: STEP_R[k] for k in ("lf", "rf", "shift")}),
                                                        **FULL_BODY)))
    assert "legs" not in parts and score > 0.99


@pytest.mark.parametrize("visible", [{"shoulder", "elbow", "wrist", "hip"}, {"shoulder", "elbow", "wrist", "hip", "knee"}])
def test_legs_out_of_view_get_half_credit(visible):
    """Never punish tracking loss, never reward it: without both whole legs in
    view a leg move's legs earn UNSEEN_LEGS (the game says SHOW YOUR FEET).
    Perfect arms then make a SUPER on NORMAL, never an X and never a PERFECT;
    with the legs not judged at all, hiding the feet scored a perfect 10000
    while being seen standing still scored ~6000."""
    m = measure(keypoints_for(STEP_R, visible=visible, **FULL_BODY))
    assert m.lower is None and not m.sees_feet
    score, parts = similarity(STEP_R, m)
    assert parts["legs"] == UNSEEN_LEGS and 0.74 <= score < 0.9
    m = measure(keypoints_for(STEP_R))  # close: the ankles are below the frame
    assert m.lower is None and 0.74 <= similarity(STEP_R, m)[0] < 0.9
    assert similarity(STEP_R, m, None)[0] > 0.99, "None leaves them out: what the arms say"
    still = similarity(STEP_R, measure(keypoints_for({**STEP_R, "lf": REST["lf"], "rf": REST["rf"],
                                                      "shift": 0.0, "squat": 0.0}, **FULL_BODY)))[0]
    assert still < similarity(STEP_R, m)[0], "seen standing still scores below hidden legs"


def test_one_leg_alone_is_not_judged():
    """A lifted foot in view with the other below the frame says too little."""
    kp = keypoints_for(KNEE_R, centre_px=(640, 330), shoulder_px=150)
    m = measure(kp)
    assert m.lower is None


def test_old_leg_angles_still_load_as_feet():
    """Charts written before feet (and the importer's drafts) give [thigh, shin]
    leg angles; they load as the same feet."""
    p = normalize_pose({"rl": [70, 0]})
    assert "rl" not in p and p["rf"][1] > 0.5, "a knee lift"
    assert normalize_pose({"ll": [3, 2], "rl": [3, 2]})["lf"] == pytest.approx([0.373, 0.0], abs=0.005)
    with pytest.raises(ValueError, match="not both"):
        normalize_pose({"rl": [70, 0], "rf": [0.4, 0]})


@pytest.mark.parametrize("bad", [{"lf": [3.0, 0]}, {"rf": [0.4, -0.1]}, {"shift": 2.0}, {"lf": [0.4]}])
def test_feet_out_of_range_are_rejected(bad):
    with pytest.raises(ValueError):
        normalize_pose(bad)


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
