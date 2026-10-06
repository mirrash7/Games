"""Hole in the Wall: matching behaviour and the round state machine."""

from __future__ import annotations

import numpy as np
import pytest

from kpapp.controls import ControlState
from kpapp.game.holeinwall.core import TargetPose, to_torso_space
from kpapp.game.holeinwall.game import HoleInWallGame, Phase, Rules
from kpapp.game.holeinwall.matching import HoleCache, evaluate
from kpapp.game.holeinwall.poses import TARGET_POSES
from kpapp.inference import Pose

STAR = np.array(
    [[.5,.08],[.47,.06],[.53,.06],[.44,.07],[.56,.07],[.38,.20],[.62,.20],
     [.30,.35],[.70,.35],[.24,.50],[.76,.50],[.42,.52],[.58,.52],
     [.40,.72],[.60,.72],[.39,.92],[.61,.92]], dtype=np.float32,
)
CONF = np.ones(17, dtype=np.float32)


def as_player(canon: np.ndarray, scale: float = 640.0, offset=(180.0, 25.0)) -> np.ndarray:
    """Canonical pose -> pixels, as a real (square-pixel) camera would see it."""
    return (canon * scale + np.asarray(offset, np.float32)).astype(np.float32)


@pytest.fixture
def star_target():
    return TargetPose("STAR", STAR), HoleCache()


def test_identical_pose_is_perfect(star_target):
    target, cache = star_target
    r = evaluate(as_player(STAR), CONF, target, cache)
    assert r.tracked and r.passed
    assert r.grade == "PERFECT"
    assert r.score == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize(
    "scale,offset",
    [(300.0, (500.0, 300.0)), (640.0, (180.0, 25.0)), (900.0, (-40.0, -90.0))],
)
def test_score_is_invariant_to_distance_and_position(star_target, scale, offset):
    """The same shape must score the same anywhere in frame, at any distance."""
    target, cache = star_target
    r = evaluate(as_player(STAR, scale, offset), CONF, target, cache)
    assert r.score == pytest.approx(1.0, abs=1e-3)


def test_wrong_pose_fails(star_target):
    target, cache = star_target
    arms_down = STAR.copy()
    arms_down[[7, 8, 9, 10]] = [[.38, .42], [.62, .42], [.40, .62], [.60, .62]]
    r = evaluate(as_player(arms_down), CONF, target, cache)
    assert not r.passed
    assert r.grade == "MISS"


def test_containment_alone_would_not_discriminate(star_target):
    """Guards the reason the metric blends joints with silhouette overlap.

    A compact pose sits entirely inside a spread-eagle opening, so containment
    saturates at 1.0 for an obviously wrong shape. If this ever becomes the
    pass condition on its own, the game stops being a game.
    """
    target, cache = star_target
    arms_down = STAR.copy()
    arms_down[[7, 8, 9, 10]] = [[.38, .42], [.62, .42], [.40, .62], [.60, .62]]
    r = evaluate(as_player(arms_down), CONF, target, cache)
    assert r.containment > 0.95
    assert not r.passed


def test_untracked_when_torso_hidden(star_target):
    target, cache = star_target
    conf = CONF.copy()
    conf[[11, 12]] = 0.0  # hips gone: no stable body frame
    assert not evaluate(as_player(STAR), conf, target, cache).tracked


def test_no_pose_is_untracked(star_target):
    target, cache = star_target
    assert not evaluate(None, None, target, cache).tracked


def test_every_library_pose_matches_itself():
    """Each authored hole must be passable by the pose that defines it."""
    cache = HoleCache()
    for target in TARGET_POSES:
        r = evaluate(as_player(target.keypoints), CONF, target, cache)
        assert r.passed, f"{target.name} scored {r.score:.3f} against itself"


def test_library_poses_are_distinguishable():
    """Different holes must not accept each other, or shapes stop mattering."""
    cache = HoleCache()
    confusable = []
    for target in TARGET_POSES:
        for other in TARGET_POSES:
            if other.name == target.name:
                continue
            if evaluate(as_player(other.keypoints), CONF, target, cache).passed:
                confusable.append((other.name, target.name))
    assert not confusable, f"poses accepted by the wrong hole: {confusable}"


# --- round lifecycle ---


def _play(game: HoleInWallGame, pose: Pose | None, seconds: float, dt: float = 1 / 35) -> None:
    for _ in range(int(seconds / dt)):
        game.update(ControlState(present=pose is not None, pose=pose), dt)


def _pose_for(target: TargetPose) -> Pose:
    k = target.keypoints - target.keypoints.mean(axis=0)
    return Pose(xy=(k * 430 + np.float32([640, 374])).astype(np.float32),
                confidence=CONF * 0.95, score=0.9)


def _wrong_pose_for(target: TargetPose) -> Pose:
    """A fully tracked player making a different, definitely-wrong shape."""
    other = next(p for p in TARGET_POSES if p.name != target.name)
    return _pose_for(other)


def test_round_advances_through_phases():
    game = HoleInWallGame((1280, 720))
    assert game.fb.phase is Phase.PREP
    _play(game, _pose_for(game.fb.target), game.rules.prep_time + 0.1)
    assert game.fb.phase is Phase.APPROACH
    _play(game, _pose_for(game.fb.target), game.rules.approach_time + 0.2)
    assert game.fb.phase is Phase.RESULT


def test_matching_the_shape_scores_and_builds_streak():
    game = HoleInWallGame((1280, 720))
    target = game.fb.target
    _play(game, _pose_for(target), game.rules.prep_time + game.rules.approach_time + 0.2)
    assert game.fb.result is not None and game.fb.result.passed
    assert game.score > 0
    assert game.streak == 1
    assert game.lives == game.rules.lives


def test_missing_the_shape_costs_a_life_and_resets_streak():
    game = HoleInWallGame((1280, 720))
    game.streak = 4
    wrong = _wrong_pose_for(game.fb.target)
    _play(game, wrong, game.rules.prep_time + game.rules.approach_time + 0.2)
    assert game.fb.phase is Phase.RESULT
    assert game.fb.result is not None and game.fb.result.tracked
    assert game.lives == game.rules.lives - 1
    assert game.streak == 0


def test_losing_tracking_costs_nothing():
    """A framing problem is not a miss: no life, no streak reset, replay.

    The player cannot fix being out of frame by posing better, so charging
    them for it reads as the game being broken.
    """
    game = HoleInWallGame((1280, 720))
    game.streak = 4
    _play(game, None, game.rules.prep_time + game.rules.approach_time + 0.2)
    assert game.fb.phase is Phase.RESULT
    assert game.fb.result is not None and not game.fb.result.tracked
    assert game.lives == game.rules.lives
    assert game.streak == 4


def test_untracked_result_explains_itself():
    """Whatever goes wrong, the player is told what to do about it."""
    game = HoleInWallGame((1280, 720))
    _play(game, None, game.rules.prep_time + 0.1)
    assert game.fb.fit.problem, "an untracked fit must carry a reason"

    partial = _pose_for(game.fb.target)
    partial.confidence = CONF * 0.95
    partial.confidence[[11, 12]] = 0.05  # hips out of frame
    game.update(ControlState(present=True, pose=partial), 1 / 35)
    assert "HIPS" in game.fb.fit.problem


def test_running_out_of_lives_ends_the_game():
    game = HoleInWallGame((1280, 720), Rules(lives=1))
    wrong = _wrong_pose_for(game.fb.target)
    _play(game, wrong,
          game.rules.prep_time + game.rules.approach_time + game.rules.result_time + 0.4)
    assert game.fb.phase is Phase.GAME_OVER


def test_game_over_is_terminal_until_reset():
    game = HoleInWallGame((1280, 720), Rules(lives=1))
    wrong = _wrong_pose_for(game.fb.target)
    _play(game, wrong,
          game.rules.prep_time + game.rules.approach_time + game.rules.result_time + 0.4)
    assert game.fb.phase is Phase.GAME_OVER
    _play(game, wrong, 5.0)
    assert game.fb.phase is Phase.GAME_OVER, "must not resume on its own"
    game.reset()
    assert game.fb.phase is Phase.PREP
    assert game.lives == game.rules.lives and game.score == 0


def test_approach_never_overshoots_on_a_stalled_frame():
    """A long dt must not skip the wall past the player."""
    game = HoleInWallGame((1280, 720))
    _play(game, None, game.rules.prep_time + 0.1)
    for _ in range(50):
        game.update(ControlState(present=False), 5.0)  # absurd stalls
        assert 0.0 <= game.fb.approach <= 1.0


def test_difficulty_ramps_but_stays_playable():
    game = HoleInWallGame((1280, 720))
    first = game._approach_duration()
    game.cleared = 100
    assert game._approach_duration() == game.rules.min_approach_time
    assert first > game.rules.min_approach_time


def test_offscreen_legs_do_not_penalise_a_correct_pose():
    """The camera not seeing your knees is not a mistake you made.

    Drawing the target with full legs while the player's legs are invisible
    collapsed silhouette overlap to 0.63 for a flawless pose.
    """
    cache = HoleCache()
    target = TargetPose("STAR", STAR)
    conf = CONF.copy()
    for i in (13, 14, 15, 16):  # knees and ankles out of frame
        conf[i] = 0.05
    r = evaluate(as_player(STAR), conf, target, cache)
    assert r.tracked
    assert r.iou > 0.9, f"visible-only overlap should stay high, got {r.iou:.2f}"
    assert r.grade == "PERFECT"


def test_shape_order_differs_between_games():
    """Hole in the Wall had the same fixed-seed bug: identical order every game."""
    def order(seed=None):
        g = HoleInWallGame((1280, 720), seed=seed)
        names = []
        for _ in range(8):
            names.append(g._pick_target().name)
        return tuple(names)
    assert len({order() for _ in range(5)}) > 1
    assert order(seed=9) == order(seed=9)
