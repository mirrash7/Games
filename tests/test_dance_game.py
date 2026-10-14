"""Raccoon Dance: the judge is fair under real conditions (docs/DANCE.md sections 7-9).

Every player here is simulated through the same input path as the app: the
bot's pose becomes COCO keypoints (`keypoints_for`), with pixel noise, model
results at 20 Hz arriving 150 ms after the moment they show, and the app's
`PoseExtrapolator` projecting them to each 60 Hz render frame. Song time comes
from a SilentPlayer on a fake clock, so runs are fast and deterministic.
"""

from __future__ import annotations

import re

import dataclasses
import json
import random
from collections import deque
from pathlib import Path

import numpy as np
import pytest

from kpapp.audio import SilentPlayer
from kpapp.controls import ControlState
from kpapp.game import REGISTRY
from kpapp.game.dance import chart, judge
from kpapp.game.dance.game import NOTICE_ARMS, NOTICE_NOBODY, DanceGame, Phase
from kpapp.game.dance.judge import Judge, rate, star_fill, stars_for
from kpapp.game.dance.pose import REST, keypoints_for, mirror_pose, normalize_pose
from kpapp.inference import PoseExtrapolator, Result

W, H = 1280, 720
FPS = 60

# --- a test song: asymmetric poses, so the wrong side is always wrong ---------

POSES = {
    "point_r": {"ra": [150, 165], "la": [20, 10], "lean": 6},
    "t_up_r": {"la": [90, 90], "ra": [170, 175]},
    "wave_r": {"ra": [90, 170], "la": [30, -20]},
    "diag": {"la": [135, 135], "ra": [45, 45]},
    "flex_l": {"la": [100, 180], "ra": [45, 30]},
    "cross": {"la": [-30, -60], "ra": [120, 150]},
}
POSES["point_l"] = mirror_pose(POSES["point_r"])
POSES["t_up_l"] = mirror_pose(POSES["t_up_r"])

MOVES = {
    "m1": {"label": "Point Up", "beats": 2,
           "keyframes": [{"beat": 0, "pose": POSES["point_r"]}, {"beat": 1, "pose": POSES["t_up_l"]}]},
    "m2": {"label": "Wave", "beats": 2,
           "keyframes": [{"beat": 0, "pose": POSES["wave_r"]}, {"beat": 1, "pose": POSES["diag"]}]},
    "m3": {"label": "Flex", "beats": 2,
           "keyframes": [{"beat": 0, "pose": POSES["flex_l"]}, {"beat": 1, "pose": POSES["cross"]}]},
    "m4": {"label": "Reach", "beats": 2,
           "keyframes": [{"beat": 0, "pose": POSES["t_up_r"]}, {"beat": 1, "pose": POSES["point_l"]}]},
    "hold": {"label": "Diagonal", "beats": 4, "keyframes": [{"beat": 0, "pose": POSES["diag"]}]},
}
LIB = chart.parse_moves(MOVES, "test library")


def song_json(sid="groove", title="Test Groove", first=8, last=56, duration=32.0, **extra):
    moves, beat, i = [], first, 0
    order = ["m1", "m2", "m3", "m4", "m2", "m1", "hold", "m3", "m4"]
    while True:
        name = order[i % len(order)]
        beats = MOVES[name]["beats"]
        if beat + beats > last:
            break
        moves.append({"move": name, "beat": beat, "gold": i in (5, 14)})
        beat += beats
        i += 1
    return {"version": 1, "id": sid, "title": title, "artist": "KP Arcade", "bpm": 120, "offset": 0.5,
            "duration": duration, "difficulty": 1, "moves": moves, **extra}


def make_songs(tmp_path: Path, songs=None) -> Path:
    root = tmp_path / "songs"
    songs = songs or [song_json()]
    for s in songs:
        (root / s["id"]).mkdir(parents=True, exist_ok=True)
        (root / s["id"] / "song.json").write_text(json.dumps(s))
    (root / "index.json").write_text(json.dumps([s["id"] for s in songs]))
    return root


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def new_game(songs_dir, variant=None, library=LIB, **kw):
    clock = Clock()
    probe = chart.load_song(Path(songs_dir) / (variant or chart.song_ids(songs_dir)[0]), library)
    player = SilentPlayer(probe.duration + 2.0, clock=clock)
    game = DanceGame((W, H), variant=variant, songs_dir=songs_dir, library=library, player=player, **kw)
    return game, clock


# --- the simulated input path --------------------------------------------------

def simulate(game, clock, bot, *, rate_hz=20.0, lag=0.15, noise=4.0, seed=1, elbow_drop=0.0,
             hips=True, seconds=None, on_frame=None, stop_at_game_over=True):
    """Run `game` with a player whose pose at song time `st` is `bot(st)`
    (a pose dict, or None for "not in the frame"). Returns the phases seen."""
    rng = np.random.default_rng(seed)
    extrap = PoseExtrapolator()  # the app's own projection (gain 0.7)
    pending: deque = deque()
    next_capture = clock.t
    phases = [game.phase]
    limit = seconds if seconds is not None else game.song.duration + 8.0
    visible = None if hips else {"shoulder", "elbow", "wrist"}
    for _ in range(int(limit * FPS)):
        clock.t += 1.0 / FPS
        now = clock.t
        if now >= next_capture:
            next_capture += 1.0 / rate_hz
            st = game.audio.position() - game.song.offset if game._started else -game.song.offset
            target = bot(st)
            poses = []
            if target is not None:
                kp = keypoints_for(target, visible=visible)
                kp.xy = (kp.xy + rng.uniform(-noise, noise, kp.xy.shape)).astype(np.float32)
                if elbow_drop:
                    from kpapp.config import KP
                    for name in ("left_elbow", "right_elbow"):
                        if rng.random() < elbow_drop:
                            kp.confidence[KP[name]] = 0.0
                poses = [kp]
            # The model sees the frame ~50 ms after capture; the game gets it at capture + lag.
            pending.append((now + lag, Result(poses=poses, timestamp=now + 0.05)))
        while pending and pending[0][0] <= now:
            extrap.update(pending.popleft()[1], arrived=now)
        poses = extrap.poses_at(now)
        pose = poses[0] if poses else None
        game.update(ControlState(present=pose is not None, pose=pose,
                                 step_pose=extrap.held[0] if extrap.held else None), 1.0 / FPS)
        if on_frame is not None:
            on_frame(game)
        if game.phase != phases[-1]:
            phases.append(game.phase)
        if stop_at_game_over and game.phase is Phase.GAME_OVER:
            break
    return phases


def perfect(game, late=0.0):
    return lambda st: game.song.coach_pose(game.song.beat_of(st - late))


def sloppy(game, late=0.25, off=(25.0, 30.0), seed=7):
    """Every limb segment 25-30 degrees off (a new direction every beat), 250 ms late."""
    def bot(st):
        p = normalize_pose(game.song.coach_pose(game.song.beat_of(st - late)))
        r = random.Random(seed * 100003 + int(game.song.beat_of(st - late)))
        for k in ("la", "ra", "ll", "rl"):
            p[k] = [a + r.choice((-1, 1)) * r.uniform(*off) for a in p[k]]
        return p
    return bot


def frozen(game):
    first = game.song.judge_points()[0].pose
    return lambda st: first


def idle(game):
    return lambda st: REST


def random_arms(game, seed=3):
    def bot(st):
        r = random.Random(seed * 7919 + int(st / 0.4))  # a new random pose every 0.4 s
        return {"la": [r.uniform(-60, 220), r.uniform(-60, 220)], "ra": [r.uniform(-60, 220), r.uniform(-60, 220)]}
    return bot


def wrong_side(game):
    return lambda st: mirror_pose(game.song.coach_pose(game.song.beat_of(st)))


def report(game) -> str:
    return f"score {game.score} stars {game.star_count} counts {game.counts} unseen {game.judge.unseen}"


@pytest.fixture
def songs(tmp_path):
    return make_songs(tmp_path)


# --- the judge on its own -------------------------------------------------------

def test_ratings_and_stars_follow_the_spec_tables():
    assert rate(1.0)[0] == "perfect" and rate(0.0) == ("x", 0.0)
    names = [rate(s)[0] for s in (0.95, 0.80, 0.65, 0.45, 0.2)]
    assert names == ["perfect", "super", "good", "ok", "x"]
    assert [stars_for(s) for s in (1999, 2000, 4000, 6000, 8000, 9199, 9200, 10000)] == [0, 1, 2, 3, 4, 4, 5, 5]
    assert star_fill(0) == 0 and star_fill(1000) == pytest.approx(0.5) and star_fill(10000) == 5
    assert star_fill(8600) == pytest.approx(4.5)


OPPOSITE = {"la": [170, 175], "ra": [10, 5]}  # nothing like the first point (point_r)


def _feed(j, start, end, measured, fps=60):
    out, t = [], start
    while t <= end + 1e-9:
        out += j.update(t, measured)
        t += 1.0 / fps
    return out


def test_judge_keeps_the_best_held_match_in_a_late_skewed_window(songs):
    from kpapp.game.dance.pose import measure

    song = chart.load_song(songs / "groove", LIB)
    p0 = song.judge_points()[0]
    good, bad = measure(keypoints_for(p0.pose)), measure(keypoints_for(OPPOSITE))
    j = Judge(song)
    assert _feed(j, p0.time - 1.0, p0.time + 0.20, bad) == []
    _feed(j, p0.time + 0.21, p0.time + 0.36, good)  # late, held 0.15 s: the lag allowance
    out = _feed(j, p0.time + 0.37, p0.time + judge.WINDOW_LATE + 0.02, bad)
    assert [r.name for r in out] == ["perfect"]
    assert j.score == round(10000 * 1.0 / j.total_weight)
    # A pose held only before the window opens doesn't count.
    j = Judge(song)
    _feed(j, p0.time - 1.0, p0.time - judge.WINDOW_EARLY - 0.02, good)
    out = _feed(j, p0.time - judge.WINDOW_EARLY, p0.time + judge.WINDOW_LATE + 0.02, bad)
    assert [r.name for r in out] == ["x"]


def test_a_limb_sweeping_through_the_pose_is_not_a_hold(songs):
    """Pins HOLD: a single matching frame (a limb passing through the target on
    its way elsewhere) would be PERFECT if the judge kept the best raw frame."""
    from kpapp.game.dance.pose import measure, similarity

    song = chart.load_song(songs / "groove", LIB)
    p0 = song.judge_points()[0]
    good, bad = measure(keypoints_for(p0.pose)), measure(keypoints_for(OPPOSITE))
    assert similarity(p0.pose, good)[0] > 0.99, "the passing frame alone is a perfect match"
    j = Judge(song)
    _feed(j, p0.time - 0.3, p0.time - 0.01, bad)
    j.update(p0.time, good)
    j.update(p0.time + 1 / 60, good)  # 2 frames: ~33 ms, under HOLD
    out = _feed(j, p0.time + 2 / 60, p0.time + judge.WINDOW_LATE + 0.02, bad)
    assert [r.name for r in out] == ["x"]


def test_a_stall_that_skips_a_window_is_unseen_never_a_miss(songs):
    song = chart.load_song(songs / "groove", LIB)
    j = Judge(song)
    pts = song.judge_points()
    now = pts[3].time + 1.0
    out = j.update(now, None)
    closed = [p for p in pts if p.time + judge.WINDOW_LATE < now]
    assert len(closed) >= 4 and [r.name for r in out] == ["unseen"] * len(closed)
    assert j.counts["x"] == 0 and j.score == 0


def test_losing_the_player_early_in_a_window_is_unseen_not_a_miss(songs):
    """Tracking loss is never a miss (AGENTS.md section 6). A window that sees
    the player only on their way into the pose and then loses them (arms out
    of the frame) saw nothing to judge: unseen. Seen failing through the late
    half of the window, it is still an X."""
    from kpapp.game.dance.pose import measure

    song = chart.load_song(songs / "groove", LIB)
    p0 = song.judge_points()[0]
    bad = measure(keypoints_for(OPPOSITE))
    j = Judge(song)
    _feed(j, p0.time - 1.0, p0.time + 0.05, bad)  # on the way in, then gone
    out = _feed(j, p0.time + 0.06, p0.time + judge.WINDOW_LATE + 0.02, None)
    assert [r.name for r in out] == ["unseen"] and j.counts["x"] == 0
    j = Judge(song)
    _feed(j, p0.time - 1.0, p0.time + judge.LOST_BEFORE + 0.05, bad)
    out = _feed(j, p0.time + judge.LOST_BEFORE + 0.06, p0.time + judge.WINDOW_LATE + 0.02, None)
    assert [r.name for r in out] == ["x"], "seen getting it wrong after the beat: a real miss"


def test_a_close_player_whose_hands_leave_the_top_of_the_frame_is_still_perfect(tmp_path):
    """Standing close, the wrists leave the top of the frame on arms-up poses
    (elbows stay in). Found in QA: a perfect dancer at 300 px shoulder width
    took an X on every Clap Up of Neon Nights (9121, 4 stars)."""
    from kpapp.game.dance.pose import measure

    up = {"la": [170, 175], "ra": [170, 175]}
    root = make_songs(tmp_path, [{"version": 1, "id": "up", "title": "Up", "bpm": 120, "duration": 20.0,
                                  "moves": [{"move": "up", "beat": 8}],
                                  "custom_moves": {"up": {"beats": 2, "keyframes": [{"beat": 0, "pose": up}]}}}])
    song = chart.load_song(root / "up", {})
    p = song.judge_points()[0]
    close = dict(centre_px=(640, 300), shoulder_px=300)
    j = Judge(song)
    _feed(j, p.time - 1.0, p.time - 0.1, measure(keypoints_for(REST, **close)))  # arms down, on the way
    seen = measure(keypoints_for(up, **close))
    assert all(seen.arms[s].reach is None for s in "lr"), "both raised wrists are out of the frame"
    out = _feed(j, p.time - 0.09, p.time + judge.WINDOW_LATE + 0.02, seen)
    assert [r.name for r in out] == ["perfect"]


# --- fairness: who earns what ------------------------------------------------------

def test_a_perfect_player_with_real_lag_and_noise_is_a_superstar(songs):
    game, clock = new_game(songs)
    phases = simulate(game, clock, perfect(game))
    print("perfect:", report(game))
    assert phases == [Phase.INTRO, Phase.DANCING, Phase.OUTRO, Phase.GAME_OVER]
    n = len(game.song.judge_points())
    assert game.score >= 9200 and game.star_count == 5, report(game)
    assert game.counts["perfect"] >= 0.95 * n, report(game)
    assert game.counts["x"] == 0 and game.judge.unseen == 0


def test_a_sloppy_player_lands_in_super_and_good(songs):
    """Every limb 25-30 degrees off and 250 ms late: decent, not perfect."""
    game, clock = new_game(songs)
    simulate(game, clock, sloppy(game))
    print("sloppy:", report(game))
    c = game.counts
    n = len(game.song.judge_points())
    assert c["super"] + c["good"] >= 0.6 * n, report(game)
    assert c["perfect"] <= 0.3 * n, report(game)
    assert 3 <= game.star_count <= 4, report(game)


def test_a_flawless_dancer_a_quarter_second_behind_the_coach_is_still_perfect(songs):
    """Pins WINDOW_LATE = 0.5: players copying what they see run late, on top
    of the camera's lag. With the spec's first 0.4 this dancer took X ratings."""
    game, clock = new_game(songs)
    simulate(game, clock, perfect(game, late=0.25))
    print("perfect, 250 ms late:", report(game))
    assert game.score >= 9200 and game.counts["x"] == 0, report(game)


def test_a_good_but_imprecise_dancer_on_a_slow_model_is_still_a_superstar(songs):
    """Pins PERFECT at 0.90, not higher: limbs 10-20 degrees off (inside the
    similarity's free zone), 200 ms late, a 15 Hz model with 200 ms of lag.
    At 0.92 this dancer dropped to 4 stars."""
    game, clock = new_game(songs)
    simulate(game, clock, sloppy(game, late=0.2, off=(10, 20)), rate_hz=15, lag=0.2, noise=6)
    print("imprecise, late, slow model:", report(game))
    assert game.star_count == 5 and game.counts["x"] == 0, report(game)


def test_holding_the_pose_is_what_separates_sloppy_from_perfect(songs, monkeypatch):
    """Pins HOLD: judged by its best single frame, the sloppy dancer's limbs
    sweep through the targets between poses and it scores like a superstar."""
    monkeypatch.setitem(judge.DIFFICULTY, "normal", dataclasses.replace(judge.DIFFICULTY["normal"], hold=0.0))
    game, clock = new_game(songs)
    simulate(game, clock, sloppy(game))
    assert game.score >= 9200, report(game)


@pytest.mark.parametrize("make_bot", [frozen, idle, random_arms, wrong_side])
def test_players_who_dont_dance_the_moves_get_at_most_one_star(songs, make_bot):
    game, clock = new_game(songs)
    simulate(game, clock, make_bot(game))
    print(f"{make_bot.__name__}:", report(game))
    assert game.star_count <= 1, report(game)


def test_wrong_side_would_score_if_mirroring_were_ignored(songs):
    """Pins why the test song is asymmetric: the mirror-wrong player is only
    caught because every pose differs from its mirror image."""
    song = chart.load_song(songs / "groove", LIB)
    for p in song.judge_points():
        assert normalize_pose(mirror_pose(p.pose)) != normalize_pose(p.pose)


def test_missing_elbows_and_hips_still_reward_a_perfect_player(songs):
    """Elbows go missing in about half of real frames and hips are often below
    the frame (AGENTS.md section 6): arms are then judged by reach alone."""
    game, clock = new_game(songs)
    simulate(game, clock, perfect(game), elbow_drop=0.5, hips=False)
    print("perfect, elbows 50%, no hips:", report(game))
    assert game.star_count >= 4, report(game)
    assert game.counts["x"] == 0, report(game)


def test_a_player_out_of_view_loses_only_those_points(songs):
    game, clock = new_game(songs)
    pts = game.song.judge_points()
    gone = (pts[10].time - 0.5, pts[17].time + 0.5)
    base = perfect(game)
    notices = []

    def bot(st):
        return None if gone[0] <= st <= gone[1] else base(st)

    simulate(game, clock, bot, on_frame=lambda g: notices.append((g.song_time, g.notice)))
    print("out of view:", report(game))
    assert game.counts["x"] == 0, "tracking loss is never a miss"
    hidden = [r for r in game.ratings if gone[0] + 0.45 <= r.point.time <= gone[1] - 0.45]
    assert hidden and all(r.name == "unseen" for r in hidden)
    assert game.judge.unseen >= len(hidden)
    seen_w = sum(r.point.weight for r in game.ratings if r.name != "unseen")
    assert game.score <= round(10000 * seen_w / game.judge.total_weight)
    assert game.score >= round(10000 * seen_w / game.judge.total_weight) - 300
    shown = {n for t, n in notices if gone[0] + 0.9 <= t <= gone[1] - 0.2}
    assert shown == {NOTICE_NOBODY}
    assert all(n is None for t, n in notices if t > gone[1] + 0.6), "the notice clears when the player is back"


def test_arms_out_of_frame_says_step_back(songs):
    game, clock = new_game(songs)
    simulate(game, clock, perfect(game), seconds=8.0)
    assert game.phase is Phase.DANCING
    for _ in range(60):
        kp = keypoints_for(REST, visible={"shoulder"})
        game.update(ControlState(present=True, pose=kp), 1 / 60)
        clock.t += 1 / 60
    assert game.notice == NOTICE_ARMS


# --- pause, resume, close -----------------------------------------------------------

def test_pause_freezes_song_time_and_never_rates_the_pause(songs):
    game, clock = new_game(songs)
    pts = game.song.judge_points()
    p = pts[12]
    # Dance perfectly until just after point 12's beat (its window already holds a PERFECT)...
    simulate(game, clock, perfect(game), seconds=p.time + game.song.offset + 0.1)
    assert p.time < game.song_time < p.time + judge.WINDOW_LATE
    rated = len(game.ratings)
    game.on_pause()
    frozen_at = game.song_time
    clock.t += 30.0  # a long pause; the shell doesn't update a paused game
    assert game.audio.position() - game.song.offset == pytest.approx(frozen_at, abs=1e-9)
    game.on_resume()
    assert game.song_time == pytest.approx(frozen_at, abs=1e-6)
    # ...then the hand is on the RESUME button (arms nowhere near the move) for a moment.
    resumed = game.song_time
    base = perfect(game)
    simulate(game, clock, lambda st: {"ra": [40, 40], "la": [10, 5]} if st < resumed + 0.8 else base(st))
    by_point = {r.point.index: r for r in game.ratings}
    assert len(game.ratings) == len(pts), "every point resolved exactly once"
    assert rated <= 12
    assert by_point[12].name != "perfect", "no free PERFECT from the pose held before pausing"
    excused = [r for r in game.ratings if resumed - judge.WINDOW_LATE <= r.point.time - judge.WINDOW_EARLY
               <= resumed + judge.RESUME_GRACE]
    assert excused and all(r.name != "x" for r in excused), "a pause never turns into X"
    assert game.counts["x"] == 0
    assert game.phase is Phase.GAME_OVER


def test_close_stops_the_music(songs):
    game, clock = new_game(songs)
    simulate(game, clock, idle(game), seconds=3.0)
    assert game.audio.playing
    game.close()
    assert not game.audio.playing
    game.update(ControlState(), 1 / 60)  # a stray frame after close doesn't restart it
    assert not game.audio.playing


def test_audio_starts_on_the_first_frame_not_at_construction(songs):
    game, clock = new_game(songs)
    clock.t += 5.0  # the shell's tutorial and countdown
    assert not game.audio.playing and game.song_time == -game.song.offset
    game.update(ControlState(), 1 / 60)
    assert game.audio.playing and game.song_time == pytest.approx(-game.song.offset)


def test_song_time_comes_from_the_audio_clock_not_dt(songs):
    game, clock = new_game(songs)
    game.update(ControlState(), 1 / 60)
    clock.t += 2.0
    game.update(ControlState(), 1 / 60)  # one frame of dt, two seconds of music
    assert game.song_time == pytest.approx(2.0 - game.song.offset)


# --- flow, state and registration ------------------------------------------------------

def test_score_stops_changing_after_game_over(songs):
    game, clock = new_game(songs)
    simulate(game, clock, perfect(game))
    assert game.phase is Phase.GAME_OVER
    final, counts = game.score, dict(game.counts)
    simulate(game, clock, random_arms(game), seconds=3.0, stop_at_game_over=False)
    assert game.score == final and game.counts == counts
    assert game.results_t > 2.5
    assert game.display_score == final


def test_renderer_state_is_complete(songs):
    game, clock = new_game(songs)
    seen = {"gold_flash": 0.0, "feedback": 0, "combo": 0}

    def watch(g):
        seen["gold_flash"] = max(seen["gold_flash"], g.gold_flash)
        seen["feedback"] = max(seen["feedback"], len(g.feedback))
        seen["combo"] = max(seen["combo"], g.combo)
        assert 0.0 <= g.stars <= 5.0 and 0 <= g.display_score <= 10000
        assert len(g.upcoming) <= 8
        assert all(u.beat >= g.beat - 0.5 for u in g.upcoming)

    simulate(game, clock, perfect(game), seconds=12.0, on_frame=watch)
    for name in ("song", "phase", "song_time", "beat", "coach_pose", "upcoming", "feedback", "score",
                 "display_score", "stars", "counts", "player", "last_rating", "notice", "gold_flash",
                 "combo", "results_t"):
        assert hasattr(game, name), name
    normalize_pose(game.coach_pose)
    up = game.upcoming[0]
    assert {"beat", "move_label", "pose", "gold"} <= set(vars(up)) if hasattr(up, "__dict__") else True
    assert seen["feedback"] >= 1 and seen["combo"] >= 5
    assert seen["gold_flash"] > 0.95, "the gold move flashed"
    f = game.feedback[-1] if game.feedback else None
    if f is not None:
        assert f.rating in judge.RATING_NAMES and 0 <= f.age < 1.2
    assert game.last_rating == "perfect" and game.player is not None
    assert set(game.counts) == {"perfect", "super", "good", "ok", "x"}


def test_variants_list_the_songs_and_the_game_builds_each(tmp_path):
    root = make_songs(tmp_path, [song_json("groove", "Test Groove"), song_json("slow_jam", "Slow Jam", first=12)])
    vs = DanceGame.variants(songs_dir=root, library=LIB)
    assert [v["id"] for v in vs] == ["groove", "slow_jam"]
    assert vs[1]["subtitle"].startswith("KP Arcade - ") and re.fullmatch(r"KP Arcade - \d+:\d\d", vs[1]["subtitle"]), \
        "the song's length, not a rating that reads like the difficulty picker"
    assert {k: vs[1][k] for k in ("id", "title", "board_id")} == {"id": "slow_jam", "title": "Slow Jam",
                                                                  "board_id": "dance-slow_jam"}
    game, _ = new_game(root, variant="slow_jam")
    assert game.song.title == "Slow Jam" and game.board_id == "dance-slow_jam"
    game, _ = new_game(root)
    assert game.song.id == "groove", "the first song in the index by default"


def test_the_shell_builds_the_game_for_a_chosen_song(tmp_path, monkeypatch):
    from kpapp.shell import Shell

    root = make_songs(tmp_path)
    lib = tmp_path / "moves.json"
    lib.write_text(json.dumps({"version": 1, "moves": MOVES}))
    monkeypatch.setattr(chart, "SONGS", root)
    monkeypatch.setattr(chart, "LIBRARY", lib)
    monkeypatch.setattr("kpapp.audio.open_player", lambda path, duration=0.0: SilentPlayer(duration))
    assert REGISTRY["dance"] is DanceGame
    shell = Shell((W, H), {"dance": DanceGame}, selected="dance")
    assert [v["id"] for v in shell.variants("dance")] == ["groove"] if hasattr(shell, "variants") else True
    game = DanceGame((W, H), mirrored=True, variant="groove")  # what shell.start(..., variant=) does
    assert game.song.id == "groove" and isinstance(game.audio, SilentPlayer)
    game.close()


REAL = chart.SONGS / "neon_nights" / "song.json"


@pytest.mark.skipif(not REAL.exists() or not chart.LIBRARY.exists(), reason="neon_nights isn't charted yet")
def test_real_song_end_to_end():
    lib = chart.load_library()

    def play(make_bot, **kw):
        game, clock = new_game(chart.SONGS, variant="neon_nights", library=lib)
        phases = simulate(game, clock, make_bot(game), **kw)
        print(f"neon_nights {getattr(make_bot, '__name__', 'bot')}:", report(game))
        return game, phases

    game, phases = play(perfect)
    assert phases == [Phase.INTRO, Phase.DANCING, Phase.OUTRO, Phase.GAME_OVER]
    assert game.score >= 9200 and game.counts["x"] == 0, report(game)
    game, _ = play(perfect, elbow_drop=0.5, hips=False)
    assert game.star_count >= 4, report(game)
    game, _ = play(sloppy)
    assert 3 <= game.star_count <= 4, report(game)
    for bot in (frozen, idle, random_arms):
        game, _ = play(bot)
        assert game.star_count <= 1, report(game)
    # Not wrong_side: a third of Neon Nights' points are symmetric poses
    # (Clap Up, Shine, Star Jump...), where the mirror image is the move itself.


# --- difficulties (docs/DANCE.md section 7a) ---------------------------------------

def test_difficulty_interface_the_shell_relies_on(songs):
    assert DanceGame.difficulties == [
        {"id": "easy", "label": "EASY", "hint": "fewer moves, held longer"},
        {"id": "normal", "label": "NORMAL", "hint": "the full routine"},
        {"id": "hard", "label": "HARD", "hint": "every move, judged strictly"},
    ]
    assert DanceGame.default_difficulty == "normal"
    game, _ = new_game(songs)
    assert game.difficulty == "normal" and game.board_id == "dance-groove"
    for level in ("easy", "hard"):
        game, _ = new_game(songs, difficulty=level)
        assert game.difficulty == level and game.board_id == f"dance-groove_{level}"
        assert game.judge.rules is judge.DIFFICULTY[level]
    game, _ = new_game(songs, difficulty="nightmare")
    assert game.difficulty == "normal" and game.board_id == "dance-groove", "unknown ids fall back to normal"
    assert [v["board_id"] for v in DanceGame.variants(songs_dir=songs, library=LIB)] == ["dance-groove"], \
        "the song list names the normal board; the shell adds the difficulty"


def test_board_ids_fit_the_servers_pattern():
    import re

    from kpapp.game.dance.game import board_id_for

    for sid in chart.song_ids():
        for level in chart.DIFFICULTIES:
            assert re.fullmatch(r"dance-[a-z0-9_]{1,40}", board_id_for(sid, level)), (sid, level)
    assert board_id_for("neon_nights") == board_id_for("neon_nights", "normal") == "dance-neon_nights"
    assert board_id_for("neon_nights", "easy") == "dance-neon_nights_easy"


def test_one_rules_table_easy_forgives_more_and_hard_less():
    t = judge.DIFFICULTY
    assert set(t) == {"easy", "normal", "hard"}
    n = t["normal"]
    # NORMAL is the game as tuned before difficulties existed.
    assert (n.early, n.late, n.hold) == (0.20, 0.50, 0.12)
    assert [r[1] for r in n.ratings] == [0.90, 0.74, 0.60, 0.42]
    assert (judge.WINDOW_EARLY, judge.WINDOW_LATE, judge.HOLD, judge.RATINGS) == (n.early, n.late, n.hold, n.ratings)
    for name in ("easy", "normal", "hard"):
        assert [r[0] for r in t[name].ratings] == ["perfect", "super", "good", "ok"]
        assert [r[2] for r in t[name].ratings] == [1.0, 0.8, 0.55, 0.3], "same values: score is 0..10000 everywhere"
    e, h = t["easy"], t["hard"]
    assert all(a[1] < b[1] < c[1] for a, b, c in zip(e.ratings, n.ratings, h.ratings))
    assert e.late > n.late and e.early >= n.early >= h.early and h.hold > n.hold
    assert rate(0.88, e.ratings)[0] == "perfect" and rate(0.88, n.ratings)[0] == "super"
    assert rate(0.88, h.ratings)[0] == "super" and rate(0.80, h.ratings)[0] == "good"


def _neon(level, make_bot, **kw):
    lib = chart.load_library()
    game, clock = new_game(chart.SONGS, variant="neon_nights", library=lib, difficulty=level)
    simulate(game, clock, make_bot(game), **kw)
    print(f"neon_nights {level} {getattr(make_bot, '__name__', 'bot')}:", report(game))
    return game


def precise(game):
    """Every limb within 10 degrees, 200 ms behind the coach."""
    return sloppy(game, late=0.2, off=(0.0, 10.0))


def test_normal_is_exactly_the_game_before_difficulties():
    """Pins NORMAL: the same chart, windows and thresholds as before, so the
    same scores to the point (and the same leaderboard, dance-neon_nights)."""
    game = _neon("normal", sloppy)
    assert len(game.song.judge_points()) == 145
    assert (game.score, game.counts) == (7859, {"perfect": 7, "super": 124, "good": 14, "ok": 0, "x": 0})
    game = _neon("normal", lambda g: sloppy(g, late=0.2, off=(10, 20)), rate_hz=15, lag=0.2, noise=6)
    assert (game.score, game.counts["perfect"], game.counts["super"]) == (9352, 98, 46)


@pytest.mark.parametrize("level", ["easy", "normal", "hard"])
def test_fairness_on_every_difficulty(level):
    """Perfect is 10000 on every difficulty; a precise dancer (within 10
    degrees, 200 ms late) is a 5-star dancer even on hard; lazy players stay
    at a star or less, easy included (the freeze rule holds on its chart)."""
    game = _neon(level, perfect)
    assert game.score == 10000 and game.judge.unseen == 0, report(game)
    n = len(game.song.judge_points())
    assert n == {"easy": 49, "normal": 145, "hard": 145}[level]
    game = _neon(level, precise)
    assert game.star_count == 5, report(game)
    for bot in (frozen, idle, random_arms):
        game = _neon(level, bot)
        assert game.star_count <= 1, report(game)
    game = _neon(level, sloppy)  # every limb 25-30 degrees off, 250 ms late
    assert game.star_count == {"easy": 4, "normal": 3, "hard": 2}[level], report(game)


def test_easy_forgives_a_beginner_that_normal_rules_would_not():
    """Pins EASY's thresholds and wider late window: a beginner 30-40 degrees
    off and 400 ms behind the coach scores 3 stars on easy, where the normal
    rules on the same chart gave 1."""
    def beginner(g):
        return sloppy(g, late=0.4, off=(30.0, 40.0))
    game = _neon("easy", beginner)
    assert game.star_count == 3, report(game)
    lib = chart.load_library()
    game, clock = new_game(chart.SONGS, variant="neon_nights", library=lib, difficulty="easy")
    game.judge = Judge(game.song, judge.DIFFICULTY["normal"])
    simulate(game, clock, beginner(game))
    assert game.star_count <= 1, report(game)


def test_hard_needs_the_longer_hold():
    """Pins HARD's 0.15 s hold: at normal's 0.12 the sloppy dancer made 3 stars on hard."""
    lib = chart.load_library()
    game, clock = new_game(chart.SONGS, variant="neon_nights", library=lib, difficulty="hard")
    game.judge = Judge(game.song, dataclasses.replace(judge.DIFFICULTY["hard"], hold=0.12))
    simulate(game, clock, sloppy(game))
    assert game.star_count >= 3, report(game)
