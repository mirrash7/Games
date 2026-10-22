"""PlayerLock: the main player through a crowd (src/kpapp/player.py).

Crowds come from tests/crowd_poses.py: a player dancing or swinging in the
middle while people walk across in front (bigger, scoring higher) and behind
(smaller), at 15-30 Hz with jitter, ~4 px noise, elbows missing and a 3%
chance per result that the model misses anyone. Identity is checked against
the ground truth (which generated pose the lock handed out), not against the
lock's own bookkeeping.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

import crowd_poses as C
from kpapp import player as P
from kpapp.inference import PoseExtrapolator, Result
from kpapp.player import PlayerLock, body_of, most_likely_player

RATES = (15, 20, 30)
SEEDS = range(5)


def passers():
    return {
        "player": C.dancer(),
        "front": C.walker(-150, 3.7, 190, t0=1.0, score=0.95),  # nearer: bigger, higher score
        "behind": C.walker(1400, -3.7, 80, t0=2.0, score=0.9),
        "same": C.walker(-100, 3.7, 120, t0=4.0, score=0.93),  # just behind the player
        "front2": C.walker(1450, -3.0, 165, t0=5.0, score=0.97),
    }


def play(people, duration=8.0, rate=20, seed=0, **kw):
    lock = PlayerLock((C.W, C.H))
    return lock, C.run(lock, C.stream(people, duration, rate, seed, **kw))


def names(log, t0=-1.0, t1=1e9):
    return {n for t, n, _ in log if t0 <= t < t1}


# --- why it exists -------------------------------------------------------------


def test_the_highest_scoring_person_is_often_not_the_player():
    """The old rule (keep the top-scoring detection) hands the game to
    passers-by: they score higher than the player in much of this crowd."""
    frames = C.stream(passers(), 8.0, 20, 0)
    top = [dets[0][0] for _, dets in frames if dets]
    assert sum(n != "player" for n in top) / len(top) > 0.3


# --- passers-by ------------------------------------------------------------------


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("seed", SEEDS)
def test_passers_by_in_front_and_behind_never_get_the_game(rate, seed):
    lock, log = play(passers(), rate=rate, seed=seed)
    first = next(t for t, n, _ in log if n is not None)
    assert first < 0.15, "a lone central player locks on their first result"
    assert names(log) <= {"player", None}
    assert lock.switches == 0
    # Only occlusions and missed detections leave the game without a pose.
    assert sum(n is None for _, n, _ in log) / len(log) < 0.2


@pytest.mark.parametrize("rate", RATES)
def test_a_passer_hiding_the_player_for_half_a_second_is_held_through(rate):
    people = {"player": C.hidden(C.dancer(), 2.0, 2.5), "front": C.walker(-150, 3.7, 190, t0=1.2, score=0.95)}
    lock, log = play(people, rate=rate, seed=1)
    hidden = [(t, n, s) for t, n, s in log if 2.0 <= t < 2.5]
    assert all(n is None and s == "holding" for _, n, s in hidden), "no pose, never the passer"
    back = next(t for t, n, _ in log if t >= 2.5 and n == "player")
    assert back - 2.5 < 0.15, "the player is handed back on their first result"
    assert names(log) <= {"player", None} and lock.switches == 0


def test_size_alone_would_not_tell_the_passer_from_the_player():
    """Pinned reason for tracking everyone: while the player is hidden, the
    passer's detection sits inside the player's motion and size gates. It is
    the passer's own track that explains it."""
    people = {"player": C.hidden(C.dancer(step=0.0), 2.0, 2.5), "front": C.walker(-150, 3.7, 150, t0=1.4, score=0.95)}
    lock = PlayerLock((C.W, C.H))
    inside = 0
    for t, dets in C.stream(people, 4.0, 20, 2):
        if 2.0 <= t < 2.5 and lock.locked is not None:
            me = lock.locked
            for name, p in dets:
                b = body_of(p)
                d = np.hypot(b.ax - me.hx, b.ay - me.hy) / me.body.scale
                gate = min(P.GATE_MAX, P.GATE_BASE + P.GATE_RATE * (t - me.last_seen))
                if name == "front" and d <= gate and P.SIZE_MIN <= b.scale / me.body.scale <= P.SIZE_MAX:
                    inside += 1
        got = lock.update([p for _, p in dets], t)
        assert got is None or got is next(p for n, p in dets if n == "player")
    assert inside >= 2


def test_a_two_second_occlusion_releases_then_recognises_the_player_again():
    people = {"player": C.hidden(C.dancer(), 2.0, 4.0), "front": C.walker(-150, 2.5, 190, t0=1.0, score=0.95)}
    for rate in RATES:
        lock, log = play(people, rate=rate, seed=3)
        assert names(log) <= {"player", None}, "the walker is never acquired: they are walking"
        released = next(t for t, _, s in log if t > 2.0 and s == "searching")
        assert 3.1 <= released <= 3.35  # HOLD_TIME after the last sighting
        back = next(t for t, n, _ in log if t >= 4.0 and n == "player")
        assert back - 4.0 < 0.15
        assert lock.switches == 0, "the same person back in the same place is not a switch"


def test_someone_standing_in_front_for_longer_than_the_hold_gets_the_lock():
    """Pinned design: a long loss releases the lock, and a still, central
    person is the most likely player. The player gets it back when the
    occluder goes (and the lock on them expires)."""
    people = {"player": C.hidden(C.dancer(step=0.0), 2.0, 4.0), "s": C.stander(640, 190, t0=1.9, t1=4.0)}
    lock, log = play(people, rate=20, seed=0)
    assert "s" in names(log, 3.0, 4.0)
    assert names(log, 5.5) == {"player"} or names(log, 5.5) == {"player", None}


# --- the player moving ---------------------------------------------------------------


@pytest.mark.parametrize("rate", RATES)
def test_fast_dance_steps_keep_the_lock(rate):
    people = {"player": C.dancer(step=1.5, bpm=130), "w": C.walker(-150, 3.7, 160, t0=1.0),
              "b": C.walker(1400, -3.7, 100, t0=2.5)}
    for seed in SEEDS:
        lock, log = play(people, rate=rate, seed=seed)
        assert names(log) <= {"player", None} and lock.switches == 0
        assert all(s != "searching" for t, _, s in log if t > 0.15), "never released"


@pytest.mark.parametrize("rate", RATES)
def test_a_golf_swing_turning_the_shoulders_keeps_the_lock(rate):
    people = {"player": C.golfer(), "w": C.walker(-150, 3.7, 170, t0=1.0), "w2": C.walker(1400, -3.7, 140, t0=2.5)}
    for seed in SEEDS:
        lock, log = play(people, rate=rate, seed=seed)
        assert names(log) <= {"player", None} and lock.switches == 0
        assert all(s != "searching" for t, _, s in log if t > 0.15)


def test_noise_dropouts_and_missing_shoulders_dont_break_the_lock():
    lock, log = play({"player": C.dancer(), "w": C.walker(-150, 3.7, 170, t0=1.0)}, rate=15, seed=4,
                     noise=6.0, elbow_drop=0.7, miss=0.1)
    assert names(log) <= {"player", None} and lock.switches == 0
    # A detection whose shoulders are gone (a guessed size) still matches.
    lock = PlayerLock((C.W, C.H))
    p = C.body_pose(C.Look(640, 150, 130))
    assert lock.update([p], 0.0) is p
    q = C.body_pose(C.Look(650, 152, 130))
    q.confidence[[5, 6]] = 0.1
    assert not body_of(q).sure
    assert lock.update([q], 0.05) is q and lock.switches == 0


def test_a_teleport_is_lost_then_found_never_followed():
    lock = PlayerLock((C.W, C.H))
    a = C.body_pose(C.Look(640, 150, 130))
    lock.update([a], 0.0)
    far = C.body_pose(C.Look(640 + 0.4 * C.W, 150, 130))  # a tracking glitch, 40% of the width
    assert lock.update([far], 0.05) is None and lock.state == "holding"
    back = C.body_pose(C.Look(645, 150, 130))
    assert lock.update([back, far], 0.1) is back


# --- who gets it ---------------------------------------------------------------------


@pytest.mark.parametrize("rate", RATES)
def test_bigger_higher_scoring_people_at_the_edges_never_steal_the_lock(rate):
    people = {"player": C.dancer(), "l": C.stander(70, 170, 0.97), "r": C.stander(1210, 170, 0.97)}
    for seed in SEEDS:
        lock, log = play(people, rate=rate, seed=seed)
        assert next(n for _, n, _ in log if n is not None) == "player", "the central player wins the first pick"
        assert names(log) <= {"player", None} and lock.switches == 0


@pytest.mark.parametrize("rate", RATES)
def test_a_crowd_walking_past_at_the_start_doesnt_get_the_lock(rate):
    people = {"player": C.dancer(), "a": C.walker(100, 3.7, 150, score=0.95), "b": C.walker(1100, -3.7, 110, score=0.95)}
    for seed in SEEDS:
        lock, log = play(people, rate=rate, seed=seed)
        first = next(t for t, n, _ in log if n is not None)
        assert first <= 0.7  # SPEED_MIN_SPAN, plus a result or two
        assert names(log) <= {"player", None} and lock.switches == 0


def test_the_player_leaving_and_a_new_one_arriving():
    people = {"player": C.leaving(C.dancer(), 3.0), "new": C.walker(-150, 3.0, 140, t0=4.0, stop_x=640)}
    stop = 4.0 + (640 + 150) / (3.0 * 140)
    for rate in RATES:
        lock, log = play(people, 10.0, rate=rate, seed=1)
        gone = next(t for t, _, s in log if t > 3.0 and s == "searching")
        assert gone < 3.6, "walking off releases the lock, no hold"
        assert "new" not in names(log, 0, stop), "not while they are walking in"
        locked = next(t for t, n, _ in log if n == "new")
        assert locked - stop < 0.6
        assert names(log, locked) <= {"new", None} and lock.switches == 1


def test_a_far_bystander_locked_in_an_empty_booth_hands_over_to_the_player():
    people = {"bg": C.stander(300, 60, 0.9), "player": C.walker(1400, -3.0, 150, t0=2.0, stop_x=640)}
    stop = 2.0 + (1400 - 640) / (3.0 * 150)
    lock, log = play(people, 8.0, rate=20, seed=0)
    assert next(n for _, n, _ in log if n is not None) == "bg"
    took = next(t for t, n, _ in log if n == "player")
    assert stop + P.TAKEOVER_TIME <= took < stop + 2.2
    assert names(log, took) <= {"player", None}


def test_repick_passes_the_lock_to_the_next_person():
    lock = PlayerLock((C.W, C.H))
    a, b = C.body_pose(C.Look(600, 150, 140)), C.body_pose(C.Look(900, 150, 120))
    t = 0.0
    for _ in range(15):
        got = lock.update([a, b], t)
        t += 0.05
    assert got is a
    lock.repick()
    assert lock.update([a, b], t) is b and lock.switches == 1 and lock.changed
    assert lock.update([a, b], t + 0.05) is b and not lock.changed
    # Alone, a repick keeps the only person there.
    solo = PlayerLock((C.W, C.H))
    solo.update([a], 0.0)
    solo.repick()
    assert solo.update([a], 0.05) is a


def test_two_similar_people_still_lock_one():
    """Claims hold within KEEP_CLAIM, so near-equal candidates can't flip-flop forever."""
    lock = PlayerLock((C.W, C.H))
    rng = np.random.default_rng(0)
    t, got = 0.0, None
    for _ in range(20):
        a = C.body_pose(C.Look(520, 150, 130), rng, noise=4.0)
        b = C.body_pose(C.Look(760, 150, 130), rng, noise=4.0)
        got = lock.update([a, b], t)
        t += 0.05
    assert got is not None


def test_most_likely_player_prefers_big_and_central_over_score():
    centre = C.body_pose(C.Look(640, 150, 130, score=0.6))
    edge = C.body_pose(C.Look(100, 150, 150, score=0.99))
    far = C.body_pose(C.Look(700, 250, 60, score=0.99))
    assert most_likely_player([edge, far, centre], C.W) is centre
    assert most_likely_player([], C.W) is None


# --- wiring and cost -----------------------------------------------------------------


def test_others_and_state_for_the_debug_view():
    lock = PlayerLock((C.W, C.H))
    a, b = C.body_pose(C.Look(640, 150, 130)), C.body_pose(C.Look(200, 170, 90))
    assert lock.update([b, a], 0.0) is None and lock.state == "searching"
    for i in range(1, 15):
        lock.update([b, a], 0.05 * i)
    assert lock.player is a and lock.others == [b] and lock.state == "locked"
    lock.update([b], 0.75)
    assert lock.state == "holding" and lock.others == [b] and lock.held_for == pytest.approx(0.05)
    assert lock.player_box is not None


def test_extrapolator_forgets_motion_when_the_identity_changes():
    """A velocity measured between two different people would fling the skeleton."""
    ex = PoseExtrapolator()
    a, b = C.body_pose(C.Look(300, 150, 130)), C.body_pose(C.Look(900, 150, 130))
    ex.update(Result(poses=[a], timestamp=0.0), arrived=0.0)
    ex.forget()
    ex.update(Result(poses=[b], timestamp=0.05), arrived=0.05)
    assert np.allclose(ex.poses_at(0.1)[0].xy, b.xy)


def test_selection_costs_a_few_hundredths_of_a_millisecond():
    people = {"player": C.dancer()}
    for k in range(7):
        people[f"p{k}"] = C.stander(80 + 160 * k, 60 + 15 * k) if k % 2 else C.walker(-150 + 200 * k, 1.0, 90 + 10 * k)
    frames = C.stream(people, 10.0, 30, 0, miss=0.0, occlude=False)
    assert np.mean([len(d) for _, d in frames]) >= 5
    lock = PlayerLock((C.W, C.H))
    spent = []
    for t, dets in frames:
        poses = [p for _, p in dets]
        t0 = time.perf_counter()
        lock.update(poses, t)
        spent.append(time.perf_counter() - t0)
    assert np.median(spent) * 1000 < 0.3  # measured ~0.05 ms; a 60 Hz frame has 16.7 ms


def test_one_person_reported_twice_is_one_person():
    """DETR has no NMS and sometimes reports a person twice (seen live on the
    crowd photo): the duplicate must not become a second identity."""
    lock = PlayerLock((C.W, C.H))
    a = C.body_pose(C.Look(640, 150, 130))
    twin = C.body_pose(C.Look(642, 151, 130, score=0.55))
    for i in range(12):
        got = lock.update([a, twin], 0.05 * i)
    assert got is a and lock.others == [twin] and len(lock.tracks) == 1
