"""Dance charts: the move library, the shipped songs, the importer and the CLI.

Every song in assets/dance/songs/index.json is validated here, so a hand-edited
or imported chart that breaks a rule fails loudly (docs/DANCE.md sections 4-6, 11).
"""

from __future__ import annotations

import json
import math
import re
import sys

import cv2
import numpy as np
import pytest

from kpapp.game.dance import chart
from kpapp.game.dance.chart import load_library, load_song, load_songs, song_ids
from kpapp.game.dance.importer import (
    chart_from_poses, format_song_json, merge_into_song, pose_from_measured, video_samples,
)
from kpapp.game.dance.pose import REST, keypoints_for, measure, normalize_pose, similarity

LIB = load_library()
SONGS = load_songs(library=LIB)
NEON = next(s for s in SONGS if s.id == "neon_nights")
# Every song on every difficulty: the rules below hold for all of them.
LEVELS = [s.for_difficulty(d) for s in SONGS for d in chart.DIFFICULTIES]

ELBOWS_HIDDEN = {"shoulder", "wrist", "hip"}
FREEZE_MAX = 0.55  # a player frozen in the previous pose must score below this


def sim(target: dict, performed: dict, visible=None, **kw) -> float:
    """How well a perfect performance of `performed` matches `target`."""
    return similarity(target, measure(keypoints_for(performed, visible=visible, **kw)))[0]


# --- every song --------------------------------------------------------------

def test_every_listed_song_loads_and_validates():
    ids = song_ids()
    assert ids and ids[0] == "neon_nights"
    assert [s.id for s in SONGS] == ids
    for song in SONGS:
        jp = song.judge_points()
        assert jp, f"{song.id} has nothing to judge"
        assert song.first_move_beat >= 16, f"{song.id}: moves must wait for the intro"
        assert all(p.beat < song.beat_of(song.duration) for p in jp)


@pytest.mark.parametrize("song", SONGS, ids=lambda s: s.id)
def test_audio_is_delivered(song):
    if not song.audio.exists():
        pytest.skip(f"{song.audio.name} not generated yet (tools/generate_dance_song.py)")
    assert song.audio.stat().st_size > 10_000


@pytest.mark.parametrize("song", LEVELS, ids=lambda s: f"{s.id}-{s.level}")
def test_a_frozen_player_cannot_score(song):
    """Consecutive judged poses differ enough that holding the previous one
    earns less than 0.55 on the next, with or without elbows in view. On
    every difficulty: EASY's derived chart must not let a player stand still."""
    assert FREEZE_MAX == chart.FREEZE_MAX
    jp = song.judge_points()
    for a, b in zip(jp, jp[1:]):
        full = sim(b.pose, a.pose)
        reach = sim(b.pose, a.pose, ELBOWS_HIDDEN)
        assert full < FREEZE_MAX, f"{song.id}: '{a.label}' at beat {a.beat:g} scores {full:.2f} on beat {b.beat:g}"
        assert reach < FREEZE_MAX, (f"{song.id}: without elbows '{a.label}' at beat {a.beat:g} "
                                    f"scores {reach:.2f} on beat {b.beat:g}")


@pytest.mark.parametrize("song", LEVELS, ids=lambda s: f"{s.id}-{s.level}")
def test_a_perfect_player_matches_every_judged_pose(song):
    for p in song.judge_points():
        assert sim(p.pose, p.pose) > 0.97, f"{p.label} at {p.beat:g}"
        assert sim(p.pose, p.pose, ELBOWS_HIDDEN) > 0.97, f"{p.label} at {p.beat:g}, elbows hidden"
        close = sim(p.pose, p.pose, centre_px=(640, 480), shoulder_px=200)  # hips below the frame
        assert close > 0.97, f"{p.label} at {p.beat:g}, standing close"


# --- Neon Nights: the spec in docs/DANCE.md section 6 ----------------------------------

SECTIONS = [("intro", 0), ("verse 1", 16), ("chorus 1", 48), ("verse 2", 80), ("chorus 2", 112),
            ("bridge", 144), ("final chorus", 160), ("outro", 192)]


def section_moves(song, lo, hi):
    return [p for p in song.chart if lo <= p.beat < hi]


def test_neon_nights_metadata():
    data = json.loads((NEON.folder / "song.json").read_text())
    for key, value in {"id": "neon_nights", "title": "Neon Nights", "artist": "KP Arcade", "bpm": 120,
                       "offset": 0.0, "duration": 106.0, "audio": "audio.m4a", "difficulty": 1,
                       "coach": {"outfit": "neon"}, "preview": [48, 64]}.items():
        assert data[key] == value, key
    assert [(s["name"], s["beat"]) for s in NEON.sections] == SECTIONS
    assert NEON.beat_of(NEON.duration) == 212, "52 bars plus a 2 s tail"


def test_neon_nights_structure():
    assert not section_moves(NEON, 0, 16), "the intro has no moves"
    assert NEON.first_move_beat == 16
    choruses = [section_moves(NEON, lo, lo + 32) for lo in (48, 112, 160)]
    signature = [p.move.name for p in choruses[0]]
    for c in choruses:
        assert [p.move.name for p in c] == signature, "every chorus dances the same signature"
        assert c[-1].end <= c[0].beat + 32
    assert signature[:len(signature) // 2] == signature[len(signature) // 2:], "the phrase repeats within a chorus"
    assert [sum(p.gold for p in c) for c in choruses] == [1, 1, 2]
    assert not any(p.gold for p in NEON.chart if not any(c0[0].beat <= p.beat < c0[0].beat + 32
                                                          for c0 in choruses)), "gold only in choruses"
    bridge = section_moves(NEON, 144, 160)
    assert bridge and all(p.move.beats == 4 for p in bridge), "the bridge slows down to 4-beat moves"
    outro = section_moves(NEON, 192, 212)
    assert outro[0].beat == 192 and outro[0].move.judge, "a judged final pose on beat 192"
    assert outro[-1].move.name == "bow" and not outro[-1].move.judge, "then an unjudged bow"
    assert NEON.judge_points()[-1].beat == 192
    assert NEON.last_judge_time < NEON.time_of(208), "judging ends before the outro does"


def test_verse_1_is_the_easiest():
    """Verse 1 judges at most one pose every 2 beats on average and uses the
    gentlest moves; the choruses judge about one per beat."""
    def density(lo, hi):
        return sum(1 for p in NEON.judge_points() if lo <= p.beat < hi) / (hi - lo)
    assert density(16, 48) <= 0.65
    assert density(16, 48) < density(80, 112) <= density(48, 80)
    assert density(144, 160) <= 0.5


def test_sides_alternate():
    """A one-sided move is followed by its twin, so both arms get the work."""
    names = [p.move.name for p in NEON.chart]
    for i, n in enumerate(names):
        if n.endswith("_r") and f"{n[:-2]}_l" in names:
            assert f"{n[:-2]}_l" in names[i + 1:i + 3] or f"{n[:-2]}_l" in names[max(0, i - 2):i], n


# --- difficulties (docs/DANCE.md section 7a) ---------------------------------------

def test_difficulty_ids_fall_back_to_normal():
    assert chart.DIFFICULTIES == ("easy", "normal", "hard")
    assert [chart.difficulty_id(d) for d in ("easy", "HARD", "normal", None, "expert", "")] == \
        ["easy", "hard", "normal", "normal", "normal", "normal"]
    assert NEON.level == "normal" and NEON.for_difficulty("normal") is NEON
    assert load_song(NEON.folder, LIB, difficulty="nightmare").level == "normal"


def test_hard_dances_the_normal_chart():
    hard = NEON.for_difficulty("hard")
    assert hard.level == "hard"
    assert [(p.beat, p.label, p.gold) for p in hard.judge_points()] == \
        [(p.beat, p.label, p.gold) for p in NEON.judge_points()]
    assert load_song(NEON.folder, LIB, difficulty="hard").judge_points() == hard.judge_points()


@pytest.mark.parametrize("song", SONGS, ids=lambda s: s.id)
def test_easy_judges_about_one_pose_per_bar(song):
    """EASY: fewer moves, held longer. At least two beats between judged poses,
    at most one per bar where the normal chart's moves are short, and far
    fewer points overall."""
    easy, normal = song.for_difficulty("easy"), song.judge_points()
    pts = easy.judge_points()
    assert pts and len(pts) <= 0.4 * len(normal), (len(pts), len(normal))
    beats = [p.beat for p in pts]
    assert all(b - a >= chart.EASY_MIN_GAP for a, b in zip(beats, beats[1:]))
    short = {p.beat for p in pts if easy.chart[p.placed].move.name.endswith("~held")}
    bars = [math.floor(b / chart.BAR_BEATS) for b in sorted(short)]
    assert len(bars) == len(set(bars)), "one held pose per bar"
    if "easy" not in song.charts:  # derived, not hand-made
        assert all(any(abs(p.beat - q.beat) < 1e-9 and p.pose == q.pose for q in normal) for p in pts), \
            "easy only asks for poses the normal chart asks for, on the same beats"
    # The score still tops out at 10000, and the server's rate limit (400 points
    # a second) still lets a perfect run through.
    assert easy.last_judge_time >= 25.0


@pytest.mark.parametrize("song", SONGS, ids=lambda s: s.id)
def test_easy_coach_holds_the_kept_pose(song):
    """What the player sees is what is asked: a held move is one keyframe, so
    between two kept poses the coach holds the first and eases into the next
    (no in-between poses that aren't judged)."""
    easy = song.for_difficulty("easy")
    for p in easy.chart:
        if p.move.name.endswith("~held"):
            assert len(p.move.keyframes) == 1 and p.move.judge == (0,)
    pts = easy.judge_points()
    for a, b in zip(pts, pts[1:]):
        if not easy.chart[a.placed].move.name.endswith("~held"):
            continue
        for u in (0.0, 0.5, 1.0):  # held from its beat until the ease into the next
            t = a.beat + u * max(0.0, b.beat - a.beat - chart.EASE_BEATS)
            assert sim(a.pose, easy.coach_pose(t)) > 0.97, f"coach left {a.label} at beat {t:g}"


def test_neon_nights_easy():
    easy = NEON.for_difficulty("easy")
    pts = easy.judge_points()
    assert len(NEON.judge_points()) == 145 and len(pts) == 49
    # Every section with gold keeps gold.
    for name, lo in SECTIONS:
        hi = next((b for _, b in SECTIONS if b > lo), 212)
        if any(p.gold for p in NEON.judge_points() if lo <= p.beat < hi):
            assert any(p.gold for p in pts if lo <= p.beat < hi), f"{name} lost its gold"
    # The bridge's 4-beat moves stay whole (the coach dances them in full), judged as in normal.
    bridge = section_moves(easy, 144, 160)
    assert [p.move for p in bridge] == [p.move for p in section_moves(NEON, 144, 160)]
    # The final pose and the bow stay.
    outro = section_moves(easy, 192, 212)
    assert [(p.beat, p.move.name) for p in outro] == [(192, "finale"), (196, "bow")]
    assert pts[-1].beat == 192
    # Verse 1 goes from a pose every 2 beats to one a bar.
    assert sum(1 for p in pts if 16 <= p.beat < 48) == 8


def _write_song(tmp_path, **extra):
    folder = tmp_path / "s"
    folder.mkdir(exist_ok=True)
    data = {"title": "T", "bpm": 120, "duration": 40.0,
            "moves": [{"move": "shine", "beat": 16}, {"move": "rise", "beat": 20}, {"move": "star", "beat": 24}],
            **extra}
    (folder / "song.json").write_text(json.dumps(data))
    return folder


def test_a_song_can_hand_make_its_easy_and_hard_charts(tmp_path):
    folder = _write_song(tmp_path, charts={"easy": [{"move": "rise", "beat": 20, "gold": True}],
                                           "hard": [{"move": "shine", "beat": 16}, {"move": "star", "beat": 18},
                                                    {"move": "rise", "beat": 20}]})
    easy = load_song(folder, LIB, difficulty="easy")
    assert [(p.move.name, p.beat, p.gold) for p in easy.chart] == [("rise", 20.0, True)]
    hard = load_song(folder, LIB, difficulty="hard")
    assert [p.move.name for p in hard.chart] == ["shine", "star", "rise"]
    normal = load_song(folder, LIB)
    assert [p.move.name for p in normal.chart] == ["shine", "rise", "star"]
    assert normal.for_difficulty("easy").chart == easy.chart, "the hand-made chart wins over the derived one"


@pytest.mark.parametrize("charts, message", [
    ({"easy": [{"move": "nope", "beat": 16}]}, "charts.easy: chart entry 0 at beat 16.0: unknown move 'nope'"),
    ({"hard": [{"move": "rise", "beat": 16}, {"move": "shine", "beat": 18}]}, "charts.hard: move 'shine' at beat 18"),
    ({"hard": [{"move": "rise", "beat": 78}]}, "runs past the song's end"),
    ({"normal": []}, "only 'easy' and 'hard' charts can be hand-made"),
    ({"expert": []}, "only 'easy' and 'hard'"),
    ({"easy": {"move": "rise"}}, "charts.easy: expected a list"),
    ([], "'charts' must be an object"),
])
def test_hand_made_charts_are_validated_like_the_normal_one(tmp_path, charts, message):
    folder = _write_song(tmp_path, charts=charts)
    with pytest.raises(chart.ChartError, match=re.escape(message)):
        load_song(folder, LIB)  # whichever difficulty is asked for


def test_easy_moves_gold_to_the_kept_pose_when_the_gold_one_would_freeze():
    """A gold point that a player frozen in the previous kept pose would score
    on can't be kept, but its bar keeps gold: the kept pose turns gold."""
    up, out = {"la": [170, 175], "ra": [170, 175]}, {"la": [90, 90], "ra": [90, 90]}
    moves = chart.parse_moves({"up": {"beats": 2, "keyframes": [{"beat": 0, "pose": up}]},
                               "out": {"beats": 2, "keyframes": [{"beat": 0, "pose": out}]}}, "t")
    normal = [chart.Placed(moves["up"], 16), chart.Placed(moves["out"], 18),
              chart.Placed(moves["up"], 20, gold=True), chart.Placed(moves["out"], 22)]
    easy = chart.derive_easy(normal)
    assert [(p.beat, p.move.label, p.gold) for p in easy] == [(16, "Up", False), (22, "Out", True)]
    assert easy[0].end == 20 and easy[1].end == 24, "held to the end of the bar's moves"


# --- the move library ----------------------------------------------------------

def test_library_size_twins_and_labels():
    assert 22 <= len(LIB) <= 32
    raw = json.loads(chart.LIBRARY.read_text())["moves"]
    for name, spec in raw.items():
        if name.endswith("_l"):
            assert spec.get("mirror_of") == name[:-2] + "_r", f"{name}: author one side, mirror the other"
    for m in LIB.values():
        assert 0 < len(m.label) <= 15, f"{m.name}: labels are short (pictograms)"
    assert "bow" in LIB and LIB["bow"].judge == ()
    assert any(m.beats == 4 and len(m.judge) >= 2 for m in LIB.values()), "slow moves for the bridge"


@pytest.mark.parametrize("name", sorted(LIB))
def test_judged_keyframes_are_clear_arm_shapes(name):
    move = LIB[name]
    rest = normalize_pose(REST)
    for i, k in enumerate(move.keyframes):
        for s in "lr":
            upper, fore = k.pose[f"{s}a"]
            assert -110 <= upper <= 260 and -110 <= fore <= 260, f"{name}[{i}] {s}a out of the authoring range"
            assert abs(fore - upper) <= 160, f"{name}[{i}] {s}a folds the elbow too far"
        if i in move.judge:
            assert sim(k.pose, rest) < FREEZE_MAX, f"{name}[{i}] is too close to just standing there"
    judged = [move.keyframes[i].pose for i in move.judge]
    for a, b in zip(judged, judged[1:]):
        assert sim(b, a) < FREEZE_MAX and sim(a, b) < FREEZE_MAX, f"{name}: its judged keyframes look alike"


# --- importer ---------------------------------------------------------------

def synth_samples(song, hz, seed, noise_px=4.0, until=100.0, drop_elbows=0.0):
    """A dancer who performs the song's coach exactly, seen by a model at `hz`
    with keypoint noise and (optionally) elbows missing in some frames."""
    rng = np.random.default_rng(seed)
    t = float(rng.uniform(0, 1 / hz))
    out = []
    while t < until:
        visible = ELBOWS_HIDDEN if rng.random() < drop_elbows else None
        kp = keypoints_for(song.coach_pose(song.beat_of(t)), visible=visible)
        kp.xy += rng.normal(0, noise_px, kp.xy.shape).astype(np.float32)
        out.append((t, kp))
        t += 1 / hz
    return out


def import_song(tmp_path, samples, **kw):
    imported = chart_from_poses(samples, 120, 0.0, 16, 196, **kw)
    data = json.loads((NEON.folder / "song.json").read_text())
    data = merge_into_song(data, imported, 0, 212, LIB)
    folder = tmp_path / "imported"
    folder.mkdir(exist_ok=True)
    (folder / "song.json").write_text(format_song_json(data))
    return imported, load_song(folder, LIB)


@pytest.mark.parametrize("hz", [15, 30])
def test_importer_round_trip(tmp_path, hz):
    imported, song = import_song(tmp_path, synth_samples(NEON, hz, seed=hz, drop_elbows=0.3))
    assert len(imported["moves"]) == 180, "one move per beat from 16 to 196"
    got = {p.beat: p for p in song.judge_points()}
    for p in NEON.judge_points():
        assert p.beat in got, f"no imported keyframe on beat {p.beat:g}"
        score = sim(p.pose, got[p.beat].pose)
        assert score > 0.9, f"{p.label} at beat {p.beat:g} imported as {got[p.beat].pose} ({score:.2f})"
    assert all(m.startswith("imp_") for m in imported["custom_moves"])
    assert song.title == "Neon Nights" and song.sections == NEON.sections, "metadata survives the import"


def test_importer_merges_held_poses(tmp_path):
    imported, song = import_song(tmp_path, synth_samples(NEON, 30, seed=3), merge=True)
    assert len(imported["moves"]) < 180
    held = [m for m in imported["custom_moves"].values() if m["beats"] > 1]
    assert held, "the 2-beat Lightning holds become one judged move"
    for p in NEON.judge_points():
        assert sim(p.pose, song.coach_pose(p.beat)) > 0.9, f"{p.label} at beat {p.beat:g}"


def test_importer_skips_beats_it_cannot_see_and_handles_half_beats():
    pose = {"la": [90, 90], "ra": [170, 175]}
    samples = [(t / 30, keypoints_for(pose)) for t in range(0, 63)]  # up to 2.07 s: beats 0-4
    samples += [(t / 30, keypoints_for(pose, visible={"shoulder"})) for t in range(63, 120)]  # no arms
    out = chart_from_poses(samples, 120, 0.0, 0, 8, every=0.5)
    beats = [m["beat"] for m in out["moves"]]
    assert beats == [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], "after beat 4 no arms were in view"
    assert all(out["custom_moves"][m["move"]]["beats"] == 0.5 for m in out["moves"])
    assert "imp_1_5" in out["custom_moves"]


def test_reach_only_arms_import_as_straight_arms():
    m = measure(keypoints_for({"la": [90, 90], "ra": [150, 150]}, visible=ELBOWS_HIDDEN))
    pose = pose_from_measured(m)
    assert pose["la"] == [90, 90] and pose["ra"] == [150, 150]
    assert pose_from_measured(measure(keypoints_for({}, visible={"shoulder"}))) is None


def test_imported_angles_use_the_chart_convention():
    """Measured angles wrap at 180; the chart's run past vertical (up to ~260),
    so the coach doesn't swing an arm through the body between keyframes."""
    pose = pose_from_measured(measure(keypoints_for({"la": [165, 212], "ra": [-35, -45]})))
    assert pose["la"] == [165, 212] and pose["ra"] == [-35, -45]


def test_merge_keeps_moves_outside_the_range():
    data = {"title": "T", "bpm": 120, "duration": 30,
            "moves": [{"move": "shine", "beat": 4}, {"move": "imp_10", "beat": 10}, {"move": "rise", "beat": 20}],
            "custom_moves": {"imp_10": {"beats": 1, "keyframes": [{"beat": 0, "pose": {}}]}}}
    imported = {"moves": [{"move": "imp_9", "beat": 9}],
                "custom_moves": {"imp_9": {"beats": 1, "keyframes": [{"beat": 0, "pose": {"la": [90, 90]}}]}}}
    out = merge_into_song(data, imported, 8, 16, LIB)
    assert [m["beat"] for m in out["moves"]] == [4, 9, 20]
    assert set(out["custom_moves"]) == {"imp_9"}, "stale imported moves are dropped"
    assert out["title"] == "T"


def test_video_samples_pick_frames_near_the_beats(tmp_path):
    path = tmp_path / "take.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
    for i in range(90):  # 3 s
        frame = np.zeros((48, 64, 3), np.uint8)
        frame[:, :8] = 255  # a white stripe on the left, to see the mirroring
        writer.write(frame)
    writer.release()
    seen = []

    def detect(frame):
        seen.append(frame[:, -4:].mean() > 128)  # mirrored: the stripe is on the right
        return "pose"

    wanted = [0.5, 1.0, 1.5]  # song times
    out = video_samples(path, detect, wanted, video_start=1.0, song_offset=0.0)  # beat 0 at 1 s into the video
    assert out and all(seen)
    times = [t for t, _ in out]
    for w in wanted:
        near = [t for t in times if abs(t - w) <= 0.061]
        assert near, f"no frame near song time {w}"
    assert all(min(abs(t - w) for w in wanted) <= 0.061 for t in times), "only frames near a beat are run"


# --- CLI ---------------------------------------------------------------------

def run_cli(monkeypatch, *argv):
    from kpapp import app

    monkeypatch.setattr(sys, "argv", ["kp", *argv])
    return app.main()


def test_cli_dance_import_parses(monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        run_cli(monkeypatch, "dance-import", "--help")
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--song", "--video", "--start", "--every", "--no-mirror", "--out", "--live"):
        assert flag in out


def test_cli_dance_preview_writes_a_contact_sheet(monkeypatch, tmp_path):
    out = tmp_path / "sheet.png"
    assert run_cli(monkeypatch, "dance-preview", "--song", "neon_nights", "--out", str(out)) == 0
    img = cv2.imread(str(out))
    assert img is not None and img.shape[1] >= 1200 and img.shape[0] > 1000


def test_cli_dance_preview_draws_a_difficulty(monkeypatch, tmp_path):
    out = tmp_path / "easy.png"
    assert run_cli(monkeypatch, "dance-preview", "--song", "neon_nights", "--difficulty", "easy", "--out", str(out)) == 0
    easy = cv2.imread(str(out))
    normal = tmp_path / "normal.png"
    assert run_cli(monkeypatch, "dance-preview", "--song", "neon_nights", "--out", str(normal)) == 0
    assert easy is not None and easy.shape[0] < cv2.imread(str(normal)).shape[0], "fewer moves, a shorter sheet"
