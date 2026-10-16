"""Raccoon Dance rendering: every phase draws, the coach stays in its box, the
frame budget holds, the tutorial cards stay in their rects, the art is deterministic.

The game state is faked from chart.py with a small inline song and move library,
not the real chart (which the choreographer may still be changing).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kpapp import figure
from kpapp.game.dance import render as R
from kpapp.game.dance import tutorial as T
from kpapp.game.dance.art import AssetsMissingError, load_art, stub_art
from kpapp.game.dance.chart import load_song, parse_moves
from kpapp.game.dance.pose import keypoints_for, measure

ROOT = Path(__file__).resolve().parents[1]
SIZE = (1280, 720)

# --- an inline song ----------------------------------------------------------

MOVES = {
    "arms_up": {"label": "Arms Up", "beats": 2, "keyframes": [{"beat": 0, "pose": {"la": [170, 175], "ra": [170, 175]}}]},
    "t_pose": {"label": "T Pose", "beats": 2, "keyframes": [{"beat": 0, "pose": {"la": [90, 90], "ra": [90, 90]}}]},
    "disco_r": {"label": "Disco Point", "beats": 2, "keyframes": [
        {"beat": 0, "pose": {"ra": [150, 165], "la": [20, 10], "lean": 8}},
        {"beat": 1, "pose": {"ra": [-30, -40], "la": [40, 20], "lean": -4}}], "judge": [0, 1]},
    "disco_l": {"mirror_of": "disco_r"},
    "squat_clap": {"label": "Squat Clap", "beats": 2, "keyframes": [
        {"beat": 0, "pose": {"la": [60, -20], "ra": [60, -20], "squat": 0.9, "ll": [30, 5], "rl": [30, 5]}}]},
    "lean_r": {"label": "Lean", "beats": 2, "keyframes": [
        {"beat": 0, "pose": {"lean": 16, "head": 8, "la": [120, 150], "ra": [40, 30], "shift": 0.4}}]},
    "cross": {"label": "Cross Arms", "beats": 2, "keyframes": [
        {"beat": 0, "pose": {"la": [-30, -80], "ra": [-30, -80]}}]},
}
CHART = ["disco_r", "disco_l", "t_pose", "arms_up", "squat_clap", "lean_r", "cross", "disco_r",
         "arms_up", "t_pose", "disco_l", "squat_clap", "lean_r", "cross", "arms_up", "t_pose",
         "disco_r", "disco_l", "t_pose", "arms_up", "squat_clap", "lean_r"]


def make_song(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    data = {
        "version": 1, "id": "test_groove", "title": "Test Groove", "artist": "KP Arcade",
        "bpm": 120, "offset": 0.0, "duration": 34.0, "difficulty": 1, "coach": {"outfit": "neon"},
        "sections": [{"name": "intro", "beat": 0}, {"name": "verse", "beat": 16},
                     {"name": "chorus", "beat": 32}, {"name": "outro", "beat": 60}],
        "moves": [{"move": m, "beat": 16 + 2 * i, **({"gold": True} if i in (8, 9) else {})}
                  for i, m in enumerate(CHART)],
    }
    (folder / "song.json").write_text(json.dumps(data))
    return load_song(folder, parse_moves(MOVES, "test library"))


class FakeGame:
    """The state docs/DANCE.md section 8 promises the renderer, driven by a beat."""

    def __init__(self, song, seed: int = 0) -> None:
        self.song = song
        self.rng = np.random.default_rng(seed)
        self.counts = {"perfect": 0, "super": 0, "good": 0, "ok": 0, "x": 0}
        self.feedback: list = []
        self.score = 0
        self.display_score = 0.0
        self.stars = 0.0
        self.combo = 0
        self.gold_flash = 0.0
        self.last_rating = None
        self.notice = None
        self.results_t = 0.0
        self.player = None
        self._judged: set[int] = set()
        self.at(-2.0)

    def at(self, beat: float, phase: str | None = None, dt: float = 1 / 60) -> "FakeGame":
        song = self.song
        self.beat = beat
        self.song_time = song.time_of(beat)
        fm = song.first_move_beat
        last = song.judge_points()[-1].beat
        if phase is None:
            phase = "intro" if beat < fm else ("dancing" if beat < last + 1 else "outro")
        self.phase = SimpleNamespace(value=phase)
        self.coach_pose = song.coach_pose(beat)
        self.upcoming = [SimpleNamespace(beat=p.beat, label=p.label, move_label=p.label, pose=p.pose, gold=p.gold)
                         for p in song.pictograms() if p.beat > beat - 1.0][:8]
        for fb in self.feedback:
            fb.age += dt
        self.feedback = [fb for fb in self.feedback if fb.age < 1.2]
        self.gold_flash = max(0.0, self.gold_flash - dt * 1.5)
        for jp in song.judge_points():
            if jp.index in self._judged or beat < jp.beat + 0.8:
                continue
            self._judged.add(jp.index)
            rating = ["perfect", "super", "good", "ok", "x"][int(self.rng.integers(0, 5)) if jp.index % 3 else 0]
            self.rate(rating, jp.gold)
        self.display_score += (self.score - self.display_score) * min(1.0, dt * 8)
        self.player = measure(keypoints_for(self.coach_pose, centre_px=(640, 300), shoulder_px=170))
        return self

    def rate(self, rating: str, gold: bool = False) -> None:
        self.counts[rating] += 1
        self.feedback.append(SimpleNamespace(rating=rating, gold=gold, age=0.0))
        self.last_rating = rating
        self.score = min(10000, self.score + {"perfect": 520, "super": 400, "good": 260, "ok": 140, "x": 0}[rating])
        self.stars = min(5.0, self.score / 2000.0)
        self.combo = self.combo + 1 if rating in ("perfect", "super") else 0
        if gold and rating in ("perfect", "super"):
            self.gold_flash = 1.0


@pytest.fixture(scope="module")
def song(tmp_path_factory):
    return make_song(tmp_path_factory.mktemp("songs") / "test_groove")


def _real_art_or_skip():
    try:
        return load_art()
    except AssetsMissingError:
        pytest.skip("dance art not generated")


def camera_frame(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(30, 220, (SIZE[1], SIZE[0], 3), dtype=np.uint8)


# --- every phase draws ---------------------------------------------------------

@pytest.mark.parametrize("use_real", [False, True])
def test_every_phase_renders(song, use_real):
    art = _real_art_or_skip() if use_real else stub_art()
    r = R.DanceRenderer(SIZE, art)
    g = FakeGame(song)
    moments = [(-1.0, None), (2.0, None), (9.0, None), (12.2, None), (14.1, None), (15.3, None), (20.0, None)]
    for beat, phase in moments:
        frame = camera_frame()
        r.render(frame, g.at(beat, phase))
        assert frame.shape == (720, 1280, 3) and frame.dtype == np.uint8
    # Dancing with every kind of feedback, a gold flash, a combo and a notice.
    g.at(34.0)
    for k in ("perfect", "super", "good", "ok", "x"):
        g.rate(k, gold=k == "perfect")
    g.combo, g.notice = 7, "STEP BACK - SHOW YOUR ARMS"
    for _ in range(30):
        frame = camera_frame()
        r.render(frame, g.at(g.beat + 0.03))
    g.notice = None
    for beat in (61.0, 62.0, 64.0, 66.0):
        r.render(camera_frame(), g.at(beat, "outro"))
    g.at(66.0, "game_over")
    g.score, g.stars = 9700, 5.0
    for t in (0.0, 0.2, 0.6, 1.4, 2.5, 4.0):
        g.results_t = t
        frame = camera_frame()
        r.render(frame, g)
        # The bottom quarter belongs to the shell's buttons: the results leave it alone
        # apart from the stage itself.
        assert frame.shape == (720, 1280, 3)


def test_renderer_never_mutates_the_game(song):
    r = R.DanceRenderer(SIZE, stub_art())
    g = FakeGame(song).at(34.0)
    g.rate("perfect", gold=True)
    before = {k: (v if not isinstance(v, (list, dict)) else repr(v)) for k, v in vars(g).items()
              if k not in ("rng", "phase", "song", "player")}
    r.render(camera_frame(), g)
    after = {k: (v if not isinstance(v, (list, dict)) else repr(v)) for k, v in vars(g).items()
             if k not in ("rng", "phase", "song", "player")}
    assert before == after


def test_results_leave_the_bottom_quarter_to_the_shell(song, monkeypatch):
    art = _real_art_or_skip()
    g = FakeGame(song).at(66.0, "game_over")
    g.score, g.stars, g.results_t = 9700, 5.0, 5.0
    g.counts = {"perfect": 30, "super": 9, "good": 4, "ok": 2, "x": 1}
    with_panel = camera_frame()
    R.DanceRenderer(SIZE, art).render(with_panel, g)
    monkeypatch.setattr(R.DanceRenderer, "_draw_results", lambda *a: None)
    without = camera_frame()
    R.DanceRenderer(SIZE, art).render(without, g)
    changed = np.any(with_panel != without, axis=2)
    ys, xs = np.nonzero(changed)
    assert len(ys) > 10000, "the results panel draws"
    assert ys.max() < 720 * 0.75, "the bottom 25% belongs to the shell's buttons"
    assert xs.min() >= R.RESULTS_RECT[0] - 4


def test_pip_shows_the_camera_as_received(song):
    r = R.DanceRenderer(SIZE, stub_art())
    g = FakeGame(song).at(20.0)
    g.player = None
    frame = np.zeros((720, 1280, 3), np.uint8)
    frame[:, 640:] = 200  # the right half of the camera is bright
    r.render(frame, g)
    x0, y0, x1, y1 = R.PIP_RECT
    pip = frame[y0 + 20:y1 - 20, x0 + 20:x1 - 20]
    left, right = pip[:, : pip.shape[1] // 2 - 5], pip[:, pip.shape[1] // 2 + 5:]
    assert right.mean() > 150 and left.mean() < 40


# --- the coach -----------------------------------------------------------------------

POSES = [
    {}, {"la": [170, 175], "ra": [170, 175]}, {"la": [90, 90], "ra": [90, 90]},
    {"ra": [150, 165], "la": [20, 10], "lean": 10}, {"la": [60, -20], "ra": [60, -20], "squat": 1.0},
    {"lean": 18, "head": 10, "la": [120, 150], "ra": [40, 30], "shift": 1.0},
    {"lean": -18, "la": [40, 30], "ra": [120, 150], "shift": -1.0},
    {"la": [-30, -80], "ra": [-30, -80]}, {"la": [200, 230], "ra": [200, 230], "ll": [40, 20], "rl": [40, 20]},
]


@pytest.mark.parametrize("pose", POSES)
def test_the_coach_stays_inside_its_roi(pose):
    art = _real_art_or_skip()
    painter = R.CoachPainter(art)
    geo = painter.layout(pose, beat=0.3)
    frame = np.zeros((720, 1280, 3), np.uint8)
    painter.paint(frame, geo, clip=R.COACH_ROI)
    x0, y0, x1, y1 = R.COACH_ROI
    inside = np.zeros((720, 1280), bool)
    inside[y0:y1, x0:x1] = True
    drawn = frame.any(axis=2)
    assert not drawn[~inside].any()
    # ...and it isn't clipped by it: the ROI's edge rows/columns stay empty.
    assert not drawn[y0, x0:x1].any() and not drawn[y1 - 1, x0:x1].any()
    assert not drawn[y0:y1, x0].any() and not drawn[y0:y1, x1 - 1].any()
    assert drawn.sum() > 20000, "the coach should be a sizeable figure"


def test_the_coach_is_the_size_the_layout_promises():
    art = _real_art_or_skip()
    painter = R.CoachPainter(art)
    frame = np.zeros((720, 1280, 3), np.uint8)
    painter.paint(frame, painter.layout({}), clip=R.COACH_ROI)
    # The figure itself: the soft contact shadow under the feet (SHADOW_INK, very dark) is the floor's.
    ys, xs = np.nonzero(frame.max(axis=2) > max(R.SHADOW_INK) + 4)
    assert 640 <= ys.max() <= 668, "feet at y ~660"
    assert 470 <= ys.max() - ys.min() <= 560, "about 520 px head to toe"
    assert abs((xs.min() + xs.max()) / 2 - R.COACH_X) < 60, "centred near x 520"


def test_mirror_convention_screen_right_arm():
    """'ra' raised must lift the arm on the screen's right (the player's real right)."""
    art = _real_art_or_skip()
    painter = R.CoachPainter(art)
    geo = painter.layout({"ra": [170, 175]})
    assert geo.j["r_hand"][0] > R.COACH_X and geo.j["r_hand"][1] < geo.head_c[1]
    assert geo.j["l_hand"][1] > geo.j["hip"][1] - 60


# --- performance -----------------------------------------------------------------------

def test_render_cost_under_4ms_per_frame(song):
    art = _real_art_or_skip()
    r = R.DanceRenderer(SIZE, art)
    g = FakeGame(song)
    frame = camera_frame()
    cam = frame.copy()
    beats = np.arange(-2.0, 66.0, 1 / 30.0)  # 60 fps at 120 bpm: 1/30 beat a frame
    for b in beats:  # warm-up: sprite, card and rotation caches
        np.copyto(frame, cam)
        r.render(frame, g.at(float(b)))
    g = FakeGame(song, seed=1)
    times = []
    for b in beats[:600]:  # 10 s of dancing
        np.copyto(frame, cam)
        g.at(float(b) + 16.0)
        t = time.perf_counter()
        r.render(frame, g)
        times.append(time.perf_counter() - t)
    ms = 1000 * float(np.mean(times))
    assert ms < 4.0, f"{ms:.2f} ms per frame"


# --- tutorial ------------------------------------------------------------------------------

RECTS = [(69 + i * 400, 134, 411 + i * 400, 334) for i in range(3)]


def test_tutorial_has_three_cards():
    assert [s.title for s in T.DANCE_TUTORIAL] == ["STAND BACK", "COPY LIKE A MIRROR", "HIT THE POSES ON THE BEAT"]
    assert all(isinstance(s, figure.TutorialStep) for s in T.DANCE_TUTORIAL)


@pytest.mark.parametrize("t", [0.0, 0.4, 1.1, 1.9, 2.6, 3.3, 5.05, 7.7])
def test_tutorial_cards_draw_inside_their_rects(t):
    for i, (step, rect) in enumerate(zip(T.DANCE_TUTORIAL, RECTS)):
        frame = np.random.default_rng(i).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
        before = frame.copy()
        step.draw(frame, rect, t)
        changed = np.any(frame != before, axis=2)
        x0, y0, x1, y1 = rect
        inside = np.zeros((720, 1280), bool)
        inside[y0:y1, x0:x1] = True
        assert not changed[~inside].any(), f"{step.title} drew outside its rect at t={t}"
        assert changed[inside].mean() > 0.03


def test_tutorial_without_art(monkeypatch):
    monkeypatch.setattr(T, "_art", None)
    monkeypatch.setattr(T, "_art_missing", True)
    for step, rect in zip(T.DANCE_TUTORIAL, RECTS):
        frame = np.zeros((720, 1280, 3), np.uint8)
        step.draw(frame, rect, 1.0)
        assert frame.any()


def test_tutorial_costs_under_1_5ms_for_all_three():
    frame = np.zeros((720, 1280, 3), np.uint8)
    times = [k / 60 for k in range(60 * 6)]
    for t in times:
        for step, rect in zip(T.DANCE_TUTORIAL, RECTS):
            step.draw(frame, rect, t)
    start = time.perf_counter()
    for t in times:
        for step, rect in zip(T.DANCE_TUTORIAL, RECTS):
            step.draw(frame, rect, t)
    ms = (time.perf_counter() - start) * 1000 / len(times)
    assert ms < 1.5, f"{ms:.2f} ms per frame"


# --- the generator --------------------------------------------------------------------------

def test_generator_is_deterministic():
    spec = importlib.util.spec_from_file_location("_gen_dance", ROOT / "tools" / "generate_dance_assets.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    # A light check on a subset: the word, star and card makers (the stage and head
    # share the same dither and drawing paths).
    makers = [lambda: gen.make_word("PERFECT", gen.RATINGS["perfect"]["colour"], 40, stream=40),
              lambda: gen.make_star(full=True), lambda: gen.make_card(True), lambda: gen.make_sparkle(),
              lambda: gen.make_floor_glow()]
    for make in makers:
        a, b = make(), make()
        assert hashlib.md5(a.tobytes()).hexdigest() == hashlib.md5(b.tobytes()).hexdigest()


def test_generated_manifest_matches_the_loader():
    art = _real_art_or_skip()
    outfit = art.outfit("neon")
    for key in ("fur", "top", "pants", "accent", "glove", "shoe", "ink", "rim", "glow", "fur_dark"):
        assert key in outfit and len(outfit[key]) == 3
    assert art.outfit("no-such-outfit") == outfit, "unknown outfits fall back to neon"
    for k in ("perfect", "super", "good", "ok", "x"):
        assert art.words[k].shape[2] == 4 and art.rating_colour(k) != (255, 255, 255)
    assert art.stage.shape == (720, 1280, 3)


def test_rating_key_accepts_strings_and_enums():
    from enum import Enum

    class Rating(Enum):
        PERFECT = "perfect"
        X = "x"

    assert R.rating_key("PERFECT") == "perfect"
    assert R.rating_key(Rating.X) == "x"
    assert R.rating_key(None) is None and R.rating_key("unseen") is None
    assert math.isfinite(R._kick(3.5))


def test_the_coachs_bow_waits_for_the_charts_own_ending():
    """Neon Nights ends finale (judged, beat 192) -> bow (unjudged, beat 196).
    The renderer's bow starts with the chart's bow, so the finale is held on
    its beat and the coach bows once (QA: it bowed at the outro start, cutting
    the finale short, then bowed again with the chart)."""
    import math

    from kpapp.game.dance import chart
    from kpapp.game.dance.render import _ending_start

    song = chart.load_song(chart.SONGS / "neon_nights")
    trailing = [p for p in song.chart if p.beat > song.judge_points()[-1].beat]
    assert trailing and not any(p.move.judge for p in trailing)
    assert _ending_start(song) == song.time_of(trailing[0].beat)
    lib = chart.parse_moves({"m": {"beats": 2, "keyframes": [{"beat": 0, "pose": {}}]}}, "t")
    song.chart = [chart.Placed(lib["m"], 16.0)]
    assert _ending_start(song) == -math.inf, "no chart ending: bow when the outro starts"


def test_the_results_have_landed_before_the_shell_asks_for_a_name(song, monkeypatch):
    """The shell covers the game's results with its name entry after
    ENTRY_DELAY and stops updating the game, so the count-up must be done by
    then (QA: it froze at "9419" behind a board that said 9463)."""
    from kpapp.shell import ENTRY_DELAY

    art = _real_art_or_skip()
    g = FakeGame(song).at(66.0, "game_over")
    g.score, g.stars, g.results_t = 9463, 5.0, ENTRY_DELAY - 0.1
    drawn = []
    real = R.text_c
    monkeypatch.setattr(R, "text_c", lambda img, s, *a, **k: (drawn.append(s), real(img, s, *a, **k)))
    R.DanceRenderer(SIZE, art).render(camera_frame(), g)
    assert "9463" in drawn, drawn


def test_lane_cards_every_beat_grow_smoothly_and_never_overlap():
    """Every judge point has a card, so cards can come every beat. Their size
    goes by distance alone (no pop when the "next" card changes, no shrink
    when a card passes the ring), and two cards a beat apart never overlap."""
    step = 2.0 / 60  # one frame at 120 BPM, in beats
    ds = np.arange(-0.65, 6.0, step / 4)
    widths = [R.card_width(d) for d in ds]
    assert max(abs(a - b) for a, b in zip(widths, widths[1:])) <= 4, "grows a step at a time"
    assert all(R.card_width(d) == R.CARD_BIG for d in ds if d <= 0), "full size on and past the ring"
    assert all(R.card_width(d) == R.CARD_SMALL for d in ds if d >= R.GROW_BEATS)
    for d in ds[ds >= 0]:
        gap = R.PX_PER_BEAT - (R.card_width(d) + R.card_width(d + 1.0)) / 2
        assert gap >= 8, f"cards a beat apart overlap at d={d:.2f}"


def test_an_older_popup_is_gone_once_the_newer_one_is_a_few_frames_old(song, monkeypatch):
    """QA: with a 0.12 s hand-over, "PERFECT" read stacked over "PERFECT" for ~7 frames."""
    assert R.SUPERSEDE_FADE <= 0.06
    r = R.DanceRenderer(SIZE, stub_art())
    g = FakeGame(song).at(30.0)
    g.feedback = [SimpleNamespace(rating="perfect", gold=False, age=0.5),
                  SimpleNamespace(rating="perfect", gold=False, age=R.SUPERSEDE_FADE)]
    calls = []
    words = r._words["perfect"]
    real = words.get
    monkeypatch.setattr(words, "get", lambda s: (calls.append(s), real(s))[1])
    r.render(camera_frame(), g)
    assert len(calls) == 1, "only the newer word is drawn"


# --- difficulty badge and the bigger lane / PiP ------------------------------------------

def _diff(song, g_setup, monkeypatch=None, use_real=False, **patch):
    """Pixels that change when the game has a difficulty (badge on) vs not."""
    art = _real_art_or_skip() if use_real else stub_art()
    out = []
    for level in ("easy", None):
        r = R.DanceRenderer(SIZE, art)
        g = g_setup(FakeGame(song))
        if level is not None:
            g.difficulty = level
        frame = camera_frame()
        r.render(frame, g)
        out.append(frame)
    return np.any(out[0] != out[1], axis=2)


@pytest.mark.parametrize("where", ["intro", "dancing", "results"])
def test_the_difficulty_badge_is_drawn(song, where):
    def setup(g):
        if where == "intro":
            return g.at(2.0)
        if where == "dancing":
            return g.at(30.0)
        g.at(66.0, "game_over")
        g.score, g.stars, g.results_t = 9000, 4.5, 2.0
        return g
    changed = _diff(song, setup)
    ys, xs = np.nonzero(changed)
    assert len(ys) > 300, "a badge appears"
    if where == "dancing":  # right of the score, top-left
        assert xs.min() >= R.HUD_BADGE_ORG[0] - 2 and ys.max() < 100 and xs.max() < R.PIP_RECT[0]
    elif where == "intro":  # under the title card (and the HUD's, which shows from the start)
        y = R.INTRO_C[1] + 122
        assert changed[y - 20:y + 20, R.INTRO_C[0] - 60:R.INTRO_C[0] + 60].sum() > 300
        assert not changed[R.INTRO_C[1] + 100:, :R.INTRO_C[0] - 120].any()
    else:  # in the results panel's top line
        x0, y0, x1, y1 = R.RESULTS_RECT
        assert xs.min() >= x0 and xs.max() <= x1 and ys.max() < y0 + 60


def test_badge_colours_read_easy_green_normal_cyan_hard_magenta():
    r = R.DanceRenderer(SIZE, _real_art_or_skip())
    b, g, rr = zip(*(r.difficulty_colours[k] for k in ("easy", "normal", "hard")))
    assert g[0] > max(b[0], rr[0]), "easy is green"
    assert b[1] > rr[1] and g[1] > rr[1], "normal is cyan"
    assert rr[2] > g[2] and b[2] > g[2], "hard is magenta"
    assert r.badge("expert") is None and r.badge(None) is None
    assert r.badge("easy").w > 40 and r.badge("hard") is r.badge("hard"), "cached"


def test_cards_and_pip_are_bigger_and_stay_in_bounds():
    """The user asked for slightly bigger cards and PiP: ~+20%, 16:9 kept."""
    assert 1.15 <= R.CARD_SMALL / 76 <= 1.25 and 1.15 <= R.CARD_BIG / 108 <= 1.25
    x0, y0, x1, y1 = R.PIP_RECT
    assert (x1 - x0, y1 - y0) == (384, 216)
    assert x0 >= 0 and y0 >= 16 and x1 <= SIZE[0] - 16
    ch = R.CARD_BIG * 208 / 192  # the card art's aspect
    assert R.LANE_Y + ch / 2 <= R.LANE_CLIP_BOTTOM < R.PAUSE_RECT[1], "a full-size card isn't clipped"
    assert R.LANE_Y + R.RING / 2 < R.LABEL_Y - 18, "the label sits under the ring"
    assert R.LABEL_Y + 8 < R.FOOTER_Y
    assert R.LANE_RECT[3] < R.PAUSE_RECT[1]
    # A card only reaches full size within a beat of the ring, well left of PAUSE.
    assert R.TARGET_X + R.GROW_BEATS * R.PX_PER_BEAT + R.CARD_BIG / 2 < R.PAUSE_RECT[0]


def _overlay_pixels(song, beats, monkeypatch, parts, art):
    """Pixels the given renderer parts draw, over a run of frames (vs the same
    frames with those parts switched off)."""
    hit = np.zeros((SIZE[1], SIZE[0]), bool)
    g_on, g_off = FakeGame(song), FakeGame(song)
    r_on, r_off = R.DanceRenderer(SIZE, art), R.DanceRenderer(SIZE, art)
    for name in parts:
        monkeypatch.setattr(r_off, name, lambda *a, **k: None)
    for b in beats:
        fa, fb = camera_frame(), camera_frame()
        g_on.at(float(b))
        g_off.at(float(b))
        g_on.difficulty = g_off.difficulty = "hard"
        r_on.render(fa, g_on)
        r_off.render(fb, g_off)
        hit |= np.any(fa != fb, axis=2)
    return hit


def test_nothing_of_ours_enters_the_pause_button_or_the_footer(song, monkeypatch):
    """The lane (cards every beat, gold sparkles, the label), the HUD, the
    feedback words and the PiP over a whole song: PAUSE and the footer stay
    the stage's own pixels."""
    art = _real_art_or_skip()
    beats = np.arange(10.0, 62.0, 0.19)
    hit = _overlay_pixels(song, beats, monkeypatch,
                          ("_draw_lane", "_draw_hud", "_draw_feedback", "_draw_pip", "_draw_intro"), art)
    x0, y0, x1, y1 = R.PAUSE_RECT
    assert not hit[y0:y1, x0:x1].any(), "something drew into the PAUSE button"
    assert not hit[R.FOOTER_Y:, :].any(), "something drew into the footer"
    assert hit[R.LANE_RECT[1]:R.LANE_RECT[3], R.LANE_RECT[0]:R.LANE_RECT[2]].mean() > 0.3, "the lane draws"


def test_the_intro_text_stays_clear_of_the_bigger_pip(song, monkeypatch):
    """Title card, badge, GET READY and the 3-2-1 (once popped in) sit under the PiP."""
    art = _real_art_or_skip()
    fm = song.first_move_beat
    beats = [-1.0, 2.0, fm - 9.0, fm - 7.0, fm - 6.0] + [fm - 4 + i + 0.5 for i in range(4)]
    hit = _overlay_pixels(song, beats, monkeypatch, ("_draw_intro",), art)
    x0, y0, x1, y1 = R.PIP_RECT
    assert hit.any()
    assert not hit[y0:y1 + 8, x0 - 8:x1].any(), "intro text under the PiP"
