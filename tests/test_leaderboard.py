"""Leaderboards: storage, ranking rules, the server client, and the shell's
tutorial / name-entry / results flow (mirrors web/tests/leaderboard.test.mjs)."""

from __future__ import annotations

import io
import json
import time
import urllib.error

import numpy as np
import pytest

from kpapp.config import KP
from kpapp.controls import ControlState
from kpapp.game.base import Game
from kpapp.inference import Pose
from kpapp.leaderboard import (
    BOARD_SIZE, FallbackBoard, LocalBoard, ServerBoard, clean_name, create_board, name_allowed,
    qualifies, site_url, sort_entries,
)
from kpapp.shell import ENTRY_DELAY, KEY_DWELL, KEY_ENTER, KEY_ESC, KEY_SPACE, RESULTS_DELAY, Screen, Shell

W, H = 1280, 720


def test_names_are_three_characters_a_to_z_and_digits():
    assert clean_name("ab") == "AB"
    assert clean_name("a.b c-d") == "ABC"
    assert clean_name("x9!") == "X9"
    assert name_allowed("ACE")
    assert not name_allowed("")
    assert not name_allowed("abcd"), "only already-clean names pass"
    assert not name_allowed("ASS"), "obvious abuse is refused"


def test_ranking_ties_go_to_whoever_got_there_first():
    s = sort_entries([{"name": "B", "score": 5, "at": 2}, {"name": "A", "score": 5, "at": 1},
                      {"name": "C", "score": 9, "at": 3}])
    assert [e["name"] for e in s] == ["C", "A", "B"]


def test_qualifies():
    assert not qualifies([], 0)
    assert qualifies([], 1)
    full = [{"name": "X", "score": 10 + i, "at": i} for i in range(BOARD_SIZE)]
    assert not qualifies(full, 10), "a tie with last place does not displace it"
    assert qualifies(full, 11)


def test_local_board_keeps_top_ten_per_game_and_survives_restart(tmp_path):
    path = tmp_path / "scores.json"
    b = LocalBoard(path)
    for i in range(1, 13):
        b.submit("snack", "AA", i).result()
    b.submit("flappy", "ZZ", 3).result()
    snack = b.top("snack").result()
    assert len(snack) == BOARD_SIZE and snack[0]["score"] == 12 and snack[-1]["score"] == 3
    assert [e["name"] for e in LocalBoard(path).top("flappy").result()] == ["ZZ"]


def test_local_board_without_a_writable_file_keeps_scores_for_the_run(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    b = LocalBoard(blocker / "sub" / "scores.json")  # parent is a file: can't be created
    _, entry_id = b.submit("snack", "ME", 7).result()
    assert b.top("snack").result()[0]["id"] == entry_id


class FakeServer:
    """Stands in for urlopen against the leaderboard Worker."""

    def __init__(self, refuse: str | None = None):
        self.calls = []
        self.agents = []
        self.refuse = refuse

    def __call__(self, req, timeout=None):
        body = json.loads(req.data) if req.data else None
        self.agents.append(req.get_header("User-agent"))
        self.calls.append((req.get_method(), req.full_url, body))
        if req.full_url.endswith("/sessions"):
            data = {"session": "s-1"}
        elif req.get_method() == "POST" and self.refuse:
            raise urllib.error.HTTPError(req.full_url, 422, "refused", {},
                                         io.BytesIO(json.dumps({"error": self.refuse}).encode()))
        elif req.get_method() == "POST":
            data = {"id": "9", "entries": [{"id": "9", "name": body["name"], "score": body["score"], "at": 1}]}
        else:
            data = {"entries": [{"id": "9", "name": "ACE", "score": 9, "at": 1}]}
        return io.BytesIO(json.dumps(data).encode())


def test_server_board_sends_the_rounds_session_with_its_score():
    fake = FakeServer()
    b = ServerBoard("https://lb.example.dev/", opener=fake)
    session = b.begin("flappy").result()
    assert session == "s-1"
    entries, entry_id = b.submit("flappy", "ace!", 9.4, session).result()
    assert entry_id == "9"
    post = next(c for c in fake.calls if c[0] == "POST" and c[1].endswith("/scores"))
    assert post[2] == {"game": "flappy", "name": "ACE", "score": 9, "session": "s-1"}
    assert b.top("flappy").result()[0]["name"] == "ACE"
    assert ("GET", "https://lb.example.dev/scores?game=flappy", None) in fake.calls
    # Cloudflare answers 403 to Python's default user agent.
    assert all(ua.startswith("kp-arcade-desktop") for ua in fake.agents)


def test_a_refused_score_is_kept_on_this_computer(tmp_path):
    local = LocalBoard(tmp_path / "s.json")
    board = FallbackBoard(ServerBoard("https://lb.example.dev", opener=FakeServer(refuse="too high")), local)
    session = board.begin("snack").result()
    board.submit("snack", "ME", 5, session).result()
    assert board.scope == "device"
    assert local.top("snack").result()[0]["name"] == "ME"


def test_no_session_means_this_computer(tmp_path):
    local = LocalBoard(tmp_path / "s.json")
    fake = FakeServer()
    board = FallbackBoard(ServerBoard("https://lb.example.dev", opener=fake), local)
    board.submit("snack", "ME", 5, None).result()
    assert not any(c[0] == "POST" for c in fake.calls), "nothing posted without a round"


def test_board_choice(tmp_path):
    cfg = tmp_path / "config.js"
    cfg.write_text('export const LEADERBOARD = {\n  url: "https://lb.example.dev",\n};\n')
    assert site_url(cfg) == "https://lb.example.dev"
    cfg.write_text('export const LEADERBOARD = {\n  url: "",\n};\n')
    assert site_url(cfg) == ""
    local = LocalBoard(tmp_path / "s.json")
    assert create_board("local", local) is local
    assert isinstance(create_board("https://lb.example.dev", local), FallbackBoard)


# --- the shell flow ---


class Scored(Game):
    name = "scored"
    title = "SCORED"
    tutorial = [type("Step", (), {"title": "ONE", "text": "Do the thing.", "draw": staticmethod(lambda *a: None)})()]

    def __init__(self, size, **options):
        super().__init__(size, **options)
        self.score = 0
        self.phase = type("P", (), {"value": "playing"})()

    def update(self, controls, dt):
        pass

    def render(self, frame):
        pass


NONE = ControlState()


def step(shell, seconds, controls=NONE):
    for _ in range(round(seconds * 60)):
        shell.update(controls, 1 / 60)


def settle(shell, seconds=0.5):
    """Let background leaderboard work finish and be collected."""
    end = time.time() + seconds
    while time.time() < end:
        shell.update(NONE, 0.0)
        time.sleep(0.005)


def shell_with(tmp_path, entries=(), board=None):
    board = board or LocalBoard(tmp_path / "s.json")
    for name, score in entries:
        board.submit("scored", name, score).result()
    sh = Shell((W, H), {"scored": Scored}, board=board)
    settle(sh, 0.05)
    return sh


def play_to(sh, score):
    sh.handle_key(ord("1"))
    if sh.screen is Screen.TUTORIAL:
        sh.handle_key(KEY_ENTER)
    step(sh, 3.1)
    assert sh.screen is Screen.PLAYING
    sh.game.score = score
    sh.game.phase.value = "game_over"


def test_first_round_shows_the_cards_and_a_replay_does_not(tmp_path):
    sh = shell_with(tmp_path)
    sh.handle_key(ord("1"))
    assert sh.screen is Screen.TUTORIAL
    step(sh, 5)
    assert sh.screen is Screen.TUTORIAL, "the cards wait for the player"
    sh.handle_key(KEY_ENTER)
    assert sh.screen is Screen.COUNTDOWN
    step(sh, 3.1)
    sh.handle_key(ord("r"))
    assert sh.screen is Screen.COUNTDOWN, "restart skips the cards"
    sh.handle_key(KEY_ESC)
    sh.handle_key(ord("1"))
    assert sh.screen is Screen.COUNTDOWN, "already seen this run"


def test_a_score_that_makes_the_board_asks_for_a_name(tmp_path):
    sh = shell_with(tmp_path, [("OLD", 3)])
    play_to(sh, 8)
    step(sh, ENTRY_DELAY + 0.1)
    assert sh.screen is Screen.ENTRY
    for k in "hrxq":  # h, r and q type here; they don't swap hands, restart or quit
        assert sh.handle_key(ord(k)) is False
    assert sh.entry["name"] == "HRX", "three characters at most"
    assert sh.hand == "right"
    sh.handle_key(127)  # backspace
    sh.handle_key(KEY_ENTER)
    settle(sh)
    assert sh.screen is Screen.RESULTS
    rows = sh.scores["scored"]
    assert [(e["name"], e["score"]) for e in rows] == [("HR", 8), ("OLD", 3)]
    assert rows[0]["id"] == sh.last_entry_id
    sh.handle_key(ord("r"))
    assert sh.screen is Screen.COUNTDOWN


def test_a_score_that_misses_the_board_goes_straight_to_it(tmp_path):
    sh = shell_with(tmp_path, [(f"P{i}", 50 + i) for i in range(BOARD_SIZE)])
    play_to(sh, 12)
    step(sh, RESULTS_DELAY + 0.1)
    assert sh.screen is Screen.RESULTS and sh.last_entry_id is None


def test_empty_and_abusive_names_are_refused_and_skip_saves_nothing(tmp_path):
    sh = shell_with(tmp_path)
    play_to(sh, 5)
    step(sh, ENTRY_DELAY + 0.1)
    sh.handle_key(KEY_ENTER)
    assert sh.screen is Screen.ENTRY and sh.entry["message"]
    for k in "ass":
        sh.handle_key(ord(k))
    sh.handle_key(KEY_ENTER)
    assert sh.screen is Screen.ENTRY
    sh.handle_key(KEY_ESC)
    assert sh.screen is Screen.RESULTS
    assert sh.scores.get("scored", []) == []


def test_letters_can_be_typed_by_hovering_the_palm(tmp_path):
    sh = shell_with(tmp_path)
    play_to(sh, 5)
    step(sh, ENTRY_DELAY + 0.1)
    key = next(b for b in sh._layout() if b.action == "key:K")
    x0, y0, x1, y1 = key.rect
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    xy = np.zeros((17, 2), np.float32)
    c = np.zeros(17, np.float32)
    for name, p in (("left_shoulder", (700, 500)), ("right_shoulder", (560, 500)),
                    ("left_elbow", (cx, cy + 138)), ("left_wrist", (cx, cy + 38))):
        xy[KP[name]] = p
        c[KP[name]] = 0.9
    controls = ControlState(present=True, pose=Pose(xy=xy, confidence=c, score=0.9))
    step(sh, 0.5)  # hand away first: the screen change blocks until it leaves
    step(sh, KEY_DWELL + 0.6, controls)
    assert sh.entry["name"] == "K"


def test_clicking_works_like_hovering(tmp_path):
    sh = shell_with(tmp_path)
    x0, y0, _, _ = sh._layout()[0].rect
    sh.handle_click(x0 + 10, y0 + 10)
    assert sh.screen is Screen.TUTORIAL
    sh.handle_key(KEY_ENTER)
    step(sh, 3.1)
    sh.handle_click(1200, 670)  # the PAUSE button
    assert sh.screen is Screen.PAUSED


def test_the_server_round_starts_when_play_begins(tmp_path):
    fake = FakeServer()
    board = FallbackBoard(ServerBoard("https://lb.example.dev", opener=fake), LocalBoard(tmp_path / "s.json"))
    sh = shell_with(tmp_path, board=board)
    settle(sh)
    sh.handle_key(ord("1"))
    sh.handle_key(KEY_ENTER)
    assert not any(c[1].endswith("/sessions") for c in fake.calls), "not during the countdown"
    step(sh, 3.1)
    settle(sh)
    assert sum(c[1].endswith("/sessions") for c in fake.calls) == 1
    sh.game.score = 12
    sh.game.phase.value = "game_over"
    step(sh, ENTRY_DELAY + 0.1)
    for k in "ab":
        sh.handle_key(ord(k))
    sh.handle_key(KEY_ENTER)
    settle(sh)
    post = next(c for c in fake.calls if c[0] == "POST" and c[1].endswith("/scores"))
    assert post[2] == {"game": "scored", "name": "AB", "score": 12, "session": "s-1"}
    assert sh.screen is Screen.RESULTS and sh.board_scope == "world"


@pytest.mark.parametrize("screen", ["tutorial", "entry", "results"])
def test_new_screens_render(tmp_path, screen):
    from kpapp.game.flappy.game import FlappyGame
    from kpapp.game.fruitninja.game import FruitNinjaGame

    board = LocalBoard(tmp_path / "s.json")
    board.submit("fruitninja", "ACE", 40).result()
    sh = Shell((W, H), {"fruitninja": FruitNinjaGame, "flappy": FlappyGame}, board=board)
    settle(sh, 0.05)
    sh.handle_key(ord("1"))
    if screen != "tutorial":
        sh.handle_key(KEY_ENTER)
        step(sh, 3.1)
        sh.game.score = 41
        sh.game.phase = type("P", (), {"value": "game_over"})()
        step(sh, ENTRY_DELAY + 0.1)
        if screen == "results":
            for k in "zed":
                sh.handle_key(ord(k))
            sh.handle_key(KEY_ENTER)
            settle(sh)
    assert sh.screen.value == screen
    frame = np.full((H, W, 3), 80, np.uint8)
    step(sh, 1.0)
    sh.render(frame)
    assert frame.std() > 5


def test_desktop_games_share_the_browser_and_server_leaderboard_ids():
    """A desktop score must land on the same board as the site's, or the
    server refuses it as an unknown game and it stays on this computer.
    Games with songs have one board per song: those ids must match the
    server's pattern for them (rules.js)."""
    import re
    from pathlib import Path

    from kpapp.game import REGISTRY

    root = Path(__file__).resolve().parents[1]
    rules = (root / "leaderboard/src/rules.js").read_text()
    server_fixed = set(re.findall(r"^\s+(\w+): \{ perSecond", rules, re.M))
    server_dance = re.compile(re.search(r"DANCE_ID = /(.+)/;", rules).group(1))
    web = set(re.findall(r'static id = "(\w+)"', "".join(
        p.read_text() for p in (root / "web/js/games").glob("*/game.js"))))
    sh = Shell((W, H), REGISTRY)
    # Desktop games whose browser port (and server rule) is still to come. The
    # port adds the game to web/js/games and leaderboard/src/rules.js and takes
    # it out of here; until then its scores stay on this computer.
    pending_port = {"golf"}
    fixed = {sh._board_name(n) for n in sh.menu_games if not sh.variants(n)} - pending_port
    per_song = [k for n in sh.menu_games if sh.variants(n) for k in sh.board_keys() if k.startswith(sh._board_name(n))]
    assert fixed == server_fixed, (fixed, server_fixed)
    assert fixed <= web, (fixed, web)
    assert per_song, "the dance game lists its songs"
    assert all(server_dance.fullmatch(k) for k in per_song), per_song



# --- variants (songs) ---


class Songs(Scored):
    name = "songs"
    title = "SONGS"
    board_id = "songs"
    tutorial = []
    calls: list = []

    @classmethod
    def variants(cls):
        return [{"id": "a", "title": "Alpha", "subtitle": "first", "board_id": "songs-a"},
                {"id": "b", "title": "Beta", "subtitle": "second", "board_id": "songs-b"}]

    def __init__(self, size, variant=None, **options):
        super().__init__(size, **options)
        self.variant = variant

    def on_pause(self):
        Songs.calls.append(("pause", self.variant))

    def on_resume(self):
        Songs.calls.append(("resume", self.variant))

    def close(self):
        Songs.calls.append(("close", self.variant))


def test_a_game_with_songs_opens_a_song_list_and_each_song_has_its_board(tmp_path):
    board = LocalBoard(tmp_path / "s.json")
    board.submit("songs-b", "BBB", 50).result()
    Songs.calls = []
    sh = Shell((W, H), {"songs": Songs}, board=board)
    settle(sh, 0.05)
    assert sh.scores.get("songs-b")[0]["name"] == "BBB", "every song's board is fetched"
    sh.handle_key(ord("1"))
    assert sh.screen is Screen.SONGS
    assert [b.action for b in sh._layout()] == ["song:a", "song:b", "menu"]
    sh.handle_key(ord("2"))
    assert sh.screen is Screen.COUNTDOWN and sh.game.variant == "b" and sh.board_key == "songs-b"
    step(sh, 3.1)
    sh.handle_key(KEY_SPACE)  # pause
    assert ("pause", "b") in Songs.calls
    sh.handle_key(KEY_SPACE)
    assert ("resume", "b") in Songs.calls
    sh.game.score = 60
    sh.game.phase.value = "game_over"
    step(sh, ENTRY_DELAY + 0.1)
    assert sh.screen is Screen.ENTRY, "60 beats BBB on this song's board"
    for k in "new":
        sh.handle_key(ord(k))
    sh.handle_key(KEY_ENTER)
    settle(sh)
    assert [e["name"] for e in board.top("songs-b").result()] == ["NEW", "BBB"]
    assert board.top("songs-a").result() == []
    sh.handle_key(ord("r"))
    assert sh.game.variant == "b", "play again keeps the song"
    assert ("close", "b") in Songs.calls, "the old round's music stops"
    sh.handle_key(KEY_ESC)
    sh.handle_key(KEY_ESC)
    assert sh.screen is Screen.MENU


def test_song_list_escape_and_click(tmp_path):
    sh = Shell((W, H), {"songs": Songs}, board=LocalBoard(tmp_path / "s.json"))
    sh.handle_key(ord("1"))
    sh.handle_key(KEY_ESC)
    assert sh.screen is Screen.MENU
    sh.handle_key(ord("1"))
    x0, y0, _, _ = sh._layout()[0].rect
    sh.handle_click(x0 + 5, y0 + 5)
    assert sh.screen is Screen.COUNTDOWN and sh.game.variant == "a"
    frame = np.full((H, W, 3), 60, np.uint8)
    sh.to_menu()
    sh.handle_key(ord("1"))
    step(sh, 0.2)
    sh.render(frame)  # the song list draws
    assert frame.std() > 5


def test_autostarting_a_game_with_songs_plays_the_first_song_on_its_board():
    """`kp game --game dance --autostart` skips the song list: the round must
    still be a song's, or its score goes to "dance", a board the server refuses."""
    Songs.calls = []
    sh = Shell((W, H), {"songs": Songs}, selected="songs", autostart=True)
    assert sh.screen is Screen.PLAYING
    assert sh.game.variant == "a" and sh.variant == "a" and sh.board_key == "songs-a"
    sh.restart()
    assert sh.game.variant == "a" and sh.board_key == "songs-a"


def test_games_with_their_own_count_in_skip_the_shell_countdown(tmp_path):
    """Raccoon Dance counts 3-2-1 on the music; a second countdown before it read as a glitch."""
    class Counted(Songs):
        own_countdown = True
        tutorial = [type("Step", (), {"title": "T", "text": "t", "draw": staticmethod(lambda *a: None)})()]

    sh = Shell((W, H), {"songs": Counted}, board=LocalBoard(tmp_path / "s.json"))
    sh.handle_key(ord("1"))
    sh.handle_key(ord("1"))
    assert sh.screen is Screen.TUTORIAL
    sh.handle_key(KEY_ENTER)
    assert sh.screen is Screen.PLAYING, "straight into the song after the cards"
    sh.handle_key(ord("r"))
    assert sh.screen is Screen.PLAYING, "and on play again"


# --- difficulties ---


class Levels(Songs):
    difficulties = [{"id": "easy", "label": "EASY", "hint": "fewer moves"},
                    {"id": "normal", "label": "NORMAL", "hint": "the routine"},
                    {"id": "hard", "label": "HARD", "hint": "strict"}]
    default_difficulty = "normal"

    def __init__(self, size, variant=None, difficulty="normal", **options):
        super().__init__(size, variant=variant, **options)
        self.difficulty = difficulty


def test_difficulty_picker_sets_the_game_and_its_own_board(tmp_path):
    board = LocalBoard(tmp_path / "s.json")
    board.submit("songs-a", "OLD", 50).result()  # scores from before difficulties existed
    sh = Shell((W, H), {"songs": Levels}, board=board)
    settle(sh, 0.05)
    assert {"songs-a", "songs-a_easy", "songs-a_hard", "songs-b_easy"} <= set(sh.board_keys())
    sh.handle_key(ord("1"))
    assert [b.action for b in sh._layout()][:3] == ["diff:easy", "diff:normal", "diff:hard"]
    assert sh.difficulty_of("songs") == "normal"
    line, _ = sh._tile_scoreline("song:a")
    assert "OLD" in line, "normal keeps the song's existing board"
    sh._do("diff:easy")
    assert "NO HIGH SCORES" in sh._tile_scoreline("song:a")[0], "easy has its own board"
    sh.handle_key(ord("1"))
    assert sh.game.difficulty == "easy" and sh.board_key == "songs-a_easy"
    assert sh._round_title() == "ALPHA - EASY"
    sh.handle_key(ord("r"))
    assert sh.game.difficulty == "easy", "play again keeps the difficulty"
    sh.handle_key(KEY_ESC)
    sh.handle_key(KEY_ESC)
    sh.handle_key(ord("1"))
    assert sh.difficulty_of("songs") == "easy", "the choice is remembered"
    sh.handle_key(9)  # TAB cycles
    assert sh.difficulty_of("songs") == "normal"
    frame = np.full((H, W, 3), 60, np.uint8)
    step(sh, 0.2)
    sh.render(frame)
    assert frame.std() > 5


def test_games_without_difficulties_are_unchanged(tmp_path):
    sh = Shell((W, H), {"songs": Songs}, board=LocalBoard(tmp_path / "s.json"))
    sh.handle_key(ord("1"))
    assert not any(b.action.startswith("diff:") for b in sh._layout())
    sh.handle_key(ord("1"))
    assert sh.difficulty is None and sh.board_key == "songs-a"
