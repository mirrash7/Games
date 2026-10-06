"""The arcade shell: nothing starts on its own, and start/pause/stop all work
by hand-hover as well as by key."""

from __future__ import annotations

from enum import Enum

import numpy as np

from kpapp.config import KP
from kpapp.controls import ControlState
from kpapp.game.base import Game
from kpapp.inference import Pose
from kpapp.shell import KEY_ESC, KEY_SPACE, Screen, Shell

W, H = 1280, 720
DT = 1 / 60


class _Phase(str, Enum):
    PLAYING = "playing"
    GAME_OVER = "game_over"


class FakeGame(Game):
    name = "fake"
    title = "FAKE GAME"
    blurb = "A stand-in."

    def __init__(self, size, **options):
        super().__init__(size, **options)
        self.updates = 0
        self.resumed = 0
        self.phase = _Phase.PLAYING

    def update(self, controls, dt):
        self.updates += 1

    def render(self, frame):
        pass

    def on_resume(self):
        self.resumed += 1


class OtherGame(FakeGame):
    name = "other"
    title = "OTHER GAME"


GAMES = {"fake": FakeGame, "other": OtherGame}


def hand_at(x: float, y: float) -> ControlState:
    """Player's real right hand at (x, y) - left_wrist on a mirrored feed."""
    xy = np.zeros((17, 2), np.float32)
    c = np.zeros(17, np.float32)
    xy[KP["left_wrist"]] = (x, y)
    c[KP["left_wrist"]] = 0.9
    return ControlState(present=True, pose=Pose(xy=xy, confidence=c, score=0.9))


NOBODY = ControlState(present=False)


def run(shell: Shell, controls: ControlState, seconds: float) -> None:
    for _ in range(int(seconds / DT)):
        shell.update(controls, DT)


def centre(shell: Shell, action: str) -> tuple[float, float]:
    shell._buttons = shell._layout()
    b = next(b for b in shell._buttons if b.action == action)
    x0, y0, x1, y1 = b.rect
    return (x0 + x1) / 2, (y0 + y1) / 2


def playing(shell: Shell) -> Shell:
    shell.handle_key(ord("1"))
    run(shell, NOBODY, 3.2)  # countdown
    assert shell.screen is Screen.PLAYING
    return shell


# --- nothing starts on its own ---


def test_opens_on_the_welcome_screen_with_no_game_running():
    shell = Shell((W, H), GAMES)
    run(shell, NOBODY, 10.0)
    assert shell.screen is Screen.MENU
    assert shell.game is None


def test_lists_every_titled_game():
    assert Shell((W, H), GAMES).menu_games == ["fake", "other"]


def test_autostart_skips_the_menu_for_benchmarks():
    shell = Shell((W, H), GAMES, selected="other", autostart=True)
    assert shell.screen is Screen.PLAYING
    assert isinstance(shell.game, OtherGame)


# --- starting ---


def test_number_key_starts_that_game_after_a_countdown():
    shell = Shell((W, H), GAMES)
    shell.handle_key(ord("2"))
    assert shell.screen is Screen.COUNTDOWN
    assert isinstance(shell.game, OtherGame)
    run(shell, NOBODY, 1.0)
    assert shell.game.updates == 0, "the game must not run during the countdown"
    run(shell, NOBODY, 2.2)
    assert shell.screen is Screen.PLAYING


def test_hovering_a_game_starts_it():
    shell = Shell((W, H), GAMES)
    hand = hand_at(*centre(shell, "start:fake"))
    run(shell, hand, 0.6)
    assert shell.screen is Screen.MENU, "fired too early"
    run(shell, hand, 0.6)
    assert shell.screen is Screen.COUNTDOWN
    assert isinstance(shell.game, FakeGame)


def test_moving_off_a_tile_cancels_the_hover():
    shell = Shell((W, H), GAMES)
    tile = hand_at(*centre(shell, "start:fake"))
    run(shell, tile, 0.7)
    run(shell, hand_at(20, 20), 0.2)
    run(shell, tile, 0.7)
    assert shell.screen is Screen.MENU


# --- stopping ---


def test_space_pauses_and_the_game_stops_updating():
    shell = playing(Shell((W, H), GAMES))
    shell.handle_key(KEY_SPACE)
    assert shell.screen is Screen.PAUSED
    before = shell.game.updates
    run(shell, NOBODY, 1.0)
    assert shell.game.updates == before
    shell.handle_key(KEY_SPACE)
    assert shell.screen is Screen.PLAYING
    assert shell.game.resumed == 1, "games must be told play resumed"


def test_holding_the_hand_on_pause_pauses():
    shell = playing(Shell((W, H), GAMES))
    b = shell._pause_button()
    x0, y0, x1, y1 = b.rect
    run(shell, hand_at((x0 + x1) / 2, (y0 + y1) / 2), 1.7)
    assert shell.screen is Screen.PAUSED


def test_a_passing_hand_does_not_pause():
    shell = playing(Shell((W, H), GAMES))
    x0, y0, x1, y1 = shell._pause_button().rect
    run(shell, hand_at((x0 + x1) / 2, (y0 + y1) / 2), 0.8)
    assert shell.screen is Screen.PLAYING


def test_escape_from_play_returns_to_the_menu_and_only_quits_from_there():
    shell = playing(Shell((W, H), GAMES))
    assert shell.handle_key(KEY_ESC) is False
    assert shell.screen is Screen.MENU and shell.game is None
    assert shell.handle_key(KEY_ESC) is True


def test_hovering_resume_resumes():
    shell = playing(Shell((W, H), GAMES))
    shell.handle_key(KEY_SPACE)
    run(shell, NOBODY, 0.1)  # hand away, so the block clears
    run(shell, hand_at(*centre(shell, "resume")), 1.2)
    assert shell.screen is Screen.PLAYING


# --- game over ---


def test_game_over_offers_play_again_by_hover():
    shell = playing(Shell((W, H), GAMES))
    shell.game.phase = _Phase.GAME_OVER
    run(shell, NOBODY, 0.1)
    old = shell.game
    run(shell, hand_at(*centre(shell, "again")), 1.2)
    assert shell.screen is Screen.COUNTDOWN
    assert shell.game is not old, "play again must start a fresh game"


def test_r_plays_again_after_game_over():
    shell = playing(Shell((W, H), GAMES))
    shell.game.phase = _Phase.GAME_OVER
    shell.handle_key(ord("r"))
    assert shell.screen is Screen.COUNTDOWN


# --- buttons that overlap across screens ---


def test_a_resting_hand_does_not_click_through_a_screen_change():
    """MENU on the pause screen sits inside a game tile on the welcome screen.

    Pressing it must not leave the hand dwelling on that tile and launch a new
    game a second later: after any screen change, the hand has to move off
    first.
    """
    shell = playing(Shell((W, H), GAMES))
    shell.handle_key(KEY_SPACE)
    run(shell, NOBODY, 0.1)
    menu_btn = hand_at(*centre(shell, "menu"))
    run(shell, menu_btn, 1.2)
    assert shell.screen is Screen.MENU
    run(shell, menu_btn, 3.0)  # hand never moves
    assert shell.screen is Screen.MENU, "a resting hand launched a game"


def test_hand_already_on_play_again_must_move_off_first():
    """The game can end while the hand happens to rest where PLAY AGAIN appears."""
    shell = playing(Shell((W, H), GAMES))
    shell.game.phase = _Phase.GAME_OVER
    spot = hand_at(*centre(shell, "again"))
    run(shell, spot, 2.0)
    assert shell.screen is Screen.PLAYING and shell.game_over, "restarted without the hand moving"


# --- hands ---


def test_menu_cursor_falls_back_to_the_other_hand():
    shell = Shell((W, H), GAMES)
    xy = np.zeros((17, 2), np.float32)
    c = np.zeros(17, np.float32)
    xy[KP["right_wrist"]] = (300, 300)  # player's LEFT hand, mirrored
    c[KP["right_wrist"]] = 0.9
    shell.update(ControlState(present=True, pose=Pose(xy=xy, confidence=c, score=0.9)), DT)
    assert shell.cursor is not None


def test_render_every_screen_without_error():
    shell = Shell((W, H), GAMES)
    frame = np.zeros((H, W, 3), np.uint8)
    shell.render(frame)
    shell.handle_key(ord("1")); shell.render(frame)
    run(shell, NOBODY, 3.2); shell.render(frame)
    shell.handle_key(KEY_SPACE); shell.render(frame)
    shell.handle_key(KEY_SPACE)
    shell.game.phase = _Phase.GAME_OVER
    shell.update(NOBODY, DT); shell.render(frame)
