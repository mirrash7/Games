"""The arcade around the games: welcome screen, how-to-play cards, start,
pause, stop, replay, and the leaderboards (the browser build's web/js/shell.js
is a port of this file; keep them in step).

Games used to start the instant the window opened. On recorded play that meant
fruit was already falling while the player was still stepping back, and three
rounds ended in about three seconds each. Now nothing starts until the player
chooses, and a countdown gives them time to get into position.

Everything can be done standing back from the computer: hovering the hand over
a button for a moment presses it, Kinect-style. Keys and mouse clicks do the
same for anyone at the keyboard.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from .controls import ControlState
from .hand import HandTracker
from .leaderboard import BOARD_SIZE, NAME_MAX, clean_name, name_allowed, qualifies
from .overlay import outlined_text
from .theme import ACCENT as _ACCENT
from .theme import CYAN as _CYAN
from .theme import DIM as _DIM
from .theme import GOOD as _GOOD
from .theme import WARN as _WARN
from .theme import PANEL as _PANEL
from .theme import WHITE as _WHITE
from .theme import centred_text as _text
from .theme import panel as _panel
from .theme import wrap as _wrap

COUNTDOWN_SECONDS = 3.0
MENU_DWELL = 1.0  # hover time to press a button
PAUSE_DWELL = 1.5  # longer in play, so a passing hand cannot pause by accident
KEY_DWELL = 0.7  # name entry: quicker, three letters should not take forever
ENTRY_DELAY = 1.6  # seconds of the game's own game-over art before asking for a name
RESULTS_DELAY = 2.2  # ...or before showing the leaderboard
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

KEY_ESC, KEY_ENTER, KEY_SPACE = 27, 13, 32
KEY_BACKSPACE = (8, 127)
_HOT_FILL = (46, 38, 34)
_IDLE_BORDER = (110, 110, 120)
_HINT = (150, 150, 165)


class Screen(str, Enum):
    MENU = "menu"
    TUTORIAL = "tutorial"  # how-to-play cards, before a player's first round of a game
    COUNTDOWN = "countdown"
    PLAYING = "playing"
    PAUSED = "paused"
    ENTRY = "entry"  # typing a name for the leaderboard
    RESULTS = "results"  # the leaderboard after a round


@dataclass
class Button:
    action: str
    label: str
    rect: tuple[int, int, int, int]  # x0, y0, x1, y1
    sub: str = ""
    key_hint: str = ""

    def contains(self, p: np.ndarray | None, pad: int = 0) -> bool:
        if p is None:
            return False
        x0, y0, x1, y1 = self.rect
        return x0 - pad <= p[0] <= x1 + pad and y0 - pad <= p[1] <= y1 + pad


class Dwell:
    """Hover-to-press. Holding the hand on a button fills it, then it fires.

    After firing, the same button will not fire again until the hand leaves
    it, so a hand left resting on PLAY AGAIN does not loop forever.
    """

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.target: str | None = None
        self.progress = 0.0
        self._spent: str | None = None

    def reset(self) -> None:
        self.target, self.progress, self._spent = None, 0.0, None

    def block(self) -> None:
        """Ignore everything until the hand is over no button at all.

        Used on every screen change: buttons on the new screen can sit right
        where the hand already is (MENU on the pause screen overlaps a game
        tile on the welcome screen), and must not fire just because of that.
        """
        self.target, self.progress, self._spent = None, 0.0, "*"

    def update(self, hovered: str | None, dt: float) -> str | None:
        if self._spent == "*":
            if hovered is None:
                self._spent = None
            return None
        if hovered != self._spent:
            self._spent = None
        if hovered is None or hovered == self._spent:
            self.target, self.progress = hovered if hovered == self._spent else None, 0.0
            return None
        if hovered != self.target:
            self.target, self.progress = hovered, 0.0
        self.progress += dt / self.seconds
        if self.progress >= 1.0:
            self.progress = 0.0
            self._spent = hovered
            return hovered
        return None


class Shell:
    """Owns the current game and the screens around it."""

    def __init__(
        self,
        size: tuple[int, int],
        games: dict,
        mirrored: bool = True,
        selected: str | None = None,
        autostart: bool = False,
        board=None,
    ) -> None:
        """`board`: a leaderboard (leaderboard.py); None turns leaderboards off."""
        self.width, self.height = size
        self.mirrored = mirrored
        self.registry = games
        # The menu lists games that describe themselves; a directly requested
        # game is included even if it does not.
        self.menu_games = [n for n, g in games.items()
                           if getattr(g, "title", "") and getattr(g, "menu", True)]
        if selected and selected not in self.menu_games:
            self.menu_games.append(selected)
        self.selected = selected or (self.menu_games[0] if self.menu_games else None)
        self.hand = "right"
        self._trackers = {h: HandTracker(h, mirrored) for h in ("right", "left")}
        self._clock = 0.0

        self.screen = Screen.MENU
        self.game = None
        self.countdown = 0.0
        self.cursor: np.ndarray | None = None
        self.person_seen = False
        self.dwell = Dwell(MENU_DWELL)
        self.pause_dwell = Dwell(PAUSE_DWELL)
        self._buttons: list[Button] = []
        self._was_over = False

        self.board = board
        self.scores: dict[str, list[dict]] = {}  # game -> top entries, as last fetched
        self._fetching: dict[str, object] = {}  # game -> pending Future
        self.last_entry_id: str | None = None  # the player's own row on the results screen
        self.key_dwell = Dwell(KEY_DWELL)
        self.entry: dict | None = None  # {name, message, saving} while entering a name
        self._session = None  # Future of the leaderboard round, from play start
        self._saving = None  # Future of a score being saved
        self._over_t = 0.0
        self._qualified = False
        self._tutored: set[str] = set()  # games whose cards were seen this run
        self.tutorial_t = 0.0
        self.refresh_scores()

        if autostart and self.selected:
            self.start(self.selected, countdown=False)

    # --- transitions ---

    def start(self, name: str, countdown: bool = True) -> None:
        self.selected = name
        self.game = self.registry[name]((self.width, self.height), mirrored=self.mirrored)
        if self.hand != "right" and hasattr(self.game, "swap_hand"):
            self.game.swap_hand()
        self.countdown = COUNTDOWN_SECONDS
        self.screen = Screen.COUNTDOWN if countdown else Screen.PLAYING
        self._session = None
        if not countdown:
            self._begin_play()
        # A player's first round of each game starts with how to play it.
        cls = self.registry[name]
        if countdown and getattr(cls, "tutorial", None) and name not in self._tutored:
            self.screen = Screen.TUTORIAL
            self.tutorial_t = 0.0
        self._was_over = False
        self.entry = None
        self.last_entry_id = None
        self._saving = None
        self.dwell.reset()
        self.pause_dwell.reset()

    def _begin_play(self) -> None:
        """Play starts: tell the leaderboard, which times the round on its side."""
        self.screen = Screen.PLAYING
        self._session = self.board.begin(self._board_name(self.selected)) if self.board is not None else None

    def end_tutorial(self) -> None:
        """Leave the how-to-play cards for the countdown."""
        if self.screen is not Screen.TUTORIAL:
            return
        self._tutored.add(self.selected)
        self.screen = Screen.COUNTDOWN
        self.countdown = COUNTDOWN_SECONDS
        self.dwell.block()

    # --- leaderboard ---

    def refresh_scores(self, names: list[str] | None = None) -> None:
        """Fetch the boards again (on start-up and when going back to the menu)."""
        if self.board is None:
            return
        for name in names or self.menu_games:
            if name not in self._fetching:
                self._fetching[name] = self.board.top(self._board_name(name))

    def _poll_board(self) -> None:
        """Collect finished leaderboard work. Never blocks."""
        for name, fut in list(self._fetching.items()):
            if fut.done():
                del self._fetching[name]
                try:
                    self.scores[name] = fut.result()
                except Exception as exc:  # noqa: BLE001
                    print(f"[kp] leaderboard: {exc}")
        e = self.entry
        if e is not None and e["saving"] and self._saving is None:
            # Saving waits for the round's session, then posts the score.
            if self._session is None or self._session.done():
                session = None
                if self._session is not None:
                    try:
                        session = self._session.result()
                    except Exception:  # noqa: BLE001
                        session = None
                self._saving = self.board.submit(self._board_name(self.selected), e["name"], self.score, session)
        if self._saving is not None and self._saving.done():
            fut, self._saving = self._saving, None
            try:
                entries, entry_id = fut.result()
                self.scores[self.selected] = entries
                if self.screen is Screen.ENTRY:
                    self.last_entry_id = entry_id
                    self.show_results()
            except Exception as exc:  # noqa: BLE001
                print(f"[kp] could not save the score: {exc}")
                if self.screen is Screen.ENTRY and self.entry is not None:
                    self.entry["saving"] = False
                    self.entry["message"] = "Couldn't save - try again, or SKIP"

    def _board_name(self, name: str) -> str:
        """The game's name on the shared leaderboard (the browser build's id)."""
        return getattr(self.registry[name], "board_id", name)

    @property
    def board_scope(self) -> str:
        return getattr(self.board, "scope", "device")

    @property
    def score(self) -> int:
        return max(0, int(round(getattr(self.game, "score", 0) or 0)))

    def enter_name(self) -> None:
        """The name-entry screen, for a score that made the board."""
        self.screen = Screen.ENTRY
        self.entry = {"name": "", "message": "", "saving": False}
        self.key_dwell.block()

    def _type(self, action: str) -> None:
        e = self.entry
        if e is None or e["saving"]:
            return
        e["message"] = ""
        if action == "del":
            e["name"] = e["name"][:-1]
        elif action == "ok":
            if not e["name"]:
                e["message"] = "Pick at least one letter"
            elif not name_allowed(e["name"]):
                e["message"] = "Please pick another name"
            else:
                e["saving"] = True  # _poll_board posts it once the round's session is known
        elif action == "skip":
            self.show_results()
        elif action.startswith("key:") and len(e["name"]) < NAME_MAX:
            e["name"] = clean_name(e["name"] + action[4:])

    def show_results(self) -> None:
        self.screen = Screen.RESULTS
        self.entry = None
        self.dwell.block()

    def restart(self) -> None:
        if self.selected:
            self.start(self.selected)

    def pause(self) -> None:
        if self.screen is Screen.PLAYING and not self.game_over:
            self.screen = Screen.PAUSED
            self.dwell.block()

    def resume(self) -> None:
        if self.screen is Screen.PAUSED:
            self.screen = Screen.PLAYING
            self.pause_dwell.reset()
            self.game.on_resume()

    def to_menu(self) -> None:
        self.screen = Screen.MENU
        self.game = None
        self.entry = None
        self.dwell.block()
        self.refresh_scores()

    def swap_hand(self) -> None:
        self.hand = "left" if self.hand == "right" else "right"
        if self.game is not None and hasattr(self.game, "swap_hand"):
            self.game.swap_hand()

    @property
    def game_over(self) -> bool:
        phase = getattr(self.game, "phase", None)
        return getattr(phase, "value", None) == "game_over"

    def _do(self, action: str) -> None:
        if action.startswith("start:"):
            self.start(action.split(":", 1)[1])
        elif action == "resume":
            self.resume()
        elif action == "menu":
            self.to_menu()
        elif action == "again":
            self.restart()
        elif action == "pause":
            self.pause()
        elif action == "play":
            self.end_tutorial()
        elif action.startswith("key:") or action in ("del", "ok", "skip"):
            self._type(action)

    # --- input ---

    def handle_key(self, key: int) -> bool:
        """Apply a keypress. Returns True if the app should quit."""
        if self.screen is Screen.ENTRY:
            # Every key types here, H and R included.
            ch = chr(key).lower() if 0 <= key < 128 else ""
            if ch.isascii() and ch.isalnum():
                self._type(f"key:{ch.upper()}")
            elif key in KEY_BACKSPACE:
                self._type("del")
            elif key == KEY_ENTER:
                self._type("ok")
            elif key == KEY_ESC:
                self._type("skip")
            return False
        if key == ord("h"):
            self.swap_hand()
            return False

        if self.screen is Screen.MENU:
            if key == KEY_ESC:
                return True
            if ord("1") <= key <= ord("9"):
                i = key - ord("1")
                if i < len(self.menu_games):
                    self.start(self.menu_games[i])
            elif key in (KEY_ENTER, KEY_SPACE) and self.selected:
                self.start(self.selected)
        elif self.screen is Screen.TUTORIAL:
            if key in (KEY_ENTER, KEY_SPACE):
                self.end_tutorial()
            elif key == KEY_ESC:
                self.to_menu()
        elif self.screen is Screen.RESULTS:
            if key in (ord("r"), KEY_SPACE, KEY_ENTER):
                self.restart()
            elif key == KEY_ESC:
                self.to_menu()
        elif self.screen is Screen.COUNTDOWN:
            if key == KEY_ESC:
                self.to_menu()
        elif self.screen is Screen.PAUSED:
            if key in (KEY_SPACE, ord("p")):
                self.resume()
            elif key == ord("r"):
                self.restart()
            elif key == KEY_ESC:
                self.to_menu()
        elif self.screen is Screen.PLAYING:
            if self.game_over:
                if self._qualified and key in (KEY_SPACE, KEY_ENTER):
                    self.enter_name()
                elif key in (ord("r"), KEY_SPACE, KEY_ENTER):
                    self.restart()
                elif key == KEY_ESC:
                    self.to_menu()
            elif key in (KEY_SPACE, ord("p")):
                self.pause()
            elif key == ord("r"):
                self.restart()
            elif key == KEY_ESC:
                self.to_menu()
        return False

    def handle_click(self, x: float, y: float) -> None:
        """A mouse click at frame coordinates."""
        p = np.array([x, y], np.float32)
        if self.screen is Screen.PLAYING and not self.game_over:
            if self._pause_button().contains(p):
                self.pause()
            return
        for b in self._layout():
            if b.contains(p, pad=6):
                self._do(b.action)
                return

    def _locate_hand(self, controls: ControlState, dt: float) -> None:
        self._clock += dt
        pose = controls.pose
        self.person_seen = pose is not None
        other = "left" if self.hand == "right" else "right"
        main = self._trackers[self.hand].update(pose, self._clock)
        spare = self._trackers[other].update(pose, self._clock)
        # Menus are forgiving: point with whichever hand is up.
        self.cursor = main if main is not None else spare

    def update(self, controls: ControlState, dt: float) -> None:
        self._locate_hand(controls, dt)
        self._poll_board()
        self._buttons = self._layout()

        if self.screen is Screen.COUNTDOWN:
            self.countdown -= dt
            if self.countdown <= 0.0:
                self._begin_play()
            return
        if self.screen is Screen.TUTORIAL:
            self.tutorial_t += dt

        if self.screen is Screen.PLAYING and not self.game_over:
            self.game.update(controls, dt)
            pause_btn = self._pause_button()
            fired = self.pause_dwell.update(
                pause_btn.action if pause_btn.contains(self.cursor) else None, dt)
            if fired:
                self.pause()
            return

        if self.screen is Screen.PLAYING and self.game_over:
            if not self._was_over:
                self.dwell.block()  # PLAY AGAIN may appear under the hand
                self._over_t = 0.0
                # A score good enough for the board: no PLAY AGAIN button to hit
                # by accident; the name entry follows the game's own game-over moment.
                self._qualified = (self.board is not None
                                   and qualifies(self.scores.get(self.selected, []), self.score))
            self._over_t += dt
            self.game.update(controls, dt)  # let debris finish falling
            if self.board is not None and self._over_t >= (ENTRY_DELAY if self._qualified else RESULTS_DELAY):
                if self._qualified:
                    self.enter_name()
                else:
                    self.screen = Screen.RESULTS  # same buttons, same place: keep the hover going
                self._was_over = False
                self._buttons = self._layout()
                return
        self._was_over = self.screen is Screen.PLAYING and self.game_over
        self._buttons = self._layout()

        entry = self.screen is Screen.ENTRY
        dwell = self.key_dwell if entry else self.dwell
        hovered = next((b.action for b in self._buttons if b.contains(self.cursor, pad=0 if entry else 6)), None)
        fired = dwell.update(hovered, dt)
        if fired:
            self._do(fired)

    # --- layout ---

    def _pause_button(self) -> Button:
        w, h = self.width, self.height
        return Button("pause", "II  PAUSE", (w - 190, h - 74, w - 22, h - 22))

    def _layout(self) -> list[Button]:
        w, h = self.width, self.height
        if self.screen is Screen.MENU:
            n = max(1, len(self.menu_games))
            tw, th, gap = int(w * 0.30), int(h * 0.34), int(w * 0.04)
            total = n * tw + (n - 1) * gap
            x = (w - total) // 2
            y0 = int(h * 0.36)
            buttons = []
            for i, name in enumerate(self.menu_games):
                g = self.registry[name]
                buttons.append(Button(f"start:{name}", getattr(g, "title", "") or name.upper(),
                                      (x, y0, x + tw, y0 + th), getattr(g, "blurb", ""), str(i + 1)))
                x += tw + gap
            return buttons
        if self.screen is Screen.ENTRY:
            return self._keyboard()
        bw, bh, gap = int(w * 0.20), 74, int(w * 0.04)
        y0 = int(h * 0.56) if self.screen is Screen.PAUSED else int(h * 0.80)
        if self.screen is Screen.PAUSED:
            pair = (("resume", "RESUME", "SPACE"), ("menu", "MENU", "ESC"))
        elif self.screen is Screen.TUTORIAL:
            pair = (("play", "LET'S GO", "ENTER"), ("menu", "MENU", "ESC"))
        elif self.screen is Screen.RESULTS or (self.screen is Screen.PLAYING and self.game_over
                                               and not self._qualified):
            pair = (("again", "PLAY AGAIN", "R"), ("menu", "MENU", "ESC"))
        else:
            return []
        x = w // 2 - bw - gap // 2
        out = []
        for action, label, key in pair:
            out.append(Button(action, label, (x, y0, x + bw, y0 + bh), key_hint=key))
            x += bw + gap
        return out

    def _keyboard(self) -> list[Button]:
        """A-Z, DEL and OK in a 7x4 grid, big enough to hit with a palm from 2 m; SKIP below."""
        cols, kw, kh, gap = 7, 112, 78, 10
        x0, y0 = (self.width - (cols * kw + (cols - 1) * gap)) // 2, 250
        keys = []
        for i, label in enumerate([*LETTERS, "DEL", "OK"]):
            r, c = divmod(i, cols)
            x, y = x0 + c * (kw + gap), y0 + r * (kh + gap)
            action = {"DEL": "del", "OK": "ok"}.get(label, f"key:{label}")
            keys.append(Button(action, label, (x, y, x + kw, y + kh)))
        sw = 180
        keys.append(Button("skip", "SKIP", (self.width // 2 - sw // 2, 612, self.width // 2 + sw // 2, 664),
                           key_hint="ESC"))
        return keys

    # --- drawing ---

    def render(self, canvas: np.ndarray) -> None:
        self._buttons = self._layout()
        if self.screen is Screen.MENU:
            self._draw_menu(canvas)
        elif self.screen is Screen.TUTORIAL:
            self._draw_tutorial(canvas)
        else:
            self.game.render(canvas)
            if self.screen is Screen.COUNTDOWN:
                self._draw_countdown(canvas)
            elif self.screen is Screen.PAUSED:
                self._draw_paused(canvas)
            elif self.screen is Screen.ENTRY:
                self._draw_entry(canvas)
            elif self.screen is Screen.RESULTS:
                self._draw_results(canvas)
            elif self.game_over:
                if self._qualified:
                    pulse = 1.0 + 0.06 * np.sin(self._over_t * 9.0)
                    _text(canvas, "NEW HIGH SCORE!", self.width // 2, int(self.height * 0.86), 1.2 * pulse, _CYAN, 3)
                self._draw_buttons(canvas, self.dwell)
                self._draw_cursor(canvas, self.dwell)
            else:
                self._draw_play_chrome(canvas)
        self._draw_footer(canvas)

    def _dim(self, canvas, amount: float = 0.55) -> None:
        cv2.addWeighted(canvas, 1.0 - amount, np.zeros_like(canvas), amount, 0, canvas)

    def _draw_menu(self, canvas) -> None:
        self._dim(canvas, 0.62)
        w, h = self.width, self.height
        _text(canvas, "MOTION ARCADE", w // 2, int(h * 0.17), 2.0, _ACCENT, 4)
        _text(canvas, "Point with your right hand and hover over a game to start",
              w // 2, int(h * 0.25), 0.72, _WHITE, 1, cv2.FONT_HERSHEY_SIMPLEX)
        self._draw_buttons(canvas, self.dwell, tiles=True)

        if not self.person_seen:
            status, colour = "Step in front of the camera", (90, 180, 255)
        elif self.cursor is None:
            status, colour = "Raise your hand to point", (90, 180, 255)
        else:
            status, colour = "Hand found - hold it over a game", (140, 235, 150)
        _text(canvas, status, w // 2, int(h * 0.82), 0.8, colour, 2)
        self._draw_cursor(canvas, self.dwell)

    def _draw_buttons(self, canvas, dwell: Dwell, tiles: bool = False) -> None:
        for b in self._buttons:
            hot = dwell.target == b.action
            x0, y0, x1, y1 = b.rect
            _panel(canvas, b.rect, colour=(46, 38, 34) if hot else _PANEL, alpha=0.85,
                   border=_ACCENT if hot else (110, 110, 120), thickness=3 if hot else 2)
            cx = (x0 + x1) // 2
            if tiles:
                _text(canvas, b.label, cx, y0 + 62, 1.05, _ACCENT if hot else _WHITE, 2)
                for i, line in enumerate(_wrap(b.sub, x1 - x0 - 40, 0.58)):
                    _text(canvas, line, cx, y0 + 108 + i * 28, 0.58, _DIM, 1,
                          cv2.FONT_HERSHEY_SIMPLEX, shadow=False)
                _text(canvas, f"hover to play  -  or press {b.key_hint}", cx, y1 - 24, 0.5,
                      (150, 150, 165), 1, cv2.FONT_HERSHEY_SIMPLEX, shadow=False)
                if self.board is not None:
                    top = (self.scores.get(b.action.split(":", 1)[1]) or [None])[0]
                    line = f"HIGH SCORE   {top['name']}   {top['score']}" if top else "NO HIGH SCORES YET - SET ONE"
                    _text(canvas, line, cx, y1 - 62, 0.56, _ACCENT if top else _DIM, 1, shadow=False)
            else:
                _text(canvas, b.label, cx, y0 + 44, 0.85, _ACCENT if hot else _WHITE, 2)
                if b.key_hint:
                    _text(canvas, b.key_hint, cx, y1 + 22, 0.45, (150, 150, 165), 1,
                          cv2.FONT_HERSHEY_SIMPLEX, shadow=False)
            if hot and dwell.progress > 0:
                fill = int((x1 - x0 - 24) * min(1.0, dwell.progress))
                cv2.rectangle(canvas, (x0 + 12, y1 - 10), (x0 + 12 + fill, y1 - 5), _ACCENT, -1)

    def _draw_cursor(self, canvas, dwell: Dwell) -> None:
        if self.cursor is None:
            return
        c = (int(self.cursor[0]), int(self.cursor[1]))
        cv2.circle(canvas, c, 22, (20, 20, 20), 5, cv2.LINE_AA)
        cv2.circle(canvas, c, 22, _WHITE, 2, cv2.LINE_AA)
        cv2.circle(canvas, c, 5, _WHITE, -1, cv2.LINE_AA)
        if dwell.target and dwell.progress > 0:
            cv2.ellipse(canvas, c, (30, 30), -90, 0, 360 * min(1.0, dwell.progress),
                        _ACCENT, 5, cv2.LINE_AA)

    def _draw_tutorial(self, canvas) -> None:
        self._dim(canvas, 0.78)
        w = self.width
        cls = self.registry[self.selected]
        _text(canvas, f"HOW TO PLAY {cls.title}", w // 2, 84, 1.25, _ACCENT, 3)
        steps = cls.tutorial
        n = len(steps)
        cw, gap, ch, y0 = 370, 30, 380, 120
        x = (w - (n * cw + (n - 1) * gap)) // 2
        for i, step in enumerate(steps):
            # Cards appear one after another, so the eye reads them in order.
            a = min(1.0, max(0.0, (self.tutorial_t - i * 0.25) / 0.35))
            if a <= 0.0:
                x += cw + gap
                continue
            rect = (x, y0, x + cw, y0 + ch)
            before = canvas[y0:y0 + ch + 1, x:x + cw + 1].copy() if a < 1.0 else None
            _panel(canvas, rect, alpha=0.9, border=_IDLE_BORDER)
            step.draw(canvas, (x + 14, y0 + 14, x + cw - 14, y0 + 214), self.tutorial_t)
            _text(canvas, str(i + 1), x + 34, y0 + 48, 0.9, _ACCENT, 2)  # over the demo
            _text(canvas, step.title, x + cw // 2, y0 + 256, 0.7, _WHITE, 2)
            for j, line in enumerate(_wrap(step.text, cw - 44, 0.52)):
                _text(canvas, line, x + cw // 2, y0 + 292 + j * 26, 0.52, _DIM, 1,
                      cv2.FONT_HERSHEY_SIMPLEX, shadow=False)
            if before is not None:
                roi = canvas[y0:y0 + ch + 1, x:x + cw + 1]
                cv2.addWeighted(roi, a, before, 1.0 - a, 0, roi)
            x += cw + gap
        self._draw_buttons(canvas, self.dwell)
        self._draw_cursor(canvas, self.dwell)

    def _draw_entry(self, canvas) -> None:
        self._dim(canvas, 0.88)  # the game's own GAME OVER text would show through the slots
        w, e = self.width, self.entry
        cls = self.registry[self.selected]
        rank = 1 + sum(1 for x in self.scores.get(self.selected, []) if x["score"] >= self.score)
        _text(canvas, "NEW HIGH SCORE!", w // 2, 76, 1.35, _ACCENT, 3)
        _text(canvas, f"{cls.title}   {self.score}   #{rank}", w // 2, 116, 0.65, _WHITE, 2)
        # Three slots; the next one to fill blinks.
        sw, sg = 78, 16
        sx = w // 2 - (NAME_MAX * sw + (NAME_MAX - 1) * sg) // 2
        for i in range(NAME_MAX):
            x = sx + i * (sw + sg)
            active = i == len(e["name"]) and not e["saving"] and int(self._clock * 2.5) % 2 == 0
            _panel(canvas, (x, 136, x + sw, 222), alpha=0.9, radius=12,
                   border=_CYAN if active else _IDLE_BORDER, thickness=3 if active else 2)
            if i < len(e["name"]):
                _text(canvas, e["name"][i], x + sw // 2, 200, 1.6, _ACCENT, 3)
        msg = "SAVING..." if e["saving"] else (e["message"] or "Hover over letters to spell your name - or type it")
        _text(canvas, msg, w // 2, 242, 0.5, _WARN if e["message"] else _DIM, 1,
              cv2.FONT_HERSHEY_SIMPLEX, shadow=False)
        for b in self._buttons:
            hot = self.key_dwell.target == b.action
            ready = b.action == "ok" and len(e["name"]) > 0
            x0, y0, x1, y1 = b.rect
            _panel(canvas, b.rect, colour=_HOT_FILL if hot else _PANEL, alpha=0.88, radius=12,
                   border=_ACCENT if hot else _CYAN if ready else _IDLE_BORDER, thickness=3 if hot or ready else 2)
            small = len(b.label) > 1
            _text(canvas, b.label, (x0 + x1) // 2, (y0 + y1) // 2 + (10 if small else 15), 0.72 if small else 1.05,
                  _ACCENT if hot else _CYAN if ready else _WHITE, 2)
            if hot and self.key_dwell.progress > 0:
                fill = int((x1 - x0 - 16) * min(1.0, self.key_dwell.progress))
                cv2.rectangle(canvas, (x0 + 8, y1 - 9), (x0 + 8 + fill, y1 - 5), _ACCENT, -1)
        self._draw_cursor(canvas, self.key_dwell)

    def _draw_results(self, canvas) -> None:
        self._dim(canvas, 0.8)
        w = self.width
        cls = self.registry[self.selected]
        entries = self.scores.get(self.selected, [])
        px0, px1, y0, y1 = w // 2 - 300, w // 2 + 300, 70, 548
        _panel(canvas, (px0, y0, px1, y1), alpha=0.96, border=_ACCENT)
        _text(canvas, f"{cls.title}  TOP {BOARD_SIZE}", w // 2, y0 + 50, 0.95, _ACCENT, 2)
        _text(canvas, "WORLDWIDE" if self.board_scope == "world" else "ON THIS DEVICE", w // 2, y0 + 78, 0.46,
              _DIM, 1, cv2.FONT_HERSHEY_SIMPLEX, shadow=False)
        if not entries:
            _text(canvas, "No scores yet - be the first!", w // 2, y0 + 200, 0.62, _DIM, 1, cv2.FONT_HERSHEY_SIMPLEX)
        for i, en in enumerate(entries[:BOARD_SIZE]):
            y = y0 + 122 + i * 32
            mine = self.last_entry_id is not None and en["id"] == self.last_entry_id
            if mine:
                _panel(canvas, (px0 + 24, y - 24, px1 - 24, y + 8), colour=(120, 100, 40), alpha=0.6, radius=8)
            rank = f"{i + 1}."
            (rw, _), _ = cv2.getTextSize(rank, cv2.FONT_HERSHEY_DUPLEX, 0.62, 1)
            cv2.putText(canvas, rank, (px0 + 110 - rw, y), cv2.FONT_HERSHEY_DUPLEX, 0.62,
                        _CYAN if mine else _DIM, 1, cv2.LINE_AA)
            cv2.putText(canvas, en["name"], (px0 + 150, y), cv2.FONT_HERSHEY_DUPLEX, 0.66,
                        _CYAN if mine else _WHITE, 2, cv2.LINE_AA)
            sc = str(en["score"])
            (sw_, _), _ = cv2.getTextSize(sc, cv2.FONT_HERSHEY_DUPLEX, 0.66, 2)
            cv2.putText(canvas, sc, (px1 - 110 - sw_, y), cv2.FONT_HERSHEY_DUPLEX, 0.66,
                        _CYAN if mine else _ACCENT, 2, cv2.LINE_AA)
        if self.last_entry_id is None:
            _text(canvas, f"YOUR SCORE   {self.score}", w // 2, y1 - 18, 0.62, _WHITE, 2, shadow=False)
        self._draw_buttons(canvas, self.dwell)
        self._draw_cursor(canvas, self.dwell)

    def _draw_countdown(self, canvas) -> None:
        self._dim(canvas, 0.45)
        w, h = self.width, self.height
        n = max(1, int(np.ceil(self.countdown)))
        _text(canvas, "GET READY", w // 2, int(h * 0.30), 1.4, _ACCENT, 3)
        _text(canvas, str(n), w // 2, int(h * 0.58), 5.0, _WHITE, 10)
        tip = getattr(self.registry[self.selected], "tip", None) or {
            "fruitninja": "Stand back so your upper body is in frame",
            "flappy": "Stand back so both arms are in frame - flap to fly",
        }.get(self.selected, "Stand back so your hips are in frame")
        _text(canvas, tip, w // 2, int(h * 0.72), 0.75, _DIM, 1, cv2.FONT_HERSHEY_SIMPLEX)

    def _draw_paused(self, canvas) -> None:
        self._dim(canvas, 0.6)
        _text(canvas, "PAUSED", self.width // 2, int(self.height * 0.40), 2.0, _ACCENT, 4)
        self._draw_buttons(canvas, self.dwell)
        self._draw_cursor(canvas, self.dwell)

    def _draw_play_chrome(self, canvas) -> None:
        """The pause button, with a cursor only while the hand is near it."""
        b = self._pause_button()
        near = b.contains(self.cursor, pad=60)
        hot = self.pause_dwell.target == b.action
        _panel(canvas, b.rect, alpha=0.7 if near else 0.45, radius=14,
               border=_ACCENT if hot else (120, 120, 130))
        x0, y0, x1, y1 = b.rect
        _text(canvas, b.label, (x0 + x1) // 2, y0 + 33, 0.62,
              _ACCENT if hot else (210, 210, 220), 1, cv2.FONT_HERSHEY_DUPLEX, shadow=False)
        if hot and self.pause_dwell.progress > 0:
            fill = int((x1 - x0 - 20) * min(1.0, self.pause_dwell.progress))
            cv2.rectangle(canvas, (x0 + 10, y1 - 9), (x0 + 10 + fill, y1 - 5), _ACCENT, -1)
        if near:
            self._draw_cursor(canvas, self.pause_dwell)

    def _draw_footer(self, canvas) -> None:
        keys = {
            Screen.MENU: "1-%d or click to start   H switch hand   ESC quit" % max(1, len(self.menu_games)),
            Screen.COUNTDOWN: "ESC menu",
            Screen.TUTORIAL: "ENTER play   ESC menu",
            Screen.ENTRY: "type your name   BACKSPACE delete   ENTER save   ESC skip",
            Screen.RESULTS: "R play again   ESC menu",
            Screen.PAUSED: "SPACE resume   R restart   ESC menu",
            Screen.PLAYING: ("R play again   ESC menu" if self.game_over
                             else "SPACE pause   R restart   ESC menu   H switch hand"),
        }[self.screen]
        hand = f"{self.hand.upper()} HAND"
        # Outlined: the footer sits over every game's art, light skies included.
        outlined_text(canvas, f"{hand}   {keys}", (18, self.height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                      (198, 198, 205), 1, width=1)
