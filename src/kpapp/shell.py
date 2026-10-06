"""The arcade around the games: welcome screen, start, pause, stop, replay.

Games used to start the instant the window opened. On recorded play that meant
fruit was already falling while the player was still stepping back, and three
rounds ended in about three seconds each. Now nothing starts until the player
chooses, and a countdown gives them time to get into position.

Everything can be done standing back from the computer: hovering the hand over
a button for a moment presses it, Kinect-style. Keys do the same for anyone at
the keyboard.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from .controls import ControlState
from .hand import HandTracker
from .theme import ACCENT as _ACCENT
from .theme import CYAN as _CYAN
from .theme import DIM as _DIM
from .theme import PANEL as _PANEL
from .theme import WHITE as _WHITE
from .theme import centred_text as _text
from .theme import panel as _panel
from .theme import wrap as _wrap

COUNTDOWN_SECONDS = 3.0
MENU_DWELL = 1.0  # hover time to press a button
PAUSE_DWELL = 1.5  # longer in play, so a passing hand cannot pause by accident

KEY_ESC, KEY_ENTER, KEY_SPACE = 27, 13, 32


class Screen(str, Enum):
    MENU = "menu"
    COUNTDOWN = "countdown"
    PLAYING = "playing"
    PAUSED = "paused"


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
    ) -> None:
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
        self._was_over = False
        self.dwell.reset()
        self.pause_dwell.reset()

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
        self.dwell.block()

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

    # --- input ---

    def handle_key(self, key: int) -> bool:
        """Apply a keypress. Returns True if the app should quit."""
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
                if key in (ord("r"), KEY_SPACE, KEY_ENTER):
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
        self._buttons = self._layout()

        if self.screen is Screen.COUNTDOWN:
            self.countdown -= dt
            if self.countdown <= 0.0:
                self.screen = Screen.PLAYING
            return

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
            self.game.update(controls, dt)  # let debris finish falling
        self._was_over = self.screen is Screen.PLAYING and self.game_over

        hovered = next((b.action for b in self._buttons if b.contains(self.cursor, pad=6)), None)
        fired = self.dwell.update(hovered, dt)
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
        bw, bh, gap = int(w * 0.20), 74, int(w * 0.04)
        y0 = int(h * 0.56) if self.screen is Screen.PAUSED else int(h * 0.80)
        if self.screen is Screen.PAUSED:
            pair = (("resume", "RESUME", "SPACE"), ("menu", "MENU", "ESC"))
        elif self.screen is Screen.PLAYING and self.game_over:
            pair = (("again", "PLAY AGAIN", "R"), ("menu", "MENU", "ESC"))
        else:
            return []
        x = w // 2 - bw - gap // 2
        out = []
        for action, label, key in pair:
            out.append(Button(action, label, (x, y0, x + bw, y0 + bh), key_hint=key))
            x += bw + gap
        return out

    # --- drawing ---

    def render(self, canvas: np.ndarray) -> None:
        self._buttons = self._layout()
        if self.screen is Screen.MENU:
            self._draw_menu(canvas)
        else:
            self.game.render(canvas)
            if self.screen is Screen.COUNTDOWN:
                self._draw_countdown(canvas)
            elif self.screen is Screen.PAUSED:
                self._draw_paused(canvas)
            elif self.game_over:
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

    def _draw_countdown(self, canvas) -> None:
        self._dim(canvas, 0.45)
        w, h = self.width, self.height
        n = max(1, int(np.ceil(self.countdown)))
        _text(canvas, "GET READY", w // 2, int(h * 0.30), 1.4, _ACCENT, 3)
        _text(canvas, str(n), w // 2, int(h * 0.58), 5.0, _WHITE, 10)
        tip = {
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
            Screen.MENU: "1-%d start   ENTER start   H switch hand   ESC quit" % max(1, len(self.menu_games)),
            Screen.COUNTDOWN: "ESC menu",
            Screen.PAUSED: "SPACE resume   R restart   ESC menu",
            Screen.PLAYING: ("R play again   ESC menu" if self.game_over
                             else "SPACE pause   R restart   ESC menu   H switch hand"),
        }[self.screen]
        hand = f"{self.hand.upper()} HAND"
        cv2.putText(canvas, f"{hand}   {keys}", (18, self.height - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (175, 175, 185), 1, cv2.LINE_AA)
