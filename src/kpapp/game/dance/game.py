"""Raccoon Dance: copy the coach like a mirror (docs/DANCE.md).

The game is a clock, a judge and some juice:

- **Clock.** Song time is the audio player's position minus the song's
  offset, never a sum of `dt`. The player dances to what they hear, so the
  judge must use the same clock as the speakers. `dt` only animates juice.
- **Judge.** `judge.Judge` keeps the best match per judge point inside a
  late-skewed window and rates it when the window closes. The player is read
  from `controls.pose`, the extrapolated pose: for matching a held pose the
  lower latency matters more than the steps.
- **Juice.** Everything the renderer needs is plain state on the game
  (docs/DANCE.md section 8). The renderer reads it and never writes it.

The phases are intro (title card, count-in) -> dancing (moves judged) ->
outro (the last point has resolved, the music plays out) -> game_over.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from ... import theme
from ...controls import ControlState
from ..base import Game
from . import chart
from .judge import COMBO_FROM, UNSEEN, Judge, Rating, rules_for
from .pose import Measured, forward, measure

try:  # the designer's cards; the game must import without them
    from .tutorial import DANCE_TUTORIAL as _TUTORIAL
except ImportError:  # pragma: no cover - only while tutorial.py doesn't exist yet
    _TUTORIAL = []


class Phase(str, Enum):
    INTRO = "intro"
    DANCING = "dancing"
    OUTRO = "outro"
    GAME_OVER = "game_over"


@dataclass(frozen=True)
class PictogramItem:
    beat: float  # absolute beat the move is hit on
    move_label: str
    pose: dict
    gold: bool


@dataclass
class Feedback:
    """A rating popup: pops in above the coach's head, rises and fades."""
    rating: str  # "perfect" | "super" | "good" | "ok" | "x"
    gold: bool
    age: float = 0.0  # seconds since it appeared
    x: float = 520.0  # where it pops (docs/DANCE.md section 10)
    y: float = 150.0
    label: str = ""  # the move's label
    value: float = 0.0
    similarity: float = 0.0
    combo: int = 0  # the combo after this rating

    @property
    def life(self) -> float:
        """1 -> 0 over FEEDBACK_SECONDS."""
        return max(0.0, 1.0 - self.age / FEEDBACK_SECONDS)


FEEDBACK_SECONDS = 1.1
FEEDBACK_MAX = 6
UPCOMING = 8  # pictograms in the lane
UPCOMING_KEEP_BEATS = 0.5  # a card stays this long after its beat, to slide through the ring
DANCE_LEAD_BEATS = 1.0  # dancing starts this long before the first move
OUTRO_HOLD = 3.0  # s after the last move's end: game over even if the audio rings on
AUDIO_TAIL = 2.0  # s of ring-out after `duration` (docs/DANCE.md section 6)
NOTICE_AFTER = 0.6  # s of not seeing the player's arms (or feet, when the moves need them) before saying so
FEET_AHEAD = 4.0  # beats: ask for the feet this long before a judged leg move (and all through the intro)
GOLD_FLASH_SECONDS = 0.8
SCORE_EASE = 7.0  # 1/s: display_score catches up with score
AUDIO_STALL = 2.0  # s of a frozen audio clock while playing before falling back to a silent one

NOTICE_ARMS = "STEP BACK - SHOW YOUR ARMS"
NOTICE_NOBODY = "STEP INTO THE FRAME"
NOTICE_FEET = "STEP BACK - SHOW YOUR FEET"


def board_id_for(song_id: str, difficulty: str | None = None) -> str:
    """A song's leaderboard on a difficulty: "dance-<song>" on normal (the
    board it had before difficulties), "dance-<song>_<difficulty>" otherwise.
    The server accepts /^dance-[a-z0-9_]{1,40}$/."""
    level = chart.difficulty_id(difficulty)
    return f"dance-{song_id}" if level == chart.DEFAULT_DIFFICULTY else f"dance-{song_id}_{level}"


def _default_library() -> dict:
    """The shared move library, or none if it isn't there (songs may define
    all their moves in custom_moves; unknown moves still fail loudly)."""
    try:
        return chart.load_library(chart.LIBRARY)
    except chart.ChartError as exc:
        if "missing move library" in str(exc):
            return {}
        raise


class DanceGame(Game):
    name = "dance"
    title = "RACCOON DANCE"
    own_countdown = True  # the song counts in 3-2-1 on the beat; the shell skips its own
    blurb = "Bust a move with the raccoon. Copy every pose like a mirror!"
    tip = "Stand back so your whole body, feet too, is in frame - copy the coach like a mirror"
    board_id = "dance"  # per song: "dance-<song id>" (variants()), "_<difficulty>" off normal
    tutorial = _TUTORIAL
    # The shell's difficulty picker reads these (docs/DANCE.md section 7a). NORMAL
    # keeps each song's original board, so scores from before difficulties stay put.
    difficulties = [
        {"id": "easy", "label": "EASY", "hint": "fewer moves, held longer"},
        {"id": "normal", "label": "NORMAL", "hint": "the full routine"},
        {"id": "hard", "label": "HARD", "hint": "every move, judged strictly"},
    ]
    default_difficulty = chart.DEFAULT_DIFFICULTY

    def __init__(self, size: tuple[int, int], mirrored: bool = True, seed: int | None = None,
                 variant: str | None = None, art=None, player=None, songs_dir=None,
                 library: dict | None = None, difficulty: str | None = "normal",
                 **options: object) -> None:
        super().__init__(size, **options)
        self.mirrored = mirrored
        self.seed = seed  # nothing random yet; kept for the game contract
        self.songs_dir = Path(songs_dir) if songs_dir is not None else chart.SONGS
        lib = library if library is not None else _default_library()
        if variant is None:
            ids = chart.song_ids(self.songs_dir)
            if not ids:
                raise chart.ChartError(f"no songs in {self.songs_dir / 'index.json'}")
            variant = ids[0]
        self.variant = variant
        self.difficulty = chart.difficulty_id(difficulty)  # unknown ids are normal
        self.song = chart.load_song(self.songs_dir / variant, lib, difficulty=self.difficulty)
        self.rules = rules_for(self.difficulty)
        self.board_id = board_id_for(variant, self.difficulty)
        if player is None:
            from ...audio import open_player

            player = open_player(self.song.audio, self.song.duration + AUDIO_TAIL)
        self.audio = player
        self.art = art
        self._renderer = None
        self._renderer_failed = False
        self._pictos = [PictogramItem(p.beat, p.label, p.pose, p.gold) for p in self.song.pictograms()]
        self._picto_beats = [p.beat for p in self._pictos]
        self._last_move_end = self.song.time_of(max((p.end for p in self.song.chart), default=0.0))
        # Judged poses that work the legs: the feet must be in the picture for them.
        self._leg_beats = [p.beat for p in self.song.judge_points() if chart.leg_led(p.pose)]
        self.judge = Judge(self.song, self.rules)
        self._closed = False
        self.reset()

    # --- lifecycle ---

    def reset(self) -> None:
        if getattr(self, "_started", False):
            self.audio.stop()
        self.judge.reset()
        self.phase = Phase.INTRO
        self._started = False
        self._paused = False
        self.song_time = -self.song.offset
        self.beat = self.song.beat_of(self.song_time)
        self.coach_pose = self.song.coach_pose(self.beat)
        self.upcoming: list[PictogramItem] = self._pictos[:UPCOMING]
        self.feedback: list[Feedback] = []
        self.display_score = 0.0
        self.player: Measured | None = None
        self.notice: str | None = None
        self.gold_flash = 0.0
        self.score_bump = 0.0  # 1 -> 0 after the score changes (HUD pulse)
        self.results_t = 0.0
        self.ratings: list[Rating] = []  # every resolved point, unseen included
        self._unseen_for = 0.0
        self._feetless_for = 0.0
        self._clock = 0.0  # sum of clamped dt: juice only
        self._last_pos: float | None = None
        self._pos_still = 0.0

    def on_pause(self) -> None:
        if self._started and not self._paused:
            self.audio.pause()
        self._paused = True

    def on_resume(self) -> None:
        """Resume the music and forget what the judge saw before the pause:
        the player moved (to the PAUSE and RESUME buttons) while it stood still."""
        if self._paused and self._started and not self._closed:
            self.audio.resume()
        self._paused = False
        self.song_time = self._audio_time()
        self.judge.on_resume(self.song_time)
        self._unseen_for = 0.0
        self._feetless_for = 0.0
        self.notice = None
        self._pos_still = 0.0

    def close(self) -> None:
        """Stop the music; the shell calls this whenever it drops the game."""
        self._closed = True
        self.audio.stop()

    # --- state the renderer and shell read ---

    @property
    def score(self) -> int:
        return self.judge.score

    @property
    def stars(self) -> float:
        """0..5, continuous: the star bar's fill."""
        return self.judge.star_fill

    @property
    def star_count(self) -> int:
        return self.judge.stars

    @property
    def superstar(self) -> bool:
        return self.judge.superstar

    @property
    def counts(self) -> dict[str, int]:
        return self.judge.counts

    @property
    def combo(self) -> int:
        return self.judge.combo

    @property
    def last_rating(self) -> str | None:
        return self.judge.last_rating

    @classmethod
    def variants(cls, songs_dir=None, library: dict | None = None) -> list[dict]:
        """The songs, in menu order: [{id, title, subtitle, board_id}]."""
        d = Path(songs_dir) if songs_dir is not None else chart.SONGS
        lib = library if library is not None else _default_library()
        out = []
        for sid in chart.song_ids(d):
            try:
                s = chart.load_song(d / sid, lib)
            except chart.ChartError as exc:
                print(f"[kp] dance: skipping song '{sid}': {exc}")
                continue
            # ASCII only: OpenCV's Hershey fonts can't draw a middle dot.
            # The song's length, not its old "difficulty 1" rating: that read like the
            # EASY / NORMAL / HARD picker above the cards.
            length = f"{int(s.duration // 60)}:{int(round(s.duration % 60)):02d}"
            sub = f"{s.artist} - {length}" if s.artist else length
            out.append({"id": sid, "title": s.title, "subtitle": sub, "board_id": f"dance-{sid}"})
        return out

    # --- the clock ---

    def _audio_time(self) -> float:
        return self.audio.position() - self.song.offset

    def _watch_audio(self, dt: float) -> None:
        """A real device that stops reporting progress must not freeze the
        game forever: after AUDIO_STALL seconds, keep time silently."""
        pos = self.audio.position()
        if self._paused or not self._started or self._closed or getattr(self.audio, "finished", False):
            self._last_pos, self._pos_still = pos, 0.0
            return
        if self._last_pos is not None and pos == self._last_pos:
            self._pos_still += dt
        else:
            self._pos_still = 0.0
        self._last_pos = pos
        if self._pos_still > AUDIO_STALL:
            from ...audio import SilentPlayer

            print(f"[kp] dance: the audio clock stopped at {pos:.2f}s; keeping time silently")
            self.audio.stop()
            self.audio = SilentPlayer(self.song.duration + AUDIO_TAIL)
            self.audio.play(pos)
            self._pos_still = 0.0

    # --- update ---

    def update(self, controls: ControlState, dt: float) -> None:
        dt = min(max(float(dt), 0.0), 0.1)
        self._clock += dt
        if not self._started and not self._closed:
            self.audio.play(0.0)
            self._started = True
        if not self._paused:
            self._watch_audio(dt)
        self.song_time = self._audio_time()
        self.beat = self.song.beat_of(self.song_time)
        self.coach_pose = self.song.coach_pose(self.beat)
        i = bisect.bisect_left(self._picto_beats, self.beat - UPCOMING_KEEP_BEATS)
        self.upcoming = self._pictos[i:i + UPCOMING]

        pose = getattr(controls, "pose", None) if controls is not None else None
        self.player = measure(pose, self.mirrored)
        seen = self.player is not None and self.player.sees_arms

        if self.phase is not Phase.GAME_OVER:
            for r in self.judge.update(self.song_time, self.player):
                self._on_rating(r)
            self._advance_phase()

        if self.phase is Phase.DANCING and not seen:
            self._unseen_for += dt
        else:
            self._unseen_for = 0.0
        # Feet: never a miss (the legs of a player whose feet are out of view are
        # not judged, the arms carry the point), but say what to fix.
        if seen and not self.player.sees_feet and self._legs_coming():
            self._feetless_for += dt
        else:
            self._feetless_for = 0.0
        if self._unseen_for > NOTICE_AFTER:
            self.notice = NOTICE_NOBODY if self.player is None else NOTICE_ARMS
        elif self._feetless_for > NOTICE_AFTER:
            self.notice = NOTICE_FEET
        else:
            self.notice = None

        self._animate(dt)

    def _legs_coming(self) -> bool:
        """Is a judged leg move open or due within FEET_AHEAD beats (the intro counts)?"""
        if not self._leg_beats or self.phase in (Phase.OUTRO, Phase.GAME_OVER):
            return False
        if self.phase is Phase.INTRO:
            return True
        i = bisect.bisect_left(self._leg_beats, self.beat - 1.0)
        return i < len(self._leg_beats) and self._leg_beats[i] <= self.beat + FEET_AHEAD

    def _advance_phase(self) -> None:
        if self.phase is Phase.INTRO and self.judge.points and \
                self.beat >= self.song.first_move_beat - DANCE_LEAD_BEATS:
            self.phase = Phase.DANCING
        if self.phase in (Phase.INTRO, Phase.DANCING) and self.judge.done and \
                (self.phase is Phase.DANCING or not self.judge.points):
            self.phase = Phase.OUTRO
        if self.phase is Phase.OUTRO and (getattr(self.audio, "finished", False)
                                          or self.song_time >= self._last_move_end + OUTRO_HOLD):
            self.phase = Phase.GAME_OVER
            self.results_t = 0.0

    def _on_rating(self, r: Rating) -> None:
        self.ratings.append(r)
        if r.name == UNSEEN:
            return
        self.feedback.append(Feedback(r.name, r.gold, label=r.point.label, value=r.value,
                                      similarity=r.similarity or 0.0, combo=self.judge.combo))
        del self.feedback[:-FEEDBACK_MAX]
        if r.value > 0:
            self.score_bump = 1.0
        if r.gold and r.name in COMBO_FROM:
            self.gold_flash = 1.0

    def _animate(self, dt: float) -> None:
        for f in self.feedback:
            f.age += dt
        self.feedback = [f for f in self.feedback if f.age < FEEDBACK_SECONDS]
        self.gold_flash = max(0.0, self.gold_flash - dt / GOLD_FLASH_SECONDS)
        self.score_bump = max(0.0, self.score_bump - dt / 0.35)
        target = float(self.score)
        self.display_score += (target - self.display_score) * (1.0 - math.exp(-SCORE_EASE * dt))
        if abs(target - self.display_score) < 1.0:
            self.display_score = target
        if self.phase is Phase.GAME_OVER:
            self.results_t += dt

    # --- render ---

    def render(self, frame: np.ndarray) -> None:
        renderer = self._get_renderer()
        if renderer is not None:
            renderer.render(frame, self)
        else:
            self._render_debug(frame)

    def preload(self) -> None:
        """Load the art, build the renderer and fill its caches now, not on the first
        frame of the countdown, where it showed as a hitch. The shell calls
        this when it builds the game, on the song pick's screen change."""
        renderer = self._get_renderer()
        if renderer is not None:  # one throwaway frame fills its sprite caches (title, cards, words)
            renderer.render(np.zeros((self.height, self.width, 3), np.uint8), self)

    def _get_renderer(self):
        if self._renderer is None and not self._renderer_failed:
            try:
                from .render import DanceRenderer
                art = self.art
                if art is None:
                    from . import art as art_mod

                    missing = getattr(art_mod, "AssetsMissingError", FileNotFoundError)
                    try:
                        art = art_mod.load_art()
                    except (missing, FileNotFoundError) as exc:
                        print(f"[kp] dance: {exc}; showing the debug view")
                        self._renderer_failed = True
                        return None
                self._renderer = DanceRenderer((self.width, self.height), art)
            except ImportError as exc:
                print(f"[kp] dance: renderer not available ({exc}); showing the debug view")
                self._renderer_failed = True
        return self._renderer

    def _render_debug(self, frame: np.ndarray) -> None:
        """A bare view so the game runs before the real renderer exists: the
        coach as a stick figure, the score, the last rating."""
        cv2.addWeighted(frame, 0.35, np.zeros_like(frame), 0.65, 0, dst=frame)
        sw = 110.0 * self.height / 720.0
        ox, oy = self.width * 520 / 1280, self.height * 660 / 720 - 1.67 * sw
        j = forward(self.coach_pose)

        def px(name):
            x, y = j[name]
            return int(ox + x * sw), int(oy + y * sw)

        bones = [("neck", "l_shoulder"), ("neck", "r_shoulder"), ("neck", "hip"),
                 ("l_hip", "r_hip")]
        for s in "lr":
            bones += [(f"{s}_shoulder", f"{s}_elbow"), (f"{s}_elbow", f"{s}_wrist"),
                      (f"{s}_wrist", f"{s}_hand"), (f"{s}_hip", f"{s}_knee"), (f"{s}_knee", f"{s}_ankle")]
        for a, b in bones:
            cv2.line(frame, px(a), px(b), theme.CYAN, 8, cv2.LINE_AA)
        cv2.circle(frame, px("head"), int(0.3 * sw), theme.CYAN, 6, cv2.LINE_AA)
        theme.centred_text(frame, f"{self.score:05d}", 120, 60, 1.2, theme.ACCENT, 2)
        theme.centred_text(frame, self.song.title.upper(), self.width // 2, 50, 0.9, theme.ACCENT, 2)
        if self.feedback:
            f = self.feedback[-1]
            colour = theme.DANGER if f.rating == "x" else (theme.ACCENT if f.gold else theme.CYAN)
            theme.centred_text(frame, f.rating.upper(), int(ox), 150, 1.6, colour, 3)
        if self.upcoming:
            theme.centred_text(frame, f"NEXT: {self.upcoming[0].move_label.upper()}", 1010, 600, 0.8,
                               theme.DIM, 2)
        if self.notice:
            theme.centred_text(frame, self.notice, self.width // 2, self.height // 2, 1.0, theme.WARN, 2)
        if self.phase is Phase.GAME_OVER:
            theme.centred_text(frame, f"{self.star_count} STARS", self.width // 2, 260, 1.4, theme.ACCENT, 3)
