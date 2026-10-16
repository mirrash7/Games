"""Raccoon Golf: three island holes, swung with your arms.

The rules and flow follow resort-island motion golf (Wii Sports Resort's
Golf): a behind-the-ball view of the hole; auto-aim at the pin (or down the
fairway on a dogleg) and an auto-chosen club, both adjustable; a landing
target and dotted arc for a full-power straight shot; a power gauge that
fills on the backswing with a red overswing zone that hooks or slices the
ball; practice swings until you address the ball; a putter, slope grid and
distance in feet on the green; water hazards (+1, drop near where it went in)
and out of bounds (+1, replay from the same spot); and a pick-up at par + 5 so
no hole drags on. The art, course and characters are original.

The input is a whole-body swing seen by a camera (swing.py). Everything about
it is tuned for ~100 ms of lag and 15-30 Hz poses: power comes from how high
the hands went (a held position, not a speed), and impact is found on the
grip's swept path, so lag delays a shot but never changes it.

Scoring for the leaderboard (higher is better) is per hole, by strokes
against par, while the HUD shows the golf score (E, +1, -2) as Wii does:

    hole in one 1000, albatross or better 800, eagle 500, birdie 300,
    par 200, bogey 100, double bogey 50, anything worse (or a pick-up) 10

A perfect round is 3000. Points arrive only when a hole is finished (the
fastest possible: an ace on the par 3, see FASTEST_HOLE_SECONDS).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ...controls import ControlState
from ...hand import HandTracker
from ...shell import KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_UP  # arrows, as the shell passes them
from ..base import Game
from . import physics as P
from .course import FEET_PER_M, MPH, YARDS_PER_M, Course, Hole, load_course
from .swing import SwingDetector, SwingEvent
from .tutorial import GOLF_TUTORIAL


class Phase(str, Enum):
    INTRO = "intro"  # a flyover of the hole, with its number, par and length
    AIM = "aim"  # pre-shot: address, practice swings, adjust aim/club, swing
    FLIGHT = "flight"  # the ball in the air and rolling; the camera follows
    RESULT = "result"  # where it ended up; then the next shot
    HOLED = "holed"  # the hole's result, then the scorecard
    GAME_OVER = "game_over"  # the shell watches for exactly this value


# --- pacing (seconds) ---
INTRO_SECONDS = 3.0
RESULT_SECONDS = 1.6
HAZARD_SECONDS = 2.4
HOLED_MESSAGE = 2.2
SCORECARD_SECONDS = 3.6
AIR_SPEED = 1.25  # simulated seconds per real second while the ball flies
ROLL_SPEED = 1.7  # ...and while it rolls (a long roll is dull at 1x)
PUTT_SPEED = 1.3
FOLLOW_SECONDS = 0.3  # Lenny's follow-through
CHASE_BLEND = 0.9  # s from the address view to the chase camera after a shot
HOLED_PULL = 0.9  # s from the last shot's view to the hole-out view
# The hole-out view (holed_camera): back and up from the pin, turned so the flag
# stands right of the result text with Lenny celebrating left of it, and pitched
# down far enough that the horizon (and the trees beyond the green) sit behind
# the top HUD row's panels only as sky, not as trunks and canopies.
HOLED_BACK, HOLED_UP, HOLED_PITCH = 10.0, 6.5, math.radians(21.0)
HOLED_TURN = math.radians(-14.0)  # look left of the pin: it lands right of centre
LENNY_BY_PIN = (-6.8, 0.5)  # Lenny's spot from the pin, m: (right of the line, along it)
MAX_OVER_PAR = 5  # pick up at par + 5

# --- scoring ---
POINTS_ACE = 1000
POINTS_BY_DIFF = {-3: 800, -2: 500, -1: 300, 0: 200, 1: 100, 2: 50}
POINTS_WORSE = 10
# The fastest a hole can be finished after play starts: the intro, an address
# hold, a swing, and an ace's flight at AIR_SPEED (a 150 m carry flies ~6 s).
# For the leaderboard's plausibility rules: at most 1000 points per hole and a
# hole can't finish sooner than this.
FASTEST_HOLE_SECONDS = INTRO_SECONDS + 0.4 + 0.5 + 6.0 / AIR_SPEED

AIM_STEP = math.radians(2.0)  # per press / repeat of the aim buttons
AIM_STEP_PUTT = math.radians(1.0)  # finer on the green, where a putt's line is read off the slope grid
AIM_MAX = math.radians(40.0)
HOVER_FIRE = 0.6  # s on an in-game button before it fires...
HOVER_REPEAT = 0.22  # ...then again every this while held



def result_name(strokes: int, par: int) -> str:
    if strokes == 1:
        return "HOLE IN ONE!"
    d = strokes - par
    return {-4: "CONDOR!", -3: "ALBATROSS!", -2: "EAGLE!", -1: "BIRDIE!", 0: "PAR", 1: "BOGEY",
            2: "DOUBLE BOGEY", 3: "TRIPLE BOGEY"}.get(d, f"+{d}" if d > 0 else f"{d}")


def hole_points(strokes: int, par: int, picked_up: bool = False) -> int:
    if picked_up:
        return POINTS_WORSE
    if strokes == 1:
        return POINTS_ACE
    d = strokes - par
    if d <= -3:
        return POINTS_BY_DIFF[-3]
    return POINTS_BY_DIFF.get(d, POINTS_WORSE)


def to_par_text(diff: int) -> str:
    return "E" if diff == 0 else (f"+{diff}" if diff > 0 else str(diff))


@dataclass
class Camera:
    x: float = 0.0
    y: float = 0.0
    z: float = 4.0
    yaw: float = 0.0  # heading the camera looks along (0 = +y)
    pitch: float = 0.18  # radians, positive looks down

    def copy(self) -> "Camera":
        return Camera(self.x, self.y, self.z, self.yaw, self.pitch)


def holed_camera(pin: tuple[float, float], heading: float) -> tuple[Camera, tuple[float, float]]:
    """The hole-out view along `heading` (the last shot's), and where Lenny celebrates."""
    px, py = pin
    sh, ch = math.sin(heading), math.cos(heading)
    cam = Camera(px - sh * HOLED_BACK, py - ch * HOLED_BACK, HOLED_UP, heading + HOLED_TURN, HOLED_PITCH)
    right, along = LENNY_BY_PIN
    spot = (px + ch * right + sh * along, py - sh * right + ch * along)
    return cam, spot


def address_camera(x: float, y: float, heading: float, putt: bool) -> Camera:
    back, height, pitch = (5.0, 2.3, math.radians(16.0)) if putt else (9.0, 4.2, math.radians(10.0))
    return Camera(x - math.sin(heading) * back, y - math.cos(heading) * back, height, heading, pitch)


@dataclass
class Message:
    text: str
    colour: tuple
    size: float = 2.0
    life: float = 0.0  # seconds shown so far
    duration: float = 1.8
    sub: str = ""


@dataclass
class Popup:
    """A "+300" that pops in, rises and fades at the point of scoring."""

    text: str
    colour: tuple
    pos: tuple[float, float]
    life: float = 0.0
    duration: float = 1.2


@dataclass
class Particle:
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    kind: str  # "water" or "sand"
    life: float = 0.0
    duration: float = 0.9


@dataclass
class CardRow:
    par: int
    strokes: int = 0
    points: int = 0
    picked_up: bool = False
    done: bool = False


@dataclass
class Effects:
    shake: float = 0.0
    flash: float = 0.0
    score_bump: float = 0.0
    particles: list[Particle] = field(default_factory=list)
    popups: list[Popup] = field(default_factory=list)


class GolfGame(Game):
    name = "golf"
    title = "RACCOON GOLF"
    blurb = "Swing your arms like a golf club. Three island holes - can you beat par?"
    tip = "Stand back so your hands show at waist height"
    tutorial = GOLF_TUTORIAL
    board_id = "golf"

    def __init__(self, size, mirrored: bool = True, seed: int | None = None, art=None,
                 course: Course | None = None, holes: int | None = None, detector=None, **options) -> None:
        super().__init__(size, **options)
        self.mirrored = mirrored
        self.seed = seed
        self._art = art  # None: the real art, loaded at the first render
        self.course = course or load_course()
        self.n_holes = min(len(self.course.holes), holes or len(self.course.holes))
        self.detector = detector or SwingDetector(frame_width=self.width, frame_height=self.height)
        self.hand = "right"
        self.tracker = HandTracker(self.hand, mirrored)
        self.reset()

    # --- lifecycle ---

    def reset(self) -> None:
        self.rng = random.Random(self.seed)  # unseeded by default: wind differs every round
        self.clock = 0.0
        self.card = [CardRow(h.par) for h in self.course.holes[: self.n_holes]]
        self.fx = Effects()
        self.message: Message | None = None
        self.flight: P.Flight | None = None
        self.preview: P.Flight | None = None
        self.last_shot: dict | None = None
        self.cursor = None
        self.hover: str | None = None
        self.hover_t = 0.0
        self.hover_fired = 0
        self.lenny_angle = 0.0
        self.follow_t = -1.0  # time since the last swing's impact (-1: none)
        self.follow_from = 0.0
        self.practice_t = -1.0
        self.last_pose = None
        self.shots_taken = 0
        self.celebrate_spot: tuple[float, float] | None = None  # where Lenny cheers after holing out
        self._start_hole(0)

    def on_resume(self) -> None:
        self.detector.reset()  # the player moved during the pause
        self.tracker.reset()

    def swap_hand(self) -> None:
        self.hand = "left" if self.hand == "right" else "right"
        self.tracker = HandTracker(self.hand, self.mirrored)

    def preload(self) -> None:
        self._renderer_or_none()

    # --- properties the HUD and the shell read ---

    @property
    def hole(self) -> Hole:
        return self.course.holes[self.hole_i]

    @property
    def score(self) -> int:
        """Leaderboard points (higher is better)."""
        return sum(r.points for r in self.card)

    @property
    def strokes_total(self) -> int:
        return sum(r.strokes for r in self.card if r.done) + (0 if self.card[self.hole_i].done else self.strokes)

    @property
    def to_par(self) -> int:
        """Strokes against par so far, counting only finished holes (as on a scorecard)."""
        return sum(r.strokes - r.par for r in self.card if r.done)

    @property
    def club(self) -> P.Club:
        return P.CLUBS[self.club_i]

    @property
    def putting(self) -> bool:
        return self.club.putter

    @property
    def heading(self) -> float:
        return self.auto_heading + self.aim_offset

    @property
    def distance_to_pin(self) -> float:
        return self.hole.distance_to_pin(*self.ball)

    def distance_text(self) -> str:
        d = self.distance_to_pin
        if self.lie == "green":
            return f"{d * FEET_PER_M:.0f} ft"
        return f"{d * YARDS_PER_M:.0f} yd"

    def club_range_text(self) -> str:
        if self.putting:
            return f"{P.putt_range_m() * FEET_PER_M:.0f} ft"
        reach = self.club.carry_m * P.LIE_FACTOR.get(self.lie, 1.0)
        return f"{reach * YARDS_PER_M:.0f} yd"

    @property
    def wind_mph(self) -> float:
        return self._wind_mph

    @property
    def wind_dir(self) -> float:
        """Heading the wind blows toward (0 = down the hole, +y)."""
        return self._wind_dir

    @property
    def wind(self) -> tuple[float, float]:
        s = self._wind_mph * MPH
        return (s * math.sin(self._wind_dir), s * math.cos(self._wind_dir))

    # --- holes and shots ---

    def _start_hole(self, i: int) -> None:
        self.hole_i = i
        h = self.hole
        self.ball = h.tee
        self.prev_spot = h.tee
        self.lie = "tee"
        self.strokes = 0
        lo, hi = h.wind_range
        self._wind_mph = float(round(self.rng.uniform(lo, hi)))
        self._wind_dir = self.rng.uniform(0.0, 2 * math.pi)
        self.phase = Phase.INTRO
        self.phase_t = 0.0
        self.flight = None
        self._setup_shot()
        # The intro flies down from high above the tee to the address view.
        hd = self.auto_heading
        self.intro_from = Camera(h.tee[0] - math.sin(hd) * 30, h.tee[1] - math.cos(hd) * 30, 62.0, hd,
                                 math.radians(15.0))
        self.camera = self.intro_from.copy()

    def auto_club(self) -> int:
        """Wii-style: the shortest club that reaches the pin (putter on the green)."""
        if self.lie == "green":
            return len(P.CLUBS) - 1
        d = self.distance_to_pin
        factor = P.LIE_FACTOR.get(self.lie, 1.0)
        usable = [i for i, c in enumerate(P.CLUBS[:-1]) if not c.tee_only or self.lie == "tee"]
        for i in reversed(usable):  # shortest first
            if P.CLUBS[i].carry_m * factor >= d * 0.97:
                return i
        return usable[0]

    def _setup_shot(self) -> None:
        """Ready the next shot from self.ball: club, aim, swing mode, camera."""
        self.club_i = self.auto_club()
        self.aim_offset = 0.0
        self._aim()
        self.detector.set_mode("putt" if self.putting else "full")
        self.detector.reset()
        self.lenny_angle = 0.0
        self.follow_t = -1.0
        self.camera = address_camera(*self.ball, self.heading, self.putting)

    def _aim(self) -> None:
        """Auto-aim for the current club, and the preview of a full straight shot."""
        reach = self.club.carry_m * P.LIE_FACTOR.get(self.lie, 1.0) if not self.putting else math.inf
        tx, ty = self.hole.aim_point(*self.ball, reach) if not self.putting else self.hole.pin
        self.auto_heading = math.atan2(tx - self.ball[0], ty - self.ball[1])
        self.preview = P.preview(self.hole, self.ball, self.club, self.heading, self.lie)

    def change_club(self, step: int) -> None:
        if self.phase is not Phase.AIM or self.detector.swinging:
            return
        n = len(P.CLUBS)
        i = self.club_i
        for _ in range(n):
            i = (i + step) % n
            c = P.CLUBS[i]
            if c.tee_only and self.lie != "tee":
                continue
            if self.lie == "bunker" and c.putter:
                continue
            break
        was_putt = self.putting
        self.club_i = i
        self._aim()
        if self.putting != was_putt:
            self.detector.set_mode("putt" if self.putting else "full")
            self.camera = address_camera(*self.ball, self.heading, self.putting)

    def adjust_aim(self, step: int) -> None:
        if self.phase is not Phase.AIM or self.detector.swinging:
            return
        step_size = AIM_STEP_PUTT if self.putting else AIM_STEP
        self.aim_offset = max(-AIM_MAX, min(AIM_MAX, self.aim_offset + step * step_size))
        self.preview = P.preview(self.hole, self.ball, self.club, self.heading, self.lie)
        self.camera = address_camera(*self.ball, self.heading, self.putting)

    def handle_key(self, key: int) -> bool:
        """Arrow keys: aim and club. Returns True if the key was used."""
        if key == KEY_LEFT:
            self.adjust_aim(-1)
        elif key == KEY_RIGHT:
            self.adjust_aim(1)
        elif key == KEY_UP:
            self.change_club(-1)  # up the list: a longer club
        elif key == KEY_DOWN:
            self.change_club(1)
        else:
            return False
        return True

    def _hit(self, ev: SwingEvent) -> None:
        curve = 0.0
        if not self.putting and ev.overswing > 0:
            # Overswing: an unpredictable hook or slice, bigger the deeper into the red.
            curve = self.rng.choice((-1.0, 1.0)) * ev.overswing * self.rng.uniform(0.55, 1.0)
        shot = P.Shot(self.club, ev.power, self.heading, self.lie, quality=ev.quality, curve=curve, wind=self.wind)
        self.prev_spot = self.ball
        self.start_lie = self.lie
        self.strokes += 1
        self.shots_taken += 1
        self.flight = P.Flight(self.hole, self.ball, shot)
        self.last_shot = {"power": ev.power, "quality": ev.quality, "curve": curve, "club": self.club.key,
                          "overswing": ev.overswing}
        self.phase = Phase.FLIGHT
        self.phase_t = 0.0
        self.shot_from_cam = self.camera.copy()
        self.message = None
        if ev.overswing > 0 and not self.putting:
            self._say("OVERSWING!", (60, 80, 255), 1.3, 1.1)
        elif ev.quality < 0.5 and not self.putting:
            self._say("THIN!", (90, 190, 255), 1.3, 1.1)

    def _say(self, text: str, colour, size: float = 2.0, duration: float = 1.8, sub: str = "") -> None:
        self.message = Message(text, colour, size, 0.0, duration, sub)

    # --- the frame ---

    def update(self, controls: ControlState, dt: float) -> None:
        dt = min(max(dt, 0.0), 0.1)  # a stalled frame must not skip the game forward
        self.clock += dt
        self.phase_t += dt
        self.last_pose = controls.pose
        self._effects(dt)
        if self.phase is Phase.INTRO:
            u = min(1.0, self.phase_t / INTRO_SECONDS)
            e = u * u * (3 - 2 * u)
            self.camera = _lerp_cam(self.intro_from, address_camera(*self.ball, self.heading, self.putting), e)
            if u >= 1.0:
                self.phase, self.phase_t = Phase.AIM, 0.0
            return
        if self.phase is Phase.AIM:
            self._update_aim(controls, dt)
        elif self.phase is Phase.FLIGHT:
            self._update_flight(dt)
        elif self.phase is Phase.RESULT:
            if self.phase_t >= self._result_for:
                self._after_result()
        elif self.phase is Phase.HOLED:
            u = min(1.0, self.phase_t / HOLED_PULL)  # pull back to look at the green
            self.camera = _lerp_cam(self.holed_from, self.holed_to, u * u * (3 - 2 * u))
            if self.phase_t >= HOLED_MESSAGE + SCORECARD_SECONDS:
                if self.hole_i + 1 < self.n_holes:
                    self._start_hole(self.hole_i + 1)
                else:
                    self.phase, self.phase_t = Phase.GAME_OVER, 0.0
        self._animate_lenny(dt)

    def _update_aim(self, controls: ControlState, dt: float) -> None:
        # Stepped poses: the detector times motion between model updates.
        step = controls.step_pose if controls.step_pose is not None else controls.pose
        ev = self.detector.update(step, self.clock)
        self._update_buttons(controls, dt)
        if ev is None:
            return
        self.follow_t, self.follow_from = 0.0, self.lenny_angle
        if ev.practice:
            self.practice_t = 0.0
            return
        self._hit(ev)

    # --- the in-game hover buttons (aim and club) ---

    def buttons(self) -> list[tuple[str, tuple[int, int, int, int]]]:
        """The adjust buttons' rects. Only while the hands are apart: a grip
        (hands together) is swinging, and a backswing must never press one."""
        if self.phase is not Phase.AIM:
            return []
        return [("club_up", (16, 222, 158, 288)), ("club_down", (174, 222, 316, 288)),
                ("aim_left", (200, 400, 330, 480)), ("aim_right", (950, 400, 1080, 480))]

    @property
    def buttons_active(self) -> bool:
        d = self.detector
        return (self.phase is Phase.AIM and self.cursor is not None and not d.swinging and not d.together
                and d.stage.value == "no_grip")

    def _update_buttons(self, controls: ControlState, dt: float) -> None:
        self.cursor = self.tracker.update(controls.pose, self.clock)
        hovered = None
        if self.buttons_active and self.cursor is not None:
            cx, cy = float(self.cursor[0]), float(self.cursor[1])
            for name, (x0, y0, x1, y1) in self.buttons():
                if x0 - 8 <= cx <= x1 + 8 and y0 - 8 <= cy <= y1 + 8:
                    hovered = name
                    break
        if hovered != self.hover:
            self.hover, self.hover_t, self.hover_fired = hovered, 0.0, 0
            return
        if hovered is None:
            return
        self.hover_t += dt
        due = HOVER_FIRE + self.hover_fired * HOVER_REPEAT
        if self.hover_t >= due:
            self.hover_fired += 1
            {"club_up": lambda: self.change_club(-1), "club_down": lambda: self.change_club(1),
             "aim_left": lambda: self.adjust_aim(-1), "aim_right": lambda: self.adjust_aim(1)}[hovered]()

    # --- flight ---

    def _update_flight(self, dt: float) -> None:
        f = self.flight
        speed = (PUTT_SPEED if self.flight.shot.club.putter else
                 AIR_SPEED if f.state == "air" else ROLL_SPEED)
        was_air = f.state == "air"
        bounces = f.bounces
        f.step(dt * speed)
        if f.bounces > bounces and f.landing is not None and bounces == 0:
            lie = self.hole.lie_at(*f.landing)
            if lie == "bunker":
                self._burst(f.landing, "sand", 22)
        if was_air and f.state == "water" or f.state == "water" and not was_air:
            self._burst(f.water_entry or (f.x, f.y), "water", 30)
            self.fx.shake = 1.0
        self._follow_camera(dt)
        if f.done:
            self._resolve()

    def _follow_camera(self, dt: float) -> None:
        f = self.flight
        hd = f.shot.heading
        putt = f.shot.club.putter
        back, up, pitch = (5.0, 2.3, math.radians(16.0)) if putt else (13.0, 2.6, math.radians(9.0))
        tz = up + (f.z if not putt else 0.0)  # ride above the ball: it stays just under the centre
        tx, ty = f.x - math.sin(hd) * back, f.y - math.cos(hd) * back
        target = Camera(tx, ty, tz, hd, pitch)
        # Ease from the address view into the chase over CHASE_BLEND, then ride
        # along behind the ball (the ball's own path is smooth enough).
        u = min(1.0, self.phase_t / CHASE_BLEND)
        self.camera = _lerp_cam(self.shot_from_cam, target, u * u * (3 - 2 * u))

    def _resolve(self) -> None:
        f = self.flight
        h = self.hole
        self._result_for = RESULT_SECONDS
        if f.state == "holed":
            self.ball = h.pin
            self._finish_hole()
            return
        if f.state == "water":
            self.strokes += 1
            self.ball = P.drop_point(h, f)
            self.lie = h.lie_at(*self.ball)
            self._say("WATER HAZARD", (255, 190, 90), 1.7, HAZARD_SECONDS, sub="+1 stroke - drop near where it went in")
            self._result_for = HAZARD_SECONDS
        elif f.state == "ob":
            self.strokes += 1
            self.ball = self.prev_spot
            self.lie = self.start_lie
            self._say("OUT OF BOUNDS", (70, 70, 255), 1.7, HAZARD_SECONDS,
                      sub="+1 stroke - play again from the same spot")
            self.fx.shake = max(self.fx.shake, 0.8)
            self._result_for = HAZARD_SECONDS
        else:
            self.ball = (f.x, f.y)
            self.lie = h.lie_at(*self.ball)
            self._comment()
        self.phase, self.phase_t = Phase.RESULT, 0.0

    def _comment(self) -> None:
        """Juice for where a (dry, unholed) shot came to rest."""
        f, shot = self.flight, self.flight.shot
        gold, cyan = (70, 210, 255), (255, 220, 110)
        if shot.club.putter:
            if self.distance_to_pin < 0.6:
                self._say("SO CLOSE!", cyan, 1.5, RESULT_SECONDS)
            return
        if self.lie == "green" and self.start_lie != "green":
            self._say("NICE ON!", gold, 2.2, RESULT_SECONDS)
        elif f.hit_trees:
            self._say("TIMBER!", (90, 190, 255), 1.6, RESULT_SECONDS)
        elif 0.85 <= shot.power <= 1.0 and shot.quality >= 0.8 and self.lie in ("fairway", "green"):
            self._say("NICE SHOT!", gold, 2.2, RESULT_SECONDS)
        elif self.lie == "bunker":
            self._say("BUNKER", (90, 190, 255), 1.4, RESULT_SECONDS)
        elif self.lie == "rough":
            self._say("ROUGH", (175, 175, 190), 1.2, RESULT_SECONDS)

    def _after_result(self) -> None:
        if self.strokes >= self.hole.par + MAX_OVER_PAR:
            self._finish_hole(picked_up=True)
            return
        self.flight = None
        self.message = None
        self.phase, self.phase_t = Phase.AIM, 0.0
        self._setup_shot()

    def _finish_hole(self, picked_up: bool = False) -> None:
        row = self.card[self.hole_i]
        row.strokes = min(self.strokes, self.hole.par + MAX_OVER_PAR)
        row.picked_up = picked_up
        row.points = hole_points(row.strokes, self.hole.par, picked_up)
        row.done = True
        self.fx.score_bump = 1.0
        name = "PICKED UP" if picked_up else result_name(row.strokes, self.hole.par)
        good = not picked_up and row.strokes <= self.hole.par
        colour = (70, 210, 255) if good else (255, 220, 110) if row.strokes == self.hole.par + 1 else (175, 175, 190)
        big = 2.6 if row.strokes < self.hole.par or row.strokes == 1 else 2.0
        self._say(name, colour, big, HOLED_MESSAGE + 0.4)
        self.fx.popups.append(Popup(f"+{row.points}", (70, 210, 255), (self.width / 2, self.height * 0.56)))
        if good:
            self.fx.flash = 0.6 if row.strokes < self.hole.par else 0.0
        self.phase, self.phase_t = Phase.HOLED, 0.0
        hd = self.flight.shot.heading if self.flight is not None else self.auto_heading
        self.holed_from = self.camera.copy()
        self.holed_to, self.celebrate_spot = holed_camera(self.hole.pin, hd)

    # --- effects and Lenny ---

    def _burst(self, at, kind: str, n: int) -> None:
        r = np.random.default_rng()  # cosmetic only
        for _ in range(n):
            a = r.uniform(0, 2 * math.pi)
            s = r.uniform(1.0, 4.0) if kind == "water" else r.uniform(0.8, 2.5)
            self.fx.particles.append(Particle(at[0], at[1], 0.05, math.cos(a) * s * 0.5, math.sin(a) * s * 0.5,
                                              r.uniform(3.0, 7.0) if kind == "water" else r.uniform(2.0, 4.5),
                                              kind, 0.0, r.uniform(0.6, 1.1)))

    def _effects(self, dt: float) -> None:
        fx = self.fx
        fx.shake = max(0.0, fx.shake - dt * 2.0)
        fx.flash = max(0.0, fx.flash - dt * 2.0)
        fx.score_bump = max(0.0, fx.score_bump - dt * 2.5)
        for p in fx.particles:
            p.life += dt
            p.vz -= 9.81 * dt
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.z = max(0.0, p.z + p.vz * dt)
        fx.particles = [p for p in fx.particles if p.life < p.duration]
        for p in fx.popups:
            p.life += dt
        fx.popups = [p for p in fx.popups if p.life < p.duration]
        if self.message is not None:
            self.message.life += dt
            if self.message.life >= self.message.duration:
                self.message = None

    def _animate_lenny(self, dt: float) -> None:
        """Lenny's swing angle: 0 address, 1 top of a full backswing, < 0 follow-through.
        Before impact it follows the player's own grip; after it, a follow-through."""
        putt = self.putting
        finish = -0.45 if putt else -0.95
        if self.follow_t >= 0.0:
            self.follow_t += dt
            t = self.follow_t
            if t < FOLLOW_SECONDS:
                u = t / FOLLOW_SECONDS
                self.lenny_angle = self.follow_from + (finish - self.follow_from) * (1 - (1 - u) ** 3)
            elif self.phase is Phase.AIM and t > FOLLOW_SECONDS + 0.7:
                self.lenny_angle += (0.0 - self.lenny_angle) * min(1.0, dt * 6.0)  # a practice swing: back to address
                if abs(self.lenny_angle) < 0.01:
                    self.follow_t = -1.0
            else:
                self.lenny_angle = finish
            return
        if self.phase is Phase.AIM:
            target = self.detector.swing_angle if self.detector.swinging or self.detector.ready else 0.0
            if putt:
                target *= 0.5
            self.lenny_angle += (target - self.lenny_angle) * min(1.0, dt * 18.0)

    # --- drawing ---

    def _renderer_or_none(self):
        if not hasattr(self, "_renderer"):
            from .art import load_art
            from .render import GolfRenderer

            art = self._art if self._art is not None else load_art()
            self._renderer = GolfRenderer((self.width, self.height), art)
        return self._renderer

    def render(self, frame: np.ndarray) -> None:
        self._renderer_or_none().draw(frame, self)


def _lerp_cam(a: Camera, b: Camera, u: float) -> Camera:
    return Camera(a.x + (b.x - a.x) * u, a.y + (b.y - a.y) * u, a.z + (b.z - a.z) * u,
                  a.yaw + (b.yaw - a.yaw) * u, a.pitch + (b.pitch - a.pitch) * u)
