"""Golf swings from a webcam: address, backswing (power), downswing (impact).

The game depends on exactly this surface (`SwingDetector.update` returning
`SwingEvent`s, plus the read-only state the HUD shows); keep it stable.

**What is measured.** The *grip*: the midpoint of the two wrists. Not the
palms (`hand.hand_point`), which matter for pointing but hurt here: the palm
sits 0.38 of a forearm past the wrist only in frames where the elbow is seen
(it is missing in about half of them), so the grip would jump ~0.25 sw from
frame to frame - more than the address hold tolerates, and enough to move a
backswing's peak. Without an elbow, `hand_point` assumes a raised pointing
forearm, which at a golf grip (forearms hanging) would put the palm above the
wrist. A swing is measured relative to its own address, so the wrists' fixed
offset from the hands costs nothing. Positions are in
shoulder widths (sw) in a body frame: `u` sideways from the shoulder midpoint
(screen direction), `h` height above the shoulder line (+ up). Shoulder width
is the steadiest body scale we get, and heights come from the shoulder line,
not the hips, which are often out of frame (AGENTS.md section 6).

**The body frame freezes during a swing.** Seen from the front, the shoulders
turn away at the top of a backswing and their apparent width shrinks by half
or more; normalising by that width would make the hands look twice as high.
The frame (shoulder midpoint and width) is smoothed while the player stands at
address and frozen from the start of the backswing to the end of the swing.

**Stepped input.** Like `FlapDetector`, the detector reads `controls.step_pose`
(poses at 15-30 Hz, held between model updates), keeps only *distinct*
observations, and dates a held one to max(first seen, last seen - one model
period): a grip held still at the top and then seen far below one step later
moved in one period, not over the whole hold.

**Stages.**

    NO_GRIP     hands apart, hidden, or not low: show what to fix
    ADDRESSING  both hands together (palms within TOGETHER sw), low (h <=
                ADDRESS_MAX_H, about the waist), near the midline, and still:
                READY after ADDRESS_HOLD seconds of that
    READY       Lenny addresses the ball (Wii's "hold A"). The address height
                and position are the mean over the hold.
    BACKSWING   the grip rose START_RISE above address (or moved START_SIDE to
                either side - left- and right-handed swings both work). The
                gauge follows the grip; the peak sets the power.
    DOWNSWING   the grip fell DOWN_DROP below its peak.

Impact is the grip's *swept segment* between two observations crossing the
address line (u = address u) toward the other side - from the front, a
golf swing's hands arc down through the bottom centre - below the shoulder
line, with the grip moving at least HIT_SPEED (windowed over SPEED_WINDOW).
The crossing point is interpolated along the segment, never "is the grip
there now": at 15 Hz a fast swing jumps ~0.7 sw between samples. A swing whose
backswing went straight up (no side) hits when the grip falls fast through
CHOP_LINE above address instead.

A slow lowering of the hands is not a hit: crossing (or coming back down to
address) slower than HIT_SPEED cancels the swing with "SWING THROUGH!".

**Power** = (peak - address height) / (FULL_H - address height): hands back up
at about head height (FULL_H above the shoulder line) is 100%. Above that is
the overswing zone, up to OVERSWING_MAX (115%): the game adds a hook or slice.

**Strike quality** comes from the grip's height where it crossed, against the
address height: within CLEAN_H it is clean (1.0); THIN_H or more above it is
thin (0.0: lower and ~70% as far, in physics.py). Measured on the swept
segment, so latency does not change it.

**Practice swings.** A swing that starts before READY (hands together, but not
yet held still) is reported with `practice=True`: the gauge and Lenny animate,
nothing is hit. That is Wii's practice swing without holding A.

**Putting** (`mode="putt"`): a pendulum. The backstroke is the grip's
sideways travel from address (PUTT_START to start); power = peak travel /
PUTT_FULL (no overswing). Impact = crossing back over the address point
within PUTT_STROKE seconds of starting forward. Timed, not a speed threshold:
a short putt's stroke is as slow as keypoint jitter (0.15 sw in 0.4 s), but no
real stroke takes over a second to come back through the ball.

**Tracking.** A jump larger than TELEPORT (a fraction of the frame width)
between observations is a glitch: history is dropped (lost, then found) and
no speed or crossing is measured across it. Losing the grip for longer than
LOSS_KEEP cancels a swing in progress - never a stroke for tracking loss.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum

from ...config import KP
from ...inference import Pose

MIN_CONF = 0.4

# --- address -----------------------------------------------------------------
TOGETHER = 0.7  # sw between wrists: a two-handed grip reads ~0.1-0.3
ADDRESS_MAX_H = -0.7  # grip at least this far below the shoulder line (waist is ~-1.2)
ADDRESS_MAX_U = 0.6  # ...and within this of the body's midline
STEADY = 0.15  # sw: how far the grip may wander during the hold (~6x jitter)
ADDRESS_HOLD = 0.4  # s held still -> READY
FRAME_TAU = 0.35  # s: smoothing of the body frame while it is live

# --- the swing ---------------------------------------------------------------
START_RISE = 0.30  # sw above address: the backswing has begun
START_SIDE = 0.55  # ...or this far to either side (a low, wide takeaway)
FULL_H = 0.75  # sw above the shoulder line: hands at about head height = 100%
OVERSWING_MAX = 1.15
DOWN_DROP = 0.25  # sw below the peak: the downswing has begun
SIDE_MIN = 0.25  # sw of sideways travel at the peak to know which side it went
HIT_SPEED = 3.5  # sw/s through the hitting zone. A real downswing peaks at 8-14;
#                  lowering the hands over 1.5 s peaks under 2.5
HIT_MAX_H = -0.15  # the crossing must be below the shoulder line
CHOP_LINE = 0.45  # sw above address: the hitting line for a swing with no side
CLEAN_H = 0.35  # crossing within this of address height: a clean strike
THIN_H = 1.0  # this far above it: fully thin
SWING_TIMEOUT = 1.2  # s a downswing may take before it counts as lowering the hands

# --- putting -----------------------------------------------------------------
PUTT_START = 0.09  # sw sideways: the backstroke has begun. ~6x the smoothed jitter at address
#                   (0.014 sw at 5 px of noise); the gentlest putt rolls ~1.2 m, under capture speed
PUTT_FULL = 1.2  # sw of backstroke = full putter power
PUTT_RETURN = 0.05  # sw back from the peak: the forward stroke has begun
PUTT_SMOOTH = 3  # observations averaged for the putter's sideways position
PUTT_STROKE = 1.0  # s from the top of the backstroke back to the ball, at most

# --- tracking ----------------------------------------------------------------
SPEED_WINDOW = 0.10  # s; spans at least one model period down to 10 Hz
MAX_GAP = 0.25  # s between observations beyond which speed is not measured
TELEPORT = 0.35  # of the frame width between observations: a tracking glitch
LOSS_KEEP = 0.30  # s without a grip before a swing in progress is dropped
ONE_HAND_KEEP = 0.30  # s a swing may continue on one visible palm
HOLD_EPS = 0.5  # px: a grip that moved less is a repeated (held) frame
DEFAULT_PERIOD = 1.0 / 20.0
HINT_SECONDS = 1.6
# Framing: room below the shoulder line, in sw, needed to see the hands at
# address. Separate on/off levels so the prompt does not flicker.
TOO_CLOSE_ON = 1.3
TOO_CLOSE_OFF = 1.5


class Stage(str, Enum):
    NO_GRIP = "no_grip"
    ADDRESSING = "addressing"
    READY = "ready"
    BACKSWING = "backswing"
    DOWNSWING = "downswing"


@dataclass
class SwingEvent:
    t: float  # same clock as update()
    power: float  # 0..OVERSWING_MAX (putts 0..1)
    quality: float  # 1 clean .. 0 thin (putts: 1)
    side: int  # -1 / +1: the side of the body (screen) the backswing went; 0 unknown
    speed: float  # sw/s through the hitting zone (for juice)
    practice: bool  # swung without addressing first: animate only, no stroke

    @property
    def overswing(self) -> float:
        """0 .. 1 through the red zone."""
        return max(0.0, min(1.0, (self.power - 1.0) / (OVERSWING_MAX - 1.0)))


def wrist(pose: Pose, index: int) -> tuple[float, float] | None:
    w = pose.point(index, MIN_CONF)
    return None if w is None else (float(w[0]), float(w[1]))


class SwingDetector:
    def __init__(self, mode: str = "full", frame_width: float = 1280.0, frame_height: float = 720.0) -> None:
        self.frame_width = float(frame_width)
        self.frame_height = float(frame_height)
        self.mode = mode
        self.reset()

    # --- lifecycle ---

    def reset(self) -> None:
        """Forget everything (on resume, between shots, on a mode change)."""
        self.stage = Stage.NO_GRIP
        self._t: float | None = None
        self._frame: tuple[float, float, float] | None = None  # shoulder mid x, y, sw (px)
        self._frame_t = 0.0
        self._obs: deque = deque()  # [first seen, last seen, u, h] distinct grip observations
        self._grip_px: tuple[float, float] | None = None
        self._lost_since: float | None = None
        self._one_hand_since: float | None = None
        self._hold: deque = deque()  # (t, u, h) during ADDRESSING
        self._low = (0.0, 0.0)  # lowest grip since ADDRESSING began
        self._hold_start = 0.0
        self.address: tuple[float, float] | None = None  # (u, h) at READY
        self.armed = False  # this swing started from READY
        self.peak_h = -math.inf
        self.peak_u = 0.0
        self.side = 0
        self.power = 0.0  # live gauge: 0..OVERSWING_MAX
        self.locked = False  # the gauge stopped following (downswing)
        self._swing_t = 0.0
        self._down_t = 0.0
        self.hint: str | None = None
        self._hint_until = -math.inf
        self.too_close = False
        self.visible = False  # shoulders seen
        self.together = False
        self.hands_seen = 0  # 0, 1 or 2 palms this frame
        self._period = DEFAULT_PERIOD
        self._intervals: deque = deque(maxlen=12)
        self._last_new: float | None = None
        self.last_event: SwingEvent | None = None
        self._pu: float | None = None  # smoothed sideways grip (putting)
        self._pu_prev: float | None = None
        self._putt_out = 0

    def set_mode(self, mode: str) -> None:
        if mode != self.mode:
            self.mode = mode
            self.reset()

    # --- read-only state for the HUD ---

    @property
    def ready(self) -> bool:
        return self.stage is Stage.READY

    @property
    def swinging(self) -> bool:
        return self.stage in (Stage.BACKSWING, Stage.DOWNSWING)

    @property
    def hold_progress(self) -> float:
        if self.stage is Stage.ADDRESSING and self._hold:
            return min(1.0, (self._hold[-1][0] - self._hold_start) / ADDRESS_HOLD)
        return 1.0 if self.stage in (Stage.READY, Stage.BACKSWING, Stage.DOWNSWING) and self.armed else 0.0

    @property
    def grip(self) -> tuple[float, float] | None:
        """The grip in frame pixels (for the camera window)."""
        return self._grip_px

    @property
    def body_frame(self) -> tuple[float, float, float] | None:
        """(shoulder midpoint x, y, shoulder width) in px, as the detector uses it."""
        return self._frame

    @property
    def grip_body(self) -> tuple[float, float] | None:
        """The newest grip in the body frame (u, h), sw."""
        return (self._obs[-1][2], self._obs[-1][3]) if self._obs else None

    @property
    def swing_angle(self) -> float:
        """How far round the swing is, for Lenny: 0 address, 1 top of a full backswing,
        beyond 1 overswing. Follows the live grip, so Lenny's arms move with the player's."""
        if self.mode == "putt":
            if self.address is None or not self._obs:
                return 0.0
            return max(-1.0, min(1.0, abs(self._obs[-1][2] - self.address[0]) / PUTT_FULL))
        if self.address is None or not self._obs or self.stage in (Stage.NO_GRIP, Stage.ADDRESSING):
            return 0.0
        return max(0.0, self._power_at(self._obs[-1][3]))

    # --- the main entry point ---

    def update(self, pose: Pose | None, t: float) -> SwingEvent | None:
        """Feed one (stepped) pose per game frame. Returns a SwingEvent on the
        frame a swing (or practice swing) is recognised, else None."""
        if self._t is not None and t <= self._t:
            return None
        self._t = t
        if t >= self._hint_until:
            self.hint = None
        grip = self._read(pose, t)
        if grip is None:
            self._lose(t)
            return None
        self._lost_since = None
        u, h, new = grip
        if not new:
            return self._held(t)
        return self._advance(t, u, h)

    # --- reading the pose ---

    def _read(self, pose: Pose | None, t: float):
        """(u, h, is_new) for this frame's grip, or None when there is none."""
        self.visible = False
        self.together = False
        self.hands_seen = 0
        if pose is None:
            return None
        ls = pose.point(KP["left_shoulder"], MIN_CONF)
        rs = pose.point(KP["right_shoulder"], MIN_CONF)
        frozen = self.stage in (Stage.BACKSWING, Stage.DOWNSWING)
        if ls is not None and rs is not None:
            self.visible = True
            mx, my = float(ls[0] + rs[0]) / 2, float(ls[1] + rs[1]) / 2
            sw = max(1.0, math.hypot(float(ls[0] - rs[0]), float(ls[1] - rs[1])))
            room = (self.frame_height - my) / sw
            self.too_close = room < (TOO_CLOSE_OFF if self.too_close else TOO_CLOSE_ON)
            if not frozen:
                if self._frame is None:
                    self._frame = (mx, my, sw)
                else:
                    a = 1.0 - math.exp(-max(0.0, t - self._frame_t) / FRAME_TAU)
                    fx, fy, fs = self._frame
                    self._frame = (fx + (mx - fx) * a, fy + (my - fy) * a, fs + (sw - fs) * a)
                self._frame_t = t
        elif self._frame is None or not frozen:
            return None
        pl = wrist(pose, KP["left_wrist"])
        pr = wrist(pose, KP["right_wrist"])
        self.hands_seen = (pl is not None) + (pr is not None)
        fx, fy, sw = self._frame
        if pl is not None and pr is not None:
            self.together = math.hypot(pl[0] - pr[0], pl[1] - pr[1]) <= TOGETHER * sw
            gx, gy = (pl[0] + pr[0]) / 2, (pl[1] + pr[1]) / 2
            if not self.together:
                if frozen:
                    return None  # hands flew apart mid-swing: treat as lost
                self._grip_px = (gx, gy)
                self._obs.clear()
                self._to_no_grip(t)
                return None
            self._one_hand_since = None
        elif (pl is not None or pr is not None) and frozen:
            # Mid-swing a wrist can drop out for a frame or two (it passes the
            # body, or blurs): carry on with the one that is seen, briefly.
            if self._one_hand_since is None:
                self._one_hand_since = t
            if t - self._one_hand_since > ONE_HAND_KEEP:
                return None
            gx, gy = pl if pl is not None else pr
        else:
            return None
        self._grip_px = (gx, gy)
        u, h = (gx - fx) / sw, (fy - gy) / sw
        new = self._observe(t, u, h, gx, gy)
        return u, h, new

    def _observe(self, t: float, u: float, h: float, gx: float, gy: float) -> bool:
        """Record a distinct observation; False for a repeated (held) frame."""
        if self._obs:
            last = self._obs[-1]
            sw = self._frame[2]
            du, dh = u - last[2], h - last[3]
            if math.hypot(du, dh) * sw < HOLD_EPS:
                last[1] = t
                return False
            if math.hypot(du, dh) * sw > TELEPORT * self.frame_width:
                # A glitch: lost, then found here. No speed, peak or crossing
                # across it; mid-swing the found sample only starts new history.
                self._obs.clear()
                if self.swinging:
                    self._obs.append([t, t, u, h])
                    return False
        if self._last_new is not None:
            gap = t - self._last_new
            if 0.0 < gap < MAX_GAP:
                self._intervals.append(gap)
                s = sorted(self._intervals)
                self._period = s[len(s) // 2]
        self._last_new = t
        self._obs.append([t, t, u, h])
        while len(self._obs) > 2 and self._obs[0][1] < t - 1.0:
            self._obs.popleft()
        return True

    def _speed(self) -> float:
        """Grip speed (sw/s) over SPEED_WINDOW ending at the newest observation."""
        if len(self._obs) < 2:
            return 0.0
        t_now, _, u1, h1 = self._obs[-1]
        ref = None
        for first, last, u0, h0 in reversed(list(self._obs)[:-1]):
            when = max(first, last - self._period)
            if ref is None or when >= t_now - SPEED_WINDOW - 1e-9:
                ref = (when, u0, h0)
            else:
                break
        when, u0, h0 = ref
        dt = t_now - when
        if dt <= 1e-6 or dt > MAX_GAP:
            return 0.0
        return math.hypot(u1 - u0, h1 - h0) / dt

    # --- stages ---

    def _say(self, hint: str, t: float) -> None:
        self.hint = hint
        self._hint_until = t + HINT_SECONDS

    def _to_no_grip(self, t: float) -> None:
        if self.stage is not Stage.NO_GRIP:
            self.stage = Stage.NO_GRIP
        self._hold.clear()
        self.armed = False
        self.locked = False
        self.power = 0.0

    def _lose(self, t: float) -> None:
        if self._lost_since is None:
            self._lost_since = t
        if self.swinging and t - self._lost_since <= LOSS_KEEP:
            return  # a short dropout mid-swing: keep going
        if self.stage is not Stage.NO_GRIP:
            self._obs.clear()
            self._to_no_grip(t)

    def _power_at(self, h: float) -> float:
        if self.mode == "putt" or self.address is None:
            return 0.0
        span = max(0.5, FULL_H - self.address[1])
        return min(OVERSWING_MAX, max(0.0, (h - self.address[1]) / span))

    def _held(self, t: float) -> SwingEvent | None:
        """A repeated frame: time passes, nothing moved."""
        if self.stage is Stage.ADDRESSING and self._hold:
            self._check_hold(t)
        elif self.stage is Stage.DOWNSWING and t - self._down_t > (PUTT_STROKE if self.mode == "putt"
                                                                   else SWING_TIMEOUT):
            self._cancel(t)
        return None

    def _addressable(self, u: float, h: float) -> bool:
        return h <= ADDRESS_MAX_H and abs(u) <= ADDRESS_MAX_U

    def _advance(self, t: float, u: float, h: float) -> SwingEvent | None:
        if self.mode == "putt":
            # A putt moves the hands only ~0.1-1.2 sw, so jitter matters: use
            # the mean of the last PUTT_SMOOTH observations (noise / sqrt(3)).
            recent = [o[2] for o in list(self._obs)[-PUTT_SMOOTH:] if o[0] >= t - 0.25]
            self._pu_prev = self._pu
            u = self._pu = sum(recent) / len(recent) if recent else u
        st = self.stage
        if st is Stage.NO_GRIP:
            if self._addressable(u, h):
                self._begin_hold(t, u, h)
            return None
        if st is Stage.ADDRESSING:
            if not self._addressable(u, h):
                # Hands rising from an unsteady address: a practice swing.
                if self._hold and self._starts_swing(u, h, self._hold_ref()):
                    self._begin_swing(t, armed=False, ref=self._hold_ref())
                    return self._advance(t, u, h)
                self._to_no_grip(t)
                return None
            self._hold.append((t, u, h))
            if h < self._low[1]:
                self._low = (u, h)
            ref = self._hold_ref()
            if self._starts_swing(u, h, ref):
                self._begin_swing(t, armed=False, ref=ref)
                return None
            if max(abs(u - p[1]) + abs(h - p[2]) for p in self._hold) > STEADY * 1.5:
                # Still moving: restart the hold from here.
                while len(self._hold) > 1 and (abs(u - self._hold[0][1]) + abs(h - self._hold[0][2])
                                               > STEADY * 1.5):
                    self._hold.popleft()
                self._hold_start = self._hold[0][0]
            self._check_hold(t)
            return None
        if st is Stage.READY:
            assert self.address is not None
            if self._starts_swing(u, h, self.address):
                self._begin_swing(t, armed=True, ref=self.address)
                return self._advance(t, u, h)
            if h < self.address[1] - 0.35:
                # Hands dropped (the player bent down or stepped): address again.
                self._begin_hold(t, u, h)
            return None
        if st is Stage.BACKSWING:
            return self._backswing(t, u, h)
        if st is Stage.DOWNSWING:
            return self._downswing(t, u, h)
        return None

    def _hold_ref(self) -> tuple[float, float]:
        """Where a practice swing is measured from: the lowest point the grip
        reached since it came into the address zone (the hold itself is pruned
        while the hands move, so it would drift along with them)."""
        return self._low

    def _begin_hold(self, t: float, u: float, h: float) -> None:
        self.stage = Stage.ADDRESSING
        self._low = (u, h)
        self._putt_out = 0
        self._hold.clear()
        self._hold.append((t, u, h))
        self._hold_start = t
        self.address = None
        self.armed = False
        self.power = 0.0
        self.locked = False

    def _check_hold(self, t: float) -> None:
        if t - self._hold_start >= ADDRESS_HOLD:
            pts = list(self._hold)
            self.address = (sum(p[1] for p in pts) / len(pts), sum(p[2] for p in pts) / len(pts))
            self.stage = Stage.READY
            self.armed = True

    def _starts_swing(self, u: float, h: float, ref: tuple[float, float]) -> bool:
        if self.mode == "putt":
            # Two observations in a row past PUTT_START: one jittery sample is not a backstroke.
            self._putt_out = self._putt_out + 1 if abs(u - ref[0]) >= PUTT_START else 0
            return self._putt_out >= 2
        return h - ref[1] >= START_RISE or abs(u - ref[0]) >= START_SIDE

    def _begin_swing(self, t: float, armed: bool, ref: tuple[float, float]) -> None:
        self.stage = Stage.BACKSWING
        self._putt_out = 0
        self.armed = armed
        self.address = ref
        self.peak_h = ref[1]
        self.peak_u = ref[0]
        self.side = 0
        self.power = 0.0
        self.locked = False
        self._swing_t = t
        self._hold.clear()

    def _backswing(self, t: float, u: float, h: float) -> SwingEvent | None:
        a_u, a_h = self.address
        if self.mode == "putt":
            travel = abs(u - a_u)
            if travel >= abs(self.peak_u - a_u):
                self.peak_u = u
                self.side = 1 if u > a_u else -1
            self.power = min(1.0, abs(self.peak_u - a_u) / PUTT_FULL)
            if travel <= abs(self.peak_u - a_u) - PUTT_RETURN:
                self.stage = Stage.DOWNSWING
                self.locked = True
                self._down_t = t
                return self._downswing(t, u, h)
            return None
        if h >= self.peak_h:
            self.peak_h, self.peak_u = h, u
        self.power = self._power_at(max(h, a_h))
        if h <= self.peak_h - DOWN_DROP:
            off = self.peak_u - a_u
            self.side = (1 if off > 0 else -1) if abs(off) >= SIDE_MIN else 0
            self.power = self._power_at(self.peak_h)
            self.locked = True
            self.stage = Stage.DOWNSWING
            self._down_t = t
            return self._downswing(t, u, h)
        return None

    def _downswing(self, t: float, u1: float, h1: float) -> SwingEvent | None:
        a_u, a_h = self.address
        if len(self._obs) < 2:
            return None
        _, _, u0, h0 = self._obs[-2]
        speed = self._speed()
        putt = self.mode == "putt"
        if putt:
            u0 = self._pu_prev if self._pu_prev is not None else u0
            crossed = self.side != 0 and (u0 - a_u) * self.side > 0 >= (u1 - a_u) * self.side
            if crossed:
                if t - self._down_t <= PUTT_STROKE:
                    return self._fire(t, self.power, 1.0, speed)
                return self._cancel(t)
            if t - self._down_t > PUTT_STROKE:
                return self._cancel(t)
            return None

        if self.side != 0:
            crossed = (u0 - a_u) * self.side > 0 >= (u1 - a_u) * self.side
            if crossed:
                f = (u0 - a_u) / (u0 - u1) if u0 != u1 else 1.0
                h_c = h0 + (h1 - h0) * f
            else:
                h_c = None
        else:
            line = a_h + CHOP_LINE
            crossed = h0 > line >= h1
            h_c = line if crossed else None
        if crossed and h_c is not None and h_c <= HIT_MAX_H:
            if speed >= HIT_SPEED:
                q = 1.0 - (h_c - a_h - CLEAN_H) / (THIN_H - CLEAN_H)
                return self._fire(t, self.power, min(1.0, max(0.0, q)), speed)
            return self._cancel(t)
        # Back down at address height without crossing, slowly: lowering, not swinging.
        if h1 <= a_h + 0.25 and speed < HIT_SPEED:
            return self._cancel(t)
        if t - self._down_t > SWING_TIMEOUT:
            return self._cancel(t)
        return None

    def _fire(self, t: float, power: float, quality: float, speed: float) -> SwingEvent:
        ev = SwingEvent(t=t, power=power, quality=quality, side=self.side, speed=speed,
                        practice=not self.armed)
        self.last_event = ev
        # After a swing the player must address again (and the game decides
        # whether a ball was hit).
        self.stage = Stage.NO_GRIP
        self.armed = False
        self.locked = True
        self.address = None
        self._hold.clear()
        return ev

    def _cancel(self, t: float) -> None:
        if self.armed or self.swinging:
            self._say("SWING THROUGH!", t)
        self.stage = Stage.NO_GRIP
        self.armed = False
        self.locked = False
        self.power = 0.0
        self.address = None
        self._hold.clear()
        return None

    # --- guidance ---

    def guidance(self) -> str | None:
        """What the player should do next, for the HUD (None = nothing to fix)."""
        if self.hint:
            return self.hint
        if not self.visible and self._frame is None:
            return "STEP IN FRONT OF THE CAMERA"
        if self.too_close and self.stage in (Stage.NO_GRIP, Stage.ADDRESSING):
            return "STEP BACK - SHOW BOTH HANDS"
        if self.stage is Stage.NO_GRIP:
            if self.hands_seen < 2:
                return "SHOW BOTH HANDS"
            if not self.together:
                return "HOLD BOTH HANDS TOGETHER"
            return "HANDS LOW, IN FRONT OF YOU"
        if self.stage is Stage.ADDRESSING:
            return "HOLD STILL"
        return None
