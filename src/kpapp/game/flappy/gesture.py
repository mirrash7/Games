"""Arm-flap detection: one FlapEvent per wing-beat downstroke.

The game depends on exactly this surface; keep it stable.

**What is measured.** For each arm, the wrist's height above its own shoulder,
in shoulder widths (`h`: 0 = shoulder line, +1 = one shoulder width above it,
negative = below). Shoulder width is the steadiest body scale we get, so the
same flap reads the same near the camera or far from it. Elbows and hips are
not used: on recorded play elbows were missing in 54% of frames with a visible
wrist, and hips are usually out of frame.

**Stepped input.** Poses come from the model at 15-30 Hz but arrive here every
game frame (~60 Hz), so motion shows up as runs of identical frames then a
jump. Each arm keeps only *distinct* observations, stamped with the time they
first appeared. Speed is "how far did the wrist fall from its highest point
within the last SPEED_WINDOW", taken over those distinct observations - so a
value that sat unchanged for 60 ms is not mistaken for one that just arrived,
and a single model step is measured over the real time it took.

**Firing.** A flap fires when the arms are *armed* and the wrists fall at
least FIRE_DROP within SPEED_WINDOW (i.e. faster than FIRE_DROP/SPEED_WINDOW).
Both arms are averaged; one arm alone can fire if it is the only one tracked,
or if it moves SOLO_FACTOR times the threshold. It fires on the downstroke
itself, ~50 ms into a brisk flap (plus up to one model period when poses
arrive in steps), not at the bottom of it.

**Leaving the frame.** On recorded play the player stands close: the shoulder
line sits only 0.1-0.4 shoulder widths above the bottom of the frame, so arms
at the sides are invisible and every downstroke carries the wrists out of
frame. A wrist that vanishes while falling fast still counts as having
completed its downstroke (`exit credit`), otherwise a short, fast flap that
leaves the frame within one or two model steps would be missed.

**Re-arming.** After a flap the detector is disarmed; an arm must then rise
REARM_RISE above its lowest point since the flap before another can fire. This
is the hysteresis that stops one stroke firing twice and stops jitter from
machine-gunning. A REFRACTORY period backs it up.

**Tracking loss.** Any gap in an arm's tracking ends its history; speed is
never measured across a gap, so a wrist that reappears somewhere else cannot
look like a fast stroke.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from ...config import KP
from ...inference import Pose

# --- Thresholds. Heights/distances are in shoulder widths (sw); a shoulder
# width is ~0.35-0.4 m on an adult, so 1 sw/s ~ 0.4 m/s of wrist speed. ------

MIN_CONF = 0.4  # keypoint confidence to count as visible

# Speed window. Must span at least one model period, or a step could fall
# between two windows and the stroke would never be seen whole. 0.10 s covers
# models down to 10 Hz; RF-DETR runs at 15-30 Hz here.
SPEED_WINDOW = 0.10

# Fall within SPEED_WINDOW needed to fire: 0.33 sw in 0.1 s = 3.3 sw/s
# (~1.2-1.3 m/s of wrist speed). Brisk flaps average 6-12 sw/s on the way down
# (2 sw in 0.15-0.3 s) and peak higher; deliberately lowering the arms over
# 1.5 s+ peaks below ~2 sw/s, and even a fairly quick 1.2 s lowering peaks at
# ~2.6. 0.33 sw is also ~10x keypoint jitter (a few px on a ~420 px shoulder
# width at play distance, ~2% of sw even far away). It sets the latency: a
# half-cosine 2 sw downstroke over 0.18 s covers 0.33 sw in ~48 ms.
FIRE_DROP = 0.33
FIRE_SPEED = FIRE_DROP / SPEED_WINDOW  # 3.3 sw/s; also gates exit credit

# Two tracked arms are averaged, so a glitch on one wrist must be twice as big
# to fire. One arm can still carry a flap alone at 2x the threshold (0.66 sw
# in 0.1 s): a vigorous one-armed flap works, a wobble of one arm does not.
SOLO_FACTOR = 2.0

# Exit credit: a wrist that vanishes while falling (at least EXIT_MIN_DROP,
# ~5x jitter, at >= EXIT_SPEED_MIN) is taken to have finished its downstroke
# out of frame if either
#  - it was falling at >= FIRE_SPEED, or
#  - it vanished from well above the lowest point wrists have recently been
#    seen at (the frame's bottom edge when standing close), i.e. it must have
#    covered that gap at >= FIRE_SPEED within one model period.
# At 15-20 Hz a 3 Hz flap may show only 2-3 samples per stroke in view, often
# just 0.1-0.15 sw of fall before the wrist is gone. Lowering the arms slowly
# out of frame is safe: the wrist vanishes right at the lowest point seen (the
# floor follows it down), so its implied speed is ~0. The credit counts as
# one arm's worth of a flap for EXIT_HOLD, so when both wrists are tracked
# both must leave (or the other must fall too): one wrist dropping out while
# the other stays up is not a flap.
EXIT_MIN_DROP = 0.10
EXIT_SPEED_MIN = 1.0
EXIT_HOLD = 0.12
FLOOR_RISE = 0.05  # sw/s: how fast the remembered floor forgets old lows

# Re-arm hysteresis: rise this far above the lowest point since the last flap.
# 0.35 sw is ~10x jitter and well under the ~0.8-1.5 sw an arm travels in a
# rapid 3-4 Hz flap, but larger than the small rebound arms make at the bottom
# of a stroke, so one stroke cannot fire twice.
REARM_RISE = 0.35

# Re-entry: standing close, a wrist that left the bottom of the frame is
# often visible again for only the top 0.5 sw of the next upstroke, sometimes
# just 2-3 model frames - too little to see a full REARM_RISE. A wrist that
# vanished after a flap while falling, or near/below the shoulder line
# (REENTRY_LEVEL), and then comes back REENTRY_RISE (~5x jitter) above where
# it vanished, or climbs that much once back, has made an upstroke.
REENTRY_LEVEL = 0.1
REENTRY_RISE = 0.15

# A flap starts from raised arms: the stroke's high point must be at least
# this high. -0.6 sw is wrists at chest height (wrists hang ~-1.3 sw at the
# sides, a horizontal arm is ~0). It keeps fidgeting with the arms down from
# counting.
ARM_LEVEL = -0.6

# Fresh start (first sight, or after losing tracking): arms already up and
# tracked steadily this long are armed without having to be seen rising.
SETTLE = 0.15

# An arm needs this much continuous tracking before its motion counts, so a
# wrist that has just reappeared (often misplaced for a frame) cannot fire.
MIN_TRACK = 0.05

# Minimum time between flaps. Players spam at 3-4 flaps/s (250-330 ms apart);
# 0.12 s never gets in the way of that, and catches anything hysteresis misses.
REFRACTORY = 0.12

# A jump faster than this between consecutive observations is a tracking
# glitch, not motion. A full 2.4 sw flap at 4 Hz peaks near 30 sw/s (~11 m/s
# at the wrist - about the limit of a human arm), which is 2 sw in one 15 Hz
# step; a fixed step limit would cut real strokes at low model rates.
MAX_SPEED = 40.0  # sw/s

# A wrist that moved less than this many pixels is a repeated frame, not a new
# observation. Real updates differ by more than this from jitter alone.
HOLD_EPS_PX = 0.5

HISTORY = 0.4  # seconds of distinct observations kept per arm
FALL_TOL = 0.05  # sw: jitter allowed within one continuous descent
DEFAULT_PERIOD = 1.0 / 30.0  # model period until one has been measured
LOSS_KEEP = 0.5  # a tracking loss shorter than this keeps the armed state
SCALE_TAU = 0.5  # smoothing of shoulder width; it shrinks ~20% with arms up

# UI. Wing 0 = wrists hanging at the sides (~-1.2 sw), 1 = a little above the
# shoulder line. While the wrists are out of frame (they leave at the bottom of
# every stroke when the player stands close) arms stay "visible" for
# VIS_GRACE, and the wing eases to 0 if they were last seen low.
WING_DOWN, WING_UP = -1.2, 0.2
WING_TAU = 0.05
VIS_GRACE = 0.5
WING_EXIT_LEVEL = 0.3  # last seen below this when vanishing -> went down

_SIDES = (
    (KP["left_shoulder"], KP["left_wrist"]),
    (KP["right_shoulder"], KP["right_wrist"]),
)


@dataclass
class FlapEvent:
    t: float  # time of the flap, same clock passed to update()
    strength: float  # 0..1, how vigorous the downstroke was (for visual juice only)


class _Arm:
    """One arm's recent distinct observations.

    Each entry is [first seen, last seen, wrist height above shoulder in px].
    A repeated (held) frame only extends `last seen`.
    """

    def __init__(self) -> None:
        self.samples: deque[list[float]] = deque()
        self.start = -math.inf  # when this continuous track began
        self.low: float | None = None  # lowest height (px) since the last flap
        self.visible = False
        self.exit_t = -math.inf  # exit credit: when it was given...
        self.exit_speed = 0.0  # ...and for the strength of the flap
        self.exit_peak = 0.0
        self.gone_low_t = -math.inf  # when it last vanished near/below the shoulders
        self.gone_low_px = 0.0  # ...and at what height

    def end_track(self) -> None:
        self.samples.clear()
        self.low = None
        self.visible = False

    def observe(self, t: float, rise_px: float, scale: float, period: float) -> None:
        self.visible = True
        if self.samples:
            last = self.samples[-1]
            if abs(rise_px - last[2]) < HOLD_EPS_PX:
                last[1] = t  # repeated frame: nothing new
                return
            # Elapsed time since the previous model output, not since the
            # held value first appeared: a value held for a second does not
            # make a teleport plausible.
            if abs(rise_px - last[2]) / scale > MAX_SPEED * max(t - last[1], period):
                self.end_track()  # glitch: start over from here
                self.visible = True
        if not self.samples:
            self.start = t
        self.samples.append([t, t, rise_px])
        self.low = rise_px if self.low is None else min(self.low, rise_px)
        while len(self.samples) > 2 and self.samples[0][1] < t - HISTORY:
            self.samples.popleft()

    def height(self, scale: float) -> float:
        return self.samples[-1][2] / scale

    def fall(self, scale: float, t: float, period: float) -> tuple[float, float, float]:
        """(drop, speed, peak) over SPEED_WINDOW ending at the newest sample.

        drop: fall from the highest point of the current descent within the
              window to now, in sw. The descent ends (looking back) where an
              earlier sample is lower by more than FALL_TOL, so an arm that
              bottomed out and is rising again reads no drop.
        speed: that drop over the time since the high point, in sw/s.
        peak: height of that high point, in sw.

        An observation held over several frames is dated to when it first
        appeared - unless it was held longer than a model period, in which
        case the arm really was there until about one period before it last
        showed. That keeps a step from looking faster than it was, without
        making an arm held perfectly still look like it left its spot long ago.
        """
        t_now, _, now = self.samples[-1]
        if t - t_now > SPEED_WINDOW:
            return 0.0, 0.0, now / scale  # nothing new for a while: stopped
        best, t_best, later = now, t_now, now
        for first, last, v in reversed(list(self.samples)[:-1]):
            when = max(first, last - period)
            if when < t_now - SPEED_WINDOW - 1e-6 or v < later - FALL_TOL * scale:
                break
            later = v
            if v > best:
                best, t_best = v, when
        drop = (best - now) / scale
        dt = t_now - t_best
        speed = drop / dt if dt > 1e-6 else 0.0
        return drop, speed, best / scale


class FlapDetector:
    def __init__(self, mirrored: bool = True) -> None:
        # Both arms are used symmetrically, so the mirrored labelling (the
        # player's right arm comes back as left_*) does not change anything.
        self.mirrored = mirrored
        self.reset()

    def reset(self) -> None:
        """Forget all history (called on pause/resume and when tracking is lost)."""
        self._arms = [_Arm(), _Arm()]
        self._scale: float | None = None
        self._t: float | None = None
        self._armed = False
        self._need_rise = False  # True after a flap until the arms rise again
        self._last_flap = -math.inf
        self._lost_since: float | None = None
        self._shoulders = False
        self._wrist_seen = -math.inf
        self._wing: float | None = None
        self._wing_target = 0.0
        self._last_h = 1.0
        self._floor: float | None = None  # lowest wrist height seen lately, sw
        # Model update period, measured from how often the pose changes.
        self._period = DEFAULT_PERIOD
        self._intervals: deque[float] = deque(maxlen=15)
        self._sig = None
        self._sig_t = 0.0

    # --- public ----------------------------------------------------------

    def update(self, pose: Pose | None, t: float) -> FlapEvent | None:
        """Feed one pose per game frame (~60 Hz). Returns a FlapEvent on the
        frame a flap is recognised, else None. At most one event per flap."""
        if self._t is not None and t <= self._t:
            return None
        dt = 0.0 if self._t is None else t - self._t
        self._t = t

        frame = self._read(pose)
        if frame is None:
            self._lose(t)
            return None
        if self._lost_since is not None:
            if t - self._lost_since > LOSS_KEEP:
                self._armed = self._need_rise = False
            self._lost_since = None

        assert pose is not None
        line_y, width = frame
        self._update_period(pose, t)
        self._update_scale(width, dt)
        scale = self._scale
        assert scale is not None

        for arm, (_, wi) in zip(self._arms, _SIDES):
            wrist = pose.point(wi, MIN_CONF)
            if wrist is None:
                if arm.visible:
                    self._credit_exit(arm, t, scale)
                    falling = arm.fall(scale, t, self._period)[0] >= EXIT_MIN_DROP
                    if falling or arm.height(scale) <= REENTRY_LEVEL:
                        arm.gone_low_t = t
                        arm.gone_low_px = arm.samples[-1][2]
                    arm.end_track()
                continue
            arm.observe(t, line_y[wi] - float(wrist[1]), scale, self._period)
            h = arm.height(scale)
            self._floor = h if self._floor is None else min(h, self._floor + FLOOR_RISE * dt)

        visible = [a for a in self._arms if a.visible]
        if visible:
            self._wrist_seen = t
        self._update_wing(visible, scale, dt)

        self._update_arming(visible, t, scale)
        return self._maybe_fire(t, scale)

    @property
    def arms_visible(self) -> bool:
        """True when enough of the arms is tracked to detect flaps at all.

        Shoulders plus at least one wrist, with a VIS_GRACE allowance: when the
        player stands close the wrists leave the bottom of the frame on every
        downstroke, and that is still flapping, not "can't see you".
        """
        if not self._shoulders or self._t is None:
            return False
        return self._t - self._wrist_seen <= VIS_GRACE

    @property
    def wing(self) -> float | None:
        """Current arm height for UI feedback: ~0 arms down at sides, ~1 arms
        raised to shoulder height or above. None if not visible."""
        return self._wing if self.arms_visible else None

    # --- internals -------------------------------------------------------

    def _read(self, pose: Pose | None) -> tuple[dict[int, float], float] | None:
        """Shoulder line y per side, and shoulder width in px; None if lost.

        With one shoulder missing, its partner's height stands in for it and
        the remembered width is used - only if a width has been measured.
        """
        if pose is None:
            return None
        ls = pose.point(KP["left_shoulder"], MIN_CONF)
        rs = pose.point(KP["right_shoulder"], MIN_CONF)
        if ls is not None and rs is not None:
            width = math.hypot(float(ls[0] - rs[0]), float(ls[1] - rs[1]))
            if width < 8.0:
                return None
            ly, ry = float(ls[1]), float(rs[1])
        elif self._scale is not None and (ls is not None or rs is not None):
            only = ls if ls is not None else rs
            ly = ry = float(only[1])
            width = 0.0  # no new measurement
        else:
            return None
        return {KP["left_wrist"]: ly, KP["right_wrist"]: ry}, width

    def _update_period(self, pose: Pose, t: float) -> None:
        """Median interval between changes of the pose: the model's period
        (~0.033-0.067 s), or one game frame if poses are fresh every frame."""
        sig = pose.xy[[KP["left_shoulder"], KP["right_shoulder"],
                       KP["left_wrist"], KP["right_wrist"]]].copy()
        if self._sig is not None and float(abs(sig - self._sig).max()) < HOLD_EPS_PX:
            return
        if self._sig is not None and t - self._sig_t <= 0.2:
            self._intervals.append(t - self._sig_t)
            ordered = sorted(self._intervals)
            self._period = min(max(ordered[len(ordered) // 2], 1.0 / 120.0), SPEED_WINDOW)
        self._sig, self._sig_t = sig, t

    def _update_scale(self, width: float, dt: float) -> None:
        if width <= 0.0:
            return
        if self._scale is None:
            self._scale = width
            return
        # Turning side-on collapses the measured width; do not follow that.
        width = min(max(width, 0.6 * self._scale), 1.6 * self._scale)
        a = 1.0 - math.exp(-dt / SCALE_TAU) if dt > 0 else 0.0
        self._scale += a * (width - self._scale)

    def _lose(self, t: float) -> None:
        """Pose or shoulders gone. Histories end now so no speed is ever
        measured across the gap; the armed state survives a brief loss."""
        for arm in self._arms:
            arm.end_track()
            arm.exit_t = -math.inf
        self._shoulders = False
        self._wing = None
        if self._lost_since is None:
            self._lost_since = t
        if t - self._lost_since > LOSS_KEEP:
            self._scale = None

    def _credit_exit(self, arm: _Arm, t: float, scale: float) -> None:
        """A wrist vanished: if it was falling fast, it finished its stroke
        out of frame."""
        if t - arm.start < MIN_TRACK or len(arm.samples) < 2:
            return
        drop, speed, peak = arm.fall(scale, t, self._period)
        if drop < EXIT_MIN_DROP or peak < ARM_LEVEL or speed < EXIT_SPEED_MIN:
            return
        last_seen_h = arm.height(scale)
        floor = self._floor if self._floor is not None else last_seen_h
        gap_t = max(t - arm.samples[-1][0], self._period)
        implied = (last_seen_h - floor) / gap_t
        if speed >= FIRE_SPEED or implied >= FIRE_SPEED:
            arm.exit_t = t
            arm.exit_speed = max(speed, implied)
            arm.exit_peak = peak

    def _update_arming(self, visible: list[_Arm], t: float, scale: float) -> None:
        if self._armed:
            return
        for arm in visible:
            h = arm.height(scale)
            if h < ARM_LEVEL:
                continue
            # Back in view after leaving low since the last flap: it was at
            # least that low, and a shorter climb proves the upstroke.
            reentered = arm.start > self._last_flap and arm.gone_low_t >= self._last_flap
            low = arm.low
            if reentered and low is not None:
                low = min(low, arm.gone_low_px)
            need = REENTRY_RISE if reentered else REARM_RISE
            rose = low is not None and (arm.samples[-1][2] - low) / scale >= need
            settled = not self._need_rise and t - arm.start >= SETTLE
            if rose or settled:
                self._armed = True
                self._need_rise = False
                return

    def _maybe_fire(self, t: float, scale: float) -> FlapEvent | None:
        evidence: list[float] = []
        speeds: list[float] = []
        peaks: list[float] = []
        for arm in self._arms:
            if arm.visible:
                if t - arm.start < MIN_TRACK:
                    continue
                drop, speed, peak = arm.fall(scale, t, self._period)
                e = drop / FIRE_DROP if peak >= ARM_LEVEL else 0.0
                evidence.append(e)
                if e > 0:
                    speeds.append(speed)
                    peaks.append(peak)
            elif t - arm.exit_t <= EXIT_HOLD:
                evidence.append(1.0)  # one arm's worth of a flap
                speeds.append(arm.exit_speed)
                peaks.append(arm.exit_peak)
        if not evidence or not self._armed or t - self._last_flap < REFRACTORY:
            return None
        mean = sum(evidence) / len(evidence)
        if mean < 1.0 and max(evidence) < SOLO_FACTOR:
            return None

        self._armed = False
        self._need_rise = True
        self._last_flap = t
        for arm in self._arms:
            arm.exit_t = -math.inf
            arm.low = arm.samples[-1][2] if arm.visible else None
        return FlapEvent(t=t, strength=_strength(max(speeds, default=FIRE_SPEED),
                                                 max(peaks, default=0.0)))

    def _update_wing(self, visible: list[_Arm], scale: float, dt: float) -> None:
        self._shoulders = True
        if visible:
            h = sum(a.height(scale) for a in visible) / len(visible)
            self._wing_target = _clip01((h - WING_DOWN) / (WING_UP - WING_DOWN))
            self._last_h = h
        elif self._last_h < WING_EXIT_LEVEL:
            self._wing_target = 0.0  # wrists left the bottom of the frame
        if self._wing is None or dt <= 0.0:
            self._wing = self._wing_target
        else:
            a = 1.0 - math.exp(-dt / WING_TAU)
            self._wing += a * (self._wing_target - self._wing)


def _strength(speed: float, peak: float) -> float:
    """Mostly how fast the wrists were falling when the flap fired (3 sw/s is
    the firing threshold, 12 sw/s a hard flap), plus how high the stroke
    started (shoulder line = 0, a full sw above = 1)."""
    s_speed = _clip01((speed - 3.0) / 9.0)
    s_amp = _clip01((peak + 0.3) / 1.3)
    return _clip01(0.15 + 0.6 * s_speed + 0.25 * s_amp)


def _clip01(x: float) -> float:
    return min(max(x, 0.0), 1.0)
