"""The coach's motion: how it moves between hits and grooves while it holds them.

`Song.coach_pose(beat)` is the *target* timeline: it hits every keyframe on its
beat and eases between keyframes with a plain smoothstep. Bots, tests and the
parity fixture copy it, so it stays as it is. This module is the *performance*
layered on top for the drawn coach only (docs/DANCE.md section 4a):

- **Transitions** keep the chart's timing (the move starts EASE_BEATS before
  its beat) but anticipate (a small wind-up the other way), arrive a little
  early with a small overshoot, and settle back onto the target exactly on the
  beat. The upper arm leads and the forearm follows (overlapping action), so
  limbs swing in soft arcs instead of rotating as rigid sticks.
- **Groove** locked to the beat: a dip on every beat (knees bend, the torso
  squashes), weight shifting from foot to foot, alternate shoulder rolls, a
  little lean sway and torso twist, a head bob that lags the body.
- **Follow-through** from lagged samples of the same pure timeline: the head
  tilt trails the torso, the tail swings against the body's motion, the hands
  flick behind the forearm, and a two-footed move hops.
- **Depth**: a fake torso yaw (twist) and a depth for each arm, which the
  painter uses for draw order (an arm crossing the body passes in front, the
  far arm of a twist goes behind the jacket) and foreshortening.

The guarantee the judge depends on: **at every keyframe beat the arm and leg
angles in the torso frame equal the chart's exactly**. Everything else here
is either zero at keyframe beats (overshoot, anticipation, hops) or moves
whole body parts without turning a limb (bounce, sway, shoulder roll,
squash, twist); the few groove terms that do touch judged numbers (the dip's
knee bend, the lean sway, the weight shift) stay within a few degrees, and
tests/test_dance_motion.py pins the drawn coach within 6 degrees of every
judged pose on every song and difficulty.

Everything is a pure function of (song keyframes, beat): no wall clock, no
state, so `web/js/games/dance/motion.js` is a line-for-line port and
web/tests/fixtures/dance_motion.json (tools/make_dance_motion_fixture.py)
pins the two together.
"""

from __future__ import annotations

import math

from .chart import EASE_BEATS, idle_groove
from .pose import FOREARM, HAND, SCALARS, UPPER_ARM, normalize_pose, smoothstep

# --- transitions ---------------------------------------------------------------

ANTICIPATE = 0.10  # wind-up, as a fraction of the move (capped below)
OVERSHOOT = 0.10  # overshoot before settling on the beat, as a fraction of the move
# Caps on both, per channel (degrees, or shoulder widths for shift, 0..1 for squat).
CAPS = {"limb": 8.0, "lean": 2.5, "head": 4.0, "shift": 0.03, "squat": 0.04}
ARRIVE = (0.12, 0.80)  # the main ease runs over this part of the transition
ANTIC_SPAN = (0.0, 0.32)
OVER_SPAN = (0.60, 1.0)  # zero (and flat) again exactly on the beat
LEAD_UPPER = 0.92  # the upper arm finishes at 92% of the transition...
LAG_FORE = 0.06  # ...the forearm starts 6% late; both land on the beat
LAG_LEG = 0.03

# --- groove ------------------------------------------------------------------------

SECTION_GROOVE = (("chorus", 1.0), ("verse", 0.8), ("bridge", 0.6), ("intro", 0.65), ("outro", 0.5))
DIP_SQUAT = 0.045  # knee bend added at the bottom of each beat's dip (thigh +1.7 degrees)
LEAN_SWAY = 2.0  # degrees, over two beats; zero on every beat
SWAY = 0.035  # shoulder widths of weight shift (upper body over the feet), over two beats
SHOULDER_ROLL = 0.035  # shoulder widths, alternate shoulders, over two beats
SQUASH = 0.045  # torso length lost at the bottom of the dip
STRETCH = 0.018  # and gained on the off-beat
TWIST_GROOVE = 7.0  # degrees of torso yaw, over two beats
TWIST_CROSS = 24.0  # degrees of torso yaw when one arm reaches right across the body
HEAD_BOB = 0.035  # shoulder widths
HEAD_SWAY = 3.0  # degrees
TAIL_SWAY = 12.0  # degrees

# --- follow-through --------------------------------------------------------------

LAG = 0.2  # beats: how far behind the follow-through samples the timeline
LAG_SHORT = 0.08  # beats: the hands' (quicker) follow-through, and the twist's smoothing step
HAND_FLICK, HAND_MAX = 0.6, 30.0  # a glove trails its forearm by this much of the lag, at most this many degrees
HOP = 0.22  # shoulder widths: a two-footed leg change hops this high
HOP_LEGS = 20.0  # degrees of leg change (each leg) that makes a hop


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _bump(x: float, lo: float, hi: float) -> float:
    """0 outside (lo, hi), rising to 1 in the middle, flat (zero slope) at both ends."""
    if x <= lo or x >= hi:
        return 0.0
    s = math.sin(math.pi * (x - lo) / (hi - lo))
    return s * s


def ease_value(v0: float, v1: float, x: float, cap: float) -> float:
    """One channel of a transition at x (0..1): anticipate, arrive early,
    overshoot, settle. Exactly v0 at x <= 0 and v1 at x >= 1."""
    if x <= 0.0:
        return v0
    if x >= 1.0:
        return v1
    d = v1 - v0
    s = smoothstep((x - ARRIVE[0]) / (ARRIVE[1] - ARRIVE[0]))
    sign = 1.0 if d >= 0 else -1.0
    mag = abs(d)
    wind = min(mag * ANTICIPATE, cap) * _bump(x, *ANTIC_SPAN)
    over = min(mag * OVERSHOOT, cap) * _bump(x, *OVER_SPAN)
    return v0 + d * s + sign * (over - wind)


def ease_pose(p0: dict, p1: dict, x: float) -> dict:
    """A transition between two complete poses at x (0..1), with the upper arm
    leading and the forearm and legs following."""
    out = {}
    lim = CAPS["limb"]
    xu = _clamp(x / LEAD_UPPER, 0.0, 1.0)
    xf = _clamp((x - LAG_FORE) / (1.0 - LAG_FORE), 0.0, 1.0)
    xl = _clamp((x - LAG_LEG) / (1.0 - LAG_LEG), 0.0, 1.0)
    for k in ("la", "ra"):
        out[k] = [ease_value(p0[k][0], p1[k][0], xu, lim), ease_value(p0[k][1], p1[k][1], xf, lim)]
    for k in ("ll", "rl"):
        out[k] = [ease_value(p0[k][0], p1[k][0], xl, lim), ease_value(p0[k][1], p1[k][1], xl, lim)]
    for k in SCALARS:
        out[k] = ease_value(p0[k], p1[k], x, CAPS[k])
    out["squat"] = _clamp(out["squat"], 0.0, 1.0)
    return out


def _segment(frames, beat: float):
    """(p0, p1, x, b1, ease) for the transition in progress at `beat`, or
    (pose, None, 1.0, ...) while a pose is held. Before the first keyframe the
    idle groove eases into it; after the last, the last pose holds."""
    if not frames or beat < frames[0][0] - EASE_BEATS:
        return idle_groove(beat), None, 1.0, None, 0.0
    if beat < frames[0][0]:
        b1 = frames[0][0]
        return idle_groove(beat), frames[0][1], (beat - (b1 - EASE_BEATS)) / EASE_BEATS, b1, EASE_BEATS
    lo, hi = 0, len(frames) - 1
    if beat >= frames[hi][0]:
        return frames[hi][1], None, 1.0, frames[hi][0], 0.0
    while hi - lo > 1:  # last frame at or before `beat`
        mid = (lo + hi) // 2
        if frames[mid][0] <= beat:
            lo = mid
        else:
            hi = mid
    (b0, p0), (b1, p1) = frames[lo], frames[lo + 1]
    ease = min(EASE_BEATS, b1 - b0)
    if ease <= 0 or beat <= b1 - ease:
        return p0, None, 1.0, b0, 0.0
    return p0, p1, (beat - (b1 - ease)) / ease, b1, ease


def performed_pose(frames, beat: float) -> dict:
    """The coach's limbs at `beat` before the groove: the chart's keyframes with
    anticipation, overshoot and overlapping action. Equals the keyframe on its beat."""
    p0, p1, x, _, _ = _segment(frames, beat)
    if p1 is None:
        return normalize_pose(p0)
    return ease_pose(normalize_pose(p0), normalize_pose(p1), x)


def _hop(frames, beat: float) -> float:
    """Height of a hop (shoulder widths) when a transition changes both legs a
    lot (a star jump): up over the move, down just before its beat."""
    p0, p1, x, _, _ = _segment(frames, beat)
    if p1 is None:
        return 0.0
    a, b = normalize_pose(p0), normalize_pose(p1)
    change = min(max(abs(b[k][0] - a[k][0]), abs(b[k][1] - a[k][1])) for k in ("ll", "rl"))
    if change < HOP_LEGS:
        return 0.0
    k = min(1.0, change / (HOP_LEGS * 1.6))
    u = _clamp((x - 0.18) / 0.62, 0.0, 1.0)  # airborne 18%..80%, landed before the beat
    return HOP * k * math.sin(math.pi * u)


def _hand_x(pose: dict, side: str) -> float:
    """Where a hand is across the body (torso frame, shoulder widths): positive
    = past the midline towards the *other* side."""
    up, fore = pose[f"{side}a"]
    return -(0.5 + UPPER_ARM * math.sin(math.radians(up)) + (FOREARM + HAND) * math.sin(math.radians(fore)))


def _crossing(pose: dict, side: str) -> float:
    return _clamp((_hand_x(pose, side) + 0.1) / 0.6, 0.0, 1.0)


def section_energy(song, beat: float) -> float:
    """How hard the coach grooves: by the song's section at `beat`."""
    name = ""
    if song is not None and hasattr(song, "section_at"):
        try:
            name = str(song.section_at(beat)).lower()
        except Exception:  # noqa: BLE001 - a malformed section list must not stop the show
            name = ""
    for key, e in SECTION_GROOVE:
        if key in name:
            return e
    return 0.8


def _dip(beat: float) -> float:
    """1 at the bottom of the dip (on the beat), 0 on the off-beat."""
    c = 0.5 + 0.5 * math.cos(2.0 * math.pi * beat)
    return c * c


def coach_motion(song, beat: float, energy: float | None = None, frames=None) -> dict:
    """The drawn coach at `beat`: the performed pose plus the groove and
    follow-through channels the painter applies (all plain numbers, so the
    browser port can be pinned by a fixture).

    `frames` overrides the song's keyframes ([(beat, pose)], ascending); with
    neither, the coach does its idle groove. Channels (SW = shoulder widths):
    pose       the pose to draw (limb angles exact on keyframe beats)
    lift       SW the whole body is off the floor (hops)
    squash     torso length factor (< 1 squashed); its width goes the other way
    sway       SW the upper body sits over the feet (+ = screen-right)
    shoulders  [left, right] SW each shoulder rises (+ = up)
    twist      degrees of torso yaw (+ = the chest turns to screen-right)
    head_yaw   degrees the head turns (same sign); head_tilt extra degrees; head_bob SW down
    tail       degrees the tail swings
    hands      [left, right] degrees each glove trails its forearm
    depth      {"l": [upper, hand], "r": [...]}: -1 far .. +1 near the viewer
    """
    if frames is None:
        frames = getattr(song, "_frames", None) or []
    e = section_energy(song, beat) if energy is None else float(energy)
    pose = performed_pose(frames, beat)
    lag = performed_pose(frames, beat - LAG)
    near = performed_pose(frames, beat - LAG_SHORT)
    nearer = performed_pose(frames, beat - 2 * LAG_SHORT)

    dip = _dip(beat)
    sway2 = math.sin(math.pi * beat)  # two-beat sway, zero on every beat
    swing = math.cos(math.pi * beat)  # weight on one foot on each beat
    pose["squat"] = _clamp(pose["squat"] + DIP_SQUAT * e * dip, 0.0, 1.0)
    pose["lean"] += LEAN_SWAY * e * sway2

    off = 0.5 - 0.5 * math.cos(2.0 * math.pi * beat)  # 1 on the off-beat
    squash = 1.0 - SQUASH * e * dip + STRETCH * e * off * off
    # The chest turns towards an arm reaching across it, a little behind the arm.
    twist_pose = TWIST_CROSS * sum(_crossing(q, "l") - _crossing(q, "r") for q in (pose, near, nearer)) / 3.0
    twist = twist_pose + TWIST_GROOVE * e * math.sin(math.pi * (beat - 0.25))
    shoulder = SHOULDER_ROLL * e * math.sin(math.pi * (beat - 0.15))

    # Follow-through: parts that trail the body, from the lagged timeline.
    head_tilt = 0.55 * (lag["lean"] - pose["lean"]) + 0.45 * (lag["head"] - pose["head"])
    head_tilt += HEAD_SWAY * e * math.sin(math.pi * (beat - 0.3))
    tail = TAIL_SWAY * e * math.sin(math.pi * (beat - 0.4)) + 0.8 * (lag["lean"] - pose["lean"])
    tail += 40.0 * (lag["shift"] - pose["shift"])
    hands = [_clamp(HAND_FLICK * (near[f"{s}a"][1] - pose[f"{s}a"][1]), -HAND_MAX, HAND_MAX) for s in "lr"]

    yaw = math.sin(math.radians(twist))
    depth = {}
    for s in "lr":
        up, fore = pose[f"{s}a"]
        turn = yaw if s == "l" else -yaw  # twisting to screen-right brings the screen-left side forward
        across = _crossing(pose, s)
        upper_z = _clamp(-up / 60.0, 0.0, 1.0) + turn
        hand_z = max(across, _clamp(-fore / 70.0, 0.0, 1.0), 0.6 * _clamp((fore - 195.0) / 40.0, 0.0, 1.0)) + 0.5 * turn
        depth[s] = [_clamp(upper_z, -1.0, 1.0), _clamp(hand_z, -1.0, 1.0)]

    return {
        "pose": pose,
        "lift": _hop(frames, beat),
        "squash": squash,
        "sway": SWAY * e * swing,
        "shoulders": [shoulder, -shoulder],
        "twist": twist,
        "head_yaw": 0.7 * twist + 5.0 * e * math.sin(math.pi * (beat - 0.45)),
        "head_tilt": head_tilt,
        "head_bob": HEAD_BOB * e * _dip(beat - 0.1),
        "tail": tail,
        "hands": hands,
        "depth": depth,
    }


def still_motion(pose: dict | None) -> dict:
    """A pose with no motion layered on: what the tutorial and tests draw."""
    return {"pose": normalize_pose(pose), "lift": 0.0, "squash": 1.0, "sway": 0.0, "shoulders": [0.0, 0.0],
            "twist": 0.0, "head_yaw": 0.0, "head_tilt": 0.0, "head_bob": 0.0, "tail": 0.0, "hands": [0.0, 0.0],
            "depth": {"l": [0.0, 0.0], "r": [0.0, 0.0]}}

