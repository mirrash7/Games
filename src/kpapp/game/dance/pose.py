"""Dance poses: angles, forward kinematics, measuring a player, similarity.

See docs/DANCE.md section 3 for the representation. In short: a pose is arm
angles in degrees in the torso frame (0 = straight down, positive = outward
then up, 180 = straight up), in screen space as the player sees it, plus the
lower body as a dancer thinks of it: where each foot is on the floor (`lf`,
`rf`: [out, up]), where the hips are over the feet (`shift`) and how deep the
knees bend (`squat`), plus lean/head. The legs follow from those by two-bone
IK, knees bending outward.

The coach is drawn from `forward(pose)`; the player is read with `measure()`;
`similarity()` compares the two. The lower body is judged on features both
sides compute with the same function (`read_lower`) from joint positions:
stance width, weight over the feet, which foot is lifted and how much each
knee bends. So the coach's `shift` and `squat` mean exactly what the camera
reads off a player (docs/DANCE.md section 3a).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ...config import KP

# --- representation ---------------------------------------------------------

ARMS = ("la", "ra")  # "l"/"r" = screen-left/right
FEET = ("lf", "rf")  # [out, up]: SW outward from the floor's centre line, SW above the floor
LIMBS = ARMS + FEET
LEGACY_LEGS = ("ll", "rl")  # [thigh, shin] angles: still accepted (old and imported charts), turned into feet
SCALARS = ("lean", "head", "shift", "squat")
FOOT_REST = 0.37  # each ankle this far out from the centre line when standing
REST: dict = {"la": [10.0, 5.0], "ra": [10.0, 5.0], "lf": [FOOT_REST, 0.0], "rf": [FOOT_REST, 0.0],
              "lean": 0.0, "head": 0.0, "shift": 0.0, "squat": 0.0}

# Body proportions in shoulder widths (docs/DANCE.md section 3).
TORSO, HIP_W, NECK, HEAD_R = 1.25, 0.6, 0.2, 0.3
UPPER_ARM, FOREARM, HAND = 0.62, 0.58, 0.16
THIGH, SHIN = 0.85, 0.82
LEG = THIGH + SHIN
# The floor: the ankles' height below the rest hip centre (standing legs are straight).
FLOOR = math.sqrt(LEG * LEG - (FOOT_REST - HIP_W / 2) ** 2)
SQUAT_BEND = 90.0  # degrees of knee bend at squat 1 (on the supporting leg)
FOOT_RANGE = ((-1.0, 2.2), (0.0, 1.4))  # authoring range of [out, up]
SHIFT_RANGE = (-1.5, 1.5)
_REACH = LEG * (1.0 - 1e-7)  # a leg never quite locks: the knee keeps a direction

SIDE = {"l": -1.0, "r": 1.0}  # sign of "outward" in screen x


def _legacy_feet(pose: dict) -> dict:
    """[thigh, shin] leg angles (the old format; the importer's before feet)
    -> feet: forward kinematics from the rest hips, the lower foot on the floor."""
    ankles = {}
    for s in "lr":
        thigh, shin = (float(v) for v in pose.get(f"{s}l", (3.0, 2.0)))
        a, b = math.radians(thigh), math.radians(shin)
        ankles[s] = (SIDE[s] * (HIP_W / 2 + THIGH * math.sin(a) + SHIN * math.sin(b)),
                     THIGH * math.cos(a) + SHIN * math.cos(b))
    floor = max(y for _, y in ankles.values())
    return {f"{s}f": [round(SIDE[s] * x, 4), round(floor - y, 4)] for s, (x, y) in ankles.items()}


def normalize_pose(pose: dict | None) -> dict:
    """A complete pose: missing keys from REST, values as floats. Raises
    ValueError on unknown keys or malformed values (charts are validated with this)."""
    out = {k: (list(v) if isinstance(v, list) else v) for k, v in REST.items()}
    items = dict(pose or {})
    legacy = [k for k in LEGACY_LEGS if k in items]
    if legacy:
        if any(k in items for k in FEET):
            raise ValueError("give the legs as feet ('lf'/'rf') or as angles ('ll'/'rl'), not both")
        for k in legacy:
            v = items[k]
            if not (isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v)):
                raise ValueError(f"'{k}' must be [thigh, shin] angles in degrees, got {v!r}")
        items.update(_legacy_feet({k: items.pop(k) for k in legacy}))
    for key, value in items.items():
        if key in LIMBS:
            if not (isinstance(value, (list, tuple)) and len(value) == 2
                    and all(isinstance(x, (int, float)) for x in value)):
                what = "[upper, lower] angles in degrees" if key in ARMS else "[out, up] in shoulder widths"
                raise ValueError(f"'{key}' must be {what}, got {value!r}")
            out[key] = [float(value[0]), float(value[1])]
        elif key in SCALARS:
            if not isinstance(value, (int, float)):
                raise ValueError(f"'{key}' must be a number, got {value!r}")
            out[key] = float(value)
        else:
            raise ValueError(f"unknown pose key '{key}' (expected {', '.join(LIMBS + SCALARS)})")
    if not 0.0 <= out["squat"] <= 1.0:
        raise ValueError(f"'squat' must be 0..1, got {out['squat']}")
    for k in FEET:
        (lo, hi), (ulo, uhi) = FOOT_RANGE
        if not (lo <= out[k][0] <= hi and ulo <= out[k][1] <= uhi):
            raise ValueError(f"'{k}' must be [out {lo:g}..{hi:g}, up {ulo:g}..{uhi:g}], got {out[k]}")
    if not SHIFT_RANGE[0] <= out["shift"] <= SHIFT_RANGE[1]:
        raise ValueError(f"'shift' must be {SHIFT_RANGE[0]:g}..{SHIFT_RANGE[1]:g}, got {out['shift']}")
    return out


def mirror_pose(pose: dict) -> dict:
    """The same move done on the other side of the screen."""
    p = normalize_pose(pose)
    return {"la": list(p["ra"]), "ra": list(p["la"]), "lf": list(p["rf"]), "rf": list(p["lf"]),
            "lean": -p["lean"], "head": -p["head"], "shift": -p["shift"], "squat": p["squat"]}


def smoothstep(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3.0 - 2.0 * u)


def lerp_pose(a: dict, b: dict, u: float) -> dict:
    """Blend two complete poses (u = 0 -> a, 1 -> b). Angles blend linearly, so
    authors keep consecutive keyframes within 180 degrees of each other; feet
    slide across the floor (the drawn coach lifts a stepping foot: motion.py)."""
    out = {}
    for k in LIMBS:
        out[k] = [a[k][0] + (b[k][0] - a[k][0]) * u, a[k][1] + (b[k][1] - a[k][1]) * u]
    for k in SCALARS:
        out[k] = a[k] + (b[k] - a[k]) * u
    return out


# --- geometry ---------------------------------------------------------------

def limb_dir(side: str, angle_deg: float) -> tuple[float, float]:
    """Unit vector (torso frame, y down) for a limb angle on a side."""
    a = math.radians(angle_deg)
    return SIDE[side] * math.sin(a), math.cos(a)


def limb_angle(side: str, vx: float, vy: float) -> float:
    """Inverse of limb_dir: torso-frame vector -> angle in (-180, 180]."""
    return math.degrees(math.atan2(SIDE[side] * vx, vy))


def _rot(v: tuple[float, float], deg: float) -> tuple[float, float]:
    """Torso frame -> screen: rotate by lean (positive tips the head screen-right)."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return v[0] * c - v[1] * s, v[0] * s + v[1] * c


def _add(p, v, k=1.0):
    return p[0] + v[0] * k, p[1] + v[1] * k


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def angle_diff(a: float, b: float) -> float:
    """Absolute difference between two angles, 0..180."""
    d = (a - b) % 360.0
    return 360.0 - d if d > 180.0 else d


def squat_reach(squat: float) -> float:
    """Hip-to-ankle distance of a leg whose knee bends SQUAT_BEND * squat degrees."""
    b = math.radians(SQUAT_BEND * squat)
    return math.sqrt(THIGH * THIGH + SHIN * SHIN + 2.0 * THIGH * SHIN * math.cos(b))


def foot_position(pose: dict, side: str) -> tuple[float, float]:
    """Where an ankle is (SW, y down, origin at the rest hip centre)."""
    out, up = pose[f"{side}f"]
    return SIDE[side] * out, FLOOR - up


def pelvis(pose: dict) -> tuple[float, float]:
    """The hip centre for a complete pose: `shift` across, and as low as the
    squat and the feet need. The supporting leg (planted, nearest under the
    hips) bends SQUAT_BEND * squat degrees; a foot planted wide pulls the hips
    down until the leg reaches it (a wide stance or a lunge sits lower)."""
    px = pose["shift"]
    want = squat_reach(pose["squat"])
    num = den = 0.0
    low_y, low_sq = -math.inf, 0.0
    reach_y = -math.inf
    for s in "lr":
        ax, ay = foot_position(pose, s)
        dx = abs(ax - (px + SIDE[s] * HIP_W / 2))
        reach_y = max(reach_y, ay - math.sqrt(max(0.0, _REACH * _REACH - dx * dx)))
        y_sq = ay - math.sqrt(max(0.0, want * want - dx * dx))
        planted = _clamp(1.0 - pose[f"{s}f"][1] / 0.15, 0.0, 1.0)
        w = planted / ((dx + 0.15) * (dx + 0.15))
        num += w * y_sq
        den += w
        if ay > low_y:
            low_y, low_sq = ay, y_sq
    y = num / den if den > 0.0 else low_sq  # both feet off the floor: the lower one
    return px, max(y, reach_y)


def _knee(hip: tuple[float, float], ankle: tuple[float, float], side: str):
    """Two-bone IK: (knee, ankle) for a leg from `hip` reaching for `ankle`, the
    knee bending outward. An ankle out of reach is pulled in along the leg."""
    dx, dy = ankle[0] - hip[0], ankle[1] - hip[1]
    d = math.hypot(dx, dy)
    if d > _REACH:
        dx, dy = dx * _REACH / d, dy * _REACH / d
        ankle = (hip[0] + dx, hip[1] + dy)
        d = _REACH
    if d < 1e-9:
        return (hip[0] + SIDE[side] * THIGH, hip[1]), ankle
    ux, uy = dx / d, dy / d
    a = math.acos(_clamp((THIGH * THIGH + d * d - SHIN * SHIN) / (2.0 * THIGH * d), -1.0, 1.0))
    t = -SIDE[side] * a  # rotate the hip-to-ankle line outward
    c, s = math.cos(t), math.sin(t)
    return (hip[0] + THIGH * (ux * c - uy * s), hip[1] + THIGH * (ux * s + uy * c)), ankle


def forward(pose: dict) -> dict[str, tuple[float, float]]:
    """Joint positions in shoulder widths, y down, origin at the rest hip centre.

    The feet stand where `lf`/`rf` put them on the floor (FLOOR below the rest
    hips), the hips sit at `pelvis(pose)`, and the knees bend outward to join
    them (two-bone IK). Arms hang from the leaning torso; legs stay in the
    world frame, so the feet stay planted.
    """
    p = normalize_pose(pose)
    lean = p["lean"]
    hip = pelvis(p)
    neck = _add(hip, _rot((0.0, -TORSO), lean))
    j: dict[str, tuple[float, float]] = {"hip": hip, "neck": neck}
    j["head"] = _add(_add(neck, _rot((0.0, -NECK), lean)), _rot((0.0, -HEAD_R), lean + p["head"]))
    for s in "lr":
        sh = _add(neck, _rot((SIDE[s] * 0.5, 0.0), lean))
        up, fore = p[f"{s}a"]
        el = _add(sh, _rot(limb_dir(s, up), lean), UPPER_ARM)
        wr = _add(el, _rot(limb_dir(s, fore), lean), FOREARM)
        j[f"{s}_shoulder"], j[f"{s}_elbow"], j[f"{s}_wrist"] = sh, el, wr
        j[f"{s}_hand"] = _add(wr, _rot(limb_dir(s, fore), lean), HAND)
        hp = (hip[0] + SIDE[s] * HIP_W / 2, hip[1])
        kn, an = _knee(hp, foot_position(p, s), s)
        j[f"{s}_hip"], j[f"{s}_knee"], j[f"{s}_ankle"] = hp, kn, an
    return j


def target_reach(pose: dict, side: str) -> float:
    """The target's shoulder-to-wrist direction (torso frame), for players whose
    elbow isn't visible."""
    p = normalize_pose(pose)
    up, fore = p[f"{side}a"]
    ux, uy = limb_dir(side, up)
    fx, fy = limb_dir(side, fore)
    return limb_angle(side, ux * UPPER_ARM + fx * FOREARM, uy * UPPER_ARM + fy * FOREARM)


# --- the lower body, as the camera sees it ----------------------------------------

@dataclass
class LowerBody:
    """Lower-body features in shoulder widths (scaled by the longer leg, so a
    player's own proportions and a twisted torso don't matter):

    stance   screen-right ankle x minus screen-left ankle x (negative = crossed)
    weight   hip centre x minus the ankles' midpoint (+ = over the screen-right foot)
    lift     screen-left ankle y minus screen-right ankle y (+ = the screen-right foot is higher)
    bend     per screen side, the knee's bend in degrees (0 = straight)
    """
    scale: float  # px (or SW) per shoulder width
    stance: float
    weight: float
    lift: float
    bend: dict[str, float] = field(default_factory=dict)


def _bend(hp, kn, an) -> float:
    ax, ay = kn[0] - hp[0], kn[1] - hp[1]
    bx, by = an[0] - kn[0], an[1] - kn[1]
    return abs(math.degrees(math.atan2(ax * by - ay * bx, ax * bx + ay * by)))


def read_lower(get) -> LowerBody | None:
    """Lower-body features from joints: `get(name)` gives (x, y) or None for
    l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle (screen sides). The same
    function reads the coach (forward()) and the player (measure()). None
    unless both legs are whole in view: one leg alone (a lifted foot with the
    other below the frame) says too little, and is never judged."""
    pts = {}
    longest = 0.0
    for s in "lr":
        hp, kn, an = get(f"{s}_hip"), get(f"{s}_knee"), get(f"{s}_ankle")
        if hp is None or kn is None or an is None:
            return None
        pts[s] = (hp, kn, an)
        length = math.hypot(kn[0] - hp[0], kn[1] - hp[1]) + math.hypot(an[0] - kn[0], an[1] - kn[1])
        longest = max(longest, length)
    if longest <= 1e-9:
        return None
    k = longest / LEG
    (hl, _, al), (hr, _, ar) = pts["l"], pts["r"]
    return LowerBody(scale=k, stance=(ar[0] - al[0]) / k,
                     weight=((hl[0] + hr[0]) / 2 - (al[0] + ar[0]) / 2) / k,
                     lift=(al[1] - ar[1]) / k,
                     bend={s: _bend(*pts[s]) for s in "lr"})


# What makes a target's lower body worth judging: how far it is from standing.
STEP_STANCE, STEP_WEIGHT = 0.30, 0.15  # SW from the rest stance / of weight over a foot: a step or a sway
LIFT_MIN = 0.15  # SW: a foot off the floor (knee lift, kick, heel)
BEND_MIN = 25.0  # degrees of knee bend: a squat, a dip, a bent free leg


@dataclass(frozen=True)
class LegTarget:
    """A target's lower body: its features and which of them are judged."""
    lower: LowerBody
    step: bool  # stance and weight
    lift: bool
    bend: tuple[str, ...]  # screen sides whose knee bend is judged

    @property
    def active(self) -> bool:
        return self.step or self.lift or bool(self.bend)


REST_LOWER = read_lower(forward(REST).get)
_LEG_TARGETS: dict[tuple, LegTarget] = {}


def leg_target(pose: dict) -> LegTarget:
    """The lower-body features of a (complete) target pose and which are judged.
    Cached: the judge asks every frame."""
    key = (tuple(pose["lf"]), tuple(pose["rf"]), pose["shift"], pose["squat"])
    t = _LEG_TARGETS.get(key)
    if t is None:
        lb = read_lower(forward(pose).get)
        step = abs(lb.stance - REST_LOWER.stance) > STEP_STANCE or abs(lb.weight) > STEP_WEIGHT
        bend = tuple(s for s in "lr" if lb.bend[s] > BEND_MIN)
        t = LegTarget(lb, step, abs(lb.lift) > LIFT_MIN, bend)
        if len(_LEG_TARGETS) > 4096:
            _LEG_TARGETS.clear()
        _LEG_TARGETS[key] = t
    return t


# --- measuring a player ---------------------------------------------------------

MIN_CONF = 0.4


def side_keypoints(mirrored: bool = True) -> dict[str, str]:
    """COCO prefix for each screen side. On a mirrored feed the model labels
    limbs as they appear, so the screen-right limbs are left_* (AGENTS.md section 6)."""
    return {"r": "left", "l": "right"} if mirrored else {"r": "right", "l": "left"}


@dataclass
class ArmReading:
    upper: float | None = None  # degrees, torso frame
    fore: float | None = None
    reach: float | None = None  # shoulder -> wrist


@dataclass
class LegReading:
    thigh: float | None = None
    shin: float | None = None


@dataclass
class Measured:
    lean: float  # degrees
    has_hips: bool
    shoulder_px: float  # shoulder width in pixels
    centre_px: tuple[float, float]  # shoulder centre in pixels
    arms: dict[str, ArmReading] = field(default_factory=dict)  # "l"/"r" (screen side)
    legs: dict[str, LegReading] = field(default_factory=dict)  # thigh/shin angles (the importer, the PiP)
    lower: LowerBody | None = None  # what the judge reads of the legs (read_lower)

    @property
    def sees_arms(self) -> bool:
        """Can any arm be judged: a wrist, or at least an elbow (similarity)."""
        return any(a.reach is not None or a.upper is not None for a in self.arms.values())

    @property
    def sees_feet(self) -> bool:
        """Are both legs, feet included, in view: the lower body can be judged."""
        return self.lower is not None


def measure(kp_pose, mirrored: bool = True) -> Measured | None:
    """A player's keypoints -> torso-frame angles. None without both shoulders."""
    if kp_pose is None:
        return None
    sides = side_keypoints(mirrored)

    def pt(name):
        p = kp_pose.point(KP[name], MIN_CONF)
        return None if p is None else (float(p[0]), float(p[1]))

    sl, sr = pt(f"{sides['l']}_shoulder"), pt(f"{sides['r']}_shoulder")
    if sl is None or sr is None:
        return None
    sw = math.hypot(sr[0] - sl[0], sr[1] - sl[1])
    if sw < 8.0:
        return None
    centre = ((sl[0] + sr[0]) / 2, (sl[1] + sr[1]) / 2)
    hl, hr = pt(f"{sides['l']}_hip"), pt(f"{sides['r']}_hip")
    has_hips = hl is not None and hr is not None
    if has_hips:
        hip_c = ((hl[0] + hr[0]) / 2, (hl[1] + hr[1]) / 2)
        upx, upy = centre[0] - hip_c[0], centre[1] - hip_c[1]
    else:
        # Perpendicular to the shoulder line, pointing up.
        sx, sy = sr[0] - sl[0], sr[1] - sl[1]
        upx, upy = sy, -sx
    lean = math.degrees(math.atan2(upx, -upy))

    def torso(v):
        return _rot(v, -lean)

    m = Measured(lean=lean, has_hips=has_hips, shoulder_px=sw, centre_px=centre)
    joints: dict[str, tuple[float, float] | None] = {}
    for s, sh in (("l", sl), ("r", sr)):
        el, wr = pt(f"{sides[s]}_elbow"), pt(f"{sides[s]}_wrist")
        reading = ArmReading()
        if wr is not None:
            reading.reach = limb_angle(s, *torso((wr[0] - sh[0], wr[1] - sh[1])))
            if el is not None:
                reading.upper = limb_angle(s, *torso((el[0] - sh[0], el[1] - sh[1])))
                reading.fore = limb_angle(s, *torso((wr[0] - el[0], wr[1] - el[1])))
        elif el is not None:
            reading.upper = limb_angle(s, *torso((el[0] - sh[0], el[1] - sh[1])))
        m.arms[s] = reading
        hp, kn, an = pt(f"{sides[s]}_hip"), pt(f"{sides[s]}_knee"), pt(f"{sides[s]}_ankle")
        joints[f"{s}_hip"], joints[f"{s}_knee"], joints[f"{s}_ankle"] = hp, kn, an
        if hp is not None and kn is not None and an is not None:
            # Legs are measured in the world frame, as forward() draws them.
            m.legs[s] = LegReading(limb_angle(s, kn[0] - hp[0], kn[1] - hp[1]),
                                   limb_angle(s, an[0] - kn[0], an[1] - kn[1]))
    m.lower = read_lower(joints.get)
    return m


# --- similarity -------------------------------------------------------------

PERFECT_WITHIN = 20.0  # degrees: no penalty up to here
ZERO_AT = 90.0  # degrees: no credit from here
ARM_WEIGHT, LEAN_WEIGHT = 1.0, 0.3
# A target whose lower body is active (leg_target) weighs its legs as much as
# both arms together: a leg-led move is half the score (docs/DANCE.md section 3a).
LEGS_WEIGHT = 2.0
LEAN_MIN = 8.0  # a target lean smaller than this isn't judged
# Lower-body tolerances: (free, span): full credit up to `free`, none from free + span.
STANCE_TOL = (0.20, 0.45)  # SW
WEIGHT_TOL = (0.10, 0.25)  # SW
LIFT_TOL = (0.15, 0.30)  # SW
BEND_TOL = (20.0, 35.0)  # degrees
BEND_SHARE = 0.5  # each knee's bend counts half as much as stance, weight or lift
# Legs the camera can't see aren't judged (None): the arms carry the point.
# Feet are optional, because a crowd or a close camera often hides them. A
# number here would give unseen legs that much credit instead (0.5 made hiding
# the feet never beat dancing in full view, but asked every player for feet).
UNSEEN_LEGS = None


def segment_score(target: float, seen: float) -> float:
    d = angle_diff(target, seen)
    return min(1.0, max(0.0, 1.0 - (d - PERFECT_WITHIN) / (ZERO_AT - PERFECT_WITHIN)))


def _tol(d: float, tol: tuple[float, float]) -> float:
    return min(1.0, max(0.0, 1.0 - (abs(d) - tol[0]) / tol[1]))


def legs_score(target: LegTarget, seen: LowerBody | None) -> float | None:
    """0..1 match of a player's lower body on the features the target asks for,
    or None if the target's legs aren't judged or the player's can't be seen."""
    if seen is None or not target.active:
        return None
    t = target.lower
    num = den = 0.0
    if target.step:
        num += _tol(t.stance - seen.stance, STANCE_TOL) + _tol(t.weight - seen.weight, WEIGHT_TOL)
        den += 2.0
    if target.lift:
        num += _tol(t.lift - seen.lift, LIFT_TOL)
        den += 1.0
    for s in target.bend:
        num += BEND_SHARE * _tol(t.bend[s] - seen.bend[s], BEND_TOL)
        den += BEND_SHARE
    return num / den


def similarity(target: dict, m: Measured | None,
               unseen_legs: float | None = UNSEEN_LEGS) -> tuple[float | None, dict[str, float]]:
    """0..1 match between a target pose and a measured player, plus the
    per-part scores. None when no arm is visible: "can't see you", not a miss.
    Legs out of view on a leg move earn `unseen_legs`: None (UNSEEN_LEGS, in play
    too) leaves them unjudged, so the arms carry the point; feet are optional."""
    if m is None:
        return None, {}
    t = normalize_pose(target)
    parts: dict[str, float] = {}
    weights: dict[str, float] = {}
    for s in "lr":
        a = m.arms.get(s)
        if a is None:
            continue
        tu, tf = t[f"{s}a"]
        if a.upper is not None and a.fore is not None:
            parts[f"{s}a"] = 0.5 * (segment_score(tu, a.upper) + segment_score(tf, a.fore))
        elif a.reach is not None:
            parts[f"{s}a"] = segment_score(target_reach(t, s), a.reach)
        elif a.upper is not None:
            # Elbow seen, wrist not: a player standing close raises their hands
            # out of the top of the frame on every arms-up move. The upper arm
            # still says where the arm points; ignoring it left the window with
            # only the arm on its way up, an X for a perfect Clap Up.
            parts[f"{s}a"] = segment_score(tu, a.upper)
        else:
            continue
        weights[f"{s}a"] = ARM_WEIGHT
    if not parts:
        return None, {}
    lt = leg_target(t)
    legs = legs_score(lt, m.lower)
    if legs is None and lt.active:
        legs = unseen_legs
    if legs is not None:
        parts["legs"] = legs
        weights["legs"] = LEGS_WEIGHT
    if m.has_hips and abs(t["lean"]) > LEAN_MIN:
        parts["lean"] = min(1.0, max(0.0, 1.0 - max(0.0, abs(t["lean"] - m.lean) - 6.0) / 24.0))
        weights["lean"] = LEAN_WEIGHT
    total = acc = 0.0
    for k in weights:
        total += weights[k]
    for k in parts:
        acc += parts[k] * weights[k]
    return acc / total, parts


# --- synthetic players (tests, QA, demo mode) ---------------------------------------

# A player about 2.5-3 m back: the whole body in a 1280 x 720 frame, feet included.
FULL_BODY = {"centre_px": (640.0, 250.0), "shoulder_px": 110.0}


def keypoints_for(pose: dict, centre_px=(640.0, 330.0), shoulder_px: float = 150.0,
                  mirrored: bool = True, visible: set[str] | None = None, frame_h: int = 720):
    """A COCO `Pose` that performs `pose`: what a perfect player looks like to
    the model. Keypoints outside the frame (or not in `visible`) get confidence 0.
    The default framing is a close player whose ankles are below the frame
    (FULL_BODY shows the feet)."""
    import numpy as np

    from ...inference import Pose

    j = forward(pose)
    sides = side_keypoints(mirrored)
    origin = (centre_px[0] - (j["l_shoulder"][0] + j["r_shoulder"][0]) / 2 * shoulder_px,
              centre_px[1] - (j["l_shoulder"][1] + j["r_shoulder"][1]) / 2 * shoulder_px)

    def px(p):
        return origin[0] + p[0] * shoulder_px, origin[1] + p[1] * shoulder_px

    xy = np.zeros((17, 2), np.float32)
    conf = np.zeros(17, np.float32)
    head = px(j["head"])
    for name, at in (("nose", head), ("left_eye", head), ("right_eye", head)):
        xy[KP[name]] = at
        conf[KP[name]] = 0.9
    for s in "lr":
        for part in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle"):
            name = f"{sides[s]}_{part}"
            if visible is not None and part not in visible:
                continue
            x, y = px(j[f"{s}_{part}"])
            xy[KP[name]] = (x, y)
            conf[KP[name]] = 0.9 if -5 <= y <= frame_h + 5 else 0.0
    return Pose(xy=xy, confidence=conf, score=0.9)
