"""Dance poses: angles, forward kinematics, measuring a player, similarity.

See docs/DANCE.md section 3 for the representation. In short: a pose is limb angles in
degrees in the torso frame (0 = straight down, positive = outward then up,
180 = straight up), in screen space as the player sees it, plus lean/head/
shift/squat. The coach is drawn from `forward(pose)`; the player is read into
the same angles with `measure()`; `similarity()` compares the two.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ...config import KP

# --- representation ---------------------------------------------------------

LIMBS = ("la", "ra", "ll", "rl")  # arms then legs; "l"/"r" = screen-left/right
SCALARS = ("lean", "head", "shift", "squat")
REST: dict = {"la": [10.0, 5.0], "ra": [10.0, 5.0], "ll": [3.0, 2.0], "rl": [3.0, 2.0],
              "lean": 0.0, "head": 0.0, "shift": 0.0, "squat": 0.0}

# Body proportions in shoulder widths (docs/DANCE.md section 3).
TORSO, HIP_W, NECK, HEAD_R = 1.25, 0.6, 0.2, 0.3
UPPER_ARM, FOREARM, HAND = 0.62, 0.58, 0.16
THIGH, SHIN = 0.85, 0.82

SIDE = {"l": -1.0, "r": 1.0}  # sign of "outward" in screen x


def normalize_pose(pose: dict | None) -> dict:
    """A complete pose: missing keys from REST, values as floats. Raises
    ValueError on unknown keys or malformed values (charts are validated with this)."""
    out = {k: (list(v) if isinstance(v, list) else v) for k, v in REST.items()}
    for key, value in (pose or {}).items():
        if key in LIMBS:
            if not (isinstance(value, (list, tuple)) and len(value) == 2
                    and all(isinstance(x, (int, float)) for x in value)):
                raise ValueError(f"'{key}' must be [upper, lower] angles in degrees, got {value!r}")
            out[key] = [float(value[0]), float(value[1])]
        elif key in SCALARS:
            if not isinstance(value, (int, float)):
                raise ValueError(f"'{key}' must be a number, got {value!r}")
            out[key] = float(value)
        else:
            raise ValueError(f"unknown pose key '{key}' (expected {', '.join(LIMBS + SCALARS)})")
    if not 0.0 <= out["squat"] <= 1.0:
        raise ValueError(f"'squat' must be 0..1, got {out['squat']}")
    return out


def mirror_pose(pose: dict) -> dict:
    """The same move done on the other side of the screen."""
    p = normalize_pose(pose)
    return {"la": list(p["ra"]), "ra": list(p["la"]), "ll": list(p["rl"]), "rl": list(p["ll"]),
            "lean": -p["lean"], "head": -p["head"], "shift": -p["shift"], "squat": p["squat"]}


def smoothstep(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3.0 - 2.0 * u)


def lerp_pose(a: dict, b: dict, u: float) -> dict:
    """Blend two complete poses (u = 0 -> a, 1 -> b). Angles blend linearly, so
    authors keep consecutive keyframes within 180 degrees of each other."""
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


def angle_diff(a: float, b: float) -> float:
    """Absolute difference between two angles, 0..180."""
    d = (a - b) % 360.0
    return 360.0 - d if d > 180.0 else d


def forward(pose: dict) -> dict[str, tuple[float, float]]:
    """Joint positions in shoulder widths, y down, origin at the rest hip centre.

    `shift` moves the body sideways; `squat` lowers the hips and bends the
    knees outward (cosmetic). Arms hang from the leaning torso; legs stay in
    the world frame so the feet stay planted.
    """
    p = normalize_pose(pose)
    lean = p["lean"]
    hip = (p["shift"], p["squat"] * 0.42)
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
        thigh, shin = p[f"{s}l"]
        # Squatting bends the knees outward and the shins back in.
        thigh += 38.0 * p["squat"]
        shin -= 12.0 * p["squat"]
        hp = _add(hip, (SIDE[s] * HIP_W / 2, 0.0))
        kn = _add(hp, limb_dir(s, thigh), THIGH)
        j[f"{s}_hip"], j[f"{s}_knee"] = hp, kn
        j[f"{s}_ankle"] = _add(kn, limb_dir(s, shin), SHIN)
    return j


def target_reach(pose: dict, side: str) -> float:
    """The target's shoulder-to-wrist direction (torso frame), for players whose
    elbow isn't visible."""
    p = normalize_pose(pose)
    up, fore = p[f"{side}a"]
    ux, uy = limb_dir(side, up)
    fx, fy = limb_dir(side, fore)
    return limb_angle(side, ux * UPPER_ARM + fx * FOREARM, uy * UPPER_ARM + fy * FOREARM)


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
    legs: dict[str, LegReading] = field(default_factory=dict)

    @property
    def sees_arms(self) -> bool:
        """Can any arm be judged: a wrist, or at least an elbow (similarity)."""
        return any(a.reach is not None or a.upper is not None for a in self.arms.values())


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
        if hp is not None and kn is not None and an is not None:
            # Legs are measured in the world frame, as forward() draws them.
            m.legs[s] = LegReading(limb_angle(s, kn[0] - hp[0], kn[1] - hp[1]),
                                   limb_angle(s, an[0] - kn[0], an[1] - kn[1]))
    return m


# --- similarity -------------------------------------------------------------

PERFECT_WITHIN = 20.0  # degrees: no penalty up to here
ZERO_AT = 90.0  # degrees: no credit from here
ARM_WEIGHT, LEG_WEIGHT, LEAN_WEIGHT = 1.0, 0.35, 0.3
LEG_REST_MARGIN = 25.0  # a target leg this close to rest isn't judged
LEAN_MIN = 8.0  # a target lean smaller than this isn't judged


def segment_score(target: float, seen: float) -> float:
    d = angle_diff(target, seen)
    return min(1.0, max(0.0, 1.0 - (d - PERFECT_WITHIN) / (ZERO_AT - PERFECT_WITHIN)))


def similarity(target: dict, m: Measured | None) -> tuple[float | None, dict[str, float]]:
    """0..1 match between a target pose and a measured player, plus the
    per-part scores. None when no arm is visible: "can't see you", not a miss."""
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
    for s in "lr":
        leg = m.legs.get(s)
        tt, ts = t[f"{s}l"]
        rt, rs = REST[f"{s}l"]
        if leg is None or (abs(tt - rt) <= LEG_REST_MARGIN and abs(ts - rs) <= LEG_REST_MARGIN):
            continue
        parts[f"{s}l"] = 0.5 * (segment_score(tt, leg.thigh) + segment_score(ts, leg.shin))
        weights[f"{s}l"] = LEG_WEIGHT
    if m.has_hips and abs(t["lean"]) > LEAN_MIN:
        parts["lean"] = min(1.0, max(0.0, 1.0 - max(0.0, abs(t["lean"] - m.lean) - 6.0) / 24.0))
        weights["lean"] = LEAN_WEIGHT
    total = sum(weights.values())
    return sum(parts[k] * weights[k] for k in parts) / total, parts


# --- synthetic players (tests, QA, demo mode) ---------------------------------------

def keypoints_for(pose: dict, centre_px=(640.0, 330.0), shoulder_px: float = 150.0,
                  mirrored: bool = True, visible: set[str] | None = None, frame_h: int = 720):
    """A COCO `Pose` that performs `pose`: what a perfect player looks like to
    the model. Keypoints outside the frame (or not in `visible`) get confidence 0."""
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
