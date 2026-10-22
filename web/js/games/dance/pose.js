// Dance poses: angles, forward kinematics, measuring a player, similarity
// (port of src/kpapp/game/dance/pose.py; docs/DANCE.md section 3).
//
// A pose is arm angles in degrees in the torso frame (0 = straight down,
// positive = outward then up, 180 = straight up), in screen space as the
// player sees it, plus the lower body as a dancer thinks of it: where each
// foot is on the floor (lf, rf: [out, up]), where the hips are over the feet
// (shift) and how deep the knees bend (squat), plus lean/head. The legs follow
// by two-bone IK, knees bending outward. The coach is drawn from
// forward(pose); the player is read with measure(); similarity() compares the
// two, the lower body on features both sides compute with readLower().
//
// No DOM: this module runs under Node for the tests.

import { KP, Pose } from "../../core/pose.js";

const RAD = Math.PI / 180;

// --- representation ---------------------------------------------------------

export const ARMS = ["la", "ra"]; // "l"/"r" = screen-left/right
export const FEET = ["lf", "rf"]; // [out, up]: SW outward from the floor's centre line, SW above the floor
export const LIMBS = [...ARMS, ...FEET];
export const LEGACY_LEGS = ["ll", "rl"]; // [thigh, shin] angles: still accepted (old and imported charts), turned into feet
export const SCALARS = ["lean", "head", "shift", "squat"];
export const FOOT_REST = 0.37; // each ankle this far out from the centre line when standing
export const REST = Object.freeze({
  la: Object.freeze([10.0, 5.0]), ra: Object.freeze([10.0, 5.0]),
  lf: Object.freeze([FOOT_REST, 0.0]), rf: Object.freeze([FOOT_REST, 0.0]),
  lean: 0.0, head: 0.0, shift: 0.0, squat: 0.0,
});

// Body proportions in shoulder widths (docs/DANCE.md section 3).
export const TORSO = 1.25, HIP_W = 0.6, NECK = 0.2, HEAD_R = 0.3;
export const UPPER_ARM = 0.62, FOREARM = 0.58, HAND = 0.16;
export const THIGH = 0.85, SHIN = 0.82;
export const LEG = THIGH + SHIN;
// The floor: the ankles' height below the rest hip centre (standing legs are straight).
export const FLOOR = Math.sqrt(LEG * LEG - (FOOT_REST - HIP_W / 2) ** 2);
export const SQUAT_BEND = 90.0; // degrees of knee bend at squat 1 (on the supporting leg)
export const FOOT_RANGE = Object.freeze([Object.freeze([-1.0, 2.2]), Object.freeze([0.0, 1.4])]); // [out, up]
export const SHIFT_RANGE = Object.freeze([-1.5, 1.5]);
const REACH = LEG * (1.0 - 1e-7); // a leg never quite locks: the knee keeps a direction

export const SIDE = Object.freeze({ l: -1.0, r: 1.0 }); // sign of "outward" in screen x

/** Python's repr() of a value, for error messages that read like the reference's. */
export function pyRepr(v) {
  if (v === null || v === undefined) return "None";
  if (v === true) return "True";
  if (v === false) return "False";
  if (typeof v === "number") return Number.isFinite(v) ? String(v) : pyNum(v); // JSON ints stay ints
  if (typeof v === "string") return `'${v}'`;
  if (Array.isArray(v)) return `[${v.map(pyRepr).join(", ")}]`;
  if (typeof v === "object") return `{${Object.entries(v).map(([k, x]) => `'${k}': ${pyRepr(x)}`).join(", ")}}`;
  return String(v);
}

/** A float formatted as Python prints it (2 -> "2.0"). */
export function pyNum(x) {
  if (!Number.isFinite(x)) return x > 0 ? "inf" : x < 0 ? "-inf" : "nan";
  return Number.isInteger(x) ? `${x}.0` : String(x);
}

const isNum = (v) => typeof v === "number" || typeof v === "boolean";
const isPair = (v) => Array.isArray(v) && v.length === 2 && v.every(isNum);
const round4 = (x) => Math.round(x * 1e4) / 1e4;
const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

/**
 * [thigh, shin] leg angles (the old format) -> feet: forward kinematics from
 * the rest hips, the lower foot on the floor.
 */
function legacyFeet(legs) {
  const ankles = {};
  for (const s of ["l", "r"]) {
    const v = legs[`${s}l`] ?? [3.0, 2.0];
    const a = Number(v[0]) * RAD, b = Number(v[1]) * RAD;
    ankles[s] = [SIDE[s] * (HIP_W / 2 + THIGH * Math.sin(a) + SHIN * Math.sin(b)),
      THIGH * Math.cos(a) + SHIN * Math.cos(b)];
  }
  const floor = Math.max(ankles.l[1], ankles.r[1]);
  const out = {};
  for (const s of ["l", "r"]) out[`${s}f`] = [round4(SIDE[s] * ankles[s][0]), round4(floor - ankles[s][1])];
  return out;
}

/**
 * A complete pose: missing keys from REST, values as numbers. Throws on
 * unknown keys or malformed values (charts are validated with this).
 */
export function normalizePose(pose) {
  const out = {
    la: [...REST.la], ra: [...REST.ra], lf: [...REST.lf], rf: [...REST.rf],
    lean: REST.lean, head: REST.head, shift: REST.shift, squat: REST.squat,
  };
  let items = Object.entries(pose ?? {});
  const legacy = items.filter(([k]) => LEGACY_LEGS.includes(k));
  if (legacy.length) {
    if (items.some(([k]) => FEET.includes(k))) {
      throw new Error("give the legs as feet ('lf'/'rf') or as angles ('ll'/'rl'), not both");
    }
    for (const [k, v] of legacy) {
      if (!isPair(v)) throw new Error(`'${k}' must be [thigh, shin] angles in degrees, got ${pyRepr(v)}`);
    }
    items = [...items.filter(([k]) => !LEGACY_LEGS.includes(k)), ...Object.entries(legacyFeet(Object.fromEntries(legacy)))];
  }
  for (const [key, value] of items) {
    if (LIMBS.includes(key)) {
      if (!isPair(value)) {
        const what = ARMS.includes(key) ? "[upper, lower] angles in degrees" : "[out, up] in shoulder widths";
        throw new Error(`'${key}' must be ${what}, got ${pyRepr(value)}`);
      }
      out[key] = [Number(value[0]), Number(value[1])];
    } else if (SCALARS.includes(key)) {
      if (!isNum(value)) throw new Error(`'${key}' must be a number, got ${pyRepr(value)}`);
      out[key] = Number(value);
    } else {
      throw new Error(`unknown pose key '${key}' (expected ${[...LIMBS, ...SCALARS].join(", ")})`);
    }
  }
  if (!(out.squat >= 0.0 && out.squat <= 1.0)) throw new Error(`'squat' must be 0..1, got ${pyNum(out.squat)}`);
  const [[lo, hi], [ulo, uhi]] = FOOT_RANGE;
  for (const k of FEET) {
    if (!(out[k][0] >= lo && out[k][0] <= hi && out[k][1] >= ulo && out[k][1] <= uhi)) {
      throw new Error(`'${k}' must be [out ${lo}..${hi}, up ${ulo}..${uhi}], got [${pyNum(out[k][0])}, ${pyNum(out[k][1])}]`);
    }
  }
  if (!(out.shift >= SHIFT_RANGE[0] && out.shift <= SHIFT_RANGE[1])) {
    throw new Error(`'shift' must be ${SHIFT_RANGE[0]}..${SHIFT_RANGE[1]}, got ${pyNum(out.shift)}`);
  }
  return out;
}

/** The same move done on the other side of the screen. */
export function mirrorPose(pose) {
  const p = normalizePose(pose);
  return {
    la: [...p.ra], ra: [...p.la], lf: [...p.rf], rf: [...p.lf],
    lean: -p.lean, head: -p.head, shift: -p.shift, squat: p.squat,
  };
}

export function smoothstep(u) {
  u = Math.min(1.0, Math.max(0.0, u));
  return u * u * (3.0 - 2.0 * u);
}

/**
 * Blend two complete poses (u = 0 -> a, 1 -> b). Angles blend linearly, so
 * authors keep consecutive keyframes within 180 degrees of each other; feet
 * slide across the floor (the drawn coach lifts a stepping foot: motion.js).
 */
export function lerpPose(a, b, u) {
  const out = {};
  for (const k of LIMBS) {
    out[k] = [a[k][0] + (b[k][0] - a[k][0]) * u, a[k][1] + (b[k][1] - a[k][1]) * u];
  }
  for (const k of SCALARS) out[k] = a[k] + (b[k] - a[k]) * u;
  return out;
}

// --- geometry ---------------------------------------------------------------

/** Unit vector (torso frame, y down) for a limb angle on a side. */
export function limbDir(side, angleDeg) {
  const a = angleDeg * RAD;
  return [SIDE[side] * Math.sin(a), Math.cos(a)];
}

/** Inverse of limbDir: torso-frame vector -> angle in (-180, 180]. */
export function limbAngle(side, vx, vy) {
  return Math.atan2(SIDE[side] * vx, vy) / RAD;
}

/** Torso frame -> screen: rotate by lean (positive tips the head screen-right). */
function rot(v, deg) {
  const a = deg * RAD;
  const c = Math.cos(a), s = Math.sin(a);
  return [v[0] * c - v[1] * s, v[0] * s + v[1] * c];
}

const add = (p, v, k = 1.0) => [p[0] + v[0] * k, p[1] + v[1] * k];

/** Python's float modulo (the result takes the divisor's sign). */
const pyMod = (a, b) => ((a % b) + b) % b;

/** Absolute difference between two angles, 0..180. */
export function angleDiff(a, b) {
  const d = pyMod(a - b, 360.0);
  return d > 180.0 ? 360.0 - d : d;
}

/** Hip-to-ankle distance of a leg whose knee bends SQUAT_BEND * squat degrees. */
export function squatReach(squat) {
  const b = SQUAT_BEND * squat * RAD;
  return Math.sqrt(THIGH * THIGH + SHIN * SHIN + 2.0 * THIGH * SHIN * Math.cos(b));
}

/** Where an ankle is (SW, y down, origin at the rest hip centre). */
export function footPosition(pose, side) {
  const [out, up] = pose[`${side}f`];
  return [SIDE[side] * out, FLOOR - up];
}

/**
 * The hip centre for a complete pose: `shift` across, and as low as the squat
 * and the feet need. The supporting leg (planted, nearest under the hips)
 * bends SQUAT_BEND * squat degrees; a foot planted wide pulls the hips down
 * until the leg reaches it (a wide stance or a lunge sits lower).
 */
export function pelvis(pose) {
  const px = pose.shift;
  const want = squatReach(pose.squat);
  let num = 0.0, den = 0.0;
  let lowY = -Infinity, lowSq = 0.0;
  let reachY = -Infinity;
  for (const s of ["l", "r"]) {
    const [ax, ay] = footPosition(pose, s);
    const dx = Math.abs(ax - (px + SIDE[s] * HIP_W / 2));
    reachY = Math.max(reachY, ay - Math.sqrt(Math.max(0.0, REACH * REACH - dx * dx)));
    const ySq = ay - Math.sqrt(Math.max(0.0, want * want - dx * dx));
    const planted = clamp(1.0 - pose[`${s}f`][1] / 0.15, 0.0, 1.0);
    const w = planted / ((dx + 0.15) * (dx + 0.15));
    num += w * ySq;
    den += w;
    if (ay > lowY) {
      lowY = ay;
      lowSq = ySq;
    }
  }
  const y = den > 0.0 ? num / den : lowSq; // both feet off the floor: the lower one
  return [px, Math.max(y, reachY)];
}

/**
 * Two-bone IK: [knee, ankle] for a leg from `hip` reaching for `ankle`, the
 * knee bending outward. An ankle out of reach is pulled in along the leg.
 */
function kneeIK(hip, ankle, side) {
  let dx = ankle[0] - hip[0], dy = ankle[1] - hip[1];
  let d = Math.hypot(dx, dy);
  if (d > REACH) {
    dx = dx * REACH / d;
    dy = dy * REACH / d;
    ankle = [hip[0] + dx, hip[1] + dy];
    d = REACH;
  }
  if (d < 1e-9) return [[hip[0] + SIDE[side] * THIGH, hip[1]], ankle];
  const ux = dx / d, uy = dy / d;
  const a = Math.acos(clamp((THIGH * THIGH + d * d - SHIN * SHIN) / (2.0 * THIGH * d), -1.0, 1.0));
  const t = -SIDE[side] * a; // rotate the hip-to-ankle line outward
  const c = Math.cos(t), s = Math.sin(t);
  return [[hip[0] + THIGH * (ux * c - uy * s), hip[1] + THIGH * (ux * s + uy * c)], ankle];
}

/**
 * Joint positions in shoulder widths, y down, origin at the rest hip centre.
 *
 * The feet stand where lf/rf put them on the floor (FLOOR below the rest
 * hips), the hips sit at pelvis(pose), and the knees bend outward to join
 * them (two-bone IK). Arms hang from the leaning torso; legs stay in the world
 * frame, so the feet stay planted.
 */
export function forward(pose) {
  const p = normalizePose(pose);
  const lean = p.lean;
  const hip = pelvis(p);
  const neck = add(hip, rot([0.0, -TORSO], lean));
  const j = { hip, neck };
  j.head = add(add(neck, rot([0.0, -NECK], lean)), rot([0.0, -HEAD_R], lean + p.head));
  for (const s of ["l", "r"]) {
    const sh = add(neck, rot([SIDE[s] * 0.5, 0.0], lean));
    const [up, fore] = p[`${s}a`];
    const el = add(sh, rot(limbDir(s, up), lean), UPPER_ARM);
    const wr = add(el, rot(limbDir(s, fore), lean), FOREARM);
    j[`${s}_shoulder`] = sh;
    j[`${s}_elbow`] = el;
    j[`${s}_wrist`] = wr;
    j[`${s}_hand`] = add(wr, rot(limbDir(s, fore), lean), HAND);
    const hp = [hip[0] + SIDE[s] * HIP_W / 2, hip[1]];
    const [kn, an] = kneeIK(hp, footPosition(p, s), s);
    j[`${s}_hip`] = hp;
    j[`${s}_knee`] = kn;
    j[`${s}_ankle`] = an;
  }
  return j;
}

/**
 * The target's shoulder-to-wrist direction (torso frame), for players whose
 * elbow isn't visible.
 */
export function targetReach(pose, side) {
  const p = normalizePose(pose);
  const [up, fore] = p[`${side}a`];
  const [ux, uy] = limbDir(side, up);
  const [fx, fy] = limbDir(side, fore);
  return limbAngle(side, ux * UPPER_ARM + fx * FOREARM, uy * UPPER_ARM + fy * FOREARM);
}

// --- the lower body, as the camera sees it ------------------------------------

function bendOf(hp, kn, an) {
  const ax = kn[0] - hp[0], ay = kn[1] - hp[1];
  const bx = an[0] - kn[0], by = an[1] - kn[1];
  return Math.abs(Math.atan2(ax * by - ay * bx, ax * bx + ay * by) / RAD);
}

/**
 * Lower-body features from joints (pose.read_lower): `get(name)` gives [x, y]
 * or null for l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle (screen sides).
 * The same function reads the coach (forward()) and the player (measure()).
 * In shoulder widths, scaled by the longer leg: {scale, stance (screen-right
 * ankle x minus screen-left), weight (hips over the ankles' midpoint), lift
 * (+ = the screen-right foot is higher), bend: {l, r} (degrees)}. null unless
 * both legs are whole in view.
 */
export function readLower(get) {
  const pts = {};
  let longest = 0.0;
  for (const s of ["l", "r"]) {
    const hp = get(`${s}_hip`), kn = get(`${s}_knee`), an = get(`${s}_ankle`);
    if (hp == null || kn == null || an == null) return null;
    pts[s] = [hp, kn, an];
    const length = Math.hypot(kn[0] - hp[0], kn[1] - hp[1]) + Math.hypot(an[0] - kn[0], an[1] - kn[1]);
    longest = Math.max(longest, length);
  }
  if (longest <= 1e-9) return null;
  const k = longest / LEG;
  const [hl, , al] = pts.l, [hr, , ar] = pts.r;
  return {
    scale: k,
    stance: (ar[0] - al[0]) / k,
    weight: ((hl[0] + hr[0]) / 2 - (al[0] + ar[0]) / 2) / k,
    lift: (al[1] - ar[1]) / k,
    bend: { l: bendOf(...pts.l), r: bendOf(...pts.r) },
  };
}

// What makes a target's lower body worth judging: how far it is from standing.
export const STEP_STANCE = 0.30, STEP_WEIGHT = 0.15; // SW from the rest stance / of weight over a foot
export const LIFT_MIN = 0.15; // SW: a foot off the floor (knee lift, kick, heel)
export const BEND_MIN = 25.0; // degrees of knee bend: a squat, a dip, a bent free leg

const getter = (j) => (name) => j[name] ?? null;
export const REST_LOWER = readLower(getter(forward(REST)));
const LEG_TARGETS = new Map();

/**
 * The lower-body features of a (complete) target pose and which are judged:
 * {lower, step (stance and weight), lift, bend: [screen sides], active}.
 * Cached: the judge asks every frame.
 */
export function legTarget(pose) {
  const key = `${pose.lf[0]},${pose.lf[1]},${pose.rf[0]},${pose.rf[1]},${pose.shift},${pose.squat}`;
  let t = LEG_TARGETS.get(key);
  if (t == null) {
    const lb = readLower(getter(forward(pose)));
    const step = Math.abs(lb.stance - REST_LOWER.stance) > STEP_STANCE || Math.abs(lb.weight) > STEP_WEIGHT;
    const bend = Object.freeze(["l", "r"].filter((s) => lb.bend[s] > BEND_MIN));
    const lift = Math.abs(lb.lift) > LIFT_MIN;
    t = Object.freeze({ lower: lb, step, lift, bend, active: step || lift || bend.length > 0 });
    if (LEG_TARGETS.size > 4096) LEG_TARGETS.clear();
    LEG_TARGETS.set(key, t);
  }
  return t;
}

// --- measuring a player -----------------------------------------------------

export const MIN_CONF = 0.4;

/**
 * COCO prefix for each screen side. On a mirrored feed the model labels
 * limbs as they appear, so the screen-right limbs are left_* (AGENTS.md section 6).
 */
export function sideKeypoints(mirrored = true) {
  return mirrored ? { r: "left", l: "right" } : { r: "right", l: "left" };
}

/**
 * A player's reading: lean (degrees), hasHips, shoulderPx, centrePx, and per
 * screen side `arms[s] = {upper, fore, reach}` (degrees or null) and, when the
 * hip, knee and ankle are all visible, `legs[s] = {thigh, shin}`; `lower` is
 * what the judge reads of the legs (readLower), null without both whole legs.
 */
export class Measured {
  constructor(lean, hasHips, shoulderPx, centrePx) {
    this.lean = lean;
    this.hasHips = hasHips;
    this.shoulderPx = shoulderPx;
    this.centrePx = centrePx;
    this.arms = {};
    this.legs = {};
    this.lower = null;
  }

  /** Can any arm be judged: a wrist, or at least an elbow (similarity). */
  get seesArms() {
    return Object.values(this.arms).some((a) => a.reach != null || a.upper != null);
  }

  /** Are both legs, feet included, in view: the lower body can be judged. */
  get seesFeet() {
    return this.lower != null;
  }
}

/** A player's keypoints -> torso-frame angles. null without both shoulders. */
export function measure(kpPose, mirrored = true) {
  if (kpPose == null) return null;
  const sides = sideKeypoints(mirrored);
  const pt = (name) => {
    const p = kpPose.point(KP[name], MIN_CONF);
    return p == null ? null : [Number(p[0]), Number(p[1])];
  };
  const sl = pt(`${sides.l}_shoulder`), sr = pt(`${sides.r}_shoulder`);
  if (sl == null || sr == null) return null;
  const sw = Math.hypot(sr[0] - sl[0], sr[1] - sl[1]);
  if (sw < 8.0) return null;
  const centre = [(sl[0] + sr[0]) / 2, (sl[1] + sr[1]) / 2];
  const hl = pt(`${sides.l}_hip`), hr = pt(`${sides.r}_hip`);
  const hasHips = hl != null && hr != null;
  let upx, upy;
  if (hasHips) {
    const hipC = [(hl[0] + hr[0]) / 2, (hl[1] + hr[1]) / 2];
    upx = centre[0] - hipC[0];
    upy = centre[1] - hipC[1];
  } else {
    // Perpendicular to the shoulder line, pointing up.
    const sx = sr[0] - sl[0], sy = sr[1] - sl[1];
    upx = sy;
    upy = -sx;
  }
  const lean = Math.atan2(upx, -upy) / RAD;
  const torso = (v) => rot(v, -lean);

  const m = new Measured(lean, hasHips, sw, centre);
  const joints = {};
  for (const [s, sh] of [["l", sl], ["r", sr]]) {
    const el = pt(`${sides[s]}_elbow`), wr = pt(`${sides[s]}_wrist`);
    const reading = { upper: null, fore: null, reach: null };
    if (wr != null) {
      reading.reach = limbAngle(s, ...torso([wr[0] - sh[0], wr[1] - sh[1]]));
      if (el != null) {
        reading.upper = limbAngle(s, ...torso([el[0] - sh[0], el[1] - sh[1]]));
        reading.fore = limbAngle(s, ...torso([wr[0] - el[0], wr[1] - el[1]]));
      }
    } else if (el != null) {
      reading.upper = limbAngle(s, ...torso([el[0] - sh[0], el[1] - sh[1]]));
    }
    m.arms[s] = reading;
    const hp = pt(`${sides[s]}_hip`), kn = pt(`${sides[s]}_knee`), an = pt(`${sides[s]}_ankle`);
    joints[`${s}_hip`] = hp;
    joints[`${s}_knee`] = kn;
    joints[`${s}_ankle`] = an;
    if (hp != null && kn != null && an != null) {
      // Legs are measured in the world frame, as forward() draws them.
      m.legs[s] = {
        thigh: limbAngle(s, kn[0] - hp[0], kn[1] - hp[1]),
        shin: limbAngle(s, an[0] - kn[0], an[1] - kn[1]),
      };
    }
  }
  m.lower = readLower((name) => joints[name] ?? null);
  return m;
}

// --- similarity -------------------------------------------------------------

export const PERFECT_WITHIN = 20.0; // degrees: no penalty up to here
export const ZERO_AT = 90.0; // degrees: no credit from here
export const ARM_WEIGHT = 1.0, LEAN_WEIGHT = 0.3;
// A target whose lower body is active (legTarget) weighs its legs as much as
// both arms together: a leg-led move is half the score (docs/DANCE.md section 3a).
export const LEGS_WEIGHT = 2.0;
// Legs the camera can't see aren't judged (null): the arms carry the point.
// Feet are optional, because a crowd or a close camera often hides them. A
// number here would give unseen legs that much credit instead (0.5 made hiding
// the feet never beat dancing in full view, but asked every player for feet).
export const UNSEEN_LEGS = null;
export const LEAN_MIN = 8.0; // a target lean smaller than this isn't judged
// Lower-body tolerances: [free, span]: full credit up to `free`, none from free + span.
export const STANCE_TOL = Object.freeze([0.20, 0.45]); // SW
export const WEIGHT_TOL = Object.freeze([0.10, 0.25]); // SW
export const LIFT_TOL = Object.freeze([0.15, 0.30]); // SW
export const BEND_TOL = Object.freeze([20.0, 35.0]); // degrees
export const BEND_SHARE = 0.5; // each knee's bend counts half as much as stance, weight or lift

export function segmentScore(target, seen) {
  const d = angleDiff(target, seen);
  return Math.min(1.0, Math.max(0.0, 1.0 - (d - PERFECT_WITHIN) / (ZERO_AT - PERFECT_WITHIN)));
}

const tol = (d, t) => Math.min(1.0, Math.max(0.0, 1.0 - (Math.abs(d) - t[0]) / t[1]));

/**
 * 0..1 match of a player's lower body on the features the target asks for,
 * or null if the target's legs aren't judged or the player's can't be seen.
 */
export function legsScore(target, seen) {
  if (seen == null || !target.active) return null;
  const t = target.lower;
  let num = 0.0, den = 0.0;
  if (target.step) {
    num += tol(t.stance - seen.stance, STANCE_TOL) + tol(t.weight - seen.weight, WEIGHT_TOL);
    den += 2.0;
  }
  if (target.lift) {
    num += tol(t.lift - seen.lift, LIFT_TOL);
    den += 1.0;
  }
  for (const s of target.bend) {
    num += BEND_SHARE * tol(t.bend[s] - seen.bend[s], BEND_TOL);
    den += BEND_SHARE;
  }
  return num / den;
}

/**
 * 0..1 match between a target pose and a measured player, plus the per-part
 * scores: [score, parts]. score is null when no arm is visible: "can't see
 * you", not a miss. Legs out of view are not judged (never a miss either):
 * the arms carry it.
 */
export function similarity(target, m, unseenLegs = UNSEEN_LEGS) {
  if (m == null) return [null, {}];
  const t = normalizePose(target);
  const parts = {};
  const weights = {};
  for (const s of ["l", "r"]) {
    const a = m.arms[s];
    if (a == null) continue;
    const [tu, tf] = t[`${s}a`];
    if (a.upper != null && a.fore != null) {
      parts[`${s}a`] = 0.5 * (segmentScore(tu, a.upper) + segmentScore(tf, a.fore));
    } else if (a.reach != null) {
      parts[`${s}a`] = segmentScore(targetReach(t, s), a.reach);
    } else if (a.upper != null) {
      // Elbow seen, wrist not: a player standing close raises their hands out
      // of the top of the frame on every arms-up move. The upper arm still says
      // where the arm points; ignoring it left the window with only the arm on
      // its way up, an X for a perfect Clap Up.
      parts[`${s}a`] = segmentScore(tu, a.upper);
    } else continue;
    weights[`${s}a`] = ARM_WEIGHT;
  }
  if (!Object.keys(parts).length) return [null, {}];
  const lt = legTarget(t);
  let legs = legsScore(lt, m.lower);
  if (legs == null && lt.active) legs = unseenLegs;
  if (legs != null) {
    parts.legs = legs;
    weights.legs = LEGS_WEIGHT;
  }
  if (m.hasHips && Math.abs(t.lean) > LEAN_MIN) {
    parts.lean = Math.min(1.0, Math.max(0.0, 1.0 - Math.max(0.0, Math.abs(t.lean - m.lean) - 6.0) / 24.0));
    weights.lean = LEAN_WEIGHT;
  }
  // Same summation order as the Python (parts in insertion order).
  let total = 0, sum = 0;
  for (const k of Object.keys(weights)) total += weights[k];
  for (const k of Object.keys(parts)) sum += parts[k] * weights[k];
  return [sum / total, parts];
}

// --- synthetic players (tests, QA, the dev harness) -------------------------

const PARTS = ["shoulder", "elbow", "wrist", "hip", "knee", "ankle"];

// A player about 2.5-3 m back: the whole body in a 1280 x 720 frame, feet included.
export const FULL_BODY = Object.freeze({ centrePx: Object.freeze([640.0, 250.0]), shoulderPx: 110.0 });

/**
 * A COCO Pose that performs `pose`: what a perfect player looks like to the
 * model. Keypoints outside the frame (or whose part is not in `visible`) get
 * confidence 0. The default framing is a close player whose ankles are below
 * the frame (FULL_BODY shows the feet).
 */
export function keypointsFor(pose, { centrePx = [640.0, 330.0], shoulderPx = 150.0, mirrored = true,
  visible = null, frameH = 720 } = {}) {
  const j = forward(pose);
  const sides = sideKeypoints(mirrored);
  const origin = [
    centrePx[0] - (j.l_shoulder[0] + j.r_shoulder[0]) / 2 * shoulderPx,
    centrePx[1] - (j.l_shoulder[1] + j.r_shoulder[1]) / 2 * shoulderPx,
  ];
  const px = (p) => [origin[0] + p[0] * shoulderPx, origin[1] + p[1] * shoulderPx];
  const xy = new Float32Array(34);
  const conf = new Float32Array(17);
  const head = px(j.head);
  for (const name of ["nose", "left_eye", "right_eye"]) {
    xy[2 * KP[name]] = head[0];
    xy[2 * KP[name] + 1] = head[1];
    conf[KP[name]] = 0.9;
  }
  const show = visible == null ? null : new Set(visible);
  for (const s of ["l", "r"]) {
    for (const part of PARTS) {
      if (show != null && !show.has(part)) continue;
      const i = KP[`${sides[s]}_${part}`];
      const [x, y] = px(j[`${s}_${part}`]);
      xy[2 * i] = x;
      xy[2 * i + 1] = y;
      conf[i] = y >= -5 && y <= frameH + 5 ? 0.9 : 0.0;
    }
  }
  return new Pose(xy, conf, 0.9);
}
