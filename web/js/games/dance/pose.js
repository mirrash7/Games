// Dance poses: angles, forward kinematics, measuring a player, similarity
// (port of src/kpapp/game/dance/pose.py; docs/DANCE.md section 3).
//
// A pose is limb angles in degrees in the torso frame (0 = straight down,
// positive = outward then up, 180 = straight up), in screen space as the
// player sees it, plus lean/head/shift/squat. The coach is drawn from
// forward(pose); the player is read into the same angles with measure();
// similarity() compares the two.
//
// No DOM: this module runs under Node for the tests.

import { KP, Pose } from "../../core/pose.js";

// --- representation ---------------------------------------------------------

export const LIMBS = ["la", "ra", "ll", "rl"]; // arms then legs; "l"/"r" = screen-left/right
export const SCALARS = ["lean", "head", "shift", "squat"];
export const REST = Object.freeze({
  la: Object.freeze([10.0, 5.0]), ra: Object.freeze([10.0, 5.0]),
  ll: Object.freeze([3.0, 2.0]), rl: Object.freeze([3.0, 2.0]),
  lean: 0.0, head: 0.0, shift: 0.0, squat: 0.0,
});

// Body proportions in shoulder widths (docs/DANCE.md section 3).
export const TORSO = 1.25, HIP_W = 0.6, NECK = 0.2, HEAD_R = 0.3;
export const UPPER_ARM = 0.62, FOREARM = 0.58, HAND = 0.16;
export const THIGH = 0.85, SHIN = 0.82;

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

/**
 * A complete pose: missing keys from REST, values as numbers. Throws on
 * unknown keys or malformed values (charts are validated with this).
 */
export function normalizePose(pose) {
  const out = {
    la: [...REST.la], ra: [...REST.ra], ll: [...REST.ll], rl: [...REST.rl],
    lean: REST.lean, head: REST.head, shift: REST.shift, squat: REST.squat,
  };
  for (const [key, value] of Object.entries(pose ?? {})) {
    if (LIMBS.includes(key)) {
      if (!(Array.isArray(value) && value.length === 2 && value.every(isNum))) {
        throw new Error(`'${key}' must be [upper, lower] angles in degrees, got ${pyRepr(value)}`);
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
  return out;
}

/** The same move done on the other side of the screen. */
export function mirrorPose(pose) {
  const p = normalizePose(pose);
  return {
    la: [...p.ra], ra: [...p.la], ll: [...p.rl], rl: [...p.ll],
    lean: -p.lean, head: -p.head, shift: -p.shift, squat: p.squat,
  };
}

export function smoothstep(u) {
  u = Math.min(1.0, Math.max(0.0, u));
  return u * u * (3.0 - 2.0 * u);
}

/**
 * Blend two complete poses (u = 0 -> a, 1 -> b). Angles blend linearly, so
 * authors keep consecutive keyframes within 180 degrees of each other.
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

const RAD = Math.PI / 180;

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

/**
 * Joint positions in shoulder widths, y down, origin at the rest hip centre.
 *
 * `shift` moves the body sideways; `squat` lowers the hips and bends the
 * knees outward (cosmetic). Arms hang from the leaning torso; legs stay in
 * the world frame so the feet stay planted.
 */
export function forward(pose) {
  const p = normalizePose(pose);
  const lean = p.lean;
  const hip = [p.shift, p.squat * 0.42];
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
    let [thigh, shin] = p[`${s}l`];
    // Squatting bends the knees outward and the shins back in.
    thigh += 38.0 * p.squat;
    shin -= 12.0 * p.squat;
    const hp = add(hip, [SIDE[s] * HIP_W / 2, 0.0]);
    const kn = add(hp, limbDir(s, thigh), THIGH);
    j[`${s}_hip`] = hp;
    j[`${s}_knee`] = kn;
    j[`${s}_ankle`] = add(kn, limbDir(s, shin), SHIN);
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
 * hip, knee and ankle are all visible, `legs[s] = {thigh, shin}`.
 */
export class Measured {
  constructor(lean, hasHips, shoulderPx, centrePx) {
    this.lean = lean;
    this.hasHips = hasHips;
    this.shoulderPx = shoulderPx;
    this.centrePx = centrePx;
    this.arms = {};
    this.legs = {};
  }

  /** Can any arm be judged: a wrist, or at least an elbow (similarity). */
  get seesArms() {
    return Object.values(this.arms).some((a) => a.reach != null || a.upper != null);
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
    if (hp != null && kn != null && an != null) {
      // Legs are measured in the world frame, as forward() draws them.
      m.legs[s] = {
        thigh: limbAngle(s, kn[0] - hp[0], kn[1] - hp[1]),
        shin: limbAngle(s, an[0] - kn[0], an[1] - kn[1]),
      };
    }
  }
  return m;
}

// --- similarity -------------------------------------------------------------

export const PERFECT_WITHIN = 20.0; // degrees: no penalty up to here
export const ZERO_AT = 90.0; // degrees: no credit from here
export const ARM_WEIGHT = 1.0, LEG_WEIGHT = 0.35, LEAN_WEIGHT = 0.3;
export const LEG_REST_MARGIN = 25.0; // a target leg this close to rest isn't judged
export const LEAN_MIN = 8.0; // a target lean smaller than this isn't judged

export function segmentScore(target, seen) {
  const d = angleDiff(target, seen);
  return Math.min(1.0, Math.max(0.0, 1.0 - (d - PERFECT_WITHIN) / (ZERO_AT - PERFECT_WITHIN)));
}

/**
 * 0..1 match between a target pose and a measured player, plus the per-part
 * scores: [score, parts]. score is null when no arm is visible: "can't see
 * you", not a miss.
 */
export function similarity(target, m) {
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
  for (const s of ["l", "r"]) {
    const leg = m.legs[s];
    const [tt, ts] = t[`${s}l`];
    const [rt, rs] = REST[`${s}l`];
    if (leg == null || (Math.abs(tt - rt) <= LEG_REST_MARGIN && Math.abs(ts - rs) <= LEG_REST_MARGIN)) continue;
    parts[`${s}l`] = 0.5 * (segmentScore(tt, leg.thigh) + segmentScore(ts, leg.shin));
    weights[`${s}l`] = LEG_WEIGHT;
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

/**
 * A COCO Pose that performs `pose`: what a perfect player looks like to the
 * model. Keypoints outside the frame (or whose part is not in `visible`) get
 * confidence 0.
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
