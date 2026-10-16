// The coach's motion (port of src/kpapp/game/dance/motion.py): how it moves
// between hits and grooves while it holds them.
//
// Song.coachPose(beat) is the *target* timeline (bots and the judge's tests copy
// it); this is the performance layered on top for the drawn coach only:
// anticipation, an early arrival with a small overshoot that settles exactly on
// the beat, the upper arm leading and the forearm following, a groove locked to
// the beat (dip, squash, weight shift, shoulder roll, sway, twist, head bob),
// follow-through from lagged samples (head tilt, tail, hand flick), hops, steps
// (the supporting foot stays planted while the coach travels over it, the
// stepping foot lifts in an arc; a jump keeps the stance centre and hops), and
// a depth per arm for draw order and foreshortening.
//
// On every keyframe beat the arm angles and the feet are exactly the chart's. Everything is
// a pure function of (keyframes, beat), so web/tests/fixtures/dance_motion.json
// (tools/make_dance_motion_fixture.py) pins this port to the Python.
//
// Output keys are the Python's, in camelCase: pose, travel, lift, squash, sway,
// shoulders, twist, headYaw, headTilt, headBob, tail, hands, depth.

import { EASE_BEATS, idleGroove } from "./chart.js";
import { FEET, FOREARM, HAND, SCALARS, UPPER_ARM, normalizePose, smoothstep } from "./pose.js";

export const ANTICIPATE = 0.10, OVERSHOOT = 0.10;
export const CAPS = Object.freeze({ limb: 8.0, foot: 0.04, lean: 2.5, head: 4.0, shift: 0.03, squat: 0.04 });
const ARRIVE = [0.12, 0.80];
const ANTIC_SPAN = [0.0, 0.32];
const OVER_SPAN = [0.60, 1.0];
const LEAD_UPPER = 0.92, LAG_FORE = 0.06, LAG_LEG = 0.03;

const SECTION_GROOVE = [["chorus", 1.0], ["verse", 0.8], ["bridge", 0.6], ["intro", 0.65], ["outro", 0.5]];
const DIP_SQUAT = 0.045, LEAN_SWAY = 2.0, SWAY = 0.035, SHOULDER_ROLL = 0.035;
const SQUASH = 0.045, STRETCH = 0.018, TWIST_GROOVE = 7.0, TWIST_CROSS = 24.0;
const HEAD_BOB = 0.035, HEAD_SWAY = 3.0, TAIL_SWAY = 12.0;

export const LAG = 0.2, LAG_SHORT = 0.08;
const HAND_FLICK = 0.6, HAND_MAX = 30.0;
const HOP = 0.22;
export const HOP_FEET = 0.25; // SW each foot moves, in opposite directions, that makes a jump...
export const CENTRED = 0.08; // ...when the weight stays centred (SW) before and after

export const STEP_LIFT = 0.16; // SW a stepping foot lifts at most, mid-step
export const STEP_FULL = 0.4; // SW of step that lifts the foot fully (smaller steps lift less)
export const TRAVEL_MAX = 1.8; // SW the coach may travel from its spot (a lint keeps phrases balanced)
// Every transition also drifts the coach back towards its spot: this much of
// the way, at most RECENTRE_MAX SW (the supporting foot slides that little).
export const RECENTRE = 0.15, RECENTRE_MAX = 0.06;

const RAD = Math.PI / 180;
const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

function bump(x, lo, hi) {
  if (x <= lo || x >= hi) return 0.0;
  const s = Math.sin(Math.PI * (x - lo) / (hi - lo));
  return s * s;
}

/** One channel of a transition at x (0..1): anticipate, arrive early, overshoot, settle. */
export function easeValue(v0, v1, x, cap) {
  if (x <= 0.0) return v0;
  if (x >= 1.0) return v1;
  const d = v1 - v0;
  const s = smoothstep((x - ARRIVE[0]) / (ARRIVE[1] - ARRIVE[0]));
  const sign = d >= 0 ? 1.0 : -1.0;
  const mag = Math.abs(d);
  const wind = Math.min(mag * ANTICIPATE, cap) * bump(x, ANTIC_SPAN[0], ANTIC_SPAN[1]);
  const over = Math.min(mag * OVERSHOOT, cap) * bump(x, OVER_SPAN[0], OVER_SPAN[1]);
  return v0 + d * s + sign * (over - wind);
}

/** A transition between two complete poses, the upper arm leading, forearm and legs following. */
export function easePose(p0, p1, x) {
  const out = {};
  const lim = CAPS.limb;
  const xu = clamp(x / LEAD_UPPER, 0.0, 1.0);
  const xf = clamp((x - LAG_FORE) / (1.0 - LAG_FORE), 0.0, 1.0);
  const xl = clamp((x - LAG_LEG) / (1.0 - LAG_LEG), 0.0, 1.0);
  for (const k of ["la", "ra"]) out[k] = [easeValue(p0[k][0], p1[k][0], xu, lim), easeValue(p0[k][1], p1[k][1], xf, lim)];
  const fc = CAPS.foot;
  for (const k of FEET) { // a foot never dips below the floor
    out[k] = [easeValue(p0[k][0], p1[k][0], xl, fc), Math.max(0.0, easeValue(p0[k][1], p1[k][1], xl, fc))];
  }
  for (const k of SCALARS) out[k] = easeValue(p0[k], p1[k], x, CAPS[k]);
  out.squat = clamp(out.squat, 0.0, 1.0);
  return out;
}

/** [p0, p1, x, i]: the transition in progress at `beat` (keyframe i to i + 1; -1 before the first), p1 null while held. */
function segment(frames, beat) {
  if (!frames.length || beat < frames[0][0] - EASE_BEATS) return [idleGroove(beat), null, 1.0, -1];
  if (beat < frames[0][0]) {
    const b1 = frames[0][0];
    return [idleGroove(beat), frames[0][1], (beat - (b1 - EASE_BEATS)) / EASE_BEATS, -1];
  }
  let lo = 0, hi = frames.length - 1;
  if (beat >= frames[hi][0]) return [frames[hi][1], null, 1.0, hi];
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (frames[mid][0] <= beat) lo = mid;
    else hi = mid;
  }
  const [b0, p0] = frames[lo], [b1, p1] = frames[lo + 1];
  const ease = Math.min(EASE_BEATS, b1 - b0);
  if (ease <= 0 || beat <= b1 - ease) return [p0, null, 1.0, lo];
  return [p0, p1, (beat - (b1 - ease)) / ease, lo];
}

/** Screen x of the screen-left and screen-right ankle, relative to the body. */
const feetX = (p) => [-p.lf[0], p.rf[0]];

/** The hips over the feet (SW, + = screen-right), as the judge reads it. */
function weightOf(p) {
  const [lx, rx] = feetX(p);
  return p.shift - (lx + rx) / 2;
}

/**
 * Which foot stays planted from p0 to p1: "l", "r", or null for a jump
 * (motion.support: centred weight before and after, both feet moving
 * HOP_FEET or more one each way). Otherwise the foot under the hips at p0;
 * from a centred stance, the foot away from where the weight is going.
 */
export function support(p0, p1) {
  const [l0, r0] = feetX(p0), [l1, r1] = feetX(p1);
  const dl = l1 - l0, dr = r1 - r0;
  const w0 = weightOf(p0), w1 = weightOf(p1);
  if (Math.abs(w0) < CENTRED && Math.abs(w1) < CENTRED && Math.abs(dl) >= HOP_FEET && Math.abs(dr) >= HOP_FEET
      && dl * dr < 0) return null;
  const nl = Math.abs(l0 - p0.shift), nr = Math.abs(r0 - p0.shift);
  if (Math.abs(nl - nr) > 1e-6) return nl < nr ? "l" : "r";
  if (Math.abs(w1) > 1e-6) return w1 > 0 ? "l" : "r";
  return Math.abs(dl) <= Math.abs(dr) ? "l" : "r";
}

/** The point that stays put on the stage: the supporting foot, or for a jump the stance centre. */
function anchor(p, side) {
  const [lx, rx] = feetX(p);
  return side == null ? (lx + rx) / 2 : side === "l" ? lx : rx;
}

const TRAVEL = new WeakMap();

/**
 * Where the coach stands (SW from its spot) on each keyframe, the supporting
 * foot of each transition and its drift back towards the spot:
 * [offsets, sides, drift]. Cached per frames array.
 */
export function keyframeTravel(frames) {
  const hit = TRAVEL.get(frames);
  if (hit) return hit;
  const poses = frames.map(([, p]) => normalizePose(p));
  const offs = [0.0], sides = [], drift = [];
  for (let i = 1; i < poses.length; i++) {
    const p0 = poses[i - 1], p1 = poses[i];
    const s = support(p0, p1);
    const o = offs[offs.length - 1] + anchor(p0, s) - anchor(p1, s);
    const d = -clamp(o * RECENTRE, -RECENTRE_MAX, RECENTRE_MAX);
    sides.push(s);
    drift.push(d);
    offs.push(clamp(o + d, -TRAVEL_MAX, TRAVEL_MAX));
  }
  const out = [offs, sides, drift];
  TRAVEL.set(frames, out);
  return out;
}

/**
 * The coach's limbs before the groove: the keyframes with anticipation,
 * overshoot and overlap, and the stepping foot lifted in an arc.
 */
export function performedPose(frames, beat) {
  const [p0, p1, x] = segment(frames, beat);
  if (p1 == null) return normalizePose(p0);
  const a = normalizePose(p0), b = normalizePose(p1);
  const out = easePose(a, b, x);
  const s = support(a, b);
  if (s != null) {
    const free = s === "l" ? "r" : "l";
    const step = Math.abs(b[`${free}f`][0] - a[`${free}f`][0]);
    if (step > 0.05) out[`${free}f`][1] += STEP_LIFT * Math.min(1.0, step / STEP_FULL) * bump(x, 0.08, 0.9);
  }
  return out;
}

/** SW the coach stands from its spot at `beat`, `pose` being the performed pose then. */
function travelAt(frames, beat, pose) {
  const [, p1, x, i] = segment(frames, beat);
  if (i < 0) return 0.0;
  const [offs, sides, drift] = keyframeTravel(frames);
  if (p1 == null) return offs[i];
  const s = sides[i];
  const o = offs[i] + anchor(normalizePose(frames[i][1]), s) - anchor(pose, s) + drift[i] * smoothstep(x);
  return clamp(o, -TRAVEL_MAX, TRAVEL_MAX);
}

/** Height of a hop on a jump (both feet spreading or closing): up over the move, down before its beat. */
function hop(frames, beat) {
  const [p0, p1, x] = segment(frames, beat);
  if (p1 == null) return 0.0;
  const a = normalizePose(p0), b = normalizePose(p1);
  if (support(a, b) != null) return 0.0;
  const [l0, r0] = feetX(a), [l1, r1] = feetX(b);
  const k = Math.min(1.0, Math.min(Math.abs(l1 - l0), Math.abs(r1 - r0)) / (HOP_FEET * 1.6));
  const u = clamp((x - 0.18) / 0.62, 0.0, 1.0);
  return HOP * k * Math.sin(Math.PI * u);
}

function handX(pose, side) {
  const [up, fore] = pose[`${side}a`];
  return -(0.5 + UPPER_ARM * Math.sin(up * RAD) + (FOREARM + HAND) * Math.sin(fore * RAD));
}

const crossing = (pose, side) => clamp((handX(pose, side) + 0.1) / 0.6, 0.0, 1.0);

/** How hard the coach grooves: by the song's section at `beat`. */
export function sectionEnergy(song, beat) {
  let name = "";
  try {
    name = String(song?.sectionAt?.(beat) ?? "").toLowerCase();
  } catch {
    name = "";
  }
  for (const [key, e] of SECTION_GROOVE) if (name.includes(key)) return e;
  return 0.8;
}

function dip(beat) {
  const c = 0.5 + 0.5 * Math.cos(2.0 * Math.PI * beat);
  return c * c;
}

/**
 * The drawn coach at `beat` (motion.py coach_motion): the performed pose plus
 * the groove and follow-through channels the painter applies.
 * `frames` overrides the song's keyframes; with neither, the idle groove.
 */
export function coachMotion(song, beat, { energy = null, frames = null } = {}) {
  frames = frames ?? song?._frames ?? [];
  const e = energy == null ? sectionEnergy(song, beat) : Number(energy);
  const pose = performedPose(frames, beat);
  const travel = frames.length ? travelAt(frames, beat, pose) : 0.0;
  const lag = performedPose(frames, beat - LAG);
  const near = performedPose(frames, beat - LAG_SHORT);
  const nearer = performedPose(frames, beat - 2 * LAG_SHORT);

  const d = dip(beat);
  const sway2 = Math.sin(Math.PI * beat);
  const swing = Math.cos(Math.PI * beat);
  pose.squat = clamp(pose.squat + DIP_SQUAT * e * d, 0.0, 1.0);
  pose.lean += LEAN_SWAY * e * sway2;

  const off = 0.5 - 0.5 * Math.cos(2.0 * Math.PI * beat);
  const squash = 1.0 - SQUASH * e * d + STRETCH * e * off * off;
  let cross = 0;
  for (const q of [pose, near, nearer]) cross += crossing(q, "l") - crossing(q, "r");
  const twistPose = TWIST_CROSS * cross / 3.0;
  const twist = twistPose + TWIST_GROOVE * e * Math.sin(Math.PI * (beat - 0.25));
  const shoulder = SHOULDER_ROLL * e * Math.sin(Math.PI * (beat - 0.15));

  let headTilt = 0.55 * (lag.lean - pose.lean) + 0.45 * (lag.head - pose.head);
  headTilt += HEAD_SWAY * e * Math.sin(Math.PI * (beat - 0.3));
  let tail = TAIL_SWAY * e * Math.sin(Math.PI * (beat - 0.4)) + 0.8 * (lag.lean - pose.lean);
  tail += 40.0 * (lag.shift - pose.shift);
  const hands = ["l", "r"].map((s) => clamp(HAND_FLICK * (near[`${s}a`][1] - pose[`${s}a`][1]), -HAND_MAX, HAND_MAX));

  const yaw = Math.sin(twist * RAD);
  const depth = {};
  for (const s of ["l", "r"]) {
    const [up, fore] = pose[`${s}a`];
    const turn = s === "l" ? yaw : -yaw;
    const across = crossing(pose, s);
    const upperZ = clamp(-up / 60.0, 0.0, 1.0) + turn;
    const handZ = Math.max(across, clamp(-fore / 70.0, 0.0, 1.0), 0.6 * clamp((fore - 195.0) / 40.0, 0.0, 1.0)) + 0.5 * turn;
    depth[s] = [clamp(upperZ, -1.0, 1.0), clamp(handZ, -1.0, 1.0)];
  }

  return {
    pose,
    travel,
    lift: hop(frames, beat),
    squash,
    sway: SWAY * e * swing,
    shoulders: [shoulder, -shoulder],
    twist,
    headYaw: 0.7 * twist + 5.0 * e * Math.sin(Math.PI * (beat - 0.45)),
    headTilt,
    headBob: HEAD_BOB * e * dip(beat - 0.1),
    tail,
    hands,
    depth,
  };
}

/** A pose with no motion layered on: what the tutorial and tests draw. */
export function stillMotion(pose) {
  return {
    pose: normalizePose(pose), travel: 0.0, lift: 0.0, squash: 1.0, sway: 0.0, shoulders: [0.0, 0.0], twist: 0.0, headYaw: 0.0,
    headTilt: 0.0, headBob: 0.0, tail: 0.0, hands: [0.0, 0.0], depth: { l: [0.0, 0.0], r: [0.0, 0.0] },
  };
}
