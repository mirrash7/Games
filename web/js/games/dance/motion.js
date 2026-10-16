// The coach's motion (port of src/kpapp/game/dance/motion.py): how it moves
// between hits and grooves while it holds them.
//
// Song.coachPose(beat) is the *target* timeline (bots and the judge's tests copy
// it); this is the performance layered on top for the drawn coach only:
// anticipation, an early arrival with a small overshoot that settles exactly on
// the beat, the upper arm leading and the forearm following, a groove locked to
// the beat (dip, squash, weight shift, shoulder roll, sway, twist, head bob),
// follow-through from lagged samples (head tilt, tail, hand flick), hops, and a
// depth per arm for draw order and foreshortening.
//
// On every keyframe beat the limb angles are exactly the chart's. Everything is
// a pure function of (keyframes, beat), so web/tests/fixtures/dance_motion.json
// (tools/make_dance_motion_fixture.py) pins this port to the Python.
//
// Output keys are the Python's, in camelCase: pose, lift, squash, sway,
// shoulders, twist, headYaw, headTilt, headBob, tail, hands, depth.

import { EASE_BEATS, idleGroove } from "./chart.js";
import { FOREARM, HAND, SCALARS, UPPER_ARM, normalizePose, smoothstep } from "./pose.js";

export const ANTICIPATE = 0.10, OVERSHOOT = 0.10;
export const CAPS = Object.freeze({ limb: 8.0, lean: 2.5, head: 4.0, shift: 0.03, squat: 0.04 });
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
const HOP = 0.22, HOP_LEGS = 20.0;

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
  for (const k of ["ll", "rl"]) out[k] = [easeValue(p0[k][0], p1[k][0], xl, lim), easeValue(p0[k][1], p1[k][1], xl, lim)];
  for (const k of SCALARS) out[k] = easeValue(p0[k], p1[k], x, CAPS[k]);
  out.squat = clamp(out.squat, 0.0, 1.0);
  return out;
}

function segment(frames, beat) {
  if (!frames.length || beat < frames[0][0] - EASE_BEATS) return [idleGroove(beat), null, 1.0];
  if (beat < frames[0][0]) {
    const b1 = frames[0][0];
    return [idleGroove(beat), frames[0][1], (beat - (b1 - EASE_BEATS)) / EASE_BEATS];
  }
  let lo = 0, hi = frames.length - 1;
  if (beat >= frames[hi][0]) return [frames[hi][1], null, 1.0];
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (frames[mid][0] <= beat) lo = mid;
    else hi = mid;
  }
  const [b0, p0] = frames[lo], [b1, p1] = frames[lo + 1];
  const ease = Math.min(EASE_BEATS, b1 - b0);
  if (ease <= 0 || beat <= b1 - ease) return [p0, null, 1.0];
  return [p0, p1, (beat - (b1 - ease)) / ease];
}

/** The coach's limbs before the groove: the keyframes with anticipation, overshoot and overlap. */
export function performedPose(frames, beat) {
  const [p0, p1, x] = segment(frames, beat);
  if (p1 == null) return normalizePose(p0);
  return easePose(normalizePose(p0), normalizePose(p1), x);
}

function hop(frames, beat) {
  const [p0, p1, x] = segment(frames, beat);
  if (p1 == null) return 0.0;
  const a = normalizePose(p0), b = normalizePose(p1);
  const change = Math.min(...["ll", "rl"].map((k) => Math.max(Math.abs(b[k][0] - a[k][0]), Math.abs(b[k][1] - a[k][1]))));
  if (change < HOP_LEGS) return 0.0;
  const k = Math.min(1.0, change / (HOP_LEGS * 1.6));
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
    pose: normalizePose(pose), lift: 0.0, squash: 1.0, sway: 0.0, shoulders: [0.0, 0.0], twist: 0.0, headYaw: 0.0,
    headTilt: 0.0, headBob: 0.0, tail: 0.0, hands: [0.0, 0.0], depth: { l: [0.0, 0.0], r: [0.0, 0.0] },
  };
}
