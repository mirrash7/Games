// Synthetic golfers for the Raccoon Golf tests (port of tests/golf_poses.py).
//
// A golfer faces the camera. Their grip moves on keyframed paths in the body
// frame: `u` sideways from the shoulder midpoint and `h` above the shoulder
// line, both in shoulder widths (sw). Poses come out the way the real pipeline
// delivers them: from a model running at 15-30 Hz (each pose held until the
// next, as controls.stepPose is), late by a fixed lag, with ~5 px of keypoint
// noise, and elbows mostly missing.

import { KP, Pose } from "../js/core/pose.js";
import { Rng } from "../js/core/random.js";
import { FULL_H } from "../js/games/golf/swing.js";

export const GAME_HZ = 60.0;
export const SW = 200.0;
export const CENTRE = [640.0, 330.0];
export const RADIUS = 1.35;

/** A pose whose grip (midpoint of the wrists) is at (u, h) sw; u = null: no hands. */
export function gripPose(u, h, { sw = SW, centre = CENTRE, noise = 0.0, rng = null, gap = 0.15, elbows = false,
  hands = 2, shoulders = true } = {}) {
  rng = rng ?? new Rng(0);
  const xy = new Float32Array(34), c = new Float32Array(17);
  const [cx, cy] = centre;
  const pts = { nose: [cx, cy - 0.7 * sw] };
  if (shoulders) {
    pts.left_shoulder = [cx + sw / 2, cy];
    pts.right_shoulder = [cx - sw / 2, cy];
  }
  if (u != null) {
    const gx = cx + u * sw, gy = cy - h * sw;
    if (hands >= 1) pts.left_wrist = [gx + (gap * sw) / 2, gy];
    if (hands >= 2) pts.right_wrist = [gx - (gap * sw) / 2, gy];
    if (elbows) {
      pts.left_elbow = [gx + 0.35 * sw, gy - 0.55 * sw];
      pts.right_elbow = [gx - 0.35 * sw, gy - 0.55 * sw];
    }
  }
  for (const [name, p] of Object.entries(pts)) {
    const i = KP[name];
    xy[2 * i] = p[0] + (noise ? rng.gauss(0, noise) : 0);
    xy[2 * i + 1] = p[1] + (noise ? rng.gauss(0, noise) : 0);
    c[i] = 0.9;
  }
  return new Pose(xy, c, 0.9);
}

/** (u, h)(t) through [t, u, h] keyframes with cosine easing between them. */
export function path(...kf) {
  return (t) => {
    if (t <= kf[0][0]) return [kf[0][1], kf[0][2]];
    for (let i = 0; i + 1 < kf.length; i++) {
      const [t0, u0, h0] = kf[i], [t1, u1, h1] = kf[i + 1];
      if (t <= t1) {
        const e = (1 - Math.cos((Math.PI * (t - t0)) / (t1 - t0))) / 2;
        return [u0 + (u1 - u0) * e, h0 + (h1 - h0) * e];
      }
    }
    const last = kf[kf.length - 1];
    return [last[1], last[2]];
  };
}

/** The grip on a circle round the shoulders: 0 = hanging straight down, + swings out to `side` and up. */
export const arc = (angle, radius = RADIUS, side = 1) => [side * radius * Math.sin(angle), -radius * Math.cos(angle)];

/** The backswing angle whose top height gives `power` (swing.js's formula). */
export function topAngleFor(power, radius = RADIUS) {
  const hAddr = -radius;
  const h = hAddr + power * (FULL_H - hAddr);
  return Math.acos(Math.max(-1.0, Math.min(1.0, -h / radius)));
}

/** A full swing: address, backswing to `power`, a fast downswing, a follow-through. Returns [motion, tImpact]. */
export function swingPath({ power = 1.0, side = 1, start = 0.8, back = 0.8, pause = 0.15, down = 0.30, follow = 0.35,
  addressHold = 0.8, radius = RADIUS, lowCross = 0.0 } = {}) {
  const top = topAngleFor(power, radius);
  const t0 = start + addressHold, tTop = t0 + back, tDown = tTop + pause, tEnd = tDown + down + follow;
  const at = (t) => {
    if (t < t0) return arc(0.0, radius, side);
    if (t < tTop) return arc(top * ((1 - Math.cos((Math.PI * (t - t0)) / back)) / 2), radius, side);
    if (t < tDown) return arc(top, radius, side);
    if (t < tEnd) {
      const k = (t - tDown) / (down + follow);
      const a = top + (-0.9 * top - top) * (k * k * (3 - 2 * k));
      const [u, h] = arc(a, radius, side);
      return [u, h + lowCross * Math.max(0.0, Math.cos(a)) ** 4];
    }
    return arc(-0.9 * top, radius, side);
  };
  let lo = 0.0, hi = 1.0;
  const e = 1 / 1.9;
  for (let i = 0; i < 40; i++) {
    const m = (lo + hi) / 2;
    if (m * m * (3 - 2 * m) < e) lo = m;
    else hi = m;
  }
  return [at, tDown + lo * (down + follow)];
}

export function modelClock(modelHz, duration, rng, jitter = 0.25, phase = 0.0) {
  const times = [];
  let t = phase - 1.0 / modelHz;
  while (t < duration + 1.0) {
    times.push(t);
    t += (1.0 / modelHz) * (1.0 + rng.uniform(-jitter, jitter));
  }
  return times;
}

/** The last index i with arr[i] <= x (numpy's searchsorted(x + eps) - 1). */
export function held(arr, x) {
  let lo = 0, hi = arr.length;
  while (lo < hi) {
    const m = (lo + hi) >> 1;
    if (arr[m] <= x + 1e-9) lo = m + 1;
    else hi = m;
  }
  return lo - 1;
}

/**
 * [t, pose] at GAME_HZ for a grip moving along motion(t) -> [u, h] or null.
 * The model publishes at modelHz (jittered); each pose shows the player as they
 * were `lag` seconds before it is delivered, and is held until the next.
 */
export function* stream(motion, duration, { modelHz = 20.0, lag = 0.0, noise = 0.0, seed = 0, elbowsEvery = 0,
  gap = 0.15, hands = null, lost = null, sw = SW, centre = CENTRE } = {}) {
  const rng = new Rng(seed);
  const clock = modelHz ? modelClock(modelHz, duration, rng) : null;
  let heldKey = null, pose = null, n = 0;
  for (let i = 0; i < Math.trunc(duration * GAME_HZ); i++) {
    const t = i / GAME_HZ;
    const ts = clock ? clock[held(clock, t)] : t;
    if (lost && lost(ts)) {
      yield [t, null];
      continue;
    }
    if (ts !== heldKey) {
      heldKey = ts;
      n += 1;
      const m = motion(ts - lag);
      const [u, h] = m == null ? [null, null] : m;
      pose = gripPose(u, h, { sw, centre, noise, rng, gap, elbows: !!elbowsEvery && n % elbowsEvery === 0,
        hands: hands ? hands(ts) : 2 });
    }
    yield [t, pose];
  }
}

/** Feed a detector; return the events. */
export function runDetector(det, motion, duration, opts = {}) {
  const events = [];
  for (const [t, pose] of stream(motion, duration, opts)) {
    const ev = det.update(pose, t);
    if (ev) events.push(ev);
  }
  return events;
}
