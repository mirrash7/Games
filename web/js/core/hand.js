// Where the player's hand is - shared by the blade and the menu cursor (port of hand.py).
// Points are [x, y] arrays in frame pixels.

import { KP } from "./pose.js";

// How far past the wrist, as a fraction of the forearm, the palm sits.
// 0.30 landed on the lower palm on recorded play; 0.38 is the middle of it.
export const PALM_REACH = 0.38;
// Forearm length relative to shoulder width (median on recorded play).
export const FOREARM_PER_SHOULDER = 0.76;
// How long a remembered forearm stays usable after the elbow drops out.
export const FOREARM_MEMORY = 0.6;

/**
 * COCO wrist keypoint for the player's real hand. On a mirrored (selfie) feed
 * the model labels limbs as they appear, so the real right hand is left_wrist.
 */
export function handKeypoint(hand = "right", mirrored = true) {
  const right = hand === "right";
  if (mirrored) return right ? KP.left_wrist : KP.right_wrist;
  return right ? KP.right_wrist : KP.left_wrist;
}

const elbowFor = (wrist) => (wrist === KP.left_wrist ? KP.left_elbow : KP.right_elbow);

/** Forearm estimate without an elbow: straight up, sized from shoulder width. */
function uprightForearm(pose, minConf) {
  const ls = pose.point(KP.left_shoulder, minConf);
  const rs = pose.point(KP.right_shoulder, minConf);
  if (!ls || !rs) return null;
  const width = Math.hypot(ls[0] - rs[0], ls[1] - rs[1]);
  if (width < 1e-3) return null;
  return [0, -width * FOREARM_PER_SHOULDER];
}

/** Stateless palm position: the wrist pushed out along the forearm. */
export function handPoint(pose, hand = "right", mirrored = true, reach = PALM_REACH, minConf = 0.4) {
  if (!pose) return null;
  const w = handKeypoint(hand, mirrored);
  const wrist = pose.point(w, minConf);
  if (!wrist) return null;
  if (reach <= 0) return wrist;
  const elbow = pose.point(elbowFor(w), minConf);
  const fore = elbow ? [wrist[0] - elbow[0], wrist[1] - elbow[1]] : uprightForearm(pose, minConf);
  if (!fore) return wrist;
  return [wrist[0] + fore[0] * reach, wrist[1] + fore[1] * reach];
}

/**
 * Adaptive low-pass filter for a 2D pointer (Casiez et al., CHI 2012): smooth
 * when slow, nearly transparent when fast. Defaults tuned at 20 Hz poses read
 * at 60 Hz: still-hand jitter 4.6 -> 2.0 px, ~4 ms lag on a fast swipe.
 */
export class OneEuroFilter {
  constructor(minCutoff = 3.0, beta = 0.02, dCutoff = 1.5) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.reset();
  }

  reset() {
    this._x = null;
    this._dx = [0, 0];
    this._t = 0;
  }

  static alpha(dt, cutoff) {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }

  filter(x, t) {
    if (this._x === null) {
      this._x = [x[0], x[1]];
      this._t = t;
      return [x[0], x[1]];
    }
    const dt = t - this._t;
    if (dt <= 0) return [this._x[0], this._x[1]];
    const ad = OneEuroFilter.alpha(dt, this.dCutoff);
    this._dx = [
      ad * ((x[0] - this._x[0]) / dt) + (1 - ad) * this._dx[0],
      ad * ((x[1] - this._x[1]) / dt) + (1 - ad) * this._dx[1],
    ];
    const cutoff = this.minCutoff + this.beta * Math.hypot(this._dx[0], this._dx[1]);
    const a = OneEuroFilter.alpha(dt, cutoff);
    this._x = [a * x[0] + (1 - a) * this._x[0], a * x[1] + (1 - a) * this._x[1]];
    this._t = t;
    return [this._x[0], this._x[1]];
  }
}

/**
 * Stable palm position over time: remembered forearm, then smoothing. The
 * elbow was missing in 54% of frames with the wrist visible; snapping to the
 * bare wrist then made the cursor jump ~39 px/frame.
 */
export class HandTracker {
  constructor(hand = "right", mirrored = true, { reach = PALM_REACH, minConf = 0.4, smooth = true } = {}) {
    this.hand = hand;
    this.mirrored = mirrored;
    this.reach = reach;
    this.minConf = minConf;
    this.filter = smooth ? new OneEuroFilter() : null;
    this._fore = null;
    this._foreT = -1e9;
  }

  reset() {
    this._fore = null;
    this._foreT = -1e9;
    if (this.filter) this.filter.reset();
  }

  /** Palm [x, y] or null. `t` is the caller's own clock in seconds. */
  update(pose, t) {
    if (!pose) {
      this.reset();
      return null;
    }
    const w = handKeypoint(this.hand, this.mirrored);
    const wrist = pose.point(w, this.minConf);
    if (!wrist) {
      this.reset();
      return null;
    }
    const elbow = pose.point(elbowFor(w), this.minConf);
    let fore;
    if (elbow) {
      fore = [wrist[0] - elbow[0], wrist[1] - elbow[1]];
      this._fore = fore;
      this._foreT = t;
    } else if (this._fore && t - this._foreT <= FOREARM_MEMORY) {
      fore = this._fore;
    } else {
      fore = uprightForearm(pose, this.minConf);
    }
    const palm = fore ? [wrist[0] + fore[0] * this.reach, wrist[1] + fore[1] * this.reach] : wrist;
    return this.filter ? this.filter.filter(palm, t) : palm;
  }
}
