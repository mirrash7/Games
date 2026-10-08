// Pose -> ControlState, with hysteresis on the boolean gestures (port of controls.py).
// Normalised against the player's own shoulder width so the same gesture reads
// the same near the camera or across the room.

import { KP } from "./pose.js";

const MIN_CONF = 0.5;
const clip = (v, a, b) => Math.min(Math.max(v, a), b);

/** One frame of player input. This is the contract games code against. */
export function emptyControls(pose = null) {
  return {
    present: false, // a player with shoulders visible
    steer: 0, // -1 (left) .. +1 (right)
    throttle: 0, // 0 hands down .. 1 hands up
    leftHand: null, // [x, y] normalised 0..1
    rightHand: null,
    handsUp: false,
    armsOut: false,
    crouching: false,
    pose, // the Pose projected to this frame (or null)
    // The same player updated only when the model produces a new result
    // (projected to when it arrived), so it moves in steps. Gesture detectors
    // that time motion between model updates want this; see PoseExtrapolator.update.
    stepPose: null,
  };
}

export class ControlMapper {
  constructor(width, height, deadzone = 0.08) {
    this.width = width;
    this.height = height;
    this.deadzone = deadzone;
    this._latched = {};
  }

  _latch(name, value, on, off) {
    const state = this._latched[name] ? value > off : value > on;
    this._latched[name] = state;
    return state;
  }

  _dz(v) {
    if (Math.abs(v) < this.deadzone) return 0;
    return clip((Math.abs(v) - this.deadzone) / (1 - this.deadzone), 0, 1) * Math.sign(v);
  }

  map(pose) {
    if (!pose) {
      this._latched = {};
      return emptyControls();
    }
    const ls = pose.point(KP.left_shoulder, MIN_CONF);
    const rs = pose.point(KP.right_shoulder, MIN_CONF);
    if (!ls || !rs) return emptyControls(pose);

    const mid = [(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2];
    const sw = Math.max(Math.hypot(ls[0] - rs[0], ls[1] - rs[1]), 1);
    const lh = pose.point(KP.left_wrist, MIN_CONF);
    const rh = pose.point(KP.right_wrist, MIN_CONF);
    const lhip = pose.point(KP.left_hip, MIN_CONF);
    const rhip = pose.point(KP.right_hip, MIN_CONF);

    const s = emptyControls(pose);
    s.present = true;
    s.leftHand = lh ? [lh[0] / this.width, lh[1] / this.height] : null;
    s.rightHand = rh ? [rh[0] / this.width, rh[1] / this.height] : null;

    if (lh && rh) {
      const offset = ((lh[0] + rh[0]) / 2 - mid[0]) / sw;
      s.steer = this._dz(clip(offset / 0.9, -1, 1));
    } else if (lhip && rhip) {
      const lean = (mid[0] - (lhip[0] + rhip[0]) / 2) / sw;
      s.steer = this._dz(clip(lean / 0.5, -1, 1));
    }

    const heights = [lh, rh].filter(Boolean).map((h) => -(h[1] - mid[1]) / sw);
    if (heights.length) s.throttle = clip(Math.max(...heights) / 1.2, 0, 1);
    s.handsUp = this._latch("handsUp", s.throttle, 0.55, 0.4);

    if (lh && rh) {
      const spread = (Math.abs(lh[0] - mid[0]) + Math.abs(rh[0] - mid[0])) / sw;
      s.armsOut = this._latch("armsOut", spread, 1.7, 1.4);
    }
    if (lhip && rhip) {
      const torso = ((lhip[1] + rhip[1]) / 2 - mid[1]) / sw;
      s.crouching = this._latch("crouch", -torso, -0.85, -1.0);
    }
    return s;
  }
}
