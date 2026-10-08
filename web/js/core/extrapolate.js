// Projects keypoints forward to render time with per-keypoint velocity
// (port of inference.PoseExtrapolator). Inference lands at 15-30 Hz and each
// result is already one model-latency old; projecting removes most of that
// lag. Gain < 1 under-shoots on purpose: overshoot at a direction reversal
// reads as a snap-back, which feels worse than a little lag.

import { Pose } from "./pose.js";

export class PoseExtrapolator {
  constructor({ gain = 0.7, maxLead = 0.12, velocitySmoothing = 0.5, maxGap = 0.25 } = {}) {
    this.gain = gain;
    this.maxLead = maxLead;
    this.velocitySmoothing = velocitySmoothing;
    this.maxGap = maxGap;
    this._poses = [];
    this._vel = new Map();
    this._prev = new Map(); // slot -> {xy, conf}
    this._prevT = null;
    this.lastLeadMs = 0;
    this.held = []; // the newest result projected to when it arrived, held until the next
  }

  /**
   * Feed one fresh result: poses (Pose[]) and the time (s) its frame entered
   * the model. `arrived` is when the render loop picked it up: `held` is then
   * the result projected to that moment, held until the next result.
   *
   * Use `held` for gesture detectors that measure speed between model updates
   * (FlapDetector). posesAt() moves every render frame, so each new result
   * shows up as a jump between two frames 8-16 ms apart, which the detector's
   * teleport guard reads as a tracking glitch, dropping its history. Simulated
   * at 120 Hz render with 15-30 Hz poses: 0 of 30 flaps caught; with `held`, 30/30.
   */
  update(poses, timestamp, arrived = null) {
    if (this.gain > 0 && this._prevT !== null) {
      const dt = timestamp - this._prevT;
      if (dt > 0 && dt <= this.maxGap) {
        poses.forEach((pose, slot) => {
          const prev = this._prev.get(slot);
          if (!prev) return;
          const raw = new Float32Array(pose.xy.length);
          for (let k = 0; k < pose.confidence.length; k++) {
            // Confidence in BOTH frames: a keypoint that just reappeared has
            // no usable history and would be flung across the frame.
            if (pose.confidence[k] >= 0.5 && prev.conf[k] >= 0.5) {
              raw[2 * k] = (pose.xy[2 * k] - prev.xy[2 * k]) / dt;
              raw[2 * k + 1] = (pose.xy[2 * k + 1] - prev.xy[2 * k + 1]) / dt;
            }
          }
          const old = this._vel.get(slot);
          if (old) {
            const a = this.velocitySmoothing;
            for (let i = 0; i < raw.length; i++) raw[i] = a * old[i] + (1 - a) * raw[i];
          }
          this._vel.set(slot, raw);
        });
      } else {
        this._vel.clear(); // stalled worker or first result
      }
    }
    this._prev = new Map(poses.map((p, i) => [i, { xy: p.xy.slice(), conf: p.confidence.slice() }]));
    this._prevT = timestamp;
    this._poses = poses;
    for (const slot of [...this._vel.keys()]) if (slot >= poses.length) this._vel.delete(slot);
    this.held = arrived == null ? poses : this.posesAt(arrived);
  }

  /** Poses projected to time `now` (s, same clock as the timestamps). */
  posesAt(now) {
    if (this.gain <= 0 || !this._poses.length || this._prevT === null) {
      this.lastLeadMs = 0;
      return this._poses;
    }
    const lead = Math.min(Math.max(now - this._prevT, 0), this.maxLead) * this.gain;
    this.lastLeadMs = lead * 1000;
    if (lead <= 0) return this._poses;
    return this._poses.map((pose, slot) => {
      const vel = this._vel.get(slot);
      if (!vel) return pose;
      const xy = new Float32Array(pose.xy.length);
      for (let i = 0; i < xy.length; i++) xy[i] = pose.xy[i] + vel[i] * lead;
      return new Pose(xy, pose.confidence, pose.score, pose.box);
    });
  }

  clear() {
    this._poses = [];
    this._vel.clear();
    this._prev.clear();
    this._prevT = null;
    this.held = [];
  }
}
