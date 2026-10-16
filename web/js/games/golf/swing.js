// Golf swings from a webcam: address, backswing (power), downswing (impact)
// (port of src/kpapp/game/golf/swing.py - keep the two in step;
// web/tests/golf_parity.test.mjs replays the Python detector's recorded input).
//
// The game depends on exactly this surface: SwingDetector.update(pose, t)
// returning a swing event ({t, power, quality, side, speed, practice,
// overswing}) or null, plus the read-only state the HUD shows.
//
// **What is measured.** The grip: the midpoint of the two wrists (not the
// palms: hand.js's palm sits past the wrist only when the elbow is seen, so the
// grip would jump between frames). Positions are in shoulder widths (sw) in a
// body frame: `u` sideways from the shoulder midpoint (screen direction), `h`
// height above the shoulder line (+ up). Hips are often out of frame, so
// heights come from the shoulder line (AGENTS.md section 6).
//
// **The body frame freezes during a swing.** Seen from the front the shoulders
// turn away at the top of a backswing and their apparent width halves; the
// frame is smoothed at address and frozen from the start of the backswing.
//
// **Stepped input.** Like FlapDetector this reads controls.stepPose, keeps only
// distinct observations, and dates a held one to max(first seen, last seen -
// one model period).
//
// **Stages.** NO_GRIP -> ADDRESSING (hands together, low, near the midline,
// still) -> READY after ADDRESS_HOLD -> BACKSWING (grip rose START_RISE above
// address, or moved START_SIDE to either side) -> DOWNSWING (fell DOWN_DROP
// below its peak). Impact is the grip's swept segment crossing the address
// line toward the other side, below the shoulder line, at HIT_SPEED or more;
// a straight-up backswing hits when the grip falls through CHOP_LINE instead.
// Slower than HIT_SPEED: a lowering of the hands, cancelled ("SWING THROUGH!").
//
// **Power** = (peak - address height) / (FULL_H - address height), up to
// OVERSWING_MAX in the red zone. **Quality** from the crossing height against
// address (CLEAN_H clean .. THIN_H thin). A swing that starts before READY is a
// practice swing (practice: true). **Putting** (mode "putt") is a pendulum:
// sideways travel for power, impact on crossing back over the address point
// within PUTT_STROKE seconds.
//
// **Tracking.** A jump larger than TELEPORT (of the frame width) is a glitch:
// history is dropped. Losing the grip for longer than LOSS_KEEP cancels a swing
// in progress - never a stroke for tracking loss.
//
// Pose keypoints are float32, and the Python does some sums in float32 (numpy);
// Math.fround reproduces those exactly so the two builds agree to the bit.

import { KP } from "../../core/pose.js";

export const MIN_CONF = 0.4;

// --- address ---
export const TOGETHER = 0.7;
export const ADDRESS_MAX_H = -0.7;
export const ADDRESS_MAX_U = 0.6;
export const STEADY = 0.15;
export const ADDRESS_HOLD = 0.4;
export const FRAME_TAU = 0.35;

// --- the swing ---
export const START_RISE = 0.30;
export const START_SIDE = 0.55;
export const FULL_H = 0.75;
export const OVERSWING_MAX = 1.15;
export const DOWN_DROP = 0.25;
export const SIDE_MIN = 0.25;
export const HIT_SPEED = 3.5;
export const HIT_MAX_H = -0.15;
export const CHOP_LINE = 0.45;
export const CLEAN_H = 0.35;
export const THIN_H = 1.0;
export const SWING_TIMEOUT = 1.2;

// --- putting ---
export const PUTT_START = 0.09;
export const PUTT_FULL = 1.2;
export const PUTT_RETURN = 0.05;
export const PUTT_SMOOTH = 3;
export const PUTT_STROKE = 1.0;

// --- tracking ---
export const SPEED_WINDOW = 0.10;
export const MAX_GAP = 0.25;
export const TELEPORT = 0.35;
export const LOSS_KEEP = 0.30;
export const ONE_HAND_KEEP = 0.30;
export const HOLD_EPS = 0.5;
export const DEFAULT_PERIOD = 1.0 / 20.0;
export const HINT_SECONDS = 1.6;
export const TOO_CLOSE_ON = 1.3;
export const TOO_CLOSE_OFF = 1.5;

export const Stage = Object.freeze({
  NO_GRIP: "no_grip", ADDRESSING: "addressing", READY: "ready", BACKSWING: "backswing", DOWNSWING: "downswing",
});

/** A recognised swing. overswing: 0 .. 1 through the red zone. */
export function swingEvent(t, power, quality, side, speed, practice) {
  return { t, power, quality, side, speed, practice,
    overswing: Math.max(0.0, Math.min(1.0, (power - 1.0) / (OVERSWING_MAX - 1.0))) };
}

const f32 = Math.fround;

/**
 * Python 3.12's sum() of floats: Neumaier-compensated, so a mean here is
 * bit-identical to the desktop's (a plain loop can differ in the last digit).
 */
export function pySum(values) {
  let total = 0.0, c = 0.0;
  for (const x of values) {
    const t = total + x;
    if (Math.abs(total) >= Math.abs(x)) c += (total - t) + x;
    else c += (x - t) + total;
    total = t;
  }
  return c !== 0 && Number.isFinite(c) ? total + c : total;
}

export class SwingDetector {
  constructor(mode = "full", frameWidth = 1280.0, frameHeight = 720.0) {
    this.frameWidth = Number(frameWidth);
    this.frameHeight = Number(frameHeight);
    this.mode = mode;
    this.reset();
  }

  // --- lifecycle ---

  /** Forget everything (on resume, between shots, on a mode change). */
  reset() {
    this.stage = Stage.NO_GRIP;
    this._t = null;
    this._frame = null; // [shoulder mid x, y, sw] px
    this._frameT = 0.0;
    this._obs = []; // [first seen, last seen, u, h] distinct grip observations
    this._gripPx = null;
    this._lostSince = null;
    this._oneHandSince = null;
    this._hold = []; // [t, u, h] during ADDRESSING
    this._low = [0.0, 0.0];
    this._holdStart = 0.0;
    this.address = null; // [u, h] at READY
    this.armed = false;
    this.peakH = -Infinity;
    this.peakU = 0.0;
    this.side = 0;
    this.power = 0.0;
    this.locked = false;
    this._swingT = 0.0;
    this._downT = 0.0;
    this.hint = null;
    this._hintUntil = -Infinity;
    this.tooClose = false;
    this.visible = false;
    this.together = false;
    this.handsSeen = 0;
    this._period = DEFAULT_PERIOD;
    this._intervals = [];
    this._lastNew = null;
    this.lastEvent = null;
    this._pu = null;
    this._puPrev = null;
    this._puttOut = 0;
  }

  setMode(mode) {
    if (mode !== this.mode) {
      this.mode = mode;
      this.reset();
    }
  }

  // --- read-only state for the HUD ---

  get ready() { return this.stage === Stage.READY; }

  get swinging() { return this.stage === Stage.BACKSWING || this.stage === Stage.DOWNSWING; }

  get holdProgress() {
    if (this.stage === Stage.ADDRESSING && this._hold.length) {
      return Math.min(1.0, (this._hold[this._hold.length - 1][0] - this._holdStart) / ADDRESS_HOLD);
    }
    return (this.stage === Stage.READY || this.swinging) && this.armed ? 1.0 : 0.0;
  }

  /** The grip in frame pixels (for the camera window). */
  get grip() { return this._gripPx; }

  /** [shoulder midpoint x, y, shoulder width] in px, as the detector uses it. */
  get bodyFrame() { return this._frame; }

  /** The newest grip in the body frame [u, h], sw. */
  get gripBody() {
    const o = this._obs[this._obs.length - 1];
    return o ? [o[2], o[3]] : null;
  }

  /** How far round the swing is, for Lenny: 0 address, 1 top of a full backswing, beyond 1 overswing. */
  get swingAngle() {
    const last = this._obs[this._obs.length - 1];
    if (this.mode === "putt") {
      if (this.address == null || !last) return 0.0;
      return Math.max(-1.0, Math.min(1.0, Math.abs(last[2] - this.address[0]) / PUTT_FULL));
    }
    if (this.address == null || !last || this.stage === Stage.NO_GRIP || this.stage === Stage.ADDRESSING) return 0.0;
    return Math.max(0.0, this._powerAt(last[3]));
  }

  // --- the main entry point ---

  /** Feed one (stepped) pose per game frame. Returns a swing event on the frame one is recognised, else null. */
  update(pose, t) {
    if (this._t != null && t <= this._t) return null;
    this._t = t;
    if (t >= this._hintUntil) this.hint = null;
    const grip = this._read(pose, t);
    if (grip == null) {
      this._lose(t);
      return null;
    }
    this._lostSince = null;
    const [u, h, isNew] = grip;
    if (!isNew) return this._held(t);
    return this._advance(t, u, h);
  }

  // --- reading the pose ---

  _read(pose, t) {
    this.visible = false;
    this.together = false;
    this.handsSeen = 0;
    if (pose == null) return null;
    const ls = pose.point(KP.left_shoulder, MIN_CONF);
    const rs = pose.point(KP.right_shoulder, MIN_CONF);
    const frozen = this.swinging;
    if (ls != null && rs != null) {
      this.visible = true;
      const mx = f32(ls[0] + rs[0]) / 2, my = f32(ls[1] + rs[1]) / 2;
      const sw = Math.max(1.0, Math.hypot(f32(ls[0] - rs[0]), f32(ls[1] - rs[1])));
      const room = (this.frameHeight - my) / sw;
      this.tooClose = room < (this.tooClose ? TOO_CLOSE_OFF : TOO_CLOSE_ON);
      if (!frozen) {
        if (this._frame == null) this._frame = [mx, my, sw];
        else {
          const a = 1.0 - Math.exp(-Math.max(0.0, t - this._frameT) / FRAME_TAU);
          const [fx, fy, fs] = this._frame;
          this._frame = [fx + (mx - fx) * a, fy + (my - fy) * a, fs + (sw - fs) * a];
        }
        this._frameT = t;
      }
    } else if (this._frame == null || !frozen) return null;
    const pl = pose.point(KP.left_wrist, MIN_CONF);
    const pr = pose.point(KP.right_wrist, MIN_CONF);
    this.handsSeen = (pl != null) + (pr != null);
    const [fx, fy, sw] = this._frame;
    let gx, gy;
    if (pl != null && pr != null) {
      this.together = Math.hypot(pl[0] - pr[0], pl[1] - pr[1]) <= TOGETHER * sw;
      gx = (pl[0] + pr[0]) / 2;
      gy = (pl[1] + pr[1]) / 2;
      if (!this.together) {
        if (frozen) return null; // hands flew apart mid-swing: treat as lost
        this._gripPx = [gx, gy];
        this._obs = [];
        this._toNoGrip(t);
        return null;
      }
      this._oneHandSince = null;
    } else if ((pl != null || pr != null) && frozen) {
      // Mid-swing a wrist can drop out for a frame or two: carry on with the one that is seen, briefly.
      if (this._oneHandSince == null) this._oneHandSince = t;
      if (t - this._oneHandSince > ONE_HAND_KEEP) return null;
      [gx, gy] = pl != null ? pl : pr;
    } else return null;
    this._gripPx = [gx, gy];
    const u = (gx - fx) / sw, h = (fy - gy) / sw;
    const isNew = this._observe(t, u, h);
    return [u, h, isNew];
  }

  /** Record a distinct observation; false for a repeated (held) frame. */
  _observe(t, u, h) {
    if (this._obs.length) {
      const last = this._obs[this._obs.length - 1];
      const sw = this._frame[2];
      const du = u - last[2], dh = h - last[3];
      if (Math.hypot(du, dh) * sw < HOLD_EPS) {
        last[1] = t;
        return false;
      }
      if (Math.hypot(du, dh) * sw > TELEPORT * this.frameWidth) {
        // A glitch: lost, then found here. Mid-swing the found sample only starts new history.
        this._obs = [];
        if (this.swinging) {
          this._obs.push([t, t, u, h]);
          return false;
        }
      }
    }
    if (this._lastNew != null) {
      const gap = t - this._lastNew;
      if (gap > 0.0 && gap < MAX_GAP) {
        this._intervals.push(gap);
        if (this._intervals.length > 12) this._intervals.shift();
        const s = [...this._intervals].sort((a, b) => a - b);
        this._period = s[Math.floor(s.length / 2)];
      }
    }
    this._lastNew = t;
    this._obs.push([t, t, u, h]);
    while (this._obs.length > 2 && this._obs[0][1] < t - 1.0) this._obs.shift();
    return true;
  }

  /** Grip speed (sw/s) over SPEED_WINDOW ending at the newest observation. */
  _speed() {
    const obs = this._obs;
    if (obs.length < 2) return 0.0;
    const [tNow, , u1, h1] = obs[obs.length - 1];
    let ref = null;
    for (let i = obs.length - 2; i >= 0; i--) {
      const [first, last, u0, h0] = obs[i];
      const when = Math.max(first, last - this._period);
      if (ref == null || when >= tNow - SPEED_WINDOW - 1e-9) ref = [when, u0, h0];
      else break;
    }
    const [when, u0, h0] = ref;
    const dt = tNow - when;
    if (dt <= 1e-6 || dt > MAX_GAP) return 0.0;
    return Math.hypot(u1 - u0, h1 - h0) / dt;
  }

  // --- stages ---

  _say(hint, t) {
    this.hint = hint;
    this._hintUntil = t + HINT_SECONDS;
  }

  _toNoGrip(_t) {
    this.stage = Stage.NO_GRIP;
    this._hold = [];
    this.armed = false;
    this.locked = false;
    this.power = 0.0;
  }

  _lose(t) {
    if (this._lostSince == null) this._lostSince = t;
    if (this.swinging && t - this._lostSince <= LOSS_KEEP) return; // a short dropout mid-swing: keep going
    if (this.stage !== Stage.NO_GRIP) {
      this._obs = [];
      this._toNoGrip(t);
    }
  }

  _powerAt(h) {
    if (this.mode === "putt" || this.address == null) return 0.0;
    const span = Math.max(0.5, FULL_H - this.address[1]);
    return Math.min(OVERSWING_MAX, Math.max(0.0, (h - this.address[1]) / span));
  }

  /** A repeated frame: time passes, nothing moved. */
  _held(t) {
    if (this.stage === Stage.ADDRESSING && this._hold.length) this._checkHold(t);
    else if (this.stage === Stage.DOWNSWING
      && t - this._downT > (this.mode === "putt" ? PUTT_STROKE : SWING_TIMEOUT)) this._cancel(t);
    return null;
  }

  _addressable(u, h) {
    return h <= ADDRESS_MAX_H && Math.abs(u) <= ADDRESS_MAX_U;
  }

  _advance(t, u, h) {
    if (this.mode === "putt") {
      // A putt moves the hands only ~0.1-1.2 sw, so jitter matters: the mean of the last PUTT_SMOOTH observations.
      const recent = this._obs.slice(-PUTT_SMOOTH).filter((o) => o[0] >= t - 0.25).map((o) => o[2]);
      this._puPrev = this._pu;
      u = this._pu = recent.length ? pySum(recent) / recent.length : u;
    }
    const st = this.stage;
    if (st === Stage.NO_GRIP) {
      if (this._addressable(u, h)) this._beginHold(t, u, h);
      return null;
    }
    if (st === Stage.ADDRESSING) {
      if (!this._addressable(u, h)) {
        // Hands rising from an unsteady address: a practice swing.
        if (this._hold.length && this._startsSwing(u, h, this._holdRef())) {
          this._beginSwing(t, false, this._holdRef());
          return this._advance(t, u, h);
        }
        this._toNoGrip(t);
        return null;
      }
      this._hold.push([t, u, h]);
      if (h < this._low[1]) this._low = [u, h];
      const ref = this._holdRef();
      if (this._startsSwing(u, h, ref)) {
        this._beginSwing(t, false, ref);
        return null;
      }
      let wander = 0;
      for (const p of this._hold) wander = Math.max(wander, Math.abs(u - p[1]) + Math.abs(h - p[2]));
      if (wander > STEADY * 1.5) {
        // Still moving: restart the hold from here.
        while (this._hold.length > 1
          && Math.abs(u - this._hold[0][1]) + Math.abs(h - this._hold[0][2]) > STEADY * 1.5) this._hold.shift();
        this._holdStart = this._hold[0][0];
      }
      this._checkHold(t);
      return null;
    }
    if (st === Stage.READY) {
      if (this._startsSwing(u, h, this.address)) {
        this._beginSwing(t, true, this.address);
        return this._advance(t, u, h);
      }
      if (h < this.address[1] - 0.35) this._beginHold(t, u, h); // hands dropped: address again
      return null;
    }
    if (st === Stage.BACKSWING) return this._backswing(t, u, h);
    if (st === Stage.DOWNSWING) return this._downswing(t, u, h);
    return null;
  }

  /** Where a practice swing is measured from: the lowest point since the grip came into the address zone. */
  _holdRef() {
    return this._low;
  }

  _beginHold(t, u, h) {
    this.stage = Stage.ADDRESSING;
    this._low = [u, h];
    this._puttOut = 0;
    this._hold = [[t, u, h]];
    this._holdStart = t;
    this.address = null;
    this.armed = false;
    this.power = 0.0;
    this.locked = false;
  }

  _checkHold(t) {
    if (t - this._holdStart >= ADDRESS_HOLD) {
      const pts = this._hold;
      this.address = [pySum(pts.map((p) => p[1])) / pts.length, pySum(pts.map((p) => p[2])) / pts.length];
      this.stage = Stage.READY;
      this.armed = true;
    }
  }

  _startsSwing(u, h, ref) {
    if (this.mode === "putt") {
      // Two observations in a row past PUTT_START: one jittery sample is not a backstroke.
      this._puttOut = Math.abs(u - ref[0]) >= PUTT_START ? this._puttOut + 1 : 0;
      return this._puttOut >= 2;
    }
    return h - ref[1] >= START_RISE || Math.abs(u - ref[0]) >= START_SIDE;
  }

  _beginSwing(t, armed, ref) {
    this.stage = Stage.BACKSWING;
    this._puttOut = 0;
    this.armed = armed;
    this.address = ref;
    this.peakH = ref[1];
    this.peakU = ref[0];
    this.side = 0;
    this.power = 0.0;
    this.locked = false;
    this._swingT = t;
    this._hold = [];
  }

  _backswing(t, u, h) {
    const [aU, aH] = this.address;
    if (this.mode === "putt") {
      const travel = Math.abs(u - aU);
      if (travel >= Math.abs(this.peakU - aU)) {
        this.peakU = u;
        this.side = u > aU ? 1 : -1;
      }
      this.power = Math.min(1.0, Math.abs(this.peakU - aU) / PUTT_FULL);
      if (travel <= Math.abs(this.peakU - aU) - PUTT_RETURN) {
        this.stage = Stage.DOWNSWING;
        this.locked = true;
        this._downT = t;
        return this._downswing(t, u, h);
      }
      return null;
    }
    if (h >= this.peakH) {
      this.peakH = h;
      this.peakU = u;
    }
    this.power = this._powerAt(Math.max(h, aH));
    if (h <= this.peakH - DOWN_DROP) {
      const off = this.peakU - aU;
      this.side = Math.abs(off) >= SIDE_MIN ? (off > 0 ? 1 : -1) : 0;
      this.power = this._powerAt(this.peakH);
      this.locked = true;
      this.stage = Stage.DOWNSWING;
      this._downT = t;
      return this._downswing(t, u, h);
    }
    return null;
  }

  _downswing(t, u1, h1) {
    const [aU, aH] = this.address;
    if (this._obs.length < 2) return null;
    let [, , u0, h0] = this._obs[this._obs.length - 2];
    const speed = this._speed();
    if (this.mode === "putt") {
      u0 = this._puPrev != null ? this._puPrev : u0;
      const crossed = this.side !== 0 && (u0 - aU) * this.side > 0 && 0 >= (u1 - aU) * this.side;
      if (crossed) {
        if (t - this._downT <= PUTT_STROKE) return this._fire(t, this.power, 1.0, speed);
        return this._cancel(t);
      }
      if (t - this._downT > PUTT_STROKE) return this._cancel(t);
      return null;
    }
    let crossed, hC;
    if (this.side !== 0) {
      crossed = (u0 - aU) * this.side > 0 && 0 >= (u1 - aU) * this.side;
      if (crossed) {
        const f = u0 !== u1 ? (u0 - aU) / (u0 - u1) : 1.0;
        hC = h0 + (h1 - h0) * f;
      } else hC = null;
    } else {
      const line = aH + CHOP_LINE;
      crossed = h0 > line && line >= h1;
      hC = crossed ? line : null;
    }
    if (crossed && hC != null && hC <= HIT_MAX_H) {
      if (speed >= HIT_SPEED) {
        const q = 1.0 - (hC - aH - CLEAN_H) / (THIN_H - CLEAN_H);
        return this._fire(t, this.power, Math.min(1.0, Math.max(0.0, q)), speed);
      }
      return this._cancel(t);
    }
    // Back down at address height without crossing, slowly: lowering, not swinging.
    if (h1 <= aH + 0.25 && speed < HIT_SPEED) return this._cancel(t);
    if (t - this._downT > SWING_TIMEOUT) return this._cancel(t);
    return null;
  }

  _fire(t, power, quality, speed) {
    const ev = swingEvent(t, power, quality, this.side, speed, !this.armed);
    this.lastEvent = ev;
    // After a swing the player must address again (and the game decides whether a ball was hit).
    this.stage = Stage.NO_GRIP;
    this.armed = false;
    this.locked = true;
    this.address = null;
    this._hold = [];
    return ev;
  }

  _cancel(t) {
    if (this.armed || this.swinging) this._say("SWING THROUGH!", t);
    this.stage = Stage.NO_GRIP;
    this.armed = false;
    this.locked = false;
    this.power = 0.0;
    this.address = null;
    this._hold = [];
    return null;
  }

  // --- guidance ---

  /** What the player should do next, for the HUD (null = nothing to fix). */
  guidance() {
    if (this.hint) return this.hint;
    if (!this.visible && this._frame == null) return "STEP IN FRONT OF THE CAMERA";
    if (this.tooClose && (this.stage === Stage.NO_GRIP || this.stage === Stage.ADDRESSING)) return "STEP BACK - SHOW BOTH HANDS";
    if (this.stage === Stage.NO_GRIP) {
      if (this.handsSeen < 2) return "SHOW BOTH HANDS";
      if (!this.together) return "HOLD BOTH HANDS TOGETHER";
      return "HANDS LOW, IN FRONT OF YOU";
    }
    if (this.stage === Stage.ADDRESSING) return "HOLD STILL";
    return null;
  }
}
