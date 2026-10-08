// Arm-flap detection: one flap event per wing-beat downstroke
// (port of src/kpapp/game/flappy/gesture.py - keep the two in step).
//
// The game depends on exactly this surface; keep it stable:
//   new FlapDetector(mirrored)
//   update(pose, t) -> {t, strength} | null
//   armsVisible, wing (getters), reset()
//
// **What is measured.** For each arm, the wrist's height above its own shoulder,
// in shoulder widths (`h`: 0 = shoulder line, +1 = one shoulder width above it,
// negative = below). Shoulder width is the steadiest body scale we get, so the
// same flap reads the same near the camera or far from it. Elbows and hips are
// not used: on recorded play elbows were missing in 54% of frames with a visible
// wrist, and hips are usually out of frame.
//
// **Stepped input.** Poses come from the model at 15-30 Hz but arrive here every
// game frame (~60 Hz), so motion shows up as runs of identical frames then a
// jump. Each arm keeps only *distinct* observations, stamped with the time they
// first appeared. Speed is "how far did the wrist fall from its highest point
// within the last SPEED_WINDOW", taken over those distinct observations - so a
// value that sat unchanged for 60 ms is not mistaken for one that just arrived,
// and a single model step is measured over the real time it took.
//
// **Firing.** A flap fires when the arms are *armed* and the wrists fall at
// least FIRE_DROP within SPEED_WINDOW (i.e. faster than FIRE_DROP/SPEED_WINDOW).
// Both arms are averaged; one arm alone can fire if it is the only one tracked,
// or if it moves SOLO_FACTOR times the threshold. It fires on the downstroke
// itself, ~50 ms into a brisk flap (plus up to one model period when poses
// arrive in steps), not at the bottom of it.
//
// **Leaving the frame.** On recorded play the player stands close: the shoulder
// line sits only 0.1-0.4 shoulder widths above the bottom of the frame, so arms
// at the sides are invisible and every downstroke carries the wrists out of
// frame. A wrist that vanishes while falling fast still counts as having
// completed its downstroke (`exit credit`), otherwise a short, fast flap that
// leaves the frame within one or two model steps would be missed.
//
// **Re-arming.** After a flap the detector is disarmed; an arm must then rise
// REARM_RISE above its lowest point since the flap before another can fire. This
// is the hysteresis that stops one stroke firing twice and stops jitter from
// machine-gunning. A REFRACTORY period backs it up.
//
// **Tracking loss.** Any gap in an arm's tracking ends its history; speed is
// never measured across a gap, so a wrist that reappears somewhere else cannot
// look like a fast stroke.

import { KP } from "../../core/pose.js";

// --- Thresholds. Heights/distances are in shoulder widths (sw); a shoulder
// width is ~0.35-0.4 m on an adult, so 1 sw/s ~ 0.4 m/s of wrist speed. ------

export const MIN_CONF = 0.4; // keypoint confidence to count as visible

// Speed window. Must span at least one model period, or a step could fall
// between two windows and the stroke would never be seen whole. 0.10 s covers
// models down to 10 Hz; RF-DETR runs at 15-30 Hz here.
export const SPEED_WINDOW = 0.10;

// Fall within SPEED_WINDOW needed to fire: 0.33 sw in 0.1 s = 3.3 sw/s
// (~1.2-1.3 m/s of wrist speed). Brisk flaps average 6-12 sw/s on the way down
// (2 sw in 0.15-0.3 s) and peak higher; deliberately lowering the arms over
// 1.5 s+ peaks below ~2 sw/s, and even a fairly quick 1.2 s lowering peaks at
// ~2.6. 0.33 sw is also ~10x keypoint jitter (a few px on a ~420 px shoulder
// width at play distance, ~2% of sw even far away). It sets the latency: a
// half-cosine 2 sw downstroke over 0.18 s covers 0.33 sw in ~48 ms.
export const FIRE_DROP = 0.33;
export const FIRE_SPEED = FIRE_DROP / SPEED_WINDOW; // 3.3 sw/s; also gates exit credit

// Two tracked arms are averaged, so a glitch on one wrist must be twice as big
// to fire. One arm can still carry a flap alone at 2x the threshold (0.66 sw
// in 0.1 s): a vigorous one-armed flap works, a wobble of one arm does not.
export const SOLO_FACTOR = 2.0;

// Exit credit: a wrist that vanishes while falling (at least EXIT_MIN_DROP,
// ~5x jitter, at >= EXIT_SPEED_MIN) is taken to have finished its downstroke
// out of frame if either
//  - it was falling at >= FIRE_SPEED, or
//  - it vanished from well above the lowest point wrists have recently been
//    seen at (the frame's bottom edge when standing close), i.e. it must have
//    covered that gap at >= FIRE_SPEED within one model period.
// At 15-20 Hz a 3 Hz flap may show only 2-3 samples per stroke in view, often
// just 0.1-0.15 sw of fall before the wrist is gone. Lowering the arms slowly
// out of frame is safe: the wrist vanishes right at the lowest point seen (the
// floor follows it down), so its implied speed is ~0. The credit counts as
// one arm's worth of a flap for EXIT_HOLD, so when both wrists are tracked
// both must leave (or the other must fall too): one wrist dropping out while
// the other stays up is not a flap.
export const EXIT_MIN_DROP = 0.10;
export const EXIT_SPEED_MIN = 1.0;
export const EXIT_HOLD = 0.12;
export const FLOOR_RISE = 0.05; // sw/s: how fast the remembered floor forgets old lows

// Re-arm hysteresis: rise this far above the lowest point since the last flap.
// 0.35 sw is ~10x jitter and well under the ~0.8-1.5 sw an arm travels in a
// rapid 3-4 Hz flap, but larger than the small rebound arms make at the bottom
// of a stroke, so one stroke cannot fire twice.
export const REARM_RISE = 0.35;

// Re-entry: standing close, a wrist that left the bottom of the frame is
// often visible again for only the top 0.5 sw of the next upstroke, sometimes
// just 2-3 model frames - too little to see a full REARM_RISE. A wrist that
// vanished after a flap while falling, or near/below the shoulder line
// (REENTRY_LEVEL), and then comes back REENTRY_RISE (~5x jitter) above where
// it vanished, or climbs that much once back, has made an upstroke.
export const REENTRY_LEVEL = 0.1;
export const REENTRY_RISE = 0.15;

// A flap starts from raised arms: the stroke's high point must be at least
// this high. -0.6 sw is wrists at chest height (wrists hang ~-1.3 sw at the
// sides, a horizontal arm is ~0). It keeps fidgeting with the arms down from
// counting.
export const ARM_LEVEL = -0.6;

// Fresh start (first sight, or after losing tracking): arms already up and
// tracked steadily this long are armed without having to be seen rising.
export const SETTLE = 0.15;

// An arm needs this much continuous tracking before its motion counts, so a
// wrist that has just reappeared (often misplaced for a frame) cannot fire.
export const MIN_TRACK = 0.05;

// Minimum time between flaps. Players spam at 3-4 flaps/s (250-330 ms apart);
// 0.12 s never gets in the way of that, and catches anything hysteresis misses.
export const REFRACTORY = 0.12;

// A jump faster than this between consecutive observations is a tracking
// glitch, not motion. A full 2.4 sw flap at 4 Hz peaks near 30 sw/s (~11 m/s
// at the wrist - about the limit of a human arm), which is 2 sw in one 15 Hz
// step; a fixed step limit would cut real strokes at low model rates.
export const MAX_SPEED = 40.0; // sw/s

// A wrist that moved less than this many pixels is a repeated frame, not a new
// observation. Real updates differ by more than this from jitter alone.
export const HOLD_EPS_PX = 0.5;

export const HISTORY = 0.4; // seconds of distinct observations kept per arm
export const FALL_TOL = 0.05; // sw: jitter allowed within one continuous descent
export const DEFAULT_PERIOD = 1.0 / 30.0; // model period until one has been measured
export const LOSS_KEEP = 0.5; // a tracking loss shorter than this keeps the armed state
export const SCALE_TAU = 0.5; // smoothing of shoulder width; it shrinks ~20% with arms up

// UI. Wing 0 = wrists hanging at the sides (~-1.2 sw), 1 = a little above the
// shoulder line. While the wrists are out of frame (they leave at the bottom of
// every stroke when the player stands close) arms stay "visible" for
// VIS_GRACE, and the wing eases to 0 if they were last seen low.
export const WING_DOWN = -1.2;
export const WING_UP = 0.2;
export const WING_TAU = 0.05;
export const VIS_GRACE = 0.5;
export const WING_EXIT_LEVEL = 0.3; // last seen below this when vanishing -> went down

const SIDES = [
  [KP.left_shoulder, KP.left_wrist],
  [KP.right_shoulder, KP.right_wrist],
];
const SIG_KEYS = [KP.left_shoulder, KP.right_shoulder, KP.left_wrist, KP.right_wrist];

const clip01 = (x) => Math.min(Math.max(x, 0.0), 1.0);

/**
 * One arm's recent distinct observations.
 *
 * Each entry is [first seen, last seen, wrist height above shoulder in px].
 * A repeated (held) frame only extends `last seen`.
 */
class Arm {
  constructor() {
    this.samples = [];
    this.start = -Infinity; // when this continuous track began
    this.low = null; // lowest height (px) since the last flap
    this.visible = false;
    this.exitT = -Infinity; // exit credit: when it was given...
    this.exitSpeed = 0.0; // ...and for the strength of the flap
    this.exitPeak = 0.0;
    this.goneLowT = -Infinity; // when it last vanished near/below the shoulders
    this.goneLowPx = 0.0; // ...and at what height
  }

  endTrack() {
    this.samples.length = 0;
    this.low = null;
    this.visible = false;
  }

  observe(t, risePx, scale, period) {
    this.visible = true;
    if (this.samples.length) {
      const last = this.samples[this.samples.length - 1];
      if (Math.abs(risePx - last[2]) < HOLD_EPS_PX) {
        last[1] = t; // repeated frame: nothing new
        return;
      }
      // Elapsed time since the previous model output, not since the
      // held value first appeared: a value held for a second does not
      // make a teleport plausible.
      if (Math.abs(risePx - last[2]) / scale > MAX_SPEED * Math.max(t - last[1], period)) {
        this.endTrack(); // glitch: start over from here
        this.visible = true;
      }
    }
    if (!this.samples.length) this.start = t;
    this.samples.push([t, t, risePx]);
    this.low = this.low == null ? risePx : Math.min(this.low, risePx);
    while (this.samples.length > 2 && this.samples[0][1] < t - HISTORY) this.samples.shift();
  }

  height(scale) {
    return this.samples[this.samples.length - 1][2] / scale;
  }

  /**
   * [drop, speed, peak] over SPEED_WINDOW ending at the newest sample.
   *
   * drop: fall from the highest point of the current descent within the
   *       window to now, in sw. The descent ends (looking back) where an
   *       earlier sample is lower by more than FALL_TOL, so an arm that
   *       bottomed out and is rising again reads no drop.
   * speed: that drop over the time since the high point, in sw/s.
   * peak: height of that high point, in sw.
   *
   * An observation held over several frames is dated to when it first
   * appeared - unless it was held longer than a model period, in which
   * case the arm really was there until about one period before it last
   * showed. That keeps a step from looking faster than it was, without
   * making an arm held perfectly still look like it left its spot long ago.
   */
  fall(scale, t, period) {
    const n = this.samples.length;
    const [tNow, , now] = this.samples[n - 1];
    if (t - tNow > SPEED_WINDOW) return [0.0, 0.0, now / scale]; // nothing new for a while: stopped
    let best = now, tBest = tNow, later = now;
    for (let i = n - 2; i >= 0; i--) {
      const [first, last, v] = this.samples[i];
      const when = Math.max(first, last - period);
      if (when < tNow - SPEED_WINDOW - 1e-6 || v < later - FALL_TOL * scale) break;
      later = v;
      if (v > best) {
        best = v;
        tBest = when;
      }
    }
    const drop = (best - now) / scale;
    const dt = tNow - tBest;
    const speed = dt > 1e-6 ? drop / dt : 0.0;
    return [drop, speed, best / scale];
  }
}

export class FlapDetector {
  constructor(mirrored = true) {
    // Both arms are used symmetrically, so the mirrored labelling (the
    // player's right arm comes back as left_*) does not change anything.
    this.mirrored = mirrored;
    this.reset();
  }

  /** Forget all history (called on pause/resume and when tracking is lost). */
  reset() {
    this._arms = [new Arm(), new Arm()];
    this._scale = null;
    this._t = null;
    this._armed = false;
    this._needRise = false; // true after a flap until the arms rise again
    this._lastFlap = -Infinity;
    this._lostSince = null;
    this._shoulders = false;
    this._wristSeen = -Infinity;
    this._wing = null;
    this._wingTarget = 0.0;
    this._lastH = 1.0;
    this._floor = null; // lowest wrist height seen lately, sw
    // Model update period, measured from how often the pose changes.
    this._period = DEFAULT_PERIOD;
    this._intervals = []; // last 15
    this._sig = null;
    this._sigT = 0.0;
  }

  // --- public ----------------------------------------------------------

  /**
   * Feed one pose per game frame (~60 Hz). Returns {t, strength} on the
   * frame a flap is recognised, else null. At most one event per flap.
   * strength is 0..1, how vigorous the downstroke was (visual juice only).
   */
  update(pose, t) {
    if (this._t != null && t <= this._t) return null;
    const dt = this._t == null ? 0.0 : t - this._t;
    this._t = t;

    const frame = this._read(pose);
    if (frame == null) {
      this._lose(t);
      return null;
    }
    if (this._lostSince != null) {
      if (t - this._lostSince > LOSS_KEEP) this._armed = this._needRise = false;
      this._lostSince = null;
    }

    const [lineY, width] = frame;
    this._updatePeriod(pose, t);
    this._updateScale(width, dt);
    const scale = this._scale;

    for (let k = 0; k < 2; k++) {
      const arm = this._arms[k];
      const wi = SIDES[k][1];
      const wrist = pose.point(wi, MIN_CONF);
      if (wrist == null) {
        if (arm.visible) {
          this._creditExit(arm, t, scale);
          const falling = arm.fall(scale, t, this._period)[0] >= EXIT_MIN_DROP;
          if (falling || arm.height(scale) <= REENTRY_LEVEL) {
            arm.goneLowT = t;
            arm.goneLowPx = arm.samples[arm.samples.length - 1][2];
          }
          arm.endTrack();
        }
        continue;
      }
      arm.observe(t, lineY[k] - wrist[1], scale, this._period);
      const h = arm.height(scale);
      this._floor = this._floor == null ? h : Math.min(h, this._floor + FLOOR_RISE * dt);
    }

    const visible = this._arms.filter((a) => a.visible);
    if (visible.length) this._wristSeen = t;
    this._updateWing(visible, scale, dt);

    this._updateArming(visible, t, scale);
    return this._maybeFire(t, scale);
  }

  /**
   * True when enough of the arms is tracked to detect flaps at all.
   *
   * Shoulders plus at least one wrist, with a VIS_GRACE allowance: when the
   * player stands close the wrists leave the bottom of the frame on every
   * downstroke, and that is still flapping, not "can't see you".
   */
  get armsVisible() {
    if (!this._shoulders || this._t == null) return false;
    return this._t - this._wristSeen <= VIS_GRACE;
  }

  /**
   * Current arm height for UI feedback: ~0 arms down at sides, ~1 arms
   * raised to shoulder height or above. null if not visible.
   */
  get wing() {
    return this.armsVisible ? this._wing : null;
  }

  // --- internals -------------------------------------------------------

  /**
   * [shoulder line y per side (left, right), shoulder width px]; null if lost.
   *
   * With one shoulder missing, its partner's height stands in for it and
   * the remembered width is used - only if a width has been measured.
   */
  _read(pose) {
    if (pose == null) return null;
    const ls = pose.point(KP.left_shoulder, MIN_CONF);
    const rs = pose.point(KP.right_shoulder, MIN_CONF);
    let ly, ry, width;
    if (ls != null && rs != null) {
      width = Math.hypot(ls[0] - rs[0], ls[1] - rs[1]);
      if (width < 8.0) return null;
      ly = ls[1];
      ry = rs[1];
    } else if (this._scale != null && (ls != null || rs != null)) {
      const only = ls != null ? ls : rs;
      ly = ry = only[1];
      width = 0.0; // no new measurement
    } else {
      return null;
    }
    return [[ly, ry], width];
  }

  /**
   * Median interval between changes of the pose: the model's period
   * (~0.033-0.067 s), or one game frame if poses are fresh every frame.
   */
  _updatePeriod(pose, t) {
    const sig = new Float64Array(8);
    SIG_KEYS.forEach((i, j) => {
      sig[2 * j] = pose.x(i);
      sig[2 * j + 1] = pose.y(i);
    });
    if (this._sig != null) {
      let maxd = 0;
      for (let j = 0; j < 8; j++) maxd = Math.max(maxd, Math.abs(sig[j] - this._sig[j]));
      if (maxd < HOLD_EPS_PX) return;
      if (t - this._sigT <= 0.2) {
        this._intervals.push(t - this._sigT);
        if (this._intervals.length > 15) this._intervals.shift();
        const ordered = [...this._intervals].sort((a, b) => a - b);
        const median = ordered[Math.floor(ordered.length / 2)];
        this._period = Math.min(Math.max(median, 1.0 / 120.0), SPEED_WINDOW);
      }
    }
    this._sig = sig;
    this._sigT = t;
  }

  _updateScale(width, dt) {
    if (width <= 0.0) return;
    if (this._scale == null) {
      this._scale = width;
      return;
    }
    // Turning side-on collapses the measured width; do not follow that.
    width = Math.min(Math.max(width, 0.6 * this._scale), 1.6 * this._scale);
    const a = dt > 0 ? 1.0 - Math.exp(-dt / SCALE_TAU) : 0.0;
    this._scale += a * (width - this._scale);
  }

  /**
   * Pose or shoulders gone. Histories end now so no speed is ever
   * measured across the gap; the armed state survives a brief loss.
   */
  _lose(t) {
    for (const arm of this._arms) {
      arm.endTrack();
      arm.exitT = -Infinity;
    }
    this._shoulders = false;
    this._wing = null;
    if (this._lostSince == null) this._lostSince = t;
    if (t - this._lostSince > LOSS_KEEP) this._scale = null;
  }

  /** A wrist vanished: if it was falling fast, it finished its stroke out of frame. */
  _creditExit(arm, t, scale) {
    if (t - arm.start < MIN_TRACK || arm.samples.length < 2) return;
    const [drop, speed, peak] = arm.fall(scale, t, this._period);
    if (drop < EXIT_MIN_DROP || peak < ARM_LEVEL || speed < EXIT_SPEED_MIN) return;
    const lastSeenH = arm.height(scale);
    const floor = this._floor != null ? this._floor : lastSeenH;
    const gapT = Math.max(t - arm.samples[arm.samples.length - 1][0], this._period);
    const implied = (lastSeenH - floor) / gapT;
    if (speed >= FIRE_SPEED || implied >= FIRE_SPEED) {
      arm.exitT = t;
      arm.exitSpeed = Math.max(speed, implied);
      arm.exitPeak = peak;
    }
  }

  _updateArming(visible, t, scale) {
    if (this._armed) return;
    for (const arm of visible) {
      const h = arm.height(scale);
      if (h < ARM_LEVEL) continue;
      // Back in view after leaving low since the last flap: it was at
      // least that low, and a shorter climb proves the upstroke.
      const reentered = arm.start > this._lastFlap && arm.goneLowT >= this._lastFlap;
      let low = arm.low;
      if (reentered && low != null) low = Math.min(low, arm.goneLowPx);
      const need = reentered ? REENTRY_RISE : REARM_RISE;
      const rose = low != null && (arm.samples[arm.samples.length - 1][2] - low) / scale >= need;
      const settled = !this._needRise && t - arm.start >= SETTLE;
      if (rose || settled) {
        this._armed = true;
        this._needRise = false;
        return;
      }
    }
  }

  _maybeFire(t, scale) {
    const evidence = [];
    const speeds = [];
    const peaks = [];
    for (const arm of this._arms) {
      if (arm.visible) {
        if (t - arm.start < MIN_TRACK) continue;
        const [drop, speed, peak] = arm.fall(scale, t, this._period);
        const e = peak >= ARM_LEVEL ? drop / FIRE_DROP : 0.0;
        evidence.push(e);
        if (e > 0) {
          speeds.push(speed);
          peaks.push(peak);
        }
      } else if (t - arm.exitT <= EXIT_HOLD) {
        evidence.push(1.0); // one arm's worth of a flap
        speeds.push(arm.exitSpeed);
        peaks.push(arm.exitPeak);
      }
    }
    if (!evidence.length || !this._armed || t - this._lastFlap < REFRACTORY) return null;
    const mean = evidence.reduce((a, b) => a + b, 0) / evidence.length;
    if (mean < 1.0 && Math.max(...evidence) < SOLO_FACTOR) return null;

    this._armed = false;
    this._needRise = true;
    this._lastFlap = t;
    for (const arm of this._arms) {
      arm.exitT = -Infinity;
      arm.low = arm.visible ? arm.samples[arm.samples.length - 1][2] : null;
    }
    return {
      t,
      strength: strength(speeds.length ? Math.max(...speeds) : FIRE_SPEED,
        peaks.length ? Math.max(...peaks) : 0.0),
    };
  }

  _updateWing(visible, scale, dt) {
    this._shoulders = true;
    if (visible.length) {
      const h = visible.reduce((s, a) => s + a.height(scale), 0) / visible.length;
      this._wingTarget = clip01((h - WING_DOWN) / (WING_UP - WING_DOWN));
      this._lastH = h;
    } else if (this._lastH < WING_EXIT_LEVEL) {
      this._wingTarget = 0.0; // wrists left the bottom of the frame
    }
    if (this._wing == null || dt <= 0.0) {
      this._wing = this._wingTarget;
    } else {
      const a = 1.0 - Math.exp(-dt / WING_TAU);
      this._wing += a * (this._wingTarget - this._wing);
    }
  }
}

/**
 * Mostly how fast the wrists were falling when the flap fired (3 sw/s is
 * the firing threshold, 12 sw/s a hard flap), plus how high the stroke
 * started (shoulder line = 0, a full sw above = 1).
 */
export function strength(speed, peak) {
  const sSpeed = clip01((speed - 3.0) / 9.0);
  const sAmp = clip01((peak + 0.3) / 1.3);
  return clip01(0.15 + 0.6 * sSpeed + 0.25 * sAmp);
}
