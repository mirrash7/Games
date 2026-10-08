// Arm-flap detection on synthetic, realistically awkward pose streams
// (port of tests/test_flappy_gesture.py).
//
// Heights are in shoulder widths (sw) above the shoulder line: 0 = shoulder
// height, ~-1.3 = wrists hanging at the sides. Poses are fed at a 60 Hz game
// rate; the model is simulated at 15-30 Hz by holding each pose until the next
// model update, which is how motion actually arrives (steps, not a ramp).

import { test } from "node:test";
import assert from "node:assert/strict";
import { KP, Pose } from "../js/core/pose.js";
import { Rng } from "../js/core/random.js";
import { FlapDetector } from "../js/games/flappy/gesture.js";
import { PoseExtrapolator } from "../js/core/extrapolate.js";

const GAME_HZ = 60.0;
const TOP = 0.7, BOTTOM = -1.3; // a full flap: wrists above the shoulders -> at the sides

// --- synthetic input ------------------------------------------------------

/**
 * Shoulders a shoulder-width `sw` apart; each wrist `h` sw above its shoulder
 * (null = wrist not visible). Elbows and hips are never visible.
 */
function makePose(hl, hr, { sw = 420.0, center = [640.0, 560.0], noise = 0.0, rng = null,
  shoulders = true } = {}) {
  rng = rng ?? new Rng(0);
  const xy = new Float32Array(34);
  const c = new Float32Array(17);
  const [cx, cy] = center;
  const pts = {};
  if (shoulders) {
    pts.left_shoulder = [cx + sw / 2, cy];
    pts.right_shoulder = [cx - sw / 2, cy];
  }
  for (const [name, side, h] of [["left_wrist", 1, hl], ["right_wrist", -1, hr]]) {
    if (h != null) pts[name] = [cx + side * 0.9 * sw, cy - h * sw];
  }
  pts.nose = [cx, cy - 0.8 * sw];
  for (const [name, p] of Object.entries(pts)) {
    const i = KP[name];
    xy[2 * i] = p[0] + (noise ? rng.gauss(0, noise) : 0);
    xy[2 * i + 1] = p[1] + (noise ? rng.gauss(0, noise) : 0);
    c[i] = 0.9;
  }
  c[KP.left_elbow] = c[KP.right_elbow] = 0.1;
  return new Pose(xy, c, 0.9);
}

/** h(t) through [t, h] keyframes with cosine easing between them. */
function keyframes(...kf) {
  return (t) => {
    if (t <= kf[0][0]) return kf[0][1];
    for (let i = 0; i + 1 < kf.length; i++) {
      const [t0, h0] = kf[i], [t1, h1] = kf[i + 1];
      if (t <= t1) {
        const u = (t - t0) / (t1 - t0);
        return h0 + (h1 - h0) * (1 - Math.cos(Math.PI * u)) / 2;
      }
    }
    return kf[kf.length - 1][1];
  };
}

/** Arms raised and held from t=0, then `n` sinusoidal flaps from `start`. Returns [h(t), downstroke starts]. */
function wingbeats(start, n, hz, top = TOP, bottom = BOTTOM) {
  const period = 1.0 / hz;
  const mid = (top + bottom) / 2, amp = (top - bottom) / 2;
  const end = start + n * period;
  const h = (t) => (t < start || t >= end ? top : mid + amp * Math.cos(2 * Math.PI * (t - start) / period));
  return [h, Array.from({ length: n }, (_, k) => start + k * period)];
}

/**
 * Times at which the model publishes a pose. jitter=0.3 spreads each interval
 * over +-30% of the nominal period, as real inference does.
 */
function modelClock(modelHz, phase, jitter, duration, rng) {
  const times = [];
  let t = phase - 1.0 / modelHz;
  while (t < duration + 1.0) {
    times.push(t);
    t += (1.0 / modelHz) * (1.0 + rng.uniform(-jitter, jitter));
  }
  return times;
}

/**
 * Feed a detector and collect events.
 *
 * left/right: h(t) per arm (right defaults to left; both=false hides it).
 * modelHz: hold each pose for 1/modelHz s (stepped input); null = fresh pose
 *   every game frame. jitter: randomise the model's update interval.
 * hidden(arm, t) -> true hides that wrist; lost(t) -> true feeds pose=null.
 * frameBottom: wrists lower than this (in sw) are out of frame.
 */
function run(left, { right = null, duration = 3.0, modelHz = null, phase = 0.0, sw = 420.0,
  center = [640.0, 560.0], noise = 0.0, seed = 0, hidden = null, lost = null,
  frameBottom = null, both = true, det = null, jitter = 0.0 } = {}) {
  const rng = new Rng(seed);
  det = det ?? new FlapDetector();
  right = right ?? left;
  const events = [];
  const wings = [];
  let heldKey = null, heldPose = null;
  const clock = modelHz == null ? null : modelClock(modelHz, phase, jitter, duration, rng);
  let ci = 0;
  const frames = Math.trunc(duration * GAME_HZ);
  for (let i = 0; i < frames; i++) {
    const t = i / GAME_HZ;
    let ts = t;
    if (clock) {
      while (ci + 1 < clock.length && clock[ci + 1] < t + 1e-9) ci++;
      ts = clock[ci];
    }
    let pose;
    if (lost && lost(ts)) {
      pose = null;
    } else {
      if (ts !== heldKey) {
        let hl = left(ts), hr = both ? right(ts) : null;
        if (frameBottom != null) {
          if (hl != null && hl < frameBottom) hl = null;
          if (hr != null && hr < frameBottom) hr = null;
        }
        if (hidden) {
          if (hidden(0, ts)) hl = null;
          if (hidden(1, ts)) hr = null;
        }
        heldKey = ts;
        heldPose = makePose(hl, hr, { sw, center, noise, rng });
      }
      pose = heldPose;
    }
    const ev = det.update(pose, t);
    wings.push([t, det.wing, det.armsVisible]);
    if (ev) events.push(ev);
  }
  return { events, det, wings };
}

/**
 * Latency of each downstroke's event; null where a flap was missed.
 * Asserts every event belongs to some downstroke (no phantoms).
 */
function match(events, starts, maxLat = 0.2) {
  const lats = [];
  const used = new Set();
  for (const s of starts) {
    const hit = events.filter((e) => s <= e.t && e.t <= s + maxLat);
    lats.push(hit.length ? hit[0].t - s : null);
    if (hit.length) used.add(hit[0]);
    assert.ok(hit.length <= 1, `double fire on the stroke at ${s.toFixed(3)}: ${hit.map((e) => e.t)}`);
  }
  const stray = events.filter((e) => !used.has(e)).map((e) => e.t);
  assert.deepEqual(stray, [], `phantom flaps at ${stray}`);
  return lats;
}

/** Arms rise from the sides, pause up, one brisk downstroke at tDown. */
function singleFlap({ tDown = 1.2, down = 0.18, top = TOP, bottom = BOTTOM } = {}) {
  return keyframes([0.0, bottom], [0.3, bottom], [0.8, top], [tDown, top], [tDown + down, bottom]);
}

const allHit = (lats) => lats.every((l) => l != null);

// --- tests ----------------------------------------------------------------

test("one clean flap fires exactly once and fast", () => {
  const { events } = run(singleFlap(), { duration: 2.5 });
  assert.equal(events.length, 1);
  const lat = events[0].t - 1.2;
  assert.ok(lat > 0 && lat <= 0.06, `latency ${lat}`);
  assert.ok(events[0].strength >= 0 && events[0].strength <= 1);
});

test("stepped 15-30 Hz input still detects the flap within one model period", () => {
  for (const modelHz of [15.0, 20.0, 30.0]) {
    for (const phase of [0.0, 0.013, 0.031, 0.049]) {
      const { events } = run(singleFlap(), { duration: 2.5, modelHz, phase, noise: 2.0 });
      const [lat] = match(events, [1.2]);
      assert.ok(lat != null, `missed at ${modelHz} Hz phase ${phase}`);
      // Cannot beat the model: allow one model period on top of the 60 Hz budget.
      assert.ok(lat <= 0.06 + 1.0 / modelHz, `${modelHz} Hz phase ${phase}: ${lat}`);
    }
  }
});

test("latency across stroke speeds and input rates", () => {
  for (const down of [0.15, 0.18, 0.25]) {
    for (const hz of [null, 30.0, 15.0]) {
      const lats = [];
      for (let k = 0; k < 6; k++) {
        const phase = (k / 6) * (1 / (hz ?? 60.0));
        const { events } = run(singleFlap({ down }), { duration: 2.0, modelHz: hz, phase });
        const [lat] = match(events, [1.2]);
        assert.ok(lat != null, `missed: down ${down} hz ${hz} phase ${phase}`);
        lats.push(lat);
      }
      if (hz == null && down <= 0.18) assert.ok(Math.max(...lats) <= 0.06, `${down}: ${lats}`);
    }
  }
});

test("slowly lowering the arms does not flap", () => {
  for (const seconds of [1.2, 2.0, 3.0]) {
    for (const modelHz of [null, 15.0]) {
      const h = keyframes([0.0, TOP], [0.5, TOP], [0.5 + seconds, BOTTOM]);
      const { events } = run(h, { duration: seconds + 1.5, modelHz, noise: 3.0 });
      assert.deepEqual(events, [], `${seconds}s at ${modelHz}`);
    }
  }
});

test("still arms with noise never flap", () => {
  for (const level of [TOP, 0.0, -0.6, BOTTOM]) {
    for (const sw of [420.0, 150.0]) {
      const { events } = run(() => level, { duration: 4.0, modelHz: 20.0, sw, noise: 4.0, seed: 3 });
      assert.deepEqual(events, [], `level ${level} sw ${sw}`);
    }
  }
});

test("jitter and small wobbles around a level do not flap", () => {
  // Arms wobbling +-0.12 sw (0.24 sw peak to peak) plus pixel noise.
  for (const around of [-0.6, 0.0, 0.5]) {
    for (const hz of [2.0, 4.0, 7.0]) {
      const h = (t) => around + 0.12 * Math.sin(2 * Math.PI * hz * t);
      const { events } = run(h, { duration: 4.0, modelHz: 20.0, noise: 3.0, seed: 7 });
      assert.deepEqual(events, [], `around ${around} at ${hz} Hz`);
    }
  }
});

test("rapid flapping counts every flap (once per downstroke)", () => {
  for (const hz of [3.0, 3.5, 4.0]) {
    for (const modelHz of [null, 30.0, 15.0]) {
      for (const [top, bottom] of [[TOP, BOTTOM], [0.5, -0.5]]) {
        const n = Math.trunc(3.0 * hz);
        const [h, starts] = wingbeats(0.6, n, hz, top, bottom);
        const { events } = run(h, { duration: 0.6 + 3.0 + 0.5, modelHz, noise: 2.0, seed: 11 });
        const lats = match(events, starts);
        assert.ok(allHit(lats), `missed flaps (${hz} Hz, model ${modelHz}, ${top}/${bottom}): ${lats}`);
        assert.equal(events.length, n);
      }
    }
  }
});

test("the same flap reads the same near and far", () => {
  for (const modelHz of [null, 15.0]) {
    const near = run(singleFlap(), { modelHz, sw: 420.0, center: [640.0, 560.0] }).events;
    const far = run(singleFlap(), { modelHz, sw: 140.0, center: [600.0, 250.0] }).events;
    assert.equal(near.length, 1);
    assert.equal(far.length, 1);
    assert.ok(Math.abs(near[0].t - far[0].t) <= 1.0 / GAME_HZ + 1e-9);
    assert.ok(Math.abs(near[0].strength - far[0].strength) < 0.05);
  }
});

test("moving toward the camera does not flap", () => {
  // Scale changing (player steps in) with arms held up is not a flap.
  const det = new FlapDetector();
  for (let i = 0; i < 240; i++) {
    const t = i / GAME_HZ;
    const sw = 200.0 + 220.0 * Math.min(t / 2.0, 1.0);
    assert.equal(det.update(makePose(0.4, 0.4, { sw, center: [640, 300 + 0.6 * sw] }), t), null);
  }
});

test("one arm visible still works (solo arm)", () => {
  for (const modelHz of [null, 15.0]) {
    const [h, starts] = wingbeats(0.6, 9, 3.0);
    const { events } = run(h, { duration: 4.0, modelHz, both: false, noise: 2.0 });
    assert.ok(allHit(match(events, starts)), `model ${modelHz}`);
  }
});

test("one arm flapping while the other is held still works", () => {
  const [h, starts] = wingbeats(0.6, 6, 2.0);
  const { events } = run(h, { right: () => 0.3, duration: 4.0, modelHz: 20.0, noise: 2.0 });
  assert.ok(allHit(match(events, starts)));
});

test("a wrist dropout mid-stroke fires once, not twice", () => {
  for (const gapAt of [1.24, 1.27, 1.30]) {
    const hidden = (arm, t) => arm === 0 && gapAt <= t && t < gapAt + 0.06;
    const { events } = run(singleFlap(), { duration: 2.5, modelHz: 20.0, hidden, noise: 2.0 });
    assert.equal(events.length, 1, `gap at ${gapAt}`);
  }
});

test("wrist dropouts while still do not flap", () => {
  for (const arm of [0, 1]) {
    const hidden = (a, t) => a === arm && (t % 0.5) < 0.1;
    for (const level of [TOP, -0.3, BOTTOM]) {
      const { events } = run(() => level, { duration: 3.0, modelHz: 20.0, hidden, noise: 3.0 });
      assert.deepEqual(events, [], `arm ${arm} level ${level}`);
    }
  }
});

test("teleport guard: a glitched wrist jump is not a flap", () => {
  // One model output misplaces one wrist (0.5 sw: within the averaging;
  // 3 sw: an impossible speed), then it snaps back.
  for (const jump of [-0.5, 3.0]) {
    const left = (t) => (t >= 1.50 && t < 1.55 ? TOP - Math.abs(jump) : TOP);
    const { events } = run(left, { right: () => TOP, duration: 3.0, modelHz: 20.0, noise: 2.0 });
    assert.deepEqual(events, [], `jump ${jump}`);
  }
});

test("tracking lost then reacquired elsewhere is not a flap", () => {
  for (const gap of [0.1, 0.4, 1.0]) {
    for (const [before, after] of [[TOP, BOTTOM], [TOP, -0.2], [BOTTOM, TOP], [0.0, -0.6]]) {
      const h = (t) => (t < 1.0 ? before : after);
      const lost = (t) => 1.0 - gap <= t && t < 1.0;
      const { events } = run(h, { duration: 3.0, modelHz: 20.0, lost, noise: 2.0 });
      assert.deepEqual(events, [], `gap ${gap}: ${before} -> ${after}`);
    }
  }
});

test("shoulders lost then reacquired is not a flap", () => {
  const det = new FlapDetector();
  for (let i = 0; i < 180; i++) {
    const t = i / GAME_HZ;
    let pose;
    if (t < 1.0) pose = makePose(TOP, TOP);
    else if (t < 1.3) pose = makePose(TOP, TOP, { shoulders: false });
    else pose = makePose(-0.3, -0.3);
    assert.equal(det.update(pose, t), null, `t=${t}`);
  }
});

test("flapping works again after tracking comes back", () => {
  const lost = (t) => t >= 1.0 && t < 1.6;
  const [h, starts] = wingbeats(2.2, 4, 2.0);
  const { events } = run(h, { duration: 4.5, modelHz: 20.0, lost });
  assert.ok(allHit(match(events, starts)));
});

test("close framing: wrists leave the frame every stroke (exit credit, re-entry re-arming)", () => {
  // As recorded: shoulders ~0.25 sw above the bottom edge, so each downstroke
  // carries both wrists out of frame part-way down.
  for (const [top, modelHz] of [[0.8, null], [0.8, 30.0], [0.8, 20.0], [0.3, null], [0.3, 30.0]]) {
    const [h, starts] = wingbeats(0.6, 10, 3.0, top, -1.3);
    const { events, wings } = run(h, { duration: 4.2, modelHz, frameBottom: -0.25, noise: 2.0 });
    const lats = match(events, starts);
    assert.ok(allHit(lats), `missed (top ${top}, model ${modelHz}): ${lats}`);
    // Still "visible" while flapping, even though the wrists keep leaving.
    assert.ok(wings.filter(([t]) => t >= 0.7 && t <= 3.8).every(([, , vis]) => vis));
  }
});

test("close framing hit rate with model timing jitter: no phantoms, ~all at 30 Hz", () => {
  // At 15-20 Hz a 3-4 Hz flap leaves only 1-3 model frames per stroke in view,
  // often all on the way up, and some strokes are simply not observable.
  for (const top of [0.5, 0.8, 1.1]) {
    for (const hz of [30.0, 20.0, 15.0]) {
      const rates = [];
      for (const flapHz of [3.0, 4.0]) {
        let hits = 0, total = 0;
        for (let seed = 0; seed < 6; seed++) {
          const [h, starts] = wingbeats(0.6, Math.trunc(3 * flapHz), flapHz, top);
          const { events } = run(h, { duration: 4.2, modelHz: hz, phase: seed * 0.011,
            frameBottom: -0.25, noise: 2.0, seed, jitter: 0.3 });
          const lats = match(events, starts); // asserts: no phantoms, no doubles
          hits += lats.filter((l) => l != null).length;
          total += lats.length;
        }
        rates.push(hits / total);
      }
      if (hz >= 30.0 && top >= 0.8) assert.ok(Math.min(...rates) >= 0.97, `top ${top}: ${rates}`);
    }
  }
});

test("rapid flapping with model timing jitter", () => {
  // Wrists in view throughout; model intervals vary +-30%.
  for (const modelHz of [20.0, 15.0]) {
    for (const flapHz of [3.0, 4.0]) {
      for (let seed = 0; seed < 5; seed++) {
        const n = Math.trunc(3.0 * flapHz);
        const [h, starts] = wingbeats(0.6, n, flapHz, 1.1, -1.3);
        const { events } = run(h, { duration: 4.2, modelHz, phase: seed * 0.013, noise: 2.0,
          seed, jitter: 0.3 });
        const lats = match(events, starts);
        assert.ok(allHit(lats), `model ${modelHz} flap ${flapHz} seed ${seed}: ${lats}`);
      }
    }
  }
});

test("both wrists dropping out while held up is not a flap", () => {
  // Raise the arms (re-arming on a seen upstroke), hold, lose both wrists.
  const h = keyframes([0.0, -0.2], [0.5, -0.2], [0.8, 0.8]);
  for (const gapAt of [0.85, 1.0, 1.2]) {
    const hidden = (arm, t) => gapAt <= t && t < gapAt + 0.2;
    const { events } = run(h, { duration: 2.0, modelHz: 20.0, hidden, noise: 3.0, frameBottom: -0.25 });
    assert.deepEqual(events, [], `gap at ${gapAt}`);
  }
});

test("close framing: slow lowering out of frame does not flap", () => {
  for (const modelHz of [null, 15.0]) {
    const h = keyframes([0.0, TOP], [0.6, TOP], [2.0, BOTTOM]);
    const { events } = run(h, { duration: 3.0, modelHz, frameBottom: -0.25, noise: 3.0 });
    assert.deepEqual(events, [], `model ${modelHz}`);
  }
});

test("one arm leaving the frame while the other is held up is not a flap", () => {
  const left = keyframes([0.0, 0.2], [1.0, 0.2], [1.25, -1.3]);
  const { events } = run(left, { right: () => 0.6, duration: 2.5, modelHz: 20.0,
    frameBottom: -0.25, noise: 2.0 });
  assert.deepEqual(events, []);
});

test("needs to rise again before the next flap (re-arm hysteresis)", () => {
  // Two drops without lifting the arms in between: only the first fires.
  const h = keyframes([0.0, TOP], [0.6, TOP], [0.75, 0.1], [1.2, 0.1], [1.35, -0.5]);
  const { events } = run(h, { duration: 2.0 });
  assert.equal(events.length, 1);
});

test("refractory: two drops 80 ms apart fire once", () => {
  // Not in the Python suite: pins REFRACTORY directly. A full drop, a quick
  // rebound above the re-arm level, then a second drop inside 0.12 s.
  const det = new FlapDetector();
  const events = [];
  let t = 0;
  const feed = (h) => {
    t += 1 / GAME_HZ;
    const ev = det.update(makePose(h, h), t);
    if (ev) events.push(ev);
  };
  for (let i = 0; i < 40; i++) feed(TOP);
  // Drop 0.9 sw in 3 frames (fires), jump back up 0.5 sw (re-arms), drop again.
  for (const h of [0.4, 0.1, -0.2, 0.3, 0.0, -0.3, -0.6]) feed(h);
  assert.equal(events.length, 1, `events at ${events.map((e) => e.t)}`);
});

test("strength tracks vigour", () => {
  const brisk = run(singleFlap({ down: 0.12, top: 1.0 })).events;
  const lazy = run(singleFlap({ down: 0.40, top: 0.0 })).events;
  assert.equal(brisk.length, 1);
  assert.equal(lazy.length, 1);
  assert.ok(lazy[0].strength >= 0 && lazy[0].strength < brisk[0].strength && brisk[0].strength <= 1);
});

test("wing and visibility", () => {
  const det = new FlapDetector();
  assert.equal(det.wing, null);
  assert.equal(det.armsVisible, false);
  det.update(null, 0.0);
  assert.equal(det.wing, null);
  assert.equal(det.armsVisible, false);
  let t = 0.0;
  for (let i = 0; i < 30; i++) {
    t += 1 / GAME_HZ;
    det.update(makePose(0.5, 0.5), t);
  }
  assert.ok(det.armsVisible && det.wing != null && det.wing > 0.95);
  for (let i = 0; i < 30; i++) {
    t += 1 / GAME_HZ;
    det.update(makePose(-1.3, -1.3), t);
  }
  assert.ok(det.wing != null && det.wing < 0.1);
  det.update(makePose(null, null, { shoulders: false }), t + 0.02);
  assert.equal(det.wing, null);
  assert.equal(det.armsVisible, false);
  // Shoulders only, for longer than the grace period: not visible.
  t += 0.02;
  for (let i = 0; i < 60; i++) {
    t += 1 / GAME_HZ;
    det.update(makePose(null, null), t);
  }
  assert.equal(det.armsVisible, false);
  assert.equal(det.wing, null);
});

test("reset clears everything", () => {
  const det = new FlapDetector();
  let t = 0.0;
  for (let i = 0; i < 60; i++) {
    t += 1 / GAME_HZ;
    det.update(makePose(TOP, TOP), t);
  }
  det.reset();
  assert.equal(det.wing, null);
  assert.equal(det.armsVisible, false);
  // Arms suddenly much lower after the reset: not a flap.
  for (let i = 0; i < 60; i++) {
    t += 1 / GAME_HZ;
    assert.equal(det.update(makePose(-0.5, -0.5), t), null);
  }
});

test("time not advancing is ignored", () => {
  const det = new FlapDetector();
  det.update(makePose(TOP, TOP), 1.0);
  assert.equal(det.update(makePose(BOTTOM, BOTTOM), 1.0), null);
  assert.equal(det.update(makePose(BOTTOM, BOTTOM), 0.5), null);
});

// --- through the real pose path: model results at 15-30 Hz, extrapolated per render frame ---

/**
 * Flap `n` times (wrists +0.9 sw at the top, -1.3 sw at the bottom) with model
 * results every 1/modelHz s arriving 50 ms late, read at `renderHz`. `use`
 * picks what the detector sees: "held" (the fix) or "projected" (per-frame
 * extrapolation, what the game used to get). Returns the flaps detected.
 */
function flapThroughPipeline({ renderHz, modelHz, beat, n = 30, use = "held", seed = 1 }) {
  const rng = new Rng(seed);
  const sw = 200, shY = 330, latency = 0.05;
  const h = (t) => {
    const ph = (t % beat) / beat;
    if (ph < 0.65) return -1.3 + 2.2 * (0.5 - 0.5 * Math.cos((Math.PI * ph) / 0.65));
    return 0.9 - 2.2 * (0.5 - 0.5 * Math.cos((Math.PI * (ph - 0.65)) / 0.35));
  };
  const poseAt = (t) => {
    const xy = new Float32Array(34), c = new Float32Array(17);
    const put = (k, x, y) => {
      xy[2 * k] = x + rng.gauss(0, 2.5);
      xy[2 * k + 1] = y + rng.gauss(0, 2.5);
      c[k] = y >= 0 && y <= 720 ? 0.9 : 0.05;
    };
    put(KP.left_shoulder, 740, shY);
    put(KP.right_shoulder, 540, shY);
    put(KP.left_wrist, 860, shY - h(t) * sw);
    put(KP.right_wrist, 420, shY - h(t) * sw);
    return new Pose(xy, c);
  };
  const det = new FlapDetector(true);
  const ex = new PoseExtrapolator({ gain: 0.7, maxLead: 0.12 });
  const queue = [];
  let next = 0, flaps = 0;
  for (let t = 0; t < n * beat + 0.3; t += 1 / renderHz) {
    while (next <= t) {
      queue.push({ at: next + latency, t0: next, pose: poseAt(next) });
      next += 1 / modelHz;
    }
    while (queue.length && queue[0].at <= t) {
      const r = queue.shift();
      ex.update([r.pose], r.t0, t);
    }
    const pose = use === "held" ? ex.held[0] : ex.posesAt(t)[0];
    if (det.update(pose ?? null, t)) flaps++;
  }
  return flaps;
}

test("every flap is caught through the real pipeline, at 60 and 120 Hz render", () => {
  for (const renderHz of [60, 120]) {
    for (const modelHz of [15, 20, 25, 30]) {
      for (const beat of [0.4, 0.6]) {
        const got = flapThroughPipeline({ renderHz, modelHz, beat });
        assert.ok(got >= 30 && got <= 31, `${renderHz} Hz render, ${modelHz} Hz model, ${beat}s beat: ${got} of 30`);
      }
    }
  }
});

test("why the detector reads held poses: per-frame extrapolation loses flaps", () => {
  // A new result shows up as a jump between two render frames 8 ms apart,
  // which the teleport guard treats as a glitch. This is the "only a few
  // flaps work, then the bird falls" bug seen in the browser on a 120 Hz screen.
  const got = flapThroughPipeline({ renderHz: 120, modelHz: 25, beat: 0.6, use: "projected" });
  assert.ok(got < 15, `projected poses caught ${got} of 30`);
});
