// Dance core: pose angles, forward kinematics, measuring players, similarity, charts
// (port of tests/test_dance_pose.py).

import { test } from "node:test";
import assert from "node:assert/strict";
import { KP } from "../js/core/pose.js";
import { ChartError, idleGroove, parseMoves, parseSong } from "../js/games/dance/chart.js";
import {
  FULL_BODY, LEGS_WEIGHT, REST, SQUAT_BEND, angleDiff, forward, keypointsFor, legTarget, measure, mirrorPose, normalizePose, pelvis, similarity,
  UNSEEN_LEGS,
} from "../js/games/dance/pose.js";

const ARMS_UP = { la: [170, 175], ra: [170, 175] };
const T_POSE = { la: [90, 90], ra: [90, 90] };
const DISCO = { ra: [150, 165], la: [20, 10], lean: 10 };
const POSES = [{}, ARMS_UP, T_POSE, DISCO, { ra: [-40, -70], la: [120, 200] }, { lean: -14, la: [60, 150] }];

const y = (kp, name) => kp.xy[2 * KP[name] + 1];
const x = (kp, name) => kp.xy[2 * KP[name]];

test("angle convention", () => {
  let j = forward(ARMS_UP);
  assert.ok(j.r_wrist[1] < j.r_shoulder[1] - 1.0, "180 is straight up");
  j = forward(T_POSE);
  assert.ok(j.r_wrist[0] > j.r_shoulder[0] + 1.0, "90 on the right arm points screen-right");
  assert.ok(j.l_wrist[0] < j.l_shoulder[0] - 1.0, "and screen-left on the left arm");
  j = forward({});
  assert.ok(j.r_wrist[1] > j.r_shoulder[1] + 0.9, "rest hangs down");
  assert.ok(j.head[1] < j.neck[1] && j.neck[1] < 0, "head on top, origin at the hips");
});

test("a perfect player measures back to the target", () => {
  for (const pose of POSES) {
    for (const mirrored of [true, false]) {
      const m = measure(keypointsFor(pose, { mirrored }), mirrored);
      const full = normalizePose(pose);
      for (const s of ["l", "r"]) {
        assert.ok(angleDiff(m.arms[s].upper, full[`${s}a`][0]) < 1.0, JSON.stringify(pose));
        assert.ok(angleDiff(m.arms[s].fore, full[`${s}a`][1]) < 1.0, JSON.stringify(pose));
      }
      assert.ok(Math.abs(m.lean - full.lean) < 1.0);
      assert.ok(similarity(pose, m)[0] > 0.99);
    }
  }
});

test("screen-right is the player's real right arm on a mirrored feed", () => {
  // The coach's screen-right arm is copied with the player's real right arm,
  // which a mirrored feed labels left_* (AGENTS.md section 6).
  const kp = keypointsFor({ ra: [170, 175] }, { mirrored: true });
  assert.ok(y(kp, "left_wrist") < y(kp, "left_shoulder"), "left_* is the raised one");
  assert.ok(x(kp, "left_wrist") > x(kp, "right_wrist"), "and it is on the screen's right");
});

test("without elbows the reach direction is judged", () => {
  const m = measure(keypointsFor(DISCO, { visible: ["shoulder", "wrist", "hip"] }));
  assert.ok(m.arms.r.upper == null && m.arms.r.reach != null);
  assert.ok(similarity(DISCO, m)[0] > 0.97);
  assert.ok(similarity(ARMS_UP, m)[0] < 0.6);
});

test("similarity discriminates", () => {
  const up = measure(keypointsFor(ARMS_UP));
  assert.ok(similarity(ARMS_UP, up)[0] > 0.99);
  assert.equal(similarity({}, up)[0], 0.0, "arms down vs arms up shares nothing");
  assert.ok(similarity(T_POSE, up)[0] < 0.2);
  assert.ok(similarity(mirrorPose(DISCO), measure(keypointsFor(DISCO)))[0] < 0.5, "the wrong side is wrong");
});

test("no shoulders or no arms is unseen, not a miss", () => {
  assert.equal(measure(null), null);
  const m = measure(keypointsFor(ARMS_UP, { visible: ["shoulder"] }));
  assert.deepEqual(similarity(ARMS_UP, m), [null, {}]);
});

test("hips out of frame still measure arms", () => {
  const m = measure(keypointsFor(DISCO, { centrePx: [640, 520], shoulderPx: 260 })); // close: hips below the frame
  assert.ok(!m.hasHips);
  assert.ok(similarity(DISCO, m)[0] > 0.9);
});

test("mirrorPose round-trips", () => {
  assert.deepEqual(normalizePose(mirrorPose(mirrorPose(DISCO))), normalizePose(DISCO));
});

test("normalizePose rejects bad keys", () => {
  assert.throws(() => normalizePose({ arm: [1, 2] }), /unknown pose key 'arm'/);
  assert.throws(() => normalizePose({ la: [1] }), /'la' must be \[upper, lower\] angles in degrees, got \[1\]/);
  assert.throws(() => normalizePose({ squat: 2 }), /'squat' must be 0..1, got 2.0/);
});

test("idle groove is a valid pose", () => {
  normalizePose(idleGroove(3.3));
});

// --- charts ---

const LIB = parseMoves({
  point_r: { label: "Point", beats: 2, keyframes: [{ beat: 0, pose: DISCO }, { beat: 1, pose: {} }], judge: [0] },
  point_l: { mirror_of: "point_r" },
  hold: { beats: 4, keyframes: [{ beat: 0, pose: ARMS_UP }] },
}, "test");

const song = (moves, extra = {}) => parseSong({ title: "T", bpm: 120, duration: 20, moves, ...extra }, "s", LIB);

test("song timeline", () => {
  const s = song([{ move: "point_r", beat: 4 }, { move: "point_l", beat: 6, gold: true }, { move: "hold", beat: 8 }]);
  const jp = s.judgePoints();
  assert.deepEqual(jp.map((p) => p.beat), [4, 6, 8]);
  assert.ok(jp[1].gold && jp[1].weight === 2.0);
  assert.ok(Math.abs(jp[0].time - 2.0) < 1e-12);
  assert.deepEqual(normalizePose(jp[1].pose), normalizePose(mirrorPose(DISCO)));
  assert.deepEqual(s.coachPose(4.0), jp[0].pose, "the coach hits the keyframe on its beat");
  assert.deepEqual(s.coachPose(4.1), jp[0].pose, "and holds it");
  assert.deepEqual(s.pictograms().map((p) => p.beat), [4, 6, 8]);
  assert.equal(LIB.get("hold").label, "Hold", "a default label is the name in title case");
});

test("every judged keyframe has its own card, with that pose", () => {
  // A move with two judged poses shows two cards, each on its own beat with the
  // pose judged there, under the move's label (QA: "Point Up" was labelled over
  // the move's first, horizontal pose and the raised arm had no card).
  const lib = parseMoves({ two: { label: "Point Up", beats: 2, pictogram: 1, judge: [0, 1],
    keyframes: [{ beat: 0, pose: T_POSE }, { beat: 1, pose: ARMS_UP }] } }, "t");
  const s = parseSong({ title: "T", bpm: 120, duration: 20, moves: [{ move: "two", beat: 4, gold: true }] }, "s", lib);
  const cards = s.pictograms();
  assert.deepEqual(cards.map((c) => c.beat), s.judgePoints().map((p) => p.beat));
  assert.deepEqual(cards.map((c) => c.beat), [4, 5]);
  assert.deepEqual(cards.map((c) => normalizePose(c.pose)), [normalizePose(T_POSE), normalizePose(ARMS_UP)]);
  assert.ok(cards.every((c) => c.label === "Point Up" && c.gold));
});

test("bad charts fail with a clear message", () => {
  for (const [moves, message] of [
    [[{ move: "nope", beat: 0 }], /song 's': chart entry 0 at beat 0.0: unknown move 'nope'/],
    [[{ move: "hold", beat: 0 }, { move: "hold", beat: 2 }], /move 'hold' at beat 2.0 overlaps the previous move \(ends at 4.0\)/],
    [[{ move: "hold", beat: 38 }], /move 'hold' at beat 38.0 runs past the song's end \(beat 40.0\)/],
    [[{ beat: 1 }], /chart entry 0 needs 'move' and 'beat'/],
  ]) {
    assert.throws(() => song(moves), (err) => err instanceof ChartError && message.test(err.message));
  }
  assert.throws(() => parseSong({ title: "T", bpm: 20, duration: 20 }, "s", LIB), /bpm must be 40-240/);
  assert.throws(() => parseSong({ bpm: 120, duration: 20 }, "s", LIB), /song.json needs 'title'/);
});

test("bad custom moves fail", () => {
  assert.throws(() => song([], { custom_moves: { x: { beats: 2, keyframes: [{ beat: 3, pose: {} }] } } }),
    /custom_moves: move 'x' keyframe 0 at beat 3.0 is outside 0..2.0/);
  assert.throws(() => song([], { custom_moves: { x: { beats: 2, keyframes: [{ beat: 0, pose: { bad: 1 } }] } } }),
    /unknown pose key/);
  assert.throws(() => parseMoves({ a: { mirror_of: "b" }, b: { mirror_of: "a" } }, "t"), /mirror each other in a loop/);
  assert.throws(() => parseMoves({ a: { mirror_of: "zz" } }, "t"), /mirrors unknown move 'zz'/);
  assert.throws(() => parseMoves({ a: { beats: 2, keyframes: [{ beat: 0 }], judge: [3] } }, "t"),
    /judges a keyframe that doesn't exist/);
  assert.throws(() => parseMoves({ constructor: { mirror_of: "toString" } }, "t"), /mirrors unknown move/,
    "move names never find Object's own properties");
});

// --- the lower body (docs/DANCE.md section 3a; as tests/test_dance_pose.py) ---

const STEP_R = { lf: [0.7, 0], rf: [0.7, 0], shift: 0.32, squat: 0.12, ra: [95, 100], la: [20, -88] };
const KNEE_R = { lf: [0.37, 0], rf: [0.2, 0.62], shift: -0.235, la: [138, 145], ra: [40, -45] };
const PLIE = { lf: [0.72, 0], rf: [0.72, 0], squat: 0.6, la: [90, 90], ra: [90, 90] };
const CROSS = { lf: [-0.06, 0], rf: [-0.06, 0], shift: 0.04, squat: 0.25, la: [168, 175], ra: [168, 175] };
const LEG_POSES = [STEP_R, mirrorPose(STEP_R), KNEE_R, PLIE, CROSS, { lf: [0.37, 0], rf: [1.25, 0.5], shift: 0.06 }];

test("the camera reads the coach's lower body back", () => {
  for (const pose of LEG_POSES) {
    for (const mirrored of [true, false]) {
      const want = legTarget(normalizePose(pose)).lower;
      const m = measure(keypointsFor(pose, { mirrored, ...FULL_BODY }), mirrored);
      assert.ok(m.seesFeet);
      for (const k of ["stance", "weight", "lift"]) assert.ok(Math.abs(m.lower[k] - want[k]) < 0.01, k);
      for (const s of ["l", "r"]) assert.ok(Math.abs(m.lower.bend[s] - want.bend[s]) < 0.5, s);
      assert.ok(similarity(pose, m)[0] > 0.99);
    }
  }
});

test("shift and squat mean what the camera reads", () => {
  for (const shift of [-0.4, 0.0, 0.25]) {
    const p = normalizePose({ lf: [0.5, 0], rf: [0.9, 0], shift });
    assert.ok(Math.abs(legTarget(p).lower.weight - (shift - (0.9 - 0.5) / 2)) < 1e-9);
    assert.ok(Math.abs(pelvis(p)[0] - shift) < 1e-12);
  }
  for (const squat of [0.2, 0.5, 0.8]) {
    assert.ok(Math.abs(legTarget(normalizePose({ squat })).lower.bend.l - SQUAT_BEND * squat) < 0.5);
    assert.ok(Math.abs(forward({ squat }).l_ankle[1] - forward({}).l_ankle[1]) < 1e-9, "the feet stay planted");
  }
});

test("legs weigh half of a leg-led move, and legs out of view aren't judged", () => {
  assert.equal(LEGS_WEIGHT, 2.0);
  const still = { ...STEP_R, lf: [...REST.lf], rf: [...REST.rf], shift: 0.0, squat: 0.0 };
  const [score, parts] = similarity(STEP_R, measure(keypointsFor(still, FULL_BODY)));
  assert.ok(parts.legs < 0.15 && score < 0.6, "perfect arms, legs standing still");
  for (const visible of [["shoulder", "elbow", "wrist", "hip"], ["shoulder", "elbow", "wrist", "hip", "knee"]]) {
    const m = measure(keypointsFor(STEP_R, { visible, ...FULL_BODY }));
    assert.ok(m.lower === null && !m.seesFeet);
    const [s2, p2] = similarity(STEP_R, m);
    assert.ok(UNSEEN_LEGS === null && !("legs" in p2) && s2 > 0.99, "feet are optional: the arms carry it");
    assert.ok(similarity(STEP_R, m, 0.5)[0] < 0.9, "a number gives unseen legs that much credit");
  }
  const up = { la: [170, 175], ra: [170, 175] };
  assert.ok(!legTarget(normalizePose(up)).active, "an arms move leaves the legs free");
});

test("old leg angles still load as feet", () => {
  const p = normalizePose({ rl: [70, 0] });
  assert.ok(!("rl" in p) && p.rf[1] > 0.5);
  assert.throws(() => normalizePose({ rl: [70, 0], rf: [0.4, 0] }), /not both/);
  assert.throws(() => normalizePose({ lf: [3.0, 0] }), /'lf' must be \[out -1\.\.2\.2, up 0\.\.1\.4\], got \[3\.0, 0\.0\]/);
});
