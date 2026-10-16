// Raccoon Golf parity: the browser port replays what the desktop computed on
// fixed inputs (web/tests/fixtures/golf_parity.json, made by
// tools/make_golf_parity_fixture.py; tests/test_golf_parity.py fails when the
// fixture is stale) and must agree:
//
//   swing    every stage on every frame, every event (frame, power, quality,
//            side, speed, practice), every guidance line; power and Lenny's
//            swing angle per frame to 1e-9
//   physics  final state, resting point, landing, apex, bounces, trees, path
//            and drop point of ~100 shots to 1e-6 m (the two builds use
//            different libm's, so the last digit may differ)
//   course   lie_at on a 3 m grid over every hole (exact), green gradients and
//            aim points to 1e-9
//   game     auto-club, auto-aim, the preview's landing, HUD texts, club
//            changes, scoring and names, and whole scripted holes (water, OB,
//            a pick-up, birdies) shot by shot

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { Pose } from "../js/core/pose.js";
import { emptyControls } from "../js/core/controls.js";
import { parseCourse } from "../js/games/golf/course.js";
import * as P from "../js/games/golf/physics.js";
import { SwingDetector, swingEvent } from "../js/games/golf/swing.js";
import { Phase, RaccoonGolfGame, holePoints, holedCamera, resultName, toParText } from "../js/games/golf/game.js";
import { stubArt } from "../js/games/golf/art.js";

const FIXTURE = JSON.parse(readFileSync(new URL("./fixtures/golf_parity.json", import.meta.url), "utf8"));
const COURSE = parseCourse(JSON.parse(readFileSync(new URL("../../assets/golf/course.json", import.meta.url), "utf8")));
const STAGE_CODE = { no_grip: "N", addressing: "A", ready: "R", backswing: "B", downswing: "D" };

const close = (got, want, tol, what) => {
  assert.ok(Math.abs(got - want) <= tol, `${what}: ${got} vs ${want}`);
};
const closeArr = (got, want, tol, what) => {
  if (want == null) return assert.equal(got, null, what);
  assert.equal(got.length, want.length, what);
  want.forEach((w, i) => close(got[i], w, tol, `${what}[${i}]`));
};

function toPose(p) {
  if (p == null) return null;
  const xy = Float32Array.from(p.xy, (v) => v / 10.0);
  const conf = new Float32Array(17);
  for (const i of p.v) conf[i] = 0.9;
  return new Pose(xy, conf, 0.9);
}

// --- swing ---------------------------------------------------------------------

for (const s of FIXTURE.swing) {
  test(`swing parity: ${s.name}`, () => {
    const det = new SwingDetector(s.mode);
    const poses = s.poses.map(toPose);
    const resets = new Set(s.resets);
    const events = [], guide = [];
    let stages = "", lastGuide;
    s.frames.forEach(([t, idx], i) => {
      if (resets.has(i)) det.reset();
      const ev = det.update(idx >= 0 ? poses[idx] : null, t);
      if (ev) events.push([i, ev.t, ev.power, ev.quality, ev.side, ev.speed, ev.practice]);
      stages += STAGE_CODE[det.stage];
      close(det.power, s.power[i], 1e-9, `power at frame ${i}`);
      close(det.swingAngle, s.angle[i], 1e-9, `swing angle at frame ${i}`);
      const g = det.guidance();
      if (g !== lastGuide) {
        guide.push([i, g]);
        lastGuide = g;
      }
    });
    assert.equal(stages, s.stages, "stage on every frame");
    assert.equal(events.length, s.events.length, "events");
    events.forEach((e, k) => {
      const w = s.events[k];
      assert.equal(e[0], w[0], `event ${k} frame`);
      close(e[1], w[1], 1e-12, `event ${k} t`);
      close(e[2], w[2], 1e-9, `event ${k} power`);
      close(e[3], w[3], 1e-9, `event ${k} quality`);
      assert.equal(e[4], w[4], `event ${k} side`);
      close(e[5], w[5], 1e-7, `event ${k} speed`);
      assert.equal(e[6], w[6], `event ${k} practice`);
    });
    assert.deepEqual(guide, s.guide, "guidance lines");
    closeArr(det.address, s.end.address, 1e-9, "address");
    assert.equal(det.tooClose, s.end.too_close);
    close(det.holdProgress, s.end.hold, 1e-9, "hold progress");
    closeArr(det.bodyFrame, s.end.body_frame, 1e-9, "body frame");
  });
}

test("swing parity covers what it should", () => {
  const names = new Set(FIXTURE.swing.map((s) => s.name));
  for (const n of ["practice", "lowering", "teleport", "lost", "dropout", "reset", "overswing", "thin", "putt_slow", "still_full", "still_putt"]) {
    assert.ok(names.has(n), n);
  }
  const ev = FIXTURE.swing.flatMap((s) => s.events);
  assert.ok(ev.some((e) => e[6]), "a practice swing");
  assert.ok(ev.some((e) => e[2] >= 1.15 - 1e-9), "an overswing");
  assert.ok(ev.some((e) => e[3] < 0.3), "a thin strike");
  assert.ok(ev.some((e) => e[4] === -1) && ev.some((e) => e[4] === 1), "both sides");
});

test("swing events report overswing like SwingEvent.overswing", () => {
  assert.equal(swingEvent(0, 1.15, 1, 1, 9, false).overswing, 1);
  assert.equal(swingEvent(0, 0.9, 1, 1, 9, false).overswing, 0);
  close(swingEvent(0, 1.075, 1, 1, 9, false).overswing, 0.5, 1e-12, "half");
});

// --- physics -------------------------------------------------------------------

const TOL = 1e-6;

function checkFlight(f, want, what) {
  const hole = f.hole;
  assert.equal(f.state, want.state, `${what}: state`);
  close(f.x, want.x, TOL, `${what}: x`);
  close(f.y, want.y, TOL, `${what}: y`);
  close(f.z, want.z, TOL, `${what}: z`);
  close(f.t, want.t, 1e-9, `${what}: t`);
  assert.equal(f.bounces, want.bounces, `${what}: bounces`);
  closeArr(f.landing, want.landing, TOL, `${what}: landing`);
  close(f.apex, want.apex, TOL, `${what}: apex`);
  close(f.carry, want.carry, TOL, `${what}: carry`);
  assert.deepEqual([...f.hitTrees].sort((a, b) => a - b), want.trees, `${what}: trees`);
  assert.equal(f.path.length, want.n_path, `${what}: path length`);
  want.path.forEach((p, i) => closeArr(f.path[i * 15], p, TOL, `${what}: path ${i * 15}`));
  assert.equal(f.lie, want.lie, `${what}: lie`);
  if (want.state === "water") {
    closeArr(f.waterEntry, want.water_entry, TOL, `${what}: water entry`);
    closeArr(P.dropPoint(hole, f), want.drop, TOL, `${what}: drop point`);
  }
}

test("physics parity: every shot lands, rolls and stops where the Python's does", () => {
  const states = new Set();
  FIXTURE.physics.shots.forEach((c, i) => {
    const hole = COURSE.holes[c.hole];
    const shot = P.makeShot(P.clubByKey(c.club), c.power, c.heading, c.lie, { quality: c.quality, curve: c.curve, wind: c.wind });
    const f = P.simulate(hole, c.start, shot);
    checkFlight(f, c.result, `shot ${i} (hole ${c.hole + 1}, ${c.club} ${c.power})`);
    states.add(f.state);
  });
  assert.deepEqual([...states].sort(), ["holed", "ob", "rest", "water"]);
  assert.ok(FIXTURE.physics.shots.some((c) => c.result.trees.length), "a tree is hit");
});

test("physics parity: previews from every tee with every club", () => {
  for (const p of FIXTURE.physics.previews) {
    const hole = COURSE.holes[p.hole];
    const f = P.preview(hole, hole.tee, P.clubByKey(p.club), p.heading, "tee");
    const what = `hole ${p.hole + 1} ${p.club}`;
    closeArr(f.landing, p.landing, TOL, what);
    close(f.apex, p.apex, TOL, what);
    assert.equal(f.path.length, p.n_path, what);
    assert.equal(f.state, p.state, what);
    close(f.x, p.x, TOL, what);
    close(f.y, p.y, TOL, what);
  }
});

test("physics parity: uneven frame times step to the same end", () => {
  const s = FIXTURE.physics.stepped;
  const f = new P.Flight(COURSE.holes[0], [0, 0], P.makeShot(P.clubByKey(s.club), s.power, s.heading, s.lie, { wind: s.wind, curve: s.curve }));
  let i = 0;
  while (!f.done) {
    f.step(s.dts[i % s.dts.length]);
    i += 1;
  }
  assert.equal(i, s.steps);
  checkFlight(f, s.result, "stepped");
});

// --- course --------------------------------------------------------------------

const LIE_CODE = { ob: "o", water: "w", bunker: "b", green: "g", tee: "t", fairway: "f", rough: "r" };

test("course parity: lies, green slopes, aim points and lengths", () => {
  FIXTURE.course.forEach((c, hi) => {
    const hole = COURSE.holes[hi];
    assert.equal(hole.par, c.par);
    close(hole.length, c.length, 1e-9, `hole ${hi + 1} length`);
    const { x0, y0, step, rows } = c.grid;
    rows.forEach((row, r) => {
      let got = "";
      for (let k = 0; k < row.length; k++) got += LIE_CODE[hole.lieAt(x0 + k * step, y0 + r * step)];
      assert.equal(got, row, `hole ${hi + 1} lies, row ${r}`);
    });
    for (const [x, y, gx, gy, h] of c.gradient) {
      closeArr(hole.greenGradient(x, y), [gx, gy], 1e-12, "gradient");
      close(hole.greenHeight(x, y), h, 1e-12, "height");
    }
    for (const [x, y, reach, ax, ay] of c.aim) closeArr(hole.aimPoint(x, y, reach), [ax, ay], 1e-9, `aim from ${x},${y}`);
  });
});

// --- game ----------------------------------------------------------------------

const newGame = (seed = 1) => new RaccoonGolfGame({ seed, art: stubArt(), course: COURSE });

test("game parity: auto-club, auto-aim, previews, HUD texts and club changes", () => {
  for (const row of FIXTURE.game.aim) {
    const g = newGame();
    g._startHole(row.hole);
    g.ball = row.ball;
    g.lie = row.lie;
    g._setupShot();
    const what = `hole ${row.hole + 1} from ${row.ball.map((v) => v.toFixed(1))} (${row.lie})`;
    assert.equal(g.club.key, row.club, `${what}: club`);
    close(g.autoHeading, row.heading, 1e-12, `${what}: heading`);
    closeArr(g.preview.landing, row.landing, TOL, `${what}: landing`);
    assert.equal(g.distanceText(), row.distance, what);
    assert.equal(g.clubRangeText(), row.range, what);
    assert.equal(g.putting, row.putt, what);
    const seq = [];
    for (const step of [1, 1, 1, -1, -1, -1, -1, -1]) {
      g.changeClub(step);
      seq.push(g.club.key);
    }
    assert.deepEqual(seq, row.changes, `${what}: club changes`);
  }
});

test("game parity: scoring, result names and the to-par text", () => {
  const s = FIXTURE.game.scoring;
  for (const [strokes, par, pu, pts] of s.points) assert.equal(holePoints(strokes, par, pu), pts, `${strokes} on a par ${par}`);
  for (const [strokes, par, name] of s.names) assert.equal(resultName(strokes, par), name);
  for (const [d, t] of s.to_par) assert.equal(toParText(d), t);
});

test("game parity: the hole-out view and Lenny's spot", () => {
  for (const [pin, hd, cam, spot] of FIXTURE.game.holed) {
    const [c, sp] = holedCamera(pin, hd);
    closeArr([c.x, c.y, c.z, c.yaw, c.pitch], cam, 1e-12, "camera");
    closeArr(sp, spot, 1e-12, "spot");
  }
});

for (const [n, round] of FIXTURE.game.rounds.entries()) {
  test(`game parity: scripted hole ${n + 1} (hole ${round.hole + 1}, ${round.log.length} shots)`, () => {
    const g = newGame(5);
    g._startHole(round.hole);
    g.setWind(...round.wind);
    while (g.phase === Phase.INTRO) g.update(emptyControls(), 1 / 60);
    round.shots.forEach(([power, quality, aim, club], k) => {
      assert.equal(g.phase, Phase.AIM, `shot ${k}`);
      for (let i = 0; i < Math.abs(aim); i++) g.adjustAim(aim > 0 ? 1 : -1);
      for (let i = 0; i < Math.abs(club); i++) g.changeClub(club > 0 ? 1 : -1);
      g._hit(swingEvent(0, power, quality, 1, 9, false));
      while (g.phase === Phase.FLIGHT) g.update(emptyControls(), 1 / 60);
      const w = round.log[k], what = `shot ${k + 1}`;
      assert.equal(g.flight.shot.club.key, w.club, `${what}: club`);
      close(g.flight.shot.heading, w.heading, 1e-12, `${what}: heading`);
      assert.equal(g.phase, w.phase, `${what}: phase`);
      assert.equal(g.strokes, w.strokes, `${what}: strokes`);
      closeArr(g.ball, w.ball, TOL, `${what}: ball`);
      assert.equal(g.lie, w.lie, `${what}: lie`);
      assert.equal(g.message?.text ?? null, w.message, `${what}: message`);
      assert.equal(g.flight.state, w.flight_state, `${what}: flight`);
      close(g.flight.t, w.flight_t, 1e-9, `${what}: flight time`);
      if (w.card) {
        assert.deepEqual(g.card.map((r) => [r.strokes, r.points, r.pickedUp, r.done]), w.card, `${what}: card`);
        closeArr([g.holedTo.x, g.holedTo.y, g.holedTo.z, g.holedTo.yaw, g.holedTo.pitch], w.holed_to, 1e-9, "holed view");
        closeArr(g.celebrateSpot, w.celebrate, 1e-9, "celebrate spot");
      }
      while (g.phase === Phase.RESULT) g.update(emptyControls(), 1 / 60);
    });
    assert.equal(g.phase, round.phase);
    assert.equal(g.score, round.score);
    assert.equal(g.toPar, round.to_par);
  });
}

test("the scripted holes visit water, out of bounds, a pick-up and birdies", () => {
  const msgs = FIXTURE.game.rounds.flatMap((r) => r.log.map((e) => e.message));
  for (const m of ["WATER HAZARD", "OUT OF BOUNDS", "BIRDIE!", "PAR", "NICE ON!"]) assert.ok(msgs.includes(m), m);
  assert.ok(FIXTURE.game.rounds.some((r) => r.score === 10), "a pick-up");
});
