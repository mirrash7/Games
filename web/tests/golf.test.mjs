// Raccoon Golf in the browser (port of the behavioural half of tests/test_golf.py
// and tests/test_golf_swing.py): the swing detector on realistically awkward
// input, the game's rules, the arcade shell, the renderer and tutorial through a
// fake 2D context, and fairness bots that play whole rounds through the real
// detector. golf_parity.test.mjs checks the numbers against the Python.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { emptyControls } from "../js/core/controls.js";
import { Rng } from "../js/core/random.js";
import { parseCourse, YARDS_PER_M } from "../js/games/golf/course.js";
import * as P from "../js/games/golf/physics.js";
import { HIT_SPEED, PUTT_FULL, PUTT_START, Stage, SwingDetector, swingEvent } from "../js/games/golf/swing.js";
import {
  FASTEST_HOLE_SECONDS, KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_UP, MAX_OVER_PAR, POINTS_ACE, Phase, RaccoonGolfGame, holePoints,
} from "../js/games/golf/game.js";
import { GolfRenderer, HOLE_PANEL, CLUB_PANEL, MAP, PIP, WIND_PANEL, GAUGE_AT, GAUGE_SCALE, View, drawLayer } from "../js/games/golf/render.js";
import { GOLF_TUTORIAL } from "../js/games/golf/tutorial.js";
import { stubArt } from "../js/games/golf/art.js";
import { GAMES } from "../js/games/index.js";
import { Shell } from "../js/shell.js";
import { CENTRE, SW, arc, gripPose, held, modelClock, path, runDetector, stream, swingPath, topAngleFor } from "./golf_poses.mjs";

const COURSE = parseCourse(JSON.parse(readFileSync(new URL("../../assets/golf/course.json", import.meta.url), "utf8")));
const DT = 1 / 60;
const REAL = { modelHz: 20.0, lag: 0.15, noise: 5.0, elbowsEvery: 3 };

const newGame = (opts = {}) => new RaccoonGolfGame({ art: stubArt(), course: COURSE, ...opts });
const controls = (pose) => ({ ...emptyControls(pose), present: pose != null, stepPose: pose });

function toAim(g) {
  while (g.phase === Phase.INTRO) g.update(emptyControls(), DT);
}

function finishFlight(g) {
  let t = 0;
  while (g.phase === Phase.FLIGHT && t < 60) {
    g.update(emptyControls(), DT);
    t += DT;
  }
  assert.notEqual(g.phase, Phase.FLIGHT);
}

const hit = (g, power, quality = 1.0) => g._hit(swingEvent(g.clock, power, quality, 1, 9.0, false));

function one(events) {
  assert.equal(events.length, 1, JSON.stringify(events));
  return events[0];
}

// --- the swing ---------------------------------------------------------------

test("hands together, low and still: READY", () => {
  const det = new SwingDetector();
  let readyAt = null;
  for (const [t, pose] of stream(() => arc(0.0), 1.5, REAL)) {
    det.update(pose, t);
    if (readyAt == null && det.stage === Stage.READY) readyAt = t;
  }
  assert.ok(readyAt >= 0.4 && readyAt <= 0.8, String(readyAt));
  assert.ok(Math.abs(det.address[1] + 1.35) < 0.06);
});

for (const [why, motion, kw] of [
  ["hands apart", () => arc(0.0), { gap: 1.2 }],
  ["hands at the chest", () => [0.0, -0.4], {}],
  ["one hand hidden", () => arc(0.0), { hands: () => 1 }],
]) {
  test(`no address without a low two-handed grip: ${why}`, () => {
    const det = new SwingDetector();
    for (const [t, pose] of stream(motion, 2.0, { ...REAL, ...kw })) {
      det.update(pose, t);
      assert.notEqual(det.stage, Stage.READY, why);
    }
    assert.ok(det.guidance());
  });
}

for (const hz of [15, 20, 30]) {
  for (const side of [1, -1]) {
    test(`a full swing hits on either side (${hz} Hz, side ${side})`, () => {
      const [motion, tImpact] = swingPath({ power: 0.9, side });
      const ev = one(runDetector(new SwingDetector(), motion, 4.0, { ...REAL, modelHz: hz }));
      assert.ok(!ev.practice);
      assert.equal(ev.side, side);
      assert.ok(Math.abs(ev.power - 0.9) < 0.06, String(ev.power));
      assert.ok(ev.quality > 0.9);
      assert.ok(ev.t >= tImpact && ev.t <= tImpact + 0.15 + 2.5 / hz);
    });
  }
}

test("power rises with backswing height; overswing caps", () => {
  // A quarter-power swing crosses the ball at about HIT_SPEED, so with some
  // noise draws it reads as lowering the hands (in the Python too): seed 1 hits.
  const powers = [0.25, 0.45, 0.65, 0.85, 1.0, 1.1].map((p) =>
    one(runDetector(new SwingDetector(), swingPath({ power: p })[0], 4.0, { ...REAL, seed: 1 })).power);
  assert.deepEqual(powers, [...powers].sort((a, b) => a - b));
  assert.ok(powers[0] < 0.35 && powers[5] > 1.0);
  const ev = one(runDetector(new SwingDetector(), swingPath({ power: 1.3 })[0], 4.0, REAL));
  assert.equal(ev.overswing, 1);
});

test("a practice swing (no address first) does not count", () => {
  const top = topAngleFor(0.8);
  const motion = (t) => {
    if (t < 0.6) return [0.0, -0.4 - 0.9 * (t / 0.6)];
    if (t < 1.3) return arc((top * (t - 0.6)) / 0.7);
    if (t < 1.7) return arc(top - ((top + 1.8) * (t - 1.3)) / 0.4);
    return arc(-1.8);
  };
  assert.ok(one(runDetector(new SwingDetector(), motion, 3.0, REAL)).practice);
});

test("slowly lowering the hands is not a hit, and says SWING THROUGH!", () => {
  const top = topAngleFor(1.0);
  const motion = (t) => {
    if (t < 1.6) return arc(0.0);
    if (t < 2.4) return arc((top * (t - 1.6)) / 0.8);
    if (t < 2.6) return arc(top);
    if (t < 4.1) return arc(top * (1 - (1 - Math.cos((Math.PI * (t - 2.6)) / 1.5)) / 2));
    return arc(0.0);
  };
  const det = new SwingDetector();
  const hints = new Set();
  const events = [];
  for (const [t, pose] of stream(motion, 5.0, REAL)) {
    const ev = det.update(pose, t);
    if (ev) events.push(ev);
    if (det.hint) hints.add(det.hint);
  }
  assert.deepEqual(events, []);
  assert.ok(hints.has("SWING THROUGH!"));
  // The reason for HIT_SPEED: lowering over 1.5 s peaks under it.
  assert.ok((1.35 * top * Math.PI) / (2 * 1.5) < HIT_SPEED);
});

test("thin strike when the hands come through high; power still from the backswing", () => {
  const clean = one(runDetector(new SwingDetector(), swingPath({ power: 0.9 })[0], 4.0, REAL));
  const thin = one(runDetector(new SwingDetector(), swingPath({ power: 0.9, lowCross: 1.0 })[0], 4.0, REAL));
  assert.ok(clean.quality > 0.9 && thin.quality < 0.3);
  assert.ok(Math.abs(thin.power - clean.power) < 0.05);
});

test("tracking: a teleport is not a swing; losing the player mid-swing never hits", () => {
  const glitch = (t) => (t >= 1.5 && t < 1.55 ? [3.0, 0.5] : arc(0.0));
  assert.deepEqual(runDetector(new SwingDetector(), glitch, 3.0, { modelHz: 20, noise: 5 }), []);
  const [motion, tImpact] = swingPath({ power: 0.9 });
  assert.deepEqual(runDetector(new SwingDetector(), motion, 4.0, { ...REAL, lost: (t) => t > tImpact - 0.25 }), []);
});

test("putting: the pendulum's backstroke sets the power; a slow drift back is not a putt", () => {
  const putt = (power, side = 1) => {
    const amp = power * PUTT_FULL;
    return path([0.0, 0.0, -1.3], [1.4, 0.0, -1.3], [2.1, side * amp, -1.27], [2.3, side * amp, -1.27],
      [2.75, -side * amp * 0.8, -1.27], [4.2, -side * amp * 0.8, -1.27]);
  };
  for (const side of [1, -1]) {
    const got = [0.15, 0.35, 0.6, 0.9].map((p) => one(runDetector(new SwingDetector("putt"), putt(p, side), 4.5, REAL)).power);
    got.forEach((g, i) => assert.ok(Math.abs(g - [0.15, 0.35, 0.6, 0.9][i]) < 0.06, String(g)));
  }
  const slow = path([0.0, 0.0, -1.3], [1.4, 0.0, -1.3], [2.0, 0.6, -1.27], [2.2, 0.6, -1.27], [6.2, -0.3, -1.27]);
  assert.deepEqual(runDetector(new SwingDetector("putt"), slow, 7.0, REAL), []);
});

for (const mode of ["full", "putt"]) {
  test(`standing at address never hits by itself (${mode})`, () => {
    for (let seed = 0; seed < 4; seed++) {
      assert.deepEqual(runDetector(new SwingDetector(mode), () => arc(0.0), 20.0, { ...REAL, noise: 6.0, seed }), [], `seed ${seed}`);
    }
  });
}

// --- the course and physics ---------------------------------------------------

test("the course: three holes, par 12, tees and pins where they belong", () => {
  assert.deepEqual(COURSE.holes.map((h) => h.par), [4, 3, 5]);
  const yards = COURSE.holes.map((h) => h.length * YARDS_PER_M);
  assert.ok(yards[0] > 330 && yards[0] < 420 && yards[1] > 130 && yards[1] < 190 && yards[2] > 470 && yards[2] < 560);
  for (const h of COURSE.holes) {
    assert.equal(h.lieAt(...h.tee), "tee");
    assert.equal(h.lieAt(...h.pin), "green");
  }
});

test("physics is deterministic whatever the frame rate", () => {
  const shot = P.makeShot(P.CLUBS[2], 0.93, 0.05, "fairway", { wind: [1.5, -2.0], curve: 0.3 });
  const a = P.simulate(COURSE.holes[0], [0, 0], shot);
  const b = new P.Flight(COURSE.holes[0], [0, 0], shot);
  for (let i = 0; i < 30000 && !b.done; i++) b.step([1 / 30, 1 / 144, 0.05][i % 3]);
  assert.deepEqual([a.state, a.x, a.y], [b.state, b.x, b.y]);
});

// --- the game ------------------------------------------------------------------

test("contract basics: registered, titled, a tutorial, a stall advances at most 0.1 s", () => {
  assert.ok(GAMES.includes(RaccoonGolfGame));
  assert.equal(RaccoonGolfGame.id, "golf");
  assert.equal(RaccoonGolfGame.title, "RACCOON GOLF");
  assert.ok(RaccoonGolfGame.blurb && RaccoonGolfGame.tip && RaccoonGolfGame.tutorial.length === 3);
  const g = newGame({ seed: 1 });
  assert.equal(g.phase, Phase.INTRO);
  assert.equal(g.score, 0);
  g.update(emptyControls(), 5.0);
  assert.ok(Math.abs(g.clock - 0.1) < 1e-12);
});

test("wind is random by default and repeatable with a seed", () => {
  assert.equal(newGame({ seed: 7 })._windDir, newGame({ seed: 7 })._windDir);
  const winds = new Set(Array.from({ length: 6 }, () => newGame()._windDir.toFixed(6)));
  assert.ok(winds.size > 1);
});

test("leaderboard plausibility: an ace is the most a hole can give, and no hole is faster", () => {
  assert.equal(holePoints(1, 3), POINTS_ACE);
  assert.ok(FASTEST_HOLE_SECONDS > 8 && FASTEST_HOLE_SECONDS < 9.5, String(FASTEST_HOLE_SECONDS));
});

test("arrow keys aim and change club; the driver stays on the tee", () => {
  const g = newGame({ seed: 1 });
  toAim(g);
  const h0 = g.heading, c0 = g.clubI;
  assert.ok(g.handleKey(KEY_RIGHT) && g.heading > h0);
  assert.ok(g.handleKey(KEY_LEFT) && g.handleKey(KEY_LEFT) && g.heading < h0);
  assert.ok(g.handleKey(KEY_DOWN) && g.clubI === c0 + 1);
  assert.ok(g.handleKey(KEY_UP) && g.clubI === c0);
  assert.equal(g.handleKey("x"), false);
  g.ball = [0.0, 150.0];
  g.lie = "fairway";
  g._setupShot();
  for (let i = 0; i < 12; i++) {
    g.changeClub(-1);
    assert.notEqual(g.club.key, "1W");
  }
});

test("water: +1 stroke and a dry drop", () => {
  const g = newGame({ seed: 1 });
  g._startHole(1);
  g.setWind(0, 0);
  toAim(g);
  hit(g, 0.55);
  finishFlight(g);
  assert.equal(g.message.text, "WATER HAZARD");
  assert.equal(g.strokes, 2);
  assert.ok(g.lie !== "water" && g.lie !== "ob");
  while (g.phase === Phase.RESULT) g.update(emptyControls(), DT);
  assert.equal(g.phase, Phase.AIM);
});

test("out of bounds replays from the same spot", () => {
  const g = newGame({ seed: 1 });
  g.setWind(0, 0);
  toAim(g);
  g.aimOffset = P.radians(-40);
  g.autoHeading = P.radians(-30);
  hit(g, 1.0);
  finishFlight(g);
  assert.equal(g.message.text, "OUT OF BOUNDS");
  assert.equal(g.strokes, 2);
  assert.deepEqual(g.ball, g.hole.tee);
});

test("pick up at par + 5", () => {
  const g = newGame({ seed: 1 });
  g._startHole(1);
  toAim(g);
  for (let i = 0; i < 20 && g.phase !== Phase.HOLED; i++) {
    hit(g, 0.08);
    finishFlight(g);
    while (g.phase === Phase.RESULT) g.update(emptyControls(), DT);
  }
  assert.equal(g.phase, Phase.HOLED);
  const row = g.card[1];
  assert.ok(row.pickedUp && row.strokes === 3 + MAX_OVER_PAR && row.points === 10);
});

test("holing out scores, Lenny celebrates, and the round ends after hole three", () => {
  const g = newGame({ seed: 2 });
  for (let i = 0; i < 3; i++) {
    toAim(g);
    g.ball = [g.hole.pin[0], g.hole.pin[1] - 2.0];
    g.lie = "green";
    g._setupShot();
    g.autoHeading = 0;
    g.aimOffset = 0;
    g.strokes = g.hole.par - 2;
    let best = null;
    for (let p = 5; p < 100 && best == null; p++) {
      if (P.simulate(g.hole, g.ball, P.makeShot(P.PUTTER, p / 100, 0.0, "green")).state === "holed") best = p / 100;
    }
    hit(g, best);
    finishFlight(g);
    assert.equal(g.phase, Phase.HOLED);
    assert.equal(g.message.text, "BIRDIE!");
    assert.ok(g.celebrateSpot);
    while (g.phase === Phase.HOLED) g.update(emptyControls(), DT);
  }
  assert.equal(g.phase, "game_over");
  assert.equal(g.score, 900);
  assert.equal(g.toPar, -3);
});

test("the aim buttons are dead while the hands grip", () => {
  const g = newGame({ seed: 1 });
  toAim(g);
  const heading = g.heading;
  const [x0, y0, x1, y1] = Object.fromEntries(g.buttons()).aim_right;
  const pose = gripPose(((x0 + x1) / 2 - CENTRE[0]) / SW, (CENTRE[1] - (y0 + y1) / 2) / SW);
  for (let i = 0; i < 120; i++) g.update(controls(pose), DT);
  assert.equal(g.heading, heading);
});

test("tracking loss never costs a stroke", () => {
  const g = newGame({ seed: 1 });
  toAim(g);
  for (let i = 0; i < 600; i++) g.update(emptyControls(), DT);
  assert.equal(g.strokes, 0);
  assert.equal(g.phase, Phase.AIM);
  assert.ok(g.detector.guidance());
});

test("onResume forgets the swing", () => {
  const g = newGame({ seed: 1 });
  toAim(g);
  g.detector.stage = Stage.BACKSWING;
  g.onResume();
  assert.equal(g.detector.stage, Stage.NO_GRIP);
});

test("the HUD keeps clear of the PAUSE button and the footer", () => {
  const pause = [1090, 646, 1258, 698];
  const overlap = (a, b) => a[0] < b[2] && b[0] < a[2] && a[1] < b[3] && b[1] < a[3];
  const g = newGame({ seed: 1 });
  toAim(g);
  const rects = [PIP, MAP, HOLE_PANEL, CLUB_PANEL, WIND_PANEL, ...g.buttons().map(([, r]) => r),
    [GAUGE_AT[0], GAUGE_AT[1], GAUGE_AT[0] + 66 * GAUGE_SCALE, GAUGE_AT[1] + 400 * GAUGE_SCALE]];
  for (const r of rects) {
    assert.ok(!overlap(r, pause), String(r));
    assert.ok(r[3] <= 700, String(r));
  }
});

// --- through the arcade shell --------------------------------------------------

class StubGolf extends RaccoonGolfGame {
  constructor(opts) {
    super({ ...opts, art: stubArt(), course: COURSE });
  }
}

test("plays through the real shell; arrows reach the game, space still pauses", () => {
  const sh = new Shell({ games: [StubGolf], selected: "golf" });
  sh.handleKey("1");
  assert.equal(sh.screen, "tutorial");
  sh.handleKey("Enter");
  for (let i = 0; i < 200; i++) sh.update(emptyControls(), DT);
  assert.equal(sh.screen, "playing");
  for (let i = 0; i < 3.2 / DT; i++) sh.update(emptyControls(), DT);
  const g = sh.game;
  assert.equal(g.phase, Phase.AIM);
  const h0 = g.heading;
  assert.ok(sh.handleKey("ArrowRight"));
  assert.ok(g.heading > h0);
  assert.equal(sh.screen, "playing");
  sh.handleKey(" ");
  assert.equal(sh.screen, "paused");
});

// --- drawing through a fake context -------------------------------------------

function fakeContext(log) {
  const target = {
    canvas: { width: 1280, height: 720 },
    measureText: (s) => ({ width: String(s).length * 12 }),
    drawImage: (img, ...args) => {
      if (!img || !(img.width > 0)) throw new Error("drawImage of nothing");
      if (args.some((v) => !Number.isFinite(v))) throw new Error(`drawImage with ${args}`);
      log.push(["drawImage", img, args]);
    },
    fillText: (str, x, y) => log.push(["fillText", String(str), x, y]),
    lineTo: (x, y) => {
      if (!Number.isFinite(x) || !Number.isFinite(y)) throw new Error("lineTo NaN");
    },
  };
  return new Proxy(target, {
    get(t, k) {
      if (k in t) return t[k];
      return () => {};
    },
    set(t, k, v) {
      t[k] = v;
      return true;
    },
  });
}

const fakeCanvas = (log) => (w, h) => ({ width: Math.round(w), height: Math.round(h), getContext: () => fakeContext(log) });

test("every phase renders through a fake context without touching the game", () => {
  const log = [];
  const g = newGame({ seed: 4 });
  g._renderer = new GolfRenderer(1280, 720, g.art, { makeCanvas: fakeCanvas(log) });
  const ctx = fakeContext(log), frame = fakeCanvas(log)(1280, 720);
  const phases = new Set();
  const draw = () => {
    const before = JSON.stringify([g.clock, g.ball, g.strokes, g.phase]);
    g.render(ctx, frame);
    assert.equal(JSON.stringify([g.clock, g.ball, g.strokes, g.phase]), before);
    phases.add(g.phase);
  };
  for (let i = 0; i < 200; i++) {
    g.update(emptyControls(), DT);
    if (i % 20 === 0) draw();
  }
  hit(g, 1.0);
  while (g.phase === Phase.FLIGHT) {
    g.update(emptyControls(), DT);
    draw();
  }
  draw();
  g.ball = [g.hole.pin[0], g.hole.pin[1] - 3];
  g.lie = "green";
  g.phase = Phase.AIM;
  g._setupShot();
  draw();
  g._finishHole();
  for (let i = 0; i < 300; i++) {
    g.update(emptyControls(), DT);
    if (i % 30 === 0) draw();
  }
  g.phase = Phase.GAME_OVER;
  draw();
  for (const p of ["intro", "aim", "flight", "result", "holed", "game_over"]) assert.ok(phases.has(p), p);
  const words = log.filter((e) => e[0] === "fillText").map((e) => e[1]);
  for (const w of ["HOLE 1", "SCORECARD", "ROUND COMPLETE", "PUTT", "POWER"]) assert.ok(words.includes(w), w);
});

test("polygons that surround the camera clip at the near plane", () => {
  const g = newGame({ seed: 1 });
  const view = new View({ x: 0, y: 0, z: 4, yaw: 0, pitch: 0.17 }, 1280, 720, 1000);
  const pts = [];
  const ctx = new Proxy({ moveTo: (x, y) => pts.push([x, y]), lineTo: (x, y) => pts.push([x, y]) },
    { get: (t, k) => (k in t ? t[k] : () => {}), set: () => true });
  drawLayer(ctx, view, { colour: [255, 255, 255], polys: [[[-3000, -3000], [3000, -3000], [3000, 3000], [-3000, 3000]]], farCut: Infinity });
  assert.ok(pts.length >= 3);
  for (const [, y] of pts) assert.ok(y >= view.horizon - 1e-6, `above the horizon: ${y}`);
  assert.ok(g); // the game is untouched
});

test("tutorial cards draw (clipped) at any time", () => {
  const log = [];
  for (const t of [0.0, 0.7, 1.4, 2.2, 3.1]) {
    for (const step of GOLF_TUTORIAL) step.draw(fakeContext(log), [69, 134, 411, 334], t);
  }
  assert.ok(GOLF_TUTORIAL.every((s) => s.title && s.text));
});

// --- fairness bots: whole rounds through the real detector ---------------------

/** Feeds a game poses as the camera pipeline would: 20 Hz, 150 ms late, ~5 px of noise. */
class Player {
  constructor(game, { modelHz = 20.0, lag = 0.15, noise = 5.0, seed = 0 } = {}) {
    Object.assign(this, { game, lag, noise });
    this.rng = new Rng(seed);
    this.clock = modelClock(modelHz, 3600.0, this.rng);
    this.t = 0.0;
    this.motion = () => null; // [u, h, gap] or null at time t
    this._heldKey = null;
    this._held = null;
  }

  step() {
    const ts = this.clock[held(this.clock, this.t)];
    if (ts !== this._heldKey) {
      this._heldKey = ts;
      const m = this.motion(ts - this.lag);
      this._held = m == null ? null : gripPose(m[0], m[1], { noise: this.noise, rng: this.rng, gap: m[2] });
    }
    this.game.update(controls(this._held), DT);
    this.t += DT;
  }
}

function fullSwing(t0, power, side = 1) {
  const top = topAngleFor(power);
  const m = (t) => {
    t -= t0;
    if (t < 0) return null;
    if (t < 0.8) return [...arc(0.0, undefined, side), 0.15];
    if (t < 1.5) return [...arc((top * (1 - Math.cos((Math.PI * (t - 0.8)) / 0.7))) / 2, undefined, side), 0.15];
    if (t < 1.65) return [...arc(top, undefined, side), 0.15];
    if (t < 2.1) {
      const k = (t - 1.65) / 0.45;
      return [...arc(top - (top + 0.9 * top) * k * k * (3 - 2 * k), undefined, side), 0.15];
    }
    return null;
  };
  return [m, t0 + 2.3];
}

function puttStroke(t0, power, side = 1) {
  const amp = power * PUTT_FULL;
  const m = (t) => {
    t -= t0;
    if (t < 0) return null;
    if (t < 0.8) return [0.0, -1.3, 0.15];
    if (t < 1.4) return [(side * amp * (1 - Math.cos((Math.PI * (t - 0.8)) / 0.6))) / 2, -1.28, 0.15];
    if (t < 1.55) return [side * amp, -1.28, 0.15];
    if (t < 2.0) {
      const k = (t - 1.55) / 0.45;
      return [side * amp * (1 - 1.8 * k * k * (3 - 2 * k)), -1.28, 0.15];
    }
    return null;
  };
  return [m, t0 + 2.2];
}

/** The skilled bot's choice: aim (in button steps) and power, from the game's own simulator. */
function plan(g) {
  const { hole, ball, lie } = g;
  const step = g.putting ? P.radians(1.0) : P.radians(2.0);
  const penalty = { water: 1e4, ob: 1e4, bunker: 12.0, rough: 6.0 };
  const cost = (off, power) => {
    const f = P.simulate(hole, ball, P.makeShot(g.club, power, g.autoHeading + off * step, lie, { wind: g.wind }));
    if (f.state === "holed") return -1.0;
    return hole.distanceToPin(f.x, f.y) + (penalty[f.state === "water" || f.state === "ob" ? f.state : f.lie] ?? 0);
  };
  const lo = g.putting ? PUTT_START / PUTT_FULL + 0.01 : 0.1;
  const offs = g.putting ? [-12, -9, -6, -3, 0, 3, 6, 9, 12] : [-6, -4, -2, 0, 2, 4, 6];
  const powers = [];
  for (let p = lo; p <= 1.0001; p += g.putting ? 0.05 : 0.1) powers.push(p);
  let best = null;
  for (const o of offs) for (const p of powers) {
    const c = cost(o, p);
    if (!best || c < best[0]) best = [c, o, p];
  }
  const [, o0, p0] = best;
  const span = g.putting ? 2 : 1, dp = g.putting ? 0.01 : 0.025;
  for (let o = o0 - span; o <= o0 + span; o++) {
    for (let p = p0 - 0.05; p <= p0 + 0.0501; p += dp) {
      const pp = Math.min(1.0, Math.max(lo, p));
      const c = cost(o, pp);
      if (c < best[0]) best = [c, o, pp];
    }
  }
  return [best[1], best[2]];
}

function skilled(g, pl, rng) {
  const [o, p] = plan(g);
  const unit = g.putting ? P.radians(1.0) : P.radians(2.0);
  while (Math.round(g.aimOffset / unit) !== o) g.adjustAim(o > Math.round(g.aimOffset / unit) ? 1 : -1);
  const side = rng.random() < 0.5 ? 1 : -1;
  const [m, end] = (g.putting ? puttStroke : fullSwing)(pl.t, p, side);
  pl.motion = m;
  return end + 1.0;
}

function playRound(g, brain, limit = 900.0, seed = 0) {
  const pl = new Player(g, { seed });
  const rng = new Rng(seed + 100);
  let busyUntil = -1.0;
  while (g.phase !== "game_over" && pl.t < limit) {
    if (g.phase === Phase.AIM && pl.t > busyUntil) busyUntil = brain(g, pl, rng);
    pl.step();
  }
  return pl.t;
}

for (const seed of [11, 13]) {
  test(`a skilled bot, 150 ms late at 20 Hz with 5 px of noise, plays near par (seed ${seed})`, () => {
    const g = newGame({ seed });
    playRound(g, skilled, 900, seed);
    assert.equal(g.phase, "game_over");
    const strokes = g.card.reduce((s, r) => s + r.strokes, 0);
    assert.ok(strokes >= COURSE.par - 4 && strokes <= COURSE.par + 3, JSON.stringify(g.card.map((r) => r.strokes)));
    assert.ok(!g.card.some((r) => r.pickedUp));
    assert.ok(g.score >= 300);
  });
}

test("an idle player scores nothing and nothing breaks", () => {
  const g = newGame({ seed: 3 });
  const pl = new Player(g);
  for (let i = 0; i < 120 / DT; i++) pl.step();
  assert.equal(g.score, 0);
  assert.equal(g.strokes, 0);
  assert.equal(g.phase, Phase.AIM);
});

test("a random flailer scores poorly and never crashes", () => {
  const rng = new Rng(5);
  const g = newGame({ seed: 5 });
  const pl = new Player(g, { seed: 5 });
  const keys = [[0.0, 0.0, -1.0, 0.15]];
  let t = 0.0;
  while (t < 600) {
    t += rng.uniform(0.15, 0.7);
    keys.push([t, rng.uniform(-1.8, 1.8), rng.uniform(-1.7, 1.3), rng.choice([0.15, 0.15, 0.6, 1.4])]);
  }
  const times = keys.map((k) => k[0]);
  pl.motion = (tt) => {
    const i = Math.max(0, Math.min(keys.length - 2, held(times, tt - 1e-9)));
    const [t0, u0, h0, g0] = keys[i], [t1, u1, h1] = keys[i + 1];
    const e = Math.min(1.0, Math.max(0.0, (tt - t0) / (t1 - t0)));
    return [u0 + (u1 - u0) * e, h0 + (h1 - h0) * e, g0];
  };
  while (pl.t < 600 && g.phase !== "game_over") pl.step();
  const done = g.card.filter((r) => r.done).length;
  assert.ok(g.score <= 100 * Math.max(1, done), `${g.score} over ${done} holes`);
});
