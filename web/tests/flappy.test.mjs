// Flappy Raccoon: physics, pipes, scoring, fairness under input lag
// (port of tests/test_flappy.py; the rendering tests live in web/dev/flappy.html).

import { test, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { KP, Pose } from "../js/core/pose.js";
import { emptyControls } from "../js/core/controls.js";
import { Rng } from "../js/core/random.js";
import { FlappyRaccoonGame, Phase, Pipe, medalFor } from "../js/games/flappy/game.js";
import { FlapDetector } from "../js/games/flappy/gesture.js";

const W = 1280, H = 720;
const DT = 1 / 60;
const GROUND = 96;

// Hit radius of the art that actually ships, read from the generated manifest
// rather than hard-coded, so the fairness proof below always flies the real
// collision circle: when the bird became a raccoon, a hard-coded radius
// silently kept testing the old, smaller one.
const GEN = new URL("../../assets/flappy/generated/", import.meta.url);
const MANIFEST = JSON.parse(readFileSync(new URL("manifest.json", GEN)));
const SHIPPED_RADIUS = MANIFEST.bird_hit_radius;
function pngSize(name) {
  const b = readFileSync(new URL(name, GEN));
  return [b.readUInt32BE(16), b.readUInt32BE(20)];
}
const PIPE_W = pngSize("pipe_body.png")[0];

/** Just what the simulation reads from the art. */
const stubArt = () => ({
  birdHitRadius: SHIPPED_RADIUS,
  groundHeight: MANIFEST.ground_height ?? GROUND,
  pipeBody: { width: PIPE_W, height: 64 },
});

/** Stands in for FlapDetector: emits a flap on the next update when asked. */
class FakeDetector {
  constructor() {
    this.queued = 0;
    this.resets = 0;
    this.armsVisible = true;
    this.wing = 0.4;
    this.seen = [];
  }
  flap() { this.queued += 1; }
  reset() {
    this.resets += 1;
    this.queued = 0;
  }
  update(_pose, t) {
    this.seen.push(t);
    if (this.queued) {
      this.queued -= 1;
      return { t, strength: 0.8 };
    }
    return null;
  }
}

beforeEach(() => {
  FlappyRaccoonGame.sessionBest = 0;
});

function newGame(opts = {}) {
  const det = opts.detector ?? new FakeDetector();
  const game = new FlappyRaccoonGame({ width: W, height: H, art: stubArt(), ...opts, detector: det });
  return [game, det];
}

const idle = (game, seconds, dt = DT, controls = emptyControls()) => {
  for (let i = 0, n = Math.round(seconds / dt); i < n; i++) game.update(controls, dt);
};

function start(game, det) {
  det.flap();
  game.update(emptyControls(), DT);
  assert.equal(game.phase, Phase.PLAYING);
}

const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, msg ?? `${a} vs ${b} (+-${tol})`);

// --- the READY screen ---

test("starts ready and waits for a flap", () => {
  // Classic: the bird bobs in place and nothing happens until the first flap.
  const [game] = newGame({ seed: 1 });
  assert.equal(game.phase, "ready");
  const y0 = game.bird.y;
  idle(game, 5.0);
  assert.equal(game.phase, Phase.READY);
  assert.ok(Math.abs(game.bird.y - y0) <= game.rules.readyBob + 1);
  assert.equal(game.score, 0);
  assert.equal(game.pipes.length, 0);
});

test("first flap starts play and lifts the bird; first pipe off screen", () => {
  const [game, det] = newGame({ seed: 1 });
  idle(game, 0.5);
  start(game, det);
  assert.ok(game.bird.vy < 0);
  assert.ok(game.pipes.length, "pipes should be queued once play starts");
  assert.ok(Math.min(...game.pipes.map((p) => p.x)) >= W, "the first pipe must not pop in on screen");
});

test("feeds the detector the game's own clock", () => {
  const [game, det] = newGame();
  idle(game, 0.2);
  assert.equal(det.seen.length, 12);
  assert.deepEqual(det.seen, [...det.seen].sort((a, b) => a - b));
  assert.ok(det.seen[0] > 0);
  // A stall is clamped before it reaches the clock.
  game.update(emptyControls(), 5.0);
  close(det.seen.at(-1) - det.seen.at(-2), game.rules.maxDt, 1e-9);
});

// --- physics ---

test("without flaps the bird falls to the ground and lands", () => {
  const [game, det] = newGame({ seed: 2 });
  start(game, det);
  idle(game, 3.0);
  assert.equal(game.phase, "game_over");
  idle(game, 1.5);
  assert.ok(game.landed);
  close(game.bird.y + game.hitRadius, game.groundY, 1.0);
});

test("a flap sets rather than adds velocity", () => {
  const [game, det] = newGame({ seed: 2 });
  start(game, det);
  idle(game, 0.6); // falling now
  det.flap();
  game.update(emptyControls(), DT);
  const fallingFlap = game.bird.vy;
  det.flap();
  game.update(emptyControls(), DT); // flap again while already rising
  close(fallingFlap, game.bird.vy, 1.0);
  assert.ok(fallingFlap < -game.rules.flapVelocity * 0.9);
});

test("fall speed is capped", () => {
  const [game, det] = newGame({ seed: 2 });
  start(game, det);
  game.bird.y = -5000; // a long way to fall, so it reaches the cap
  game.pipes.length = 0;
  game.rules.firstPipeDelay = 1e9;
  for (let i = 0; i < 90; i++) {
    game.update(emptyControls(), DT);
    assert.ok(game.bird.vy <= game.rules.terminalVelocity + 1e-6);
  }
});

test("the ceiling is a soft cap", () => {
  const [game, det] = newGame({ seed: 3 });
  start(game, det);
  game.pipes = [new Pipe(1e6, 300, 230)]; // nothing nearby
  for (let i = 0; i < 240; i++) {
    det.flap();
    game.update(emptyControls(), DT);
    assert.ok(game.bird.y >= game.hitRadius - 1e-6);
  }
  assert.equal(game.phase, Phase.PLAYING);
});

test("the nose follows vertical speed", () => {
  const [game, det] = newGame({ seed: 4 });
  start(game, det);
  game.update(emptyControls(), DT);
  assert.ok(game.bird.angle < 0, "nose up while climbing");
  game.pipes = [new Pipe(1e6, 300, 230)];
  idle(game, 0.9);
  assert.ok(game.bird.angle >= game.rules.noseDown - 0.5, "nose down in a dive");
  assert.ok(game.bird.angle <= game.rules.noseDown + 1e-6, "but no further than the cap");
});

test("a stall does not teleport the bird", () => {
  const [game, det] = newGame({ seed: 5 });
  start(game, det);
  const y0 = game.bird.y;
  game.update(emptyControls(), 5.0); // absurd stall
  const r = game.rules;
  assert.ok(Math.abs(game.bird.y - y0) <= r.terminalVelocity * r.maxDt + 1);
});

test("a stall cannot carry the bird through a pipe", () => {
  const [game, det] = newGame({ seed: 5 });
  start(game, det);
  const b = game.bird;
  game.pipes = [new Pipe(b.x + game.pipeWidth / 2 + game.hitRadius + 2, b.y - 400, 200)];
  for (let i = 0; i < 30; i++) game.update(emptyControls(), 2.0);
  assert.equal(game.phase, Phase.GAME_OVER);
  assert.equal(game.score, 0);
});

// --- collision ---

test("circle collision against pipes", () => {
  const [game, det] = newGame({ seed: 6 });
  start(game, det);
  const r = game.hitRadius, half = game.pipeWidth / 2;
  game.pipes = [new Pipe(600, 360, 200)]; // opening 260..460
  const at = (x, y) => { game.bird.x = x; game.bird.y = y; return game._collides(); };
  assert.equal(at(600, 360), false, "dead centre of the gap");
  assert.equal(at(600, 260 + r - 2), true, "grazing the top pipe");
  assert.equal(at(600, 460 - r + 2), true, "grazing the bottom pipe");
  assert.equal(at(600 - half - r - 1, 100), false, "level with the top pipe but just in front of it");
  // Near the pipe's corner but outside the circle: a box test would kill here.
  assert.equal(at(600 - half - r * 0.75, 260 + r * 0.75), false, "corner miss must not count");
});

test("the ground kills", () => {
  const [game, det] = newGame({ seed: 6 });
  start(game, det);
  game.pipes.length = 0;
  game.bird.y = game.groundY - game.hitRadius + 1;
  assert.ok(game._collides());
});

// --- scoring ---

test("passing a pipe scores once, with a popup and a pulse", () => {
  const [game, det] = newGame({ seed: 7 });
  start(game, det);
  const b = game.bird;
  game.pipes = [new Pipe(b.x + 10, b.y, 300)];
  game.bird.vy = 0;
  idle(game, 0.1);
  assert.equal(game.score, 1);
  assert.deepEqual(game.fx.popups.map((p) => p.text), ["+1"]);
  assert.ok(game.fx.scoreBump > 0.5);
  idle(game, 0.05);
  assert.equal(game.score, 1, "a pipe scores once");
});

test("popups rise and expire", () => {
  const [game, det] = newGame({ seed: 7 });
  start(game, det);
  const b = game.bird;
  game.pipes = [new Pipe(b.x + 2, b.y, 400)];
  game.update(emptyControls(), DT);
  const pop = game.fx.popups[0];
  const y0 = pop.y;
  idle(game, 0.2);
  assert.ok(pop.y < y0);
  idle(game, 1.0);
  assert.equal(game.fx.popups.length, 0);
});

test("a flap sheds feathers that fade", () => {
  const [game, det] = newGame({ seed: 8 });
  start(game, det);
  assert.ok(game.fx.feathers.length >= 5);
  game.pipes = [new Pipe(1e6, 300, 230)];
  for (let i = 0; i < 50; i++) {
    if (i % 30 === 0) det.flap();
    game.update(emptyControls(), DT);
  }
  idle(game, 1.0);
  assert.ok(game.fx.feathers.every((f) => f.life > 0));
  assert.ok(game.fx.feathers.length <= 40);
});

test("medals", () => {
  assert.equal(medalFor(9), null);
  assert.equal(medalFor(10), "bronze");
  assert.equal(medalFor(19), "bronze");
  assert.equal(medalFor(20), "silver");
  assert.equal(medalFor(30), "gold");
  assert.equal(medalFor(40), "platinum");
  assert.equal(medalFor(400), "platinum");
});

// --- lifecycle ---

test("best survives reset and new games", () => {
  // The arcade builds a new game for every PLAY AGAIN: best must outlive it.
  const [game, det] = newGame({ seed: 9 });
  start(game, det);
  game.score = 12;
  idle(game, 3.0);
  assert.equal(game.phase, Phase.GAME_OVER);
  assert.equal(game.best, 12);
  assert.ok(game.newBest);
  game.reset();
  assert.equal(game.phase, Phase.READY);
  assert.equal(game.score, 0);
  assert.equal(game.best, 12);
  const [again, det2] = newGame({ seed: 9 });
  assert.equal(again.best, 12);
  start(again, det2);
  again.score = 5;
  idle(again, 3.0);
  assert.equal(again.best, 12);
  assert.equal(again.newBest, false);
});

test("game over is terminal: the world stops and nothing scores", () => {
  const [game, det] = newGame({ seed: 10 });
  start(game, det);
  idle(game, 3.0);
  assert.equal(game.phase, Phase.GAME_OVER);
  const xs = game.pipes.map((p) => p.x);
  const d = game.distance, score = game.score;
  // Even with a pipe right at the bird, nothing scores once over.
  game.pipes.push(new Pipe(game.bird.x - 1, game.bird.y, 300));
  for (let i = 0; i < 60; i++) {
    det.flap();
    game.update(emptyControls(), DT);
  }
  assert.equal(game.phase, Phase.GAME_OVER);
  assert.deepEqual(game.pipes.slice(0, xs.length).map((p) => p.x), xs);
  assert.equal(game.distance, d);
  assert.equal(game.score, score);
});

test("resume resets the detector", () => {
  // The arms moved during the pause; old history must not read as a flap.
  const [game, det] = newGame();
  const before = det.resets;
  game.onResume();
  assert.equal(det.resets, before + 1);
});

test("every game is different, but a seed reproduces", () => {
  const course = (seed) => {
    const [game, det] = newGame({ seed });
    start(game, det);
    game._fillPipes();
    while (game.pipes.length < 6) {
      const last = game.pipes.at(-1);
      game.pipes.push(game._spawnPipe(last.x + 400, last, 400, 200));
    }
    return game.pipes.map((p) => p.gapY.toFixed(3)).join(",");
  };
  assert.equal(course(42), course(42));
  assert.equal(new Set([0, 1, 2, 3, 4].map(() => course(null))).size, 5);
});

test("flap juice does not change a seeded course", () => {
  // Feathers draw from their own stream; the course depends only on the seed.
  const course = (flapEvery) => {
    const [game, det] = newGame({ seed: 5 });
    start(game, det);
    game.bird.y = 200;
    for (let i = 0; i < 12; i++) {
      if (flapEvery && i % flapEvery === 0) det.flap();
      game.update(emptyControls(), DT);
    }
    assert.equal(game.phase, Phase.PLAYING);
    return game.pipes.map((p) => p.gapY.toFixed(3)).join(",");
  };
  assert.equal(course(0), course(3));
});

// --- difficulty curve ---

test("difficulty starts generous and keeps rising (exponential curve)", () => {
  // Easy to get going, then harder for as long as the player survives. The
  // old linear ramps stopped changing at score ~30; the exponential ones keep
  // tightening (ever more gently) and never flatten.
  const [game] = newGame();
  close(game.gapSize(0), 300, 1e-9);
  assert.ok(game.gapSize(0) / game.groundY > 0.45, "first gaps are generous");
  close(game.pipeSpeed(0), 200, 1e-9);
  close(game.pipeInterval(0), 2.0, 1e-9);
  // limit + (start - limit) * exp(-score / tau), at the values the comments quote.
  const curve = (start, limit, tau, s) => limit + (start - limit) * Math.exp(-s / tau);
  for (const s of [10, 20, 30, 60, 120]) {
    close(game.gapSize(s), curve(300, 175, 18, s), 1e-9);
    close(game.pipeSpeed(s), curve(200, 270, 25, s), 1e-9);
    close(game.pipeInterval(s), curve(2.0, 1.55, 25, s), 1e-9);
  }
  close(game.gapSize(10), 247, 1);
  close(game.gapSize(20), 216, 1);
  close(game.gapSize(30), 198, 1);
  close(game.pipeSpeed(10), 223, 1);
  close(game.pipeSpeed(20), 239, 1);
  const scores = Array.from({ length: 150 }, (_, i) => i);
  const gaps = scores.map((s) => game.gapSize(s));
  const speeds = scores.map((s) => game.pipeSpeed(s));
  const intervals = scores.map((s) => game.pipeInterval(s));
  for (let i = 1; i < scores.length; i++) {
    assert.ok(gaps[i - 1] > gaps[i], "gap shrinks at every score");
    assert.ok(speeds[i - 1] < speeds[i], "speed rises at every score");
    assert.ok(intervals[i - 1] > intervals[i], "pipes come sooner at every score");
  }
  assert.ok(Math.min(...gaps) > game.rules.gapMin, "the floor is approached, never undercut");
  assert.ok(Math.max(...speeds) < game.rules.speedMax);
});

test("gaps stay in bounds and reachable", () => {
  for (let seed = 0; seed < 40; seed++) {
    const [game, det] = newGame({ seed });
    start(game, det);
    const r = game.rules;
    for (let score = 0; score < 60; score += 3) {
      game.score = score;
      const speed = game.pipeSpeed();
      const spacing = speed * game.pipeInterval();
      const prev = game.pipes.at(-1);
      const p = game._spawnPipe(prev.x + spacing, prev, spacing, speed);
      assert.ok(p.gapTop >= r.ceilingMargin - 1e-6);
      assert.ok(p.gapBottom <= game.groundY - r.groundMargin + 1e-6);
      const rise = prev.gapY - p.gapY;
      assert.ok(rise <= game.climbBudget(spacing, speed) + 1e-6);
      assert.ok(-rise <= r.maxDrop + 1e-6);
      game.pipes.push(p);
    }
  }
});

// --- fairness: a lagged bot must be able to fly every course ---

/**
 * A skilled player with camera lag and human arms.
 *
 * Decides from what is on screen, but its flaps land `delay` seconds later
 * (camera + model latency), optionally with random jitter it cannot predict.
 * It flaps no faster than the game's assumed human rate, `flapPeriod`.
 *
 * Policy: hold the bird on a line inside the next gap, chosen so a flap from
 * the line neither reaches the top pipe at its apex nor sinks onto the bottom
 * pipe. Being skilled, it predicts where the bird will be when the flap
 * actually lands (using its *nominal* delay, not the jittered real one).
 */
export class LaggedBot {
  constructor(game, det, delay, jitter = 0.0, seed = 0) {
    this.game = game;
    this.det = det;
    this.delay = delay;
    this.jitter = jitter;
    this.rng = new Rng(seed);
    this.pending = [];
    this.last = -1e9;
  }

  _predict(t) {
    const g = this.game, r = g.rules;
    let y = g.bird.y, vy = g.bird.vy;
    const flaps = [...this.pending].sort((a, b) => a - b);
    let tt = t;
    const end = t + this.delay;
    while (tt < end - 1e-9) {
      while (flaps.length && flaps[0] <= tt + 1e-9) {
        vy = -r.flapVelocity;
        flaps.shift();
      }
      vy = Math.min(vy + r.gravity * DT, r.terminalVelocity);
      y += vy * DT;
      tt += DT;
    }
    return [y, vy];
  }

  _target(ahead) {
    const g = this.game;
    const reach = g.pipeWidth / 2 + g.hitRadius;
    return g.pipes.find((p) => p.x - ahead + reach > g.bird.x) ?? null;
  }

  step(t) {
    const g = this.game, r = g.rules;
    if (t - this.last >= r.flapPeriod) {
      const [y] = this._predict(t);
      const p = this._target(g.pipeSpeed() * this.delay);
      let line;
      if (p != null) {
        const rise = r.flapVelocity ** 2 / (2 * r.gravity);
        const lo = p.gapTop + g.hitRadius + rise;
        const hi = p.gapBottom - g.hitRadius - r.terminalVelocity * DT - 2;
        line = (lo + hi) / 2;
      } else {
        line = g.groundY * 0.5;
      }
      if (y > line) {
        const lag = this.delay + (this.jitter ? this.rng.uniform(-this.jitter, this.jitter) : 0);
        this.pending.push(t + Math.max(0.0, lag));
        this.last = t;
      }
    }
    while (this.pending.length && this.pending[0] <= t + 1e-9) {
      this.pending.shift();
      this.det.flap();
    }
  }
}

function fly(seed, delay, jitter = 0.0, pipes = 60, rules = null, limit = null) {
  const [game, det] = newGame({ seed, rules });
  start(game, det);
  const bot = new LaggedBot(game, det, delay, jitter, seed);
  let t = 0.0;
  limit = limit ?? pipes * 2.2 + 10;
  const controls = emptyControls();
  while (game.phase === Phase.PLAYING && game.score < pipes && t < limit) {
    bot.step(t);
    game.update(controls, DT);
    t += DT;
  }
  return [game.score, game.phase];
}

test("every course is flyable with 150 ms of lag (30 courses x 60 pipes, 3 lag settings)", () => {
  // No course may be impossible for a player with ~150 ms of input lag. 60
  // pipes per run takes the difficulty well up the curve, across 30 random
  // courses (1800 pipes). The jittered run has the lag vary 100-200 ms
  // unpredictably, as real camera + model latency does.
  for (const [delay, jitter] of [[0.0, 0.0], [0.150, 0.0], [0.150, 0.05]]) {
    const failures = [];
    for (let seed = 0; seed < 30; seed++) {
      const [score, phase] = fly(seed, delay, jitter);
      if (score < 60) failures.push([seed, score, phase]);
    }
    assert.deepEqual(failures, [], `bot crashed at lag ${delay}+-${jitter} (seed, score, phase)`);
  }
});

test("the hardest difficulty is still flyable with lag (120 pipes)", () => {
  // Difficulty never stops rising, so prove the far end too: 120 pipes
  // reaches gap ~175, speed ~269, interval ~1.55 s with the bot at 150 ms.
  const failures = [];
  for (let seed = 0; seed < 8; seed++) {
    const [score, phase] = fly(seed, 0.150, 0.0, 120);
    if (score < 120) failures.push([seed, score, phase]);
  }
  assert.deepEqual(failures, []);
});

test("the fairness bot can fail on courses that really are too hard", () => {
  // The reach test has teeth: otherwise a broken bot would pass everything.
  for (const rules of [
    { gapStart: 95, gapMin: 95 }, // barely wider than the bird
    // 3x the speed, no climb limit, gaps allowed at the edges: climbs no human
    // flap rate can make in the time available.
    { maxDrop: 1e9, reachSafety: 4.0, speedStart: 600, speedMax: 600, intervalStart: 0.8,
      intervalMin: 0.8, ceilingMargin: 20, groundMargin: 20 },
  ]) {
    let crashed = 0;
    for (let seed = 0; seed < 8; seed++) {
      const [, phase] = fly(seed, 0.15, 0.0, 30, rules, 80);
      crashed += phase === Phase.GAME_OVER ? 1 : 0;
    }
    assert.ok(crashed >= 4, `bot survived impossible courses (${crashed}/8 crashed)`);
  }
});

test("fairness flies the shipped collision circle", () => {
  const [game] = newGame();
  close(game.hitRadius, SHIPPED_RADIUS * game.rules.birdScale, 1e-9);
  assert.ok(game.hitRadius > 20, "the raccoon's circle is ~22 px at play size");
});

// --- tracking loss and the real detector ---

test("the default detector is the real one; no pose means no flap and no crash", () => {
  const game = new FlappyRaccoonGame({ width: W, height: H, art: stubArt(), mirrored: false });
  assert.ok(game.detector instanceof FlapDetector);
  game.update(emptyControls(), DT);
  assert.equal(game.phase, Phase.READY);
});

/** A whole-body pose with wrists `h` shoulder widths above the shoulders (null = hidden). */
function armsPose(h, { sw = 300, cy = 300 } = {}) {
  const xy = new Float32Array(34);
  const c = new Float32Array(17);
  const set = (name, x, y) => {
    xy[2 * KP[name]] = x;
    xy[2 * KP[name] + 1] = y;
    c[KP[name]] = 0.9;
  };
  set("left_shoulder", 640 + sw / 2, cy);
  set("right_shoulder", 640 - sw / 2, cy);
  if (h != null) {
    set("left_wrist", 640 + 0.9 * sw, cy - h * sw);
    set("right_wrist", 640 - 0.9 * sw, cy - h * sw);
  }
  return new Pose(xy, c, 0.9);
}

test("real flapping poses start and fly the game", () => {
  const game = new FlappyRaccoonGame({ width: W, height: H, art: stubArt(), seed: 3 });
  // Raise and hold, then brisk 2 Hz flaps through the real detector.
  const h = (t) => (t < 1.0 ? 0.7 : 0.7 - 2.0 * (0.5 - 0.5 * Math.cos(2 * Math.PI * 2 * (t - 1.0))));
  let t = 0;
  for (let i = 0; i < 180; i++) {
    t += DT;
    game.update(emptyControls(armsPose(h(t))), DT);
  }
  assert.equal(game.phase, Phase.PLAYING);
  assert.ok(game.flaps >= 3, `flaps ${game.flaps}`);
});

test("tracking loss never kills (READY waits; only physics ends a run)", () => {
  // No player at all for 10 s: still READY, nothing lost.
  const [game] = newGame({ seed: 4, detector: new FlapDetector() });
  idle(game, 10.0);
  assert.equal(game.phase, Phase.READY);
  // Pose drops in and out on the READY screen: still waiting, still READY.
  for (let i = 0; i < 600; i++) {
    const pose = i % 40 < 20 ? null : armsPose(0.5);
    game.update(emptyControls(pose), DT);
  }
  assert.equal(game.phase, Phase.READY);
  assert.equal(game.score, 0);
});

// --- framing guidance ---

function shouldersAt(y, width = 300.0) {
  const xy = new Float32Array(34);
  const c = new Float32Array(17);
  xy[2 * KP.left_shoulder] = 640 + width / 2;
  xy[2 * KP.left_shoulder + 1] = y;
  xy[2 * KP.right_shoulder] = 640 - width / 2;
  xy[2 * KP.right_shoulder + 1] = y;
  c[KP.left_shoulder] = c[KP.right_shoulder] = 0.9;
  return new Pose(xy, c, 0.9);
}

test("framing: too close is flagged (STEP BACK) with hysteresis", () => {
  // Standing close sends the wrists out of frame on every flap, and at 15-20 Hz
  // the detector then misses flaps - so the game tells the player.
  const [game] = newGame();
  const room = (widths) => {
    game.update({ ...emptyControls(shouldersAt(720 - widths * 300)), present: true }, 1 / 60);
    return game.tooClose;
  };
  assert.equal(room(0.3), true, "0.3 shoulder widths of room (as recorded) is too close");
  assert.equal(room(0.9), true, "between the on/off levels it stays flagged");
  assert.equal(room(1.2), false);
  assert.equal(room(0.9), false, "and stays clear between the levels");
});

test("framing is left alone without shoulders", () => {
  const [game] = newGame();
  game.update(emptyControls(null), 1 / 60);
  assert.equal(game.tooClose, false);
});

test("menu metadata matches the Python game", () => {
  assert.equal(FlappyRaccoonGame.id, "flappy");
  assert.equal(FlappyRaccoonGame.title, "FLAPPY RACCOON");
  assert.equal(FlappyRaccoonGame.blurb, "Flap your arms to fly the raccoon. Don't touch the pipes.");
  assert.equal(FlappyRaccoonGame.tip, "Stand back so both arms are in frame - flap to fly");
});
