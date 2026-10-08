// Snack Attack: blade tracking, bite detection and the classic rules
// (port of tests/test_fruitninja.py). No DOM: stub art, scripted poses.
import { test } from "node:test";
import assert from "node:assert/strict";

import { KEYPOINT_NAMES, KP, Pose } from "../js/core/pose.js";
import { emptyControls } from "../js/core/controls.js";
import { Blade, bladeKeypoint } from "../js/games/snack/blade.js";
import { Fruit, GRAVITY, Popup, Spawner } from "../js/games/snack/entities.js";
import { SnackAttackGame } from "../js/games/snack/game.js";
import { popScale } from "../js/games/snack/render.js";

const W = 1280, H = 720;
const DT = 1 / 30;

function stubArt() {
  const spr = { width: 110, height: 110 };
  const fruits = [0, 1, 2, 3].map((i) => ({
    name: `f${i}`, whole: spr, halfA: spr, halfB: spr, juice: [220, 50, 50], radius: 52, score: 1,
  }));
  return { fruits, bomb: spr, background: null, splat: spr, flash: spr, lifeFull: spr, lifeLost: spr,
    cursorIdle: null, cursorChomp: null };
}

function handPose(x, y, conf = 0.9) {
  const xy = new Float32Array(34);
  const c = new Float32Array(17);
  const set = (name, px, py, cf) => {
    xy[2 * KP[name]] = px;
    xy[2 * KP[name] + 1] = py;
    c[KP[name]] = cf;
  };
  set("left_wrist", x, y, conf);
  for (const n of ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]) set(n, 640, 360, 0.9);
  return new Pose(xy, c, 0.9);
}

const hand = (x, y, conf) => ({ ...emptyControls(handPose(x, y, conf)), present: true });
const nobody = () => emptyControls();
const newGame = (opts = {}) => new SnackAttackGame({ width: W, height: H, art: stubArt(), ...opts });

/**
 * A fast swing, sampled the way tracking delivers it. Recorded play never
 * moved more than ~245 px between samples; a single 800 px jump is what a
 * tracking glitch looks like and is ignored by design.
 */
function swipe(game, x0, x1, y, step = 120) {
  const n = Math.max(1, Math.ceil(Math.abs(x1 - x0) / step));
  for (let i = 0; i <= n; i++) game.update(hand(x0 + ((x1 - x0) * i) / n, y), DT);
}

function still(art, x, y, extra = {}) {
  return new Fruit({ pos: [x, y], vel: [0, 0], radius: 52, art, ...extra });
}

// --- which hand is the blade ---

test("mirrored right hand is the left_wrist keypoint", () => {
  // A flipped image makes the model label the player's right hand as left.
  assert.equal(KEYPOINT_NAMES[bladeKeypoint("right", true)], "left_wrist");
  assert.equal(KEYPOINT_NAMES[bladeKeypoint("left", true)], "right_wrist");
});

test("unmirrored right hand is the right_wrist keypoint", () => {
  assert.equal(KEYPOINT_NAMES[bladeKeypoint("right", false)], "right_wrist");
  assert.equal(KEYPOINT_NAMES[bladeKeypoint("left", false)], "left_wrist");
});

test("swapHand switches the tracked wrist", () => {
  const game = newGame();
  game.spawner.timer = 1e9;
  assert.equal(game.hand, "right");
  game.update(hand(400, 300), DT);
  assert.ok(game.blade.active, "left_wrist drives the right hand on a mirrored feed");
  game.swapHand();
  assert.equal(game.hand, "left");
  assert.ok(!game.blade.active, "swapping breaks the old trail");
  game.update(hand(410, 300), DT);
  assert.ok(!game.blade.active, "only left_wrist is in the pose, and it is no longer the blade");
});

// --- bite detection ---

test("a fast swipe cannot tunnel through a snack", () => {
  // At 30 Hz a quick hand jumps a long way per sample; testing only the
  // current point would miss most cuts, and hardest on the fastest swings.
  const blade = new Blade({ hitRadius: 0 });
  blade.update([250, 300], 0);
  blade.update([550, 300], DT); // the snack sits between the samples
  assert.notEqual(blade.hits([400, 300], 55), null);
});

test("a swipe misses a snack off its path", () => {
  const blade = new Blade();
  blade.update([250, 300], 0);
  blade.update([550, 300], DT);
  assert.equal(blade.hits([400, 600], 55), null);
});

test("a slow hand does not bite", () => {
  // Resting a hand on a snack must not dissolve the board.
  const blade = new Blade();
  blade.update([400, 300], 0);
  blade.update([402, 301], DT);
  assert.ok(!blade.slicing);
  assert.equal(blade.hits([400, 300], 55), null);
});

test("cut direction follows the swing", () => {
  const blade = new Blade();
  blade.update([250, 300], 0);
  blade.update([550, 300], DT);
  const d = blade.hits([400, 300], 55);
  assert.ok(d);
  assert.ok(Math.abs(d[0] - 1) < 1e-5 && Math.abs(d[1]) < 1e-5, `direction ${d}`);
});

test("losing the hand breaks the trail", () => {
  const blade = new Blade();
  blade.update([100, 300], 0);
  blade.update([200, 300], DT);
  blade.update(null, 2 * DT);
  assert.deepEqual(blade.trail(2 * DT), []);
  assert.ok(!blade.active && !blade.slicing);
});

test("a low-confidence wrist is not tracked", () => {
  const game = newGame();
  game.update(hand(400, 300, 0.1), DT);
  assert.ok(!game.blade.active);
});

function bladeThrough(y, opts = {}) {
  const b = new Blade(opts);
  for (let i = 0, x = 200; x <= 1000; i++, x += 100) b.update([x, y], i * DT);
  return b;
}

test("blade width reaches a snack beside the path", () => {
  const fruit = [600, 380], r = 50; // 80 px off the hand's path
  assert.equal(bladeThrough(300, { hitRadius: 0 }).hits(fruit, r), null);
  assert.notEqual(bladeThrough(300, { hitRadius: 34 }).hits(fruit, r), null);
});

test("the trail bites a snack that arrives after the swing", () => {
  const b = bladeThrough(300, { hitRadius: 0 }); // isolate the trail from blade width
  // On the 700->800 segment, ~0.07 s ago; the newest segment is 150 px away.
  assert.notEqual(b.hits([750, 300], 10), null);
});

test("an old trail goes blunt", () => {
  const b = bladeThrough(300, { cutWindow: 0.15 });
  const tEnd = 8 * DT;
  for (let k = 1; k < 12; k++) b.update([1000, 300], tEnd + k * DT); // hold still ~0.37 s
  assert.equal(b.hits([400, 300], 50), null);
});

test("stepped tracking still counts as a swing (windowed speed)", () => {
  // Frame-to-frame speed alternates between zero and huge; the windowed
  // speed must still read a real swing as one, or bites drop out at random.
  const b = new Blade();
  let t = 0, x = 200;
  for (let s = 0; s < 6; s++) {
    for (let k = 0; k < 3; k++) { // three frames with no new pose
      t += 1 / 60;
      b.update([x, 300], t);
    }
    x += 120;
    t += 1 / 60;
    b.update([x, 300], t); // then a step
    assert.ok(b.slicing, `swing at ~1800 px/s read as idle (speed ${b.speed.toFixed(0)})`);
  }
});

test("a tracking glitch does not cut", () => {
  // A jump across the frame is the tracker losing the hand, not a swing.
  const b = new Blade({ maxJump: 450 });
  b.update([100, 100], 0);
  b.update([110, 105], DT);
  b.update([1100, 650], 2 * DT); // ~1100 px in one sample
  assert.equal(b.hits([600, 380], 60), null);
  assert.equal(b.trail(2 * DT).length, 1, "the glitch should start a fresh stroke");
});

test("the game's glitch guard is 35% of the width", () => {
  assert.equal(newGame().blade.maxJump, W * 0.35);
});

test("hit area matches the raccoon head", () => {
  assert.equal(newGame().blade.hitRadius, 44);
});

// --- flight ---

test("snacks peak inside the play area", () => {
  const sp = new Spawner([W, H], { seed: 3 });
  const fruits = [];
  const art = stubArt().fruits;
  for (let i = 0; i < 40; i++) {
    sp.timer = 0;
    sp.update(DT, fruits, art, 0);
  }
  assert.ok(fruits.length);
  for (const f of fruits.slice(0, 25)) {
    const peak = f.pos[1] - f.vel[1] ** 2 / (2 * GRAVITY);
    assert.ok(peak > 0 && peak < H * 0.9, `apex ${peak.toFixed(0)} outside the screen`);
  }
});

test("opening waves are gentle: one or two snacks, no bombs", () => {
  const sp = new Spawner([W, H], { seed: 5 });
  const art = stubArt().fruits;
  for (let wave = 1; wave <= 6; wave++) {
    const fruits = [];
    sp.timer = 0;
    sp.update(DT, fruits, art, 0);
    assert.ok(fruits.length >= 1 && fruits.length <= 2, `wave ${wave} launched ${fruits.length}`);
    if (wave <= 4) assert.ok(!fruits.some((f) => f.isBomb), `bomb in opening wave ${wave}`);
  }
});

// --- rules ---

test("a bite scores and spawns halves, crumbs and sparks", () => {
  const game = newGame();
  game.fruits.push(still(game.art.fruits[0], 640, 360));
  swipe(game, 200, 1000, 360);
  assert.equal(game.score, 1);
  assert.equal(game.fx.halves.length, 2);
  assert.ok(game.fx.splats.length);
  assert.equal(game.fx.sparks.length, 8);
});

test("dropping a snack costs a life", () => {
  const game = newGame();
  for (let i = 0; i < 300; i++) game.update(nobody(), DT);
  assert.ok(game.lives < game.rules.lives);
});

test("three dropped snacks end the game", () => {
  const game = newGame();
  for (let i = 0; i < 3000 && game.phase !== "game_over"; i++) game.update(nobody(), DT);
  assert.equal(game.phase, "game_over");
  assert.equal(game.lives, 0);
  assert.match(game.deathReason, /GOT AWAY/);
});

test("dropping a bomb is not punished", () => {
  const game = newGame({ rules: { lives: 3 } });
  game.fruits.push(new Fruit({ pos: [640, H], vel: [0, 600], radius: 52, isBomb: true }));
  for (let i = 0; i < 12; i++) game.update(nobody(), DT);
  assert.equal(game.lives, 3);
  assert.equal(game.phase, "playing");
});

test("biting a bomb ends the run", () => {
  const game = newGame();
  game.fruits.push(new Fruit({ pos: [640, 360], vel: [0, 0], radius: 52, isBomb: true }));
  swipe(game, 200, 1000, 360);
  assert.equal(game.phase, "game_over");
  assert.match(game.deathReason, /TRASH/);
  assert.ok(game.fx.shake > 0 && game.fx.sparks.length >= 60);
});

test("one swing through several snacks pays a combo", () => {
  const game = newGame();
  for (let i = 0; i < 4; i++) game.fruits.push(still(game.art.fruits[0], 400 + i * 100, 360));
  swipe(game, 100, 1100, 360);
  assert.equal(game.score, 4); // the bonus is not paid until the swing ends
  for (let i = 0; i < 20; i++) game.update(nobody(), DT);
  assert.ok(game.score > 4);
  assert.equal(game.score, 4 + 4 * 2);
  assert.equal(game.fx.banner.count, 4);
  assert.equal(game.bestCombo, 4);
});

test("two snacks are not a combo", () => {
  const game = newGame();
  for (let i = 0; i < 2; i++) game.fruits.push(still(game.art.fruits[0], 500 + i * 100, 360));
  swipe(game, 100, 1100, 360);
  for (let i = 0; i < 20; i++) game.update(nobody(), DT);
  assert.equal(game.score, 2);
  assert.equal(game.fx.banner.count, 0);
});

test("continuous eating still pays a combo", () => {
  // Without a maximum swing length an unbroken flurry never pays out.
  const game = newGame();
  game.spawner.timer = 1e9;
  for (let s = 0; s < 8; s++) {
    game.fruits.push(still(game.art.fruits[0], 640, 360));
    swipe(game, 520, 760, 360);
    for (let k = 0; k < 3; k++) game.update(hand(760, 360), DT); // bites ~0.2 s apart
  }
  assert.equal(game.slicedTotal, 8);
  assert.ok(game.bestCombo >= game.rules.comboMin, "an unbroken flurry never paid out");
});

test("game over is terminal until reset, and nothing scores after it", () => {
  const game = newGame();
  game.fruits.push(new Fruit({ pos: [640, 360], vel: [0, 0], radius: 52, isBomb: true }));
  swipe(game, 200, 1000, 360);
  assert.equal(game.phase, "game_over");
  const score = game.score;
  game.fruits.push(still(game.art.fruits[0], 640, 300));
  swipe(game, 1000, 200, 300);
  for (let i = 0; i < 60; i++) game.update(nobody(), DT);
  assert.equal(game.phase, "game_over");
  assert.equal(game.score, score, "no scoring once the game is over");
  game.reset();
  assert.equal(game.phase, "playing");
  assert.equal(game.score, 0);
  assert.equal(game.lives, game.rules.lives);
});

test("no spawning after game over", () => {
  const game = newGame();
  game.phase = "game_over";
  const before = game.fruits.length;
  for (let i = 0; i < 90; i++) game.update(nobody(), DT);
  assert.ok(game.fruits.length <= before);
});

test("a stalled frame does not fling everything off-screen", () => {
  const game = newGame();
  game.fruits.push(new Fruit({ pos: [640, 360], vel: [0, -800], radius: 52, art: game.art.fruits[0] }));
  game.update(nobody(), 5.0); // absurd stall
  assert.ok(game.fruits.length && Math.abs(game.fruits[0].pos[1] - 360) < 400);
});

test("an empty manifest fails loudly", () => {
  const game = newGame();
  game.art.fruits.length = 0;
  assert.throws(() => { for (let i = 0; i < 120; i++) game.update(nobody(), DT); },
    /generate_fruitninja_snacks/);
});

test("onResume does not read as a giant swing", () => {
  // The hand moves during a pause while the game clock stands still.
  const game = newGame();
  game.update(hand(200, 300), DT);
  assert.ok(game.blade.active);
  game.onResume();
  assert.ok(!game.blade.active);
  game.update(hand(1000, 300), DT); // the hand is somewhere else now
  assert.ok(!game.blade.slicing, "the first sample after a pause must not be a swing");
});

test("losing the player never costs a life on its own", () => {
  const game = newGame();
  game.spawner.timer = 1e9;
  for (let i = 0; i < 120; i++) game.update(nobody(), DT);
  assert.equal(game.lives, game.rules.lives);
});

// --- variety, fairness, feedback ---

function waveSignature(seed = null, waves = 6) {
  const sp = new Spawner([W, H], { seed });
  const art = stubArt().fruits;
  const out = [];
  for (let i = 0; i < waves; i++) {
    const fruits = [];
    sp.timer = 0;
    sp.update(DT, fruits, art, 3);
    out.push(fruits.map((f) => `${Math.round(f.pos[0])}:${f.art ? f.art.name : "bomb"}`).join(","));
  }
  return out.join("|");
}

test("every unseeded game is different", () => {
  const runs = new Set([0, 1, 2, 3, 4].map(() => waveSignature()));
  assert.equal(runs.size, 5, "unseeded games repeated the same snacks");
});

test("a seed still reproduces a game", () => {
  assert.equal(waveSignature(42), waveSignature(42));
  assert.notEqual(waveSignature(42), waveSignature(43));
});

test("a seeded game's waves do not depend on how the player ate", () => {
  // Crumbs draw from their own stream, so a test's seed pins the board.
  const a = newGame({ seed: 9 }), b = newGame({ seed: 9 });
  a.fruits.push(still(a.art.fruits[0], 640, 360));
  swipe(a, 200, 1000, 360); // 8 updates, one bite and 8 crumbs
  for (let i = 0; i < 8; i++) b.update(hand(200, 360), DT); // same clock, no bite
  assert.equal(a.score, 1);
  assert.equal(b.score, 0);
  for (let i = 0; i < 150; i++) { a.update(nobody(), DT); b.update(nobody(), DT); }
  assert.ok(a.spawner.wave >= 2);
  assert.equal(a.spawner.wave, b.spawner.wave);
  assert.equal(a.spawner.rng._s, b.spawner.rng._s, "spawner stream diverged");
});

test("bombs never crowd a snack (analytic closest approach, checked by stepping)", () => {
  // Checked independently of the spawner's own projection: real game physics,
  // stepped frame by frame for 90 s at the hardest difficulty, measuring
  // every bomb-snack distance while both are on screen.
  const sp = new Spawner([W, H], { seed: 7, clearance: 34 });
  const art = stubArt().fruits;
  let bodies = [];
  let worst = Infinity;
  const dt = 1 / 60;
  for (let i = 0; i < Math.round(90 / dt); i++) {
    sp.update(dt, bodies, art, 12);
    for (const b of bodies) b.step(dt);
    bodies = bodies.filter((b) => b.pos[1] < H + 110);
    const on = bodies.filter((b) => b.pos[0] > -40 && b.pos[0] < W + 40 && b.pos[1] > -60 && b.pos[1] < H + 10);
    for (const bomb of on) {
      if (!bomb.isBomb) continue;
      for (const f of on) {
        if (f.isBomb) continue;
        const gap = Math.hypot(bomb.pos[0] - f.pos[0], bomb.pos[1] - f.pos[1]) - sp.separation(bomb, f);
        worst = Math.min(worst, gap);
      }
    }
  }
  assert.ok(sp.wave > 40, `only ${sp.wave} waves`);
  assert.ok(worst >= -2, `a bomb came ${(-worst).toFixed(0)} px inside the safe distance of a snack`);
});

test("minGap is the exact closest approach (same gravity cancels)", () => {
  const sp = new Spawner([W, H], { seed: 1 });
  // Two bodies on crossing straight-line (relative) paths, both on screen.
  const a = new Fruit({ pos: [300, 400], vel: [400, -900], radius: 50 });
  const b = new Fruit({ pos: [900, 400], vel: [-400, -900], radius: 50, isBomb: true });
  // Relative motion is purely horizontal: they meet at t = 0.75 s (if on screen).
  assert.ok(sp.minGap(a, b) < 1e-6);
  // A pair that never shares the screen is never "close".
  const gone = new Fruit({ pos: [640, H + 400], vel: [0, 0], radius: 50 });
  assert.equal(sp.minGap(a, gone), Infinity);
});

test("bombs still appear at high difficulty", () => {
  // Safety must not quietly remove bombs from the game.
  const sp = new Spawner([W, H], { seed: 3 });
  const art = stubArt().fruits;
  let thrown = 0;
  for (let i = 0; i < 60; i++) {
    const bodies = [];
    sp.timer = 0;
    sp.update(DT, bodies, art, 12);
    thrown += bodies.filter((b) => b.isBomb).length; // judge each wave on its own
  }
  assert.ok(thrown >= 8, `only ${thrown} bombs in 60 hard waves`);
});

test("eating pops up its points, and the score pulses", () => {
  const game = newGame();
  game.spawner.timer = 1e9;
  const one = game.art.fruits[0];
  const two = { ...one, name: "cookie", juice: [240, 200, 60], score: 2 };
  game.fruits.push(still(one, 500, 360));
  game.fruits.push(still(two, 800, 360));
  swipe(game, 300, 1000, 360);
  assert.deepEqual(game.fx.popups.map((p) => p.text).sort(), ["+1", "+2"]);
  assert.ok(game.fx.popups.find((p) => p.text === "+2").big);
  assert.ok(game.fx.scoreBump > 0.5, "the score counter should pulse");
  assert.equal(game.score, 3);
});

test("popups rise and expire", () => {
  const game = newGame();
  game.spawner.timer = 1e9;
  game.fruits.push(still(game.art.fruits[0], 640, 360));
  swipe(game, 400, 900, 360);
  const pop = game.fx.popups[0];
  const y0 = pop.pos[1];
  for (let i = 0; i < 10; i++) game.update(nobody(), DT);
  assert.ok(pop.pos[1] < y0, "popup should float upward");
  for (let i = 0; i < 40; i++) game.update(nobody(), DT);
  assert.equal(game.fx.popups.length, 0, "popups should be gone after about a second");
});

test("popups pop in, overshoot and settle", () => {
  assert.ok(Math.abs(popScale(0) - 0.6) < 1e-9);
  assert.ok(Math.abs(popScale(0.12) - 1.35) < 1e-9);
  assert.equal(popScale(0.5), 1);
  assert.ok(new Popup([0, 0], "+1", [1, 2, 3]).life === 1);
});

test("a bite opens the raccoon's mouth briefly", () => {
  const game = newGame();
  game.spawner.timer = 1e9;
  game.fruits.push(still(game.art.fruits[0], 640, 360));
  swipe(game, 400, 900, 360);
  assert.ok(game.fx.chomp > 0.5, "eating should show the open-mouth frame");
  for (let i = 0; i < 12; i++) game.update(nobody(), DT);
  assert.equal(game.fx.chomp, 0, "the mouth closes again after ~0.2 s");
});

test("a fresh instance per round has full lives and an empty board", () => {
  const g = newGame();
  assert.equal(g.phase, "playing");
  assert.equal(g.lives, 3);
  assert.equal(g.fruits.length, 0);
  assert.equal(SnackAttackGame.id, "snack");
  assert.equal(SnackAttackGame.title, "SNACK ATTACK");
});
