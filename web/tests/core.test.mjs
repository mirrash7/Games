// The shared browser core: hand tracking, motion prediction and the arcade shell.
import { test } from "node:test";
import assert from "node:assert/strict";
import { KP, Pose } from "../js/core/pose.js";
import { HandTracker, handKeypoint, handPoint, OneEuroFilter } from "../js/core/hand.js";
import { PoseExtrapolator } from "../js/core/extrapolate.js";
import { ControlMapper } from "../js/core/controls.js";
import { Rng } from "../js/core/random.js";
import { Dwell, Shell, MENU_DWELL } from "../js/shell.js";
import { Game } from "../js/games/base.js";

function pose(points, conf = 0.9) {
  const xy = new Float32Array(34), c = new Float32Array(17);
  for (const [name, [x, y]] of Object.entries(points)) {
    xy[2 * KP[name]] = x;
    xy[2 * KP[name] + 1] = y;
    c[KP[name]] = conf;
  }
  return new Pose(xy, c);
}

const SHOULDERS = { left_shoulder: [700, 500], right_shoulder: [560, 500] };

test("mirrored feed: the player's real right hand is left_wrist", () => {
  assert.equal(handKeypoint("right", true), KP.left_wrist);
  assert.equal(handKeypoint("left", true), KP.right_wrist);
  assert.equal(handKeypoint("right", false), KP.right_wrist);
});

test("palm sits past the wrist along the forearm", () => {
  const p = pose({ ...SHOULDERS, left_elbow: [800, 400], left_wrist: [800, 300] });
  const [x, y] = handPoint(p);
  assert.equal(x, 800);
  assert.ok(Math.abs(y - (300 - 0.38 * 100)) < 1e-3);
});

test("no elbow: the forearm is assumed upright, sized from the shoulders", () => {
  const p = pose({ ...SHOULDERS, left_wrist: [800, 300] });
  const [x, y] = handPoint(p);
  assert.equal(x, 800);
  assert.ok(Math.abs(y - (300 - 0.38 * 0.76 * 140)) < 1e-3);
});

test("the tracker remembers the forearm across a short elbow drop-out", () => {
  const tr = new HandTracker("right", true, { smooth: false });
  const withElbow = pose({ ...SHOULDERS, left_elbow: [900, 350], left_wrist: [800, 300] });
  const without = pose({ ...SHOULDERS, left_wrist: [800, 300] });
  const a = tr.update(withElbow, 0);
  const b = tr.update(without, 0.3);
  assert.deepEqual(a, b, "no jump when the elbow vanishes");
  const c = tr.update(without, 1.0);
  assert.notDeepEqual(c, a, "after FOREARM_MEMORY it falls back to upright");
});

test("One Euro: steadies a still hand, follows a fast one", () => {
  const rng = new Rng(1);
  const f = new OneEuroFilter();
  let raw = 0, filtered = 0;
  for (let i = 0; i < 240; i++) {
    const nx = rng.gauss(0, 4);
    const out = f.filter([500 + nx, 300 + rng.gauss(0, 4)], i / 60);
    if (i > 30) {
      raw += nx * nx;
      filtered += (out[0] - 500) ** 2;
    }
  }
  assert.ok(Math.sqrt(filtered) < 0.6 * Math.sqrt(raw), "jitter at least ~halved");
  const g = new OneEuroFilter();
  let out;
  for (let i = 0; i <= 30; i++) out = g.filter([100 + 2000 * (i / 60), 300], i / 60);
  assert.ok(1100 - out[0] < 40, `lag on a 2000 px/s swipe ${(1100 - out[0]).toFixed(1)} px`);
});

test("extrapolator projects along velocity, capped, and drops stale velocity", () => {
  const ex = new PoseExtrapolator({ gain: 1, maxLead: 0.1 });
  ex.update([pose({ left_wrist: [100, 100] })], 0);
  ex.update([pose({ left_wrist: [110, 100] })], 0.05); // 200 px/s
  const ahead = ex.posesAt(0.1)[0];
  assert.ok(Math.abs(ahead.x(KP.left_wrist) - 120) < 1e-3);
  const capped = ex.posesAt(5)[0];
  assert.ok(Math.abs(capped.x(KP.left_wrist) - 130) < 1e-3, "lead clamped to maxLead");
  ex.update([pose({ left_wrist: [500, 100] })], 2.0); // stalled for 2 s
  assert.equal(ex.posesAt(2.1)[0].x(KP.left_wrist), 500);
});

test("extrapolator ignores a keypoint that was invisible in the previous result", () => {
  const ex = new PoseExtrapolator({ gain: 1 });
  ex.update([pose({ left_wrist: [100, 100] }, 0.1)], 0);
  ex.update([pose({ left_wrist: [400, 100] })], 0.05);
  assert.equal(ex.posesAt(0.1)[0].x(KP.left_wrist), 400);
});

test("controls: hands up latches with hysteresis", () => {
  const m = new ControlMapper(1280, 720);
  const up = pose({ ...SHOULDERS, left_wrist: [700, 300], right_wrist: [560, 300] });
  const mid = pose({ ...SHOULDERS, left_wrist: [700, 420], right_wrist: [560, 420] });
  assert.equal(m.map(up).handsUp, true);
  assert.equal(m.map(mid).handsUp, true, "stays on between the off and on thresholds");
  assert.equal(new ControlMapper(1280, 720).map(mid).handsUp, false);
});

test("seeded Rng repeats; unseeded games differ", () => {
  const a = new Rng(7), b = new Rng(7);
  assert.deepEqual([a.random(), a.random()], [b.random(), b.random()]);
  assert.notEqual(new Rng().random(), new Rng().random());
  const r = new Rng(3);
  for (let i = 0; i < 200; i++) {
    const v = r.randint(1, 3);
    assert.ok(v >= 1 && v <= 3);
  }
});

test("dwell fires once, then not again until the hand leaves", () => {
  const d = new Dwell(1.0);
  let fired = 0;
  for (let i = 0; i < 300; i++) if (d.update("a", 1 / 60)) fired++;
  assert.equal(fired, 1);
  d.update(null, 1 / 60);
  for (let i = 0; i < 70; i++) if (d.update("a", 1 / 60)) fired++;
  assert.equal(fired, 2);
});

test("dwell blocked on a screen change ignores a hand already over a button", () => {
  const d = new Dwell(0.5);
  d.block();
  for (let i = 0; i < 120; i++) assert.equal(d.update("x", 1 / 60), null);
});

class Dummy extends Game {
  static id = "dummy";
  static title = "DUMMY";
  constructor(o) {
    super(o);
    this.resumed = 0;
  }
  onResume() {
    this.resumed++;
  }
}

const noControls = { present: false, pose: null };

test("shell: menu -> countdown -> play -> pause -> resume -> game over -> again", () => {
  const sh = new Shell({ games: [Dummy] });
  assert.equal(sh.screen, "menu");
  sh.handleKey("1");
  assert.equal(sh.screen, "countdown");
  for (let i = 0; i < 200; i++) sh.update(noControls, 1 / 60);
  assert.equal(sh.screen, "playing");
  sh.handleKey(" ");
  assert.equal(sh.screen, "paused");
  sh.handleKey(" ");
  assert.equal(sh.game.resumed, 1);
  const first = sh.game;
  first.phase = "game_over";
  sh.handleKey("r");
  assert.notEqual(sh.game, first, "a fresh instance for every replay");
  sh.handleKey("Escape");
  sh.handleKey("Escape");
  assert.equal(sh.screen, "menu");
});

test("shell: hovering a menu tile with the hand starts that game", () => {
  const sh = new Shell({ games: [Dummy] });
  const [x0, y0, x1, y1] = sh._layout()[0].rect;
  // Palm of the real right hand (left_wrist on a mirrored feed), elbow below.
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  // Forearm 100 px straight up, so the palm sits 38 px above the wrist: on the tile centre.
  const p = pose({ ...SHOULDERS, left_elbow: [cx, cy + 138], left_wrist: [cx, cy + 38] });
  const controls = { present: true, pose: p };
  for (let i = 0; i < Math.ceil(MENU_DWELL * 60) + 10 && sh.screen === "menu"; i++) sh.update(controls, 1 / 60);
  assert.equal(sh.screen, "countdown");
});

test("shell: clicking a tile starts it; clicking PAUSE pauses", () => {
  const sh = new Shell({ games: [Dummy] });
  const [x0, y0] = sh._layout()[0].rect;
  sh.handleClick(x0 + 10, y0 + 10);
  assert.equal(sh.screen, "countdown");
  for (let i = 0; i < 200; i++) sh.update(noControls, 1 / 60);
  sh.handleClick(1200, 670);
  assert.equal(sh.screen, "paused");
});
