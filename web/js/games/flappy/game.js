// Flappy Bird, flown by flapping your arms (port of src/kpapp/game/flappy/game.py).
//
// The rules are the classic ones: one input (a flap) that sets the bird's
// vertical speed, gravity the rest of the time, a point for every pipe passed,
// and any touch of a pipe or the ground ends the run. The first flap starts play.
//
// What is *not* classic is the tuning. The original is tuned for a thumb on a
// touchscreen: ~20 ms of input lag and taps as fast as you like. Here a flap is a
// whole-arm motion seen by a camera and a pose model running at 15-30 Hz, which
// adds ~100-150 ms before the game even hears about it, and nobody can flap their
// arms much faster than about twice a second. With the original numbers the bird
// falls a third of the gap in that lag alone. Every number below is chosen for
// that input, and tests/flappy.test.mjs flies a bot with a 150 ms handicap and a
// human flap rate through thousands of pipes to prove no level is impossible.
//
// The game never looks at keypoints: FlapDetector turns poses into flap events,
// and the game only reacts to those.
//
// No DOM at module level: this file runs under Node for the tests. The renderer
// is built lazily on the first render().

import { Game } from "../base.js";
import { KP } from "../../core/pose.js";
import { Rng } from "../../core/random.js";
import { FlapDetector } from "./gesture.js";
import { cachedArt, loadArt } from "./art.js";
import { FlappyRenderer } from "./render.js";

export const Phase = Object.freeze({ READY: "ready", PLAYING: "playing", GAME_OVER: "game_over" });

/**
 * Every tuning knob, in px and seconds at 1280x720 and 60 FPS.
 *
 * Vertical numbers are written for a 720 px tall frame and scaled by the real
 * height at construction, so the feel is the same at other resolutions.
 */
export const DEFAULT_RULES = Object.freeze({
  // --- the bird ---
  // Gravity about half the original's (scaled to 720p it is ~2300 px/s^2).
  // Halving it doubles the time the bird takes to fall any distance, which is
  // exactly what absorbs the input lag: in 150 ms the bird now drops at most
  // ~80 px instead of ~150, against a gap of 175-230.
  gravity: 1150.0,
  // A flap *sets* vertical speed (classic), it does not add to it, so a flap
  // always feels the same however the bird was moving. 410 px/s gives an
  // apex 73 px above the flap point (v^2 / 2g) reached in 0.36 s, and a hover
  // cycle (flap, rise, fall back to the same height) of 2v/g = 0.71 s, i.e.
  // ~1.4 flaps a second to hold altitude: a comfortable arm rhythm. A single
  // flap rising 73 px is well under half the smallest gap, so one flap from
  // the bottom of a gap can never carry the bird into the top pipe.
  flapVelocity: 410.0,
  // Capped fall speed. Without it a long dive reaches 900+ px/s and the lag
  // alone costs 135 px. 540 px/s is reached ~0.47 s after the apex, so short
  // hops feel purely ballistic and only real dives hit the cap.
  terminalVelocity: 540.0,
  // Nose angle (degrees, + is nose down) follows vertical speed: pitched up
  // while climbing, diving once the bird falls faster than `diveAfter`.
  // The hero is a front-facing raccoon, not a side-on bird: tipped 75 degrees
  // its face just looks sideways, so the tilt only hints at climb and dive.
  noseUp: -15.0,
  noseDown: 30.0,
  diveAfter: 180.0, // px/s of fall before the nose starts to drop
  noseDropRate: 320.0, // deg/s; the nose snaps up on a flap, sinks smoothly
  // Fraction of width: ~970 px of look-ahead (4.8 s at the start). Also keeps
  // the bird, once it has fallen dead to the ground, clear of the arcade's
  // PLAY AGAIN button, which starts at 28% of the width.
  birdX: 0.24,
  // The flying raccoon is 128x112 with its wings spread; at 0.9x its head is
  // ~52 px across, about the old bird's body, and still reads from two metres
  // back. The hit circle (24 px native -> ~22) scales with it and covers the
  // head and body, so ears, wing tips and tail can graze a pipe and forgive.
  birdScale: 0.9,

  // --- the pipes ---
  // Difficulty eases in and then keeps rising for as long as the player
  // survives. Each knob approaches its limit exponentially:
  //     value(score) = limit + (start - limit) * exp(-score / tau)
  // so it changes fastest early and never flattens out. The old linear ramps
  // hit their floors at score ~30 and stopped changing.
  //
  // Gap: 300 px at the start (48% of the play height, up from 230) to make
  // the first pipes easy to clear while learning to flap; 248 at score 10,
  // 216 at 20, 198 at 30, approaching 175 (24%) without reaching it.
  gapStart: 300.0,
  gapMin: 175.0,
  gapTau: 18.0, // points
  // Horizontal speed: 200 px/s at the start, 220 at 10, 233 at 20, 270 limit.
  // Slow early so a pipe is on screen for ~4.6 s before it reaches the bird.
  speedStart: 200.0,
  speedMax: 270.0,
  speedTau: 25.0, // points
  // Time between consecutive pipes reaching the bird: 2.0 s at the start
  // (400 px apart), toward 1.55 s. The original is ~1.3 s; a player needs time
  // to see the next gap, decide, and get a full-body motion through the camera.
  intervalStart: 2.0,
  intervalMin: 1.55,
  intervalTau: 25.0, // points
  // Gaps stay away from the ceiling and ground: a gap hugging either needs
  // pinpoint flaps, and one at the ground punishes the lag the hardest.
  ceilingMargin: 80.0,
  groundMargin: 70.0,
  // How fast a human can flap their arms, sustained. Consecutive gaps are
  // never further apart vertically than this rate can climb (with a safety
  // factor) in the time between leaving one pipe and entering the next.
  flapPeriod: 0.40,
  reachSafety: 0.80,
  maxDrop: 230.0, // cap on a downward step; diving is easy, but a cliff reads as unfair
  firstPipeDelay: 4.0, // seconds of open sky after the first flap

  // --- feel ---
  maxDt: 0.05, // a stalled frame advances at most this much
  substep: 1.0 / 120.0, // physics step; 4 px a step at worst, so nothing tunnels
  readyBob: 10.0, // px of idle bobbing on the READY screen
});

const VERTICAL_KNOBS = ["gravity", "flapVelocity", "terminalVelocity", "diveAfter", "gapStart",
  "gapMin", "ceilingMargin", "groundMargin", "maxDrop", "readyBob"];

/** A fresh, mutable rule set: defaults, then `overrides`, scaled for a frame `height` px tall. */
export function makeRules(overrides = {}, height = 720) {
  const out = { ...DEFAULT_RULES, ...overrides };
  const k = height / 720.0;
  if (Math.abs(k - 1.0) >= 1e-9) for (const name of VERTICAL_KNOBS) out[name] *= k;
  return out;
}

export const MEDALS = [["platinum", 40], ["gold", 30], ["silver", 20], ["bronze", 10]];

export function medalFor(score) {
  for (const [name, need] of MEDALS) if (score >= need) return name;
  return null;
}

export const POPUP_SECONDS = 0.8;
export const FEATHER_SECONDS = 0.75;
// Flap burst in the raccoon's purples (RGB; Python's BGR tuples reversed).
// White and lavender were tried and vanished against the pale sky and clouds;
// saturated purples hold contrast.
export const FEATHER_TINTS = Object.freeze([[104, 63, 155], [150, 110, 210], [205, 150, 235]]);

/** Exact circle-rectangle overlap: distance to the closest point of the rect. */
function circleHitsRect(cx, cy, r, x0, y0, x1, y1) {
  const nx = Math.min(Math.max(cx, x0), x1);
  const ny = Math.min(Math.max(cy, y0), y1);
  return (cx - nx) ** 2 + (cy - ny) ** 2 < r * r;
}

export class Pipe {
  constructor(x, gapY, gap) {
    this.x = x; // centre
    this.gapY = gapY; // centre of the opening
    this.gap = gap; // height of the opening
    this.scored = false;
  }
  get gapTop() { return this.gapY - this.gap / 2; }
  get gapBottom() { return this.gapY + this.gap / 2; }
}

const freshEffects = () => ({
  feathers: [], // {x, y, vx, vy, angle (deg), spin (deg/s), tint (index), life}
  popups: [], // {x, y, text, life}: floating "+1", pops in, rises, fades
  scoreBump: 0.0, // 1 -> 0 after the score changes, drives the HUD pulse
  flash: 0.0, // white crash flash
  shake: 0.0,
  flap: 0.0, // 1 -> 0 after any recognised flap: the PiP "FLAP!" flash
  flapStrength: 0.0,
});

export class FlappyRaccoonGame extends Game {
  static id = "flappy";
  static title = "FLAPPY RACCOON";
  static blurb = "Flap your arms to fly the raccoon. Don't touch the pipes.";
  static tip = "Stand back so both arms are in frame - flap to fly";
  static menu = true;

  // Session best. The arcade builds a fresh game for every PLAY AGAIN, so a
  // best kept on the instance would be forgotten on every restart; on the
  // class it lasts for the whole session, as the player expects.
  static sessionBest = 0;

  // Room below the shoulder line, in shoulder widths, needed to see a full
  // downstroke. Standing close (as in recorded play, 0.1-0.4 widths) sends
  // the wrists out of the bottom of the frame on every flap; at a 15-20 Hz
  // model rate the gesture detector then misses 6-31% of normal-pace flaps.
  // With the wrists in view it catches them all. Separate on/off levels so
  // the prompt does not flicker at the boundary.
  static TOO_CLOSE_ON = 0.8;
  static TOO_CLOSE_OFF = 1.0;

  static async preload() {
    await loadArt();
  }

  /**
   * opts: {width, height, mirrored, seed, art, rules, detector}
   * `art` (or anything with birdHitRadius, groundHeight and pipeBody.width) and
   * `detector` are injectable so the simulation runs without art or a camera.
   * `rules` overrides individual tuning knobs.
   */
  constructor({ width = 1280, height = 720, mirrored = true, seed = null, art = null,
    rules = null, detector = null } = {}) {
    super({ width, height, mirrored, seed });
    this.rules = makeRules(rules ?? {}, this.height);
    this.art = art ?? cachedArt();
    if (!this.art) {
      throw new Error("Flappy Raccoon art is not loaded: await FlappyRaccoonGame.preload() first");
    }
    this.detector = detector ?? new FlapDetector(mirrored);
    this.seed = seed; // null: a different course every game
    this.pipeWidth = Number(this.art.pipeWidth ?? this.art.pipeBody.width);
    this.hitRadius = Number(this.art.birdHitRadius) * this.rules.birdScale;
    this.groundY = this.height - Number(this.art.groundHeight);
    this._renderer = null;
    this.reset();
  }

  // --- lifecycle ---

  reset() {
    const r = this.rules;
    this.phase = Phase.READY;
    this.tooClose = false; // player too near the camera for flaps to be seen
    this.roomBelow = null;
    this.score = 0;
    this.newBest = false;
    this.rng = new Rng(this.seed);
    // Feathers and shake get their own stream, so juice never changes the
    // course a seed produces (Python used an unseeded numpy rng for them).
    this.fxRng = new Rng(this.seed == null ? null : `${this.seed}:fx`);
    this.bird = { x: this.width * r.birdX, y: this._readyY(), vy: 0.0, angle: 0.0,
      wing: 0.0, wingBoost: 0.0 }; // angle: degrees, + nose down; wing: frames of the flap cycle
    this.pipes = [];
    this.fx = freshEffects();
    this.distance = 0.0; // world scroll in px; drives ground and skyline parallax
    this._clock = 0.0;
    this.deadTime = 0.0; // seconds since the crash
    this.flaps = 0;
    this.lastPose = null; // for the camera window's arm overlay
    this.detector.reset();
  }

  /**
   * The arms kept moving while the clock stood still; without a reset the
   * first frame back would compare against a pose from before the pause and
   * could read as a flap.
   */
  onResume() {
    this.detector.reset();
  }

  get best() { return Math.max(FlappyRaccoonGame.sessionBest, this.score); }
  get medal() { return medalFor(this.score); }
  get clock() { return this._clock; }

  _readyY() { return this.groundY * 0.46; }

  // --- difficulty ---

  gapSize(score = null) {
    const r = this.rules, s = score ?? this.score;
    return r.gapMin + (r.gapStart - r.gapMin) * Math.exp(-s / r.gapTau);
  }

  pipeSpeed(score = null) {
    const r = this.rules, s = score ?? this.score;
    return r.speedMax - (r.speedMax - r.speedStart) * Math.exp(-s / r.speedTau);
  }

  pipeInterval(score = null) {
    const r = this.rules, s = score ?? this.score;
    return r.intervalMin + (r.intervalStart - r.intervalMin) * Math.exp(-s / r.intervalTau);
  }

  get speed() { return this.phase === Phase.PLAYING ? this.pipeSpeed() : 0.0; }

  /**
   * Highest the next gap may sit above the last one.
   *
   * Between leaving one pipe and entering the next the bird covers the
   * spacing minus a pipe width minus its own diameter. Flapping at the
   * human rate `flapPeriod` climbs v*P - g*P^2/2 per flap; the budget is
   * that rate over the transit time, times a safety factor. The reach
   * test verifies it with lag on top.
   */
  climbBudget(spacing, speed) {
    const r = this.rules;
    const transit = Math.max(0.0, spacing - this.pipeWidth - 2 * this.hitRadius) / Math.max(speed, 1e-6);
    const p = r.flapPeriod;
    const perFlap = r.flapVelocity * p - 0.5 * r.gravity * p * p;
    return Math.max(0.0, r.reachSafety * perFlap * transit / p);
  }

  _gapBounds(gap) {
    const r = this.rules;
    const lo = r.ceilingMargin + gap / 2;
    const hi = this.groundY - r.groundMargin - gap / 2;
    return [lo, Math.max(lo, hi)];
  }

  /** A pipe at x whose gap can be reached from `prev`'s. */
  _spawnPipe(x, prev, spacing, speed) {
    const gap = this.gapSize();
    let [lo, hi] = this._gapBounds(gap);
    let centre;
    if (prev == null) {
      centre = (lo + hi) / 2; // the first gap is dead centre: a gentle opener
      centre += this.rng.uniform(-0.25, 0.25) * (hi - lo);
    } else {
      const up = this.climbBudget(spacing, speed);
      lo = Math.max(lo, prev.gapY - up);
      hi = Math.min(hi, prev.gapY + this.rules.maxDrop);
      centre = hi > lo ? this.rng.uniform(lo, hi) : Math.min(Math.max(prev.gapY, lo), hi);
    }
    return new Pipe(x, centre, gap);
  }

  /** Keep one pipe queued past the right edge, spaced by time-to-bird. */
  _fillPipes() {
    const speed = this.pipeSpeed();
    if (!this.pipes.length) {
      let x = this.bird.x + this.rules.firstPipeDelay * speed;
      // Never appear inside the frame; it would pop in under the PiP.
      x = Math.max(x, this.width + this.pipeWidth);
      this.pipes.push(this._spawnPipe(x, null, 0.0, speed));
    }
    while (this.pipes[this.pipes.length - 1].x < this.width + this.pipeWidth) {
      const spacing = speed * this.pipeInterval();
      const last = this.pipes[this.pipes.length - 1];
      this.pipes.push(this._spawnPipe(last.x + spacing, last, spacing, speed));
    }
  }

  // --- framing ---

  _updateFraming(pose) {
    if (pose == null) return;
    const ls = pose.point(KP.left_shoulder, 0.4);
    const rs = pose.point(KP.right_shoulder, 0.4);
    if (ls == null || rs == null) return;
    const width = Math.hypot(ls[0] - rs[0], ls[1] - rs[1]);
    if (width < 1.0) return;
    const room = (this.height - (ls[1] + rs[1]) * 0.5) / width;
    this.roomBelow = room;
    const C = FlappyRaccoonGame;
    this.tooClose = this.tooClose ? room < C.TOO_CLOSE_OFF : room < C.TOO_CLOSE_ON;
  }

  // --- simulation ---

  update(controls, dt) {
    dt = Math.min(Math.max(dt, 0.0), this.rules.maxDt); // a stall must not teleport the bird
    this._clock += dt;

    const pose = controls?.pose ?? null;
    this.lastPose = pose;
    this._updateFraming(pose);
    const event = this.detector.update(pose, this._clock);
    if (event != null) {
      this.fx.flap = 1.0;
      this.fx.flapStrength = Number(event.strength ?? 1.0);
    }
    this._decayEffects(dt);

    if (this.phase === Phase.READY) {
      this._updateReady(dt);
      if (event != null) {
        this.phase = Phase.PLAYING;
        this._fillPipes();
        this._flap(this.fx.flapStrength);
      }
      return;
    }

    if (this.phase === Phase.PLAYING) {
      if (event != null) this._flap(this.fx.flapStrength);
      this._stepWorld(dt);
    } else {
      this.deadTime += dt;
      this._stepCorpse(dt);
    }
    this._stepFeathers(dt);
  }

  _updateReady(dt) {
    const b = this.bird;
    b.y = this._readyY() + this.rules.readyBob * Math.sin(this._clock * 2 * Math.PI * 0.9);
    b.vy = 0.0;
    b.angle = 0.0;
    this._animateWings(dt);
    // The ground keeps rolling on the READY screen, as in the original:
    // the bird reads as flying in place rather than frozen.
    this.distance += this.rules.speedStart * dt;
    this._stepFeathers(dt);
  }

  _flap(strength) {
    const b = this.bird, rng = this.fxRng;
    b.vy = -this.rules.flapVelocity;
    b.angle = this.rules.noseUp; // the nose snaps up at once
    b.wing = 0.0; // wings snap down, then beat fast
    b.wingBoost = 1.0;
    this.flaps += 1;
    const n = 5 + Math.round(4 * Math.min(1.0, Math.max(0.0, strength)));
    const speed = this.speed;
    for (let i = 0; i < n; i++) {
      // Shed downward and backward, against the beat, and left behind by
      // the scrolling world so they hang in the air the bird flew through.
      const ang = rng.uniform(0.35, 2.8); // radians, below the horizon
      const v = rng.uniform(60, 190);
      this.fx.feathers.push({
        x: b.x - 8 + rng.uniform(-6, 6),
        y: b.y + rng.uniform(-4, 10),
        vx: Math.cos(ang) * v - speed * 0.6,
        vy: Math.sin(ang) * v,
        angle: rng.uniform(0, 360),
        spin: rng.uniform(-360, 360),
        tint: rng.randint(0, FEATHER_TINTS.length - 1),
        life: 1.0,
      });
    }
  }

  _animateWings(dt) {
    const b = this.bird;
    b.wing += dt * (7.0 + 13.0 * b.wingBoost); // frames/s: lazy glide -> hard beat
    b.wingBoost = Math.max(0.0, b.wingBoost - dt * 2.5);
  }

  _pitch(dt) {
    const r = this.rules, b = this.bird;
    let target;
    if (b.vy <= r.diveAfter) {
      target = r.noseUp;
    } else {
      const k = (b.vy - r.diveAfter) / Math.max(1.0, r.terminalVelocity - r.diveAfter);
      target = r.noseUp + (r.noseDown - r.noseUp) * Math.min(1.0, k);
    }
    if (target < b.angle) b.angle = Math.max(target, b.angle - r.noseDropRate * 2 * dt);
    else b.angle = Math.min(target, b.angle + r.noseDropRate * dt);
  }

  _stepWorld(dt) {
    const r = this.rules, b = this.bird;
    const n = Math.max(1, Math.ceil(dt / r.substep));
    const h = dt / n;
    for (let i = 0; i < n; i++) {
      const speed = this.pipeSpeed();
      b.vy = Math.min(b.vy + r.gravity * h, r.terminalVelocity);
      b.y += b.vy * h;
      if (b.y < this.hitRadius) { // the ceiling is a soft cap, not a death
        b.y = this.hitRadius;
        b.vy = Math.max(b.vy, 0.0);
      }
      for (const p of this.pipes) p.x -= speed * h;
      this.distance += speed * h;
      this._scorePass();
      if (this._collides()) {
        this._crash();
        break;
      }
    }
    this.pipes = this.pipes.filter((p) => p.x > -this.pipeWidth);
    if (this.phase === Phase.PLAYING) this._fillPipes();
    this._pitch(dt);
    this._animateWings(dt);
  }

  _scorePass() {
    const b = this.bird;
    for (const p of this.pipes) {
      if (!p.scored && p.x <= b.x) {
        p.scored = true;
        this.score += 1;
        this.fx.scoreBump = 1.0;
        this.fx.popups.push({ x: b.x, y: b.y - 36, text: "+1", life: 1.0 });
      }
    }
  }

  _collides() {
    const b = this.bird, r = this.hitRadius;
    if (b.y + r >= this.groundY) return true;
    const half = this.pipeWidth / 2;
    for (const p of this.pipes) {
      if (Math.abs(p.x - b.x) > half + r) continue;
      const x0 = p.x - half, x1 = p.x + half;
      if (circleHitsRect(b.x, b.y, r, x0, -1e6, x1, p.gapTop)
        || circleHitsRect(b.x, b.y, r, x0, p.gapBottom, x1, 1e6)) return true;
    }
    return false;
  }

  _crash() {
    this.phase = Phase.GAME_OVER;
    this.deadTime = 0.0;
    this.fx.flash = 1.0;
    this.fx.shake = 1.0;
    this.bird.vy = Math.max(this.bird.vy, 0.0); // no more lift: it drops from where it hit
    this.bird.y = Math.min(this.bird.y, this.groundY - this.hitRadius); // never sunk into the ground
    if (this.score > FlappyRaccoonGame.sessionBest) {
      this.newBest = true;
      FlappyRaccoonGame.sessionBest = this.score;
    }
  }

  /** After a crash the bird falls nose-first to the ground and stays. */
  _stepCorpse(dt) {
    const r = this.rules, b = this.bird;
    const floor = this.groundY - this.hitRadius;
    if (b.y < floor) {
      b.vy = Math.min(b.vy + r.gravity * 1.4 * dt, r.terminalVelocity * 1.6);
      b.y = Math.min(floor, b.y + b.vy * dt);
    } else {
      b.vy = 0.0;
    }
    b.angle = Math.min(90.0, b.angle + 480.0 * dt);
  }

  get landed() {
    return this.phase === Phase.GAME_OVER && this.bird.y >= this.groundY - this.hitRadius - 0.5;
  }

  _stepFeathers(dt) {
    const drag = Math.max(0.0, 1.0 - 2.2 * dt); // air drag: they float, not fall
    for (const f of this.fx.feathers) {
      f.vx *= drag;
      f.vy *= drag;
      f.vy += 260.0 * dt;
      f.x += f.vx * dt;
      f.y += f.vy * dt;
      f.angle += f.spin * dt;
      f.life -= dt / FEATHER_SECONDS;
    }
    this.fx.feathers = this.fx.feathers.filter((f) => f.life > 0).slice(-40);
  }

  _decayEffects(dt) {
    const fx = this.fx;
    fx.flash = Math.max(0.0, fx.flash - dt * 3.0);
    fx.shake = Math.max(0.0, fx.shake - dt * 2.5);
    fx.scoreBump = Math.max(0.0, fx.scoreBump - dt * 3.5);
    fx.flap = Math.max(0.0, fx.flap - dt * 2.5);
    const speed = this.speed;
    for (const pop of fx.popups) {
      pop.life -= dt / POPUP_SECONDS;
      pop.y -= 70.0 * dt * Math.max(pop.life, 0.0); // rise, easing to a stop
      pop.x -= speed * dt * 0.5; // drift back with the pipe it was for
    }
    fx.popups = fx.popups.filter((p) => p.life > 0).slice(-6);
  }

  // --- rendering ---

  render(ctx, frame) {
    if (!this._renderer) {
      this._renderer = new FlappyRenderer(this.width, this.height, this.art, {
        birdScale: this.rules.birdScale, tints: FEATHER_TINTS, seed: this.seed,
      });
    }
    this._renderer.draw(ctx, frame, this);
  }
}

export default FlappyRaccoonGame;
