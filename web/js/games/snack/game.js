// Snack Attack - Fruit Ninja rules, with a raccoon for a hand
// (port of game/fruitninja/game.py).
//
// The player's palm is a raccoon that gobbles flying snacks and must stay away
// from bags of trash. Identifiers keep their Fruit Ninja names: a "fruit" is a
// snack, a "bomb" is the bag of trash, "slicing" is the raccoon taking a bite.
//
// Scoring, lives and combos follow the arcade original: a point per snack, a
// bonus for several in one swing, three misses and you are out, and a bomb
// ends the run immediately. Everything the player does comes through `Blade`,
// fed the palm position each frame; the game never looks at keypoints itself.
//
// No DOM at import time: the renderer is only built on the first render().

import { Game } from "../base.js";
import { HandTracker } from "../../core/hand.js";
import { Rng } from "../../core/random.js";
import { loadArt, cachedArt } from "./art.js";
import { Blade, CUT_WINDOW, MIN_SLICE_SPEED } from "./blade.js";
import { POPUP_SECONDS, Half, Popup, Spark, Splat, Spawner } from "./entities.js";
import { SnackRenderer } from "./render.js";

export const COMBO_WINDOW = 0.40; // bites this close together count as one swing
export const COMBO_MIN = 3;
const BOMB_SPARK = [255, 60, 60]; // RGB (Python BGR (60, 60, 255))

/** The tuning knobs, as in Python's `Rules` dataclass. */
export function defaultRules(overrides = {}) {
  return {
    lives: 3,
    comboWindow: COMBO_WINDOW,
    comboMin: COMBO_MIN,
    maxSwing: 1.2, // a swing this long is settled up even if unbroken
    // Blade half-width, px. Wider than the plain blade's 34: the raccoon head
    // drawn on the palm is ~100 px across, and players judge a bite by the
    // head they see, so the hit area matches it.
    hitRadius: 44.0,
    cutWindow: CUT_WINDOW, // seconds the trail stays sharp
    minSpeed: MIN_SLICE_SPEED, // px/s a swing needs to cut
    ...overrides,
  };
}

function newEffects() {
  return {
    halves: [],
    splats: [],
    sparks: [],
    popups: [],
    chomp: 0, // 1 -> 0 after a bite: the raccoon shows its open mouth
    scoreBump: 0, // 1 -> 0 after the score changes, drives the HUD pulse
    flash: 0,
    shake: 0,
    banner: { count: 0, bonus: 0, life: 0 },
  };
}

export class SnackAttackGame extends Game {
  static id = "snack";
  static title = "SNACK ATTACK";
  static blurb = "Your hand is a raccoon. Gobble the snacks, dodge the trash bags.";
  static menu = true;
  static tip = "Stand back so your upper body is in frame";

  static async preload() {
    await loadArt();
  }

  constructor({ width = 1280, height = 720, mirrored = true, seed = null, art = null,
    rules = null, hand = "right" } = {}) {
    super({ width, height, mirrored, seed });
    this.rules = defaultRules(rules ?? {});
    // Injectable so the simulation can be exercised without the generated art.
    this.art = art ?? cachedArt();
    if (!this.art) throw new Error("Snack Attack art is not loaded: await SnackAttackGame.preload() first");
    this.hand = hand;
    this.tracker = new HandTracker(hand, mirrored);
    this.blade = new Blade({
      hitRadius: this.rules.hitRadius,
      cutWindow: this.rules.cutWindow,
      minSpeed: this.rules.minSpeed,
      maxJump: this.width * 0.35,
    });
    this._renderer = null;
    this.reset();
  }

  // --- lifecycle ---

  reset() {
    this.phase = "playing";
    this.score = 0;
    this.lives = this.rules.lives;
    this.bestCombo = 0;
    this.slicedTotal = 0;
    this.fruits = [];
    this.fx = newEffects();
    this.spawner = new Spawner([this.width, this.height], {
      seed: this.seed, clearance: this.rules.hitRadius,
    });
    // Cosmetic randomness (crumbs) on its own stream, so a seeded game's waves
    // do not depend on how the player happened to play.
    this.fxRng = new Rng(this.seed == null ? null : `${this.seed}:fx`);
    this.blade.loseTrack();
    this.tracker.reset();
    this._clock = 0;
    this._swing = []; // timestamps of recent bites
    this.deathReason = "";
  }

  /** Back from pause: the hand moved while the clock stood still, so the old
   * trail would read as one impossibly fast swing. */
  onResume() {
    this.blade.loseTrack();
    this.tracker.reset();
  }

  /** Flip which hand drives the blade, for when the mirror guess is wrong. */
  swapHand() {
    this.hand = this.hand === "right" ? "left" : "right";
    this.tracker = new HandTracker(this.hand, this.mirrored);
    this.blade.loseTrack();
  }

  /** Seconds of gameplay elapsed; the timebase for blade samples. */
  get clock() {
    return this._clock;
  }

  get level() {
    return Math.floor(this.slicedTotal / 8);
  }

  // --- simulation ---

  update(controls, dt) {
    dt = Math.min(dt, 0.05); // a stall must not fling everything off-screen
    this._clock += dt;

    this._trackHand(controls);
    this._decayEffects(dt);

    if (this.phase === "game_over") {
      this._stepBodies(dt);
      return;
    }

    this.spawner.update(dt, this.fruits, this.art.fruits, this.level);
    this._slicePass();
    this._stepBodies(dt);
    this._cull();
    this._resolveCombo();
  }

  _trackHand(controls) {
    const pose = controls ? controls.pose : null;
    if (!pose) {
      this.blade.loseTrack();
      this.tracker.reset();
      return;
    }
    this.blade.update(this.tracker.update(pose, this._clock), this._clock);
  }

  /** Test this frame's swept blade against everything airborne. */
  _slicePass() {
    if (!this.blade.slicing) return;
    for (const fruit of this.fruits) {
      if (fruit.sliced) continue;
      const direction = this.blade.hits(fruit.pos, fruit.radius);
      if (!direction) continue;
      fruit.sliced = true;
      if (fruit.isBomb) {
        this._explode(fruit);
        return;
      }
      this._sliceFruit(fruit, direction);
    }
  }

  _sliceFruit(fruit, direction) {
    const art = fruit.art;
    this.score += fruit.score;
    this.slicedTotal += 1;
    this._swing.push(this._clock);

    // Halves separate along the cut normal, so the split follows the swing.
    const normal = [-direction[1], direction[0]];
    for (const [sprite, sign] of [[art.halfA, 1], [art.halfB, -1]]) {
      this.fx.halves.push(new Half({
        pos: fruit.pos,
        vel: [fruit.vel[0] + normal[0] * sign * 190, fruit.vel[1] + normal[1] * sign * 190],
        angle: fruit.angle,
        spin: fruit.spin + sign * 2.6,
        radius: fruit.radius,
        sprite,
      }));
    }

    this.fx.splats.push(new Splat(fruit.pos, art.juice, 1.0));
    this.fx.chomp = 1;
    this.fx.popups.push(new Popup(fruit.pos, `+${fruit.score}`, art.juice, fruit.score > 1));
    this.fx.scoreBump = 1;
    for (let i = 0; i < 8; i++) {
      const v = [this.fxRng.gauss(0, 240), this.fxRng.gauss(0, 240)];
      this.fx.sparks.push(new Spark(fruit.pos, v, art.juice));
    }
    this.fx.flash = 1;
  }

  _explode(bomb) {
    this.phase = "game_over";
    this.deathReason = "THE RACCOON GRABBED THE TRASH";
    this.fx.shake = 1;
    this.fx.flash = 1;
    for (let i = 0; i < 60; i++) {
      const v = [this.fxRng.gauss(0, 520), this.fxRng.gauss(0, 520)];
      this.fx.sparks.push(new Spark(bomb.pos, v, BOMB_SPARK));
    }
  }

  _stepBodies(dt) {
    for (const f of this.fruits) f.step(dt);
    for (const h of this.fx.halves) h.step(dt);
    for (const s of this.fx.sparks) {
      s.vel[1] += 900 * dt;
      s.pos[0] += s.vel[0] * dt;
      s.pos[1] += s.vel[1] * dt;
      s.life -= dt * 1.5;
    }
  }

  _cull() {
    const floor = this.height + 110;
    const kept = [];
    for (const f of this.fruits) {
      if (f.pos[1] <= floor) {
        kept.push(f);
        continue;
      }
      // A dropped snack costs a life; a dropped bomb is a relief, not a loss.
      if (!f.sliced && !f.isBomb) {
        this.lives -= 1;
        if (this.lives <= 0) {
          this.phase = "game_over";
          this.deathReason = "TOO MANY SNACKS GOT AWAY";
        }
      }
    }
    this.fruits = kept;
    this.fx.halves = this.fx.halves.filter((h) => h.life > 0 && h.pos[1] < floor);
    this.fx.sparks = this.fx.sparks.filter((s) => s.life > 0);
  }

  /** A swing that ate several snacks pays a bonus once it has ended. */
  _resolveCombo() {
    if (!this._swing.length) return;
    const stillSwinging = this._clock - this._swing[this._swing.length - 1] < this.rules.comboWindow;
    // A player eating continuously never leaves a gap, so without this the
    // swing would never close and the bonus would never be paid.
    const overrun = this._clock - this._swing[0] > this.rules.maxSwing;
    if (stillSwinging && !overrun) return;
    const count = this._swing.length;
    if (count >= this.rules.comboMin) {
      const bonus = count * 2;
      this.score += bonus;
      this.bestCombo = Math.max(this.bestCombo, count);
      this.fx.banner = { count, bonus, life: 1 };
      this.fx.scoreBump = 1;
    }
    this._swing.length = 0;
  }

  _decayEffects(dt) {
    const fx = this.fx;
    fx.flash = Math.max(0, fx.flash - dt * 4);
    fx.shake = Math.max(0, fx.shake - dt * 1.6);
    fx.banner.life = Math.max(0, fx.banner.life - dt * 0.7);
    fx.scoreBump = Math.max(0, fx.scoreBump - dt * 3.5);
    fx.chomp = Math.max(0, fx.chomp - dt / 0.22); // mouth stays open ~0.2 s
    for (const pop of fx.popups) {
      pop.life -= dt / POPUP_SECONDS;
      pop.pos[1] -= 85 * dt * Math.max(pop.life, 0); // rise, easing to a stop
    }
    fx.popups = fx.popups.filter((p) => p.life > 0).slice(-12);
    for (const s of fx.splats) s.life -= dt * 0.5;
    // Debris is cosmetic and capped; oldest goes first, the least noticeable.
    fx.splats = fx.splats.filter((s) => s.life > 0).slice(-10);
    fx.sparks = fx.sparks.slice(-110);
    fx.halves = fx.halves.slice(-18);
  }

  // --- rendering ---

  render(ctx, _frame) {
    if (!this._renderer) this._renderer = new SnackRenderer(this.width, this.height, this.art);
    this._renderer.draw(ctx, this);
  }
}

export default SnackAttackGame;
