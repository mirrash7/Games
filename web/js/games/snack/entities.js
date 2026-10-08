// Tossed objects and their physics (port of game/fruitninja/entities.py).
// Names keep the Fruit Ninja theme: a "fruit" is a snack, a "bomb" is the bag
// of trash. Positions and velocities are mutable [x, y] arrays.

import { Rng } from "../../core/random.js";

export const GRAVITY = 1500.0; // px/s^2
export const POPUP_SECONDS = 0.9;

export class Body {
  constructor({ pos, vel, angle = 0, spin = 0, radius = 50 }) {
    this.pos = [pos[0], pos[1]];
    this.vel = [vel[0], vel[1]];
    this.angle = angle;
    this.spin = spin; // radians/s
    this.radius = radius;
  }

  step(dt, gravity = GRAVITY) {
    this.vel[1] += gravity * dt;
    this.pos[0] += this.vel[0] * dt;
    this.pos[1] += this.vel[1] * dt;
    this.angle += this.spin * dt;
  }
}

export class Fruit extends Body {
  constructor({ art = null, isBomb = false, sliced = false, ...body }) {
    super(body);
    this.art = art;
    this.isBomb = isBomb;
    this.sliced = sliced;
  }

  get score() {
    return this.art ? this.art.score : 1;
  }
}

export class Half extends Body {
  constructor({ sprite = null, life = 1, ...body }) {
    super(body);
    this.sprite = sprite;
    this.life = life;
  }

  step(dt, gravity = GRAVITY) {
    super.step(dt, gravity);
    // Halves fade rather than vanish, so a bite reads as a follow-through.
    this.life = Math.max(0, this.life - dt * 0.55);
  }
}

export class Splat {
  constructor(pos, colour, scale = 1, life = 1) {
    this.pos = [pos[0], pos[1]];
    this.colour = colour; // [r, g, b]
    this.scale = scale;
    this.life = life;
  }
}

/** Floating "+1" at the point of a cut: pops in, rises, fades. */
export class Popup {
  constructor(pos, text, colour, big = false, life = 1) {
    this.pos = [pos[0], pos[1]];
    this.text = text;
    this.colour = colour;
    this.big = big;
    this.life = life; // 1 -> 0 over POPUP_SECONDS
  }
}

export class Spark {
  constructor(pos, vel, colour, life = 1) {
    this.pos = [pos[0], pos[1]];
    this.vel = [vel[0], vel[1]];
    this.colour = colour;
    this.life = life;
  }
}

/**
 * Tosses waves of snacks up from below the bottom edge.
 *
 * Launch speed is solved backwards from a target apex so snacks always peak
 * inside the play area. Every game is different (seed = null); a seed is only
 * for tests.
 *
 * Bombs are only thrown where they can be avoided. Everything falls under the
 * same gravity, so whole flight paths are projected at launch; a bomb is placed
 * only if it stays at least `separation` from every snack for as long as both
 * are on screen. If no safe path is found it is not thrown.
 */
export class Spawner {
  static TRIES = 24;
  static HORIZON = 3.2; // seconds of flight to check
  // On-screen bounds, generous at the edges where objects enter and leave.
  static X_MIN = -40.0;
  static Y_MIN = -60.0;
  static Y_PAD = 10.0;
  static X_PAD = 40.0;

  constructor([width, height], { seed = null, clearance = 34.0 } = {}) {
    this.width = width;
    this.height = height;
    this.rng = new Rng(seed); // null -> fresh entropy each game
    this.clearance = clearance; // the blade's half-width
    this.timer = 1.6; // a beat after the start before anything flies
    this.wave = 0;
    this.bombsDropped = 0; // bombs skipped because no safe path existed
  }

  /** Centre distance at which a snack can be eaten without touching a bomb. */
  separation(a, b) {
    return a.radius + b.radius + this.clearance + 30.0;
  }

  /**
   * Time intervals (from now) during which a body is on screen. Solved, not
   * sampled: x(t) is linear and y(t) a parabola, so each bound is a root.
   * Sampling at 1/30 s once missed a snack rising past a falling bomb.
   */
  _onScreen(body) {
    const S = Spawner;
    const [x0, y0] = body.pos;
    const [vx, vy] = body.vel;
    const g = GRAVITY;

    // Interval where y(t) < limit (y opens upward under gravity).
    const quadBelow = (limit) => {
      const c = y0 - limit;
      const disc = vy * vy - 2 * g * c;
      if (disc <= 0) return null;
      const r = Math.sqrt(disc);
      return [(-vy - r) / g, (-vy + r) / g];
    };

    let lo = 0, hi = S.HORIZON;
    const belowBottom = quadBelow(this.height + S.Y_PAD);
    if (belowBottom === null) return [];
    lo = Math.max(lo, belowBottom[0]);
    hi = Math.min(hi, belowBottom[1]);

    if (Math.abs(vx) > 1e-9) {
      const ta = (S.X_MIN - x0) / vx;
      const tb = (this.width + S.X_PAD - x0) / vx;
      lo = Math.max(lo, Math.min(ta, tb));
      hi = Math.min(hi, Math.max(ta, tb));
    } else if (!(S.X_MIN < x0 && x0 < this.width + S.X_PAD)) {
      return [];
    }
    if (lo >= hi) return [];

    // Above the top edge: invisible between these roots, if it gets that high.
    const aboveTop = quadBelow(S.Y_MIN);
    if (aboveTop === null) return [[lo, hi]];
    const out = [];
    if (lo < aboveTop[0]) out.push([lo, Math.min(hi, aboveTop[0])]);
    if (hi > aboveTop[1]) out.push([Math.max(lo, aboveTop[1]), hi]);
    return out.filter((iv) => iv[0] < iv[1]);
  }

  /**
   * Exact closest approach while both are on screen (Infinity if never). The
   * shared gravity cancels: their separation moves in a straight line, and its
   * closest point over each shared on-screen interval is a clamp.
   */
  minGap(a, b) {
    const r0x = a.pos[0] - b.pos[0], r0y = a.pos[1] - b.pos[1];
    const vx = a.vel[0] - b.vel[0], vy = a.vel[1] - b.vel[1];
    const vv = vx * vx + vy * vy;
    let best = Infinity;
    const aOn = this._onScreen(a), bOn = this._onScreen(b);
    for (const [a0, a1] of aOn) {
      for (const [b0, b1] of bOn) {
        const t0 = Math.max(a0, b0), t1 = Math.min(a1, b1);
        if (t0 >= t1) continue;
        let t = vv < 1e-12 ? 0 : -(r0x * vx + r0y * vy) / vv;
        t = Math.min(Math.max(t, t0), t1);
        best = Math.min(best, Math.hypot(r0x + vx * t, r0y + vy * t));
      }
    }
    return best;
  }

  /** How far the tightest pairing is from being unsafe (negative = unsafe). */
  _clearanceMargin(body, others) {
    let m = Infinity;
    for (const o of others) m = Math.min(m, this.minGap(body, o) - this.separation(body, o));
    return m;
  }

  _launch(art, isBomb, delayRows = 0) {
    const w = this.width, h = this.height, rng = this.rng;
    const x = rng.uniform(w * 0.12, w * 0.88);
    const apex = rng.uniform(h * 0.10, h * 0.42);
    const rise = (h + 60) - apex;
    const vy = -Math.sqrt(2 * GRAVITY * rise);
    // Drift back toward the middle so nothing exits sideways immediately.
    const vx = (w * 0.5 - x) * rng.uniform(0.25, 0.75) + rng.uniform(-90, 90);
    const radius = art ? art.radius : 52;
    return new Fruit({
      // Stagger a wave slightly so they do not leave as one clump.
      pos: [x, h + 60 + delayRows * 26],
      vel: [vx, vy],
      angle: rng.uniform(0, 6.28),
      spin: rng.uniform(-3, 3),
      radius,
      art,
      isBomb,
    });
  }

  update(dt, fruits, artList, level) {
    this.timer -= dt;
    if (this.timer > 0) return;

    if (!artList || !artList.length) {
      throw new Error("no snacks in the manifest - run tools/generate_fruitninja_snacks.py");
    }
    const rng = this.rng;
    this.wave += 1;
    // Recorded play ended three rounds in ~3 s each: openings of two to five
    // were too much while a player is still finding the blade. Start with one
    // or two, ramp up, and keep bombs out of the first few waves entirely.
    let count;
    if (level < 2) count = 1 + rng.randint(0, 1);
    else count = Math.min(5, 2 + rng.randint(0, Math.min(3, Math.floor(level / 2))));
    const bombChance = this.wave <= 4 ? 0 : Math.min(0.3, 0.05 + level * 0.025);

    const kinds = [];
    for (let i = 0; i < count; i++) kinds.push(rng.random() < bombChance);
    if (kinds.every(Boolean)) kinds[0] = false; // a wave is never all bombs
    const airborne = fruits.filter((f) => !f.sliced);
    const liveFruit = airborne.filter((f) => !f.isBomb);
    const liveBombs = airborne.filter((f) => f.isBomb);

    // Snacks first, kept clear of any bomb already in the air. A snack with no
    // safe path sits this wave out rather than spawn beside a bomb.
    const newFruit = [];
    kinds.forEach((isBomb, row) => {
      if (isBomb) return;
      const art = rng.choice(artList);
      const tries = liveBombs.length ? Spawner.TRIES : 1;
      for (let k = 0; k < tries; k++) {
        const cand = this._launch(art, false, row);
        if (this._clearanceMargin(cand, liveBombs) >= 0) {
          newFruit.push(cand);
          break;
        }
      }
    });
    fruits.push(...newFruit);

    // Then bombs, which are optional: only thrown on a provably safe path.
    const targets = liveFruit.concat(newFruit);
    kinds.forEach((isBomb, row) => {
      if (!isBomb) return;
      for (let k = 0; k < Spawner.TRIES; k++) {
        const cand = this._launch(null, true, row);
        if (this._clearanceMargin(cand, targets) >= 0) {
          fruits.push(cand);
          return;
        }
      }
      this.bombsDropped += 1;
    });

    const gap = Math.max(0.8, 2.2 - level * 0.09);
    this.timer = rng.uniform(gap, gap + 0.7);
  }
}
