// Ball flight, bounce and roll: pure, fixed-step and deterministic
// (port of src/kpapp/game/golf/physics.py - keep the two in step;
// web/tests/golf_parity.test.mjs replays the Python's shots).
//
// One Flight per stroke. It advances in fixed SUBSTEP steps whatever the frame
// rate, so the same shot lands on the same spot at 30 or 120 FPS and in the
// desktop build (plain floats, no randomness: any randomness - the
// overswing's hook or slice - is decided by the game before launch).
//
// The model, per substep, in metres and seconds:
//
//     air:  a = gravity
//               - DRAG |v_r| v_r                     (v_r: velocity relative to the wind)
//               + LIFT spin |v_r|^2 n_up             (backspin: n_up is v_r turned 90 degrees up)
//               + SIDE curve spin |v_r|^2 n_right    (hook/slice: curve -1 hook .. +1 slice)
//           spin decays with SPIN_TAU.
//     land: a bounce keeps RESTITUTION of the vertical speed and RETAIN of the
//           horizontal (times the club's `bite` on the first bounce), until the
//           vertical speed is under ROLL_BELOW, then it rolls.
//     roll: constant deceleration per surface (FRICTION), plus the green's slope
//           (5/7 g grad h, a rolling ball) on the green.
//
// Clubs carry about their nominal yardage at full power with no wind. Power
// scales the launch speed by power^POWER_EXP, so carry is close to
// proportional to power: the gauge reads as a percentage of the club's distance.

import { YARDS_PER_M } from "./course.js";

export const GRAVITY = 9.81;
export const SUBSTEP = 1.0 / 120.0;
export const MAX_TIME = 40.0;

export const DRAG = 0.0050;
export const LIFT = 0.0042;
export const SIDE = 0.0026;
export const SPIN_TAU = 5.0;

// Per surface: vertical restitution, horizontal speed kept on the first
// bounce, kept on later bounces, rolling deceleration (m/s^2).
export const SURFACES = {
  tee: [0.30, 0.45, 0.75, 2.4],
  fairway: [0.30, 0.45, 0.75, 2.4],
  rough: [0.18, 0.25, 0.50, 5.5],
  green: [0.22, 0.30, 0.70, 0.65],
  bunker: [0.02, 0.05, 0.10, 14.0],
};
export const ROLL_BELOW = 1.2;
export const STOP_SPEED = 0.06;
export const SLOPE_GAIN = (5.0 / 7.0) * GRAVITY;

// The cup is played 0.10 m in radius (about twice the real one): a putt's line
// and speed come from a camera-tracked arm with ~100 ms of lag.
export const CUP_RADIUS = 0.10;
export const CAPTURE_SPEED = 1.7;
export const DUNK_SPEED = 9.0;

export const LIE_FACTOR = { tee: 1.0, fairway: 1.0, green: 1.0, rough: 0.8, bunker: 0.55 };

export const POWER_EXP = 0.58;
export const OVERSWING_MAX = 1.15;
export const THIN_DISTANCE = 0.7;
export const THIN_LAUNCH = 0.6;

/** Python's math.radians, operation for operation. */
export const radians = (d) => d * (Math.PI / 180.0);

const surface = (lie) => SURFACES[lie] ?? SURFACES.rough;
export const lieFactor = (lie) => LIE_FACTOR[lie] ?? 1.0;

export class Club {
  constructor(key, name, carryYd, speed, launch, spin, bite, teeOnly = false) {
    Object.assign(this, { key, name, carryYd, speed, launch, spin, bite, teeOnly });
    Object.freeze(this);
  }
  get carryM() { return this.carryYd / YARDS_PER_M; }
  get putter() { return this.key === "PT"; }
}

export const CLUBS = Object.freeze([
  new Club("1W", "DRIVER", 230, 74.3, 12.0, 0.82, 1.00, true),
  new Club("3W", "3 WOOD", 210, 66.7, 13.5, 0.90, 0.85),
  new Club("5I", "5 IRON", 180, 56.5, 17.0, 1.00, 0.60),
  new Club("7I", "7 IRON", 155, 48.6, 21.0, 1.10, 0.40),
  new Club("9I", "9 IRON", 130, 41.6, 27.0, 1.20, 0.33),
  new Club("PW", "WEDGE", 105, 35.5, 33.0, 1.30, 0.28),
  new Club("SW", "SAND WEDGE", 75, 29.0, 42.0, 1.35, 0.22),
  new Club("PT", "PUTTER", 0, 0.0, 0.0, 0.0, 1.0),
]);
export const PUTTER = CLUBS[CLUBS.length - 1];
export const PUTT_SPEED = 4.6;
export const CURVE_MAX = 1.0;

export function clubByKey(key) {
  const c = CLUBS.find((cl) => cl.key === key);
  if (!c) throw new Error(`no club ${key}`);
  return c;
}

export function puttRangeM(power = 1.0, decel = SURFACES.green[3]) {
  const v = PUTT_SPEED * Math.sqrt(Math.max(0.0, power));
  return (v * v) / (2 * decel);
}

/** {club, power, heading (radians, 0 = +y, + turns right), lie, quality, curve, wind [x, y] m/s} */
export function makeShot(club, power, heading, lie = "fairway", { quality = 1.0, curve = 0.0, wind = [0.0, 0.0] } = {}) {
  return { club, power, heading, lie, quality, curve, wind };
}

const PATH_EVERY = 4; // a path sample every 4 substeps (30 per second)

/** One stroke from launch to rest. Call step(dt) with any dt; it substeps. */
export class Flight {
  constructor(hole, start, shot) {
    this.hole = hole;
    this.start = [start[0], start[1]];
    this.shot = shot;
    this.x = 0.0; this.y = 0.0; this.z = 0.0;
    this.vx = 0.0; this.vy = 0.0; this.vz = 0.0;
    this.spin = 0.0;
    this.curve = 0.0;
    this.t = 0.0;
    this.state = "air"; // air, roll, rest, holed, water, ob
    this.bounces = 0;
    this.landing = null; // first touch of the ground
    this.apex = 0.0;
    this.waterEntry = null;
    this.path = []; // [x, y, z] every PATH_EVERY substeps
    this.hitTrees = new Set();
    this._acc = 0.0;
    this._n = 0;
    this._surface = "fairway";

    const s = shot;
    [this.x, this.y] = this.start;
    const sh = Math.sin(s.heading), ch = Math.cos(s.heading);
    if (s.club.putter) {
      const v = PUTT_SPEED * Math.sqrt(Math.max(0.0, s.power));
      this.vx = v * sh; this.vy = v * ch; this.vz = 0.0;
      this.state = "roll";
      this._surface = hole.lieAt(this.x, this.y);
    } else {
      let scale = Math.max(0.0, s.power) * lieFactor(s.lie);
      const q = Math.min(1.0, Math.max(0.0, s.quality));
      scale *= THIN_DISTANCE + (1.0 - THIN_DISTANCE) * q;
      const v = s.club.speed * scale ** POWER_EXP;
      const launch = radians(s.club.launch * (THIN_LAUNCH + (1.0 - THIN_LAUNCH) * q));
      this.vx = v * Math.cos(launch) * sh;
      this.vy = v * Math.cos(launch) * ch;
      this.vz = v * Math.sin(launch);
      this.spin = s.club.spin;
      this.curve = s.curve * CURVE_MAX;
    }
    this.path.push([this.x, this.y, this.z]);
  }

  // --- public ---

  get done() { return this.state === "rest" || this.state === "holed" || this.state === "water" || this.state === "ob"; }

  get speed() { return Math.sqrt(this.vx * this.vx + this.vy * this.vy + this.vz * this.vz); }

  /** Distance from the start to the first touch of the ground (m). */
  get carry() {
    if (this.landing == null) return 0.0;
    return Math.hypot(this.landing[0] - this.start[0], this.landing[1] - this.start[1]);
  }

  step(dt) {
    this._acc += dt;
    while (this._acc >= SUBSTEP - 1e-12 && !this.done) {
      this._acc -= SUBSTEP;
      this._substep(SUBSTEP);
    }
    if (this.done) this._acc = 0.0;
  }

  /** Simulate to the end (bots, previews, tests). */
  run() {
    while (!this.done) this._substep(SUBSTEP);
    return this;
  }

  // --- the model ---

  _substep(h) {
    this.t += h;
    this._n += 1;
    if (this.state === "air") this._fly(h);
    else if (this.state === "roll") this._roll(h);
    if (this.t >= MAX_TIME && !this.done) this._settle();
    if (this._n % PATH_EVERY === 0 || this.done) this.path.push([this.x, this.y, this.z]);
  }

  _fly(h) {
    const [wx, wy] = this.shot.wind;
    const rx = this.vx - wx, ry = this.vy - wy, rz = this.vz;
    const sp = Math.sqrt(rx * rx + ry * ry + rz * rz);
    const hs = Math.hypot(rx, ry);
    let ax = -DRAG * sp * rx;
    let ay = -DRAG * sp * ry;
    let az = -GRAVITY - DRAG * sp * rz;
    if (hs > 1e-6 && sp > 1e-6) {
      const k = (LIFT * this.spin * sp) / hs;
      ax += k * (-rx * rz);
      ay += k * (-ry * rz);
      az += k * (hs * hs);
      const c = (SIDE * this.curve * this.spin * sp * sp) / hs;
      ax += c * ry;
      ay += c * -rx;
    }
    this.vx += ax * h;
    this.vy += ay * h;
    this.vz += az * h;
    const px = this.x, py = this.y;
    this.x += this.vx * h;
    this.y += this.vy * h;
    this.z += this.vz * h;
    this.spin *= Math.exp(-h / SPIN_TAU);
    this.apex = Math.max(this.apex, this.z);
    this._checkTrees();
    if (this.z <= 0.0) {
      // Back up to the exact touchdown point along this substep.
      const f = this.z - this.vz * h;
      const u = f - this.z > 1e-9 ? f / (f - this.z) : 1.0;
      this.x = px + (this.x - px) * u;
      this.y = py + (this.y - py) * u;
      this.z = 0.0;
      this._touch();
    }
  }

  _checkTrees() {
    const trees = this.hole.trees;
    for (let i = 0; i < trees.length; i++) {
      const tr = trees[i];
      if (this.hitTrees.has(i)) continue;
      if (this.z > tr.height || this.z < tr.height * 0.35) continue; // over it, or under the canopy
      if ((this.x - tr.x) ** 2 + (this.y - tr.y) ** 2 <= tr.radius * tr.radius) {
        // Knocked down: most of the speed gone, a little bounce back.
        this.hitTrees.add(i);
        this.vx *= -0.18;
        this.vy *= -0.18;
        this.vz = Math.min(this.vz, 0.0) * 0.3;
        this.spin = 0.0;
        this.curve = 0.0;
      }
    }
  }

  _touch() {
    const lie = this.hole.lieAt(this.x, this.y);
    if (this.landing == null) this.landing = [this.x, this.y];
    if (lie === "ob") {
      this.state = "ob";
      this.vx = this.vy = this.vz = 0.0;
      return;
    }
    if (lie === "water") {
      this.state = "water";
      this.waterEntry = [this.x, this.y];
      this.vx = this.vy = this.vz = 0.0;
      return;
    }
    if (this.hole.distanceToPin(this.x, this.y) <= CUP_RADIUS && this.speed < DUNK_SPEED) {
      this._holeOut();
      return;
    }
    let [e, first, keep] = surface(lie);
    if (this.bounces === 0) keep = first * this.shot.club.bite;
    this.bounces += 1;
    this.vx *= keep;
    this.vy *= keep;
    this.spin *= 0.4;
    this.curve *= 0.3;
    if (-this.vz * e > ROLL_BELOW) this.vz = -this.vz * e;
    else {
      this.vz = 0.0;
      this.state = "roll";
    }
    this._surface = lie;
  }

  _roll(h) {
    const lie = this.hole.lieAt(this.x, this.y);
    if (lie === "ob") {
      this.state = "ob";
      this.vx = this.vy = 0.0;
      return;
    }
    if (lie === "water") {
      this.state = "water";
      this.waterEntry = [this.x, this.y];
      this.vx = this.vy = 0.0;
      return;
    }
    this._surface = lie;
    const decel = surface(lie)[3];
    let sx = 0.0, sy = 0.0;
    if (lie === "green") {
      const [gx, gy] = this.hole.greenGradient(this.x, this.y);
      sx = -SLOPE_GAIN * gx;
      sy = -SLOPE_GAIN * gy;
    }
    const sp = Math.hypot(this.vx, this.vy);
    if (sp < STOP_SPEED && Math.hypot(sx, sy) < decel) {
      this._settle();
      return;
    }
    if (sp > 1e-9) {
      // Friction never reverses the ball: clamp it at a stop.
      const dv = Math.min(sp, decel * h);
      this.vx -= (this.vx / sp) * dv;
      this.vy -= (this.vy / sp) * dv;
    }
    this.vx += sx * h;
    this.vy += sy * h;
    const px = this.x, py = this.y;
    this.x += this.vx * h;
    this.y += this.vy * h;
    // Over the cup slowly enough to drop?
    if (lie === "green" || this.hole.distanceToPin(this.x, this.y) < 1.0) {
      if (segDist(px, py, this.x, this.y, this.hole.pin[0], this.hole.pin[1]) <= CUP_RADIUS
        && Math.hypot(this.vx, this.vy) <= CAPTURE_SPEED) this._holeOut();
    }
  }

  _holeOut() {
    [this.x, this.y] = this.hole.pin;
    this.z = 0.0;
    this.vx = this.vy = this.vz = 0.0;
    this.state = "holed";
  }

  _settle() {
    this.vx = this.vy = this.vz = 0.0;
    this.z = 0.0;
    this.state = "rest";
  }

  /** Where the ball is now (meaningful once it is at rest). */
  get lie() {
    if (this.state === "holed") return "green";
    return this.hole.lieAt(this.x, this.y);
  }
}

export function segDist(ax, ay, bx, by, px, py) {
  const dx = bx - ax, dy = by - ay;
  const seg = dx * dx + dy * dy;
  const u = seg <= 1e-12 ? 0.0 : Math.min(1.0, Math.max(0.0, ((px - ax) * dx + (py - ay) * dy) / seg));
  return Math.hypot(ax + dx * u - px, ay + dy * u - py);
}

export function simulate(hole, start, shot) {
  return new Flight(hole, start, shot).run();
}

/**
 * A straight, windless shot through the air only, for the landing target and
 * the dotted arc. Stops at the first touch of the ground.
 */
export function preview(hole, start, club, heading, lie, power = 1.0) {
  const f = new Flight(hole, start, makeShot(club, power, heading, lie));
  if (club.putter) return f.run();
  while (f.state === "air" && f.t < MAX_TIME) f._substep(SUBSTEP);
  return f;
}

/**
 * Where to play from after a water hazard: on dry land just short of where
 * the ball crossed into the water (along its ground track), never nearer the
 * hole than that, as in a lateral drop.
 */
export function dropPoint(hole, flight) {
  const entry = flight.waterEntry ?? [flight.x, flight.y];
  const track = [...flight.path.map((p) => [p[0], p[1]]), entry];
  const dry = (x, y) => { const l = hole.lieAt(x, y); return l !== "water" && l !== "ob"; };
  for (let i = track.length - 1; i > 0; i--) {
    const [x, y] = track[i - 1];
    if (dry(x, y)) {
      // Two more metres back from the edge, along the track.
      const [bx, by] = track[Math.max(0, i - 2)];
      const d = Math.hypot(x - bx, y - by);
      if (d > 1e-6) {
        const k = Math.min(1.0, 2.0 / d);
        const cx = x + (bx - x) * k, cy = y + (by - y) * k;
        if (dry(cx, cy)) return [cx, cy];
      }
      return [x, y];
    }
  }
  return [flight.start[0], flight.start[1]];
}
