// Raccoon Golf: three island holes, swung with your arms
// (port of src/kpapp/game/golf/game.py - keep the two in step).
//
// The rules and flow follow resort-island motion golf (Wii Sports Resort's
// Golf): a behind-the-ball view of the hole; auto-aim at the pin (or down the
// fairway on a dogleg) and an auto-chosen club, both adjustable; a landing
// target and dotted arc for a full-power straight shot; a power gauge that
// fills on the backswing with a red overswing zone that hooks or slices the
// ball; practice swings until you address the ball; a putter, slope grid and
// distance in feet on the green; water hazards (+1, drop near where it went in)
// and out of bounds (+1, replay from the same spot); and a pick-up at par + 5 so
// no hole drags on. The art, course and characters are original.
//
// The input is a whole-body swing seen by a camera (swing.js). Power comes from
// how high the hands went (a held position, not a speed), and impact is found
// on the grip's swept path, so lag delays a shot but never changes it.
//
// Scoring for the leaderboard (higher is better) is per hole, by strokes
// against par, while the HUD shows the golf score (E, +1, -2):
//
//     hole in one 1000, albatross or better 800, eagle 500, birdie 300,
//     par 200, bogey 100, double bogey 50, anything worse (or a pick-up) 10
//
// A perfect round is 3000. Points arrive only when a hole is finished.
//
// Randomness (each hole's wind, an overswing's hook or slice) comes from
// core/random.js, so a seed gives a different (but repeatable) round from the
// desktop's; everything after those draws is the same arithmetic.
//
// No DOM at module level: this file runs under Node for the tests. The renderer
// is built lazily on the first render().

import { Game } from "../base.js";
import { HandTracker } from "../../core/hand.js";
import { Rng } from "../../core/random.js";
import * as P from "./physics.js";
import { FEET_PER_M, MPH, YARDS_PER_M, cachedCourse, loadCourse } from "./course.js";
import { SwingDetector } from "./swing.js";
import { cachedArt, loadArt } from "./art.js";
import { GolfRenderer } from "./render.js";
import { GOLF_TUTORIAL } from "./tutorial.js";

export const Phase = Object.freeze({
  INTRO: "intro", AIM: "aim", FLIGHT: "flight", RESULT: "result", HOLED: "holed", GAME_OVER: "game_over",
});

// Arrow keys, as KeyboardEvent.key names them (the shell passes keys through).
export const KEY_LEFT = "ArrowLeft", KEY_UP = "ArrowUp", KEY_RIGHT = "ArrowRight", KEY_DOWN = "ArrowDown";

// --- pacing (seconds) ---
export const INTRO_SECONDS = 3.0;
export const RESULT_SECONDS = 1.6;
export const HAZARD_SECONDS = 2.4;
export const HOLED_MESSAGE = 2.2;
export const SCORECARD_SECONDS = 3.6;
export const AIR_SPEED = 1.25; // simulated seconds per real second while the ball flies
export const ROLL_SPEED = 1.7;
export const PUTT_SPEED = 1.3;
export const FOLLOW_SECONDS = 0.3;
export const CHASE_BLEND = 0.9;
export const HOLED_PULL = 0.9; // s from the last shot's view to the hole-out view
// The hole-out view (holedCamera): back and up from the pin, turned so the flag
// stands right of the result text with Lenny celebrating left of it.
export const HOLED_BACK = 10.0, HOLED_UP = 6.5, HOLED_PITCH = P.radians(21.0);
export const HOLED_TURN = P.radians(-14.0);
export const LENNY_BY_PIN = [-6.8, 0.5]; // m from the pin: right of the line, along it
export const MAX_OVER_PAR = 5;

// --- scoring ---
export const POINTS_ACE = 1000;
export const POINTS_BY_DIFF = { "-3": 800, "-2": 500, "-1": 300, 0: 200, 1: 100, 2: 50 };
export const POINTS_WORSE = 10;
// The fastest a hole can be finished after play starts: the intro, an address
// hold, a swing, and an ace's flight at AIR_SPEED (a 150 m carry flies ~6 s).
// For the leaderboard's plausibility rules (leaderboard/src/rules.js).
export const FASTEST_HOLE_SECONDS = INTRO_SECONDS + 0.4 + 0.5 + 6.0 / AIR_SPEED;

export const AIM_STEP = P.radians(2.0);
export const AIM_STEP_PUTT = P.radians(1.0);
export const AIM_MAX = P.radians(40.0);
export const HOVER_FIRE = 0.6;
export const HOVER_REPEAT = 0.22;

// Colours (RGB; the Python's BGR tuples reversed).
const GOLD = [255, 210, 70], CYAN_MSG = [110, 220, 255], ORANGE = [255, 190, 90], GREY = [190, 175, 175];

export function resultName(strokes, par) {
  if (strokes === 1) return "HOLE IN ONE!";
  const d = strokes - par;
  const names = { "-4": "CONDOR!", "-3": "ALBATROSS!", "-2": "EAGLE!", "-1": "BIRDIE!", 0: "PAR", 1: "BOGEY",
    2: "DOUBLE BOGEY", 3: "TRIPLE BOGEY" };
  return names[d] ?? (d > 0 ? `+${d}` : `${d}`);
}

export function holePoints(strokes, par, pickedUp = false) {
  if (pickedUp) return POINTS_WORSE;
  if (strokes === 1) return POINTS_ACE;
  const d = strokes - par;
  if (d <= -3) return POINTS_BY_DIFF["-3"];
  return POINTS_BY_DIFF[d] ?? POINTS_WORSE;
}

export function toParText(diff) {
  return diff === 0 ? "E" : diff > 0 ? `+${diff}` : String(diff);
}

/** Python's round(): halves go to the even neighbour. */
export function roundHalfEven(v) {
  const r = Math.round(v);
  return Math.abs(v % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

export class Camera {
  constructor(x = 0.0, y = 0.0, z = 4.0, yaw = 0.0, pitch = 0.18) {
    Object.assign(this, { x, y, z, yaw, pitch });
  }
  copy() { return new Camera(this.x, this.y, this.z, this.yaw, this.pitch); }
}

export function addressCamera(x, y, heading, putt) {
  const [back, height, pitch] = putt ? [5.0, 2.3, P.radians(16.0)] : [9.0, 4.2, P.radians(10.0)];
  return new Camera(x - Math.sin(heading) * back, y - Math.cos(heading) * back, height, heading, pitch);
}

/** The hole-out view along `heading` (the last shot's), and where Lenny celebrates. */
export function holedCamera(pin, heading) {
  const [px, py] = pin;
  const sh = Math.sin(heading), ch = Math.cos(heading);
  const cam = new Camera(px - sh * HOLED_BACK, py - ch * HOLED_BACK, HOLED_UP, heading + HOLED_TURN, HOLED_PITCH);
  const [right, along] = LENNY_BY_PIN;
  return [cam, [px + ch * right + sh * along, py - sh * right + ch * along]];
}

export function lerpCam(a, b, u) {
  return new Camera(a.x + (b.x - a.x) * u, a.y + (b.y - a.y) * u, a.z + (b.z - a.z) * u,
    a.yaw + (b.yaw - a.yaw) * u, a.pitch + (b.pitch - a.pitch) * u);
}

const message = (text, colour, size = 2.0, duration = 1.8, sub = "") => ({ text, colour, size, life: 0.0, duration, sub });

const cardRow = (par) => ({ par, strokes: 0, points: 0, pickedUp: false, done: false });

export class RaccoonGolfGame extends Game {
  static id = "golf";
  static title = "RACCOON GOLF";
  static blurb = "Swing your arms like a golf club. Three island holes - can you beat par?";
  static tip = "Stand back so your hands show at waist height";
  static tutorial = GOLF_TUTORIAL;
  static menu = false; // hidden for now: its page, games/raccoon-golf/, still runs it

  static async preload() {
    await Promise.all([loadArt(), loadCourse()]);
  }

  /**
   * opts: {width, height, mirrored, seed, art, course, holes, detector}. `art`
   * and `course` default to the preloaded ones; tests pass a stub art and the
   * course parsed from disk.
   */
  constructor({ width = 1280, height = 720, mirrored = true, seed = null, art = null, course = null,
    holes = null, detector = null } = {}) {
    super({ width, height, mirrored, seed });
    this.art = art ?? cachedArt();
    this.course = course ?? cachedCourse();
    if (!this.course) throw new Error("Raccoon Golf's course is not loaded: await RaccoonGolfGame.preload() first");
    this.nHoles = Math.min(this.course.holes.length, holes || this.course.holes.length);
    this.detector = detector ?? new SwingDetector("full", this.width, this.height);
    this.hand = "right";
    this.tracker = new HandTracker(this.hand, mirrored);
    this._renderer = null;
    this.reset();
  }

  // --- lifecycle ---

  reset() {
    this.rng = new Rng(this.seed); // unseeded by default: wind differs every round
    // Splashes get their own stream, so juice never changes what a seed plays.
    this.fxRng = new Rng(this.seed == null ? null : `${this.seed}:fx`);
    this.clock = 0.0;
    this.card = this.course.holes.slice(0, this.nHoles).map((h) => cardRow(h.par));
    this.fx = { shake: 0.0, flash: 0.0, scoreBump: 0.0, particles: [], popups: [] };
    this.message = null;
    this.flight = null;
    this.preview = null;
    this.lastShot = null;
    this.cursor = null;
    this.hover = null;
    this.hoverT = 0.0;
    this.hoverFired = 0;
    this.lennyAngle = 0.0;
    this.followT = -1.0;
    this.followFrom = 0.0;
    this.practiceT = -1.0;
    this.lastPose = null;
    this.shotsTaken = 0;
    this.celebrateSpot = null;
    this._startHole(0);
  }

  onResume() {
    this.detector.reset(); // the player moved during the pause
    this.tracker.reset();
  }

  swapHand() {
    this.hand = this.hand === "right" ? "left" : "right";
    this.tracker = new HandTracker(this.hand, this.mirrored);
  }

  // --- properties the HUD and the shell read ---

  get hole() { return this.course.holes[this.holeI]; }

  /** Leaderboard points (higher is better). */
  get score() { return this.card.reduce((s, r) => s + r.points, 0); }

  get strokesTotal() {
    return this.card.filter((r) => r.done).reduce((s, r) => s + r.strokes, 0) + (this.card[this.holeI].done ? 0 : this.strokes);
  }

  /** Strokes against par so far, counting only finished holes (as on a scorecard). */
  get toPar() { return this.card.filter((r) => r.done).reduce((s, r) => s + r.strokes - r.par, 0); }

  get club() { return P.CLUBS[this.clubI]; }
  get putting() { return this.club.putter; }
  get heading() { return this.autoHeading + this.aimOffset; }
  get distanceToPin() { return this.hole.distanceToPin(this.ball[0], this.ball[1]); }

  distanceText() {
    const d = this.distanceToPin;
    if (this.lie === "green") return `${(d * FEET_PER_M).toFixed(0)} ft`;
    return `${(d * YARDS_PER_M).toFixed(0)} yd`;
  }

  clubRangeText() {
    if (this.putting) return `${(P.puttRangeM() * FEET_PER_M).toFixed(0)} ft`;
    const reach = this.club.carryM * P.lieFactor(this.lie);
    return `${(reach * YARDS_PER_M).toFixed(0)} yd`;
  }

  get windMph() { return this._windMph; }
  /** Heading the wind blows toward (0 = down the hole, +y). */
  get windDir() { return this._windDir; }
  get wind() {
    const s = this._windMph * MPH;
    return [s * Math.sin(this._windDir), s * Math.cos(this._windDir)];
  }

  /** Set this hole's wind (tests and the parity fixture: Python's random draws can't be reproduced). */
  setWind(mph, dir) {
    this._windMph = mph;
    this._windDir = dir;
  }

  // --- holes and shots ---

  _startHole(i) {
    this.holeI = i;
    const h = this.hole;
    this.ball = [h.tee[0], h.tee[1]];
    this.prevSpot = this.ball;
    this.lie = "tee";
    this.strokes = 0;
    const [lo, hi] = h.windRange;
    this._windMph = roundHalfEven(this.rng.uniform(lo, hi));
    this._windDir = this.rng.uniform(0.0, 2 * Math.PI);
    this.phase = Phase.INTRO;
    this.phaseT = 0.0;
    this.flight = null;
    this._setupShot();
    // The intro flies down from high above the tee to the address view.
    const hd = this.autoHeading;
    this.introFrom = new Camera(h.tee[0] - Math.sin(hd) * 30, h.tee[1] - Math.cos(hd) * 30, 62.0, hd, P.radians(15.0));
    this.camera = this.introFrom.copy();
  }

  /** Wii-style: the shortest club that reaches the pin (putter on the green). */
  autoClub() {
    if (this.lie === "green") return P.CLUBS.length - 1;
    const d = this.distanceToPin;
    const factor = P.lieFactor(this.lie);
    const usable = [];
    P.CLUBS.slice(0, -1).forEach((c, i) => { if (!c.teeOnly || this.lie === "tee") usable.push(i); });
    for (const i of [...usable].reverse()) { // shortest first
      if (P.CLUBS[i].carryM * factor >= d * 0.97) return i;
    }
    return usable[0];
  }

  /** Ready the next shot from this.ball: club, aim, swing mode, camera. */
  _setupShot() {
    this.clubI = this.autoClub();
    this.aimOffset = 0.0;
    this._aim();
    this.detector.setMode(this.putting ? "putt" : "full");
    this.detector.reset();
    this.lennyAngle = 0.0;
    this.followT = -1.0;
    this.camera = addressCamera(this.ball[0], this.ball[1], this.heading, this.putting);
  }

  /** Auto-aim for the current club, and the preview of a full straight shot. */
  _aim() {
    const reach = this.putting ? Infinity : this.club.carryM * P.lieFactor(this.lie);
    const [tx, ty] = this.putting ? this.hole.pin : this.hole.aimPoint(this.ball[0], this.ball[1], reach);
    this.autoHeading = Math.atan2(tx - this.ball[0], ty - this.ball[1]);
    this.preview = P.preview(this.hole, this.ball, this.club, this.heading, this.lie);
  }

  changeClub(step) {
    if (this.phase !== Phase.AIM || this.detector.swinging) return;
    const n = P.CLUBS.length;
    let i = this.clubI;
    for (let k = 0; k < n; k++) {
      i = (((i + step) % n) + n) % n;
      const c = P.CLUBS[i];
      if (c.teeOnly && this.lie !== "tee") continue;
      if (this.lie === "bunker" && c.putter) continue;
      break;
    }
    const wasPutt = this.putting;
    this.clubI = i;
    this._aim();
    if (this.putting !== wasPutt) {
      this.detector.setMode(this.putting ? "putt" : "full");
      this.camera = addressCamera(this.ball[0], this.ball[1], this.heading, this.putting);
    }
  }

  adjustAim(step) {
    if (this.phase !== Phase.AIM || this.detector.swinging) return;
    const size = this.putting ? AIM_STEP_PUTT : AIM_STEP;
    this.aimOffset = Math.max(-AIM_MAX, Math.min(AIM_MAX, this.aimOffset + step * size));
    this.preview = P.preview(this.hole, this.ball, this.club, this.heading, this.lie);
    this.camera = addressCamera(this.ball[0], this.ball[1], this.heading, this.putting);
  }

  /** Arrow keys: aim and club. Returns true if the key was used. */
  handleKey(key) {
    if (key === KEY_LEFT) this.adjustAim(-1);
    else if (key === KEY_RIGHT) this.adjustAim(1);
    else if (key === KEY_UP) this.changeClub(-1); // up the list: a longer club
    else if (key === KEY_DOWN) this.changeClub(1);
    else return false;
    return true;
  }

  /** Strike the ball with a swing event. `curve` overrides the overswing's random hook/slice (tests). */
  _hit(ev, curve = null) {
    if (curve == null) {
      curve = 0.0;
      if (!this.putting && ev.overswing > 0) {
        // Overswing: an unpredictable hook or slice, bigger the deeper into the red.
        curve = this.rng.choice([-1.0, 1.0]) * ev.overswing * this.rng.uniform(0.55, 1.0);
      }
    }
    const shot = P.makeShot(this.club, ev.power, this.heading, this.lie, { quality: ev.quality, curve, wind: this.wind });
    this.prevSpot = this.ball;
    this.startLie = this.lie;
    this.strokes += 1;
    this.shotsTaken += 1;
    this.flight = new P.Flight(this.hole, this.ball, shot);
    this.lastShot = { power: ev.power, quality: ev.quality, curve, club: this.club.key, overswing: ev.overswing };
    this.phase = Phase.FLIGHT;
    this.phaseT = 0.0;
    this.shotFromCam = this.camera.copy();
    this.message = null;
    if (ev.overswing > 0 && !this.putting) this._say("OVERSWING!", [255, 80, 60], 1.3, 1.1);
    else if (ev.quality < 0.5 && !this.putting) this._say("THIN!", ORANGE, 1.3, 1.1);
  }

  _say(text, colour, size = 2.0, duration = 1.8, sub = "") {
    this.message = message(text, colour, size, duration, sub);
  }

  // --- the frame ---

  update(controls, dt) {
    dt = Math.min(Math.max(dt, 0.0), 0.1); // a stalled frame must not skip the game forward
    this.clock += dt;
    this.phaseT += dt;
    this.lastPose = controls?.pose ?? null;
    this._effects(dt);
    if (this.phase === Phase.INTRO) {
      const u = Math.min(1.0, this.phaseT / INTRO_SECONDS);
      const e = u * u * (3 - 2 * u);
      this.camera = lerpCam(this.introFrom, addressCamera(this.ball[0], this.ball[1], this.heading, this.putting), e);
      if (u >= 1.0) {
        this.phase = Phase.AIM;
        this.phaseT = 0.0;
      }
      return;
    }
    if (this.phase === Phase.AIM) this._updateAim(controls, dt);
    else if (this.phase === Phase.FLIGHT) this._updateFlight(dt);
    else if (this.phase === Phase.RESULT) {
      if (this.phaseT >= this._resultFor) this._afterResult();
    } else if (this.phase === Phase.HOLED) {
      const u = Math.min(1.0, this.phaseT / HOLED_PULL); // pull back to look at the green
      this.camera = lerpCam(this.holedFrom, this.holedTo, u * u * (3 - 2 * u));
      if (this.phaseT >= HOLED_MESSAGE + SCORECARD_SECONDS) {
        if (this.holeI + 1 < this.nHoles) this._startHole(this.holeI + 1);
        else {
          this.phase = Phase.GAME_OVER;
          this.phaseT = 0.0;
        }
      }
    }
    this._animateLenny(dt);
  }

  _updateAim(controls, dt) {
    // Stepped poses: the detector times motion between model updates.
    const step = controls?.stepPose ?? controls?.pose ?? null;
    const ev = this.detector.update(step, this.clock);
    this._updateButtons(controls, dt);
    if (ev == null) return;
    this.followT = 0.0;
    this.followFrom = this.lennyAngle;
    if (ev.practice) {
      this.practiceT = 0.0;
      return;
    }
    this._hit(ev);
  }

  // --- the in-game hover buttons (aim and club) ---

  /** The adjust buttons' rects: [name, [x0, y0, x1, y1]]. */
  buttons() {
    if (this.phase !== Phase.AIM) return [];
    return [["club_up", [16, 222, 158, 288]], ["club_down", [174, 222, 316, 288]],
      ["aim_left", [200, 400, 330, 480]], ["aim_right", [950, 400, 1080, 480]]];
  }

  /** Only while the hands are apart: a grip (hands together) is swinging, and a backswing must never press one. */
  get buttonsActive() {
    const d = this.detector;
    return this.phase === Phase.AIM && this.cursor != null && !d.swinging && !d.together && d.stage === "no_grip";
  }

  _updateButtons(controls, dt) {
    this.cursor = this.tracker.update(controls?.pose ?? null, this.clock);
    let hovered = null;
    if (this.buttonsActive && this.cursor != null) {
      const [cx, cy] = this.cursor;
      for (const [name, [x0, y0, x1, y1]] of this.buttons()) {
        if (x0 - 8 <= cx && cx <= x1 + 8 && y0 - 8 <= cy && cy <= y1 + 8) {
          hovered = name;
          break;
        }
      }
    }
    if (hovered !== this.hover) {
      this.hover = hovered;
      this.hoverT = 0.0;
      this.hoverFired = 0;
      return;
    }
    if (hovered == null) return;
    this.hoverT += dt;
    const due = HOVER_FIRE + this.hoverFired * HOVER_REPEAT;
    if (this.hoverT >= due) {
      this.hoverFired += 1;
      ({ club_up: () => this.changeClub(-1), club_down: () => this.changeClub(1),
        aim_left: () => this.adjustAim(-1), aim_right: () => this.adjustAim(1) })[hovered]();
    }
  }

  // --- flight ---

  _updateFlight(dt) {
    const f = this.flight;
    const speed = f.shot.club.putter ? PUTT_SPEED : f.state === "air" ? AIR_SPEED : ROLL_SPEED;
    const wasAir = f.state === "air";
    const bounces = f.bounces;
    f.step(dt * speed);
    if (f.bounces > bounces && f.landing != null && bounces === 0) {
      if (this.hole.lieAt(f.landing[0], f.landing[1]) === "bunker") this._burst(f.landing, "sand", 22);
    }
    if ((wasAir && f.state === "water") || (f.state === "water" && !wasAir)) {
      this._burst(f.waterEntry ?? [f.x, f.y], "water", 30);
      this.fx.shake = 1.0;
    }
    this._followCamera(dt);
    if (f.done) this._resolve();
  }

  _followCamera(_dt) {
    const f = this.flight;
    const hd = f.shot.heading;
    const putt = f.shot.club.putter;
    const [back, up, pitch] = putt ? [5.0, 2.3, P.radians(16.0)] : [13.0, 2.6, P.radians(9.0)];
    const tz = up + (putt ? 0.0 : f.z); // ride above the ball: it stays just under the centre
    const target = new Camera(f.x - Math.sin(hd) * back, f.y - Math.cos(hd) * back, tz, hd, pitch);
    const u = Math.min(1.0, this.phaseT / CHASE_BLEND);
    this.camera = lerpCam(this.shotFromCam, target, u * u * (3 - 2 * u));
  }

  _resolve() {
    const f = this.flight, h = this.hole;
    this._resultFor = RESULT_SECONDS;
    if (f.state === "holed") {
      this.ball = [h.pin[0], h.pin[1]];
      this._finishHole();
      return;
    }
    if (f.state === "water") {
      this.strokes += 1;
      this.ball = P.dropPoint(h, f);
      this.lie = h.lieAt(this.ball[0], this.ball[1]);
      this._say("WATER HAZARD", [90, 190, 255], 1.7, HAZARD_SECONDS, "+1 stroke - drop near where it went in");
      this._resultFor = HAZARD_SECONDS;
    } else if (f.state === "ob") {
      this.strokes += 1;
      this.ball = this.prevSpot;
      this.lie = this.startLie;
      this._say("OUT OF BOUNDS", [255, 70, 70], 1.7, HAZARD_SECONDS, "+1 stroke - play again from the same spot");
      this.fx.shake = Math.max(this.fx.shake, 0.8);
      this._resultFor = HAZARD_SECONDS;
    } else {
      this.ball = [f.x, f.y];
      this.lie = h.lieAt(f.x, f.y);
      this._comment();
    }
    this.phase = Phase.RESULT;
    this.phaseT = 0.0;
  }

  /** Juice for where a (dry, unholed) shot came to rest. */
  _comment() {
    const f = this.flight, shot = f.shot;
    const cyan = [110, 220, 255];
    if (shot.club.putter) {
      if (this.distanceToPin < 0.6) this._say("SO CLOSE!", cyan, 1.5, RESULT_SECONDS);
      return;
    }
    if (this.lie === "green" && this.startLie !== "green") this._say("NICE ON!", GOLD, 2.2, RESULT_SECONDS);
    else if (f.hitTrees.size) this._say("TIMBER!", ORANGE, 1.6, RESULT_SECONDS);
    else if (shot.power >= 0.85 && shot.power <= 1.0 && shot.quality >= 0.8 && (this.lie === "fairway" || this.lie === "green")) {
      this._say("NICE SHOT!", GOLD, 2.2, RESULT_SECONDS);
    } else if (this.lie === "bunker") this._say("BUNKER", ORANGE, 1.4, RESULT_SECONDS);
    else if (this.lie === "rough") this._say("ROUGH", [190, 175, 175], 1.2, RESULT_SECONDS);
  }

  _afterResult() {
    if (this.strokes >= this.hole.par + MAX_OVER_PAR) {
      this._finishHole(true);
      return;
    }
    this.flight = null;
    this.message = null;
    this.phase = Phase.AIM;
    this.phaseT = 0.0;
    this._setupShot();
  }

  _finishHole(pickedUp = false) {
    const row = this.card[this.holeI], par = this.hole.par;
    row.strokes = Math.min(this.strokes, par + MAX_OVER_PAR);
    row.pickedUp = pickedUp;
    row.points = holePoints(row.strokes, par, pickedUp);
    row.done = true;
    this.fx.scoreBump = 1.0;
    const name = pickedUp ? "PICKED UP" : resultName(row.strokes, par);
    const good = !pickedUp && row.strokes <= par;
    const colour = good ? GOLD : row.strokes === par + 1 ? CYAN_MSG : GREY;
    const big = row.strokes < par || row.strokes === 1 ? 2.6 : 2.0;
    this._say(name, colour, big, HOLED_MESSAGE + 0.4);
    this.fx.popups.push({ text: `+${row.points}`, colour: GOLD, pos: [this.width / 2, this.height * 0.56], life: 0.0, duration: 1.2 });
    if (good) this.fx.flash = row.strokes < par ? 0.6 : 0.0;
    this.phase = Phase.HOLED;
    this.phaseT = 0.0;
    const hd = this.flight != null ? this.flight.shot.heading : this.autoHeading;
    this.holedFrom = this.camera.copy();
    [this.holedTo, this.celebrateSpot] = holedCamera(this.hole.pin, hd);
  }

  // --- effects and Lenny ---

  _burst(at, kind, n) {
    const r = this.fxRng; // cosmetic only
    for (let i = 0; i < n; i++) {
      const a = r.uniform(0, 2 * Math.PI);
      const s = kind === "water" ? r.uniform(1.0, 4.0) : r.uniform(0.8, 2.5);
      this.fx.particles.push({ x: at[0], y: at[1], z: 0.05, vx: Math.cos(a) * s * 0.5, vy: Math.sin(a) * s * 0.5,
        vz: kind === "water" ? r.uniform(3.0, 7.0) : r.uniform(2.0, 4.5), kind, life: 0.0, duration: r.uniform(0.6, 1.1) });
    }
  }

  _effects(dt) {
    const fx = this.fx;
    fx.shake = Math.max(0.0, fx.shake - dt * 2.0);
    fx.flash = Math.max(0.0, fx.flash - dt * 2.0);
    fx.scoreBump = Math.max(0.0, fx.scoreBump - dt * 2.5);
    for (const p of fx.particles) {
      p.life += dt;
      p.vz -= 9.81 * dt;
      p.x += p.vx * dt;
      p.y += p.vy * dt;
      p.z = Math.max(0.0, p.z + p.vz * dt);
    }
    fx.particles = fx.particles.filter((p) => p.life < p.duration);
    for (const p of fx.popups) p.life += dt;
    fx.popups = fx.popups.filter((p) => p.life < p.duration);
    if (this.message != null) {
      this.message.life += dt;
      if (this.message.life >= this.message.duration) this.message = null;
    }
  }

  /**
   * Lenny's swing angle: 0 address, 1 top of a full backswing, < 0 follow-through.
   * Before impact it follows the player's own grip; after it, a follow-through.
   */
  _animateLenny(dt) {
    const putt = this.putting;
    const finish = putt ? -0.45 : -0.95;
    if (this.followT >= 0.0) {
      this.followT += dt;
      const t = this.followT;
      if (t < FOLLOW_SECONDS) {
        const u = t / FOLLOW_SECONDS;
        this.lennyAngle = this.followFrom + (finish - this.followFrom) * (1 - (1 - u) ** 3);
      } else if (this.phase === Phase.AIM && t > FOLLOW_SECONDS + 0.7) {
        this.lennyAngle += (0.0 - this.lennyAngle) * Math.min(1.0, dt * 6.0); // a practice swing: back to address
        if (Math.abs(this.lennyAngle) < 0.01) this.followT = -1.0;
      } else this.lennyAngle = finish;
      return;
    }
    if (this.phase === Phase.AIM) {
      let target = this.detector.swinging || this.detector.ready ? this.detector.swingAngle : 0.0;
      if (putt) target *= 0.5;
      this.lennyAngle += (target - this.lennyAngle) * Math.min(1.0, dt * 18.0);
    }
  }

  // --- drawing ---

  render(ctx, frame) {
    if (!this._renderer) {
      const art = this.art ?? cachedArt();
      if (!art) return;
      this._renderer = new GolfRenderer(this.width, this.height, art);
    }
    this._renderer.draw(ctx, frame, this);
  }
}

export default RaccoonGolfGame;
