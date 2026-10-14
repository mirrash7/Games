// Judging a dance: timing windows, ratings, score and stars
// (port of src/kpapp/game/dance/judge.py; docs/DANCE.md section 7).
//
// Every judged keyframe in a song is a judge point at time `t`. While the song
// time is inside [t - early, t + late], the judge keeps the best similarity
// between the point's pose and the player that was *held* for `hold` seconds.
// When the window closes, the point resolves into a rating. The window, the
// hold and the rating thresholds depend on the difficulty: one table,
// DIFFICULTY, below.
//
// Why "best in a window" rather than "the pose at t": poses reach the game in
// steps (15-30 Hz) and 100-150 ms late, so the frame that shows the player
// hitting a pose lands at some unpredictable moment after the beat. The window
// is skewed late for the same reason.
//
// A point the judge never got a usable look at (no shoulders, no arms) is
// UNSEEN: no rating on screen, no points, and never a miss. Tracking loss is
// the camera's fault, not the player's.
//
// The numbers and the reasons for them are in judge.py and docs/DANCE.md;
// they are pinned by tests/test_dance_game.py and web/tests/dance_game.test.mjs.

import { similarity } from "./pose.js";

/** How strictly one difficulty judges: the window, the hold and the rating thresholds. */
export class Rules {
  constructor({ early, late, hold, ratings }) {
    this.early = early; // s before the point's time that a pose already counts
    this.late = late; // s after it that the window closes
    this.hold = hold; // s a match must be held before it counts
    this.ratings = ratings; // [[name, minimum best similarity, value], ...]; below the last is X
    Object.freeze(this);
  }

  /** See LOST_BEFORE. */
  get lostBefore() {
    return this.late / 2;
  }
}

const table = (rows) => Object.freeze(rows.map((r) => Object.freeze(r)));

// One table for every difficulty (the reasons for each number are in judge.py).
// NORMAL is the game as tuned before difficulties existed. EASY judges a
// quarter as many poses (chart.deriveEasy) and forgives more; HARD judges the
// normal chart strictly.
export const DIFFICULTY = Object.freeze({
  easy: new Rules({ early: 0.25, late: 0.60, hold: 0.12, ratings: table([
    ["perfect", 0.86, 1.0], ["super", 0.72, 0.8], ["good", 0.56, 0.55], ["ok", 0.40, 0.3]]) }),
  normal: new Rules({ early: 0.20, late: 0.50, hold: 0.12, ratings: table([
    ["perfect", 0.90, 1.0], ["super", 0.74, 0.8], ["good", 0.60, 0.55], ["ok", 0.42, 0.3]]) }),
  hard: new Rules({ early: 0.15, late: 0.50, hold: 0.15, ratings: table([
    ["perfect", 0.93, 1.0], ["super", 0.84, 0.8], ["good", 0.68, 0.55], ["ok", 0.50, 0.3]]) }),
});
export const DEFAULT_DIFFICULTY = "normal";

/** The Rules for a difficulty id (unknown ids and null mean normal), or a Rules as is. */
export function rulesFor(difficulty) {
  if (difficulty instanceof Rules) return difficulty;
  return Object.hasOwn(DIFFICULTY, String(difficulty)) ? DIFFICULTY[String(difficulty)]
    : DIFFICULTY[DEFAULT_DIFFICULTY];
}

// NORMAL's numbers under their old names, for code and tests that read them.
const N = DIFFICULTY.normal;
export const WINDOW_EARLY = N.early;
export const WINDOW_LATE = N.late;
export const HOLD = N.hold;
export const RATINGS = N.ratings;
export const MISS = Object.freeze(["x", 0.0]);
export const UNSEEN = "unseen";
export const RATING_NAMES = Object.freeze([...RATINGS.map((r) => r[0]), MISS[0]]);
export const COMBO_FROM = Object.freeze(["perfect", "super"]); // ratings that extend the combo

export const SCORE_MAX = 10000;
export const STAR_SCORES = Object.freeze([2000, 4000, 6000, 8000, 9200]); // 1..5 stars
export const SUPERSTAR = 9600;

// After a pause the player's hand is on the shell's RESUME button, not in the
// move. Points whose window is open at the resume, or opens within this many
// seconds after it, are "excused": what was seen before the pause is
// forgotten, and if what is seen after it would be an X, the point is unseen.
export const RESUME_GRACE = 1.0;

// Tracking loss is never a miss (AGENTS.md section 6). A window that loses the
// player before this many seconds past its point's time, and never sees them
// again, saw at most the player on their way into the pose (they reach the game
// 100-150 ms late): if that would be an X, the point is unseen instead.
export const LOST_BEFORE = N.lostBefore;

/** Python's round(): half to even, so scores match the desktop's exactly. */
export function pyRound(x) {
  return Math.abs(x % 1) === 0.5 ? 2 * Math.round(x / 2) : Math.round(x);
}

/** A best similarity -> [rating name, value], by `ratings` (default: normal's). */
export function rate(sim, ratings = null) {
  for (const [name, need, value] of ratings ?? RATINGS) if (sim >= need) return [name, value];
  return [...MISS];
}

export function starsFor(score) {
  return STAR_SCORES.filter((s) => score >= s).length;
}

/**
 * 0..5, continuous: the star bar's fill. Each star's segment fills linearly
 * from the previous threshold to its own.
 */
export function starFill(score) {
  let lo = 0.0, out = 0.0;
  for (const need of STAR_SCORES) {
    if (score >= need) {
      out += 1.0;
      lo = need;
      continue;
    }
    return out + Math.max(0.0, score - lo) / (need - lo);
  }
  return out;
}

export class Rating {
  constructor(name, value, weight, similarity, point, time) {
    this.name = name; // "perfect" | "super" | "good" | "ok" | "x" | "unseen"
    this.value = value; // 1.0 / 0.8 / 0.55 / 0.3 / 0, and 0 for unseen
    this.weight = weight; // 2 for a gold point, else 1
    this.similarity = similarity; // the best seen; null if unseen
    this.point = point; // the chart's judge point
    this.time = time; // song time it resolved at
    Object.freeze(this);
  }

  get gold() {
    return Boolean(this.point?.gold);
  }

  get seen() {
    return this.name !== UNSEEN;
  }

  get combo() {
    return COMBO_FROM.includes(this.name);
  }
}

class Window {
  constructor(point, hold) {
    this.point = point;
    this.hold = hold;
    this.best = null; // best similarity held for HOLD seconds
    this.floor = null; // worst sample seen, for a window seen too briefly to hold
    this.recent = []; // [song time, similarity], the last HOLD s
    this.excused = false;
    this.lastSeen = null; // song time of the last measurable sample
  }

  add(t, sim) {
    this.floor = this.floor == null ? sim : Math.min(this.floor, sim);
    this.lastSeen = t;
    const r = this.recent;
    r.push([t, sim]);
    // Keep the newest sample at or before t - HOLD, so the kept run spans HOLD.
    while (r.length > 1 && r[1][0] <= t - this.hold) r.shift();
    if (t - r[0][0] >= this.hold - 1e-9) {
      let held = Infinity;
      for (const [, s] of r) held = Math.min(held, s);
      if (this.best == null || held > this.best) this.best = held;
    }
  }

  forget() {
    this.best = this.floor = this.lastSeen = null;
    this.recent.length = 0;
  }

  result() {
    return this.best != null ? this.best : this.floor;
  }
}

/**
 * Feed it update(songTime, measured) every frame; it returns the ratings that
 * resolved on that frame (for juice). `rules` is a difficulty id or a Rules
 * (default normal).
 */
export class Judge {
  constructor(song, rules = null) {
    this.song = song;
    this.rules = rulesFor(rules);
    this.points = [...song.judgePoints()];
    this.totalWeight = this.points.reduce((a, p) => a + p.weight, 0);
    this.reset();
  }

  reset() {
    this._next = 0; // index of the first point not yet opened
    this._open = [];
    this.results = [];
    this.counts = Object.fromEntries(RATING_NAMES.map((n) => [n, 0]));
    this.unseen = 0;
    this.earned = 0.0; // sum of value * weight
    this.combo = 0;
    this.bestCombo = 0;
    this.lastRating = null; // the last *seen* rating
    this.songTime = -1e9;
    this._excuseUntil = -1e9;
  }

  // --- state ---

  get score() {
    if (this.totalWeight <= 0) return 0;
    return pyRound(SCORE_MAX * this.earned / this.totalWeight);
  }

  get stars() {
    return starsFor(this.score);
  }

  get starFill() {
    return starFill(this.score);
  }

  get superstar() {
    return this.score >= SUPERSTAR;
  }

  get done() {
    return this.results.length === this.points.length;
  }

  /** Points whose window contains the current song time. */
  get openPoints() {
    return this._open.map((w) => w.point);
  }

  /** Is any window open at `songTime` (default: the last update)? */
  judging(songTime = null) {
    const t = songTime == null ? this.songTime : songTime;
    return this.points.slice(0, this._next + 4)
      .some((p) => p.time - this.rules.early <= t && t <= p.time + this.rules.late);
  }

  // --- feeding ---

  update(songTime, measured) {
    this.songTime = songTime;
    const { early, late } = this.rules;
    // Open every window that has started.
    while (this._next < this.points.length && songTime >= this.points[this._next].time - early) {
      const w = new Window(this.points[this._next], this.rules.hold);
      w.excused = w.point.time - early <= this._excuseUntil;
      this._open.push(w);
      this._next += 1;
    }
    const resolved = [];
    const still = [];
    for (const w of this._open) {
      if (songTime > w.point.time + late) {
        // Closed. A frame that jumped past the whole window (a stall) saw
        // nothing of it: unseen, never a miss.
        resolved.push(this._resolve(w, songTime));
        continue;
      }
      if (measured != null) {
        const [sim] = similarity(w.point.pose, measured);
        if (sim != null) w.add(songTime, sim); // an unmeasurable frame neither breaks nor extends a hold
      }
      still.push(w);
    }
    this._open = still;
    return resolved;
  }

  /**
   * Forget what open windows saw before a pause, and excuse the points near
   * the resume from X ratings (see RESUME_GRACE).
   */
  onResume(songTime) {
    this._excuseUntil = songTime + RESUME_GRACE;
    for (const w of this._open) {
      w.forget();
      w.excused = true;
    }
  }

  _resolve(w, songTime) {
    const p = w.point;
    const best = w.result();
    let name, value;
    if (best == null) [name, value] = [UNSEEN, 0.0];
    else {
      [name, value] = rate(best, this.rules.ratings);
      const lost = w.lastSeen != null && w.lastSeen < p.time + this.rules.lostBefore;
      if (name === MISS[0] && (w.excused || lost)) [name, value] = [UNSEEN, 0.0];
    }
    const r = new Rating(name, value, p.weight, name !== UNSEEN ? best : null, p, songTime);
    this.results.push(r);
    if (name === UNSEEN) {
      this.unseen += 1; // the combo carries on: tracking loss is never punished
    } else {
      this.counts[name] += 1;
      this.earned += value * p.weight;
      this.lastRating = name;
      this.combo = r.combo ? this.combo + 1 : 0;
      this.bestCombo = Math.max(this.bestCombo, this.combo);
    }
    return r;
  }
}
