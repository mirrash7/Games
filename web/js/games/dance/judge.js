// Judging a dance: timing windows, ratings, score and stars
// (port of src/kpapp/game/dance/judge.py; docs/DANCE.md section 7).
//
// Every judged keyframe in a song is a judge point at time `t`. While the song
// time is inside [t - WINDOW_EARLY, t + WINDOW_LATE], the judge keeps the best
// similarity between the point's pose and the player that was *held* for HOLD
// seconds. When the window closes, the point resolves into a rating.
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

export const WINDOW_EARLY = 0.20; // s before the point's time that a pose already counts
export const WINDOW_LATE = 0.50; // s after it (the spec started at 0.40; see judge.py)
// A match only counts once it has been *held* this long (song time): the
// window's score is the best, over the window, of the worst similarity in the
// last HOLD seconds. A limb sweeping through the target is not a PERFECT.
export const HOLD = 0.12;

// [name, minimum best similarity, value]. Checked in order; below OK is X.
// PERFECT is 0.90 (the spec started at 0.86; see judge.py).
export const RATINGS = Object.freeze([
  ["perfect", 0.90, 1.0],
  ["super", 0.74, 0.8],
  ["good", 0.60, 0.55],
  ["ok", 0.42, 0.3],
]);
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
export const LOST_BEFORE = WINDOW_LATE / 2;

/** Python's round(): half to even, so scores match the desktop's exactly. */
export function pyRound(x) {
  return Math.abs(x % 1) === 0.5 ? 2 * Math.round(x / 2) : Math.round(x);
}

/** A best similarity -> [rating name, value]. */
export function rate(sim) {
  for (const [name, need, value] of RATINGS) if (sim >= need) return [name, value];
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
 * resolved on that frame (for juice). `hold` is overridable for the test that
 * pins why HOLD exists.
 */
export class Judge {
  constructor(song, { hold = HOLD } = {}) {
    this.song = song;
    this.hold = hold;
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
      .some((p) => p.time - WINDOW_EARLY <= t && t <= p.time + WINDOW_LATE);
  }

  // --- feeding ---

  update(songTime, measured) {
    this.songTime = songTime;
    // Open every window that has started.
    while (this._next < this.points.length && songTime >= this.points[this._next].time - WINDOW_EARLY) {
      const w = new Window(this.points[this._next], this.hold);
      w.excused = w.point.time - WINDOW_EARLY <= this._excuseUntil;
      this._open.push(w);
      this._next += 1;
    }
    const resolved = [];
    const still = [];
    for (const w of this._open) {
      if (songTime > w.point.time + WINDOW_LATE) {
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
      [name, value] = rate(best);
      const lost = w.lastSeen != null && w.lastSeen < p.time + LOST_BEFORE;
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
