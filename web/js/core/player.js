// PlayerLock: follow one main player through a crowd (port of player.py).
//
// The model sees everyone in frame: at a booth, passers-by, people queueing,
// friends watching. Handing the games the highest-scoring detection lets any
// of them take over. PlayerLock picks the main player once and follows that
// identity result by result, so every game sees only them. It runs on each
// new model result before the PoseExtrapolator, so extrapolation and
// stepPose follow one identity too. See src/kpapp/player.py for the reasons
// behind every rule; web/tests/player.test.mjs replays the Python's answers
// (web/tests/fixtures/player_parity.json, tools/make_player_parity_fixture.py).

import { KP } from "./pose.js";

export const MIN_CONF = 0.5;
const BODY = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]; // nose and the 12 limb joints

// Who can be the player.
export const MIN_SCALE = 0.015;
export const WALK_SPEED = 2.2;
export const SPEED_WINDOW = 1.0;
export const SPEED_MIN_SPAN = 0.5;
export const WALK_SPAN = 0.8;
export const CENTRAL = 0.35;
// Acquisition and hand-over.
export const ACQUIRE_TIME = 0.3;
export const KEEP_CLAIM = 0.85;
export const TAKEOVER_TIME = 1.0;
export const TAKEOVER_CENTRAL = 0.2;
export const TAKEOVER_STILL = 1.5;
export const TAKEOVER_BIGGER = 1.6;
export const TAKEOVER_ASIDE = 0.3;
// Keeping the lock.
export const HOLD_TIME = 1.2;
export const EDGE_HOLD = 0.3;
export const EDGE = 0.38;
export const RECALL_TIME = 4.0;
export const FORGET_TIME = 1.5;
export const MAX_TRACKS = 16;
export const DUPLICATE_IOU = 0.8;
// Matching.
export const GATE_BASE = 0.35;
export const GATE_RATE = 6.0;
export const GATE_MAX = 3.0;
export const SIZE_MIN = 0.8;
export const SIZE_MAX = 1.25;
export const PRED_MAX = 0.15;
export const PRED_CAP = 0.5;
export const W_SIZE = 3.0;
export const W_IOU = 0.5;
export const HOME_TAU = 1.5;
export const UNSURE_SIZE = 0.3;
export const PLAYER_PRIOR = 0.15;
export const W_GAP = 0.75;

// Plain sqrt and left-to-right sums, exactly as the Python, so both builds
// compute the same doubles.
const dist = (dx, dy) => Math.sqrt(dx * dx + dy * dy);

function mean(pts) {
  let sx = 0, sy = 0;
  for (const [x, y] of pts) { sx += x; sy += y; }
  return [sx / pts.length, sy / pts.length];
}

const vis = (pose, i) => pose.confidence[i] >= MIN_CONF;
const pt = (pose, i) => [pose.xy[2 * i], pose.xy[2 * i + 1]];

/** The detection box, or the bounding box of the visible keypoints. */
export function poseBox(pose) {
  if (pose.box != null) return [pose.box[0], pose.box[1], pose.box[2], pose.box[3]];
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < pose.confidence.length; i++) {
    if (!vis(pose, i)) continue;
    const [x, y] = pt(pose, i);
    x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y);
  }
  return x0 === Infinity ? [0, 0, 0, 0] : [x0, y0, x1, y1];
}

/** What the lock reads from one detection: {ax, ay, scale, sure, shoulders, box, quality}. */
export function bodyOf(pose) {
  const box = poseBox(pose);
  const ls = KP.left_shoulder, rs = KP.right_shoulder, lh = KP.left_hip, rh = KP.right_hip;
  const shoulders = vis(pose, ls) && vis(pose, rs);
  let ax, ay, scale, sure;
  if (shoulders) {
    const [x1, y1] = pt(pose, ls), [x2, y2] = pt(pose, rs);
    ax = (x1 + x2) * 0.5;
    ay = (y1 + y2) * 0.5;
    const sw = dist(x1 - x2, y1 - y2);
    const hips = [lh, rh].filter((i) => vis(pose, i)).map((i) => pt(pose, i));
    let torso = 0;
    if (hips.length) {
      const [hx, hy] = mean(hips);
      torso = dist(hx - ax, hy - ay);
    }
    scale = Math.max(sw, 0.7 * torso);
    sure = true;
  } else {
    // A guess at where the shoulder midpoint is, from what is visible.
    scale = 0.2 * (box[3] - box[1]);
    sure = false;
    const hips = [lh, rh].filter((i) => vis(pose, i)).map((i) => pt(pose, i));
    if (vis(pose, ls) || vis(pose, rs)) {
      [ax, ay] = pt(pose, vis(pose, ls) ? ls : rs);
    } else if (vis(pose, KP.nose)) {
      [ax, ay] = pt(pose, KP.nose);
      ay += 0.55 * scale;
    } else if (hips.length) {
      [ax, ay] = mean(hips);
      ay -= 1.45 * scale;
    } else {
      ax = (box[0] + box[2]) * 0.5;
      ay = (box[1] + box[3]) * 0.5;
    }
  }
  let seen = 0;
  for (const i of BODY) if (vis(pose, i)) seen++;
  return { ax, ay, scale: Math.max(scale, 1.0), sure, shoulders, box, quality: 0.4 + (0.6 * seen) / BODY.length };
}

export function iou(a, b) {
  const ix = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
  const iy = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  if (ix <= 0 || iy <= 0) return 0;
  const inter = ix * iy;
  const union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return union > 0 ? inter / union : 0;
}

/** How much this looks like the person playing: big, central, whole, not walking. */
export function likeness(body, width, speed) {
  const dx = Math.min(1.0, Math.abs(body.ax - width * 0.5) / (width * 0.5));
  const centre = 1.0 - 0.75 * dx * dx;
  const motion = speed == null ? 1.0 : 1.0 / (1.0 + (speed / 1.5) * (speed / 1.5));
  return (body.scale / width) * centre * body.quality * motion;
}

/** The single most player-like pose, with no history (one-off frames). Live play uses PlayerLock. */
export function mostLikelyPlayer(poses, width) {
  let best = null, bestL = -1;
  for (const p of poses) {
    const b = bodyOf(p);
    if (!b.shoulders) continue;
    const l = likeness(b, width, null);
    if (l > bestL) { best = p; bestL = l; }
  }
  return best;
}

class Track {
  constructor(id, body, pose, t) {
    this.id = id;
    this.body = body;
    this.pose = pose;
    this.firstSeen = t;
    this.lastSeen = t;
    this.vx = 0;
    this.vy = 0;
    this.hist = body.shoulders ? [[t, body.ax, body.ay]] : [];
    this.seen = true;
    this.hx = body.ax; // home: where this person has been standing (slow average)
    this.hy = body.ay;
    this.x0 = body.ax; // where they were first seen
  }

  /** Net anchor velocity (px/s) over the last SPEED_WINDOW, or null. */
  drift(minSpan = SPEED_MIN_SPAN) {
    if (this.hist.length < 2) return null;
    const [t0, x0, y0] = this.hist[0];
    const [t1, x1, y1] = this.hist[this.hist.length - 1];
    const span = t1 - t0;
    if (span < minSpan) return null;
    return [(x1 - x0) / span, (y1 - y0) / span];
  }

  /** The drift in own shoulder widths per second. */
  speed(minSpan = SPEED_MIN_SPAN) {
    const v = this.drift(minSpan);
    return v == null ? null : dist(v[0], v[1]) / this.body.scale;
  }

  predict(t, walking = true) {
    const gap = Math.max(t - this.lastSeen, 0);
    const lead = Math.min(gap, PRED_MAX);
    let dx = this.vx * lead, dy = this.vy * lead;
    const d = dist(dx, dy);
    const cap = PRED_CAP * this.body.scale;
    if (d > cap) { dx = (dx * cap) / d; dy = (dy * cap) / d; }
    if (walking && gap > PRED_MAX) {
      const v = this.drift();
      if (v != null) {
        dx += v[0] * (gap - PRED_MAX);
        dy += v[1] * (gap - PRED_MAX);
      }
    }
    return [this.body.ax + dx, this.body.ay + dy];
  }

  absorb(body, pose, t) {
    const dt = t - this.lastSeen;
    if (dt > 0 && dt <= 0.3) {
      this.vx = 0.5 * this.vx + (0.5 * (body.ax - this.body.ax)) / dt;
      this.vy = 0.5 * this.vy + (0.5 * (body.ay - this.body.ay)) / dt;
    } else {
      this.vx = this.vy = 0;
    }
    const a = Math.min(1.0, Math.max(dt, 0) / HOME_TAU);
    this.hx += a * (body.ax - this.hx);
    this.hy += a * (body.ay - this.hy);
    if (body.sure && this.body.sure) body.scale = 0.5 * this.body.scale + 0.5 * body.scale;
    this.body = body;
    this.pose = pose;
    this.lastSeen = t;
    this.seen = true;
    if (body.shoulders) this.hist.push([t, body.ax, body.ay]);
    while (this.hist.length && this.hist[0][0] < t - SPEED_WINDOW) this.hist.shift();
  }
}

export class PlayerLock {
  /**
   * update(poses, t) once per model result (t = the result's timestamp, s):
   * the locked player's pose, or null when nobody is locked or they are
   * momentarily out of sight. `others` (everyone else, for the debug view),
   * `state` ("searching" | "locked" | "holding"), `changed` (the identity
   * handed to the games differs from the previous result's: reset motion
   * history), `switches`, `playerId`, `heldFor`, `playerBox`.
   */
  constructor(width = 1280, height = 720) {
    this.width = width;
    this.height = height;
    this.reset();
  }

  reset() {
    this.tracks = [];
    this._nextId = 1;
    this.playerId = null;
    this.player = null;
    this.others = [];
    this.state = "searching";
    this.switches = 0;
    this.changed = false;
    this.heldFor = 0;
    this._challenger = null; // [track id, since]
    this._lost = null; // [body, lastSeen] of the released player
    this._lastId = null;
    this._repick = false;
    this._given = null;
    this._walked = null;
    this._owner = [];
  }

  /** Pass the lock to the next most likely player on the next result (the L key). */
  repick() {
    this._repick = true;
  }

  _track(id) {
    return this.tracks.find((tr) => tr.id === id) ?? null;
  }

  get locked() {
    return this._track(this.playerId);
  }

  get playerBox() {
    return this.locked?.body.box ?? null;
  }

  update(poses, t) {
    const bodies = poses.map(bodyOf);
    // One person reported twice (no NMS in DETR): keep the first of near-identical boxes.
    const keep = [];
    bodies.forEach((b, i) => {
      if (keep.every((k) => iou(bodies[k].box, b.box) < DUPLICATE_IOU)) keep.push(i);
    });
    this._match(keep.map((i) => poses[i]), keep.map((i) => bodies[i]), t);
    const owner = new Array(poses.length).fill(null);
    keep.forEach((i, k) => { owner[i] = this._owner[k]; });
    this._owner = owner;

    if (this._repick) {
      this._repick = false;
      this._doRepick();
    }
    let tr = this.locked;
    if (tr && tr.seen) {
      // Players play on the spot: walking for a second is leaving (or the lock
      // following somebody who walked through the player's spot). Let go.
      const speed = tr.speed(WALK_SPAN);
      if (speed != null && speed > WALK_SPEED) {
        this._release(tr, false);
        this._walked = tr.id;
        tr = null;
      }
    }
    if (tr && !tr.seen) {
      const hold = Math.abs(tr.body.ax - this.width * 0.5) >= EDGE * this.width ? EDGE_HOLD : HOLD_TIME;
      if (t - tr.lastSeen > hold) {
        this._release(tr);
        tr = null;
      }
    }
    if (!tr) this._acquire(t);
    else this._considerTakeover(t);

    tr = this.locked;
    const seen = !!tr && tr.seen;
    this.player = seen ? tr.pose : null;
    this.state = !tr ? "searching" : seen ? "locked" : "holding";
    this.heldFor = !tr || seen ? 0 : t - tr.lastSeen;
    this.others = poses.filter((_, i) => this._owner[i] !== this.playerId || !seen);
    const given = seen ? this.playerId : null;
    this.changed = given != null && this._given != null && given !== this._given;
    if (given != null) this._given = given;
    return this.player;
  }

  _match(poses, bodies, t) {
    const pairs = [];
    for (const tr of this.tracks) {
      tr.seen = false;
      const isPlayer = tr.id === this.playerId;
      const [px, py] = tr.predict(t, !isPlayer);
      const gap = Math.max(t - tr.lastSeen, 0);
      const gate = Math.min(GATE_MAX, GATE_BASE + GATE_RATE * gap) * tr.body.scale;
      bodies.forEach((b, di) => {
        let ex = px, ey = py;
        let d = dist(b.ax - px, b.ay - py);
        if (isPlayer) {
          const dh = dist(b.ax - tr.hx, b.ay - tr.hy);
          if (dh < d) { ex = tr.hx; ey = tr.hy; d = dh; }
        }
        if (d > gate) return;
        let sizeCost;
        if (b.sure && tr.body.sure) {
          const r = b.scale / tr.body.scale;
          if (r < SIZE_MIN || r > SIZE_MAX) return;
          sizeCost = W_SIZE * Math.abs(Math.log(r));
        } else {
          sizeCost = UNSURE_SIZE;
        }
        const sx = ex - tr.body.ax, sy = ey - tr.body.ay;
        const bx = tr.body.box;
        const pbox = [bx[0] + sx, bx[1] + sy, bx[2] + sx, bx[3] + sy];
        let cost = d / tr.body.scale + sizeCost + W_IOU * (1.0 - iou(pbox, b.box)) + W_GAP * gap;
        if (isPlayer) cost -= PLAYER_PRIOR;
        pairs.push([cost, tr.id, di]);
      });
    }
    pairs.sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
    const used = new Set();
    this._owner = new Array(bodies.length).fill(null);
    for (const [, tid, di] of pairs) {
      if (used.has(tid) || this._owner[di] != null) continue;
      used.add(tid);
      this._owner[di] = tid;
      this._track(tid).absorb(bodies[di], poses[di], t);
    }
    bodies.forEach((b, di) => {
      if (this._owner[di] != null) return;
      const tr = new Track(this._nextId++, b, poses[di], t);
      this.tracks.push(tr);
      this._owner[di] = tr.id;
    });
    this.tracks = this.tracks.filter((tr) => tr.seen || tr.id === this.playerId || t - tr.lastSeen <= FORGET_TIME);
    if (this.tracks.length > MAX_TRACKS) {
      const key = (tr) => [tr.id !== this.playerId ? 1 : 0, -tr.lastSeen, tr.id];
      this.tracks.sort((a, b) => {
        const ka = key(a), kb = key(b);
        return ka[0] - kb[0] || ka[1] - kb[1] || ka[2] - kb[2];
      });
      this.tracks = this.tracks.slice(0, MAX_TRACKS).sort((a, b) => a.id - b.id);
    }
  }

  _eligible(tr) {
    if (!tr.seen || !tr.body.shoulders || tr.body.scale < MIN_SCALE * this.width) return false;
    const speed = tr.speed();
    return speed == null || speed <= WALK_SPEED;
  }

  _dx(tr) {
    return Math.abs(tr.body.ax - this.width * 0.5) / this.width;
  }

  _like(tr) {
    return likeness(tr.body, this.width, tr.speed());
  }

  _ranked(exclude = null) {
    const cands = this.tracks.filter((tr) => this._eligible(tr) && tr.id !== exclude);
    const like = new Map(cands.map((tr) => [tr.id, this._like(tr)]));
    cands.sort((a, b) => like.get(b.id) - like.get(a.id) || a.id - b.id);
    return cands;
  }

  _lock(tr, recall = false) {
    if (!recall && this._lastId != null && tr.id !== this._lastId) this.switches++;
    this.playerId = tr.id;
    this._lastId = tr.id;
    this._challenger = null;
    this._lost = null;
  }

  _release(tr, recall = true) {
    this._lost = recall ? [tr.body, tr.lastSeen] : null;
    this.playerId = null;
    this._challenger = null;
  }

  _acquire(t) {
    if (this._walked != null) {
      // Banned while walking out; welcome again once they stop.
      const w = this._track(this._walked);
      const speed = w ? w.speed() : null;
      if (!w || (speed != null && speed <= TAKEOVER_STILL)) this._walked = null;
    }
    let cands = this._ranked(this._walked);
    if (!cands.length) {
      this._challenger = null;
      return;
    }
    if (this._lost) {
      const [lost, when] = this._lost;
      if (t - when > RECALL_TIME) this._lost = null;
      else {
        // The player we let go of, back where they were: the same person.
        for (const tr of cands) {
          const d = dist(tr.body.ax - lost.ax, tr.body.ay - lost.ay);
          const r = tr.body.scale / lost.scale;
          if (d <= 1.5 * lost.scale && r >= SIZE_MIN && r <= SIZE_MAX) {
            this._lock(tr, true);
            return;
          }
        }
      }
    }
    if (cands.length === 1 && this._dx(cands[0]) <= CENTRAL) {
      // Alone and central: no contest, unless they walked in from the side.
      const lone = cands[0];
      if (lone.speed() != null || Math.abs(lone.x0 - this.width * 0.5) <= CENTRAL * this.width) {
        this._lock(lone);
        return;
      }
    }
    // A contest is only between people known not to be walking past.
    cands = cands.filter((tr) => tr.speed() != null);
    if (!cands.length) {
      this._challenger = null;
      return;
    }
    const best = cands[0];
    let pick = null;
    if (this._challenger) {
      pick = cands.find((tr) => tr.id === this._challenger[0]) ?? null;
      if (pick && this._like(pick) < KEEP_CLAIM * this._like(best)) pick = null;
    }
    if (!pick) {
      const first = this._challenger == null;
      this._challenger = [best.id, first ? t - ACQUIRE_TIME : t];
      pick = best;
    }
    if (t - this._challenger[1] >= ACQUIRE_TIME) this._lock(pick);
  }

  _considerTakeover(t) {
    const cur = this.locked;
    let best = null;
    for (const tr of this._ranked(this.playerId)) {
      const speed = tr.speed();
      if (this._dx(tr) > TAKEOVER_CENTRAL || speed == null || speed > TAKEOVER_STILL) continue;
      const bigger = tr.body.scale >= TAKEOVER_BIGGER * cur.body.scale;
      const aside = this._dx(cur) >= TAKEOVER_ASIDE && tr.body.scale >= 0.8 * cur.body.scale;
      if (bigger || aside) {
        best = tr;
        break;
      }
    }
    if (!best) {
      this._challenger = null;
      return;
    }
    if (!this._challenger || this._challenger[0] !== best.id) this._challenger = [best.id, t];
    else if (t - this._challenger[1] >= TAKEOVER_TIME) this._lock(best);
  }

  _doRepick() {
    const cands = this._ranked(this.playerId);
    if (cands.length) this._lock(cands[0]);
    else if (!this.locked || !this.locked.seen) {
      this.playerId = null;
      this._challenger = null;
      this._lost = null;
    }
  }
}
