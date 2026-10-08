// Turning a tracked hand into a blade, and deciding what it cut
// (port of game/fruitninja/blade.py). Points are [x, y] arrays.
//
// Three things make cutting feel fair rather than fussy:
//
// Swept segments. Poses arrive at ~15-30 Hz, so a fast hand jumps a long way
// between samples. Every segment swept between samples is tested, so nothing
// can tunnel through a snack.
//
// A blade with width, and a trail that still cuts. The blade is a thick stroke
// (`hitRadius`), and the recent trail stays sharp for `cutWindow` seconds, so a
// snack that flies into the streak a moment after the swing is still eaten.
//
// Speed measured over a window. Inference updates the hand in steps while the
// game runs at 60 Hz, so frame-to-frame speed flickers between ~0 and huge; a
// short window average does not.
//
// A bite still needs real speed (resting a hand on a snack must not dissolve
// the board), and an impossible jump is the tracker losing the hand, not a
// swing - recorded play once drew a stroke the full height of the frame.

import { handKeypoint } from "../../core/hand.js";

// Re-exported: tests know it by this name.
export const bladeKeypoint = handKeypoint;

export const MIN_SLICE_SPEED = 260.0; // px/s, over SPEED_WINDOW
export const TRAIL_SECONDS = 0.22; // how long the drawn streak lingers
export const CUT_WINDOW = 0.15; // how long the streak stays sharp enough to cut
export const HIT_RADIUS = 34.0; // blade half-width in px, added to each snack's radius
export const SPEED_WINDOW = 0.08;
export const MAX_JUMP = 450.0; // px between consecutive samples; beyond this it's a glitch
const MAX_SAMPLES = 64;

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

export class Blade {
  constructor({
    trailSeconds = TRAIL_SECONDS,
    minSpeed = MIN_SLICE_SPEED,
    hitRadius = HIT_RADIUS,
    cutWindow = CUT_WINDOW,
    maxJump = MAX_JUMP,
  } = {}) {
    this.trailSeconds = trailSeconds;
    this.minSpeed = minSpeed;
    this.hitRadius = hitRadius;
    this.cutWindow = cutWindow;
    this.maxJump = maxJump;
    this._samples = []; // {pos, t, speed}, oldest first
    this.speed = 0;
    this.active = false; // is the hand visible at all
    this.slicing = false; // visible AND moving fast enough to cut
  }

  get pos() {
    const n = this._samples.length;
    return n ? this._samples[n - 1].pos : null;
  }

  /** The path swept since the previous sample, or null. */
  get segment() {
    const n = this._samples.length;
    if (n < 2) return null;
    return [this._samples[n - 2].pos, this._samples[n - 1].pos];
  }

  /** Recent points, newest last, for drawing the streak. */
  trail(now) {
    const cutoff = now - this.trailSeconds;
    const out = [];
    for (const s of this._samples) if (s.t >= cutoff) out.push(s.pos);
    return out;
  }

  /** Hand went missing: break the trail so it does not draw a long jump. */
  loseTrack() {
    this._samples.length = 0;
    this.speed = 0;
    this.active = false;
    this.slicing = false;
  }

  /** Path length over the last SPEED_WINDOW, divided by the time it spans. */
  _windowSpeed(now) {
    const s = this._samples;
    let length = 0;
    let start = null;
    for (let i = 1; i < s.length; i++) {
      const a = s[i - 1], b = s[i];
      if (b.t < now - SPEED_WINDOW) continue;
      length += dist(b.pos, a.pos);
      if (start === null) start = a.t;
    }
    if (start === null || now <= start) return 0;
    return length / (now - start);
  }

  update(pos, t) {
    if (pos == null) {
      this.loseTrack();
      return;
    }
    pos = [pos[0], pos[1]];
    const s = this._samples;
    if (s.length) {
      const prev = s[s.length - 1];
      if (t - prev.t <= 0) return;
      if (dist(pos, prev.pos) > this.maxJump) {
        // The tracker lost the hand and found it (or a phantom) far away.
        // Start a fresh stroke instead of drawing - and cutting along - a
        // line across the screen.
        s.length = 0;
      }
    }
    s.push({ pos, t, speed: 0 });
    if (s.length > MAX_SAMPLES) s.shift();
    while (s.length > 2 && s[0].t < t - this.trailSeconds) s.shift();

    this.speed = this._windowSpeed(t);
    s[s.length - 1].speed = this.speed;
    this.active = true;
    this.slicing = this.speed >= this.minSpeed;
  }

  /**
   * Cut direction if any sharp part of the recent trail crossed the circle.
   * Tests every segment within `cutWindow`, newest first, each widened by
   * `hitRadius`. Returns the unit direction [dx, dy] of the cutting segment
   * (used to split the halves), or null.
   */
  hits(centre, radius) {
    const s = this._samples;
    if (s.length < 2) return null;
    const now = s[s.length - 1].t;
    const reach = radius + this.hitRadius;
    for (let i = s.length - 1; i >= 1; i--) {
      const a = s[i - 1], b = s[i];
      if (b.t < now - this.cutWindow) break;
      if (b.speed < this.minSpeed) continue;
      const dx = b.pos[0] - a.pos[0], dy = b.pos[1] - a.pos[1];
      const lengthSq = dx * dx + dy * dy;
      if (lengthSq < 1e-6) continue;
      let u = ((centre[0] - a.pos[0]) * dx + (centre[1] - a.pos[1]) * dy) / lengthSq;
      u = Math.min(Math.max(u, 0), 1);
      const px = a.pos[0] + dx * u, py = a.pos[1] + dy * u;
      if (Math.hypot(px - centre[0], py - centre[1]) <= reach) {
        const len = Math.sqrt(lengthSq);
        return [dx / len, dy / len];
      }
    }
    return null;
  }
}
