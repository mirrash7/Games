// Raccoon Golf's course: holes as data, and the ground questions physics asks
// (port of src/kpapp/game/golf/course.py - keep the two in step).
//
// The course lives in assets/golf/course.json (hand-authored, original): one
// entry per hole with polygons in metres. Each hole has its own frame: the tee
// near the origin, +y running down the hole, +x to the right as seen from the
// tee, z up. The ground is flat (z = 0) everywhere; only the green has a height
// field, and it is used for roll, not for drawing.
//
//     land     the island; anything outside it is the sea, i.e. out of bounds
//     fairway  one or more polygons, mown in stripes
//     green, tee_box
//     bunkers, water (ponds and creeks: a water hazard, not OB)
//     trees    {x, y, height, radius, kind}: radius is the canopy's
//     path     the hole's centre line, tee -> pin (auto-aim follows it on doglegs)
//     slope    green height h(x, y) = a (x - pin x) + b (y - pin y)
//                                     + sum A exp(-d^2 / (2 s^2)) over bumps
//     wind     [min, max] mph; each hole draws its own wind from this range
//
// Points are [x, y] arrays. No DOM here: the tests parse the JSON under Node.

import { loadJSON } from "../../core/gfx.js";

export const COURSE_URL = new URL("../../../assets/golf/course.json", import.meta.url);

export const YARDS_PER_M = 1.0936133;
export const FEET_PER_M = 3.2808399;
export const MPH = 0.44704; // m/s per mph

// Lies, in the order they win when polygons overlap (water drawn over a
// fairway is water). Anything on the island and in none of them is rough.
export const LIES = ["ob", "water", "bunker", "green", "tee", "fairway", "rough"];

/** Even-odd ray cast. `poly` is a list of [x, y]. */
export function pointInPoly(x, y, poly) {
  let inside = false;
  const n = poly.length;
  let j = n - 1;
  for (let i = 0; i < n; i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
    j = i;
  }
  return inside;
}

/** A polygon with a cached bounding box (most lookups miss the box). */
export class Area {
  constructor(pts) {
    this.pts = pts.map((p) => [Number(p[0]), Number(p[1])]);
    const xs = this.pts.map((p) => p[0]), ys = this.pts.map((p) => p[1]);
    this.x0 = Math.min(...xs);
    this.x1 = Math.max(...xs);
    this.y0 = Math.min(...ys);
    this.y1 = Math.max(...ys);
  }

  contains(x, y) {
    if (x < this.x0 || x > this.x1 || y < this.y0 || y > this.y1) return false;
    return pointInPoly(x, y, this.pts);
  }
}

export class Hole {
  constructor(o) {
    Object.assign(this, o);
  }

  // --- where is the ball? ---

  lieAt(x, y) {
    if (!this.land.contains(x, y)) return "ob";
    for (const a of this.water) if (a.contains(x, y)) return "water";
    for (const a of this.bunkers) if (a.contains(x, y)) return "bunker";
    if (this.green.contains(x, y)) return "green";
    if (this.teeBox.contains(x, y)) return "tee";
    for (const a of this.fairway) if (a.contains(x, y)) return "fairway";
    return "rough";
  }

  distanceToPin(x, y) {
    return Math.hypot(this.pin[0] - x, this.pin[1] - y);
  }

  // --- the green's height field ---

  greenHeight(x, y) {
    const [a, b] = this.slopePlane;
    let h = a * (x - this.pin[0]) + b * (y - this.pin[1]);
    for (const [bx, by, s, amp] of this.bumps) h += amp * Math.exp(-((x - bx) ** 2 + (y - by) ** 2) / (2 * s * s));
    return h;
  }

  /** d(height)/dx, d(height)/dy: a ball rolls toward minus this. */
  greenGradient(x, y) {
    let [gx, gy] = this.slopePlane;
    for (const [bx, by, s, amp] of this.bumps) {
      const e = amp * Math.exp(-((x - bx) ** 2 + (y - by) ** 2) / (2 * s * s));
      gx -= (e * (x - bx)) / (s * s);
      gy -= (e * (y - by)) / (s * s);
    }
    return [gx, gy];
  }

  // --- aiming ---

  /**
   * Where auto-aim points from (x, y) with a club that carries `reach` m: the
   * pin if it is in range; otherwise the point on the hole's centre line
   * `reach` metres away (ahead of the ball), so a tee shot on a dogleg aims
   * down the fairway rather than over the trees at the green.
   */
  aimPoint(x, y, reach) {
    if (this.distanceToPin(x, y) <= reach * 1.05) return this.pin;
    const path = this.path;
    let bestI = 0, bestU = 0, bestD = Infinity;
    for (let i = 0; i < path.length - 1; i++) {
      const [ax, ay] = path[i], [bx, by] = path[i + 1];
      const dx = bx - ax, dy = by - ay;
      const seg = dx * dx + dy * dy;
      const u = seg <= 0 ? 0 : Math.min(1, Math.max(0, ((x - ax) * dx + (y - ay) * dy) / seg));
      const d = Math.hypot(ax + dx * u - x, ay + dy * u - y);
      if (d < bestD) [bestI, bestU, bestD] = [i, u, d];
    }
    for (let i = bestI; i < path.length - 1; i++) {
      const [ax, ay] = path[i], [bx, by] = path[i + 1];
      const u0 = i === bestI ? bestU : 0;
      const steps = 40;
      for (let k = 0; k <= steps; k++) {
        const u = u0 + ((1 - u0) * k) / steps;
        const px = ax + (bx - ax) * u, py = ay + (by - ay) * u;
        if (Math.hypot(px - x, py - y) >= reach) return [px, py];
      }
    }
    return this.pin;
  }

  /** Tee to pin along the centre line, in metres (the card's yardage). */
  get length() {
    let s = 0;
    for (let i = 0; i + 1 < this.path.length; i++) {
      s += Math.hypot(this.path[i + 1][0] - this.path[i][0], this.path[i + 1][1] - this.path[i][1]);
    }
    return s;
  }
}

const pair = (p) => [Number(p[0]), Number(p[1])];

export function holeFromDict(d, number) {
  const slope = d.slope ?? {};
  const wind = d.wind ?? [0, 8];
  return new Hole({
    number,
    name: d.name ?? `Hole ${number}`,
    par: Math.trunc(Number(d.par)),
    tee: pair(d.tee),
    pin: pair(d.pin),
    land: new Area(d.land),
    fairway: (d.fairway ?? []).map((p) => new Area(p)),
    green: new Area(d.green),
    teeBox: new Area(d.tee_box),
    bunkers: (d.bunkers ?? []).map((p) => new Area(p)),
    water: (d.water ?? []).map((p) => new Area(p)),
    trees: (d.trees ?? []).map((t) => ({ x: Number(t.x), y: Number(t.y), height: Number(t.height),
      radius: Number(t.radius), kind: t.kind ?? "palm" })),
    path: d.path.map(pair),
    slopePlane: [Number(slope.a ?? 0), Number(slope.b ?? 0)],
    bumps: (slope.bumps ?? []).map((b) => b.map(Number)),
    windRange: [Number(wind[0]), Number(wind[1])],
  });
}

export class Course {
  constructor(name, holes) {
    this.name = name;
    this.holes = holes;
  }
  get par() {
    return this.holes.reduce((s, h) => s + h.par, 0);
  }
}

/** A Course from course.json's parsed contents. */
export function parseCourse(data) {
  return new Course(data.name ?? "Course", data.holes.map((h, i) => holeFromDict(h, i + 1)));
}

let cache = null;

/** The course loaded by loadCourse() (or set by the tests), else null. */
export function cachedCourse() {
  return cache;
}

/** Use this course from now on (tests read course.json from disk and pass it here). */
export function setCourse(course) {
  cache = course;
  return course;
}

export async function loadCourse() {
  if (cache) return cache;
  return setCourse(parseCourse(await loadJSON(COURSE_URL.href)));
}
