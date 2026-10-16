// Drawing Raccoon Golf: a pinhole 3D view of the hole, Lenny, and the HUD
// (port of src/kpapp/game/golf/render.py, drawn with Canvas 2D; the layout,
// colours and sizes are the Python's - compare with web/dev/golf.html).
//
// The camera is a pinhole (focal FOCAL px at 1280x720, ~65 degrees across)
// behind the ball. The ground is flat, so each surface of the hole (island,
// fairway, mowing stripes, green, bunkers, water) is a polygon in metres that is
// transformed, clipped at the near plane (a polygon that reaches behind the
// camera would otherwise project inside out) and filled. Trees are billboards
// sorted far to near. The sky is a panorama sliced by the camera's yaw.
//
// The live camera image is not the backdrop: like Flappy Raccoon the game is a
// full scene and the camera goes picture-in-picture (top right), where it shows
// the player their tracked grip.
//
// While the camera is still (aiming, results, the scorecard) the 3D world is
// cached on an offscreen canvas and each frame is one copy plus the moving parts.
//
// No DOM at module level: game.js imports this under Node. Colours are RGB.

import { makeCanvas as defaultMakeCanvas } from "../../core/gfx.js";
import { KP } from "../../core/pose.js";
import {
  ACCENT, CYAN, DANGER, DIM, PANEL, WARN, WHITE, centredText, css, panel, roundRectPath, text,
} from "../../core/theme.js";
import * as P from "./physics.js";
import { YARDS_PER_M } from "./course.js";
import { HOLED_PULL, toParText } from "./game.js";

export const NEAR = 0.4; // m: the near clipping plane
export const STRIPE = 8.0; // m: fairway mowing stripes
export const GREEN_STRIPE = 2.5;
export const STRIPE_FAR = 170.0; // m: stripes beyond this would only shimmer
export const PIN_HEIGHT = 2.2; // m (drawn no shorter than PIN_MIN_PX)
export const PIN_MIN_PX = 34;
export const LENNY_WORLD = 0.0093; // Lenny's sprite scale per (focal / depth): 0.95 at the address view
export const LENNY_MAX = 1.15;
export const THETA_TOP = 2.75; // radians of arm swing at a full backswing (swing angle 1)
export const THETA_FINISH = 2.6;
export const ARC_TOP = 118; // px: the dotted arc stops short of the top HUD row
export const LENNY_FOOT = [133.0, 292.0]; // px in lenny.png between his shoes: where he stands when not addressing

export const PIP = [944, 16, 1264, 196]; // x0, y0, x1, y1: the live camera window
export const MAP = [1146, 208, 1264, 630]; // the course overview
export const GAUGE_AT = [22, 318]; // top-left of the power gauge (drawn at GAUGE_SCALE)
export const GAUGE_SCALE = 0.85;
export const HOLE_PANEL = [16, 14, 330, 134];
export const CLUB_PANEL = [16, 142, 330, 212];
export const WIND_PANEL = [566, 12, 714, 106];

const GOLD = ACCENT;
const INK = [24, 16, 20];
const DARK = [20, 20, 20];

// --- geometry ---------------------------------------------------------------

/** A pinhole camera: world (x, y, z) metres -> screen pixels. */
export class View {
  constructor(cam, w, h, f, ox = 0.0, oy = 0.0) {
    this.w = w;
    this.h = h;
    this.f = f;
    this.cx = w / 2 + ox;
    this.cy = h / 2 + oy;
    const sy = Math.sin(cam.yaw), cy = Math.cos(cam.yaw);
    const sp = Math.sin(cam.pitch), cp = Math.cos(cam.pitch);
    this.R = [[cy, -sy, 0.0], [sy * sp, cy * sp, cp], [sy * cp, cy * cp, -sp]];
    this.C = [cam.x, cam.y, cam.z];
    this.horizon = this.cy - f * Math.tan(cam.pitch);
    this.yaw = cam.yaw;
  }

  /** World [x, y, z] -> camera [x right, y up, z forward]. */
  toCam(p) {
    const dx = p[0] - this.C[0], dy = p[1] - this.C[1], dz = (p[2] ?? 0) - this.C[2];
    const R = this.R;
    return [R[0][0] * dx + R[0][1] * dy + R[0][2] * dz, R[1][0] * dx + R[1][1] * dy + R[1][2] * dz,
      R[2][0] * dx + R[2][1] * dy + R[2][2] * dz];
  }

  project(q) {
    return [this.cx + (this.f * q[0]) / q[2], this.cy - (this.f * q[1]) / q[2]];
  }

  /** [sx, sy, depth] or null behind the near plane. */
  point(x, y, z = 0.0) {
    const q = this.toCam([x, y, z]);
    if (q[2] < NEAR) return null;
    return [this.cx + (this.f * q[0]) / q[2], this.cy - (this.f * q[1]) / q[2], q[2]];
  }
}

/** Sutherland-Hodgman against z >= NEAR for one polygon in camera space. */
export function clipNear(q) {
  const out = [];
  const n = q.length;
  for (let i = 0; i < n; i++) {
    const a = q[i], b = q[(i + 1) % n];
    const ina = a[2] >= NEAR, inb = b[2] >= NEAR;
    if (ina) out.push(a);
    if (ina !== inb) {
      const t = (NEAR - a[2]) / (b[2] - a[2]);
      out.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]);
    }
  }
  return out;
}

/** Grow (d > 0) or shrink a simple polygon by d metres (mitred, clamped). */
export function offsetPoly(pts, d) {
  const n = pts.length;
  let area = 0;
  for (let i = 0; i < n; i++) {
    const a = pts[i], b = pts[(i + 1) % n];
    area += a[0] * b[1] - b[0] * a[1];
  }
  area *= 0.5;
  const sign = area < 0 ? 1.0 : -1.0;
  const out = [];
  for (let i = 0; i < n; i++) {
    const a = pts[(i - 1 + n) % n], b = pts[i], c = pts[(i + 1) % n];
    const e1 = [b[0] - a[0], b[1] - a[1]], e2 = [c[0] - b[0], c[1] - b[1]];
    const l1 = Math.hypot(e1[0], e1[1]) || 1.0, l2 = Math.hypot(e2[0], e2[1]) || 1.0;
    const n1 = [e1[1] / l1, -e1[0] / l1], n2 = [e2[1] / l2, -e2[0] / l2];
    let m = [n1[0] + n2[0], n1[1] + n2[1]];
    const ml = Math.hypot(m[0], m[1]);
    m = ml < 1e-9 ? n1 : [m[0] / ml, m[1] / ml];
    const k = Math.min(3.0, 1.0 / Math.max(0.2, m[0] * n1[0] + m[1] * n1[1]));
    out.push([b[0] + m[0] * d * k * -sign, b[1] + m[1] * d * k * -sign]);
  }
  return out;
}

/** Keep the part of `poly` where nx x + ny y >= c. */
export function clipHalfplane(poly, nx, ny, c) {
  const out = [];
  const n = poly.length;
  for (let i = 0; i < n; i++) {
    const a = poly[i], b = poly[(i + 1) % n];
    const da = nx * a[0] + ny * a[1] - c, db = nx * b[0] + ny * b[1] - c;
    if (da >= 0) out.push(a);
    if ((da >= 0) !== (db >= 0)) {
      const t = da / (da - db);
      out.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
    }
  }
  return out;
}

/** Every other `width`-wide band of `poly` across `axis` (mowing stripes). */
export function stripes(poly, axis, width, origin) {
  const [ax, ay] = axis;
  const s = poly.map((p) => (p[0] - origin[0]) * ax + (p[1] - origin[1]) * ay);
  const k0 = Math.floor(Math.min(...s) / width), k1 = Math.ceil(Math.max(...s) / width);
  const out = [];
  for (let k = k0; k <= k1; k++) {
    if (((k % 2) + 2) % 2) continue;
    const lo = origin[0] * ax + origin[1] * ay + k * width;
    let band = clipHalfplane(poly, ax, ay, lo);
    band = clipHalfplane(band, -ax, -ay, -(lo + width));
    if (band.length >= 3) out.push(band);
  }
  return out;
}

const layer = (colour, polys, farCut = Infinity, edge = null) => ({ colour, polys, farCut, edge });

function tracePoly(ctx, pts) {
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
  ctx.closePath();
}

export function drawLayer(ctx, view, ly) {
  if (!ly.polys.length) return;
  const shown = [];
  for (const poly of ly.polys) {
    let q = poly.map((p) => view.toCam([p[0], p[1], 0.0]));
    let zmin = Infinity, zmax = -Infinity;
    for (const v of q) {
      zmin = Math.min(zmin, v[2]);
      zmax = Math.max(zmax, v[2]);
    }
    if (zmax < NEAR || zmin > ly.farCut) continue;
    if (zmin < NEAR) {
      q = clipNear(q);
      if (q.length < 3) continue;
    }
    const s = q.map((v) => view.project(v));
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const [x, y] of s) {
      x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
    }
    if (x1 < 0 || x0 > view.w || y1 < view.horizon - 2 || y0 > view.h) continue;
    shown.push(s.map(([x, y]) => [Math.max(-60000, Math.min(60000, x)), Math.max(-60000, Math.min(60000, y))]));
  }
  if (!shown.length) return;
  ctx.fillStyle = css(ly.colour);
  for (const s of shown) {
    ctx.beginPath();
    tracePoly(ctx, s);
    ctx.fill();
  }
  if (ly.edge) {
    ctx.strokeStyle = css(ly.edge);
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (const s of shown) tracePoly(ctx, s);
    ctx.stroke();
  }
}

// --- per-hole scenery -------------------------------------------------------

/** Everything about a hole that is built once: ground layers and the green's grid. */
export class HoleScene {
  constructor(hole, g) {
    this.hole = hole;
    const [tx, ty] = hole.tee, [px, py] = hole.pin;
    const L = Math.hypot(px - tx, py - ty) || 1.0;
    this.axis = [(px - tx) / L, (py - ty) / L];
    const fair = hole.fairway.map((a) => a.pts);
    const green = hole.green.pts;
    const water = hole.water.map((w) => w.pts);
    this.ground = [
      layer(g.beach, [offsetPoly(hole.land.pts, 5.0)]),
      layer(g.rough, [hole.land.pts]),
      layer(g.fairway, fair, Infinity, g.edge),
      layer(g.fairway_stripe, fair.flatMap((f) => stripes(f, this.axis, STRIPE, hole.tee)), STRIPE_FAR),
      layer(g.fringe, [offsetPoly(green, 1.6)]),
      layer(g.green, [green]),
      layer(g.green_stripe, stripes(green, this.axis, GREEN_STRIPE, hole.pin), 60.0),
      layer(g.tee, [hole.teeBox.pts], Infinity, g.edge),
      layer(g.bunker_rim, hole.bunkers.map((b) => offsetPoly(b.pts, 0.7))),
      layer(g.bunker, hole.bunkers.map((b) => b.pts)),
      layer(g.water_rim, water.map((w) => offsetPoly(w, 1.0))),
      layer(g.water, water),
      layer(g.water_hi, water.map((w) => offsetPoly(w, -1.4))),
      layer(g.water, water.map((w) => offsetPoly(w, -2.2))),
    ];
    this.shadow = g.rough.map((v) => Math.trunc(v * 0.72));
    [this.gridLines, this.flowPts] = greenGrid(hole);
  }
}

/** numpy.arange(a, b, step) as a list. */
function arange(a, b, step) {
  const n = Math.max(0, Math.ceil((b - a) / step));
  return Array.from({ length: n }, (_, i) => a + i * step);
}

/** World-space grid lines over the green, and the points for slope arrows. */
export function greenGrid(hole) {
  const a = hole.green;
  const lines = [];
  const step = 1.5, sample = 0.3;
  const runs = (fixed, along, xFirst) => {
    for (const c of fixed) {
      let run = [];
      for (const v of along) {
        const [x, y] = xFirst ? [c, v] : [v, c];
        if (a.contains(x, y)) run.push([x, y, 0.0]);
        else {
          if (run.length > 1) lines.push(run);
          run = [];
        }
      }
      if (run.length > 1) lines.push(run);
    }
  };
  runs(arange(Math.floor(a.x0), a.x1 + step, step), arange(a.y0, a.y1 + sample, sample), true);
  runs(arange(Math.floor(a.y0), a.y1 + step, step), arange(a.x0, a.x1 + sample, sample), false);
  const pts = [];
  for (const xs of arange(a.x0 + 0.75, a.x1, step)) {
    for (const ys of arange(a.y0 + 0.75, a.y1, step)) {
      if (a.contains(xs, ys)) {
        const [gx, gy] = hole.greenGradient(xs, ys);
        pts.push([xs, ys, gx, gy]);
      }
    }
  }
  return [lines, pts];
}

/** A top-down map of the hole, tee at the bottom, pin at the top: {canvas, toMap}. */
export function buildMap(hole, g, w, h, makeCanvas = defaultMakeCanvas) {
  const [tx, ty] = hole.tee, [px, py] = hole.pin;
  const ang = Math.atan2(px - tx, py - ty); // rotate this heading to straight up
  const ca = Math.cos(ang), sa = Math.sin(ang);
  const rot = (x, y) => [(x - tx) * ca - (y - ty) * sa, (x - tx) * sa + (y - ty) * ca];
  const pts = [...hole.land.pts.map((p) => rot(p[0], p[1])), rot(px, py), rot(tx, ty)];
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const pad = 8;
  const s = Math.min((w - 2 * pad) / (Math.max(...xs) - Math.min(...xs)), (h - 2 * pad) / (Math.max(...ys) - Math.min(...ys)));
  const mx = (Math.max(...xs) + Math.min(...xs)) / 2, my = (Math.max(...ys) + Math.min(...ys)) / 2;
  const toMap = (x, y) => {
    const [rx, ry] = rot(x, y);
    return [w / 2 + (rx - mx) * s, h / 2 - (ry - my) * s];
  };
  const canvas = makeCanvas(w, h);
  const c = canvas.getContext("2d");
  c.fillStyle = css(g.sea);
  c.fillRect(0, 0, w, h);
  const fill = (poly, colour) => {
    c.fillStyle = css(colour);
    c.beginPath();
    tracePoly(c, poly.map((p) => toMap(p[0], p[1])));
    c.fill();
  };
  fill(offsetPoly(hole.land.pts, 5.0), g.beach);
  fill(hole.land.pts, g.rough);
  for (const a of hole.fairway) fill(a.pts, g.fairway);
  fill(offsetPoly(hole.green.pts, 1.6), g.fringe);
  fill(hole.green.pts, g.green);
  fill(hole.teeBox.pts, g.tee);
  for (const b of hole.bunkers) fill(b.pts, g.bunker);
  for (const wt of hole.water) fill(wt.pts, g.water);
  c.fillStyle = css([40, 100, 40]);
  for (const t of hole.trees) {
    const [x, y] = toMap(t.x, t.y);
    c.beginPath();
    c.arc(x, y, Math.max(2.0, t.radius * s), 0, Math.PI * 2);
    c.fill();
  }
  return { canvas, toMap };
}

// --- small drawing helpers --------------------------------------------------

const easeOut = (t) => 1.0 - (1.0 - Math.min(1.0, Math.max(0.0, t))) ** 3;

/** Overshoot then settle: 0.6 -> 1.25 -> 1.0 over the first 0.3 s. */
export function pop(age) {
  if (age < 0.12) return 0.6 + (age / 0.12) * 0.65;
  if (age < 0.3) return 1.25 - ((age - 0.12) / 0.18) * 0.25;
  return 1.0;
}

function capsule(ctx, a, b, r, colour) {
  ctx.lineCap = "round";
  ctx.strokeStyle = css(colour);
  ctx.lineWidth = Math.max(1, Math.round(2 * r));
  ctx.beginPath();
  ctx.moveTo(a[0], a[1]);
  ctx.lineTo(b[0], b[1]);
  ctx.stroke();
}

function disc(ctx, x, y, r, colour) {
  ctx.fillStyle = css(colour);
  ctx.beginPath();
  ctx.arc(x, y, Math.max(0.5, r), 0, Math.PI * 2);
  ctx.fill();
}

function ellipse(ctx, x, y, rx, ry, rot, colour) {
  ctx.fillStyle = css(colour);
  ctx.beginPath();
  ctx.ellipse(x, y, Math.max(0.5, rx), Math.max(0.5, ry), rot, 0, Math.PI * 2);
  ctx.fill();
}

/** A filled arrow head pointing left/right/up/down. */
function triangle(ctx, cx, cy, size, direction, colour, outline = INK) {
  const d = { left: [-1, 0], right: [1, 0], up: [0, -1], down: [0, 1] }[direction];
  const nx = -d[1], ny = d[0];
  const tip = [cx + d[0] * size, cy + d[1] * size];
  const b1 = [cx - d[0] * size * 0.6 + nx * size * 0.8, cy - d[1] * size * 0.6 + ny * size * 0.8];
  const b2 = [cx - d[0] * size * 0.6 - nx * size * 0.8, cy - d[1] * size * 0.6 - ny * size * 0.8];
  ctx.beginPath();
  tracePoly(ctx, [tip, b1, b2]);
  ctx.lineJoin = "round";
  ctx.lineWidth = 5;
  ctx.strokeStyle = css(outline);
  ctx.stroke();
  ctx.fillStyle = css(colour);
  ctx.fill();
}

/** A rounded rectangle's outline only. */
function border(ctx, rect, r, colour, thickness) {
  const [x0, y0, x1, y1] = rect;
  roundRectPath(ctx, x0, y0, x1, y1, r);
  ctx.lineWidth = thickness;
  ctx.strokeStyle = css(colour);
  ctx.stroke();
}

/** Text at a baseline, left-aligned, outlined (overlay.outlined_text). */
const label = (ctx, s, x, y, scale, colour, body = false) => text(ctx, s, x, y, scale, colour, { body });

const faded = (ctx, s, cx, y, scale, colour, alpha, body = false) => {
  if (alpha <= 0.01) return;
  centredText(ctx, s, cx, y, scale, colour, { alpha: Math.min(1, alpha), body });
};

// --- the renderer -----------------------------------------------------------

export class GolfRenderer {
  constructor(width, height, art, { makeCanvas = defaultMakeCanvas } = {}) {
    this.width = width;
    this.height = height;
    this.art = art;
    this.makeCanvas = makeCanvas;
    const m = art.manifest;
    this.m = m;
    this.k = width / 1280.0;
    this.f = Number(m.focal) * this.k;
    this.g = m.ground;
    this.panoW = Math.round(Number(m.pano_w) * this.k);
    this.panoH = Math.round(Number(m.pano_h) * this.k);
    this.treeSpec = m.trees;
    this.lennySpec = m.lenny;
    const gw = Math.max(1, Math.round(art.gauge.width * GAUGE_SCALE));
    const gh = Math.max(1, Math.round(art.gauge.height * GAUGE_SCALE));
    this.gaugeSize = [gw, gh];
    this.gaugeInner = m.gauge.inner.map((v) => v * GAUGE_SCALE);
    this.scenes = new Map();
    this.maps = new Map();
    this._cacheKey = null;
    this._cache = null;
  }

  // --- scene pieces ---

  scene(game) {
    let sc = this.scenes.get(game.holeI);
    if (!sc) this.scenes.set(game.holeI, (sc = new HoleScene(game.hole, this.g)));
    return sc;
  }

  holeMap(game) {
    let mp = this.maps.get(game.holeI);
    if (!mp) this.maps.set(game.holeI, (mp = buildMap(game.hole, this.g, MAP[2] - MAP[0], MAP[3] - MAP[1], this.makeCanvas)));
    return mp;
  }

  /** A strip of a horizontally tiling image, wrapping at its width. */
  _wrapped(ctx, img, sx, sy, sw, sh, dy, dh, tileW, srcScale) {
    const W = this.width;
    let x = ((sx % tileW) + tileW) % tileW;
    let dx = 0;
    while (dx < W) {
      const take = Math.min(tileW - x, W - dx);
      ctx.drawImage(img, x * srcScale, sy, take * srcScale, sh, dx, dy, take, dh);
      dx += take;
      x = 0;
    }
  }

  drawBackdrop(ctx, view) {
    const W = this.width, H = this.height;
    const hy = Math.round(view.horizon);
    const sky = this.art.sky, sea = this.art.sea;
    if (hy > 0) {
      const rows = Math.min(hy, H);
      // The panorama is stored at half size: source px = pano px * kx.
      const kx = sky.width / this.panoW, ky = sky.height / this.panoH;
      const x0 = Math.round((view.yaw / (2 * Math.PI)) * this.panoW);
      const top = this.panoH - rows;
      if (top >= 0) this._wrapped(ctx, sky, x0, top * ky, W, rows * ky, 0, rows, this.panoW, kx);
      else {
        this._wrapped(ctx, sky, x0, 0, W, 1, 0, -top, this.panoW, kx); // the top row, stretched
        this._wrapped(ctx, sky, x0, 0, W, sky.height, -top, this.panoH, this.panoW, kx);
      }
    }
    if (hy < H) {
      const r0 = Math.max(0, hy);
      const seaW = sea.width * this.k;
      const x0 = Math.round((view.yaw / (2 * Math.PI)) * this.panoW);
      const s0 = r0 - hy, rows = H - r0; // source rows from the horizon
      const have = Math.max(0, Math.min(rows, sea.height - s0));
      if (have > 0) this._wrapped(ctx, sea, x0, s0, W, have, r0, have, seaW, sea.width / seaW);
      if (have < rows) this._wrapped(ctx, sea, x0, sea.height - 1, W, 1, r0 + have, rows - have, seaW, sea.width / seaW);
    }
  }

  drawGround(ctx, view, sc) {
    for (const ly of sc.ground) drawLayer(ctx, view, ly);
    const hy = Math.round(view.horizon);
    if (hy >= 0 && hy < this.height) ctx.drawImage(this.art.haze, 0, hy, this.width, this.art.haze.height * this.k);
  }

  drawTrees(ctx, view, game, sc) {
    const trees = game.hole.trees;
    if (!trees.length) return;
    const q = trees.map((t) => view.toCam([t.x, t.y, 0.0]));
    const order = trees.map((_, i) => i).sort((a, b) => q[b][2] - q[a][2]);
    for (const i of order) {
      const z = q[i][2];
      if (z < 2.0) continue;
      const t = trees[i];
      const sx = view.cx + (view.f * q[i][0]) / z;
      const sy = view.cy - (view.f * q[i][1]) / z;
      const tall = (view.f * t.height) / z;
      if (tall < 4 || sx < -tall || sx > this.width + tall || sy < view.horizon - 2) continue;
      const kind = this.art.trees[t.kind] ? t.kind : "palm";
      const spec = this.treeSpec[kind] ?? this.treeSpec.palm;
      const src = this.art.trees[kind];
      const nativeTall = Math.max(1.0, spec.foot[1] - spec.top);
      const hgt = Math.max(2, Math.round(src.height * (tall / nativeTall)));
      const sk = hgt / src.height;
      const rx = Math.max(2, Math.trunc((view.f * t.radius * 0.9) / z));
      const ry = Math.max(1, Math.trunc(((view.f * t.radius * 0.9) / z) * 0.28));
      ellipse(ctx, sx, sy, rx, ry, 0, sc.shadow);
      ctx.drawImage(src, Math.round(sx - spec.foot[0] * sk), Math.round(sy - spec.foot[1] * sk),
        Math.max(1, Math.round(src.width * sk)), hgt);
    }
  }

  drawPin(ctx, view, game, t) {
    const [px, py] = game.hole.pin;
    const base = view.point(px, py, 0.0);
    if (!base) return;
    const [sx, sy, z] = base;
    const tall = Math.max((PIN_MIN_PX * this.width) / this.width, (view.f * PIN_HEIGHT) / z);
    if (z < 30) {
      const r = (view.f * P.CUP_RADIUS) / z;
      ellipse(ctx, sx, sy, Math.max(1, r), Math.max(1, r * 0.35), 0, [30, 50, 30]);
    }
    const top = [sx, sy - tall];
    const w = Math.max(1.0, tall * 0.045);
    capsule(ctx, [sx, sy], top, w + 1, [50, 40, 40]);
    capsule(ctx, [sx, sy], top, w, [250, 250, 250]);
    // The flag streams with the wind, rippling.
    const windDir = game.windDir - view.yaw;
    const side = (game.windMph > 0.5 ? Math.sin(windDir) : 1.0) >= 0 ? 1.0 : -1.0;
    const fw = tall * 0.55, fh = tall * 0.32;
    const pts = [];
    const n = 8;
    for (let i = 0; i <= n; i++) {
      const u = i / n, wave = Math.sin(t * 7.0 + u * 5.0) * fh * 0.12 * u;
      pts.push([sx + side * fw * u, top[1] + fh * 0.5 * u + wave]);
    }
    for (let i = n; i >= 0; i--) {
      const u = i / n, wave = Math.sin(t * 7.0 + u * 5.0) * fh * 0.12 * u;
      pts.push([sx + side * fw * u, top[1] + fh - fh * 0.5 * u + wave]);
    }
    ctx.beginPath();
    tracePoly(ctx, pts);
    ctx.fillStyle = css([235, 70, 60]);
    ctx.fill();
    ctx.lineWidth = 1;
    ctx.strokeStyle = css([120, 30, 30]);
    ctx.stroke();
  }

  world(ctx, view, game) {
    const sc = this.scene(game);
    this.drawBackdrop(ctx, view);
    this.drawGround(ctx, view, sc);
    this.drawTrees(ctx, view, game, sc);
  }

  // --- frame ---

  draw(ctx, frame, game) {
    const W = this.width, H = this.height;
    const shake = game.fx.shake;
    let ox = 0.0, oy = 0.0;
    if (shake > 0) {
      ox = (Math.random() * 2 - 1) * 14.0 * shake;
      oy = (Math.random() * 2 - 1) * 14.0 * shake;
    }
    const cam = game.camera;
    const view = new View(cam, W, H, this.f, ox, oy);
    const phase = game.phase;
    const moving = phase === "intro" || phase === "flight" || shake > 0 || (phase === "holed" && game.phaseT < HOLED_PULL + 0.1);
    const key = [game.holeI, cam.x.toFixed(3), cam.y.toFixed(3), cam.z.toFixed(3), cam.yaw.toFixed(5), cam.pitch.toFixed(5)].join();
    ctx.save();
    if (!moving && key === this._cacheKey && this._cache) ctx.drawImage(this._cache, 0, 0);
    else if (moving) {
      this.world(ctx, view, game);
      this._cacheKey = null;
    } else {
      if (!this._cache) this._cache = this.makeCanvas(W, H);
      const c = this._cache.getContext("2d");
      this.world(c, view, game);
      this._cacheKey = key;
      ctx.drawImage(this._cache, 0, 0);
    }

    const sc = this.scene(game);
    this.drawPin(ctx, view, game, game.clock);
    if (phase === "aim") {
      if (game.putting || game.lie === "green") this._drawGreenGrid(ctx, view, sc, game.clock);
      this._drawTarget(ctx, view, game);
    }
    if (((phase === "aim" || phase === "flight" || phase === "result") && game.flight != null && game.flight.state === "air")
      || phase === "flight") this._drawTrail(ctx, view, game);
    this._drawParticles(ctx, view, game);
    if (phase === "aim" || phase === "flight" || phase === "result" || phase === "intro") this._drawLenny(ctx, view, game);
    else if (phase === "holed") this._drawCelebration(ctx, view, game);
    this._drawBall(ctx, view, game);
    if (game.fx.flash > 0) {
      ctx.fillStyle = `rgba(255,255,255,${Math.min(1.0, game.fx.flash) * 0.6})`;
      ctx.fillRect(0, 0, W, H);
    }
    this._drawHud(ctx, game, view);
    this._drawPip(ctx, frame, game);
    ctx.restore();
  }

  // --- overlays in the world ---

  _drawGreenGrid(ctx, view, sc, t) {
    ctx.strokeStyle = css([180, 236, 196]);
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (const ln of sc.gridLines) {
      const q = ln.map((p) => view.toCam(p)).filter((v) => v[2] >= NEAR);
      if (q.length < 2) continue;
      q.forEach((v, i) => {
        const [x, y] = view.project(v);
        if (i) ctx.lineTo(x, y);
        else ctx.moveTo(x, y);
      });
    }
    ctx.stroke();
    // Slope arrows: short chevrons drifting downhill, coloured by steepness
    // (white gentle, yellow medium, orange steep), like a green's slope grid.
    const bands = [[0.0, 0.012, [255, 250, 245]], [0.012, 0.025, [255, 230, 90]], [0.025, 9.0, [255, 140, 40]]];
    const paths = bands.map(() => []);
    for (const [x, y, gx, gy] of sc.flowPts) {
      const g = Math.hypot(gx, gy);
      if (!(g > 0.002)) continue;
      const dx = -gx / Math.max(g, 1e-9), dy = -gy / Math.max(g, 1e-9);
      const nx = -dy, ny = dx;
      const phase = (((t * (0.4 + Math.min(1, Math.max(0, g / 0.03)) * 0.8) + (x * 0.37 + y * 0.61)) % 1.0) + 1.0) % 1.0;
      const tipW = [x + dx * (phase * 0.9 - 0.45), y + dy * (phase * 0.9 - 0.45)];
      const back = [tipW[0] - dx * 0.32, tipW[1] - dy * 0.32];
      const qa = view.toCam([back[0] + nx * 0.16, back[1] + ny * 0.16, 0]);
      const qt = view.toCam([tipW[0], tipW[1], 0]);
      const qb = view.toCam([back[0] - nx * 0.16, back[1] - ny * 0.16, 0]);
      if (qa[2] < NEAR || qt[2] < NEAR || qb[2] < NEAR || !(qt[2] < 40)) continue;
      const band = bands.findIndex(([lo, hi]) => g >= lo && g < hi);
      if (band >= 0) paths[band].push([view.project(qa), view.project(qt), view.project(qb)]);
    }
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    bands.forEach(([, , colour], i) => {
      if (!paths[i].length) return;
      ctx.strokeStyle = css(colour);
      ctx.beginPath();
      for (const [a, b, c] of paths[i]) {
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(b[0], b[1]);
        ctx.lineTo(c[0], c[1]);
      }
      ctx.stroke();
    });
  }

  _drawTarget(ctx, view, game) {
    const pv = game.preview;
    if (!pv) return;
    if (game.putting) {
      // The line the putt starts on, to just past the hole.
      const [hx, hy] = game.ball;
      const L = Math.min(P.puttRangeM(), game.distanceToPin + 1.5);
      const n = Math.max(2, Math.trunc(L / 0.4));
      for (let i = 1; i <= n; i++) {
        const p = view.point(hx + (Math.sin(game.heading) * L * i) / n, hy + (Math.cos(game.heading) * L * i) / n, 0.0);
        if (p) disc(ctx, p[0], p[1], 3, WHITE);
      }
      return;
    }
    // Dotted arc of a full-power straight shot, then the landing ring.
    const path = pv.path;
    if (path.length > 2) {
      let i = 0;
      for (let j = 1; j < path.length; j += 2) {
        const p = view.point(path[j][0], path[j][1], path[j][2]);
        if (!p) continue;
        if (!(p[1] > ARC_TOP)) continue;
        const r = i % 2 === 0 ? 4 : 3;
        disc(ctx, p[0], p[1], r + 2, [30, 30, 40]);
        disc(ctx, p[0], p[1], r, WHITE);
        i += 1;
      }
    }
    if (!pv.landing) return;
    const [lx, ly] = pv.landing;
    const ring = [];
    for (let k = 0; k < 28; k++) {
      const a = (2 * Math.PI * k) / 28;
      ring.push(view.toCam([lx + 5.0 * Math.cos(a), ly + 5.0 * Math.sin(a), 0.0]));
    }
    if (Math.min(...ring.map((v) => v[2])) < NEAR) return;
    const s = ring.map((v) => view.project(v));
    // Keep the ring readable when it is far away and flattened.
    const cx = s.reduce((a, p) => a + p[0], 0) / s.length, cy = s.reduce((a, p) => a + p[1], 0) / s.length;
    const ys = s.map((p) => p[1]);
    const hgt = Math.max(...ys) - Math.min(...ys);
    if (hgt < 10) for (const p of s) p[1] = cy + (p[1] - cy) * (10 / Math.max(hgt, 1e-3));
    ctx.lineJoin = "round";
    for (const [w, colour] of [[6, [40, 30, 30]], [3, GOLD]]) {
      ctx.beginPath();
      tracePoly(ctx, s);
      ctx.lineWidth = w;
      ctx.strokeStyle = css(colour);
      ctx.stroke();
    }
    // A marker hovering over it, bobbing.
    const bob = 4 * Math.sin(game.clock * 4);
    const top = Math.min(...s.map((p) => p[1])) - 14 + bob;
    triangle(ctx, cx, top - 8, 11, "down", GOLD);
  }

  _drawTrail(ctx, view, game) {
    const f = game.flight;
    if (!f || f.path.length < 2) return;
    let pts = f.path.slice(-10);
    if (f.state !== "air") pts = pts.filter((p) => p[2] > 0.05);
    const s = pts.map((p) => view.point(p[0], p[1], p[2])).filter(Boolean);
    if (s.length < 2) return;
    ctx.beginPath();
    s.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.strokeStyle = "#fff";
    ctx.stroke();
  }

  _drawParticles(ctx, view, game) {
    for (const p of game.fx.particles) {
      const pt = view.point(p.x, p.y, p.z);
      if (!pt) continue;
      const r = Math.max(1.5, (view.f * (p.kind === "water" ? 0.09 : 0.07)) / pt[2]);
      disc(ctx, pt[0], pt[1], r, p.kind === "water" ? [220, 240, 255] : [240, 214, 170]);
    }
  }

  _ballState(game) {
    const f = game.flight;
    if (game.phase === "flight" && f) return [f.x, f.y, f.z];
    if (game.phase === "holed") return null;
    return [game.ball[0], game.ball[1], 0.0];
  }

  _drawBall(ctx, view, game) {
    const b = this._ballState(game);
    if (!b) return;
    const [x, y, z] = b;
    if (game.flight && (game.flight.state === "water" || game.flight.state === "ob") && game.phase !== "aim") return;
    const ground = view.point(x, y, 0.0);
    const pt = view.point(x, y, z);
    if (!pt) return;
    const [sx, sy, d] = pt;
    const r = Math.max(2.5, (view.f * Number(this.m.ball_visual_radius_m ?? 0.075)) / d);
    if (ground) {
      const sr = r * (1.0 + Math.min(1.5, z * 0.1));
      ellipse(ctx, ground[0], ground[1], sr, Math.max(1, sr * 0.35), 0, [40, 70, 40]);
    }
    const size = Math.max(2, Math.round((2 * r * 48) / 42));
    const img = this.art.ball;
    const w = Math.max(1, Math.round(img.width * (size / img.height)));
    ctx.drawImage(img, Math.round(sx - w / 2), Math.round(sy - size / 2), w, size);
  }

  // --- Lenny ---

  lennyScale(depth) {
    return Math.min(LENNY_MAX, (LENNY_WORLD * this.f) / Math.max(depth, 0.1));
  }

  /** Lenny's body sprite at scale s with its top-left at (ox, oy); returns k, the sprite's real scale. */
  _lennyBody(ctx, s, place) {
    const img = this.art.lenny;
    const hgt = Math.max(2, Math.round(img.height * s));
    const k = hgt / img.height;
    const [ox, oy] = place(k);
    ctx.drawImage(img, Math.round(ox), Math.round(oy), Math.max(1, Math.round(img.width * k)), hgt);
    return [k, ox, oy];
  }

  /** Lenny stands left of the ball where the shot was played from; arms and club at game.lennyAngle. */
  _drawLenny(ctx, view, game) {
    if (game.phase === "intro" && game.phaseT < 1.6) return;
    const spot = game.flight && game.phase !== "aim" ? game.flight.start : game.ball;
    const pt = view.point(spot[0], spot[1], 0.0);
    if (!pt) return;
    const [bx, by, depth] = pt;
    if (depth < 2.0) return;
    const s = this.lennyScale(depth);
    if (s < 0.08) return;
    const spec = this.lennySpec;
    const putt = game.phase === "aim" ? game.putting : !!(game.flight && game.flight.shot.club.putter);
    const arm = spec.arm * s, club = spec.club * s * (putt ? 0.62 : 1.0);
    const a0 = P.radians(spec.address_dir - (putt ? 12.0 : 0.0));
    const ball = [bx - 9.0 * s, by - 2.0 * s]; // the club head rests just behind the ball
    const pivot = [ball[0] - (arm + club) * Math.sin(a0), ball[1] - (arm + club) * Math.cos(a0)];
    const sb = spec.shoulder_back, sf = spec.shoulder_front;
    const mid = [(sb[0] + sf[0]) / 2, (sb[1] + sf[1]) / 2];
    const [k, ox, oy] = this._lennyBody(ctx, s, (kk) => [pivot[0] - mid[0] * kk, pivot[1] - mid[1] * kk]);

    const ang = game.lennyAngle;
    let theta = ang * (ang >= 0 ? THETA_TOP : THETA_FINISH);
    if (putt) theta = ang * 0.9;
    const a = a0 - theta; // arm direction, from straight down, + toward the ball
    const grip = [pivot[0] + arm * Math.sin(a), pivot[1] + arm * Math.cos(a)];
    const hinge = theta >= 0 ? Math.min(theta, 1.6) * 0.85 : Math.max(theta, -1.4) * 0.6;
    const c = putt ? a : a - hinge;
    const head = [grip[0] + club * Math.sin(c), grip[1] + club * Math.cos(c)];
    const big = game.phase === "aim" && (game.club.key === "1W" || game.club.key === "3W") ? 1.35 : 1.0;
    this._drawArms(ctx, [ox, oy], k, s, grip, head, c, putt, big);
  }

  /** Lenny's club, both arms from his shoulders to the grip, and the glove. */
  _drawArms(ctx, [ox, oy], k, s, grip, head, c, putt, big) {
    const spec = this.lennySpec;
    const col = spec.colours, ink = col.ink;
    const sb = spec.shoulder_back, sf = spec.shoulder_front;
    // Club first (behind the hands), then both arms to the grip, then the glove.
    capsule(ctx, grip, head, Math.max(1.5, 3.2 * s), ink);
    capsule(ctx, grip, head, Math.max(1.0, 1.8 * s), col.shaft);
    const hd = [Math.sin(c), Math.cos(c)];
    if (putt) {
      const n = [hd[1], -hd[0]];
      const p0 = [head[0] - n[0] * 9 * s, head[1] - n[1] * 9 * s], p1 = [head[0] + n[0] * 9 * s, head[1] + n[1] * 9 * s];
      capsule(ctx, p0, p1, Math.max(1.5, 5 * s), ink);
      capsule(ctx, p0, p1, Math.max(1.0, 3.4 * s), col.head);
    } else {
      const rot = Math.atan2(hd[1], hd[0]) + Math.PI / 2;
      ellipse(ctx, head[0], head[1], Math.max(1, 11 * s * big), Math.max(1, 6 * s * big), rot, ink);
      ellipse(ctx, head[0], head[1], Math.max(1, 9 * s * big), Math.max(1, 4.4 * s * big), rot, col.head);
    }
    const shoulders = [sb, sf].map((sh) => [ox + sh[0] * k, oy + sh[1] * k]);
    for (const p of shoulders) capsule(ctx, p, grip, 10.5 * s, ink);
    for (const p of shoulders) {
      const midArm = [p[0] + (grip[0] - p[0]) * 0.42, p[1] + (grip[1] - p[1]) * 0.42];
      capsule(ctx, p, grip, 8.0 * s, col.fur);
      capsule(ctx, p, midArm, 9.0 * s, col.sleeve);
    }
    disc(ctx, grip[0], grip[1], 10 * s + 2, ink);
    disc(ctx, grip[0], grip[1], 10 * s, col.glove);
  }

  /** After holing out: Lenny by the green, hopping, club raised over his head. */
  _drawCelebration(ctx, view, game) {
    const spot = game.celebrateSpot;
    if (!spot) return;
    const pt = view.point(spot[0], spot[1], 0.0);
    if (!pt || pt[2] < 2.0) return;
    const [fx, fy, depth] = pt;
    const s = this.lennyScale(depth);
    if (s < 0.08) return;
    const t = game.phaseT;
    const hop = Math.abs(Math.sin(t * 7.0)) * 26.0 * s * Math.max(0.0, 1.0 - t / 2.4); // three hops, dying away
    const k0 = Math.max(2, Math.round(this.art.lenny.height * s)) / this.art.lenny.height;
    // A shadow that shrinks as he leaves the ground.
    const rx = Math.max(2, Math.trunc(48 * k0 * (1.0 - Math.min(0.4, hop / Math.max(1.0, 80 * k0)))));
    ellipse(ctx, fx, fy, rx, Math.max(1, Math.trunc(rx * 0.26)), 0, [46, 90, 40]);
    const [k, ox, oy] = this._lennyBody(ctx, s, (kk) => [fx - LENNY_FOOT[0] * kk, fy - LENNY_FOOT[1] * kk - hop]);
    const spec = this.lennySpec;
    const sb = spec.shoulder_back, sf = spec.shoulder_front;
    const mid = [ox + ((sb[0] + sf[0]) / 2) * k, oy + ((sb[1] + sf[1]) / 2) * k];
    const pump = Math.sin(t * 7.0) * 0.12; // the arms pump with the hops
    // Arms straight, up and out in front of his face; the club held high, nearly upright.
    const a = Math.PI - 0.85 + pump;
    const arm = spec.arm * s * 1.3, club = spec.club * s;
    const grip = [mid[0] + arm * Math.sin(a), mid[1] + arm * Math.cos(a)];
    const c = Math.PI - 0.1 - pump * 1.5;
    const head = [grip[0] + club * Math.sin(c), grip[1] + club * Math.cos(c)];
    this._drawArms(ctx, [ox, oy], k, s, grip, head, c, false, 1.0);
  }

  // --- HUD ---

  _drawHud(ctx, game, view) {
    const phase = game.phase;
    if (phase === "intro") {
      this._drawIntro(ctx, game);
      this._drawMap(ctx, game);
      return;
    }
    if (phase !== "game_over") {
      this._drawHolePanel(ctx, game);
      this._drawWind(ctx, game, view);
      this._drawMap(ctx, game);
    }
    if (phase === "aim") {
      this._drawClubPanel(ctx, game);
      this._drawGauge(ctx, game);
      this._drawStatus(ctx, game);
      this._drawButtons(ctx, game);
    } else if (phase === "flight" || phase === "result") this._drawGauge(ctx, game);
    this._drawMessage(ctx, game);
    this._drawPopups(ctx, game);
    if ((phase === "holed" && game.phaseT >= 2.2) || phase === "game_over") this._drawScorecard(ctx, game, phase === "game_over");
  }

  _drawHolePanel(ctx, game) {
    const [x0, y0] = HOLE_PANEL;
    panel(ctx, HOLE_PANEL, { alpha: 0.72 });
    const h = game.hole;
    label(ctx, `HOLE ${h.number}`, x0 + 16, y0 + 40, 1.05, GOLD);
    label(ctx, `PAR ${h.par}`, x0 + 200, y0 + 40, 0.85, WHITE);
    const dist = game.phase === "holed" ? "IN THE HOLE" : `${game.distanceText()} to pin`;
    label(ctx, dist, x0 + 16, y0 + 76, 0.75, WHITE, true);
    const shot = game.strokes + (game.phase === "aim" ? 1 : 0);
    const bump = game.fx.scoreBump;
    const col = GOLD.map((c, i) => Math.trunc(c + (WHITE[i] - c) * (1 - bump)));
    label(ctx, `SHOT ${Math.max(1, shot)}`, x0 + 16, y0 + 108, 0.66, DIM, true);
    label(ctx, `TOTAL ${toParText(game.toPar)}`, x0 + 150, y0 + 108, 0.72, col);
  }

  _drawClubPanel(ctx, game) {
    const [x0, y0] = CLUB_PANEL;
    panel(ctx, CLUB_PANEL, { alpha: 0.72 });
    label(ctx, game.club.key, x0 + 16, y0 + 46, 1.15, GOLD);
    label(ctx, game.club.name, x0 + 96, y0 + 30, 0.62, WHITE, true);
    const lie = { tee: "TEE", fairway: "FAIRWAY", rough: "ROUGH 80%", bunker: "BUNKER 55%", green: "GREEN" }[game.lie]
      ?? game.lie.toUpperCase();
    text(ctx, `${game.clubRangeText()}   ${lie}`, x0 + 96, y0 + 58, 0.55, DIM, { body: true, outlineWidth: 3 });
  }

  _drawButtons(ctx, game) {
    const active = game.buttonsActive;
    for (const [name, rect] of game.buttons()) {
      const [x0, y0, x1, y1] = rect;
      const hot = active && game.hover === name;
      const aim = name.startsWith("aim");
      const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
      if (aim && !active) {
        // Just a hint of them while the hands are gripping.
        triangle(ctx, cx, cy, 14, name === "aim_left" ? "left" : "right", [210, 200, 200]);
        continue;
      }
      panel(ctx, rect, { colour: hot ? [34, 38, 46] : PANEL, alpha: active ? 0.8 : 0.45,
        border: hot ? GOLD : [120, 110, 110], thickness: hot ? 3 : 2, radius: 14 });
      const arrow = { club_up: "up", club_down: "down", aim_left: "left", aim_right: "right" }[name];
      const word = aim ? "AIM" : "CLUB";
      const colour = hot ? GOLD : active ? WHITE : [160, 150, 150];
      if (aim) {
        triangle(ctx, arrow === "left" ? x0 + 26 : x1 - 26, cy, 13, arrow, colour);
        centredText(ctx, word, cx + (arrow === "left" ? 12 : -12), Math.trunc(cy + 10), 0.75, colour);
      } else {
        triangle(ctx, x0 + 28, cy, 12, arrow, colour);
        centredText(ctx, word, cx + 14, Math.trunc(cy + 10), 0.72, colour);
      }
      if (hot && game.hoverT > 0) {
        const fill = Math.min(1.0, game.hoverT / 0.6);
        ctx.fillStyle = css(GOLD);
        ctx.fillRect(x0 + 10, y1 - 9, Math.trunc((x1 - x0 - 20) * fill), 4);
      }
    }
    if (active && game.cursor) {
      const [x, y] = game.cursor;
      ctx.lineWidth = 5;
      ctx.strokeStyle = css(DARK);
      ctx.beginPath();
      ctx.arc(x, y, 20, 0, Math.PI * 2);
      ctx.stroke();
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#fff";
      ctx.stroke();
      disc(ctx, x, y, 4, WHITE);
    }
  }

  _drawGauge(ctx, game) {
    const [gx, gy] = GAUGE_AT;
    const [gw, gh] = this.gaugeSize;
    ctx.drawImage(this.art.gauge, gx, gy, gw, gh);
    const d = game.detector;
    let power, practice;
    if (game.phase === "aim") {
      power = d.swinging || d.locked ? d.power : 0.0;
      practice = d.swinging && !d.armed;
    } else {
      power = game.lastShot?.power ?? 0.0;
      practice = false;
    }
    const [, y0, , y1] = this.gaugeInner;
    const topFrac = Math.min(1.15, Math.max(0.0, power)) / 1.15;
    if (topFrac > 0) {
      const yy = Math.round(y1 - (y1 - y0) * topFrac);
      const img = this.art.gaugeFill, ks = img.height / gh;
      ctx.drawImage(img, 0, yy * ks, img.width, (gh - yy) * ks, gx, gy + yy, gw, gh - yy);
    }
    const word = game.putting && game.phase === "aim" ? "PUTT" : "POWER";
    centredText(ctx, word, gx + gw / 2, gy - 6, 0.5, WHITE, { outlineWidth: 3 });
    if (power > 0) {
      const colour = power > 1.0 ? DANGER : GOLD;
      centredText(ctx, `${Math.round(power * 100)}%`, gx + gw + 40, Math.trunc(gy + y1 - (y1 - y0) * topFrac + 10), 0.8, colour);
    }
    if (practice) text(ctx, "PRACTICE", gx + 4, gy + gh + 26, 0.55, DIM, { body: true, outlineWidth: 3 });
  }

  _drawStatus(ctx, game) {
    const d = game.detector;
    const cx = this.width / 2, y = 150;
    const msg = d.guidance();
    if (d.hint) {
      centredText(ctx, d.hint, cx, y, 1.1, WARN);
      return;
    }
    if (d.stage === "ready") {
      centredText(ctx, "READY", cx, y, 1.2, CYAN);
      centredText(ctx, game.putting ? "rock your hands back, then through" : "swing back, then through", cx, y + 34, 0.7, WHITE, { body: true });
    } else if (d.swinging && !d.armed) {
      centredText(ctx, "PRACTICE SWING", cx, y, 1.0, WHITE);
      centredText(ctx, "hold your hands still at your waist first", cx, y + 32, 0.62, DIM, { body: true, outlineWidth: 3 });
    } else if (d.swinging) {
      // nothing: the gauge and Lenny say it all
    } else if (d.stage === "addressing") {
      centredText(ctx, "HOLD STILL", cx, y, 1.0, WHITE);
      const p = d.holdProgress;
      if (p > 0) {
        ctx.lineWidth = 4;
        ctx.strokeStyle = css(GOLD);
        ctx.beginPath();
        ctx.arc(cx, y + 36, 18, -Math.PI / 2, -Math.PI / 2 + 2 * Math.PI * p);
        ctx.stroke();
      }
    } else if (msg) {
      const colour = msg.startsWith("STEP") || msg.startsWith("SHOW") ? WARN : WHITE;
      centredText(ctx, msg, cx, y, 0.95, colour);
      if (game.buttonsActive) centredText(ctx, "or point at AIM / CLUB to adjust", cx, y + 32, 0.6, DIM, { body: true, outlineWidth: 3 });
    }
  }

  _drawWind(ctx, game, view) {
    panel(ctx, WIND_PANEL, { alpha: 0.72 });
    const [x0, y0, x1, y1] = WIND_PANEL;
    const cx = (x0 + x1) / 2, cy = y0 + 38;
    const mph = game.windMph;
    if (mph < 0.5) centredText(ctx, "CALM", cx, Math.trunc(cy + 10), 0.8, WHITE);
    else {
      // Arrow in the camera's frame: up the screen = away from the camera.
      const a = game.windDir - view.yaw;
      const dx = Math.sin(a), dy = -Math.cos(a) * 0.75; // a little perspective
      const L = 26;
      const tail = [cx - dx * L, cy - dy * L], tip = [cx + dx * L, cy + dy * L];
      const colour = mph < 6 ? CYAN : mph < 10 ? GOLD : DANGER;
      capsule(ctx, tail, tip, 5, INK);
      capsule(ctx, tail, tip, 3, colour);
      const nx = -dy, ny = dx;
      ctx.beginPath();
      tracePoly(ctx, [tip, [tip[0] - dx * 14 + nx * 10, tip[1] - dy * 14 + ny * 10],
        [tip[0] - dx * 14 - nx * 10, tip[1] - dy * 14 - ny * 10]]);
      ctx.fillStyle = css(colour);
      ctx.fill();
    }
    centredText(ctx, mph >= 0.5 ? `${mph.toFixed(0)} mph` : "no wind", cx, y1 - 12, 0.6, WHITE, { body: true, outlineWidth: 3 });
  }

  _drawMap(ctx, game) {
    const { canvas, toMap } = this.holeMap(game);
    const [x0, y0, x1, y1] = MAP;
    ctx.save();
    roundRectPath(ctx, x0, y0, x1, y1, 12);
    ctx.clip();
    ctx.drawImage(canvas, x0, y0);
    ctx.translate(x0, y0);
    // Aim line and landing target, the pin, the ball.
    const [bx, by] = game.phase !== "flight" || !game.flight ? toMap(game.ball[0], game.ball[1]) : toMap(game.flight.x, game.flight.y);
    ctx.lineWidth = 1;
    if (game.phase === "aim" && game.preview) {
      const tgt = game.putting ? game.hole.pin : game.preview.landing;
      if (tgt) {
        const [tx, ty] = toMap(tgt[0], tgt[1]);
        ctx.strokeStyle = "#fff";
        ctx.beginPath();
        ctx.moveTo(bx, by);
        ctx.lineTo(tx, ty);
        ctx.stroke();
        ctx.lineWidth = 2;
        ctx.strokeStyle = css(GOLD);
        ctx.beginPath();
        ctx.arc(tx, ty, 5, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
    if (game.flight && (game.phase === "flight" || game.phase === "result")) {
      const pts = game.flight.path.filter((_, i) => i % 3 === 0).map((p) => toMap(p[0], p[1]));
      if (pts.length > 1) {
        ctx.lineWidth = 1;
        ctx.strokeStyle = "#fff";
        ctx.beginPath();
        pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
        ctx.stroke();
      }
    }
    const [px, py] = toMap(game.hole.pin[0], game.hole.pin[1]);
    capsule(ctx, [px, py], [px, py - 12], 1, WHITE);
    ctx.beginPath();
    tracePoly(ctx, [[px, py - 12], [px + 9, py - 9], [px, py - 6]]);
    ctx.fillStyle = css([235, 70, 60]);
    ctx.fill();
    disc(ctx, bx, by, 5, DARK);
    disc(ctx, bx, by, 3, WHITE);
    ctx.restore();
    border(ctx, [x0 - 1, y0 - 1, x1, y1], 12, DARK, 5);
    border(ctx, [x0 - 1, y0 - 1, x1, y1], 12, [235, 235, 235], 2);
  }

  _drawMessage(ctx, game) {
    const m = game.message;
    if (!m) return;
    const age = m.life;
    const alpha = age < m.duration - 0.35 ? 1.0 : Math.max(0.0, (m.duration - age) / 0.35);
    const y = Math.trunc(this.height * 0.40);
    faded(ctx, m.text, this.width / 2, y, m.size * pop(age), m.colour, alpha);
    if (m.sub) faded(ctx, m.sub, this.width / 2, y + 50, 0.75, WHITE, alpha, true);
  }

  _drawPopups(ctx, game) {
    for (const p of game.fx.popups) {
      const age = p.life / p.duration;
      const y = p.pos[1] - 60 * easeOut(age);
      const alpha = age < 0.6 ? 1.0 : Math.max(0.0, (1 - age) / 0.4);
      faded(ctx, p.text, p.pos[0], y, 1.6 * pop(p.life), p.colour, alpha);
    }
  }

  _drawIntro(ctx, game) {
    const t = game.phaseT, h = game.hole;
    const a = Math.min(1.0, t / 0.4) * (t < 2.5 ? 1.0 : Math.max(0.0, (3.0 - t) / 0.5));
    const cx = this.width / 2;
    if (a <= 0) return;
    faded(ctx, `HOLE ${h.number}`, cx, 230, 2.6 * pop(t), GOLD, a);
    faded(ctx, h.name.toUpperCase(), cx, 285, 1.0, WHITE, a);
    faded(ctx, `PAR ${h.par}   -   ${(h.length * YARDS_PER_M).toFixed(0)} yd`, cx, 335, 1.1, WHITE, a);
    faded(ctx, game.windMph >= 0.5 ? `WIND ${game.windMph.toFixed(0)} mph` : "NO WIND", cx, 375, 0.75, CYAN, a, true);
  }

  _drawScorecard(ctx, game, final) {
    const W = this.width;
    const x0 = W / 2 - 380, x1 = W / 2 + 380;
    const [y0, y1] = final ? [110, 540] : [150, 470];
    panel(ctx, [x0, y0, x1, y1], { alpha: 0.9, border: GOLD });
    centredText(ctx, final ? "ROUND COMPLETE" : "SCORECARD", W / 2, y0 + 52, 1.2, GOLD);
    const cw = 120, labelW = 190, gx = x0 + 40, gy = y0 + 104;
    const card = game.card;
    const heads = ["HOLE", ...card.map((_, i) => String(i + 1)), "TOTAL"];
    const parTotal = card.reduce((s, r) => s + r.par, 0);
    const rows = [
      ["PAR", card.map((r) => String(r.par)), String(parTotal)],
      ["STROKES", card.map((r) => (r.done ? String(r.strokes) : "-")), String(card.filter((r) => r.done).reduce((s, r) => s + r.strokes, 0))],
      ["SCORE", card.map((r) => diffText(r)), toParText(game.toPar)],
      ["POINTS", card.map((r) => (r.done ? String(r.points) : "-")), String(game.score)],
    ];
    heads.forEach((hd, j) => {
      const x = gx + (j === 0 ? 0 : labelW + (j - 1) * cw + cw / 2);
      if (j === 0) label(ctx, hd, x, gy, 0.7, DIM, true);
      else centredText(ctx, hd, x, gy, 0.75, DIM);
    });
    ctx.strokeStyle = css([120, 110, 110]);
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(gx, gy + 14);
    ctx.lineTo(x1 - 40, gy + 14);
    ctx.stroke();
    rows.forEach(([name, vals, total], i) => {
      const y = gy + 56 + i * 48;
      label(ctx, name, gx, y, 0.75, WHITE, true);
      [...vals, total].forEach((v, j) => {
        const x = gx + labelW + j * cw + cw / 2;
        const current = j === game.holeI && !final;
        let colour = j === vals.length ? GOLD : current ? CYAN : WHITE;
        if (name === "SCORE" && v !== "-" && v.startsWith("-")) colour = j < vals.length ? CYAN : GOLD;
        centredText(ctx, v, x, y, 0.95, colour);
      });
    });
    if (final) {
      const tp = game.toPar;
      const verdict = tp < 0 ? "UNDER PAR - SUPERB!" : tp === 0 ? "LEVEL PAR - NICE ROUND!" : tp <= 4 ? "WELL PLAYED!" : "KEEP SWINGING!";
      centredText(ctx, verdict, W / 2, y1 - 28, 0.9, CYAN);
    }
  }

  // --- the camera window ---

  _drawPip(ctx, frame, game) {
    const [x0, y0, x1, y1] = PIP;
    const pw = x1 - x0, ph = y1 - y0;
    const s = pw / this.width;
    const d = game.detector;
    ctx.save();
    roundRectPath(ctx, x0, y0, x1, y1, 14);
    ctx.clip();
    if (frame && frame.width > 0) ctx.drawImage(frame, x0, y0, pw, ph);
    else {
      ctx.fillStyle = "#222";
      ctx.fillRect(x0, y0, pw, ph);
    }
    ctx.translate(x0, y0);
    const pose = game.lastPose;
    if (pose) {
      const pts = {};
      for (const n of ["left_shoulder", "right_shoulder", "left_wrist", "right_wrist", "left_elbow", "right_elbow"]) {
        const i = KP[n];
        if (pose.confidence[i] >= 0.4) pts[n] = [Math.trunc(pose.x(i) * s), Math.trunc(pose.y(i) * s)];
      }
      ctx.lineCap = "round";
      for (const [a, b] of [["left_shoulder", "right_shoulder"], ["left_shoulder", "left_elbow"], ["left_elbow", "left_wrist"],
        ["right_shoulder", "right_elbow"], ["right_elbow", "right_wrist"]]) {
        if (!pts[a] || !pts[b]) continue;
        capsule(ctx, pts[a], pts[b], 2, DARK);
        capsule(ctx, pts[a], pts[b], 1, [230, 230, 230]);
      }
    }
    const g = d.grip;
    if (g) {
      const colour = d.ready ? CYAN : d.swinging ? GOLD : d.together ? WHITE : WARN;
      disc(ctx, Math.trunc(g[0] * s), Math.trunc(g[1] * s), 9, DARK);
      disc(ctx, Math.trunc(g[0] * s), Math.trunc(g[1] * s), 7, colour);
    }
    if (d.address && d.bodyFrame && game.phase === "aim") {
      const [fx, fy, sw] = d.bodyFrame;
      const [au, ah] = d.address;
      const ax = (fx + au * sw) * s, ay = (fy - ah * sw) * s;
      ctx.strokeStyle = css(CYAN);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(ax, ay - 30);
      ctx.lineTo(ax, ay + 20);
      ctx.stroke();
    }
    if (!d.visible && game.phase === "aim") {
      const cy = Math.trunc(ph / 2);
      ctx.fillStyle = css(DARK);
      ctx.fillRect(8, cy - 22, pw - 16, 36);
      centredText(ctx, "STEP INTO VIEW", pw / 2, cy + 6, 0.7, WARN);
    }
    ctx.restore();
    const colour = d.ready ? CYAN : d.swinging ? GOLD : [235, 235, 235];
    border(ctx, [x0, y0, x1 - 1, y1 - 1], 14, DARK, 6);
    border(ctx, [x0, y0, x1 - 1, y1 - 1], 14, colour, 3);
  }
}

function diffText(r) {
  if (!r.done) return "-";
  if (r.pickedUp) return "PU";
  return toParText(r.strokes - r.par);
}
