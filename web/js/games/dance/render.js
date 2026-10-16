// Drawing Raccoon Dance: the neon stage, the coach, the pictogram lane, feedback,
// HUD (port of src/kpapp/game/dance/render.py, in Canvas 2D).
//
// The camera is not the backdrop here. The player copies a coach, so the coach
// and the cards own the screen, and the camera moves to a picture-in-picture
// window (top-right) where it does one job: show the player what the judge
// sees, a skeleton tinted by the last rating.
//
// Layout (docs/DANCE.md section 10): the coach stands centre-left in a
// spotlight (x ~520, feet at y 660, ~520 px tall); the pictogram lane runs
// along the bottom right onto a target ring at x 806; the PiP is top-right; the
// score is top-left above a vertical star bar; the bottom-left footer and the
// PAUSE button (x 1090+, y 646+) stay clear.
//
// Budget: < 4 ms a frame. As in the Python, every light that pulses (beams, the
// floor pool, neon trims, the coach's halo, the gold edge glow) is summed into
// one quarter-resolution buffer and added to the frame once; sprites are tinted
// and sized once and cached; the coach is a few dozen filled paths.
//
// Colours: the Python's BGR literals are written here as bgr([b, g, r]) so they
// can be checked against render.py; art.js already converted the manifest's.
//
// The renderer reads the game's state (docs/DANCE.md section 8, camelCase) and
// never mutates it. Nothing touches the DOM at import time; canvases come from
// an injectable factory so Node tests can drive it with a fake context.

import { makeCanvas as defaultMakeCanvas } from "../../core/gfx.js";
import { ACCENT, CYAN, DIM, GOOD, WARN, css, centredText, font, fontPx, panel, text, textWidth } from "../../core/theme.js";
import { bgr } from "./art.js";
import { coachMotion, stillMotion } from "./motion.js";
import { FOREARM, HIP_W, REST, SHIN, SIDE, THIGH, TORSO, UPPER_ARM, forward, legTarget, lerpPose, limbDir, normalizePose,
  smoothstep } from "./pose.js";

// --- layout (1280 x 720) ----------------------------------------------------
export const COACH_X = 520, COACH_FEET = 660;
export const COACH_SCALE = 114.0; // px per shoulder width
export const COACH_ROI = [110, 56, 930, 704]; // the coach never draws outside this
export const POPUP_Y_MIN = 78;

export const PAUSE_RECT = [1090, 646, 1258, 698]; // the shell's PAUSE button: nothing of ours enters it
export const FOOTER_Y = 700; // the shell's footer (bottom-left) starts here

// The lane: cards ~20% bigger than at first (76 -> 108 px), re-laid out so two
// cards a beat apart still keep a gap, everything above PAUSE_RECT, and the
// label above the footer.
export const LANE_RECT = [740, 513, 1262, 631]; // the track (118 px tall)
export const TARGET_X = 812, LANE_Y = 572;
export const PX_PER_BEAT = 128.0;
export const CARD_SMALL = 92, CARD_BIG = 132; // card widths (multiples of 4: cardWidth's steps)
export const GROW_BEATS = 1.0; // a card grows from small to big over its last beat to the ring (cards come every beat)
export const RING = 146; // target ring diameter
export const LANE_CLIP_BOTTOM = PAUSE_RECT[1] - 1; // cards (and their sparkles) never draw below this
export const LABEL_Y = 670; // baseline of the move's name under the ring
export const LABEL_SCALE = 0.72;

export const PIP_RECT = [880, 16, 1264, 232]; // 384 x 216, 16:9
export const PIP_RADIUS = 16;
export const STAR_BAR = [30, 128, 50, 552]; // the fill track; star markers sit on it
export const SCORE_ORG = [78, 76];
export const HUD_BADGE_ORG = [262, 58]; // the difficulty badge's left edge and centre line, right of the score

export const INTRO_C = [958, 352]; // the title card and GET READY, centred under the PiP
export const COUNT_Y = 366; // the 3-2-1-GO words: their glow clears the PiP (y 232) once popped in
export const POPUP_X = [260, 660]; // rating words: centre x clamp (the widest word, gold, stays left of the PiP)

export const RESULTS_RECT = [700, 34, 1250, 528]; // leaves the bottom 25% to the shell

export const DIFFICULTY_LABELS = Object.freeze({ easy: "EASY", normal: "NORMAL", hard: "HARD" });

const WHITE = [255, 255, 255];
const INK = bgr([46, 24, 34]);

export const RATING_KEYS = ["perfect", "super", "good", "ok", "x"];
const RATING_FALLBACK = {
  perfect: bgr([70, 210, 255]), super: bgr([255, 220, 110]), good: bgr([140, 235, 150]),
  ok: bgr([255, 150, 200]), x: bgr([70, 70, 255]),
};
const SECTION_ENERGY = [["chorus", 1.0], ["verse", 0.72], ["bridge", 0.55], ["intro", 0.55], ["outro", 0.45]];
export const BOW_POSE = normalizePose({ la: [28, 18], ra: [-28, -70], head: 0, lean: 0, squat: 0.12 });

export const POPUP_LIFE = 0.85;
// An older popup clears this fast once a newer one appears. 0.12 s left both
// words readable for ~7 frames ("PERFECT" over "PERFECT"); at 0.04 s (squared
// fade: 34% on the first frame, 3% on the second, then gone) the old word has
// left before the new one has popped to full size.
export const SUPERSEDE_FADE = 0.04;

const RAD = Math.PI / 180;

// --- small helpers ----------------------------------------------------------

/** "perfect" | "super" | "good" | "ok" | "x" from a rating string (any case), else null. */
export function ratingKey(rating) {
  if (rating == null) return null;
  const r = String(rating.name ?? rating).toLowerCase().split(".").pop();
  return RATING_KEYS.includes(r) ? r : null;
}

/**
 * A lane card's width `d` beats before its beat: small until GROW_BEATS out,
 * full size from the ring on (and while it slides through), in steps of 4 px
 * so the card cache stays small.
 */
export function cardWidth(d) {
  const u = Math.min(1.0, Math.max(0.0, 1.0 - d / GROW_BEATS));
  return Math.round((CARD_SMALL + (CARD_BIG - CARD_SMALL) * smoothstep(u)) / 4.0) * 4;
}

/**
 * When the chart's own ending starts (song time): the first of its trailing
 * unjudged moves (Neon Nights ends finale -> "bow"). The renderer's bow waits
 * for it, so the last judged pose is held on its beat and there is one bow, not
 * two. -Infinity when the chart doesn't end on unjudged moves.
 */
export function endingStart(song) {
  const chart = song?.chart ?? [];
  let start = null;
  for (let i = chart.length - 1; i >= 0; i--) {
    if (chart[i].move?.judge?.length) break;
    start = chart[i].beat;
  }
  if (start == null || typeof song.timeOf !== "function") return -Infinity;
  return song.timeOf(start);
}

/** 1 on each beat, decaying through it: the pulse every light shares. */
export const kick = (beat, sharp = 5.0) => Math.exp(-(beat - Math.floor(beat)) * sharp);

export function easeOut(t) {
  t = Math.min(1.0, Math.max(0.0, t));
  return 1.0 - (1.0 - t) ** 3;
}

/** Scale-in with an overshoot: start -> over -> 1.0. */
export function pop(age, dur = 0.26, over = 1.28, start = 0.45) {
  if (age < dur * 0.45) return start + (over - start) * (age / (dur * 0.45));
  if (age < dur) return over - (over - 1.0) * ((age - dur * 0.45) / (dur * 0.55));
  return 1.0;
}

const clamp255 = (c, k = 1) => c.map((v) => Math.min(255, Math.max(0, Math.round(v * k))));

/** Draw `img` centred at (cx, cy), w x h, faded. */
function drawC(ctx, img, cx, cy, w = img.width, h = img.height, fade = 1.0) {
  if (fade <= 0.004 || !img) return;
  const a0 = ctx.globalAlpha;
  ctx.globalAlpha = a0 * Math.min(1, fade);
  ctx.drawImage(img, cx - w / 2, cy - h / 2, w, h);
  ctx.globalAlpha = a0;
}

/** Add `img` as light, centred at (cx, cy); gains above 1 draw it again. */
function addC(ctx, img, cx, cy, w = img.width, h = img.height, gain = 1.0) {
  if (gain <= 0.004 || !img) return;
  const a0 = ctx.globalAlpha, op = ctx.globalCompositeOperation;
  ctx.globalCompositeOperation = "lighter";
  for (let g = gain; g > 0.004; g -= 1) {
    ctx.globalAlpha = a0 * Math.min(1, g);
    ctx.drawImage(img, cx - w / 2, cy - h / 2, w, h);
  }
  ctx.globalAlpha = a0;
  ctx.globalCompositeOperation = op;
}

function path(ctx, pts) {
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
  ctx.closePath();
}

/** Fill convex polygons in one colour, as one path (all wound the same way, so it's a union). */
function fillPolys(ctx, polys, colour) {
  ctx.fillStyle = css(colour);
  ctx.beginPath();
  for (const p of polys) path(ctx, p);
  ctx.fill();
}

function strokeLine(ctx, pts, colour, width) {
  ctx.strokeStyle = css(colour);
  ctx.lineWidth = Math.max(1, width);
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
  ctx.stroke();
}

function disc(ctx, c, r, colour) {
  ctx.fillStyle = css(colour);
  ctx.beginPath();
  ctx.arc(c[0], c[1], Math.max(0.5, r), 0, 2 * Math.PI);
  ctx.fill();
}

/** cv2.ellipse: angles in degrees, clockwise on screen (as the canvas's). thick < 0 fills. */
function ellipse(ctx, c, axes, angleDeg, colour, thick = -1, a0 = 0, a1 = 360) {
  ctx.beginPath();
  ctx.ellipse(c[0], c[1], Math.max(0.5, axes[0]), Math.max(0.5, axes[1]), angleDeg * RAD, a0 * RAD, a1 * RAD);
  if (thick < 0) {
    ctx.fillStyle = css(colour);
    ctx.fill();
  } else {
    ctx.strokeStyle = css(colour);
    ctx.lineWidth = thick;
    ctx.stroke();
  }
}

/** Positive-area (screen clockwise) winding, so overlapping polygons fill as a union. */
function wound(pts) {
  let a = 0;
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i], q = pts[(i + 1) % pts.length];
    a += p[0] * q[1] - q[0] * p[1];
  }
  return a < 0 ? pts.reverse() : pts;
}

const ARC_K = 10;

/**
 * Tapered capsules as convex polygons: each the convex hull of its two end
 * circles. segs: [[x0, y0, x1, y1, r0, r1], ...] in px.
 */
export function capsules(segs, grow) {
  return segs.map(([x0, y0, x1, y1, ra, rb]) => {
    const r0 = ra + grow, r1 = rb + grow;
    const dx = x1 - x0, dy = y1 - y0;
    const length = Math.max(Math.hypot(dx, dy), 1e-6);
    const theta = Math.atan2(dy, dx);
    const alpha = Math.acos(Math.min(0.999, Math.max(-0.999, (r0 - r1) / length)));
    const pts = [];
    for (let k = 0; k < ARC_K; k++) {
      const a = theta - alpha + 2 * alpha * (k / (ARC_K - 1));
      pts.push([x1 + Math.cos(a) * r1, y1 + Math.sin(a) * r1]);
    }
    for (let k = 0; k < ARC_K; k++) {
      const a = theta + alpha + (2 * Math.PI - 2 * alpha) * (k / (ARC_K - 1));
      pts.push([x0 + Math.cos(a) * r0, y0 + Math.sin(a) * r0]);
    }
    return wound(pts);
  });
}

/** Convex hull of points (monotone chain). */
function convexHull(points) {
  const p = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (p.length < 3) return p;
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [], upper = [];
  for (const q of p) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], q) <= 0) lower.pop();
    lower.push(q);
  }
  for (let i = p.length - 1; i >= 0; i--) {
    const q = p[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], q) <= 0) upper.pop();
    upper.push(q);
  }
  return wound(lower.slice(0, -1).concat(upper.slice(0, -1)));
}

/** Convex hull of several circles. */
function hull(points, radii, grow) {
  const pts = [];
  points.forEach((c, i) => {
    const r = radii[i] + grow;
    for (let k = 0; k < 18; k++) {
      const t = (2 * Math.PI * k) / 18;
      pts.push([c[0] + Math.cos(t) * r, c[1] + Math.sin(t) * r]);
    }
  });
  return convexHull(pts);
}

function ellipsePoly(c, axes, angleDeg) {
  const a = angleDeg * RAD, ca = Math.cos(a), sa = Math.sin(a);
  const pts = [];
  for (let k = 0; k < 20; k++) {
    const t = (2 * Math.PI * k) / 20;
    const ex = Math.cos(t) * axes[0], ey = Math.sin(t) * axes[1];
    pts.push([c[0] + ex * ca - ey * sa, c[1] + ex * sa + ey * ca]);
  }
  return wound(pts);
}

const add = (p, v, k = 1) => [p[0] + v[0] * k, p[1] + v[1] * k];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1]];
const unit = (v) => {
  const n = Math.max(1e-6, Math.hypot(v[0], v[1]));
  return [v[0] / n, v[1] / n];
};

/** A sprite tinted to one colour (keeping its alpha), at w x h, on a fresh canvas. */
function tintedCanvas(mk, img, colour, w = img.width, h = img.height) {
  const c = mk(Math.max(1, Math.round(w)), Math.max(1, Math.round(h)));
  const g = c.getContext("2d");
  g.imageSmoothingQuality = "high";
  g.drawImage(img, 0, 0, c.width, c.height);
  if (colour) {
    g.globalCompositeOperation = "source-in";
    g.fillStyle = css(colour);
    g.fillRect(0, 0, c.width, c.height);
  }
  return c;
}

function roundRect(ctx, x0, y0, x1, y1, r) {
  r = Math.max(0, Math.min(r, (x1 - x0) / 2, (y1 - y0) / 2));
  ctx.moveTo(x0 + r, y0);
  ctx.arcTo(x1, y0, x1, y1, r);
  ctx.arcTo(x1, y1, x0, y1, r);
  ctx.arcTo(x0, y1, x0, y0, r);
  ctx.arcTo(x0, y0, x1, y0, r);
  ctx.closePath();
}

/**
 * A runtime word sprite in the generated words' style: fill (a light-to-colour
 * gradient), dark keyline, soft glow. For text only known at runtime (song
 * titles). Built once and cached by the caller.
 */
export function makeGlowText(mk, str, scale, colour, glow = null) {
  const px = fontPx(scale);
  const key = Math.max(3, px * 0.09), gw = Math.max(6, px * 0.26);
  const probe = mk(4, 4).getContext("2d");
  probe.font = font(scale);
  const tw = probe.measureText(str).width;
  const pad = Math.ceil(gw * 2 + key);
  const c = mk(Math.ceil(tw + 2 * pad), Math.ceil(px * 1.25 + 2 * pad));
  const g = c.getContext("2d");
  g.font = font(scale);
  g.textAlign = "center";
  g.textBaseline = "middle";
  g.lineJoin = "round";
  const x = c.width / 2, y = c.height / 2 + px * 0.04;
  g.save();
  g.shadowColor = css(glow ?? colour, 0.85);
  g.shadowBlur = gw;
  g.strokeStyle = css(glow ?? colour, 0.85);
  g.lineWidth = key * 2;
  g.strokeText(str, x, y);
  g.restore();
  g.strokeStyle = css(INK);
  g.lineWidth = key * 2;
  g.strokeText(str, x, y);
  const grad = g.createLinearGradient(0, y - px * 0.45, 0, y + px * 0.45);
  grad.addColorStop(0, css(colour.map((v) => v + (255 - v) * 0.6)));
  grad.addColorStop(1, css(colour));
  g.fillStyle = grad;
  g.fillText(str, x, y);
  return c;
}

const isPlainObject = (v) => v !== null && typeof v === "object" && !Array.isArray(v);

/**
 * A difficulty badge: the label in `colour` on a dark, slightly translucent
 * pill with a border of the same colour (port of render.make_badge).
 */
export function makeBadge(mk, label, colour, scale = 0.62) {
  const probe = mk(4, 4).getContext("2d");
  probe.font = font(scale);
  const tw = probe.measureText(label).width;
  const th = fontPx(scale) * 0.72; // cap height
  const px = Math.round(th * 0.85), py = Math.round(th * 0.5);
  const w = Math.ceil(tw + 2 * px), h = Math.ceil(th * 1.35 + 2 * py);
  const c = mk(w, h);
  const g = c.getContext("2d");
  const pill = (inset) => {
    g.beginPath();
    roundRect(g, inset, inset, w - inset, h - inset, h / 2 - inset);
  };
  pill(1.6);
  g.fillStyle = css(bgr([40, 18, 30]), 0.88);
  g.fill();
  pill(0.8);
  g.lineWidth = 1.6;
  g.strokeStyle = css(colour);
  g.stroke();
  g.font = font(scale);
  g.textAlign = "center";
  g.textBaseline = "middle";
  g.fillStyle = css(colour);
  g.fillText(label, w / 2, h / 2 + th * 0.04);
  return c;
}

// --- the coach ------------------------------------------------------------------

// Radii in shoulder widths: [start, end] of each segment.
const R_UPPER = [0.135, 0.112];
const R_FORE = [0.112, 0.092];
const R_THIGH = [0.19, 0.145];
const R_SHIN = [0.145, 0.112];
const HEAD_FACE_SW = 1.2; // the face is drawn this many shoulder widths wide: a mascot
const FOOT_BELOW_ANKLE = 0.245; // shoe sole (with keyline and rim) below the ankle, in SW
const REST_ANKLE_Y = forward(REST).l_ankle[1];

// The 3D look (render.py, docs/DANCE.md section 4a): a key light from the upper
// left. Each part is filled in its shadow colour, then in its own colour shrunk
// by SHADE_K of its radius and moved that far towards the light, leaving a
// core-shadow crescent on the lower right; limbs get a highlight streak.
const LIGHT = (() => {
  const n = Math.hypot(0.55, 0.83);
  return [-0.55 / n, -0.83 / n];
})();
const SHADE_K = 0.24;
const STREAK_OFF = 0.5, STREAK_R = 0.2, STREAK_TRIM = 0.18;
const SHADE_DARK = 0.6, SHADE_BOUNCE = 0.14, HIGHLIGHT = 0.34;
export const SHADOW_INK = bgr([20, 6, 14]);
const DEPTH_THICK = 0.12, FORESHORTEN = 0.08, GLOVE_NEAR = 0.16, BEHIND = -0.25;
const CHEST_TURN = 0.16, HEAD_TURN = 0.06, HULL_SHADE = 0.065;

/** The lit part of each capsule: radii shrunk by k, moved k of a radius towards the light. */
function litSegs(segs, k = SHADE_K) {
  return segs.map(([x0, y0, x1, y1, r0, r1]) => [x0 + LIGHT[0] * k * r0, y0 + LIGHT[1] * k * r0,
    x1 + LIGHT[0] * k * r1, y1 + LIGHT[1] * k * r1, r0 * (1 - k), r1 * (1 - k)]);
}

/** A highlight along each capsule's lit side, short of its ends. */
function streakSegs(segs) {
  return segs.map(([x0, y0, x1, y1, r0, r1]) => {
    const dx = x1 - x0, dy = y1 - y0;
    return [x0 + dx * STREAK_TRIM + LIGHT[0] * STREAK_OFF * r0, y0 + dy * STREAK_TRIM + LIGHT[1] * STREAK_OFF * r0,
      x1 - dx * STREAK_TRIM + LIGHT[0] * STREAK_OFF * r1, y1 - dy * STREAK_TRIM + LIGHT[1] * STREAK_OFF * r1,
      r0 * STREAK_R, r1 * STREAK_R];
  });
}

/** The lit part of a hull of circles: each shrunk by k px and moved k px towards the light. */
const litHull = (points, radii, k) => hull(points.map((p) => add(p, LIGHT, k)), radii.map((r) => r - k), 0.0);

/** [base, shadow, highlight] for one material (render.py _shades). */
function shades(base, glow) {
  const shadow = base.map((v, i) => Math.min(255, Math.max(0, Math.round(v * SHADE_DARK + glow[i] * SHADE_BOUNCE))));
  const hi = base.map((v) => Math.round(v + (255 - v) * HIGHLIGHT));
  return [base, shadow, hi];
}

/**
 * The coach: the arcade's purple raccoon as a bold, lit cartoon dancer.
 * layout(pose, {motion}) turns a pose and the motion layered on it (motion.js)
 * into pixel geometry (pure math); paint(ctx, geo) draws it back to front: a
 * contact shadow, a neon rim around the whole silhouette, tail, the far arm
 * when the body twists away from it, legs and pants, jacket, neck, head, and
 * arms in front, each with a dark keyline. With `shade` (the default) every
 * part is lit from the upper left; the tutorial's small coaches draw flat.
 */
export class CoachPainter {
  constructor(art, outfit = null, { scale = COACH_SCALE, origin = null, makeCanvas = defaultMakeCanvas, shade = true } = {}) {
    this.S = scale;
    this.c = art.outfit(outfit);
    this.shade = shade;
    this.inkW = Math.max(1.2, 0.026 * this.S);
    this.rimW = Math.max(1.2, 0.032 * this.S);
    this.origin = origin ?? this.originFor(COACH_X, COACH_FEET);
    const glow = this.c.glow ?? bgr([255, 80, 200]);
    this.mat = {};
    for (const k of ["fur", "fur_dark", "top", "pants", "glove", "shoe"]) this.mat[k] = shades(this.c[k] ?? [200, 100, 200], glow);
    const hm = art.manifest.head ?? {};
    this._face = hm.centre ?? [0.5, 0.565];
    this._earTop = 0.105;
    const fw = Number(hm.face_width ?? 0.67);
    const side = Math.max(8, Math.round(HEAD_FACE_SW * this.S / fw));
    this._headSide = side;
    // The head over its rim (tinted to the outfit's rim colour), one per head turn.
    let turns = shade ? (art.headTurns ?? []) : [];
    if (!turns.length) turns = [{ yaw: 0, head: art.head, rim: art.headRim }];
    this._yaws = turns.map((t) => t.yaw);
    this._heads = turns.map((t) => {
      const head = makeCanvas(side, side);
      const g = head.getContext("2d");
      g.imageSmoothingQuality = "high";
      g.drawImage(tintedCanvas(makeCanvas, t.rim, this.c.rim ?? WHITE, side, side), 0, 0);
      g.drawImage(t.head, 0, 0, side, side);
      return head;
    });
    this._head = this._heads[this._yaws.indexOf(0) >= 0 ? this._yaws.indexOf(0) : 0];
    // The contact shadow: one soft ellipse, drawn scaled.
    this._shadow = null;
    if (shade) {
      const sw = 128, sh = 48;
      const c = makeCanvas(sw, sh);
      const g = c.getContext("2d");
      g.save();
      g.translate(sw / 2, sh / 2);
      g.scale(1, sh / sw);
      const grad = g.createRadialGradient(0, 0, 0, 0, 0, sw / 2);
      grad.addColorStop(0, css(SHADOW_INK, 0.78));
      grad.addColorStop(0.45, css(SHADOW_INK, 0.72));
      grad.addColorStop(0.75, css(SHADOW_INK, 0.3));
      grad.addColorStop(1, css(SHADOW_INK, 0));
      g.fillStyle = grad;
      g.beginPath();
      g.arc(0, 0, sw / 2, 0, 2 * Math.PI);
      g.fill();
      g.restore();
      this._shadow = c;
    }
  }

  /** The rest hip centre that puts the coach's shoe soles at feetY. */
  originFor(x, feetY) {
    return [x, feetY - (FOOT_BELOW_ANKLE + REST_ANKLE_Y) * this.S];
  }

  // --- geometry ---

  /**
   * Pixel geometry for `pose`, or for `motion` (motion.js coachMotion). The feet
   * stay on the floor (the lower one is planted) unless the motion hops.
   */
  layout(pose, { bow = 0.0, beat = 0.0, origin = null, motion = null } = {}) {
    const m = motion ?? stillMotion(pose ?? REST);
    let p = normalizePose(m.pose);
    let calm = 1.0;
    if (bow > 0.0) {
      p = lerpPose(p, BOW_POSE, Math.min(1.0, bow));
      calm = 1.0 - Math.min(1.0, bow);
    }
    const jj = forward(p);
    const S = this.S, [ox, oy] = origin ?? this.origin;
    const lift = m.lift * calm;
    const dx = m.travel ?? 0.0; // steps travel; the bow stays where the dance left it
    const dy = REST_ANKLE_Y - Math.max(jj.l_ankle[1], jj.r_ankle[1]) - lift;
    const j = {};
    for (const [k, v] of Object.entries(jj)) j[k] = [ox + (v[0] + dx) * S, oy + (v[1] + dy) * S];
    if (bow > 0.0) {
      // A front-view bow: the upper body foreshortens towards the camera.
      const hipY = j.hip[1];
      const k = 1.0 - 0.24 * Math.min(1.0, bow);
      for (const name of ["neck", "head", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
        "l_hand", "r_hand"]) {
        j[name] = [j[name][0], hipY + (j[name][1] - hipY) * k];
      }
    }
    const up = unit(sub(j.neck, j.hip));
    const right = [-up[1], up[0]];
    const arm = { l: ["l_shoulder", "l_elbow", "l_wrist", "l_hand"], r: ["r_shoulder", "r_elbow", "r_wrist", "r_hand"] };
    const upper = ["neck", "head", ...arm.l, ...arm.r];
    const move = (names, d) => { for (const k of names) j[k] = add(j[k], d); };

    // Weight shift: the upper body over the feet (the knees go half way).
    const sway = m.sway * calm * S;
    move([...upper, "hip", "l_hip", "r_hip"], [sway, 0]);
    move(["l_knee", "r_knee"], [0.5 * sway, 0]);
    // Squash and stretch along the torso's axis.
    const q = 1.0 + (m.squash - 1.0) * calm;
    move(upper, [up[0] * (q - 1.0) * TORSO * S, up[1] * (q - 1.0) * TORSO * S]);
    // Shoulders: wider when squashed, narrower as the torso twists, rolled up and down.
    const yaw = m.twist * calm * RAD;
    const span = (1.0 + (1.0 - q) * 0.8) * Math.cos(yaw);
    ["l", "r"].forEach((s, i) => {
      const sd = SIDE[s];
      move(arm[s], add([right[0] * sd * 0.5 * S * (span - 1.0), right[1] * sd * 0.5 * S * (span - 1.0)],
        up, m.shoulders[i] * calm * S));
      move([`${s}_hip`, `${s}_knee`, `${s}_ankle`], [sd * HIP_W / 2 * S * (Math.cos(0.4 * yaw) - 1.0), 0]);
    });
    // Depth: thicker near the viewer, a forearm pointing at it foreshortened.
    const armScale = {}, behind = [];
    for (const s of ["l", "r"]) {
      const uz = m.depth[s][0] * calm, hz = m.depth[s][1] * calm;
      if (hz > 0.0) {
        const el = j[`${s}_elbow`], wr = j[`${s}_wrist`];
        const nw = add(el, sub(wr, el), 1.0 - FORESHORTEN * hz);
        j[`${s}_hand`] = add(j[`${s}_hand`], sub(nw, wr));
        j[`${s}_wrist`] = nw;
      }
      armScale[s] = [1.0 + DEPTH_THICK * uz, 1.0 + DEPTH_THICK * hz, 1.0 + GLOVE_NEAR * Math.max(-0.5, hz)];
      if (this.shade && uz < BEHIND && hz < 0.15) behind.push(s);
    }
    const headYaw = m.headYaw * calm;
    const headAngle = p.lean + p.head + m.headTilt * calm;
    const ha = headAngle * RAD;
    let headC = add(j.neck, [Math.sin(ha), -Math.cos(ha)], (0.50 - 0.22 * Math.min(1.0, bow)) * S);
    headC = add(add(headC, up, -m.headBob * calm * S), right, Math.sin(headYaw * RAD) * HEAD_TURN * S);
    const headTop = headC[1] - (this._face[1] - this._earTop) * this._headSide;
    const g = {
      j, up, beat, headC, headAngle, headTop, headYaw, silhouette: [], bbox: [0, 0, 0, 0],
      twist: yaw / RAD, tail: motion ? m.tail * calm : Math.sin(beat * Math.PI) * 14.0,
      hands: [m.hands[0] * calm, m.hands[1] * calm], armScale, behind,
      frontFirst: yaw > 0 ? ["r", "l"] : ["l", "r"], lift: lift * S,
    };
    const sl = this._shoe(j, "l")[0], sr = this._shoe(j, "r")[0];
    g.ground = [(sl[0] + sr[0]) / 2, oy + (REST_ANKLE_Y + 0.075 + 0.08) * S];
    g.feetSpan = Math.abs(sr[0] - sl[0]);
    g.silhouette = this._silhouette(g, this.inkW + this.rimW);
    const xs = [...Object.values(j).map((v) => v[0]), headC[0]];
    const ys = [...Object.values(j).map((v) => v[1]), headTop];
    const mg = 0.5 * S;
    g.bbox = [Math.trunc(Math.min(...xs) - mg), Math.trunc(Math.min(...ys) - 0.1 * S),
      Math.trunc(Math.max(...xs) + mg), Math.trunc(Math.max(...ys) + 0.4 * S)];
    return g;
  }

  _segs(g) {
    const S = this.S, j = g.j, sc = g.armScale;
    const arms = ["l", "r"].map((s) => [...j[`${s}_shoulder`], ...j[`${s}_elbow`], R_UPPER[0] * S * sc[s][0],
      R_UPPER[1] * S * sc[s][0]])
      .concat(["l", "r"].map((s) => [...j[`${s}_elbow`], ...j[`${s}_wrist`], R_FORE[0] * S * sc[s][0],
        R_FORE[1] * S * sc[s][1]]));
    const legs = ["l", "r"].map((s) => [...j[`${s}_hip`], ...j[`${s}_knee`], R_THIGH[0] * S, R_THIGH[1] * S])
      .concat(["l", "r"].map((s) => [...j[`${s}_knee`], ...j[`${s}_ankle`], R_SHIN[0] * S, R_SHIN[1] * S]));
    return [arms, legs];
  }

  /** The jacket's and the pants' circles. */
  _torsoParts(j, up) {
    const S = this.S;
    const right = [-up[1], up[0]];
    const waist = add(j.neck, sub(j.hip, j.neck), 0.66);
    return {
      top: [[j.l_shoulder, j.r_shoulder, add(waist, right, -0.30 * S), add(waist, right, 0.30 * S)],
        [0.2 * S, 0.2 * S, 0.13 * S, 0.13 * S]],
      pants: [[add(waist, right, -0.29 * S), add(waist, right, 0.29 * S), j.l_hip, j.r_hip],
        [0.12 * S, 0.12 * S, 0.19 * S, 0.19 * S]],
      waist, right,
    };
  }

  /** A ringed raccoon tail from behind the hips, swinging with the groove. */
  _tail(j, swing) {
    const S = this.S;
    const start = add(j.hip, [-0.22 * S, 0.12 * S]);
    const pts = [start], radii = [0.12 * S];
    let ang = 150.0 + swing * 0.5; // screen degrees, y down: left and a little down
    let p = start;
    for (const r of [0.16, 0.19, 0.19, 0.16, 0.11]) {
      ang += 14.0 + swing * 0.25;
      p = add(p, [Math.cos(ang * RAD), Math.sin(ang * RAD)], 0.2 * S);
      pts.push(p);
      radii.push(r * S);
    }
    return [pts, radii];
  }

  _shoe(j, s) {
    const side = s === "l" ? -1.0 : 1.0, S = this.S;
    return [add(j[`${s}_ankle`], [side * 0.07 * S, 0.075 * S]), [0.205 * S, 0.112 * S]];
  }

  _glove(g, s) {
    let d = unit(sub(g.j[`${s}_hand`], g.j[`${s}_wrist`]));
    const flick = g.hands[s === "l" ? 0 : 1] * RAD;
    if (flick) {
      const c = Math.cos(flick), sn = Math.sin(flick);
      d = [d[0] * c - d[1] * sn, d[0] * sn + d[1] * c];
    }
    const k = g.armScale[s][2];
    return [add(g.j[`${s}_wrist`], d, 0.1 * this.S * k), [0.15 * this.S * k, 0.125 * this.S * k],
      Math.atan2(d[1], d[0]) / RAD, d];
  }

  _silhouette(g, grow) {
    const { j, up } = g;
    const [arms, legs] = this._segs(g);
    const polys = [...capsules(arms, grow), ...capsules(legs, grow)];
    const { top, pants } = this._torsoParts(j, up);
    polys.push(hull(top[0], top[1], grow), hull(pants[0], pants[1], grow));
    const [tp, tr] = this._tail(j, g.tail);
    polys.push(...capsules(tp.slice(1).map((b, i) => [...tp[i], ...b, tr[i], tr[i + 1]]), grow));
    polys.push(...capsules([[...j.neck, ...add(j.neck, up, 0.25 * this.S), 0.14 * this.S, 0.14 * this.S]], grow));
    for (const s of ["l", "r"]) {
      const [c, ax] = this._shoe(j, s);
      polys.push(ellipsePoly(c, [ax[0] + grow, ax[1] + grow], 0.0));
      const [gc, gax, gang] = this._glove(g, s);
      polys.push(ellipsePoly(gc, [gax[0] + grow, gax[1] + grow], gang));
    }
    return polys;
  }

  // --- drawing ---

  /** Draw the coach. With `clip` [x0, y0, x1, y1] nothing lands outside it. */
  paint(ctx, g, clip = null) {
    ctx.save();
    if (clip) {
      ctx.beginPath();
      ctx.rect(clip[0], clip[1], clip[2] - clip[0], clip[3] - clip[1]);
      ctx.clip();
    }
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    this._paint(ctx, g);
    ctx.restore();
  }

  /** Capsules in a material: flat, or shadow + lit + highlight. */
  _caps(ctx, segs, mat, streak = true) {
    const [base, shadow, hi] = this.mat[mat];
    if (!this.shade) {
      fillPolys(ctx, capsules(segs, 0.0), base);
      return;
    }
    fillPolys(ctx, capsules(segs, 0.0), shadow);
    fillPolys(ctx, capsules(litSegs(segs), 0.0), base);
    if (streak) fillPolys(ctx, capsules(streakSegs(segs), 0.0), hi);
  }

  _ell(ctx, c, ax, ang, mat) {
    const [base, shadow] = this.mat[mat];
    if (!this.shade) {
      ellipse(ctx, c, ax, ang, base);
      return;
    }
    ellipse(ctx, c, ax, ang, shadow);
    const k = SHADE_K * Math.min(ax[0], ax[1]);
    ellipse(ctx, add(c, LIGHT, k), [ax[0] - k, ax[1] - k], ang, base);
  }

  _paint(ctx, g) {
    const c = this.c, S = this.S, iw = this.inkW, ink = c.ink;
    const { j, up } = g;

    // 0. a soft contact shadow under the feet, smaller and lighter in the air
    if (this.shade && this._shadow) {
      const hop = Math.min(1.0, g.lift / Math.max(1.0, 0.25 * S));
      const w = Math.max(1.4 * S, g.feetSpan + 0.85 * S) * (1.0 - 0.25 * hop) * 1.15;
      const h = 0.17 * S * (1.0 - 0.2 * hop) * 1.9;
      drawC(ctx, this._shadow, g.ground[0], g.ground[1], w, h, 1.0 - 0.5 * hop);
    }

    // 1. the neon rim around the whole silhouette (more of it on the lower right)
    const rs = this.shade ? 0.45 * this.rimW : 0.0;
    ctx.save();
    ctx.translate(rs, 0.5 * rs);
    fillPolys(ctx, g.silhouette, c.rim);
    ctx.restore();

    // 2. tail
    const [tp, tr] = this._tail(j, g.tail);
    const segs = tp.slice(1).map((b, i) => [...tp[i], ...b, tr[i], tr[i + 1]]);
    fillPolys(ctx, capsules(segs, iw), ink);
    this._caps(ctx, segs, "fur", false);
    for (const k of [2, 4]) { // dark rings across the tail
      const a = tp[k - 1], b = k + 1 < tp.length ? tp[k + 1] : tp[k];
      const d = sub(b, a);
      const n = unit([-d[1], d[0]]);
      const r = tr[k] - iw * 0.3;
      strokeLine(ctx, [add(tp[k], n, -r * 0.92), add(tp[k], n, r * 0.92)], c.fur_dark, 0.11 * S);
    }
    const tip = tp[tp.length - 1], tipR = tr[tr.length - 1];
    disc(ctx, tip, tipR, this.mat.fur_dark[this.shade ? 1 : 0]);
    if (this.shade) disc(ctx, add(tip, LIGHT, SHADE_K * tipR), tipR * (1 - SHADE_K), c.fur_dark);

    const [arms, legs] = this._segs(g);
    // 3. an arm the body has turned away from goes behind the jacket
    for (const s of g.behind) this._paintArm(ctx, g, arms, s);

    // 4. legs and pants: one shape (ink for all, then fill for all: no seam)
    const { top, pants, waist, right } = this._torsoParts(j, up);
    fillPolys(ctx, [...capsules(legs, iw), hull(pants[0], pants[1], iw)], ink);
    const [pBase, pShadow, pHi] = this.mat.pants;
    if (this.shade) {
      fillPolys(ctx, [...capsules(legs, 0.0), hull(pants[0], pants[1], 0.0)], pShadow);
      fillPolys(ctx, [...capsules(litSegs(legs), 0.0), litHull(pants[0], pants[1], HULL_SHADE * S)], pBase);
      fillPolys(ctx, capsules(streakSegs(legs), 0.0), pHi);
      // the jacket's hem shades the top of the pants
      const hem = [[...add(add(waist, right, -0.27 * S), up, -0.1 * S), ...add(add(waist, right, 0.27 * S), up, -0.1 * S),
        0.1 * S, 0.1 * S]];
      fillPolys(ctx, capsules(hem, 0.0), pShadow);
    } else {
      fillPolys(ctx, [...capsules(legs, 0.0), hull(pants[0], pants[1], 0.0)], pBase);
    }
    for (const s of ["l", "r"]) { // a stripe down the outside of each leg
      const side = s === "l" ? -1.0 : 1.0;
      const pts = [j[`${s}_hip`], j[`${s}_knee`], j[`${s}_ankle`]];
      const offPts = pts.map((p, i) => {
        const d = sub(pts[Math.min(2, i + 1)], pts[Math.max(0, i - 1)]);
        let n = unit([-d[1], d[0]]);
        if (n[0] * side < 0) n = [-n[0], -n[1]];
        const r = [R_THIGH[0], R_THIGH[1], R_SHIN[1]][i] * S * 0.58;
        return add(p, n, r);
      });
      strokeLine(ctx, offPts, c.accent, Math.round(0.045 * S));
    }
    for (const s of ["l", "r"]) { // sneakers
      const [sc, ax] = this._shoe(j, s);
      ellipse(ctx, sc, [ax[0] + iw, ax[1] + iw], 0, ink);
      this._ell(ctx, sc, ax, 0, "shoe");
      ellipse(ctx, sc, [ax[0] - 0.02 * S, ax[1] - 0.025 * S], 0, c.accent, Math.max(1, Math.round(0.05 * S)), 25, 155);
    }

    // 5. jacket: a V of chest fur and a belt that wrap round as the chest twists
    fillPolys(ctx, [hull(top[0], top[1], iw)], ink);
    const [tBase, tShadow, tHi] = this.mat.top;
    if (this.shade) {
      fillPolys(ctx, [hull(top[0], top[1], 0.0)], tShadow);
      fillPolys(ctx, [litHull(top[0], top[1], HULL_SHADE * S)], tBase);
      // a soft sheen down the lit side of the chest
      const ls = top[0][0], wl = add(waist, right, -0.30 * S);
      const a = add(add(ls, sub(wl, ls), 0.18), right, 0.12 * S), b = add(add(ls, sub(wl, ls), 0.62), right, 0.12 * S);
      fillPolys(ctx, capsules([[...a, ...b, 0.05 * S, 0.035 * S]], 0.0), tHi);
    } else {
      fillPolys(ctx, [hull(top[0], top[1], 0.0)], tBase);
    }
    const neck = j.neck;
    const turn = Math.sin(g.twist * RAD);
    const chest = [right[0] * turn * CHEST_TURN * S, right[1] * turn * CHEST_TURN * S];
    const v = [add(add(add(neck, right, -0.2 * S), up, -0.02 * S), chest, 0.6),
      add(add(add(neck, right, 0.2 * S), up, -0.02 * S), chest, 0.6), add(add(neck, up, -0.52 * S), chest)];
    fillPolys(ctx, [v], this.mat.fur[0]);
    strokeLine(ctx, [v[0], v[2], v[1]], c.accent, Math.round(0.04 * S));
    if (this.shade) { // the belt sags at the front of a round waist, and slides as it twists
      const belt = [];
      for (let i = 0; i < 9; i++) {
        const u = -1 + i / 4;
        const sag = 0.035 * S * (1.0 - ((u - 0.5 * turn) / (1.0 + 0.5 * Math.abs(turn))) ** 2);
        belt.push(add(add(waist, right, u * 0.36 * S), up, -sag));
      }
      strokeLine(ctx, belt, ink, 0.11 * S + 2 * iw * 0.7);
      strokeLine(ctx, belt, c.accent, 0.11 * S);
      // the head shades the top of the chest
      const ang = Math.atan2(right[1], right[0]) / RAD;
      ellipse(ctx, add(add(neck, up, -0.06 * S), chest, 0.5), [0.3 * S, 0.1 * S], ang, tShadow);
      ellipse(ctx, add(add(neck, up, -0.08 * S), chest, 0.6), [0.17 * S, 0.075 * S], ang, this.mat.fur[1]);
    } else {
      const wl = add(waist, right, -0.36 * S), wr = add(waist, right, 0.36 * S);
      strokeLine(ctx, [wl, wr], ink, 0.11 * S + 2 * iw * 0.7);
      strokeLine(ctx, [wl, wr], c.accent, 0.11 * S);
    }

    // 6. neck (in the head's shadow), 7. head
    const neckSeg = [[...neck, ...add(neck, up, 0.25 * S), 0.14 * S, 0.14 * S]];
    fillPolys(ctx, capsules(neckSeg, iw), ink);
    fillPolys(ctx, capsules(neckSeg, 0.0), this.mat.fur[this.shade ? 1 : 0]);
    let vi = 0;
    this._yaws.forEach((y, i) => { if (Math.abs(y - g.headYaw) < Math.abs(this._yaws[vi] - g.headYaw)) vi = i; });
    const side = this._headSide;
    ctx.save();
    ctx.translate(g.headC[0], g.headC[1]);
    if (g.headAngle) ctx.rotate(g.headAngle * RAD); // positive tips the head screen-right
    ctx.drawImage(this._heads[vi], -this._face[0] * side, -this._face[1] * side);
    ctx.restore();

    // 8. arms in front, the nearer one last
    for (const s of g.frontFirst) if (!g.behind.includes(s)) this._paintArm(ctx, g, arms, s);
  }

  /** Fur forearm, sleeve with a cuff, wristband, white glove with a thumb. */
  _paintArm(ctx, g, arms, s) {
    const c = this.c, S = this.S, iw = this.inkW, ink = c.ink, j = g.j;
    const idx = s === "l" ? 0 : 1;
    const upper = arms[idx], fore = arms[idx + 2];
    const [gc, gax, gang, d] = this._glove(g, s);
    fillPolys(ctx, capsules([upper, fore], iw), ink);
    ellipse(ctx, gc, [gax[0] + iw, gax[1] + iw], gang, ink);
    this._caps(ctx, [fore], "fur");
    this._caps(ctx, [upper], "top");
    const el = j[`${s}_elbow`], wr = j[`${s}_wrist`];
    const r = R_FORE[1] * S * 1.08 * g.armScale[s][1];
    fillPolys(ctx, capsules([[...add(el, sub(wr, el), 0.78), ...wr, r, r]], 0.0), c.accent);
    this._ell(ctx, gc, gax, gang, "glove");
    let n = [-d[1], d[0]];
    const sd = s === "l" ? -1.0 : 1.0;
    if (n[0] * sd > 0) n = [-n[0], -n[1]];
    const k = g.armScale[s][2];
    const tc = add(add(gc, n, 0.12 * S * k), d, -0.02 * S);
    disc(ctx, tc, 0.055 * S * k + iw, ink);
    disc(ctx, tc, 0.055 * S * k, c.glove);
    if (this.shade) {
      const kk = SHADE_K * Math.min(gax[0], gax[1]);
      ellipse(ctx, add(gc, LIGHT, kk), [gax[0] * 0.92 - kk, gax[1] * 0.86 - kk], gang, c.glove);
    } else {
      ellipse(ctx, gc, [gax[0] * 0.92, gax[1] * 0.86], gang, c.glove);
    }
  }
}

// --- pictograms -------------------------------------------------------------------

const PICTO_BODY = bgr([222, 205, 236]);
const PICTO_ARM = WHITE, PICTO_ARM_GOLD = bgr([170, 238, 255]);
const PICTO_MARK = bgr([255, 220, 110]); // theme CYAN: the step / lift marks on a card

/**
 * A bold pictogram dancer for `pose`, fitted into rect [x0, y0, x1, y1]: body
 * in light lavender, arms in white (gold-white on gold cards) over it with their
 * own keyline, because the arms are what the judge mostly reads. When the
 * judge reads the legs too (legTarget), they are drawn like the arms, on a
 * floor line, with an arrow on the floor towards the foot the weight goes over
 * (a step or a sway) and a chevron over a lifted foot (a knee lift, a kick).
 */
export function drawPictogramFigure(ctx, pose, rect, gold = false) {
  const p0 = normalizePose(pose);
  const j = forward(p0);
  const lt = legTarget(p0);
  const [x0, y0, x1, y1] = rect;
  const headR = 0.36;
  const pts = Object.values(j);
  const lo = [Math.min(...pts.map((p) => p[0])) - 0.25, Math.min(...pts.map((p) => p[1])) - 0.2];
  const hi = [Math.max(...pts.map((p) => p[0])) + 0.25, Math.max(...pts.map((p) => p[1])) + 0.25];
  lo[1] = Math.min(lo[1], j.head[1] - headR - 0.12);
  const floor = Math.max(j.l_ankle[1], j.r_ankle[1]) + 0.12;
  if (lt.active) hi[1] = Math.max(hi[1], floor + 0.3);
  const bw = hi[0] - lo[0], bh = hi[1] - lo[1];
  const s = Math.min((x1 - x0) / bw, (y1 - y0) / bh, (y1 - y0) / 4.3);
  const cx = (x0 + x1) / 2 - (lo[0] + bw / 2) * s, cy = (y0 + y1) / 2 - (lo[1] + bh / 2) * s;
  const P = (name) => [cx + j[name][0] * s, cy + j[name][1] * s];
  const Q = (x, y) => [cx + x * s, cy + y * s];
  const lw = 0.3 * s, ow = 0.3 * s + 0.16 * s;
  const limb = (names, width, colour) => strokeLine(ctx, names.map(P), colour, width);
  const torso = (grow, colour) => fillPolys(ctx,
    [hull([P("l_shoulder"), P("r_shoulder"), P("r_hip"), P("l_hip")], [0.16 * s, 0.16 * s, 0.16 * s, 0.16 * s], grow)],
    colour);
  // The floor line, the weight arrow and the lift chevrons (grow > 0: their keyline).
  const marks = (colour, grow) => {
    const w = Math.max(1, 0.09 * s + grow);
    const xs = [j.l_ankle[0], j.r_ankle[0]];
    strokeLine(ctx, [Q(Math.min(...xs) - 0.35, floor), Q(Math.max(...xs) + 0.35, floor)], colour, 0.07 * s + grow);
    const lb = lt.lower;
    if (lt.step && Math.abs(lb.weight) > 0.12) {
      const mid = (xs[0] + xs[1]) / 2, d = lb.weight > 0 ? 0.62 : -0.62;
      const a = Q(mid - d * 0.5, floor + 0.2), b = Q(mid + d * 0.5, floor + 0.2);
      const tip = 0.42 * Math.hypot(b[0] - a[0], b[1] - a[1]), ang = Math.atan2(b[1] - a[1], b[0] - a[0]);
      strokeLine(ctx, [a, b], colour, w);
      for (const t of [Math.PI / 4, -Math.PI / 4]) {
        strokeLine(ctx, [b, [b[0] - tip * Math.cos(ang + t), b[1] - tip * Math.sin(ang + t)]], colour, w);
      }
    }
    if (lt.lift) {
      const side = lb.lift > 0 ? "r" : "l";
      const [ax, ay] = j[`${side}_ankle`];
      const ox = SIDE[side] * 0.32;
      for (const k of [0.0, 0.2]) {
        strokeLine(ctx, [Q(ax + ox - 0.13, ay + 0.02 - k), Q(ax + ox, ay - 0.12 - k), Q(ax + ox + 0.13, ay + 0.02 - k)],
          colour, w);
      }
    }
  };
  const legs = lt.active ? (gold ? PICTO_ARM_GOLD : PICTO_ARM) : PICTO_BODY;
  ctx.save();
  ctx.lineCap = ctx.lineJoin = "round";
  if (lt.active) marks(INK, 0.12 * s);
  for (const sd of ["l", "r"]) limb([`${sd}_hip`, `${sd}_knee`, `${sd}_ankle`], ow, INK);
  torso(0.08 * s, INK);
  disc(ctx, P("head"), headR * s + 0.08 * s, INK);
  for (const sd of ["l", "r"]) limb([`${sd}_shoulder`, `${sd}_elbow`, `${sd}_hand`], ow, INK);
  if (lt.active) marks(PICTO_MARK, 0.0);
  for (const sd of ["l", "r"]) limb([`${sd}_hip`, `${sd}_knee`, `${sd}_ankle`], lw, legs);
  torso(0.0, PICTO_BODY);
  disc(ctx, P("head"), headR * s, PICTO_BODY);
  const [hx, hy] = P("head");
  for (const side of [-1, 1]) disc(ctx, [hx + side * 0.24 * s, hy - 0.3 * s], 0.12 * s, PICTO_BODY); // ears
  for (const sd of ["l", "r"]) {
    limb([`${sd}_shoulder`, `${sd}_elbow`, `${sd}_hand`], ow, INK);
    limb([`${sd}_shoulder`, `${sd}_elbow`, `${sd}_hand`], lw, gold ? PICTO_ARM_GOLD : PICTO_ARM);
  }
  ctx.restore();
}

const poseKey = (pose) => {
  const p = normalizePose(pose);
  return Object.keys(p).sort().map((k) => (Array.isArray(p[k]) ? p[k].map((v) => v.toFixed(1)).join(",")
    : p[k].toFixed(1))).join("|");
};

/** Pictogram cards (frame + figure), built once per (pose, gold, width) and cached. */
export class PictogramCards {
  constructor(art, { makeCanvas = defaultMakeCanvas } = {}) {
    this.frames = { false: art.card, true: art.cardGold };
    this.mk = makeCanvas;
    this._cache = new Map();
  }

  get(pose, gold, width) {
    width = Math.round(width);
    const key = `${poseKey(pose)}/${Boolean(gold)}/${width}`;
    let card = this._cache.get(key);
    if (!card) {
      const frame = this.frames[Boolean(gold)];
      const h = Math.round(width * frame.height / frame.width);
      card = this.mk(width, h);
      const g = card.getContext("2d");
      g.imageSmoothingQuality = "high";
      g.drawImage(frame, 0, 0, width, h);
      const inset = 0.135 * width;
      drawPictogramFigure(g, pose, [inset, inset * 0.95, width - inset, h - inset * 0.95], gold);
      this._cache.set(key, card);
    }
    return card;
  }
}

// --- the renderer -------------------------------------------------------------------

export class DanceRenderer {
  constructor(width, height, art, { makeCanvas = defaultMakeCanvas } = {}) {
    this.width = width;
    this.height = height;
    this.art = art;
    this.mk = makeCanvas;
    const man = art.manifest;
    const q = Number(man.light_scale ?? 4);
    this.q = [1, 2, 4, 8].includes(q) ? q : 4;
    const lw = Math.floor(width / this.q), lh = Math.floor(height / this.q);
    this._light = makeCanvas(lw, lh);
    this._lctx = this._light.getContext("2d");
    this._canFilter = typeof this._lctx.filter === "string";

    const cols = man.colours ?? {};
    this.cMagenta = cols.magenta ?? bgr([190, 50, 255]);
    this.cCyan = cols.cyan ?? bgr([255, 235, 70]);
    this.cViolet = cols.violet ?? bgr([255, 90, 150]);
    this.cGold = cols.gold ?? bgr([70, 205, 255]);
    const tint = (img, colour, w, h) => tintedCanvas(makeCanvas, img, colour, w, h);
    this._floor = [this.cViolet, this.cMagenta, this.cCyan, this.cMagenta].map((c) => tint(art.glowFloor, c, lw, lh));
    this._edgeGold = tint(art.glowEdge, this.cGold, lw, lh);

    // Beams: the white beam at 1/q, one tinted copy per light; rotated when drawn.
    const lights = man.stage?.lights ?? [[170, 94], [345, 94], [520, 94], [695, 94], [870, 94]];
    this._lights = lights.map(([x, y]) => [x / this.q, y / this.q]);
    const palette = [this.cMagenta, this.cCyan, bgr([235, 225, 255]), this.cCyan, this.cMagenta];
    const bw = Math.max(4, Math.floor(art.beam.width / this.q)), bh = Math.max(8, Math.floor(art.beam.height / this.q));
    const beamCache = new Map();
    this._beams = this._lights.map((_, i) => {
      const c = palette[i % palette.length];
      const k = c.join(",");
      if (!beamCache.has(k)) beamCache.set(k, tint(art.beam, c, bw, bh));
      return beamCache.get(k);
    });

    this._painter = null;
    this._painterOutfit = null;
    this._cards = new PictogramCards(art, { makeCanvas });
    this._ring = tint(art.ring, this.cCyan, RING, RING);
    this._ringGold = tint(art.ring, this.cGold, RING, RING);
    this._ringLight = tint(art.ring, WHITE, RING, RING);
    // Difficulty colours: EASY green, NORMAL cyan, HARD magenta (the manifest's
    // palette, overridable there as "difficulty": {"easy": [b, g, r], ...}).
    const dc = isPlainObject(man.difficulty) ? man.difficulty : {};
    this.difficultyColours = {
      easy: dc.easy ? bgr(dc.easy) : GOOD,
      normal: dc.normal ? bgr(dc.normal) : this.cCyan,
      hard: dc.hard ? bgr(dc.hard) : this.cMagenta,
    };
    this._badges = new Map();
    this._laneBg = this._makeLaneBg();

    // FX
    const sizes = [22, 30, 40, 52];
    this._sparkle = sizes.map((s) => tint(art.sparkle, WHITE, s, s));
    this._sparkleGold = sizes.map((s) => tint(art.sparkle, bgr([150, 230, 255]), s, s));
    this._burst = tint(art.burst, bgr([120, 215, 255]));
    this._glowDot = tint(art.glow, bgr([150, 230, 255]), 90, 90);
    this._starBar = this._makeStarBar();
    this._titleCache = new Map();

    // Animation state the game doesn't carry.
    this._scoreSeen = null;
    this._scoreT = -10.0;
    this._starsLit = new Map();
    this._outroT0 = null;
    this._clock = 0.0;
    this._comboSeen = 0;
    this._comboT = -10.0;
    this.timings = null; // set to {} to collect per-part ms
  }

  // --- pre-built pieces ---

  _makeLaneBg() {
    const [x0, y0, x1, y1] = LANE_RECT;
    const w = x1 - x0, h = y1 - y0, r = 20;
    const c = this.mk(w, h);
    const g = c.getContext("2d");
    g.beginPath();
    roundRect(g, 0, 0, w, h, r);
    g.fillStyle = css(bgr([34, 14, 22]), 0.72);
    g.fill();
    g.strokeStyle = css(this.cViolet, 0.78);
    g.lineWidth = 2;
    g.lineCap = "round";
    g.beginPath();
    g.moveTo(r, 1);
    g.lineTo(w - r, 1);
    g.moveTo(r, h - 2);
    g.lineTo(w - r, h - 2);
    g.stroke();
    // Fade towards the right edge so cards emerge rather than pop.
    g.globalCompositeOperation = "destination-in";
    const grad = g.createLinearGradient(0, 0, w, 0);
    for (let i = 0; i <= 8; i++) {
      const u = i / 8;
      const ramp = Math.min(1, Math.max(0, 1.6 * (1 - u))) ** 0.6;
      grad.addColorStop(u, `rgba(0,0,0,${0.55 + 0.45 * ramp})`);
    }
    g.fillStyle = grad;
    g.fillRect(0, 0, w, h);
    return c;
  }

  _makeStarBar() {
    const [x0, y0, x1, y1] = STAR_BAR;
    const w = x1 - x0 + 12, h = y1 - y0 + 12;
    const track = this.mk(w, h);
    let g = track.getContext("2d");
    g.fillStyle = css(bgr([60, 30, 40]), 220 / 255);
    g.fillRect(6, 6, w - 12, h - 12);
    g.strokeStyle = css(bgr([220, 150, 190]), 160 / 255);
    g.lineWidth = 2;
    g.strokeRect(5, 5, w - 10, h - 10);
    // The fill: magenta at the bottom rising to gold, with a glossy stripe.
    const fw = x1 - x0, fh = y1 - y0;
    const fill = this.mk(fw, fh);
    g = fill.getContext("2d");
    const v = g.createLinearGradient(0, 0, 0, fh);
    v.addColorStop(0, css(this.cGold));
    v.addColorStop(1, css(this.cMagenta));
    g.fillStyle = v;
    g.fillRect(0, 0, fw, fh);
    const gloss = g.createLinearGradient(0, 0, fw, 0);
    for (let i = 0; i <= 10; i++) {
      const xs = -1 + 2 * (i / 10);
      gloss.addColorStop(i / 10, `rgba(255,255,255,${Math.min(1, Math.max(0, 0.35 - Math.abs(xs + 0.3)))})`);
    }
    g.fillStyle = gloss;
    g.fillRect(0, 0, fw, fh);
    return { track, fill };
  }

  _painterFor(song) {
    const outfit = song?.coach?.outfit ?? "neon";
    if (!this._painter || this._painterOutfit !== outfit) {
      this._painter = new CoachPainter(this.art, outfit, { makeCanvas: this.mk });
      this._painterOutfit = outfit;
    }
    return this._painter;
  }

  // --- main ---

  draw(ctx, frame, game) {
    const tm = this.timings;
    let t0 = tm ? performance.now() : 0;
    const lap = (name) => {
      if (!tm) return;
      const t1 = performance.now();
      tm[name] = (tm[name] ?? 0) + (t1 - t0);
      t0 = t1;
    };
    const phase = String(game.phase ?? "");
    const song = game.song ?? null;
    let beat = Number(game.beat) || 0;
    const songTime = Number(game.songTime) || 0;
    const over = phase === "game_over";
    if (!over) this._clock = songTime;

    ctx.save();
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = "source-over";
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(this.art.stage, 0, 0, this.width, this.height);
    lap("stage");

    const painter = this._painterFor(song);
    const bow = this._bow(phase, song, songTime);
    let motion;
    if (over) { // the coach grooves in its spotlight while the results count up
      beat = (Number(game.resultsT) || 0) * 2.0;
      motion = coachMotion(null, beat, { energy: 0.5 });
    } else if (song?._frames) {
      // The performance on top of the chart's timeline (motion.js): the same
      // keyframes as game.coachPose, hit on the same beats.
      motion = coachMotion(song, beat);
    } else {
      motion = stillMotion(game.coachPose ?? REST);
    }
    const geo = painter.layout(null, { bow, beat, motion });
    const energy = this._energy(song, beat, phase);
    this._drawLights(ctx, game, geo, beat, energy, phase);
    lap("lights");

    painter.paint(ctx, geo, COACH_ROI);
    lap("coach");

    if (over) {
      this._drawResults(ctx, game, song);
      lap("results");
      ctx.restore();
      return;
    }
    if (phase === "intro" || phase === "dancing") this._drawLane(ctx, game, beat);
    lap("lane");
    this._drawFeedback(ctx, game, geo);
    lap("feedback");
    if (phase === "intro") this._drawIntro(ctx, game, song, beat, game.difficulty);
    if (phase === "outro") this._drawOutroFade(ctx);
    lap("intro_outro");
    this._drawHud(ctx, game, phase);
    lap("hud");
    this._drawPip(ctx, frame, game);
    lap("pip");
    ctx.restore();
  }

  // --- timing helpers ---

  _energy(song, beat, phase) {
    if (phase === "game_over") return 0.35;
    let name = "";
    try {
      name = String(song?.sectionAt?.(beat) ?? "").toLowerCase();
    } catch {
      name = ""; // a malformed section list must not stop the show
    }
    for (const [key, e] of SECTION_ENERGY) if (name.includes(key)) return e;
    return 0.72;
  }

  _bow(phase, song, songTime) {
    if (phase !== "outro" && phase !== "game_over") {
      this._outroT0 = null;
      return 0.0;
    }
    if (this._outroT0 == null) this._outroT0 = Math.max(songTime, endingStart(song));
    if (phase === "game_over") return 0.0;
    const bs = song?.beatSeconds ?? 0.5;
    // Down over a beat, hold two, back up over a beat.
    const u = (songTime - this._outroT0) / bs;
    if (u < 0.4) return 0.0;
    if (u < 1.4) return smoothstep(u - 0.4);
    if (u < 3.4) return 1.0;
    return 1.0 - smoothstep((u - 3.4) / 1.2);
  }

  // --- layers ---

  _drawLights(ctx, game, geo, beat, energy, phase) {
    const L = this._lctx, q = this.q;
    const lw = this._light.width, lh = this._light.height;
    L.globalCompositeOperation = "source-over";
    L.globalAlpha = 1;
    L.filter = "none";
    L.clearRect(0, 0, lw, lh);
    L.fillStyle = "#000";
    L.fillRect(0, 0, lw, lh);
    L.globalCompositeOperation = "lighter";
    const k = phase === "intro" || phase === "dancing" || phase === "outro" ? kick(beat) : 0.0;
    const bar = phase === "dancing" ? kick(beat / 4.0, 6.0) : 0.0;
    // Neon trims flare on each bar's downbeat; the floor pool on every beat.
    const gain = (img, a) => {
      for (let g = a; g > 0.004; g -= 1) {
        L.globalAlpha = Math.min(1, g);
        L.drawImage(img, 0, 0, lw, lh);
      }
    };
    gain(this.art.glowTrims, 0.10 + 0.55 * bar * energy + 0.12 * k);
    const fl = beat >= 0 ? this._floor[Math.floor(beat / 4.0) % this._floor.length] : this._floor[0];
    gain(fl, (0.18 + 0.62 * k) * (0.5 + 0.5 * energy));

    // Beams: outer lights sweep over two bars; the centre one breathes straight down.
    const n = this._lights.length;
    this._lights.forEach(([lx, ly], i) => {
      let ang, g;
      if (Math.abs(i - (n - 1) / 2) < 0.6) {
        ang = 0.0;
        g = 0.22 + 0.25 * k * energy;
      } else {
        const side = i < (n - 1) / 2 ? -1.0 : 1.0;
        const swing = energy >= 0.9 ? 24.0 : 16.0;
        ang = -side * 14.0 + swing * Math.sin(2 * Math.PI * beat / 8.0 + i * 1.9);
        g = (0.38 + 0.5 * k) * energy;
      }
      if (g <= 0.01) return;
      const img = this._beams[i];
      L.save();
      L.globalAlpha = Math.min(1, g);
      L.translate(lx, ly);
      // OpenCV's positive angle turns counter-clockwise on screen; the canvas's turns clockwise.
      L.rotate(-Math.round(ang / 2.0) * 2.0 * RAD);
      L.drawImage(img, -img.width / 2, 0);
      L.restore();
    });

    // The coach's halo: its silhouette at 1/q, blurred, in the outfit's glow colour.
    const painter = this._painter;
    if (painter) {
      const g = 0.75 + 0.35 * k;
      L.save();
      L.globalAlpha = 1;
      if (this._canFilter) L.filter = "blur(2.4px)";
      else L.globalAlpha = 0.6;
      L.scale(1 / q, 1 / q);
      L.fillStyle = css(clamp255(painter.c.glow ?? bgr([255, 80, 200]), g));
      L.beginPath();
      for (const p of geo.silhouette) path(L, p);
      L.ellipse(geo.headC[0], geo.headC[1], 0.62 * painter.S, 0.58 * painter.S, 0, 0, 2 * Math.PI);
      L.fill();
      L.restore();
    }

    const gold = phase !== "game_over" ? Number(game.goldFlash) || 0 : 0.0;
    if (gold > 0.01) gain(this._edgeGold, Math.min(1.6, gold * 1.6));
    L.globalAlpha = 1;
    L.globalCompositeOperation = "source-over";

    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(this._light, 0, 0, this.width, this.height);
    ctx.restore();
  }

  _drawLane(ctx, game, beat) {
    const items = game.upcoming ?? [];
    if (!items.length) return;
    const [x0, y0, x1, y1] = LANE_RECT;
    ctx.drawImage(this._laneBg, x0, y0);
    const k = kick(beat);
    // Beat dots along the track, scrolling with the cards.
    const b0 = Math.ceil(beat - (TARGET_X - x0) / PX_PER_BEAT);
    const b1 = Math.trunc(beat + (x1 - TARGET_X) / PX_PER_BEAT);
    for (let b = b0; b <= b1; b++) {
      const x = TARGET_X + (b - beat) * PX_PER_BEAT;
      if (x > x0 + 14 && x < x1 - 14 && Math.abs(x - TARGET_X) > RING * 0.41) {
        const big = ((b % 4) + 4) % 4 === 0;
        disc(ctx, [Math.trunc(x), y1 - 10], big ? 4 : 2, big ? bgr([240, 200, 255]) : this.cViolet);
      }
    }
    // The next card: the first one not yet past the ring.
    const nxt = items.find((it) => it.beat - beat >= -0.3) ?? null;
    drawC(ctx, nxt?.gold ? this._ringGold : this._ring, TARGET_X, LANE_Y, RING, RING, 0.75 + 0.25 * k);
    let flash = 0.0;
    if (nxt) {
      const d = nxt.beat - beat;
      if (d >= -0.3 && d <= 0.12) flash = 1.0 - Math.abs(d + 0.05) / 0.3;
    }
    addC(ctx, this._ringLight, TARGET_X, LANE_Y, RING, RING, Math.min(1.0, (0.25 + 0.5 * k) * 0.6 + flash * 0.9));

    const visible = [];
    for (const it of items) {
      const d = it.beat - beat;
      if (d < -0.65) continue;
      const x = TARGET_X + d * PX_PER_BEAT;
      if (x - CARD_SMALL / 2 > x1) continue;
      visible.push([d, x, it]);
    }
    ctx.save();
    ctx.beginPath();
    ctx.rect(x0 + 2, y0 - 16, x1 - x0 - 4, LANE_CLIP_BOTTOM - (y0 - 16));
    ctx.clip();
    // Cards come as often as every beat (one per judge point). Size goes by
    // distance alone, so nothing pops when the "next" card changes: small until
    // GROW_BEATS out, full size on the ring, and it stays full size while it
    // slides through and fades. Passed cards are drawn first, under the
    // incoming ones; then the farthest first, the next on top.
    visible.sort((a, b) => (a[0] >= 0) - (b[0] >= 0) || b[0] - a[0]);
    for (let [d, x, it] of visible) {
      const w = cardWidth(d);
      let fade = 1.0;
      if (d < 0) {
        fade = Math.max(0.0, 1.0 + d / 0.65);
        x = TARGET_X + d * PX_PER_BEAT * 0.6;
      }
      const card = this._cards.get(it.pose, it.gold, w);
      drawC(ctx, card, Math.round(x), LANE_Y, card.width, card.height, fade);
      if (it.gold && fade > 0.3) { // twinkles on gold cards
        for (let s = 0; s < 2; s++) {
          const tw = Math.abs(Math.sin((beat * 1.3 + s * 0.5) * Math.PI));
          const sx = x + (s ? -0.36 : 0.38) * card.width, sy = LANE_Y + (s ? -0.42 : 0.36) * card.height;
          const sp = this._sparkleGold[1];
          addC(ctx, sp, sx, sy, sp.width, sp.height, tw * fade);
        }
      }
    }
    ctx.restore();
    if (nxt) {
      const label = String(nxt.moveLabel ?? nxt.label ?? "").toUpperCase();
      const d = nxt.beat - beat;
      if (label && d < 4.0) {
        const a = Math.min(1.0, (4.0 - d) / 0.6);
        centredText(ctx, nxt.gold ? `${label}  x2` : label, TARGET_X, LABEL_Y, LABEL_SCALE, nxt.gold ? this.cGold : WHITE,
          { alpha: a });
      }
    }
  }

  _drawFeedback(ctx, game, geo) {
    const fbs = game.feedback ?? [];
    // The notice ("STEP BACK...") sits where the words pop; a word showing
    // through its translucent panel read as a ghost (QA).
    if (!fbs.length || game.notice) return;
    const words = this.art.words;
    const cx = Math.min(POPUP_X[1], Math.max(POPUP_X[0], geo.headC[0]));
    const baseY = Math.max(POPUP_Y_MIN + 30, geo.headTop - 26);
    const goldFlash = Number(game.goldFlash) || 0;
    if (goldFlash > 0.01) { // a burst of rays behind the popup
      const s = 0.6 + 0.5 * (1.0 - goldFlash);
      const b = this._burst;
      addC(ctx, b, cx, baseY - 10, b.width * s, b.height * s, Math.min(1.0, goldFlash * 1.1));
    }
    const shown = fbs.filter((fb) => words[ratingKey(fb.rating)]);
    shown.forEach((fb, i) => {
      const key = ratingKey(fb.rating);
      const age = Number(fb.age) || 0;
      if (age < 0 || age > POPUP_LIFE) return;
      // Judge points come every beat (0.5 s) but a popup lives 0.85 s: when a
      // newer rating arrives, the older word whisks up and out instead of
      // stacking under it.
      const superseded = i + 1 < shown.length ? Number(shown[i + 1].age) || 0 : null;
      if (superseded != null && superseded >= SUPERSEDE_FADE) return;
      const gold = Boolean(fb.gold);
      let s = pop(age);
      let x = cx;
      if (key === "x") { // a miss shakes instead of bouncing; one letter needs the extra size
        s = Math.min(1.0, 0.7 + age * 3) * 1.35;
        x += Math.sin(age * 55) * 9 * Math.max(0.0, 1 - age / 0.35);
      }
      let y = baseY - 34 * easeOut(age / POPUP_LIFE);
      let fade = age < 0.5 ? 1.0 : Math.max(0.0, 1.0 - (age - 0.5) / (POPUP_LIFE - 0.5));
      if (superseded != null) {
        const u = superseded / SUPERSEDE_FADE;
        fade *= (1.0 - u) ** 2; // mostly gone on the new word's first frame
        y -= 46 * u;
      }
      if (key === "perfect" || (gold && (key === "perfect" || key === "super"))) this._sparkles(ctx, x, y, age, gold);
      const img = words[key];
      const sc = s * (gold && key !== "x" ? 1.1 : 1.0);
      drawC(ctx, img, x, y, img.width * sc, img.height * sc, fade);
    });
    const combo = Number(game.combo) || 0;
    if (combo !== this._comboSeen) {
      if (combo > this._comboSeen) this._comboT = this._clock;
      this._comboSeen = combo;
    }
  }

  _sparkles(ctx, x, y, age, gold) {
    const u = Math.min(1.0, age / 0.7);
    const n = gold ? 8 : 6;
    const sprites = gold ? this._sparkleGold : this._sparkle;
    for (let k = 0; k < n; k++) {
      const a = (k / n) * 2 * Math.PI + (gold ? 0.4 : 0.0);
      const r = 60 + 120 * easeOut(u) * (k % 2 ? 1.0 : 0.75);
      const sx = x + Math.cos(a) * r * 1.5, sy = y + Math.sin(a) * r * 0.7 - 20 * u;
      const size = k % 2 === 0 && u < 0.5 ? 3 : u < 0.6 ? 2 : 1;
      const sp = sprites[size];
      addC(ctx, sp, sx, sy, sp.width, sp.height, Math.max(0.0, 1.0 - u) * 1.2);
    }
    if (age < 0.25) addC(ctx, this._glowDot, x, y, 90, 90, (1 - age / 0.25) * (gold ? 0.9 : 0.6));
  }

  _drawIntro(ctx, game, song, beat, difficulty = null) {
    const fm = song?.firstMoveBeat || 16.0;
    const bs = song?.beatSeconds ?? 0.5;
    const [cx, cy] = INTRO_C;
    const words = this.art.words;
    const title = this._title(song);
    if (title && beat < fm - 7.5) {
      const [tImg, aImg] = title;
      const age = beat + 4.0; // the title is up from before beat 0
      const fade = Math.min(1.0, (fm - 7.5 - beat) / 0.75);
      const dy = (1 - easeOut(age / 1.2)) * 30;
      centredText(ctx, "NOW DANCING", cx, cy - 74 + dy, 0.62, DIM, { body: true, alpha: fade });
      drawC(ctx, tImg, cx, cy + dy, tImg.width, tImg.height, fade);
      if (aImg) drawC(ctx, aImg, cx, cy + 66 + dy, aImg.width, aImg.height, fade);
      const badge = this.badge(difficulty, 0.8);
      if (badge) drawC(ctx, badge, cx, cy + 122 + dy, badge.width, badge.height, fade);
      if (game.loading) {
        centredText(ctx, "LOADING...", cx, cy + (badge ? 174 : 128) + dy, 0.6, DIM, { body: true, alpha: fade });
      }
    }
    const k = beat - (fm - 8);
    if (k >= 0 && k < 4 && words.get_ready) {
      const s = pop(k * 0.5, 0.4) * (1.0 + 0.05 * kick(beat));
      const img = words.get_ready;
      drawC(ctx, img, cx, cy, img.width * s, img.height * s, Math.min(1.0, (4 - k) / 0.4));
    }
    ["count_3", "count_2", "count_1", "count_go"].forEach((name, i) => {
      const b = fm - 4 + i;
      const img = words[name];
      if (beat - b >= 0 && beat - b < 1 && img) {
        const s = pop((beat - b) * bs, 0.22, 1.18, 1.35); // 1.6 reached up into the PiP
        const fade = beat - b < 0.6 ? 1.0 : Math.max(0.0, (1 - (beat - b)) / 0.4);
        drawC(ctx, img, cx, COUNT_Y, img.width * s, img.height * s, fade);
      }
    });
    if (beat >= fm - 7.5 && song) { // small title, top-centre
      const line = `${song.title}  -  ${song.artist ?? ""}`.replace(/[\s-]+$/, "");
      centredText(ctx, line, 640, 40, 0.55, DIM, { body: true });
    }
  }

  /** The difficulty's badge (cached canvas), or null for a game without difficulties. */
  badge(difficulty, scale = 0.62) {
    const key = difficulty != null ? String(difficulty).toLowerCase() : "";
    if (!Object.hasOwn(DIFFICULTY_LABELS, key)) return null;
    const k = `${key}/${scale}`;
    if (!this._badges.has(k)) {
      this._badges.set(k, makeBadge(this.mk, DIFFICULTY_LABELS[key], this.difficultyColours[key], scale));
    }
    return this._badges.get(k);
  }

  _title(song) {
    if (!song) return null;
    const key = `${song.id}|${song.title}|${song.artist ?? ""}`;
    let hit = this._titleCache.get(key);
    if (!hit) {
      let t = makeGlowText(this.mk, String(song.title), 2.4, this.cGold, this.cMagenta);
      if (t.width > 560) { // long titles shrink to fit the right side of the stage
        const s = this.mk(560, Math.round(t.height * 560 / t.width));
        s.getContext("2d").drawImage(t, 0, 0, s.width, s.height);
        t = s;
      }
      const artist = String(song.artist ?? "");
      hit = [t, artist ? makeGlowText(this.mk, artist, 0.95, this.cCyan) : null];
      this._titleCache.set(key, hit);
    }
    return hit;
  }

  _drawOutroFade(ctx) {
    if (this._outroT0 == null) return;
    const u = Math.min(1.0, Math.max(0.0, (this._clock - this._outroT0 - 2.0) / 3.0));
    if (u > 0.01) {
      ctx.fillStyle = `rgba(0,0,0,${0.45 * u})`;
      ctx.fillRect(0, 0, this.width, this.height);
    }
  }

  _drawHud(ctx, game, phase) {
    const score = Math.trunc(Number(game.score) || 0);
    const shown = Math.round(Number(game.displayScore ?? score) || 0);
    if (this._scoreSeen == null) this._scoreSeen = score;
    if (score !== this._scoreSeen) {
      this._scoreSeen = score;
      this._scoreT = this._clock;
    }
    const pulse = Math.max(0.0, 1.0 - (this._clock - this._scoreT) / 0.35);
    const colour = ACCENT.map((g) => Math.round(g + (255 - g) * pulse * 0.7));
    text(ctx, String(shown), SCORE_ORG[0], SCORE_ORG[1], 1.5 * (1.0 + 0.18 * pulse), colour);
    const combo = Number(game.combo) || 0;
    if (combo >= 3) {
      const cp = Math.max(0.0, 1.0 - (this._clock - this._comboT) / 0.3);
      text(ctx, `x${combo} COMBO`, SCORE_ORG[0], SCORE_ORG[1] + 44, 0.85 * (1 + 0.2 * cp), CYAN);
    }
    const badge = this.badge(game.difficulty, 0.6);
    if (badge) ctx.drawImage(badge, HUD_BADGE_ORG[0], HUD_BADGE_ORG[1] - Math.floor(badge.height / 2));
    this._drawStarBar(ctx, Number(game.stars) || 0);
    if (game.notice) {
      const str = String(game.notice);
      const tw = textWidth(ctx, str, 0.9), th = fontPx(0.9) * 0.72;
      const cx = 600, cy = phase === "intro" ? 108 : 96;
      panel(ctx, [cx - tw / 2 - 22, cy - th - 16, cx + tw / 2 + 22, cy + 14], { alpha: 0.82, border: WARN, radius: 14 });
      centredText(ctx, str, cx, cy, 0.9, WARN);
    }
  }

  _drawStarBar(ctx, stars) {
    const [x0, y0, x1, y1] = STAR_BAR;
    const sb = this._starBar;
    ctx.drawImage(sb.track, x0 - 6, y0 - 6);
    stars = Math.min(5.0, Math.max(0.0, stars));
    const h = Math.round((y1 - y0) * stars / 5.0);
    if (h > 0) {
      ctx.drawImage(sb.fill, 0, (y1 - y0) - h, x1 - x0, h, x0, y1 - h, x1 - x0, h);
      strokeLine(ctx, [[x0, y1 - h], [x1 - 1, y1 - h]], WHITE, 2);
    }
    const cx = (x0 + x1) / 2;
    const full = this.art.starFull, empty = this.art.starEmpty;
    for (let k = 1; k <= 5; k++) {
      const y = y1 - (y1 - y0) * k / 5.0;
      if (stars >= k - 1e-6) {
        if (!this._starsLit.has(k)) this._starsLit.set(k, this._clock);
        const age = this._clock - this._starsLit.get(k);
        const s = pop(age, 0.35, 1.5, 0.6) * 0.64;
        if (age < 0.6) {
          const sp = this._sparkleGold[2];
          addC(ctx, sp, cx, y, sp.width, sp.height, 1.0 - age / 0.6);
        }
        drawC(ctx, full, cx, y, full.width * s, full.height * s);
      } else {
        this._starsLit.delete(k);
        drawC(ctx, empty, cx, y, 40, 40);
      }
    }
  }

  _drawPip(ctx, frame, game) {
    if (!frame) return;
    const [x0, y0, x1, y1] = PIP_RECT;
    const key = ratingKey(game.lastRating);
    let col = key ? this.art.ratingColour(key) : bgr([235, 235, 235]);
    if (key && col.every((v) => v === 255)) col = RATING_FALLBACK[key];
    ctx.save();
    ctx.beginPath();
    roundRect(ctx, x0, y0, x1, y1, PIP_RADIUS);
    ctx.clip();
    ctx.drawImage(frame, x0, y0, x1 - x0, y1 - y0);
    if (game.player) drawMeasured(ctx, game.player, (x1 - x0) / this.width, [x0, y0], col);
    ctx.restore();
    ctx.save();
    ctx.beginPath();
    roundRect(ctx, x0, y0, x1, y1, PIP_RADIUS);
    ctx.strokeStyle = css(col);
    ctx.lineWidth = 3;
    ctx.stroke();
    ctx.restore();
    text(ctx, "YOU", x0 + 14, y1 - 14, 0.6, WHITE, { body: true });
  }

  _drawResults(ctx, game, song) {
    // Everything has landed by ~1.25 s: the shell covers this screen with its
    // name entry after ENTRY_DELAY (1.6 s) and stops updating the game, so a
    // slower count-up froze half-way behind it (QA: "9419" next to a 9463 board).
    const t = Number(game.resultsT) || 0;
    const [x0, y0, x1, y1] = RESULTS_RECT;
    const a = easeOut(t / 0.4);
    panel(ctx, [x0, y0 + (1 - a) * 30, x1, y1 + (1 - a) * 30], { alpha: 0.86 * a,
      border: a > 0.6 ? ACCENT : null, radius: 22, thickness: 3 });
    if (a < 0.6) return;
    const cx = (x0 + x1) / 2;
    const words = this.art.words;
    const badge = this.badge(game.difficulty, 0.65);
    if (!badge) centredText(ctx, "RESULTS", cx, y0 + 40, 0.6, DIM, { body: true });
    else { // "RESULTS [EASY]", centred as a pair
      const tw = textWidth(ctx, "RESULTS", 0.6, { body: true });
      const left = cx - (tw + 12 + badge.width) / 2;
      centredText(ctx, "RESULTS", left + tw / 2, y0 + 40, 0.6, DIM, { body: true });
      ctx.drawImage(badge, Math.round(left + tw + 12), Math.round(y0 + 34 - badge.height / 2));
    }
    centredText(ctx, String(song?.title ?? "").toUpperCase(), cx, y0 + 80, 1.05, ACCENT);
    const stars = Math.floor((Number(game.stars) || 0) + 1e-6);
    const full = this.art.starFull, empty = this.art.starEmpty;
    for (let k = 0; k < 5; k++) {
      const sx = cx + (k - 2) * 84, sy = y0 + 128;
      const appear = 0.35 + 0.18 * k;
      if (k < stars && t >= appear) {
        const age = t - appear;
        const s = pop(age, 0.35, 1.45, 0.5) * 1.15;
        drawC(ctx, full, sx, sy, full.width * s, full.height * s);
        if (age < 0.6) {
          const sp = this._sparkleGold[3];
          addC(ctx, sp, sx, sy, sp.width, sp.height, 1.0 - age / 0.6);
        }
      } else {
        drawC(ctx, empty, sx, sy, 64, 64);
      }
    }
    const score = Math.trunc(Number(game.score) || 0);
    const shownScore = Math.round(score * easeOut((t - 0.2) / 1.0));
    centredText(ctx, String(shownScore), cx, y0 + 222, 2.1, ACCENT);
    if (score >= 9600 && t > 1.25 && words.superstar) {
      const s = pop(t - 1.25) * 0.85, img = words.superstar;
      drawC(ctx, img, cx, y0 + 266, img.width * s, img.height * s);
    }
    const counts = game.counts ?? {};
    const rowY = y0 + 318;
    RATING_KEYS.forEach((key, i) => {
      const appear = 0.55 + 0.1 * i;
      if (t < appear) return;
      const fa = Math.min(1.0, (t - appear) / 0.25);
      const dx = (1 - easeOut((t - appear) / 0.3)) * 40;
      const y = rowY + i * 36;
      const img = words[key];
      if (img) drawC(ctx, img, x0 + 130 + dx, y, img.width * 0.5, img.height * 0.5, fa);
      const txt = String(Math.trunc(Number(counts[key]) || 0));
      const tw = textWidth(ctx, txt, 0.85);
      text(ctx, txt, x1 - 60 - tw + dx, y + 12, 0.85, WHITE, { alpha: fa });
      if (fa > 0.5) { // a dotted leader between word and count
        ctx.fillStyle = css(bgr([150, 130, 160]));
        ctx.beginPath();
        for (let xx = Math.trunc(x0 + 240 + dx); xx < x1 - 90 - tw; xx += 14) {
          ctx.moveTo(xx + 1.2, y + 4);
          ctx.arc(xx, y + 4, 1.2, 0, 2 * Math.PI);
        }
        ctx.fill();
      }
    });
  }
}

/** The player as the judge measured them (pose.Measured), scaled by k into the PiP at `org`. */
function drawMeasured(ctx, m, k, org, colour) {
  const sw = Number(m.shoulderPx);
  const lean = Number(m.lean) * RAD;
  const c = Math.cos(lean), s = Math.sin(lean);
  const rot = (v) => [v[0] * c - v[1] * s, v[0] * s + v[1] * c];
  const [cx, cy] = m.centrePx;
  const P = (v) => [org[0] + (cx + v[0]) * k, org[1] + (cy + v[1]) * k];
  const lines = [];
  const sh = { l: rot([-0.5 * sw, 0.0]), r: rot([0.5 * sw, 0.0]) };
  lines.push([sh.l, sh.r]);
  const head = rot([0.0, -0.55 * sw]);
  for (const side of ["l", "r"]) {
    const a = m.arms?.[side];
    if (!a) continue;
    const p0 = sh[side];
    if (a.upper != null && a.fore != null) {
      const d1 = rot(limbDir(side, a.upper));
      const el = add(p0, d1, UPPER_ARM * sw);
      const d2 = rot(limbDir(side, a.fore));
      lines.push([p0, el], [el, add(el, d2, FOREARM * sw)]);
    } else if (a.reach != null) {
      lines.push([p0, add(p0, rot(limbDir(side, a.reach)), (UPPER_ARM + FOREARM) * 0.92 * sw)]);
    }
  }
  if (m.hasHips) {
    const hc = rot([0.0, TORSO * sw]);
    const hips = { l: [hc[0] - HIP_W / 2 * sw, hc[1]], r: [hc[0] + HIP_W / 2 * sw, hc[1]] };
    lines.push([sh.l, hips.l], [sh.r, hips.r], [hips.l, hips.r]);
    for (const [side, leg] of Object.entries(m.legs ?? {})) {
      const h0 = hips[side];
      if (!h0 || leg.thigh == null || leg.shin == null) continue;
      const kn = add(h0, limbDir(side, leg.thigh), THIGH * sw);
      lines.push([h0, kn], [kn, add(kn, limbDir(side, leg.shin), SHIN * sw)]);
    }
  }
  ctx.lineCap = "round";
  for (const [colourNow, width] of [[bgr([24, 16, 20]), 6], [colour, 3]]) {
    ctx.strokeStyle = css(colourNow);
    ctx.lineWidth = width;
    ctx.beginPath();
    for (const [a, b] of lines) {
      ctx.moveTo(...P(a));
      ctx.lineTo(...P(b));
    }
    ctx.stroke();
  }
  const hr = 0.3 * sw * k;
  for (const [colourNow, width] of [[bgr([24, 16, 20]), 4], [colour, 2]]) {
    ctx.strokeStyle = css(colourNow);
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.arc(...P(head), hr, 0, 2 * Math.PI);
    ctx.stroke();
  }
}
