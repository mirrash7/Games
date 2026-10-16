// The "how to play" cards shown before a player's first Raccoon Dance song
// (port of src/kpapp/game/dance/tutorial.py).
//
// Three cards in the house style (core/figure.js; the shell shows them side by
// side): stand back (feet in the picture too), copy the coach like a mirror,
// hit the poses on the beat, legs included. The second and third draw the real
// coach and pictogram cards from the game's renderer, so the tutorial teaches
// what the player will actually see: the lane's cards include leg moves (white
// legs on a floor line). Without the art they fall back to the plain figure.
// Drawing happens only when the shell calls draw().

import { cachedArt, bgr } from "./art.js";
import { CoachPainter, PictogramCards, drawPictogramFigure } from "./render.js";
import { REST } from "./pose.js";
import { clipTo, drawFigure, drawStandBack, ease, lerp, reach } from "../../core/figure.js";
import { makeCanvas } from "../../core/gfx.js";
import { ACCENT, CYAN, DIM, centredText, css } from "../../core/theme.js";

const STAGE_BG = bgr([40, 16, 30]); // the stage's dark violet
const FLOOR = bgr([90, 40, 110]);
const COACH_ARM_DOWN = [14.0, 8.0], COACH_ARM_UP = [158.0, 172.0];
const PLAYER_REST = [0.42, 0.3], PLAYER_UP = [0.62, -1.15];

const painters = new Map();
let cards = null;
const rings = new Map();

function painter(art, scale) {
  const key = Math.round(scale);
  // Flat, as before the stage coach got its lighting: the cards are small instructions.
  if (!painters.has(key)) painters.set(key, new CoachPainter(art, "neon", { scale: key, shade: false }));
  return painters.get(key);
}

function ring(art, size, gold) {
  const key = `${size}/${gold}`;
  if (!rings.has(key)) {
    const c = makeCanvas(size, size);
    const g = c.getContext("2d");
    g.drawImage(art.ring, 0, 0, size, size);
    g.globalCompositeOperation = "source-in";
    g.fillStyle = css(art.manifest.colours?.[gold ? "gold" : "cyan"] ?? (gold ? bgr([70, 205, 255]) : bgr([255, 235, 70])));
    g.fillRect(0, 0, size, size);
    rings.set(key, c);
  }
  return rings.get(key);
}

function drawWord(ctx, art, name, cx, cy, scale, fade) {
  const img = art?.words?.[name];
  if (!img || fade <= 0.004) return false;
  ctx.save();
  ctx.globalAlpha *= Math.min(1, fade);
  ctx.drawImage(img, cx - img.width * scale / 2, cy - img.height * scale / 2, img.width * scale, img.height * scale);
  ctx.restore();
  return true;
}

/** 0 (down) .. 1 (up) over a 2.8 s loop: rise, hold, drop. */
function armUp(t, delay = 0.0) {
  const p = (((t - delay) % 2.8) + 2.8) % 2.8;
  if (p < 0.25) return 0.0;
  if (p < 0.8) return ease((p - 0.25) / 0.55);
  if (p < 1.9) return 1.0;
  if (p < 2.4) return 1.0 - ease((p - 1.9) / 0.5);
  return 0.0;
}

function pop(age) {
  if (age < 0.1) return 0.5 + 0.8 * age / 0.1;
  if (age < 0.22) return 1.3 - 0.3 * (age - 0.1) / 0.12;
  return 1.0;
}

const lerp2 = (a, b, u) => [lerp(a[0], b[0], u), lerp(a[1], b[1], u)];

function backdrop(ctx, [x0, y0, x1, y1]) {
  ctx.fillStyle = css(STAGE_BG, 0.92);
  ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
  const fy = y0 + (y1 - y0) * 0.86;
  ctx.strokeStyle = css(FLOOR);
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x0, fy);
  ctx.lineTo(x1, fy);
  ctx.stroke();
}

/** The coach raises its screen-right arm; the player, a beat behind, does the same. */
function drawMirror(ctx, rect, t) {
  const art = cachedArt();
  const [x0, y0, x1, y1] = rect;
  const w = x1 - x0, h = y1 - y0;
  const restore = clipTo(ctx, rect);
  backdrop(ctx, rect);
  const feet = y0 + h * 0.86;
  const uCoach = armUp(t), uYou = armUp(t, 0.3);
  const ccx = x0 + w * 0.27;
  if (art) {
    const p = painter(art, h * 0.175);
    const pose = { ...REST, la: [16.0, 12.0],
      ra: [lerp(COACH_ARM_DOWN[0], COACH_ARM_UP[0], uCoach), lerp(COACH_ARM_DOWN[1], COACH_ARM_UP[1], uCoach)] };
    p.paint(ctx, p.layout(pose, { beat: t * 2.0, origin: p.originFor(ccx, feet) }));
  } else {
    drawFigure(ctx, ccx, feet - h * 0.42, h * 0.38, { right: reach(lerp2(PLAYER_REST, PLAYER_UP, uCoach)),
      colour: "rgb(255,80,200)" });
  }
  centredText(ctx, "COACH", ccx, y1 - 6, 0.48, DIM, { body: true });

  // The mirror: a dashed line between them.
  const mx = x0 + w * 0.5;
  ctx.save();
  ctx.strokeStyle = css(bgr([190, 150, 210]));
  ctx.lineWidth = 2;
  ctx.lineCap = "round";
  ctx.beginPath();
  for (let yy = y0 + 14; yy < feet; yy += 14) {
    ctx.moveTo(mx, yy);
    ctx.lineTo(mx, Math.min(feet, yy + 7));
  }
  ctx.stroke();
  ctx.restore();

  const ycx = x0 + w * 0.73;
  const s = h * 0.29;
  drawFigure(ctx, ycx, feet - s * 0.95, s, { left: reach([-PLAYER_REST[0], PLAYER_REST[1]], "left"),
    right: reach(lerp2(PLAYER_REST, PLAYER_UP, uYou)) });
  centredText(ctx, "YOU", ycx, y1 - 6, 0.48, DIM, { body: true });

  // Both arms up on the same side of the screen: PERFECT.
  const p = (((t - 0.3) % 2.8) + 2.8) % 2.8;
  if (p >= 0.8 && p < 1.9) {
    const age = p - 0.8;
    const fade = age < 0.8 ? 1.0 : Math.max(0.0, 1 - (age - 0.8) / 0.3);
    if (!drawWord(ctx, art, "perfect", mx, y0 + h * 0.11, 0.34 * pop(age), fade)) {
      centredText(ctx, "PERFECT", mx, y0 + h * 0.17, 0.7, ACCENT, { alpha: fade });
    }
  }
  restore();
}

const DEMO_POSES = [
  { la: [90, 90], ra: [90, 90] },
  { ra: [150, 165], la: [20, 10], lean: 8, lf: [0.7, 0], rf: [0.7, 0], shift: 0.32 },
  { la: [170, 175], ra: [170, 175] },
  { la: [90, 170], ra: [90, 170], lf: [0.72, 0], rf: [0.72, 0], squat: 0.6 },
];
const BEAT = 0.6; // seconds per beat in the demo

/** Cards scroll onto the ring; each scores as it lands; the gold one counts double. */
function drawLane(ctx, rect, t) {
  const art = cachedArt();
  const [x0, y0, x1, y1] = rect;
  const w = x1 - x0, h = y1 - y0;
  const restore = clipTo(ctx, rect);
  backdrop(ctx, rect);
  const ly = y0 + h * 0.6, rx = x0 + w * 0.24, spacing = w * 0.3;
  const cardW = Math.round(h * 0.34 / 4) * 4;
  const beat = t / BEAT;
  const kick = Math.exp(-(beat - Math.floor(beat)) * 5);
  ctx.save();
  if (art) {
    const r = ring(art, Math.trunc(cardW * 1.3), false);
    ctx.globalAlpha *= 0.7 + 0.3 * kick;
    ctx.drawImage(r, rx - r.width / 2, ly - r.height / 2);
  } else {
    ctx.strokeStyle = css(CYAN);
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(rx, ly, cardW * 0.62, 0, 2 * Math.PI);
    ctx.stroke();
  }
  ctx.restore();
  // Beat dots under the lane.
  ctx.fillStyle = css(bgr([240, 160, 210]));
  for (let b = Math.floor(beat) - 2; b < Math.floor(beat) + 6; b++) {
    const x = rx + (b - beat) * spacing;
    if (x > x0 + 8 && x < x1 - 8) {
      ctx.beginPath();
      ctx.arc(x, ly + cardW * 0.82, b % 4 === 0 ? 3 : 2, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
  if (art && !cards) cards = new PictogramCards(art);
  const n0 = Math.floor(beat) - 1;
  for (let n = n0 + 5; n >= n0; n--) { // farthest first
    const d = n - beat;
    const x = rx + d * spacing;
    if (x - cardW / 2 > x1 || d < -0.6) continue;
    const gold = ((n % 3) + 3) % 3 === 2;
    const pose = DEMO_POSES[((n % DEMO_POSES.length) + DEMO_POSES.length) % DEMO_POSES.length];
    const fade = d >= 0 ? 1.0 : Math.max(0.0, 1 + d / 0.6);
    const size = cardW + (d >= 0 && d < 1 ? Math.trunc(cardW * 0.25 / 4) * 4 : 0);
    ctx.save();
    ctx.globalAlpha *= fade;
    if (cards) {
      const c = cards.get(pose, gold, size);
      ctx.drawImage(c, x - c.width / 2, ly - c.height / 2);
    } else {
      const half = size / 2;
      ctx.strokeStyle = gold ? css(ACCENT) : "rgb(230,90,255)";
      ctx.lineWidth = 2;
      ctx.strokeRect(x - half, ly - half, size, size);
      drawPictogramFigure(ctx, pose, [x - half * 0.7, ly - half * 0.8, x + half * 0.7, ly + half * 0.8], gold);
    }
    ctx.restore();
    if (gold && d > 0.2) centredText(ctx, "x2", x, ly - size * 0.62, 0.5, ACCENT);
  }
  // The card on the ring scores.
  const last = Math.floor(beat);
  const age = (beat - last) * BEAT;
  const gold = ((last % 3) + 3) % 3 === 2;
  const name = last % 2 === 0 || gold ? "perfect" : "super";
  const fade = age < 0.35 ? 1.0 : Math.max(0.0, 1 - (age - 0.35) / 0.2);
  const wx = rx + w * 0.1, wy = y0 + h * 0.17;
  if (drawWord(ctx, art, name, wx, wy, 0.52 * pop(age), fade)) {
    if (gold) {
      const img = art.words[name];
      centredText(ctx, "x2!", wx + img.width * 0.52 * pop(age) * 0.5 + 24, y0 + h * 0.2, 0.75, ACCENT, { alpha: fade });
    }
  } else {
    centredText(ctx, name.toUpperCase(), rx + w * 0.16, y0 + h * 0.2, 0.6, name === "perfect" ? ACCENT : CYAN,
      { alpha: fade });
  }
  restore();
}

export const DANCE_TUTORIAL = [
  {
    title: "STAND BACK",
    text: "About 2.5-3 m from the screen, so all of you is in the picture: head, both arms, and your feet.",
    draw: (ctx, rect, t) => drawStandBack(ctx, rect, t, { armsUp: true }),
  },
  {
    title: "COPY LIKE A MIRROR",
    text: "The coach is your mirror: when its arm on the right of the screen goes up, raise your right arm.",
    draw: drawMirror,
  },
  {
    title: "HIT THE POSES ON THE BEAT",
    text: "Strike each pose as its card reaches the ring. White legs: step, kick or squat too. Gold counts double.",
    draw: drawLane,
  },
];

