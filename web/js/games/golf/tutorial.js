// The "how to play" cards shown before a player's first Raccoon Golf round
// (port of src/kpapp/game/golf/tutorial.py; same strings and animation).
//
// Three animated cards, drawn with the shared stick figure (core/figure.js):
// address (hands together, low and still), backswing (the gauge fills with hand
// height, red above the head), and the swing through (with putting as a
// pendulum). Nothing here loads art, so the cards draw before the assets exist.

import { clipTo, drawFigure, ease, lerp, reach } from "../../core/figure.js";
import { ACCENT, CYAN, DANGER, WARN, centredText, css } from "../../core/theme.js";

const GREEN = [70, 170, 88]; // grass under the figure (RGB)
const SHAFT = [230, 220, 215];
const INK = [24, 16, 20];
const GRIP_LOW = [0.0, 0.42]; // figure units: hands together at the waist

function stroke(ctx, pts, colour, width) {
  ctx.save();
  ctx.lineCap = ctx.lineJoin = "round";
  ctx.strokeStyle = css(colour);
  ctx.lineWidth = Math.max(1, width);
  ctx.beginPath();
  pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.stroke();
  ctx.restore();
}

function disc(ctx, [x, y], r, colour) {
  ctx.beginPath();
  ctx.arc(x, y, Math.max(1, r), 0, Math.PI * 2);
  ctx.fillStyle = css(colour);
  ctx.fill();
}

const label = (ctx, s, cx, y, scale, colour, { alpha = 1, body = false } = {}) =>
  centredText(ctx, s, cx, y, scale, colour, { alpha: Math.max(0, Math.min(1, alpha)), body });

/** Both arms reaching the same grip point (two-bone IK per arm). */
const armsTo = (grip) => ({ left: reach(grip, "left"), right: reach(grip, "right") });

/** The grip on an arc round the chest: 0 hanging at the waist, + up to the viewer's left. */
const swingGrip = (angle, radius = 0.92) => [-Math.sin(angle) * radius, -0.5 + Math.cos(angle) * radius];

/** A club from the grip, pointing away from the shoulders' centre. */
function club(ctx, cx, cy, s, grip, length = 1.0, colour = SHAFT) {
  const [gx, gy] = grip;
  const dx = gx, dy = gy + 0.5;
  const n = Math.hypot(dx, dy) || 1.0;
  const hx = gx + (dx / n) * length, hy = gy + (dy / n) * length;
  const a = [cx + gx * s, cy + gy * s], b = [cx + hx * s, cy + hy * s];
  stroke(ctx, [a, b], INK, s * 0.07);
  stroke(ctx, [a, b], colour, s * 0.035);
  disc(ctx, b, s * 0.06, [70, 56, 60]);
}

function ground(ctx, [x0, , x1, y1], y) {
  ctx.fillStyle = css(GREEN);
  ctx.fillRect(x0, y, x1 - x0, y1 - y);
}

function drawAddress(ctx, rect, t) {
  const [x0, y0, x1, y1] = rect;
  const w = x1 - x0, h = y1 - y0, s = h * 0.36;
  const cx = x0 + w * 0.42, cy = y0 + h * 0.52;
  ground(ctx, rect, cy + 0.95 * s);
  const u = (t % 3.0) / 3.0;
  // Hands come together at the waist (0..0.35), then hold still (0.35..).
  const k = ease(u / 0.35);
  const gl = [lerp(-0.55, GRIP_LOW[0], k), lerp(0.25, GRIP_LOW[1], k)];
  const gr = [lerp(0.55, GRIP_LOW[0], k), lerp(0.25, GRIP_LOW[1], k)];
  drawFigure(ctx, cx, cy, s, { left: reach(gl, "left"), right: reach(gr, "right") });
  const hold = Math.max(0.0, (u - 0.35) / 0.25);
  if (hold > 0.0 && hold < 1.0) {
    ctx.save();
    ctx.strokeStyle = css(ACCENT);
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(cx + GRIP_LOW[0] * s, cy + GRIP_LOW[1] * s, s * 0.22, -Math.PI / 2, -Math.PI / 2 + 2 * Math.PI * hold);
    ctx.stroke();
    ctx.restore();
  }
  if (hold >= 1.0) {
    club(ctx, cx, cy, s, GRIP_LOW, 0.9);
    label(ctx, "READY", x0 + w * 0.78, y0 + h * 0.3, 0.75, CYAN, { alpha: (u - 0.6) * 6 });
  } else {
    label(ctx, "HOLD STILL", x0 + w * 0.76, y0 + h * 0.3, 0.55, WARN, { alpha: u * 4, body: true });
  }
}

/** A small power gauge: fill to `power` (1.15 = top), red zone above 100%. */
function gauge(ctx, x, y0, y1, power) {
  const full = y1 - (y1 - y0) / 1.15;
  ctx.save();
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = 2;
  ctx.strokeRect(x - 9, y0 - 3, 18, y1 - y0 + 6);
  ctx.fillStyle = css([150, 40, 50]);
  ctx.fillRect(x - 7, y0, 14, full - y0);
  const top = y1 - ((y1 - y0) * Math.min(1.15, power)) / 1.15;
  ctx.fillStyle = css(power > 1.0 ? DANGER : power > 0.8 ? ACCENT : CYAN);
  ctx.fillRect(x - 7, top, 14, y1 - top);
  ctx.beginPath();
  ctx.moveTo(x - 12, full);
  ctx.lineTo(x + 12, full);
  ctx.stroke();
  ctx.restore();
}

function drawBackswing(ctx, rect, t) {
  const [x0, y0, x1, y1] = rect;
  const w = x1 - x0, h = y1 - y0, s = h * 0.36;
  const cx = x0 + w * 0.42, cy = y0 + h * 0.54;
  ground(ctx, rect, cy + 0.95 * s);
  const u = (t % 3.2) / 3.2;
  // Two backswings: a full one to head height, then an overswing above it.
  const big = u >= 0.5;
  const v = (u % 0.5) / 0.5;
  const top = big ? 2.45 : 2.05;
  const a = top * (v < 0.75 ? ease(v / 0.55) : 1 - ease((v - 0.75) / 0.25));
  const grip = swingGrip(a);
  drawFigure(ctx, cx, cy, s, armsTo(grip));
  club(ctx, cx, cy, s, grip, 0.9);
  const headY = -0.86;
  const power = Math.min(1.15, Math.max(0.0, (GRIP_LOW[1] - grip[1]) / (GRIP_LOW[1] - headY)));
  gauge(ctx, x0 + w * 0.86, y0 + h * 0.12, y0 + h * 0.84, power);
  if (power > 1.0) label(ctx, "OVERSWING", x0 + w * 0.66, y0 + h * 0.13, 0.5, DANGER, { body: true });
  else if (power > 0.85) label(ctx, "FULL POWER", x0 + w * 0.64, y0 + h * 0.13, 0.5, ACCENT, { body: true });
}

function drawSwing(ctx, rect, t) {
  const [x0, y0, x1, y1] = rect;
  const w = x1 - x0, h = y1 - y0, s = h * 0.34;
  const u = (t % 3.4) / 3.4;
  if (u < 0.55) {
    const cx = x0 + w * 0.36, cy = y0 + h * 0.54;
    ground(ctx, rect, cy + 0.95 * s);
    const v = u / 0.55;
    // Hold at the top, then a fast swing down through the ball to a high finish.
    let a;
    if (v < 0.3) a = 2.05;
    else {
      const k = Math.min(1.0, (v - 0.3) / 0.25);
      a = 2.05 + (-2.0 - 2.05) * (k * k * (3 - 2 * k));
    }
    const grip = swingGrip(a);
    drawFigure(ctx, cx, cy, s, armsTo(grip));
    club(ctx, cx, cy, s, grip, 0.9);
    const impact = 0.3 + 0.25 * 0.5;
    const bx = cx + 0.95 * s, by = cy + 0.9 * s;
    if (v < impact) disc(ctx, [bx, by], s * 0.06, [255, 255, 255]);
    else {
      const f = (v - impact) / (1 - impact);
      const px = bx + f * w * 0.55;
      const py = by - Math.sin(Math.min(1.0, f * 1.2) * Math.PI) * h * 0.45;
      disc(ctx, [px, py], s * 0.06, [255, 255, 255]);
      label(ctx, "NICE SHOT!", x0 + w * 0.62, y0 + h * 0.16, 0.65, ACCENT, { alpha: f * 4 });
    }
  } else {
    const cx = x0 + w * 0.42, cy = y0 + h * 0.5;
    ground(ctx, rect, cy + 0.95 * s);
    const v = (u - 0.55) / 0.45;
    const swing = Math.sin(v * 2 * Math.PI) * 0.42;
    const grip = [GRIP_LOW[0] + swing, GRIP_LOW[1] - Math.abs(swing) * 0.1];
    drawFigure(ctx, cx, cy, s, armsTo(grip));
    club(ctx, cx, cy, s, grip, 0.55);
    label(ctx, "PUTT: SIDE TO SIDE", x0 + w * 0.5, y0 + h * 0.13, 0.5, CYAN, { body: true });
  }
}

/** A card's demo, clipped to its rect (like figure.demo in the Python). */
const demo = (fn) => (ctx, rect, t) => {
  const restore = clipTo(ctx, rect);
  try {
    fn(ctx, rect, t);
  } finally {
    restore();
  }
};

export const GOLF_TUTORIAL = [
  {
    title: "HANDS TOGETHER, LOW",
    text: "Stand back so your hands show. Hold them together at your waist, still, until READY.",
    draw: demo(drawAddress),
  },
  {
    title: "SWING BACK FOR POWER",
    text: "Lift your hands up to one side. Head height is full power - higher overswings and curves.",
    draw: demo(drawBackswing),
  },
  {
    title: "SWING THROUGH TO HIT",
    text: "Swing down fast across your body. On the green, rock your hands side to side to putt.",
    draw: demo(drawSwing),
  },
];
