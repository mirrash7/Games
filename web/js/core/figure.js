// A simple stick figure for tutorial demos: drawn procedurally so it scales
// to any card and animates from a couple of joint angles.
//
// Units: 1 = roughly a shoulder width. Origin at the chest. Angles are screen
// angles in radians (0 = pointing right, PI/2 = straight down). "left"/"right"
// are the viewer's sides - on the mirrored camera feed the player's right hand
// is on the viewer's right, exactly as in a mirror.

export const DOWN = Math.PI / 2;
const UPPER = 0.36, FORE = 0.32;

const at = (o, a, len) => [o[0] + Math.cos(a) * len, o[1] + Math.sin(a) * len];

/** Hand position in figure units for a right ("right") or left arm. */
export function handPos([a1, a2], side = "right") {
  const sh = side === "right" ? [0.3, -0.5] : [-0.3, -0.5];
  return at(at(at(sh, a1, UPPER), a2, FORE), a2, 0.1);
}

/**
 * Arm angles that put the wrist at `target` (figure units), elbow low -
 * two-bone IK, for demos that move the hand along a path.
 */
export function reach(target, side = "right") {
  const sh = side === "right" ? [0.3, -0.5] : [-0.3, -0.5];
  const dx = target[0] - sh[0], dy = target[1] - sh[1];
  const d = Math.min(Math.max(Math.hypot(dx, dy), Math.abs(UPPER - FORE) + 1e-3), UPPER + FORE - 1e-3);
  const phi = Math.atan2(dy, dx);
  const alpha = Math.acos((UPPER * UPPER + d * d - FORE * FORE) / (2 * UPPER * d));
  const best = [phi + alpha, phi - alpha]
    .map((a1) => ({ a1, e: at(sh, a1, UPPER) }))
    .sort((p, q) => q.e[1] - p.e[1])[0];
  const w = at(sh, phi, d);
  return [best.a1, Math.atan2(w[1] - best.e[1], w[0] - best.e[0])];
}

/** Mirror a right-arm angle to the left side. */
export const mirror = (a) => Math.PI - a;

/**
 * Draw the figure centred at (cx, cy) with `s` px per unit. Arms are
 * [upperAngle, foreAngle] for each side. Returns the hands' pixel positions.
 */
export function drawFigure(ctx, cx, cy, s, { left = [DOWN - 0.2, DOWN - 0.1], right = [DOWN + 0.2, DOWN + 0.1], colour = "rgb(236,232,246)", alpha = 1 } = {}) {
  const P = (p) => [cx + p[0] * s, cy + p[1] * s];
  const ls = [-0.3, -0.5], rs = [0.3, -0.5];
  const le = at(ls, left[0], UPPER), lw = at(le, left[1], FORE);
  const re = at(rs, right[0], UPPER), rw = at(re, right[1], FORE);
  ctx.save();
  ctx.globalAlpha *= alpha;
  ctx.lineCap = ctx.lineJoin = "round";
  const limb = (pts, w) => {
    ctx.beginPath();
    pts.map(P).forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    for (const [c, k] of [["rgba(10,8,16,0.9)", 1.7], [colour, 1]]) {
      ctx.strokeStyle = c;
      ctx.lineWidth = w * s * k;
      ctx.stroke();
    }
  };
  // Torso and legs first, so the arms read in front of them.
  limb([[-0.2, 0.95], [-0.17, 0.3], [0.17, 0.3], [0.2, 0.95]], 0.12);
  limb([ls, [-0.17, 0.3], [0.17, 0.3], rs, ls], 0.12);
  limb([[0, -0.52], [0, -0.66]], 0.12);
  ctx.beginPath();
  const [hx, hy] = P([0, -0.86]);
  ctx.arc(hx, hy, 0.19 * s, 0, Math.PI * 2);
  ctx.fillStyle = colour;
  ctx.fill();
  ctx.lineWidth = 0.07 * s;
  ctx.strokeStyle = "rgba(10,8,16,0.9)";
  ctx.stroke();
  limb([ls, le, lw], 0.11);
  limb([rs, re, rw], 0.11);
  ctx.restore();
  return { leftHand: P(at(lw, left[1], 0.1)), rightHand: P(at(rw, right[1], 0.1)) };
}

export const ease = (u) => 0.5 - 0.5 * Math.cos(Math.PI * Math.min(1, Math.max(0, u)));
export const lerp = (a, b, u) => a + (b - a) * u;

/** Clip drawing to a rounded card area; returns a restore function. */
export function clipTo(ctx, [x0, y0, x1, y1], r = 14) {
  ctx.save();
  ctx.beginPath();
  ctx.roundRect ? ctx.roundRect(x0, y0, x1 - x0, y1 - y0, r) : ctx.rect(x0, y0, x1 - x0, y1 - y0);
  ctx.clip();
  return () => ctx.restore();
}

/** The "stand back" demo shared by every game: too close (cropped), then in frame. */
export function drawStandBack(ctx, rect, t, { armsUp = false } = {}) {
  const [x0, y0, x1, y1] = rect;
  const w = x1 - x0, h = y1 - y0;
  const u = ease(((t % 3.2) - 0.4) / 1.4);
  const s = lerp(h * 0.95, h * 0.33, u);
  const fx = x0 + w * 0.2, fy = y0 + h * 0.08, fw = w * 0.6, fh = h * 0.84;
  const restore = clipTo(ctx, [fx, fy, fx + fw, fy + fh], 8);
  ctx.fillStyle = "rgba(40,34,58,0.9)";
  ctx.fillRect(fx, fy, fw, fh);
  const arms = armsUp
    ? { left: [mirror(-0.5), mirror(-1.3)], right: [-0.5, -1.3] }
    : { left: [mirror(0.9), mirror(-1.2)], right: [0.9, -1.2] };
  drawFigure(ctx, fx + fw / 2, fy + fh * 0.42 + s * 0.12, s, arms);
  restore();
  ctx.save();
  ctx.setLineDash([10, 7]);
  ctx.lineWidth = 3;
  const fits = u > 0.98;
  ctx.strokeStyle = fits ? "rgb(150,235,140)" : "rgb(255,190,90)";
  ctx.strokeRect(fx, fy, fw, fh);
  ctx.restore();
  if (fits) {
    ctx.save();
    ctx.lineWidth = 7;
    ctx.lineCap = ctx.lineJoin = "round";
    ctx.strokeStyle = "rgb(150,235,140)";
    const cx = fx + fw + 26, cy = fy + 26;
    ctx.beginPath();
    ctx.moveTo(cx - 13, cy);
    ctx.lineTo(cx - 3, cy + 11);
    ctx.lineTo(cx + 15, cy - 12);
    ctx.stroke();
    ctx.restore();
  }
}

/** A bold red X, for "don't". */
export function drawCross(ctx, cx, cy, r) {
  ctx.save();
  ctx.lineCap = "round";
  for (const [c, wdt] of [["rgba(10,8,16,0.9)", r * 0.42], ["rgb(255,70,70)", r * 0.26]]) {
    ctx.strokeStyle = c;
    ctx.lineWidth = wdt;
    ctx.beginPath();
    ctx.moveTo(cx - r, cy - r);
    ctx.lineTo(cx + r, cy + r);
    ctx.moveTo(cx + r, cy - r);
    ctx.lineTo(cx - r, cy + r);
    ctx.stroke();
  }
  ctx.restore();
}
