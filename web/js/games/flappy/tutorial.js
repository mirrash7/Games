// The "how to play" cards shown before a player's first Flappy Raccoon run.
// Drawing happens only when the shell calls draw().

import { cachedArt } from "./art.js";
import { drawFigure, drawStandBack, mirror, ease, lerp, clipTo, DOWN } from "../../core/figure.js";
import { blit } from "../../core/gfx.js";
import { centredText, ACCENT, CYAN } from "../../core/theme.js";

const ARM_DOWN = [DOWN - 0.3, DOWN - 0.15];
const ARM_UP = [-0.45, -1.15];

/** Arms over one 1.1 s wing-beat: rise slowly, beat down fast, rest. Returns 0 (down) .. 1 (up). */
function wingbeat(phase) {
  if (phase < 0.55) return ease(phase / 0.55);
  if (phase < 0.72) return 1 - ease((phase - 0.55) / 0.17);
  return 0;
}

const armAt = (up) => [lerp(ARM_DOWN[0], ARM_UP[0], up), lerp(ARM_DOWN[1], ARM_UP[1], up)];

function bird(art, phase) {
  const frames = art?.birdFrames;
  if (!frames?.length) return null;
  return frames[Math.floor(phase * 12) % frames.length];
}

function drawFlap(ctx, [x0, y0, x1, y1], t) {
  const art = cachedArt();
  const w = x1 - x0, h = y1 - y0;
  const restore = clipTo(ctx, [x0, y0, x1, y1]);
  const period = 1.1;
  const phase = (t % period) / period;
  const up = wingbeat(phase);
  const s = h * 0.33;
  const right = armAt(up);
  drawFigure(ctx, x0 + w * 0.33, y0 + h * 0.62, s, { right, left: [mirror(right[0]), mirror(right[1])] });

  // One hop per downstroke: the flap fires part-way down (phase ~0.62).
  const tau = ((phase - 0.62 + 1) % 1) * period;
  const hop = Math.max(0, 230 * tau - 0.5 * 420 * tau * tau);
  const img = bird(art, phase);
  if (img) blit(ctx, img, x0 + w * 0.76, y0 + h * 0.72 - hop * (h / 230), { w: h * 0.34, h: (h * 0.34 * img.height) / img.width, angle: tau < 0.25 ? -0.25 : 0.15 });
  if (tau < 0.3) centredText(ctx, "FLAP!", x0 + w * 0.33, y0 + h * 0.16, 0.7, CYAN, { alpha: 1 - tau / 0.3 });
  restore();
}

function drawPipes(ctx, [x0, y0, x1, y1], t) {
  const art = cachedArt();
  const w = x1 - x0, h = y1 - y0;
  const restore = clipTo(ctx, [x0, y0, x1, y1]);
  ctx.fillStyle = "rgb(112,197,232)";
  ctx.fillRect(x0, y0, w, h);
  const pw = 46, gap = h * 0.42, spacing = w * 0.55, speed = 90;
  const gapY = (i) => y0 + h * (0.4 + 0.18 * Math.sin(i * 2.1));
  const offset = (t * speed) % spacing;
  const first = Math.floor((t * speed) / spacing);
  const birdX = x0 + w * 0.3;
  let birdY = y0 + h * 0.5;
  for (let k = -1; k < 4; k++) {
    const i = first + k;
    const px = x0 + w * 0.65 + k * spacing - offset;
    const gy = gapY(i);
    // The bird steers for the gap of the next pipe ahead of it.
    if (px + pw / 2 > birdX - 30 && px - spacing + pw / 2 <= birdX - 30) {
      const prevY = gapY(i - 1);
      const u = 1 - (px - (birdX - 30)) / spacing;
      birdY = lerp(prevY, gy, ease(Math.min(1, Math.max(0, u * 1.6))));
    }
    if (art?.pipeBody && art?.pipeCap) {
      const capH = (pw * 1.18 * art.pipeCap.height) / art.pipeCap.width;
      ctx.drawImage(art.pipeBody, px - pw / 2, y0, pw, gy - gap / 2 - y0);
      ctx.drawImage(art.pipeBody, px - pw / 2, gy + gap / 2, pw, y1 - gy - gap / 2);
      ctx.drawImage(art.pipeCap, px - pw * 0.59, gy - gap / 2 - capH, pw * 1.18, capH);
      ctx.drawImage(art.pipeCap, px - pw * 0.59, gy + gap / 2, pw * 1.18, capH);
    }
    // +1 as each pipe goes past the bird.
    const past = (birdX - px) / speed;
    if (past > 0 && past < 0.7) centredText(ctx, "+1", birdX, birdY - 40 - past * 50, 0.7, ACCENT, { alpha: 1 - past / 0.7 });
  }
  const img = bird(art, (t % 1.1) / 1.1);
  if (img) blit(ctx, img, birdX, birdY, { w: 54, h: (54 * img.height) / img.width });
  restore();
}

export const FLAPPY_TUTORIAL = [
  {
    title: "STAND BACK",
    text: "Far enough that both arms stay in the picture, even raised above your head.",
    draw: (ctx, rect, t) => drawStandBack(ctx, rect, t, { armsUp: true }),
  },
  {
    title: "FLAP TO FLY",
    text: "Raise both arms, then beat them down like wings. Every flap is one hop.",
    draw: drawFlap,
  },
  {
    title: "FLY THROUGH THE GAPS",
    text: "Each pipe you pass is a point. Hitting a pipe or the ground ends the run.",
    draw: drawPipes,
  },
];
