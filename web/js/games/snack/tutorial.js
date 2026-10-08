// The "how to play" cards shown before a player's first Snack Attack round.
// Each card animates a stick figure with the game's own sprites (art cached by
// preload); drawing happens only when the shell calls draw().

import { cachedArt } from "./art.js";
import { drawFigure, drawStandBack, drawCross, handPos, reach, mirror, ease, lerp, clipTo, DOWN } from "../../core/figure.js";
import { blit } from "../../core/gfx.js";
import { centredText, ACCENT } from "../../core/theme.js";

const REST_LEFT = [mirror(DOWN - 0.25), mirror(DOWN - 0.1)];

/** The right arm's swipe: a diagonal slash out to the side, high to low. */
function swing(u) {
  return reach([lerp(0.42, 1.0, u), lerp(-1.12, -0.02, u)]);
}

function drawSwipe(ctx, [x0, y0, x1, y1], t) {
  const art = cachedArt();
  const w = x1 - x0, h = y1 - y0;
  const s = h * 0.4, cx = x0 + w * 0.36, cy = y0 + h * 0.6;
  const P = ([x, y]) => [cx + x * s, cy + y * s];
  const phase = t % 2.2;
  const u = ease((phase - 0.35) / 0.75);
  const restore = clipTo(ctx, [x0, y0, x1, y1]);

  // The snack waits at the top of the swing; once the hand has passed it, it
  // breaks in two and falls, with a +1.
  const snackAt = P(handPos(swing(0.5)));
  const eaten = u > 0.5;
  const since = eaten ? phase - (0.35 + 0.75 * 0.5) : 0;
  const fruit = art?.fruits?.[3] ?? art?.fruits?.[0];
  if (fruit) {
    const size = s * 0.62;
    if (!eaten) blit(ctx, fruit.whole, snackAt[0], snackAt[1], { w: size, h: (size * fruit.whole.height) / fruit.whole.width, angle: Math.sin(t * 2) * 0.2 });
    else {
      const fall = 300 * since * since;
      for (const [img, dx] of [[fruit.halfA, -1], [fruit.halfB, 1]]) {
        blit(ctx, img, snackAt[0] + dx * 70 * since, snackAt[1] + fall - 40 * since, {
          w: size, h: (size * img.height) / img.width, angle: dx * since * 3, alpha: Math.max(0, 1 - since / 1.2),
        });
      }
      centredText(ctx, "+1", snackAt[0], snackAt[1] - 30 - 40 * since, 0.8, ACCENT, { alpha: Math.max(0, 1 - since / 1.0) });
    }
  }

  // Purple trail behind the hand while it moves fast.
  ctx.save();
  ctx.lineCap = ctx.lineJoin = "round";
  for (let k = 7; k > 0; k--) {
    const ua = ease((phase - 0.35 - k * 0.035) / 0.75), ub = ease((phase - 0.35 - (k - 1) * 0.035) / 0.75);
    if (ub - ua < 0.004) continue;
    const a = P(handPos(swing(ua))), b = P(handPos(swing(ub)));
    ctx.strokeStyle = `rgba(190,110,255,${0.75 * (1 - k / 8)})`;
    ctx.lineWidth = s * 0.16 * (1 - k / 9);
    ctx.beginPath();
    ctx.moveTo(...a);
    ctx.lineTo(...b);
    ctx.stroke();
  }
  ctx.restore();

  const { rightHand } = drawFigure(ctx, cx, cy, s, { left: REST_LEFT, right: swing(u) });
  const head = eaten && since < 0.35 ? art?.cursorChomp : art?.cursorIdle;
  if (head) blit(ctx, head, rightHand[0], rightHand[1], { w: s * 0.62, h: s * 0.62 });
  restore();
}

function drawTrash(ctx, [x0, y0, x1, y1], t) {
  const art = cachedArt();
  const w = x1 - x0, h = y1 - y0;
  const restore = clipTo(ctx, [x0, y0, x1, y1]);
  const bx = x0 + w * 0.32, by = y0 + h * 0.46 + Math.sin(t * 2.4) * 8;
  if (art?.bomb) blit(ctx, art.bomb, bx, by, { w: h * 0.5, h: (h * 0.5 * art.bomb.height) / art.bomb.width, angle: Math.sin(t * 1.7) * 0.15 });
  drawCross(ctx, bx, by, h * 0.2);
  // Lives: a dropped snack costs one; the third one blinks out.
  const lx = x0 + w * 0.72, ly = y0 + h * 0.5, size = h * 0.17;
  const lost = Math.floor((t % 3) / 1) ; // 0, 1, 2 lives lost, looping
  for (let i = 0; i < 3; i++) {
    const img = i >= 3 - lost ? art?.lifeLost : art?.lifeFull;
    if (img) blit(ctx, img, lx + (i - 1) * size * 1.15, ly, { w: size, h: size });
  }
  centredText(ctx, "x3", lx, ly + size * 1.4, 0.6, [235, 230, 245], { body: true, weight: 700 });
  restore();
}

export const SNACK_TUTORIAL = [
  {
    title: "STAND BACK",
    text: "About 2 m from the screen, with your head, shoulders and hands in the picture.",
    draw: (ctx, rect, t) => drawStandBack(ctx, rect, t),
  },
  {
    title: "YOUR HAND IS THE RACCOON",
    text: "Swipe your right hand fast through the snacks to gobble them. A slow hand won't bite.",
    draw: drawSwipe,
  },
  {
    title: "AVOID THE TRASH",
    text: "Grabbing a bag of trash ends the game. Let three snacks fall and it's over too.",
    draw: drawTrash,
  },
];
