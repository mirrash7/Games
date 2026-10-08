// The arcade's look, shared by every game so they feel like one product
// (port of theme.py + overlay.outlined_text, drawn with Canvas 2D).
//
// The style in short: a dark, translucent rounded panel; bold titles in the
// gold accent with a dark outline; light body text; cyan for active/positive
// feedback; everything sized to read from ~2 m. Colours are [r, g, b] (RGB,
// not OpenCV's BGR); every function here also accepts a CSS colour string.

export const WHITE = [255, 255, 255];
export const DIM = [190, 175, 175]; // secondary text, hints
export const ACCENT = [255, 210, 70]; // gold: titles, highlights, the score
export const CYAN = [110, 220, 255]; // active / hovered / positive feedback
export const PANEL = [22, 24, 28]; // translucent panel fill
export const GOOD = [150, 235, 140]; // success, "hand found"
export const WARN = [255, 190, 90]; // guidance the player must act on ("STEP BACK")
export const DANGER = [255, 70, 70]; // game over, lost life
export const OUTLINE = [12, 12, 12];

export const TITLE_FAMILY = '"Avenir Next", Futura, "Trebuchet MS", "Segoe UI", system-ui, sans-serif';
export const BODY_FAMILY = '"Avenir Next", "Helvetica Neue", "Segoe UI", system-ui, sans-serif';

/** CSS colour from [r, g, b] (or pass a string through), with optional alpha. */
export function css(c, a = 1) {
  if (typeof c === "string") return c;
  return a >= 1 ? `rgb(${c[0]},${c[1]},${c[2]})` : `rgba(${c[0]},${c[1]},${c[2]},${a})`;
}

/**
 * Font size for an OpenCV-style `scale`. Hershey scale 1.0 has ~22 px caps;
 * Avenir's caps are ~0.7 em, so 1.0 -> 31 px keeps the Python layouts intact.
 */
export const fontPx = (scale) => Math.max(8, Math.round(31 * scale));

export function font(scale, { body = false, weight } = {}) {
  const w = weight ?? (body ? 500 : 800);
  return `${w} ${fontPx(scale)}px ${body ? BODY_FAMILY : TITLE_FAMILY}`;
}

export function textWidth(ctx, str, scale, opts = {}) {
  ctx.save();
  ctx.font = font(scale, opts);
  const w = ctx.measureText(str).width;
  ctx.restore();
  return w;
}

/**
 * Text at baseline (x, y). `shadow` draws a dark outline that stays registered
 * to the glyphs (stroke under fill, the Canvas equivalent of outlined_text).
 */
export function text(ctx, str, x, y, scale, colour, opts = {}) {
  const { align = "left", body = false, weight, shadow = true, alpha = 1,
    outline = OUTLINE, outlineWidth } = opts;
  ctx.save();
  ctx.font = font(scale, { body, weight });
  ctx.textAlign = align;
  ctx.textBaseline = "alphabetic";
  ctx.globalAlpha *= alpha;
  if (shadow) {
    ctx.lineJoin = "round";
    ctx.lineWidth = outlineWidth ?? Math.max(3, fontPx(scale) * 0.14);
    ctx.strokeStyle = css(outline);
    ctx.strokeText(str, x, y);
  }
  ctx.fillStyle = css(colour);
  ctx.fillText(str, x, y);
  ctx.restore();
}

export function centredText(ctx, str, cx, y, scale, colour, opts = {}) {
  text(ctx, str, cx, y, scale, colour, { ...opts, align: "center" });
}

export function wrap(ctx, str, widthPx, scale, opts = { body: true }) {
  ctx.save();
  ctx.font = font(scale, opts);
  const lines = [];
  let line = "";
  for (const word of str.split(/\s+/).filter(Boolean)) {
    const trial = line ? `${line} ${word}` : word;
    if (ctx.measureText(trial).width <= widthPx || !line) line = trial;
    else {
      lines.push(line);
      line = word;
    }
  }
  if (line) lines.push(line);
  ctx.restore();
  return lines;
}

export function roundRectPath(ctx, x0, y0, x1, y1, radius) {
  const r = Math.max(0, Math.min(radius, (x1 - x0) / 2, (y1 - y0) / 2));
  ctx.beginPath();
  ctx.moveTo(x0 + r, y0);
  ctx.arcTo(x1, y0, x1, y1, r);
  ctx.arcTo(x1, y1, x0, y1, r);
  ctx.arcTo(x0, y1, x0, y0, r);
  ctx.arcTo(x0, y0, x1, y0, r);
  ctx.closePath();
}

/** Translucent rounded panel with an optional border. rect = [x0, y0, x1, y1]. */
export function panel(ctx, rect, { colour = PANEL, alpha = 0.78, border = null, radius = 18, thickness = 2 } = {}) {
  const [x0, y0, x1, y1] = rect;
  if (x1 <= x0 || y1 <= y0) return;
  ctx.save();
  roundRectPath(ctx, x0, y0, x1, y1, radius);
  ctx.fillStyle = css(colour, alpha);
  ctx.fill();
  if (border) {
    ctx.lineWidth = thickness;
    ctx.strokeStyle = css(border);
    ctx.stroke();
  }
  ctx.restore();
}

/** Darken the whole canvas (menus, pause, countdown). */
export function dim(ctx, amount = 0.55) {
  ctx.save();
  ctx.fillStyle = `rgba(0,0,0,${amount})`;
  ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.restore();
}

/** Centred warning that must be seen, e.g. the camera has stopped. */
export function banner(ctx, title, sub = null) {
  const W = ctx.canvas.width, H = ctx.canvas.height;
  const tw = textWidth(ctx, title, 1.0);
  const sw = sub ? textWidth(ctx, sub, 0.55, { body: true }) : 0;
  const bw = Math.max(tw, sw) + 56, bh = sub ? 96 : 66;
  const x0 = W / 2 - bw / 2, y0 = H / 2 - bh / 2;
  panel(ctx, [x0, y0, x0 + bw, y0 + bh], { colour: [26, 20, 20], alpha: 0.92, border: WARN, radius: 14 });
  centredText(ctx, title, W / 2, y0 + 44, 1.0, WARN);
  if (sub) centredText(ctx, sub, W / 2, y0 + 78, 0.55, [210, 200, 200], { body: true, shadow: false });
}
