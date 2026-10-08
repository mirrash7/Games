// Sprite helpers for Canvas 2D (the browser side of gfx.py). Pre-scale and
// pre-tint at load time; per frame, only drawImage.

export function makeCanvas(w, h) {
  w = Math.max(1, Math.round(w));
  h = Math.max(1, Math.round(h));
  if (typeof OffscreenCanvas !== "undefined") return new OffscreenCanvas(w, h);
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  return c;
}

export async function loadImage(url, attempts = 3) {
  // onload rather than img.decode(): decode() can stall while the tab is hidden.
  // Retried: an image occasionally fails once even though the server sent it,
  // and one missing sprite would otherwise stop the whole arcade from starting.
  for (let i = 1; ; i++) {
    try {
      return await new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error(`could not load ${url}`));
        img.src = i === 1 ? url : `${url}${url.includes("?") ? "&" : "?"}retry=${i}`;
      });
    } catch (err) {
      if (i >= attempts) throw err;
      await new Promise((r) => setTimeout(r, 250 * i));
    }
  }
}

export async function loadJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`could not load ${url} (${r.status})`);
  return r.json();
}

/** A copy of `img` resized to w x h (high-quality downscale, done once). */
export function scaled(img, w, h) {
  const c = makeCanvas(w, h);
  const g = c.getContext("2d");
  g.imageSmoothingQuality = "high";
  g.drawImage(img, 0, 0, c.width, c.height);
  return c;
}

/** Scale to a target width, keeping aspect. */
export function scaledToWidth(img, w) {
  return scaled(img, w, (img.height * w) / img.width);
}

/**
 * Replace a sprite's colour entirely, keeping its alpha - for sprites drawn in
 * pure white so one image serves every tint (splats, sparks, feathers).
 */
export function tinted(img, colour) {
  const c = makeCanvas(img.width, img.height);
  const g = c.getContext("2d");
  g.drawImage(img, 0, 0);
  g.globalCompositeOperation = "source-in";
  g.fillStyle = typeof colour === "string" ? colour : `rgb(${colour[0]},${colour[1]},${colour[2]})`;
  g.fillRect(0, 0, c.width, c.height);
  return c;
}

/**
 * Draw `img` centred on (cx, cy), optionally faded, rotated (radians) and
 * scaled. `additive` blends with "lighter" for glows and sparks.
 */
export function blit(ctx, img, cx, cy, { alpha = 1, angle = 0, scale = 1, w, h, additive = false } = {}) {
  if (alpha <= 0.003) return;
  const dw = (w ?? img.width) * scale;
  const dh = (h ?? img.height) * scale;
  ctx.save();
  if (alpha < 1) ctx.globalAlpha *= alpha;
  if (additive) ctx.globalCompositeOperation = "lighter";
  if (angle) {
    ctx.translate(cx, cy);
    ctx.rotate(angle);
    ctx.drawImage(img, -dw / 2, -dh / 2, dw, dh);
  } else {
    ctx.drawImage(img, cx - dw / 2, cy - dh / 2, dw, dh);
  }
  ctx.restore();
}
