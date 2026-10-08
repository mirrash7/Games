// Drawing the Snack Attack scene over the live camera frame
// (port of game/fruitninja/render.py).
//
// The camera is blended with the backdrop rather than hidden: the blade is the
// player's own hand, and a player who cannot see their hand is aiming blind.
// Dimming the feed under the backdrop keeps the snacks readable while leaving
// the player visible.
//
// All colours are RGB (the Python originals were BGR; each is noted).
// Sprites are pre-scaled and pre-tinted once per art set; per frame this only
// calls drawImage and fills a couple of paths.

import { blit, scaled, scaledToWidth, tinted } from "../../core/gfx.js";
import { ACCENT, centredText, css, fontPx, text, textWidth, WHITE } from "../../core/theme.js";

// The raccoon's trail. A near-white core read as white with a purple edge; a
// vivid purple glow around a light-purple core reads as purple while staying
// bright enough to pop on the dark night scene.
const TRAIL_GLOW = [130, 60, 190]; // BGR (190, 60, 130)
const TRAIL_CORE = [215, 175, 245]; // BGR (245, 175, 215)
const RACCOON_PX = 104; // drawn width of the raccoon head on the palm
const GOLD = ACCENT; // BGR (70, 210, 255)
const RED = [255, 70, 70]; // BGR (70, 70, 255)
const HINT = [255, 200, 70]; // BGR (70, 200, 255)
const HINT_BG = [22, 18, 18]; // BGR (18, 18, 22)
const COMBO_GREY = [200, 190, 190]; // BGR (190, 190, 200)
const SPLAT_PX = 140; // splats are soft blobs: halving the side is not visible

export const CAMERA_WEIGHT = 0.45; // how much of the live feed survives the backdrop blend

const GLOW_CSS = css(TRAIL_GLOW);
const CORE_CSS = css(TRAIL_CORE);

// Per art set, so a fresh game each round does not redo the pre-scaling.
const _prepared = new WeakMap();

function prepare(art, width, height) {
  let p = _prepared.get(art);
  if (p && p.width === width && p.height === height) return p;
  const raccoon = (img) => (img ? scaledToWidth(img, RACCOON_PX) : null);
  const idle = raccoon(art.cursorIdle);
  p = {
    width, height,
    bg: art.background ? scaled(art.background, width, height) : null,
    splat: art.splat ? scaled(art.splat, SPLAT_PX, SPLAT_PX) : null,
    splatTints: new Map(), // "r,g,b" -> tinted splat
    raccoonIdle: idle,
    raccoonChomp: raccoon(art.cursorChomp) ?? idle,
  };
  if (p.splat) for (const f of art.fruits) tintedSplat(p, f.juice);
  _prepared.set(art, p);
  return p;
}

function tintedSplat(p, colour) {
  const key = `${colour[0]},${colour[1]},${colour[2]}`;
  let t = p.splatTints.get(key);
  if (!t) {
    t = tinted(p.splat, colour);
    p.splatTints.set(key, t);
  }
  return t;
}

const _cssCache = new Map();
function cssOf(c) {
  const key = `${c[0]},${c[1]},${c[2]}`;
  let s = _cssCache.get(key);
  if (!s) {
    s = css(c);
    _cssCache.set(key, s);
  }
  return s;
}

const clamp01 = (v) => Math.min(Math.max(v, 0), 1);

/** Corner-cutting subdivision: a polyline becomes a smooth curve. Visual only. */
export function chaikin(pts, rounds = 2) {
  for (let r = 0; r < rounds; r++) {
    if (pts.length < 3) return pts;
    const out = [pts[0]];
    for (let i = 0; i + 1 < pts.length; i++) {
      const a = pts[i], b = pts[i + 1];
      out.push([a[0] * 0.75 + b[0] * 0.25, a[1] * 0.75 + b[1] * 0.25]);
      out.push([a[0] * 0.25 + b[0] * 0.75, a[1] * 0.25 + b[1] * 0.75]);
    }
    out.push(pts[pts.length - 1]);
    pts = out;
  }
  return pts;
}

/**
 * A single tapered ribbon along the hand path, traced into ctx as a path.
 * Stroking segments separately with round caps made a lumpy worm; offsetting
 * the path into one polygon tapers cleanly from the tail to the hand.
 */
function bladePath(ctx, pts, maxWidth) {
  const n = pts.length;
  if (n < 2) return false;
  const left = [], right = [];
  for (let i = 0; i < n; i++) {
    const p = pts[i];
    const nxt = pts[Math.min(i + 1, n - 1)], prv = pts[Math.max(i - 1, 0)];
    let dx = nxt[0] - prv[0], dy = nxt[1] - prv[1];
    let norm = Math.hypot(dx, dy);
    if (norm < 1e-3) { dx = 1; dy = 0; norm = 1; }
    const px = -dy / norm, py = dx / norm;
    // Width grows toward the newest sample; the tail comes to a point.
    const w = maxWidth * Math.pow(i / (n - 1), 0.8);
    left.push([p[0] + px * w, p[1] + py * w]);
    right.push([p[0] - px * w, p[1] - py * w]);
  }
  ctx.beginPath();
  ctx.moveTo(left[0][0], left[0][1]);
  for (let i = 1; i < n; i++) ctx.lineTo(left[i][0], left[i][1]);
  for (let i = n - 1; i >= 0; i--) ctx.lineTo(right[i][0], right[i][1]);
  ctx.closePath();
  return true;
}

/** Overshoot then settle: 0.6 -> 1.35 -> 1.0 over the first 30%. */
export function popScale(age) {
  if (age < 0.12) return 0.6 + (age / 0.12) * 0.75;
  if (age < 0.30) return 1.35 - ((age - 0.12) / 0.18) * 0.35;
  return 1.0;
}

export class SnackRenderer {
  constructor(width, height, art) {
    this.width = width;
    this.height = height;
    this.art = art;
    this.p = prepare(art, width, height);
  }

  draw(ctx, game) {
    ctx.save();
    // frame*0.45 + backdrop*0.55, as cv2.addWeighted did.
    if (this.p.bg) {
      ctx.globalAlpha = 1 - CAMERA_WEIGHT;
      ctx.drawImage(this.p.bg, 0, 0);
      ctx.globalAlpha = 1;
    }
    this._drawSplats(ctx, game);
    this._drawHalves(ctx, game);
    this._drawFruits(ctx, game);
    this._drawSparks(ctx, game);
    this._drawPopups(ctx, game);
    this._drawBlade(ctx, game);
    this._drawHud(ctx, game);
    this._drawBanner(ctx, game);
    this._drawGameOver(ctx, game);
    ctx.restore();
  }

  _drawSplats(ctx, game) {
    if (!this.p.splat) return;
    for (const s of game.fx.splats) {
      blit(ctx, tintedSplat(this.p, s.colour), s.pos[0], s.pos[1],
        { alpha: clamp01(s.life) * 0.75, scale: s.scale });
    }
  }

  _drawHalves(ctx, game) {
    for (const h of game.fx.halves) {
      if (!h.sprite) continue;
      blit(ctx, h.sprite, h.pos[0], h.pos[1], { alpha: clamp01(h.life), angle: h.angle });
    }
  }

  _drawFruits(ctx, game) {
    for (const f of game.fruits) {
      if (f.sliced) continue;
      const sprite = f.isBomb ? this.art.bomb : (f.art ? f.art.whole : null);
      if (!sprite) continue;
      blit(ctx, sprite, f.pos[0], f.pos[1], { angle: f.angle });
    }
  }

  _drawSparks(ctx, game) {
    let last = null;
    for (const s of game.fx.sparks) {
      if (s.life <= 0) continue;
      const style = cssOf(s.colour);
      if (style !== last) {
        if (last !== null) ctx.fill();
        ctx.fillStyle = style;
        ctx.beginPath();
        last = style;
      }
      const r = Math.max(1, Math.floor(4 * s.life));
      ctx.moveTo(s.pos[0] + r, s.pos[1]);
      ctx.arc(s.pos[0], s.pos[1], r, 0, Math.PI * 2);
    }
    if (last !== null) ctx.fill();
  }

  /** Floating points at each cut, in the snack's crumb colour, lightened. */
  _drawPopups(ctx, game) {
    for (const pop of game.fx.popups) {
      const age = 1 - Math.max(0, pop.life);
      // Sized to read from a couple of metres back, where the player stands.
      const scale = (pop.big ? 1.9 : 1.45) * popScale(age);
      // Lighten the colour so dark snacks still read on the dark board.
      const c = pop.colour.map((v) => Math.round(v * 0.55 + 255 * 0.45));
      const alpha = age < 0.6 ? 1 : Math.max(0, (1 - age) / 0.4);
      if (alpha <= 0.003) continue;
      centredText(ctx, pop.text, pop.pos[0], pop.pos[1] - 40, scale, c, { alpha });
    }
  }

  /** Tapered streak along the recent hand path. */
  _drawBlade(ctx, game) {
    const pts = chaikin(game.blade.trail(game.clock));
    if (pts.length === 1) {
      this._drawCursor(ctx, game, pts[0]); // hand visible but still
      return;
    }
    if (pts.length < 2) return;
    if (bladePath(ctx, pts, 13)) {
      ctx.fillStyle = GLOW_CSS;
      ctx.fill();
    }
    if (bladePath(ctx, pts, 7)) {
      ctx.fillStyle = CORE_CSS;
      ctx.fill();
    }
    const tip = pts[pts.length - 1];
    if (game.blade.slicing && this.art.flash) {
      blit(ctx, this.art.flash, tip[0], tip[1], { alpha: 0.5, additive: true });
    }
    this._drawCursor(ctx, game, tip);
  }

  /**
   * The raccoon on the player's palm, with a faint ring for the hit area. The
   * ring stays because the raccoon's outline is not the exact hit area; thin
   * and in trail purple, it reads as part of the character.
   */
  _drawCursor(ctx, game, tip) {
    const r = game.blade.hitRadius;
    const ring = game.blade.slicing ? CORE_CSS : GLOW_CSS;
    ctx.beginPath();
    ctx.arc(tip[0], tip[1], r, 0, Math.PI * 2);
    ctx.lineWidth = 2;
    ctx.strokeStyle = ring;
    ctx.stroke();
    const sprite = game.fx.chomp > 0 ? this.p.raccoonChomp : this.p.raccoonIdle;
    if (!sprite) {
      ctx.beginPath();
      ctx.arc(tip[0], tip[1], 4, 0, Math.PI * 2);
      ctx.fillStyle = ring;
      ctx.fill();
      return;
    }
    blit(ctx, sprite, tip[0], tip[1]);
  }

  _drawHud(ctx, game) {
    // The score pulses when it changes: bigger, and flashing toward white.
    const bump = game.fx.scoreBump;
    const colour = GOLD.map((g) => Math.round(g + (255 - g) * bump * 0.7));
    text(ctx, `${game.score}`, 28, 66, 1.5 * (1 + 0.35 * bump), colour);

    const icon = this.art.lifeFull;
    if (icon) {
      const step = icon.width + 8;
      for (let i = 0; i < game.rules.lives; i++) {
        const spr = i < game.lives ? this.art.lifeFull : this.art.lifeLost;
        blit(ctx, spr, this.width - 30 - i * step, 40);
      }
    }

    // Only once play has run a moment: before the first update (e.g. during
    // the countdown) the game has not looked for the hand yet.
    if (!game.blade.active && game.phase !== "game_over" && game.clock > 0.5) {
      this._hint(ctx, `RAISE YOUR ${game.hand.toUpperCase()} HAND - IT'S THE RACCOON`,
        this.height - 40);
    }
  }

  _hint(ctx, str, y, colour = HINT) {
    const scale = 0.7;
    const tw = textWidth(ctx, str, scale);
    const th = fontPx(scale) * 0.72;
    const x = this.width / 2 - tw / 2;
    ctx.fillStyle = css(HINT_BG);
    ctx.fillRect(x - 14, y - th - 10, tw + 28, th + 20);
    text(ctx, str, x, y, scale, colour, { shadow: false });
  }

  _drawBanner(ctx, game) {
    const b = game.fx.banner;
    if (b.life <= 0 || b.count < game.rules.comboMin) return;
    const scale = 1.2 + 0.5 * b.life;
    centredText(ctx, `${b.count} COMBO  +${b.bonus}`, this.width / 2,
      Math.floor(this.height * 0.28), scale, GOLD, { outlineWidth: Math.max(4, fontPx(scale) * 0.18) });
  }

  _drawGameOver(ctx, game) {
    if (game.phase !== "game_over") return;
    ctx.fillStyle = "rgba(0,0,0,0.5)"; // frame //= 2
    ctx.fillRect(0, 0, this.width, this.height);
    const lines = [
      ["GAME OVER", 0.32, 1.8, RED],
      [game.deathReason, 0.42, 0.9, WHITE],
      [`SCORE ${game.score}`, 0.53, 1.3, GOLD],
      [`BEST COMBO ${game.bestCombo}`, 0.61, 0.7, COMBO_GREY],
    ];
    for (const [str, dy, scale, colour] of lines) {
      centredText(ctx, str, this.width / 2, Math.floor(this.height * dy), scale, colour, { shadow: false });
    }
  }
}
