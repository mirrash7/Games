// Drawing Flappy Raccoon: a full game scene, plus the camera as picture-in-picture
// (port of src/kpapp/game/flappy/render.py).
//
// Unlike Snack Attack, the camera is not blended under the scene. There the hand
// *is* the blade and the player must see it on the play field; here the input is
// a whole-body gesture whose position does not matter, and a busy camera image
// behind thin pipes made the gaps hard to read. The camera moves to a corner
// window instead, where it does a different job: it teaches the gesture. It shows
// the tracked shoulders and wrists, a live "wing" meter for how high the arms
// are, and flashes FLAP! the instant a flap is recognised, so a player whose
// flaps are not registering can see why.
//
// Budget: <= 4 ms a frame at 1280x720. Every static layer (shaded ground, tiled
// pipe strips, flipped caps, scaled bird frames, tinted feathers, scaled medals)
// is built once per art set and shared by every round; per frame there is only
// drawImage and a little text.
//
// Colours are RGB (the Python file's BGR tuples, reversed).

import { KP } from "../../core/pose.js";
import { Rng } from "../../core/random.js";
import { makeCanvas } from "../../core/gfx.js";
import { centredText, css, roundRectPath } from "../../core/theme.js";

const WHITE = [255, 255, 255];
const GOLD = [255, 200, 40];
const ORANGE = [235, 120, 40];
const STEP_BACK = [255, 200, 90];
const DARK = [20, 20, 20];

export const SKYLINE_PARALLAX = 0.30; // the far layer scrolls at 30% of the ground: reads as distance
export const PIP_MARGIN = 16;
export const PIP_RADIUS = 14;
const FEATHER_SCALE = 1.5; // the 24 px feather was lost against the sky at play distance
const MEDAL_SIZE = 104;

const lerp3 = (a, b, k) => a.map((v, i) => Math.trunc(v + (b[i] - v) * k));
const clamp01 = (x) => Math.min(1, Math.max(0, x));
const mod = (a, m) => ((a % m) + m) % m;

function easeOut(t) {
  t = clamp01(t);
  return 1.0 - (1.0 - t) ** 3;
}

/** Overshoot then settle: 0.6 -> 1.3 -> 1.0 over the first 30%. */
export function popScale(age) {
  if (age < 0.12) return 0.6 + (age / 0.12) * 0.7;
  if (age < 0.30) return 1.3 - ((age - 0.12) / 0.18) * 0.3;
  return 1.0;
}

/**
 * Text centred on cx at baseline y. `outline` is the Python outline width in px
 * (0 = none): OpenCV stamped the text `outline` px out in 8 directions; a
 * canvas stroke centred on the glyph edge needs twice that.
 */
function label(ctx, str, cx, y, scale, colour, { outline = 2, alpha = 1, body = false } = {}) {
  centredText(ctx, str, cx, y, scale, colour, {
    body, alpha, shadow: outline > 0, outlineWidth: 2 * outline + 1,
    // Sentences use the body face, heavier than its default so they still
    // read from ~2 m (the Python game drew them in bold Hershey Duplex).
    weight: body ? 700 : undefined,
  });
}

function flippedV(img) {
  const c = makeCanvas(img.width, img.height);
  const g = c.getContext("2d");
  g.translate(0, img.height);
  g.scale(1, -1);
  g.drawImage(img, 0, 0);
  return c;
}

function resized(img, w, h) {
  const c = makeCanvas(w, h);
  const g = c.getContext("2d");
  g.imageSmoothingEnabled = true;
  g.imageSmoothingQuality = "high";
  g.drawImage(img, 0, 0, c.width, c.height);
  return c;
}

/** Tint a white sprite by multiplying, keeping its shading (unlike gfx.tinted). */
function multiplyTint(img, rgb) {
  const c = makeCanvas(img.width, img.height);
  const g = c.getContext("2d");
  g.drawImage(img, 0, 0);
  g.globalCompositeOperation = "multiply";
  g.fillStyle = css(rgb);
  g.fillRect(0, 0, c.width, c.height);
  g.globalCompositeOperation = "destination-in";
  g.drawImage(img, 0, 0);
  return c;
}

// Static layers depend only on the art (and bird scale), so they are built once
// and shared by every round: the arcade builds a fresh game per PLAY AGAIN.
const layerCache = new WeakMap();

function buildLayers(art, W, H, birdScale, tints) {
  const groundY = H - art.groundHeight;
  const L = { groundY };

  L.sky = art.sky.width === W && art.sky.height === H ? art.sky : resized(art.sky, W, H);
  L.skyline = art.skyline;
  L.skylineY = groundY - art.skyline.height; // skyline sits on the ground

  // The arcade prints its key hints and footer over the bottom of the ground
  // in light grey, unreadable on sand. Shade the lower ground toward dark,
  // baked into the strip once so it costs nothing per frame.
  {
    const g0 = art.ground;
    const c = makeCanvas(g0.width, g0.height);
    const g = c.getContext("2d", { willReadFrequently: true });
    g.drawImage(g0, 0, 0);
    const im = g.getImageData(0, 0, c.width, c.height);
    const gh = c.height;
    for (let y = 0; y < gh; y++) {
      const ramp = clamp01((y - gh * 0.25) / (gh * 0.75));
      const k = 1.0 - 0.6 * ramp ** 1.1;
      for (let x = 0, i = y * c.width * 4; x < c.width; x++, i += 4) {
        im.data[i] *= k;
        im.data[i + 1] *= k;
        im.data[i + 2] *= k;
      }
    }
    g.putImageData(im, 0, 0);
    L.ground = c;
  }

  // Pipes: a body strip tiled once to the full play height (both
  // orientations), and the cap upright and flipped for the top pipe.
  {
    const body = art.pipeBody;
    const tallH = groundY + body.height;
    const c = makeCanvas(body.width, tallH);
    const g = c.getContext("2d");
    for (let y = 0; y < tallH; y += body.height) g.drawImage(body, 0, y);
    L.bodyDown = c; // anchored at its top
    L.bodyUp = flippedV(c); // anchored at its bottom
    L.pipeW = body.width;
    L.cap = art.pipeCap;
    L.capUp = flippedV(art.pipeCap);
    L.capW = art.pipeCap.width;
    L.capH = art.pipeCap.height;
  }

  // Bird: scaled once; rotation is a transform at draw time.
  L.bird = art.birdFrames.map((f) => (Math.abs(birdScale - 1) > 1e-3
    ? resized(f, Math.round(f.width * birdScale), Math.round(f.height * birdScale)) : f));

  // Feathers: one pre-scaled, pre-tinted sprite per tint.
  {
    const f = art.feather;
    const base = resized(f, Math.round(f.width * FEATHER_SCALE), Math.round(f.height * FEATHER_SCALE));
    L.feathers = tints.map((t) => multiplyTint(base, t));
  }

  // Game-over scoreboard pieces.
  L.panel = art.panel;
  L.medals = Object.fromEntries(Object.entries(art.medals)
    .map(([m, img]) => [m, resized(img, MEDAL_SIZE, MEDAL_SIZE)]));
  return L;
}

function layersFor(art, W, H, birdScale, tints) {
  let byKey = layerCache.get(art);
  if (!byKey) layerCache.set(art, (byKey = new Map()));
  const key = `${W}x${H}@${birdScale}`;
  let L = byKey.get(key);
  if (!L) byKey.set(key, (L = buildLayers(art, W, H, birdScale, tints)));
  return L;
}

export class FlappyRenderer {
  constructor(width, height, art, { birdScale = 1.0, tints, seed = null } = {}) {
    this.width = width;
    this.height = height;
    this.L = layersFor(art, width, height, birdScale, tints);
    this.groundY = this.L.groundY;

    // Picture-in-picture: exactly a quarter of the frame, top-right.
    this.pipW = Math.floor(width / 4);
    this.pipH = Math.floor(height / 4);
    const x1 = width - PIP_MARGIN;
    this.pipRect = [x1 - this.pipW, PIP_MARGIN, x1, PIP_MARGIN + this.pipH];
    this.pip = makeCanvas(this.pipW, this.pipH);
    this.pipCtx = this.pip.getContext("2d");

    this._shake = [0, 0];
    this._rng = new Rng(seed == null ? null : `${seed}:shake`);
  }

  // --- main ---

  draw(ctx, frame, game) {
    ctx.save();
    if (game.fx.shake > 0) {
      const k = 12.0 * game.fx.shake;
      this._shake = [this._rng.uniform(-1, 1) * k, this._rng.uniform(-1, 1) * k];
    } else {
      this._shake = [0, 0];
    }
    const ox = Math.trunc(this._shake[0]), oy = Math.trunc(this._shake[1]);

    this._drawBackdrop(ctx, game);
    this._drawPipes(ctx, game, ox, oy);
    this._drawGround(ctx, game, ox);
    this._drawFeathers(ctx, game, ox, oy);
    this._drawBird(ctx, game, ox, oy);
    this._drawPopups(ctx, game);
    this._drawScore(ctx, game);
    if (game.fx.flash > 0) {
      ctx.fillStyle = `rgba(255,255,255,${Math.min(1.0, game.fx.flash) * 0.85})`;
      ctx.fillRect(0, 0, this.width, this.height);
    }
    if (game.phase === "ready") this._drawReady(ctx, game);
    else if (game.phase === "game_over") this._drawGameOver(ctx, game);
    this._drawPip(ctx, frame, game);
    ctx.restore();
  }

  // --- world ---

  _drawBackdrop(ctx, game) {
    const L = this.L;
    ctx.drawImage(L.sky, 0, 0);
    const w = L.skyline.width;
    const off = mod(Math.trunc(game.distance * SKYLINE_PARALLAX), w);
    for (let x = -off; x < this.width; x += w) ctx.drawImage(L.skyline, x, L.skylineY);
  }

  _drawGround(ctx, game, ox) {
    const g = this.L.ground;
    const off = mod(Math.trunc(game.distance) - ox, g.width);
    for (let x = -off; x < this.width; x += g.width) ctx.drawImage(g, x, this.groundY);
  }

  _drawPipes(ctx, game, ox, oy) {
    const L = this.L;
    for (const p of game.pipes) {
      const cx = Math.round(p.x) + ox;
      if (cx + L.capW < 0 || cx - L.capW > this.width) continue;
      const bx = cx - Math.floor(L.pipeW / 2);
      const capX = cx - Math.floor(L.capW / 2);
      const top = Math.round(p.gapTop) + oy;
      const bottom = Math.round(p.gapBottom) + oy;
      // Top pipe: body down from the ceiling (pattern anchored at the cap so it
      // never crawls), flipped cap at the opening.
      const topEnd = top - L.capH;
      if (topEnd > 0) ctx.drawImage(L.bodyUp, bx, topEnd - L.bodyUp.height);
      ctx.drawImage(L.capUp, capX, topEnd);
      // Bottom pipe: cap at the opening, body down to the ground (the ground,
      // drawn next, covers whatever runs past it).
      if (bottom + L.capH < this.groundY) ctx.drawImage(L.bodyDown, bx, bottom + L.capH);
      ctx.drawImage(L.cap, capX, bottom);
    }
  }

  _drawBird(ctx, game, ox, oy) {
    const b = game.bird;
    const frames = this.L.bird;
    const n = frames.length;
    let fi;
    if (game.phase === "game_over") {
      fi = Math.min(1, n - 1); // wings still, half-folded
    } else {
      // Wing cycle: down, ..., up, ... back (frames are ordered up -> down).
      const cycle = [];
      for (let i = 0; i < n; i++) cycle.push(n - 1 - i);
      for (let i = 1; i < n - 1; i++) cycle.push(i);
      fi = cycle[Math.trunc(b.wing) % cycle.length];
    }
    const spr = frames[fi];
    const cx = Math.round(b.x) + ox, cy = Math.round(b.y) + oy;
    ctx.save();
    ctx.translate(cx, cy);
    if (b.angle) ctx.rotate((b.angle * Math.PI) / 180); // + is nose down = clockwise on screen
    ctx.drawImage(spr, -spr.width / 2, -spr.height / 2);
    ctx.restore();
  }

  _drawFeathers(ctx, game, ox, oy) {
    const sprites = this.L.feathers;
    for (const f of game.fx.feathers) {
      if (f.life <= 0) continue;
      const spr = sprites[f.tint] ?? sprites[0];
      ctx.save();
      ctx.globalAlpha = Math.min(1.0, f.life * 1.6);
      ctx.translate(Math.trunc(f.x) + ox, Math.trunc(f.y) + oy);
      ctx.rotate((f.angle * Math.PI) / 180);
      ctx.drawImage(spr, -spr.width / 2, -spr.height / 2);
      ctx.restore();
    }
  }

  // --- HUD ---

  _drawPopups(ctx, game) {
    for (const pop of game.fx.popups) {
      const age = 1.0 - Math.max(0.0, pop.life);
      const scale = 1.6 * popScale(age);
      const alpha = age < 0.6 ? 1.0 : Math.max(0.0, (1.0 - age) / 0.4);
      label(ctx, pop.text, pop.x, pop.y, scale, GOLD, { outline: 2, alpha });
    }
  }

  _drawScore(ctx, game) {
    if (game.phase !== "playing") return;
    const bump = game.fx.scoreBump;
    const colour = lerp3(GOLD, WHITE, 1 - bump); // flashes gold, settles white
    label(ctx, String(game.score), this.width / 2, 104, 2.6 * (1.0 + 0.3 * bump), colour,
      { outline: 4 });
  }

  _drawReady(ctx, game) {
    // Not before the first update: during the arcade's 3-2-1 the game is
    // rendered but not updated, and the countdown has the centre stage.
    if (game.clock <= 0.0) return;
    const W = this.width;
    const pulse = 1.0 + 0.06 * Math.sin(game.clock * 2 * Math.PI * 1.2);
    label(ctx, "FLAP TO START", W / 2, 205, 2.1 * pulse, GOLD, { outline: 4 });
    if (game.tooClose) {
      // Flaps leave the frame when standing this close; fix that first.
      label(ctx, "STEP BACK", W / 2, 468, 1.4, STEP_BACK, { outline: 3 });
      label(ctx, "until your waist is in the camera view", W / 2, 512, 0.8, WHITE,
        { outline: 2, body: true });
      return;
    }
    label(ctx, "Raise both arms, then beat them down like wings", W / 2, 470, 0.9, WHITE,
      { outline: 2, body: true });
    label(ctx, "Each flap = one hop", W / 2, 512, 0.75, [255, 240, 225], { outline: 2, body: true });
  }

  _drawGameOver(ctx, game) {
    const W = this.width, L = this.L;
    const t = game.deadTime;
    // Title drops in once the crash flash has faded.
    const k = easeOut((t - 0.25) / 0.35);
    if (k > 0) label(ctx, "GAME OVER", W / 2, 150 - (1 - k) * 60, 2.4, ORANGE, { outline: 4 });

    // Scoreboard slides up from below.
    const s = easeOut((t - 0.55) / 0.45);
    if (s <= 0) return;
    const panel = L.panel;
    const px = Math.floor(W / 2) - Math.floor(panel.width / 2);
    const py = Math.trunc(190 + (1 - s) * (this.height - 190));
    ctx.drawImage(panel, px, py);

    // Left: the medal. Right: score (counting up) and best.
    const medal = game.medal;
    const mx = px + panel.width * 0.29, my = py + panel.height * 0.56;
    label(ctx, "MEDAL", mx, py + 70, 0.85, ORANGE, { outline: 0 });
    if (medal != null) {
      const spr = L.medals[medal];
      ctx.drawImage(spr, Math.trunc(mx - spr.width / 2), Math.trunc(my - spr.height / 2 + 6));
    } else {
      ctx.beginPath();
      ctx.arc(Math.trunc(mx), Math.trunc(my + 6), 46, 0, 2 * Math.PI);
      ctx.lineWidth = 3;
      ctx.strokeStyle = css([215, 190, 150]);
      ctx.stroke();
    }

    const count = s >= 1.0 ? easeOut((t - 1.0) / 0.6) : 0.0;
    const shown = Math.round(game.score * count);
    const rx = px + panel.width * 0.70;
    label(ctx, "SCORE", rx, py + 70, 0.85, ORANGE, { outline: 0 });
    label(ctx, String(shown), rx, py + 132, 1.8, WHITE, { outline: 3 });
    label(ctx, "BEST", rx, py + 182, 0.85, ORANGE, { outline: 0 });
    label(ctx, String(game.best), rx, py + 244, 1.8, WHITE, { outline: 3 });

    if (game.newBest && count >= 1.0) {
      const pop = 1.0 + 0.08 * Math.sin(t * 2 * Math.PI * 1.5);
      label(ctx, "NEW BEST!", W / 2, py + panel.height + 50, 1.3 * pop, GOLD, { outline: 3 });
    }
  }

  // --- the camera window ---

  _drawPip(ctx, frame, game) {
    const [x0, y0, x1, y1] = this.pipRect;
    const g = this.pipCtx, pw = this.pipW, ph = this.pipH;
    const det = game.detector;
    const s = pw / this.width;
    const visible = Boolean(det?.armsVisible);
    const flap = game.fx.flap;

    g.save();
    if (frame) {
      g.drawImage(frame, 0, 0, pw, ph);
    } else {
      g.fillStyle = "rgb(40,40,40)";
      g.fillRect(0, 0, pw, ph);
    }
    if (!visible) {
      g.fillStyle = "rgba(0,0,0,0.55)";
      g.fillRect(0, 0, pw, ph);
    }

    const pose = game.lastPose;
    if (pose != null) this._drawArms(g, pose, s, det?.wing ?? null);

    const wing = visible ? (det?.wing ?? null) : null;
    this._drawWingMeter(g, wing, flap);

    if (!visible && game.clock > 0.0) {
      const cy = Math.floor(ph / 2);
      g.fillStyle = css(DARK);
      g.fillRect(8, cy - 22, pw - 32 - 8 + 1, 38 + 1);
      label(g, "SHOW BOTH ARMS", pw / 2 - 12, cy + 8, 0.75, STEP_BACK, { outline: 0 });
    } else if (game.tooClose && flap <= 0) {
      const cy = ph - 22;
      g.fillStyle = css(DARK);
      g.fillRect(8, cy - 22, pw - 32 - 8 + 1, 32 + 1);
      label(g, "STEP BACK", pw / 2 - 12, cy + 2, 0.7, STEP_BACK, { outline: 0 });
    } else if (flap > 0) {
      const age = 1.0 - flap;
      label(g, "FLAP!", pw / 2 - 10, ph / 2 + 14, 1.25 * popScale(age), GOLD, { outline: 3 });
    }
    g.restore();

    ctx.save();
    roundRectPath(ctx, x0, y0, x1, y1, PIP_RADIUS);
    ctx.clip();
    ctx.drawImage(this.pip, x0, y0);
    ctx.restore();

    const colour = lerp3([235, 235, 235], GOLD, flap);
    ctx.save();
    roundRectPath(ctx, x0, y0, x1 - 1, y1 - 1, PIP_RADIUS);
    ctx.lineWidth = 6;
    ctx.strokeStyle = css(DARK);
    ctx.stroke();
    ctx.lineWidth = flap < 0.3 ? 3 : 4;
    ctx.strokeStyle = css(colour);
    ctx.stroke();
    ctx.restore();
  }

  _drawArms(g, pose, s, wing) {
    const names = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
      "left_wrist", "right_wrist"];
    const pts = {};
    for (const n of names) {
      const i = KP[n];
      if (pose.confidence[i] >= 0.4) pts[n] = [Math.trunc(pose.x(i) * s), Math.trunc(pose.y(i) * s)];
    }
    const k = wing == null ? 0.0 : clamp01(wing);
    const arm = css(lerp3([235, 235, 235], GOLD, k));
    const dark = css(DARK);
    g.lineCap = "round";
    g.lineJoin = "round";
    const line = (a, b, colour, width) => {
      g.beginPath();
      g.moveTo(a[0], a[1]);
      g.lineTo(b[0], b[1]);
      g.lineWidth = width;
      g.strokeStyle = colour;
      g.stroke();
    };
    const dot = (p, r, colour) => {
      g.beginPath();
      g.arc(p[0], p[1], r, 0, 2 * Math.PI);
      g.fillStyle = colour;
      g.fill();
    };
    for (const [a, b, colour] of [["left_shoulder", "right_shoulder", "rgb(200,200,200)"],
      ["left_shoulder", "left_elbow", arm], ["left_elbow", "left_wrist", arm],
      ["right_shoulder", "right_elbow", arm], ["right_elbow", "right_wrist", arm]]) {
      if (pts[a] && pts[b]) {
        line(pts[a], pts[b], dark, 5);
        line(pts[a], pts[b], colour, 2);
      }
    }
    // Shoulder to wrist directly when the elbow is missing (common up close).
    for (const [sh, el, wr] of [["left_shoulder", "left_elbow", "left_wrist"],
      ["right_shoulder", "right_elbow", "right_wrist"]]) {
      if (pts[sh] && pts[wr] && !pts[el]) line(pts[sh], pts[wr], arm, 2);
    }
    for (const n of ["left_shoulder", "right_shoulder"]) if (pts[n]) dot(pts[n], 4, "rgb(230,230,230)");
    for (const n of ["left_wrist", "right_wrist"]) {
      if (pts[n]) {
        dot(pts[n], 7, dark);
        dot(pts[n], 5, arm);
      }
    }
  }

  /** Vertical gauge on the PiP's right edge: how high the arms are now. */
  _drawWingMeter(g, wing, flap) {
    const h = this.pipH;
    const x0 = this.pipW - 22, x1 = this.pipW - 10;
    const y0 = 14, y1 = h - 14;
    g.fillStyle = css(DARK);
    g.fillRect(x0 - 2, y0 - 2, x1 - x0 + 5, y1 - y0 + 5);
    if (wing != null) {
      const k = clamp01(wing);
      const top = Math.trunc(y1 - (y1 - y0) * k);
      const colour = flap > 0 ? GOLD : lerp3([210, 210, 210], GOLD, k);
      g.fillStyle = css(colour);
      g.fillRect(x0, top, x1 - x0 + 1, y1 - top + 1);
    }
    g.lineWidth = 1;
    g.strokeStyle = "rgb(235,235,235)";
    g.strokeRect(x0 - 2 + 0.5, y0 - 2 + 0.5, x1 - x0 + 4, y1 - y0 + 4);
  }
}
