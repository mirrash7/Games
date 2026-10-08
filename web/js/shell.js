// The arcade around the games: welcome screen, start, pause, stop, replay
// (port of shell.py). Nothing starts until the player chooses, and a
// countdown gives them time to step back. Everything works standing away from
// the computer: hovering the hand over a button presses it, Kinect-style.
// Keys and mouse clicks do the same for anyone at the keyboard.

import { HandTracker } from "./core/hand.js";
import {
  ACCENT, DIM, PANEL, WHITE, GOOD, WARN, centredText, panel, wrap, dim, text, css,
} from "./core/theme.js";

export const COUNTDOWN_SECONDS = 3.0;
export const MENU_DWELL = 1.0; // hover time to press a button
export const PAUSE_DWELL = 1.5; // longer in play, so a passing hand cannot pause by accident

const HOT_FILL = [34, 38, 46];
const IDLE_BORDER = [120, 110, 110];
const HINT = [165, 150, 150];

export class Button {
  constructor(action, label, rect, sub = "", keyHint = "") {
    Object.assign(this, { action, label, rect, sub, keyHint });
  }
  contains(p, pad = 0) {
    if (!p) return false;
    const [x0, y0, x1, y1] = this.rect;
    return x0 - pad <= p[0] && p[0] <= x1 + pad && y0 - pad <= p[1] && p[1] <= y1 + pad;
  }
}

/**
 * Hover-to-press. Holding the hand on a button fills it, then it fires. After
 * firing, the same button will not fire again until the hand leaves it, so a
 * hand left resting on PLAY AGAIN does not loop forever.
 */
export class Dwell {
  constructor(seconds) {
    this.seconds = seconds;
    this.reset();
  }
  reset() {
    this.target = null;
    this.progress = 0;
    this._spent = null;
  }
  /** Ignore everything until the hand is over no button at all (every screen change). */
  block() {
    this.target = null;
    this.progress = 0;
    this._spent = "*";
  }
  update(hovered, dt) {
    if (this._spent === "*") {
      if (hovered == null) this._spent = null;
      return null;
    }
    if (hovered !== this._spent) this._spent = null;
    if (hovered == null || hovered === this._spent) {
      this.target = hovered === this._spent ? hovered : null;
      this.progress = 0;
      return null;
    }
    if (hovered !== this.target) {
      this.target = hovered;
      this.progress = 0;
    }
    this.progress += dt / this.seconds;
    if (this.progress >= 1) {
      this.progress = 0;
      this._spent = hovered;
      return hovered;
    }
    return null;
  }
}

export class Shell {
  /**
   * @param {object} o
   * @param {Function[]} o.games game classes (see games/base.js)
   */
  constructor({ width = 1280, height = 720, games = [], mirrored = true, selected = null } = {}) {
    this.width = width;
    this.height = height;
    this.mirrored = mirrored;
    this.registry = Object.fromEntries(games.map((g) => [g.id, g]));
    this.menuGames = games.filter((g) => g.title && g.menu !== false).map((g) => g.id);
    if (selected && this.registry[selected] && !this.menuGames.includes(selected)) this.menuGames.push(selected);
    this.selected = selected && this.registry[selected] ? selected : this.menuGames[0] ?? null;
    this.hand = "right";
    this._trackers = { right: new HandTracker("right", mirrored), left: new HandTracker("left", mirrored) };
    this._clock = 0;
    this.screen = "menu"; // menu | countdown | playing | paused
    this.game = null;
    this.countdown = 0;
    this.cursor = null;
    this.personSeen = false;
    this.dwell = new Dwell(MENU_DWELL);
    this.pauseDwell = new Dwell(PAUSE_DWELL);
    this._buttons = [];
    this._wasOver = false;
    this.notice = null; // optional line under the menu (e.g. running on CPU)
  }

  // --- transitions ---

  start(id, countdown = true) {
    const Cls = this.registry[id];
    if (!Cls) return;
    this.selected = id;
    this.game = new Cls({ width: this.width, height: this.height, mirrored: this.mirrored });
    if (this.hand !== "right" && this.game.swapHand) this.game.swapHand();
    this.countdown = COUNTDOWN_SECONDS;
    this.screen = countdown ? "countdown" : "playing";
    this._wasOver = false;
    this.dwell.reset();
    this.pauseDwell.reset();
  }

  restart() {
    if (this.selected) this.start(this.selected);
  }

  pause() {
    if (this.screen === "playing" && !this.gameOver) {
      this.screen = "paused";
      this.dwell.block();
    }
  }

  resume() {
    if (this.screen === "paused") {
      this.screen = "playing";
      this.pauseDwell.reset();
      this.game.onResume?.();
    }
  }

  toMenu() {
    this.screen = "menu";
    this.game = null;
    this.dwell.block();
  }

  swapHand() {
    this.hand = this.hand === "right" ? "left" : "right";
    if (this.game?.swapHand) this.game.swapHand();
  }

  get gameOver() {
    return this.game?.phase === "game_over";
  }

  _do(action) {
    if (action.startsWith("start:")) this.start(action.slice(6));
    else if (action === "resume") this.resume();
    else if (action === "menu") this.toMenu();
    else if (action === "again") this.restart();
    else if (action === "pause") this.pause();
  }

  // --- input ---

  /** Apply a key (KeyboardEvent.key). Returns true if it was used. */
  handleKey(key) {
    const k = key.length === 1 ? key.toLowerCase() : key;
    if (k === "h") {
      this.swapHand();
      return true;
    }
    const esc = k === "Escape", enter = k === "Enter", space = k === " ";
    if (this.screen === "menu") {
      if (k >= "1" && k <= "9" && k.length === 1) {
        const i = Number(k) - 1;
        if (i < this.menuGames.length) this.start(this.menuGames[i]);
        return true;
      }
      if ((enter || space) && this.selected) {
        this.start(this.selected);
        return true;
      }
    } else if (this.screen === "countdown") {
      if (esc) this.toMenu();
      return esc;
    } else if (this.screen === "paused") {
      if (space || k === "p") this.resume();
      else if (k === "r") this.restart();
      else if (esc) this.toMenu();
      else return false;
      return true;
    } else if (this.screen === "playing") {
      if (this.gameOver) {
        if (k === "r" || space || enter) this.restart();
        else if (esc) this.toMenu();
        else return false;
      } else if (space || k === "p") this.pause();
      else if (k === "r") this.restart();
      else if (esc) this.toMenu();
      else return false;
      return true;
    }
    return false;
  }

  /** A mouse click or tap at canvas coordinates. */
  handleClick(x, y) {
    const p = [x, y];
    if (this.screen === "playing" && !this.gameOver) {
      if (this._pauseButton().contains(p)) this.pause();
      return;
    }
    const b = this._layout().find((btn) => btn.contains(p, 6));
    if (b) this._do(b.action);
  }

  _locateHand(controls, dt) {
    this._clock += dt;
    const pose = controls.pose;
    this.personSeen = pose != null;
    const other = this.hand === "right" ? "left" : "right";
    const main = this._trackers[this.hand].update(pose, this._clock);
    const spare = this._trackers[other].update(pose, this._clock);
    // Menus are forgiving: point with whichever hand is up.
    this.cursor = main ?? spare;
  }

  update(controls, dt) {
    this._locateHand(controls, dt);
    this._buttons = this._layout();

    if (this.screen === "countdown") {
      this.countdown -= dt;
      if (this.countdown <= 0) this.screen = "playing";
      return;
    }
    if (this.screen === "playing" && !this.gameOver) {
      this.game.update(controls, dt);
      const pb = this._pauseButton();
      if (this.pauseDwell.update(pb.contains(this.cursor) ? pb.action : null, dt)) this.pause();
      return;
    }
    if (this.screen === "playing" && this.gameOver) {
      if (!this._wasOver) this.dwell.block(); // PLAY AGAIN may appear under the hand
      this.game.update(controls, dt); // let debris finish falling
    }
    this._wasOver = this.screen === "playing" && this.gameOver;

    const hovered = this._buttons.find((b) => b.contains(this.cursor, 6))?.action ?? null;
    const fired = this.dwell.update(hovered, dt);
    if (fired) this._do(fired);
  }

  // --- layout ---

  _pauseButton() {
    const w = this.width, h = this.height;
    return new Button("pause", "II  PAUSE", [w - 190, h - 74, w - 22, h - 22]);
  }

  _layout() {
    const w = this.width, h = this.height;
    if (this.screen === "menu") {
      const n = Math.max(1, this.menuGames.length);
      const tw = Math.floor(w * 0.3), th = Math.floor(h * 0.34), gap = Math.floor(w * 0.04);
      let x = Math.floor((w - (n * tw + (n - 1) * gap)) / 2);
      const y0 = Math.floor(h * 0.36);
      return this.menuGames.map((id, i) => {
        const g = this.registry[id];
        const b = new Button(`start:${id}`, g.title || id.toUpperCase(), [x, y0, x + tw, y0 + th], g.blurb || "", String(i + 1));
        x += tw + gap;
        return b;
      });
    }
    const bw = Math.floor(w * 0.2), bh = 74, gap = Math.floor(w * 0.04);
    const y0 = this.screen === "paused" ? Math.floor(h * 0.56) : Math.floor(h * 0.8);
    let pair;
    if (this.screen === "paused") pair = [["resume", "RESUME", "SPACE"], ["menu", "MENU", "ESC"]];
    else if (this.screen === "playing" && this.gameOver) pair = [["again", "PLAY AGAIN", "R"], ["menu", "MENU", "ESC"]];
    else return [];
    let x = Math.floor(w / 2) - bw - Math.floor(gap / 2);
    return pair.map(([action, label, key]) => {
      const b = new Button(action, label, [x, y0, x + bw, y0 + bh], "", key);
      x += bw + gap;
      return b;
    });
  }

  // --- drawing ---

  render(ctx, frame) {
    this._buttons = this._layout();
    if (this.screen === "menu") this._drawMenu(ctx);
    else {
      this.game.render(ctx, frame);
      if (this.screen === "countdown") this._drawCountdown(ctx);
      else if (this.screen === "paused") this._drawPaused(ctx);
      else if (this.gameOver) {
        this._drawButtons(ctx, this.dwell);
        this._drawCursor(ctx, this.dwell);
      } else this._drawPlayChrome(ctx);
    }
    this._drawFooter(ctx);
  }

  _drawMenu(ctx) {
    dim(ctx, 0.62);
    const w = this.width, h = this.height;
    centredText(ctx, "MOTION ARCADE", w / 2, Math.floor(h * 0.17), 2.0, ACCENT);
    centredText(ctx, "Point with your right hand and hover over a game to start", w / 2, Math.floor(h * 0.25),
      0.72, WHITE, { body: true });
    this._drawButtons(ctx, this.dwell, true);
    let status, colour;
    if (!this.personSeen) [status, colour] = ["Step in front of the camera", WARN];
    else if (!this.cursor) [status, colour] = ["Raise your hand to point", WARN];
    else [status, colour] = ["Hand found - hold it over a game", GOOD];
    centredText(ctx, status, w / 2, Math.floor(h * 0.82), 0.8, colour);
    if (this.notice) {
      centredText(ctx, this.notice.text, w / 2, Math.floor(h * 0.89), 0.56, this.notice.colour ?? DIM,
        { body: true });
    }
    this._drawCursor(ctx, this.dwell);
  }

  _drawButtons(ctx, dwell, tiles = false) {
    for (const b of this._buttons) {
      const hot = dwell.target === b.action;
      const [x0, y0, x1, y1] = b.rect;
      panel(ctx, b.rect, { colour: hot ? HOT_FILL : PANEL, alpha: 0.85, border: hot ? ACCENT : IDLE_BORDER,
        thickness: hot ? 3 : 2 });
      const cx = (x0 + x1) / 2;
      if (tiles) {
        centredText(ctx, b.label, cx, y0 + 62, 1.05, hot ? ACCENT : WHITE);
        wrap(ctx, b.sub, x1 - x0 - 40, 0.58).forEach((line, i) =>
          centredText(ctx, line, cx, y0 + 108 + i * 28, 0.58, DIM, { body: true, shadow: false }));
        centredText(ctx, `hover to play  -  or press ${b.keyHint}`, cx, y1 - 24, 0.5, HINT,
          { body: true, shadow: false });
      } else {
        centredText(ctx, b.label, cx, y0 + 46, 0.85, hot ? ACCENT : WHITE);
        if (b.keyHint) centredText(ctx, b.keyHint, cx, y1 + 24, 0.45, HINT, { body: true, shadow: false });
      }
      if (hot && dwell.progress > 0) {
        const fill = (x1 - x0 - 24) * Math.min(1, dwell.progress);
        ctx.fillStyle = css(ACCENT);
        ctx.fillRect(x0 + 12, y1 - 10, fill, 5);
      }
    }
  }

  _drawCursor(ctx, dwell) {
    if (!this.cursor) return;
    const [x, y] = this.cursor;
    ctx.save();
    ctx.beginPath();
    ctx.arc(x, y, 22, 0, Math.PI * 2);
    ctx.lineWidth = 5;
    ctx.strokeStyle = "rgb(20,20,20)";
    ctx.stroke();
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#fff";
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fillStyle = "#fff";
    ctx.fill();
    if (dwell.target && dwell.progress > 0) {
      ctx.beginPath();
      ctx.arc(x, y, 30, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * Math.min(1, dwell.progress));
      ctx.lineWidth = 5;
      ctx.lineCap = "round";
      ctx.strokeStyle = css(ACCENT);
      ctx.stroke();
    }
    ctx.restore();
  }

  _drawCountdown(ctx) {
    dim(ctx, 0.45);
    const w = this.width, h = this.height;
    const n = Math.max(1, Math.ceil(this.countdown));
    centredText(ctx, "GET READY", w / 2, Math.floor(h * 0.3), 1.4, ACCENT);
    centredText(ctx, String(n), w / 2, Math.floor(h * 0.58), 5.0, WHITE, { outlineWidth: 10 });
    const tip = this.registry[this.selected]?.tip ?? "Stand back so your upper body is in frame";
    centredText(ctx, tip, w / 2, Math.floor(h * 0.72), 0.75, DIM, { body: true });
  }

  _drawPaused(ctx) {
    dim(ctx, 0.6);
    centredText(ctx, "PAUSED", this.width / 2, Math.floor(this.height * 0.4), 2.0, ACCENT);
    this._drawButtons(ctx, this.dwell);
    this._drawCursor(ctx, this.dwell);
  }

  /** The pause button, with a cursor only while the hand is near it. */
  _drawPlayChrome(ctx) {
    const b = this._pauseButton();
    const near = b.contains(this.cursor, 60);
    const hot = this.pauseDwell.target === b.action;
    panel(ctx, b.rect, { alpha: near ? 0.7 : 0.45, radius: 14, border: hot ? ACCENT : [130, 120, 120] });
    const [x0, y0, x1, y1] = b.rect;
    centredText(ctx, b.label, (x0 + x1) / 2, y0 + 34, 0.62, hot ? ACCENT : [220, 210, 210],
      { shadow: false, weight: 700 });
    if (hot && this.pauseDwell.progress > 0) {
      ctx.fillStyle = css(ACCENT);
      ctx.fillRect(x0 + 10, y1 - 9, (x1 - x0 - 20) * Math.min(1, this.pauseDwell.progress), 4);
    }
    if (near) this._drawCursor(ctx, this.pauseDwell);
  }

  _drawFooter(ctx) {
    const n = Math.max(1, this.menuGames.length);
    const keys = {
      menu: `1-${n} or click to start   H switch hand`,
      countdown: "ESC menu",
      paused: "SPACE resume   R restart   ESC menu",
      playing: this.gameOver ? "R play again   ESC menu" : "SPACE pause   R restart   ESC menu   H switch hand",
    }[this.screen];
    // Outlined: the footer sits over every game's art, light skies included.
    text(ctx, `${this.hand.toUpperCase()} HAND   ${keys}`, 18, this.height - 14, 0.44, [205, 198, 198],
      { body: true, weight: 600, outlineWidth: 3 });
  }
}
