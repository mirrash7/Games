// The arcade around the games: welcome screen, how-to-play cards, start,
// pause, stop, replay, and the leaderboards (port of shell.py; the tutorial
// and leaderboard screens are browser-only). Nothing starts until the player chooses, and a
// countdown gives them time to step back. Everything works standing away from
// the computer: hovering the hand over a button presses it, Kinect-style.
// Keys and mouse clicks do the same for anyone at the keyboard.

import { HandTracker } from "./core/hand.js";
import { qualifies, nameAllowed, cleanName, NAME_MAX, BOARD_SIZE } from "./leaderboard.js";
import {
  ACCENT, CYAN, DIM, PANEL, WHITE, GOOD, WARN, centredText, panel, wrap, dim, text, css, roundRectPath,
} from "./core/theme.js";

export const COUNTDOWN_SECONDS = 3.0;
export const MENU_DWELL = 1.0; // hover time to press a button
export const PAUSE_DWELL = 1.5; // longer in play, so a passing hand cannot pause by accident
export const KEY_DWELL = 0.7; // name entry: quicker, three letters should not take forever
export const ENTRY_DELAY = 1.6; // seconds of the game's own game-over art before asking for a name
export const RESULTS_DELAY = 2.2; // ...or before showing the leaderboard
const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

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
   * @param {object} [o.board] leaderboard (leaderboard.js); null turns leaderboards off
   */
  constructor({ width = 1280, height = 720, games = [], mirrored = true, selected = null, board = null } = {}) {
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

    this.board = board;
    this.scores = {}; // game id -> top entries, as last fetched
    this.lastEntryId = null; // highlights the player's own row on the results screen
    this.keyDwell = new Dwell(KEY_DWELL);
    this.entry = null; // {name, message, saving} while entering a name
    this._overT = 0;
    this._qualified = false;
    this._tutored = new Set(); // games whose how-to-play cards were seen this visit
    this.tutorialT = 0;
    this.refreshScores();
  }

  /** Fetch the boards again (on start-up, and when going back to the menu). */
  refreshScores(ids = this.menuGames) {
    if (!this.board) return Promise.resolve();
    return Promise.all(ids.map((id) => this.board.top(id).then((entries) => {
      this.scores[id] = entries;
    }).catch((err) => console.warn("[kp] leaderboard:", err))));
  }

  get boardScope() {
    return this.board?.scope ?? "device";
  }

  get score() {
    return Math.max(0, Math.round(this.game?.score ?? 0));
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
    this._session = null;
    if (!countdown) this._beginPlay();
    // A player's first round of each game this visit starts with how to play it.
    if (countdown && Cls.tutorial?.length && !this._tutored.has(id)) {
      this.screen = "tutorial";
      this.tutorialT = 0;
    }
    this._wasOver = false;
    this.entry = null;
    this.lastEntryId = null;
    this.dwell.reset();
    this.pauseDwell.reset();
  }

  /** Play starts: tell the leaderboard, which times the round on its side. */
  _beginPlay() {
    this.screen = "playing";
    this._session = this.board?.begin ? this.board.begin(this.selected).catch(() => null) : null;
  }

  /** Leave the how-to-play cards for the countdown. */
  endTutorial() {
    if (this.screen !== "tutorial") return;
    this._tutored.add(this.selected);
    this.screen = "countdown";
    this.countdown = COUNTDOWN_SECONDS;
    this.dwell.block();
  }

  /** The name-entry screen, for a score that made the board. */
  enterName() {
    this.screen = "entry";
    this.entry = { name: "", message: "", saving: false };
    this.keyDwell.block();
  }

  /** Type one character, delete one, or save. */
  _type(action) {
    const e = this.entry;
    if (!e || e.saving) return;
    e.message = "";
    if (action === "del") e.name = e.name.slice(0, -1);
    else if (action === "ok") this._saveName();
    else if (action === "skip") this.showResults();
    else if (action.startsWith("key:") && e.name.length < NAME_MAX) e.name = cleanName(e.name + action.slice(4));
  }

  _saveName() {
    const e = this.entry;
    if (!e.name) {
      e.message = "Pick at least one letter";
      return;
    }
    if (!nameAllowed(e.name)) {
      e.message = "Please pick another name";
      return;
    }
    e.saving = true;
    const id = this.selected, game = this.game;
    Promise.resolve(this._session).then((session) => this.board.submit(id, e.name, this.score, session))
      .then(({ entries, id: entryId }) => {
      this.scores[id] = entries;
      if (this.game === game && this.screen === "entry") {
        this.lastEntryId = entryId;
        this.showResults();
      }
    }).catch((err) => {
      console.warn("[kp] could not save the score:", err);
      if (this.screen === "entry") {
        e.saving = false;
        e.message = "Couldn't save - try again, or SKIP";
      }
    });
  }

  showResults() {
    this.screen = "results";
    this.entry = null;
    this.dwell.block();
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
    this.entry = null;
    this.dwell.block();
    this.refreshScores();
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
    else if (action === "play") this.endTutorial();
    else if (/^(key:.|del$|ok$|skip$)/.test(action)) this._type(action);
  }

  // --- input ---

  /** Apply a key (KeyboardEvent.key). Returns true if it was used. */
  handleKey(key) {
    const k = key.length === 1 ? key.toLowerCase() : key;
    if (this.screen === "entry") {
      // Every key types here, H and R included.
      if (/^[a-z0-9]$/.test(k)) this._type(`key:${k.toUpperCase()}`);
      else if (k === "Backspace") this._type("del");
      else if (k === "Enter") this._type("ok");
      else if (k === "Escape") this._type("skip");
      else return false;
      return true;
    }
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
    } else if (this.screen === "tutorial") {
      if (enter || space) this.endTutorial();
      else if (esc) this.toMenu();
      else return false;
      return true;
    } else if (this.screen === "results") {
      if (k === "r" || space || enter) this.restart();
      else if (esc) this.toMenu();
      else return false;
      return true;
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
        if (this._qualified && (space || enter)) this.enterName();
        else if (k === "r" || space || enter) this.restart();
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
      if (this.countdown <= 0) this._beginPlay();
      return;
    }
    if (this.screen === "tutorial") this.tutorialT += dt;
    if (this.screen === "playing" && !this.gameOver) {
      this.game.update(controls, dt);
      const pb = this._pauseButton();
      if (this.pauseDwell.update(pb.contains(this.cursor) ? pb.action : null, dt)) this.pause();
      return;
    }
    if (this.screen === "playing" && this.gameOver) {
      if (!this._wasOver) {
        this.dwell.block(); // PLAY AGAIN may appear under the hand
        this._overT = 0;
        // A score good enough for the board: no PLAY AGAIN button to hit by
        // accident; the name entry follows the game's own game-over moment.
        this._qualified = !!this.board && qualifies(this.scores[this.selected] ?? [], this.score);
      }
      this._overT += dt;
      this.game.update(controls, dt); // let debris finish falling
      if (this.board && this._overT >= (this._qualified ? ENTRY_DELAY : RESULTS_DELAY)) {
        if (this._qualified) this.enterName();
        else this.screen = "results"; // same buttons in the same place: keep the hover going
        this._wasOver = false;
        this._buttons = this._layout();
        return;
      }
    }
    this._wasOver = this.screen === "playing" && this.gameOver;
    this._buttons = this._layout();

    const dwell = this.screen === "entry" ? this.keyDwell : this.dwell;
    const hovered = this._buttons.find((b) => b.contains(this.cursor, this.screen === "entry" ? 0 : 6))?.action ?? null;
    const fired = dwell.update(hovered, dt);
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
    if (this.screen === "entry") return this._keyboard();
    const bw = Math.floor(w * 0.2), bh = 74, gap = Math.floor(w * 0.04);
    const y0 = this.screen === "paused" ? Math.floor(h * 0.56) : Math.floor(h * 0.8);
    let pair;
    if (this.screen === "paused") pair = [["resume", "RESUME", "SPACE"], ["menu", "MENU", "ESC"]];
    else if (this.screen === "tutorial") pair = [["play", "LET'S GO", "ENTER"], ["menu", "MENU", "ESC"]];
    else if (this.screen === "results") pair = [["again", "PLAY AGAIN", "R"], ["menu", "MENU", "ESC"]];
    else if (this.screen === "playing" && this.gameOver && !this._qualified) pair = [["again", "PLAY AGAIN", "R"], ["menu", "MENU", "ESC"]];
    else return [];
    let x = Math.floor(w / 2) - bw - Math.floor(gap / 2);
    return pair.map(([action, label, key]) => {
      const b = new Button(action, label, [x, y0, x + bw, y0 + bh], "", key);
      x += bw + gap;
      return b;
    });
  }

  /** A-Z, DEL and OK in a 7x4 grid, big enough to hit with a palm from 2 m; SKIP below. */
  _keyboard() {
    const cols = 7, kw = 112, kh = 78, gap = 10;
    const x0 = Math.round((this.width - (cols * kw + (cols - 1) * gap)) / 2), y0 = 250;
    const labels = [...LETTERS, "DEL", "OK"];
    const keys = labels.map((label, i) => {
      const r = Math.floor(i / cols), c = i % cols;
      const x = x0 + c * (kw + gap), y = y0 + r * (kh + gap);
      const action = label === "DEL" ? "del" : label === "OK" ? "ok" : `key:${label}`;
      return new Button(action, label, [x, y, x + kw, y + kh]);
    });
    const sw = 180;
    keys.push(new Button("skip", "SKIP", [this.width / 2 - sw / 2, 612, this.width / 2 + sw / 2, 664], "", "ESC"));
    return keys;
  }

  // --- drawing ---

  render(ctx, frame) {
    this._buttons = this._layout();
    if (this.screen === "menu") this._drawMenu(ctx);
    else if (this.screen === "tutorial") this._drawTutorial(ctx);
    else {
      this.game.render(ctx, frame);
      if (this.screen === "countdown") this._drawCountdown(ctx);
      else if (this.screen === "paused") this._drawPaused(ctx);
      else if (this.screen === "entry") this._drawEntry(ctx);
      else if (this.screen === "results") this._drawResults(ctx);
      else if (this.gameOver) {
        if (this._qualified) this._drawNewBest(ctx);
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
        if (this.board) {
          const top = this.scores[b.action.slice(6)]?.[0];
          centredText(ctx, top ? `HIGH SCORE   ${top.name}   ${top.score}` : "NO HIGH SCORES YET - SET ONE",
            cx, y1 - 62, 0.56, top ? ACCENT : DIM, { shadow: false, weight: 700 });
        }
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

  _drawTutorial(ctx) {
    dim(ctx, 0.78);
    const w = this.width, Cls = this.registry[this.selected];
    centredText(ctx, `HOW TO PLAY ${Cls.title}`, w / 2, 84, 1.25, ACCENT);
    const steps = Cls.tutorial, n = steps.length;
    const cw = 370, gap = 30, ch = 380, y0 = 120;
    let x = (w - (n * cw + (n - 1) * gap)) / 2;
    steps.forEach((step, i) => {
      // Cards appear one after another, so the eye reads them in order.
      const a = Math.min(1, Math.max(0, (this.tutorialT - i * 0.25) / 0.35));
      ctx.save();
      ctx.globalAlpha = a;
      panel(ctx, [x, y0, x + cw, y0 + ch], { alpha: 0.9, border: [120, 110, 110] });
      const demo = [x + 14, y0 + 14, x + cw - 14, y0 + 214];
      step.draw(ctx, demo, this.tutorialT);
      centredText(ctx, String(i + 1), x + 34, y0 + 48, 0.9, ACCENT); // over the demo, which may fill its area
      centredText(ctx, step.title, x + cw / 2, y0 + 256, 0.7, WHITE);
      wrap(ctx, step.text, cw - 44, 0.52).forEach((line, j) =>
        centredText(ctx, line, x + cw / 2, y0 + 292 + j * 26, 0.52, DIM, { body: true, shadow: false }));
      ctx.restore();
      x += cw + gap;
    });
    this._drawButtons(ctx, this.dwell);
    this._drawCursor(ctx, this.dwell);
  }

  _drawNewBest(ctx) {
    const pulse = 1 + 0.06 * Math.sin(this._overT * 9);
    centredText(ctx, "NEW HIGH SCORE!", this.width / 2, Math.floor(this.height * 0.86), 1.2 * pulse, CYAN);
  }

  _drawEntry(ctx) {
    dim(ctx, 0.88); // the game's own GAME OVER text would show through the slots
    const w = this.width, e = this.entry, Cls = this.registry[this.selected];
    const rank = 1 + (this.scores[this.selected] ?? []).filter((x) => x.score >= this.score).length;
    centredText(ctx, "NEW HIGH SCORE!", w / 2, 76, 1.35, ACCENT);
    centredText(ctx, `${Cls.title}   ${this.score}   #${rank}`, w / 2, 116, 0.65, WHITE, { weight: 700 });
    // Three slots; the next one to fill blinks.
    const sw = 78, sg = 16, sx = w / 2 - (NAME_MAX * sw + (NAME_MAX - 1) * sg) / 2;
    for (let i = 0; i < NAME_MAX; i++) {
      const x = sx + i * (sw + sg);
      const active = i === e.name.length && !e.saving && Math.floor(this._clock * 2.5) % 2 === 0;
      panel(ctx, [x, 136, x + sw, 222], { alpha: 0.9, radius: 12, border: active ? CYAN : [120, 110, 110], thickness: active ? 3 : 2 });
      if (e.name[i]) centredText(ctx, e.name[i], x + sw / 2, 200, 1.6, ACCENT);
    }
    const msg = e.saving ? "SAVING..." : e.message || "Hover over letters to spell your name - or type it";
    centredText(ctx, msg, w / 2, 242, 0.5, e.message ? WARN : DIM, { body: true, shadow: false, weight: 600 });
    for (const b of this._buttons) {
      const hot = this.keyDwell.target === b.action;
      const ready = b.action === "ok" && e.name.length > 0;
      const [x0, y0, x1, y1] = b.rect;
      panel(ctx, b.rect, { colour: hot ? HOT_FILL : PANEL, alpha: 0.88, radius: 12,
        border: hot ? ACCENT : ready ? CYAN : IDLE_BORDER, thickness: hot || ready ? 3 : 2 });
      const small = b.label.length > 1;
      centredText(ctx, b.label, (x0 + x1) / 2, (y0 + y1) / 2 + (small ? 10 : 15), small ? 0.72 : 1.05,
        hot ? ACCENT : ready ? CYAN : WHITE);
      if (hot && this.keyDwell.progress > 0) {
        ctx.fillStyle = css(ACCENT);
        ctx.fillRect(x0 + 8, y1 - 9, (x1 - x0 - 16) * Math.min(1, this.keyDwell.progress), 4);
      }
    }
    this._drawCursor(ctx, this.keyDwell);
  }

  _drawResults(ctx) {
    dim(ctx, 0.8);
    const w = this.width, Cls = this.registry[this.selected];
    const entries = this.scores[this.selected] ?? [];
    const px0 = w / 2 - 300, px1 = w / 2 + 300, y0 = 70, y1 = 548;
    panel(ctx, [px0, y0, px1, y1], { alpha: 0.88, border: ACCENT });
    centredText(ctx, `${Cls.title}  TOP ${BOARD_SIZE}`, w / 2, y0 + 50, 0.95, ACCENT);
    centredText(ctx, this.boardScope === "world" ? "WORLDWIDE" : "ON THIS DEVICE", w / 2, y0 + 78, 0.46, DIM,
      { body: true, shadow: false, weight: 700 });
    if (!entries.length) {
      centredText(ctx, "No scores yet - be the first!", w / 2, y0 + 200, 0.62, DIM, { body: true });
    }
    entries.slice(0, BOARD_SIZE).forEach((en, i) => {
      const y = y0 + 122 + i * 32;
      const mine = this.lastEntryId != null && en.id === this.lastEntryId;
      if (mine) {
        ctx.save();
        roundRectPath(ctx, px0 + 24, y - 24, px1 - 24, y + 8, 8);
        ctx.fillStyle = css(CYAN, 0.18);
        ctx.fill();
        ctx.restore();
      }
      const colour = mine ? CYAN : WHITE;
      text(ctx, `${i + 1}.`, px0 + 110, y, 0.62, mine ? CYAN : DIM, { align: "right", shadow: false, weight: 700 });
      text(ctx, en.name, px0 + 150, y, 0.66, colour, { shadow: false, weight: 800 });
      text(ctx, String(en.score), px1 - 110, y, 0.66, mine ? CYAN : ACCENT, { align: "right", shadow: false, weight: 800 });
    });
    if (this.lastEntryId == null) {
      centredText(ctx, `YOUR SCORE   ${this.score}`, w / 2, y1 - 18, 0.62, WHITE, { weight: 700, shadow: false });
    }
    this._drawButtons(ctx, this.dwell);
    this._drawCursor(ctx, this.dwell);
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
      tutorial: "ENTER play   ESC menu",
      entry: "type your name   BACKSPACE delete   ENTER save   ESC skip",
      results: "R play again   ESC menu",
      paused: "SPACE resume   R restart   ESC menu",
      playing: this.gameOver ? "R play again   ESC menu" : "SPACE pause   R restart   ESC menu   H switch hand",
    }[this.screen];
    // Outlined: the footer sits over every game's art, light skies included.
    text(ctx, `${this.hand.toUpperCase()} HAND   ${keys}`, 18, this.height - 14, 0.44, [205, 198, 198],
      { body: true, weight: 600, outlineWidth: 3 });
  }
}
