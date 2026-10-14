// Raccoon Dance: copy the coach like a mirror (port of src/kpapp/game/dance/game.py;
// docs/DANCE.md).
//
// The game is a clock, a judge and some juice:
//
// - Clock. Song time is the audio player's position minus the song's offset,
//   never a sum of dt. The player dances to what they hear, so the judge must
//   use the same clock as the speakers. dt only animates juice. In the browser
//   the track is fetched and decoded after the game is built (during the
//   shell's tutorial and countdown); song time starts when it actually plays,
//   and until then the intro says LOADING.
// - Judge. judge.Judge keeps the best held match per judge point inside a
//   late-skewed window and rates it when the window closes. The player is read
//   from controls.pose, the extrapolated pose: for matching a held pose the
//   lower latency matters more than the steps.
// - Juice. Everything the renderer needs is plain state on the game
//   (docs/DANCE.md section 8, camelCased). The renderer reads it, never writes it.
//
// Phases: intro (title card, count-in) -> dancing (moves judged) -> outro (the
// last point has resolved, the music plays out) -> game_over.
//
// No DOM at module level: this file runs under Node for the tests.

import { Game } from "../base.js";
import { loadJSON, makeCanvas } from "../../core/gfx.js";
import { openPlayer as defaultOpenPlayer, SilentPlayer } from "../../core/audio.js";
import { ACCENT, CYAN, DANGER, DIM, WARN, centredText, css } from "../../core/theme.js";
import { ChartError, DEFAULT_DIFFICULTY, difficultyId, parseLibrary, parseSong, parseSongIds } from "./chart.js";
import { COMBO_FROM, Judge, UNSEEN, rulesFor } from "./judge.js";
import { forward, measure } from "./pose.js";
import { cachedArt, loadArt } from "./art.js";
import { DanceRenderer } from "./render.js";
import { DANCE_TUTORIAL } from "./tutorial.js";

export const Phase = Object.freeze({ INTRO: "intro", DANCING: "dancing", OUTRO: "outro", GAME_OVER: "game_over" });

export const FEEDBACK_SECONDS = 1.1;
export const FEEDBACK_MAX = 6;
export const UPCOMING = 8; // pictograms in the lane
export const UPCOMING_KEEP_BEATS = 0.5; // a card stays this long after its beat, to slide through the ring
export const DANCE_LEAD_BEATS = 1.0; // dancing starts this long before the first move
export const OUTRO_HOLD = 3.0; // s after the last move's end: game over even if the audio rings on
export const AUDIO_TAIL = 2.0; // s of ring-out after `duration` (docs/DANCE.md section 6)
export const NOTICE_AFTER = 0.6; // s of not seeing the player's arms before saying so
export const GOLD_FLASH_SECONDS = 0.8;
export const SCORE_EASE = 7.0; // 1/s: displayScore catches up with score
export const AUDIO_STALL = 2.0; // s of a frozen audio clock while playing before falling back to a silent one
// Browser only: s of game frames spent waiting for the track to decode before
// dancing without sound (openPlayer already falls back on an error).
export const AUDIO_LOAD_TIMEOUT = 15.0;

export const NOTICE_ARMS = "STEP BACK - SHOW YOUR ARMS";
export const NOTICE_NOBODY = "STEP INTO THE FRAME";

export const ASSET_URL = new URL("../../../assets/dance/", import.meta.url);

/** A rating popup: pops in above the coach's head, rises and fades. */
export class Feedback {
  constructor(rating, gold, { age = 0.0, x = 520.0, y = 150.0, label = "", value = 0.0, similarity = 0.0,
    combo = 0 } = {}) {
    this.rating = rating; // "perfect" | "super" | "good" | "ok" | "x"
    this.gold = gold;
    this.age = age; // seconds since it appeared
    this.x = x; // where it pops (docs/DANCE.md section 10)
    this.y = y;
    this.label = label; // the move's label
    this.value = value;
    this.similarity = similarity;
    this.combo = combo; // the combo after this rating
  }

  /** 1 -> 0 over FEEDBACK_SECONDS. */
  get life() {
    return Math.max(0.0, 1.0 - this.age / FEEDBACK_SECONDS);
  }
}

// --- loading (preload) ------------------------------------------------------

let loaded = null; // {library, songs: [Song], ids}

/** The songs preload() loaded (null before it ran). */
export function loadedSongs() {
  return loaded;
}

/**
 * Fetch moves.json, songs/index.json and every song.json, validated. A song
 * that fails validation is skipped with a warning (like DanceGame.variants()),
 * a missing move library means "no shared moves" (songs may define their own),
 * and a malformed library is an error.
 */
export async function loadSongs({ base = ASSET_URL, fetchJSON = loadJSON } = {}) {
  const url = (p) => new URL(p, base).href;
  let library = new Map();
  let libData = null;
  try {
    libData = await fetchJSON(url("moves.json"));
  } catch (err) {
    console.warn(`[kp] dance: no move library (${err.message})`);
  }
  if (libData != null) library = parseLibrary(libData, "assets/dance/moves.json");
  let ids = [];
  try {
    ids = parseSongIds(await fetchJSON(url("songs/index.json")), "assets/dance/songs/index.json");
  } catch (err) {
    console.warn(`[kp] dance: no song list (${err.message})`);
  }
  const songs = (await Promise.all(ids.map(async (id) => {
    try {
      let data;
      try {
        data = await fetchJSON(url(`songs/${id}/song.json`));
      } catch (err) {
        throw new ChartError(`song '${id}': missing song.json (${err.message})`);
      }
      return parseSong(data, id, library);
    } catch (err) {
      console.warn(`[kp] dance: skipping song '${id}': ${err.message}`);
      return null;
    }
  }))).filter(Boolean);
  return { library, songs, ids: songs.map((s) => s.folder) };
}

/**
 * A song's leaderboard on a difficulty: "dance-<song>" on normal (the board it
 * had before difficulties), "dance-<song>_<difficulty>" otherwise. The server
 * accepts /^dance-[a-z0-9_]{1,40}$/.
 */
export function boardIdFor(songId, difficulty = null) {
  const level = difficultyId(difficulty);
  return level === DEFAULT_DIFFICULTY ? `dance-${songId}` : `dance-${songId}_${level}`;
}

/** A song's length as M:SS (the song cards' subtitle). */
export function songLength(duration) {
  const m = Math.floor(duration / 60);
  // Python's round(): half to even.
  const r = duration % 60;
  const ss = Math.abs(r % 1) === 0.5 ? 2 * Math.round(r / 2) : Math.round(r);
  return `${m}:${String(ss).padStart(2, "0")}`;
}

/** Variant entries for a list of songs: [{id, title, subtitle, boardId}]. */
export function songVariants(songs) {
  return songs.map((s) => ({
    id: s.folder,
    title: s.title,
    // The song's length, not its old "difficulty 1" rating: that read like the
    // EASY / NORMAL / HARD picker above the cards.
    subtitle: s.artist ? `${s.artist} - ${songLength(s.duration)}` : songLength(s.duration),
    boardId: `dance-${s.folder}`,
  }));
}

// --- the game ---------------------------------------------------------------

export class DanceRaccoonGame extends Game {
  static id = "dance";
  static title = "RACCOON DANCE";
  static blurb = "Bust a move with the raccoon. Copy every pose like a mirror!";
  static tip = "Stand back so your arms and upper body are in frame - copy the coach like a mirror";
  static tutorial = DANCE_TUTORIAL; // the shell's how-to-play cards
  static menu = true;
  static ownCountdown = true; // the song counts in 3-2-1 on the beat; the shell skips its own
  // The shell's difficulty picker reads these (docs/DANCE.md section 7a). NORMAL
  // keeps each song's original board, so scores from before difficulties stay put.
  static difficulties = Object.freeze([
    { id: "easy", label: "EASY", hint: "fewer moves, held longer" },
    { id: "normal", label: "NORMAL", hint: "the full routine" },
    { id: "hard", label: "HARD", hint: "every move, judged strictly" },
  ]);
  static defaultDifficulty = DEFAULT_DIFFICULTY;

  /** Load the songs once at boot. `opts` ({base, fetchJSON}) are for tests. */
  static async preload(opts = {}) {
    const [songs] = await Promise.all([
      loadSongs(opts),
      // Without the art the game still runs, on the debug view (like the desktop's).
      opts.art === false ? null : loadArt().catch((err) => console.warn(`[kp] dance: ${err.message}`)),
    ]);
    loaded = songs;
  }

  /** The songs, in menu order. [] before preload(). */
  static variants() {
    return loaded ? songVariants(loaded.songs) : [];
  }

  /**
   * @param {object} opts
   *   variant      song folder id (default: the first song)
   *   difficulty   "easy" | "normal" | "hard" (unknown ids are normal)
   *   songs        [Song] to pick from (default: what preload() loaded)
   *   player       an audio player (SilentPlayer with a fake clock in tests);
   *                default: the song's track via openPlayer, loaded asynchronously
   *   openPlayer   (url, duration) => Promise<player>, for tests
   *   assetBase    URL of assets/dance/ (the song's audio is songs/<id>/<audio>)
   */
  constructor({ width = 1280, height = 720, mirrored = true, seed = null, variant = null, songs = null,
    difficulty = DEFAULT_DIFFICULTY, player = null, openPlayer = defaultOpenPlayer, assetBase = ASSET_URL, art = null } = {}) {
    super({ width, height, mirrored, seed }); // nothing random yet; seed kept for the contract
    const list = songs ?? loaded?.songs ?? [];
    if (!list.length) throw new ChartError("no songs: await DanceRaccoonGame.preload() first");
    const base = variant == null ? list[0] : list.find((s) => s.folder === variant || s.id === variant);
    if (!base) throw new ChartError(`no song '${variant}'`);
    this.difficulty = difficultyId(difficulty);
    const song = base.forDifficulty(this.difficulty);
    this.song = song;
    this.rules = rulesFor(this.difficulty);
    this.variant = song.folder ?? song.id;
    this.boardId = boardIdFor(this.variant, this.difficulty);
    this.art = art;
    this._renderer = null;

    this._closed = false;
    this._loadingFor = 0.0;
    this.audio = player;
    if (player == null) {
      const url = new URL(`songs/${this.variant}/${song.audio}`, assetBase).href;
      Promise.resolve()
        .then(() => openPlayer(url, song.duration + AUDIO_TAIL))
        .catch((err) => {
          console.warn(`[kp] dance: audio unavailable (${err?.message ?? err}); playing silently`);
          return new SilentPlayer(song.duration + AUDIO_TAIL);
        })
        .then((p) => {
          if (this._closed || this.audio != null) p.close?.(); // dropped, or timed out meanwhile
          else this.audio = p;
        });
    }

    this._pictos = song.pictograms().map((p) => Object.freeze({ beat: p.beat, moveLabel: p.label, pose: p.pose,
      gold: p.gold }));
    this._pictoBeats = this._pictos.map((p) => p.beat);
    this._lastMoveEnd = song.timeOf(song.chart.reduce((m, p) => Math.max(m, p.end), 0.0));
    this.judge = new Judge(song, this.rules);
    this.reset();
  }

  // --- lifecycle ---

  reset() {
    if (this._started) this.audio?.stop();
    this.judge.reset();
    this.phase = Phase.INTRO;
    this._started = false;
    this._paused = false;
    this.songTime = -this.song.offset;
    this.beat = this.song.beatOf(this.songTime);
    this.coachPose = this.song.coachPose(this.beat);
    this.upcoming = this._pictos.slice(0, UPCOMING);
    this.feedback = [];
    this.displayScore = 0.0;
    this.player = null; // the latest Measured (or null)
    this.notice = null;
    this.goldFlash = 0.0;
    this.scoreBump = 0.0; // 1 -> 0 after the score changes (HUD pulse)
    this.resultsT = 0.0;
    this.ratings = []; // every resolved point, unseen included
    this._unseenFor = 0.0;
    this._clock = 0.0; // sum of clamped dt: juice only
    this._lastPos = null;
    this._posStill = 0.0;
  }

  onPause() {
    if (this._started && !this._paused) this.audio.pause();
    this._paused = true;
  }

  /**
   * Resume the music and forget what the judge saw before the pause: the
   * player moved (to the PAUSE and RESUME buttons) while it stood still.
   */
  onResume() {
    if (this._paused && this._started && !this._closed) this.audio.resume();
    this._paused = false;
    this.songTime = this._audioTime();
    this.judge.onResume(this.songTime);
    this._unseenFor = 0.0;
    this.notice = null;
    this._posStill = 0.0;
  }

  /** Stop the music; the shell calls this whenever it drops the game. */
  close() {
    this._closed = true;
    this.audio?.stop();
  }

  // --- state the renderer and shell read ---

  /** True until the track has loaded (the intro shows LOADING...). */
  get loading() {
    return this.audio == null;
  }

  get score() {
    return this.judge.score;
  }

  /** 0..5, continuous: the star bar's fill. */
  get stars() {
    return this.judge.starFill;
  }

  get starCount() {
    return this.judge.stars;
  }

  get superstar() {
    return this.judge.superstar;
  }

  get counts() {
    return this.judge.counts;
  }

  get combo() {
    return this.judge.combo;
  }

  get lastRating() {
    return this.judge.lastRating;
  }

  // --- the clock ---

  _audioTime() {
    return (this.audio ? this.audio.position() : 0.0) - this.song.offset;
  }

  /**
   * A real device that stops reporting progress must not freeze the game
   * forever: after AUDIO_STALL seconds, keep time silently.
   */
  _watchAudio(dt) {
    const pos = this.audio.position();
    if (this._paused || !this._started || this._closed || this.audio.finished) {
      this._lastPos = pos;
      this._posStill = 0.0;
      return;
    }
    if (this._lastPos != null && pos === this._lastPos) this._posStill += dt;
    else this._posStill = 0.0;
    this._lastPos = pos;
    if (this._posStill > AUDIO_STALL) {
      console.warn(`[kp] dance: the audio clock stopped at ${pos.toFixed(2)}s; keeping time silently`);
      this.audio.stop();
      this.audio = new SilentPlayer(this.song.duration + AUDIO_TAIL);
      this.audio.play(pos);
      this._posStill = 0.0;
    }
  }

  // --- update ---

  update(controls, dt) {
    dt = Math.min(Math.max(Number(dt) || 0, 0.0), 0.1);
    this._clock += dt;
    if (!this._started && !this._closed) {
      if (this.audio == null) {
        this._loadingFor += dt;
        if (this._loadingFor > AUDIO_LOAD_TIMEOUT) {
          console.warn("[kp] dance: the music is taking too long to load; dancing without it");
          this.audio = new SilentPlayer(this.song.duration + AUDIO_TAIL);
        }
      }
      if (this.audio != null) {
        this.audio.play(0.0);
        this._started = true;
      }
    }
    if (!this._paused && this.audio != null) this._watchAudio(dt);
    this.songTime = this._audioTime();
    this.beat = this.song.beatOf(this.songTime);
    this.coachPose = this.song.coachPose(this.beat);
    const i = bisectLeft(this._pictoBeats, this.beat - UPCOMING_KEEP_BEATS);
    this.upcoming = this._pictos.slice(i, i + UPCOMING);

    const pose = controls?.pose ?? null;
    this.player = measure(pose, this.mirrored);
    const seen = this.player != null && this.player.seesArms;

    if (this.phase !== Phase.GAME_OVER && this._started) {
      for (const r of this.judge.update(this.songTime, this.player)) this._onRating(r);
      this._advancePhase();
    }

    if (this.phase === Phase.DANCING && !seen) this._unseenFor += dt;
    else this._unseenFor = 0.0;
    this.notice = this._unseenFor > NOTICE_AFTER ? (this.player == null ? NOTICE_NOBODY : NOTICE_ARMS) : null;

    this._animate(dt);
  }

  _advancePhase() {
    if (this.phase === Phase.INTRO && this.judge.points.length
        && this.beat >= this.song.firstMoveBeat - DANCE_LEAD_BEATS) {
      this.phase = Phase.DANCING;
    }
    if ((this.phase === Phase.INTRO || this.phase === Phase.DANCING) && this.judge.done
        && (this.phase === Phase.DANCING || !this.judge.points.length)) {
      this.phase = Phase.OUTRO;
    }
    if (this.phase === Phase.OUTRO && (this.audio?.finished || this.songTime >= this._lastMoveEnd + OUTRO_HOLD)) {
      this.phase = Phase.GAME_OVER;
      this.resultsT = 0.0;
    }
  }

  _onRating(r) {
    this.ratings.push(r);
    if (r.name === UNSEEN) return;
    this.feedback.push(new Feedback(r.name, r.gold, { label: r.point.label, value: r.value,
      similarity: r.similarity ?? 0.0, combo: this.judge.combo }));
    if (this.feedback.length > FEEDBACK_MAX) this.feedback.splice(0, this.feedback.length - FEEDBACK_MAX);
    if (r.value > 0) this.scoreBump = 1.0;
    if (r.gold && COMBO_FROM.includes(r.name)) this.goldFlash = 1.0;
  }

  _animate(dt) {
    for (const f of this.feedback) f.age += dt;
    this.feedback = this.feedback.filter((f) => f.age < FEEDBACK_SECONDS);
    this.goldFlash = Math.max(0.0, this.goldFlash - dt / GOLD_FLASH_SECONDS);
    this.scoreBump = Math.max(0.0, this.scoreBump - dt / 0.35);
    const target = this.score;
    this.displayScore += (target - this.displayScore) * (1.0 - Math.exp(-SCORE_EASE * dt));
    if (Math.abs(target - this.displayScore) < 1.0) this.displayScore = target;
    if (this.phase === Phase.GAME_OVER) this.resultsT += dt;
  }

  // --- render ---

  /**
   * Build the renderer and fill its caches now, not on the first frame of the
   * song, where it showed as a hitch. The shell calls this right after it
   * builds the game, on the song pick. (The static preload() loads the assets once.)
   */
  preload() {
    const art = this.art ?? cachedArt();
    if (!art || this._renderer) return;
    try {
      const canvas = makeCanvas(this.width, this.height);
      // One throwaway frame fills the sprite caches (title, cards, words).
      this.render(canvas.getContext("2d"), canvas);
    } catch (err) {
      console.warn("[kp] dance: could not prepare the renderer early", err);
    }
  }

  render(ctx, frame) {
    const art = this.art ?? cachedArt();
    if (!art) {
      this._renderDebug(ctx);
      return;
    }
    if (!this._renderer) this._renderer = new DanceRenderer(this.width, this.height, art);
    this._renderer.draw(ctx, frame, this);
  }

  /**
   * A bare view so the game runs before the real renderer exists: the coach
   * as a stick figure, the score, the last rating.
   */
  _renderDebug(ctx) {
    const W = this.width, H = this.height;
    ctx.save();
    ctx.fillStyle = "rgba(0,0,0,0.65)";
    ctx.fillRect(0, 0, W, H);
    const sw = 110.0 * H / 720.0;
    const ox = W * 520 / 1280, oy = H * 660 / 720 - 1.67 * sw;
    const j = forward(this.coachPose);
    const px = (name) => [ox + j[name][0] * sw, oy + j[name][1] * sw];
    const bones = [["neck", "l_shoulder"], ["neck", "r_shoulder"], ["neck", "hip"], ["l_hip", "r_hip"]];
    for (const s of ["l", "r"]) {
      bones.push([`${s}_shoulder`, `${s}_elbow`], [`${s}_elbow`, `${s}_wrist`], [`${s}_wrist`, `${s}_hand`],
        [`${s}_hip`, `${s}_knee`], [`${s}_knee`, `${s}_ankle`]);
    }
    ctx.strokeStyle = css(CYAN);
    ctx.lineCap = "round";
    ctx.lineWidth = 8;
    ctx.beginPath();
    for (const [a, b] of bones) {
      ctx.moveTo(...px(a));
      ctx.lineTo(...px(b));
    }
    ctx.stroke();
    ctx.lineWidth = 6;
    ctx.beginPath();
    ctx.arc(...px("head"), 0.3 * sw, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.restore();
    centredText(ctx, String(this.score).padStart(5, "0"), 120, 60, 1.2, ACCENT);
    centredText(ctx, this.song.title.toUpperCase(), W / 2, 50, 0.9, ACCENT);
    if (this.loading) centredText(ctx, "LOADING...", W / 2, 110, 0.8, DIM);
    if (this.feedback.length) {
      const f = this.feedback[this.feedback.length - 1];
      const colour = f.rating === "x" ? DANGER : f.gold ? ACCENT : CYAN;
      centredText(ctx, f.rating.toUpperCase(), ox, 150, 1.6, colour);
    }
    if (this.upcoming.length) {
      centredText(ctx, `NEXT: ${this.upcoming[0].moveLabel.toUpperCase()}`, 1010, 600, 0.8, DIM);
    }
    if (this.notice) centredText(ctx, this.notice, W / 2, H / 2, 1.0, WARN);
    if (this.phase === Phase.GAME_OVER) centredText(ctx, `${this.starCount} STARS`, W / 2, 260, 1.4, ACCENT);
  }
}

/** Python's bisect.bisect_left on a sorted array. */
function bisectLeft(a, x) {
  let lo = 0, hi = a.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (a[mid] < x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}
