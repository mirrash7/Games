// Raccoon Dance: the judge is fair under real conditions (port of
// tests/test_dance_game.py; docs/DANCE.md sections 7-9).
//
// Every player here goes through the same input path as the site: the bot's
// pose becomes COCO keypoints (keypointsFor) with pixel noise, model results
// at 20 Hz arriving 150 ms after the moment they show, and the site's
// PoseExtrapolator projecting them to each 60 Hz render frame. Song time comes
// from a SilentPlayer on a fake clock, so runs are fast and deterministic.
// The bots draw their randomness from core/random.js rather than Python's
// Mersenne Twister, so scores differ slightly from the Python run; the bounds
// are the same. Exact parity is pinned by dance_parity.test.mjs.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { SilentPlayer } from "../js/core/audio.js";
import { emptyControls } from "../js/core/controls.js";
import { PoseExtrapolator } from "../js/core/extrapolate.js";
import { KP } from "../js/core/pose.js";
import { Rng } from "../js/core/random.js";
import { GAMES } from "../js/games/index.js";
import { parseLibrary, parseMoves, parseSong } from "../js/games/dance/chart.js";
import {
  AUDIO_LOAD_TIMEOUT, AUDIO_STALL, DanceRaccoonGame, NOTICE_ARMS, NOTICE_NOBODY, Phase, songVariants,
} from "../js/games/dance/game.js";
import { Judge, LOST_BEFORE, RATING_NAMES, RESUME_GRACE, WINDOW_EARLY, WINDOW_LATE, rate, starFill, starsFor }
  from "../js/games/dance/judge.js";
import { REST, keypointsFor, measure, mirrorPose, normalizePose, similarity } from "../js/games/dance/pose.js";

const W = 1280, H = 720;
const FPS = 60;
const near = (a, b, tol = 1e-6) => Math.abs(a - b) <= tol;

// --- a test song: asymmetric poses, so the wrong side is always wrong --------

const POSES = {
  point_r: { ra: [150, 165], la: [20, 10], lean: 6 },
  t_up_r: { la: [90, 90], ra: [170, 175] },
  wave_r: { ra: [90, 170], la: [30, -20] },
  diag: { la: [135, 135], ra: [45, 45] },
  flex_l: { la: [100, 180], ra: [45, 30] },
  cross: { la: [-30, -60], ra: [120, 150] },
};
POSES.point_l = mirrorPose(POSES.point_r);
POSES.t_up_l = mirrorPose(POSES.t_up_r);

export const MOVES = {
  m1: { label: "Point Up", beats: 2, keyframes: [{ beat: 0, pose: POSES.point_r }, { beat: 1, pose: POSES.t_up_l }] },
  m2: { label: "Wave", beats: 2, keyframes: [{ beat: 0, pose: POSES.wave_r }, { beat: 1, pose: POSES.diag }] },
  m3: { label: "Flex", beats: 2, keyframes: [{ beat: 0, pose: POSES.flex_l }, { beat: 1, pose: POSES.cross }] },
  m4: { label: "Reach", beats: 2, keyframes: [{ beat: 0, pose: POSES.t_up_r }, { beat: 1, pose: POSES.point_l }] },
  hold: { label: "Diagonal", beats: 4, keyframes: [{ beat: 0, pose: POSES.diag }] },
};
const LIB = parseMoves(MOVES, "test library");

function songJson({ id = "groove", title = "Test Groove", first = 8, last = 56, duration = 32.0 } = {}) {
  const order = ["m1", "m2", "m3", "m4", "m2", "m1", "hold", "m3", "m4"];
  const moves = [];
  let beat = first;
  for (let i = 0; ; i++) {
    const name = order[i % order.length];
    if (beat + MOVES[name].beats > last) break;
    moves.push({ move: name, beat, gold: i === 5 || i === 14 });
    beat += MOVES[name].beats;
  }
  return { version: 1, id, title, artist: "KP Arcade", bpm: 120, offset: 0.5, duration, difficulty: 1, moves };
}

const makeSongs = (specs = [songJson()], lib = LIB) => specs.map((s) => parseSong(s, s.id, lib));

function newGame(songs = makeSongs(), variant = null) {
  const clock = { t: 1000.0 };
  const probe = variant == null ? songs[0] : songs.find((s) => s.folder === variant);
  const player = new SilentPlayer(probe.duration + 2.0, { clock: () => clock.t });
  const game = new DanceRaccoonGame({ width: W, height: H, songs, variant, player });
  return { game, clock };
}

// --- the simulated input path -------------------------------------------------

/**
 * Run `game` with a player whose pose at song time `st` is `bot(st)` (a pose,
 * or null for "not in the frame"). Returns the phases seen.
 */
function simulate(game, clock, bot, { rateHz = 20.0, lag = 0.15, noise = 4.0, seed = 1, elbowDrop = 0.0,
  hips = true, seconds = null, onFrame = null, stopAtGameOver = true } = {}) {
  const rng = new Rng(seed);
  const extrap = new PoseExtrapolator(); // the site's own projection (gain 0.7)
  const pending = [];
  let nextCapture = clock.t;
  const phases = [game.phase];
  const limit = seconds ?? game.song.duration + 8.0;
  const visible = hips ? null : ["shoulder", "elbow", "wrist"];
  const frames = Math.trunc(limit * FPS);
  for (let f = 0; f < frames; f++) {
    clock.t += 1.0 / FPS;
    const now = clock.t;
    if (now >= nextCapture) {
      nextCapture += 1.0 / rateHz;
      const st = game._started ? game.audio.position() - game.song.offset : -game.song.offset;
      const target = bot(st);
      const poses = [];
      if (target != null) {
        const kp = keypointsFor(target, { visible });
        for (let i = 0; i < kp.xy.length; i++) kp.xy[i] += rng.uniform(-noise, noise);
        if (elbowDrop) {
          for (const name of ["left_elbow", "right_elbow"]) if (rng.random() < elbowDrop) kp.confidence[KP[name]] = 0;
        }
        poses.push(kp);
      }
      // The model sees the frame ~50 ms after capture; the game gets it at capture + lag.
      pending.push([now + lag, poses, now + 0.05]);
    }
    while (pending.length && pending[0][0] <= now) {
      const [, poses, ts] = pending.shift();
      extrap.update(poses, ts, now);
    }
    const poses = extrap.posesAt(now);
    const pose = poses[0] ?? null;
    const controls = { ...emptyControls(pose), present: pose != null, stepPose: extrap.held[0] ?? null };
    game.update(controls, 1.0 / FPS);
    onFrame?.(game);
    if (game.phase !== phases[phases.length - 1]) phases.push(game.phase);
    if (stopAtGameOver && game.phase === Phase.GAME_OVER) break;
  }
  return phases;
}

const perfect = (game, late = 0.0) => (st) => game.song.coachPose(game.song.beatOf(st - late));

/** Every limb segment 25-30 degrees off (a new direction every beat), 250 ms late. */
function sloppy(game, { late = 0.25, off = [25.0, 30.0], seed = 7 } = {}) {
  return (st) => {
    const b = game.song.beatOf(st - late);
    const p = normalizePose(game.song.coachPose(b));
    const r = new Rng(seed * 100003 + Math.trunc(b));
    for (const k of ["la", "ra", "ll", "rl"]) p[k] = p[k].map((a) => a + r.choice([-1, 1]) * r.uniform(...off));
    return p;
  };
}

const frozen = (game) => { const first = game.song.judgePoints()[0].pose; return () => first; };
const idle = () => () => REST;
function randomArms(game, seed = 3) {
  return (st) => {
    const r = new Rng(seed * 7919 + Math.trunc(st / 0.4)); // a new random pose every 0.4 s
    return { la: [r.uniform(-60, 220), r.uniform(-60, 220)], ra: [r.uniform(-60, 220), r.uniform(-60, 220)] };
  };
}
const wrongSide = (game) => (st) => mirrorPose(game.song.coachPose(game.song.beatOf(st)));

const report = (g) => `score ${g.score} stars ${g.starCount} counts ${JSON.stringify(g.counts)} unseen ${g.judge.unseen}`;

// --- the judge on its own ---------------------------------------------------

test("ratings and stars follow the spec tables", () => {
  assert.equal(rate(1.0)[0], "perfect");
  assert.deepEqual(rate(0.0), ["x", 0.0]);
  assert.deepEqual([0.95, 0.80, 0.65, 0.45, 0.2].map((s) => rate(s)[0]), ["perfect", "super", "good", "ok", "x"]);
  assert.deepEqual([1999, 2000, 4000, 6000, 8000, 9199, 9200, 10000].map(starsFor), [0, 1, 2, 3, 4, 4, 5, 5]);
  assert.ok(starFill(0) === 0 && near(starFill(1000), 0.5) && starFill(10000) === 5);
  assert.ok(near(starFill(8600), 4.5));
});

const OPPOSITE = { la: [170, 175], ra: [10, 5] }; // nothing like the first point (point_r)

function feed(j, start, end, measured, fps = 60) {
  const out = [];
  for (let t = start; t <= end + 1e-9; t += 1.0 / fps) out.push(...j.update(t, measured));
  return out;
}

test("the judge keeps the best held match in a late-skewed window", () => {
  const song = makeSongs()[0];
  const p0 = song.judgePoints()[0];
  const good = measure(keypointsFor(p0.pose)), bad = measure(keypointsFor(OPPOSITE));
  let j = new Judge(song);
  assert.deepEqual(feed(j, p0.time - 1.0, p0.time + 0.20, bad), []);
  feed(j, p0.time + 0.21, p0.time + 0.36, good); // late, held 0.15 s: the lag allowance
  let out = feed(j, p0.time + 0.37, p0.time + WINDOW_LATE + 0.02, bad);
  assert.deepEqual(out.map((r) => r.name), ["perfect"]);
  assert.equal(j.score, Math.round(10000 * 1.0 / j.totalWeight));
  // A pose held only before the window opens doesn't count.
  j = new Judge(song);
  feed(j, p0.time - 1.0, p0.time - WINDOW_EARLY - 0.02, good);
  out = feed(j, p0.time - WINDOW_EARLY, p0.time + WINDOW_LATE + 0.02, bad);
  assert.deepEqual(out.map((r) => r.name), ["x"]);
});

test("a limb sweeping through the pose is not a hold", () => {
  // Pins HOLD: a single matching frame would be PERFECT if the judge kept the best raw frame.
  const song = makeSongs()[0];
  const p0 = song.judgePoints()[0];
  const good = measure(keypointsFor(p0.pose)), bad = measure(keypointsFor(OPPOSITE));
  assert.ok(similarity(p0.pose, good)[0] > 0.99, "the passing frame alone is a perfect match");
  const j = new Judge(song);
  feed(j, p0.time - 0.3, p0.time - 0.01, bad);
  j.update(p0.time, good);
  j.update(p0.time + 1 / 60, good); // 2 frames: ~33 ms, under HOLD
  const out = feed(j, p0.time + 2 / 60, p0.time + WINDOW_LATE + 0.02, bad);
  assert.deepEqual(out.map((r) => r.name), ["x"]);
});

test("a stall that skips a window is unseen, never a miss", () => {
  const song = makeSongs()[0];
  const j = new Judge(song);
  const pts = song.judgePoints();
  const now = pts[3].time + 1.0;
  const out = j.update(now, null);
  const closed = pts.filter((p) => p.time + WINDOW_LATE < now);
  assert.ok(closed.length >= 4);
  assert.deepEqual(out.map((r) => r.name), closed.map(() => "unseen"));
  assert.ok(j.counts.x === 0 && j.score === 0);
});

test("losing the player early in a window is unseen, not a miss", () => {
  // A window that sees the player only on their way into the pose and then
  // loses them saw nothing to judge: unseen. Seen failing through the late half
  // of the window, it is still an X.
  const song = makeSongs()[0];
  const p0 = song.judgePoints()[0];
  const bad = measure(keypointsFor(OPPOSITE));
  let j = new Judge(song);
  feed(j, p0.time - 1.0, p0.time + 0.05, bad); // on the way in, then gone
  let out = feed(j, p0.time + 0.06, p0.time + WINDOW_LATE + 0.02, null);
  assert.deepEqual(out.map((r) => r.name), ["unseen"]);
  assert.equal(j.counts.x, 0);
  j = new Judge(song);
  feed(j, p0.time - 1.0, p0.time + LOST_BEFORE + 0.05, bad);
  out = feed(j, p0.time + LOST_BEFORE + 0.06, p0.time + WINDOW_LATE + 0.02, null);
  assert.deepEqual(out.map((r) => r.name), ["x"], "seen getting it wrong after the beat: a real miss");
});

test("a close player whose hands leave the top of the frame is still perfect", () => {
  // Standing close, the wrists leave the top of the frame on arms-up poses
  // (elbows stay in). Found in QA: a perfect dancer at 300 px shoulder width
  // took an X on every Clap Up of Neon Nights.
  const up = { la: [170, 175], ra: [170, 175] };
  const song = parseSong({ version: 1, id: "up", title: "Up", bpm: 120, duration: 20.0,
    moves: [{ move: "up", beat: 8 }], custom_moves: { up: { beats: 2, keyframes: [{ beat: 0, pose: up }] } } },
  "up", new Map());
  const p = song.judgePoints()[0];
  const close = { centrePx: [640, 300], shoulderPx: 300 };
  const j = new Judge(song);
  feed(j, p.time - 1.0, p.time - 0.1, measure(keypointsFor(REST, close))); // arms down, on the way
  const seen = measure(keypointsFor(up, close));
  assert.ok(["l", "r"].every((s) => seen.arms[s].reach == null), "both raised wrists are out of the frame");
  assert.ok(seen.seesArms, "the elbows still count as seeing the arms");
  const out = feed(j, p.time - 0.09, p.time + WINDOW_LATE + 0.02, seen);
  assert.deepEqual(out.map((r) => r.name), ["perfect"]);
});

// --- fairness: who earns what -------------------------------------------------

test("a perfect player with real lag and noise is a superstar", () => {
  const { game, clock } = newGame();
  const phases = simulate(game, clock, perfect(game));
  assert.deepEqual(phases, [Phase.INTRO, Phase.DANCING, Phase.OUTRO, Phase.GAME_OVER]);
  const n = game.song.judgePoints().length;
  assert.ok(game.score >= 9200 && game.starCount === 5, report(game));
  assert.ok(game.counts.perfect >= 0.95 * n, report(game));
  assert.ok(game.counts.x === 0 && game.judge.unseen === 0, report(game));
});

test("a sloppy player lands in super and good", () => {
  const { game, clock } = newGame();
  simulate(game, clock, sloppy(game));
  const c = game.counts, n = game.song.judgePoints().length;
  assert.ok(c.super + c.good >= 0.6 * n, report(game));
  assert.ok(c.perfect <= 0.3 * n, report(game));
  assert.ok(game.starCount >= 3 && game.starCount <= 4, report(game));
});

test("a flawless dancer a quarter second behind the coach is still perfect", () => {
  // Pins WINDOW_LATE = 0.5.
  const { game, clock } = newGame();
  simulate(game, clock, perfect(game, 0.25));
  assert.ok(game.score >= 9200 && game.counts.x === 0, report(game));
});

test("a good but imprecise dancer on a slow model is still a superstar", () => {
  // Pins PERFECT at 0.90: limbs 10-20 degrees off, 200 ms late, 15 Hz with 200 ms of lag.
  const { game, clock } = newGame();
  simulate(game, clock, sloppy(game, { late: 0.2, off: [10, 20] }), { rateHz: 15, lag: 0.2, noise: 6 });
  assert.ok(game.starCount === 5 && game.counts.x === 0, report(game));
});

test("holding the pose is what separates sloppy from perfect", () => {
  // Pins HOLD: judged by its best single frame, the sloppy dancer scores like a superstar.
  const { game, clock } = newGame();
  game.judge = new Judge(game.song, { hold: 0.0 });
  simulate(game, clock, sloppy(game));
  assert.ok(game.score >= 9200, report(game));
});

test("players who don't dance the moves get at most one star", () => {
  for (const make of [frozen, idle, randomArms, wrongSide]) {
    const { game, clock } = newGame();
    simulate(game, clock, make(game));
    assert.ok(game.starCount <= 1, `${make.name}: ${report(game)}`);
  }
});

test("wrong side would score if mirroring were ignored", () => {
  for (const p of makeSongs()[0].judgePoints()) {
    assert.notDeepEqual(normalizePose(mirrorPose(p.pose)), normalizePose(p.pose));
  }
});

test("missing elbows and hips still reward a perfect player", () => {
  const { game, clock } = newGame();
  simulate(game, clock, perfect(game), { elbowDrop: 0.5, hips: false });
  assert.ok(game.starCount >= 4, report(game));
  assert.equal(game.counts.x, 0, report(game));
});

test("a player out of view loses only those points", () => {
  const { game, clock } = newGame();
  const pts = game.song.judgePoints();
  const gone = [pts[10].time - 0.5, pts[17].time + 0.5];
  const base = perfect(game);
  const notices = [];
  simulate(game, clock, (st) => (gone[0] <= st && st <= gone[1] ? null : base(st)),
    { onFrame: (g) => notices.push([g.songTime, g.notice]) });
  assert.equal(game.counts.x, 0, "tracking loss is never a miss");
  const hidden = game.ratings.filter((r) => gone[0] + 0.45 <= r.point.time && r.point.time <= gone[1] - 0.45);
  assert.ok(hidden.length && hidden.every((r) => r.name === "unseen"));
  assert.ok(game.judge.unseen >= hidden.length);
  const seenW = game.ratings.filter((r) => r.name !== "unseen").reduce((a, r) => a + r.point.weight, 0);
  const cap = Math.round(10000 * seenW / game.judge.totalWeight);
  assert.ok(game.score <= cap && game.score >= cap - 300, report(game));
  const shown = new Set(notices.filter(([t]) => gone[0] + 0.9 <= t && t <= gone[1] - 0.2).map(([, n]) => n));
  assert.deepEqual([...shown], [NOTICE_NOBODY]);
  assert.ok(notices.filter(([t]) => t > gone[1] + 0.6).every(([, n]) => n == null), "the notice clears");
});

test("arms out of frame says step back", () => {
  const { game, clock } = newGame();
  simulate(game, clock, perfect(game), { seconds: 8.0 });
  assert.equal(game.phase, Phase.DANCING);
  for (let i = 0; i < 60; i++) {
    const kp = keypointsFor(REST, { visible: ["shoulder"] });
    game.update({ ...emptyControls(kp), present: true }, 1 / 60);
    clock.t += 1 / 60;
  }
  assert.equal(game.notice, NOTICE_ARMS);
});

// --- pause, resume, close ---------------------------------------------------

test("pause freezes song time and never rates the pause", () => {
  const { game, clock } = newGame();
  const pts = game.song.judgePoints();
  const p = pts[12];
  // Dance perfectly until just after point 12's beat (its window already holds a PERFECT)...
  simulate(game, clock, perfect(game), { seconds: p.time + game.song.offset + 0.1 });
  assert.ok(p.time < game.songTime && game.songTime < p.time + WINDOW_LATE, `${game.songTime}`);
  const rated = game.ratings.length;
  game.onPause();
  const frozenAt = game.songTime;
  clock.t += 30.0; // a long pause; the shell doesn't update a paused game
  assert.ok(near(game.audio.position() - game.song.offset, frozenAt, 1e-9));
  game.onResume();
  assert.ok(near(game.songTime, frozenAt));
  // ...then the hand is on the RESUME button (arms nowhere near the move) for a moment.
  const resumed = game.songTime;
  const base = perfect(game);
  simulate(game, clock, (st) => (st < resumed + 0.8 ? { ra: [40, 40], la: [10, 5] } : base(st)));
  const byPoint = new Map(game.ratings.map((r) => [r.point.index, r]));
  assert.equal(game.ratings.length, pts.length, "every point resolved exactly once");
  assert.ok(rated <= 12);
  assert.notEqual(byPoint.get(12).name, "perfect", "no free PERFECT from the pose held before pausing");
  const excused = game.ratings.filter((r) => resumed - WINDOW_LATE <= r.point.time - WINDOW_EARLY
    && r.point.time - WINDOW_EARLY <= resumed + RESUME_GRACE);
  assert.ok(excused.length && excused.every((r) => r.name !== "x"), "a pause never turns into X");
  assert.equal(game.counts.x, 0, report(game));
  assert.equal(game.phase, Phase.GAME_OVER);
});

test("close stops the music", () => {
  const { game, clock } = newGame();
  simulate(game, clock, idle(game), { seconds: 3.0 });
  assert.ok(game.audio.playing);
  game.close();
  assert.ok(!game.audio.playing);
  game.update(emptyControls(), 1 / 60); // a stray frame after close doesn't restart it
  assert.ok(!game.audio.playing);
});

test("audio starts on the first frame, not at construction", () => {
  const { game, clock } = newGame();
  clock.t += 5.0; // the shell's tutorial and countdown
  assert.ok(!game.audio.playing && game.songTime === -game.song.offset);
  game.update(emptyControls(), 1 / 60);
  assert.ok(game.audio.playing && near(game.songTime, -game.song.offset));
});

test("song time comes from the audio clock, not dt", () => {
  const { game, clock } = newGame();
  game.update(emptyControls(), 1 / 60);
  clock.t += 2.0;
  game.update(emptyControls(), 1 / 60); // one frame of dt, two seconds of music
  assert.ok(near(game.songTime, 2.0 - game.song.offset));
});

// --- the browser's asynchronous audio ---------------------------------------

function deferred() {
  let resolve, reject;
  const promise = new Promise((a, b) => { resolve = a; reject = b; });
  return { promise, resolve, reject };
}

test("the round starts only when the track has loaded", async () => {
  const clock = { t: 0 };
  const d = deferred();
  const asked = [];
  const game = new DanceRaccoonGame({ songs: makeSongs(), openPlayer: (url, duration) => {
    asked.push([url, duration]);
    return d.promise;
  } });
  await null;
  assert.equal(asked.length, 1);
  assert.match(asked[0][0], /assets\/dance\/songs\/groove\/audio\.m4a$/);
  assert.equal(asked[0][1], 34.0, "duration + the ring-out tail");
  assert.ok(game.loading);
  for (let i = 0; i < 120; i++) game.update(emptyControls(), 1 / 60); // 2 s of LOADING...
  assert.equal(game.phase, Phase.INTRO);
  assert.equal(game.songTime, -game.song.offset, "song time waits for the music");
  const player = new SilentPlayer(34.0, { clock: () => clock.t });
  d.resolve(player);
  await new Promise((r) => setTimeout(r, 0));
  assert.ok(!game.loading && game.audio === player);
  game.update(emptyControls(), 1 / 60);
  assert.ok(player.playing && near(game.songTime, -game.song.offset));
  clock.t += 1.5;
  game.update(emptyControls(), 1 / 60);
  assert.ok(near(game.songTime, 1.0));
});

test("a game dropped while loading closes the late player; a failed load plays silently", async () => {
  const d = deferred();
  const game = new DanceRaccoonGame({ songs: makeSongs(), openPlayer: () => d.promise });
  game.close();
  let closed = false;
  d.resolve({ play() {}, close() { closed = true; }, stop() {}, position: () => 0 });
  await new Promise((r) => setTimeout(r, 0));
  assert.ok(closed && game.audio == null);

  const warn = console.warn;
  console.warn = () => {};
  try {
    const broken = new DanceRaccoonGame({ songs: makeSongs(), openPlayer: () => Promise.reject(new Error("nope")) });
    await new Promise((r) => setTimeout(r, 0));
    assert.ok(broken.audio instanceof SilentPlayer);
    const slow = new DanceRaccoonGame({ songs: makeSongs(), openPlayer: () => new Promise(() => {}) });
    for (let t = 0; t <= AUDIO_LOAD_TIMEOUT + 0.2; t += 0.1) slow.update(emptyControls(), 0.1);
    assert.ok(slow.audio instanceof SilentPlayer && slow.audio.playing, "a hung load gives up and dances silently");
  } finally {
    console.warn = warn;
  }
});

test("an audio clock that stops moving falls back to a silent one", () => {
  const stuck = { playing: false, finished: false, play() { this.playing = true; }, pause() {}, resume() {},
    stop() { this.playing = false; }, close() {}, position: () => 3.0 };
  const game = new DanceRaccoonGame({ songs: makeSongs(), player: stuck });
  const warn = console.warn;
  console.warn = () => {};
  try {
    for (let t = 0; t <= AUDIO_STALL + 0.2; t += 1 / 60) game.update(emptyControls(), 1 / 60);
  } finally {
    console.warn = warn;
  }
  assert.ok(game.audio instanceof SilentPlayer && game.audio.playing);
  assert.ok(game.audio.position() >= 3.0, "it carries on from where the music stopped");
});

// --- flow, state and registration ---------------------------------------------

test("score stops changing after game over", () => {
  const { game, clock } = newGame();
  simulate(game, clock, perfect(game));
  assert.equal(game.phase, Phase.GAME_OVER);
  const final = game.score, counts = { ...game.counts };
  simulate(game, clock, randomArms(game), { seconds: 3.0, stopAtGameOver: false });
  assert.equal(game.score, final);
  assert.deepEqual(game.counts, counts);
  assert.ok(game.resultsT > 2.5);
  assert.equal(game.displayScore, final);
});

test("the renderer state is complete", () => {
  const { game, clock } = newGame();
  const seen = { goldFlash: 0, feedback: 0, combo: 0 };
  simulate(game, clock, perfect(game), { seconds: 12.0, onFrame: (g) => {
    seen.goldFlash = Math.max(seen.goldFlash, g.goldFlash);
    seen.feedback = Math.max(seen.feedback, g.feedback.length);
    seen.combo = Math.max(seen.combo, g.combo);
    assert.ok(g.stars >= 0 && g.stars <= 5 && g.displayScore >= 0 && g.displayScore <= 10000);
    assert.ok(g.upcoming.length <= 8);
    assert.ok(g.upcoming.every((u) => u.beat >= g.beat - 0.5));
  } });
  for (const name of ["song", "phase", "songTime", "beat", "coachPose", "upcoming", "feedback", "score",
    "displayScore", "stars", "counts", "player", "lastRating", "notice", "goldFlash", "combo", "resultsT",
    "scoreBump", "starCount", "superstar", "ratings", "loading"]) {
    assert.ok(name in game, name);
  }
  normalizePose(game.coachPose);
  const up = game.upcoming[0];
  for (const k of ["beat", "moveLabel", "pose", "gold"]) assert.ok(k in up, k);
  assert.ok(seen.feedback >= 1 && seen.combo >= 5);
  assert.ok(seen.goldFlash > 0.95, "the gold move flashed");
  const f = game.feedback[game.feedback.length - 1];
  if (f) assert.ok(RATING_NAMES.includes(f.rating) && f.age >= 0 && f.age < 1.2 && f.life > 0);
  assert.ok(game.lastRating === "perfect" && game.player != null);
  assert.deepEqual(Object.keys(game.counts).sort(), ["good", "ok", "perfect", "super", "x"]);
});

test("variants list the songs and the game builds each", () => {
  const songs = makeSongs([songJson({ id: "groove", title: "Test Groove" }),
    songJson({ id: "slow_jam", title: "Slow Jam", first: 12 })]);
  const vs = songVariants(songs);
  assert.deepEqual(vs.map((v) => v.id), ["groove", "slow_jam"]);
  assert.deepEqual(vs[1], { id: "slow_jam", title: "Slow Jam", subtitle: "KP Arcade - difficulty 1",
    boardId: "dance-slow_jam" });
  let { game } = newGame(songs, "slow_jam");
  assert.ok(game.song.title === "Slow Jam" && game.boardId === "dance-slow_jam" && game.variant === "slow_jam");
  ({ game } = newGame(songs));
  assert.equal(game.song.id, "groove", "the first song in the index by default");
  assert.throws(() => newGame(songs, "nope"));
});

// --- the shipped song, through the real shell ---------------------------------

const ASSETS = new URL("../../assets/dance/", import.meta.url);
const readJSON = (p) => JSON.parse(readFileSync(new URL(p, ASSETS), "utf8"));
const diskFetch = async (href) => JSON.parse(readFileSync(new URL(href), "utf8"));

test("the shell lists Neon Nights and builds the game for it", async () => {
  const { Shell } = await import("../js/shell.js");
  await DanceRaccoonGame.preload({ base: ASSETS, fetchJSON: diskFetch, art: false });
  assert.ok(GAMES.includes(DanceRaccoonGame), "registered");
  assert.equal(GAMES.indexOf(DanceRaccoonGame), GAMES.length - 1, "after Flappy");
  assert.deepEqual(DanceRaccoonGame.variants().map((v) => [v.id, v.boardId]), [["neon_nights", "dance-neon_nights"]]);
  const sh = new Shell({ games: GAMES });
  assert.ok(sh.boardKeys().includes("dance-neon_nights"));
  const warn = console.warn;
  console.warn = () => {}; // no Web Audio under Node: the game falls back to silence
  try {
    sh.choose("dance");
    assert.equal(sh.screen, "songs");
    sh._do("song:neon_nights");
    assert.ok(sh.game instanceof DanceRaccoonGame && sh.game.song.title === "Neon Nights");
    assert.equal(sh.boardKey, "dance-neon_nights");
    await new Promise((r) => setTimeout(r, 0));
    assert.ok(sh.game.audio instanceof SilentPlayer);
    const game = sh.game;
    sh.toMenu();
    assert.ok(!game.audio.playing, "leaving stops the music");
  } finally {
    console.warn = warn;
  }
});

test("Neon Nights end to end", () => {
  const lib = parseLibrary(readJSON("moves.json"));
  const songs = [parseSong(readJSON("songs/neon_nights/song.json"), "neon_nights", lib)];
  const play = (make, opts = {}) => {
    const { game, clock } = newGame(songs);
    const phases = simulate(game, clock, make(game), opts);
    return { game, phases };
  };
  let { game, phases } = play(perfect);
  assert.deepEqual(phases, [Phase.INTRO, Phase.DANCING, Phase.OUTRO, Phase.GAME_OVER]);
  assert.ok(game.score >= 9200 && game.counts.x === 0, report(game));
  ({ game } = play(perfect, { elbowDrop: 0.5, hips: false }));
  assert.ok(game.starCount >= 4, report(game));
  ({ game } = play(sloppy));
  assert.ok(game.starCount >= 3 && game.starCount <= 4, report(game));
  for (const bot of [frozen, idle, randomArms]) {
    ({ game } = play(bot));
    assert.ok(game.starCount <= 1, `${bot.name}: ${report(game)}`);
  }
  // Not wrongSide: a third of Neon Nights' points are symmetric poses.
});
