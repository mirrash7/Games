// Raccoon Dance rendering under Node (port of the logic half of
// tests/test_dance_render.py): every phase draws through a fake 2D context,
// the renderer never mutates the game, the coach has the size and the mirror
// convention the layout promises, and the tutorial cards draw. The pixels are
// checked by eye in web/dev/dance.html.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { SilentPlayer } from "../js/core/audio.js";
import { ENTRY_DELAY } from "../js/shell.js";
import { emptyControls } from "../js/core/controls.js";
import { Placed, parseLibrary, parseMoves, parseSong } from "../js/games/dance/chart.js";
import { DanceRaccoonGame } from "../js/games/dance/game.js";
import { stubArt } from "../js/games/dance/art.js";
import {
  CARD_BIG, CARD_SMALL, COACH_ROI, COACH_X, CoachPainter, DanceRenderer, FOOTER_Y, GROW_BEATS, HUD_BADGE_ORG, INTRO_C,
  LABEL_Y, LANE_CLIP_BOTTOM, LANE_RECT, LANE_Y, PAUSE_RECT, PIP_RECT, PX_PER_BEAT, RESULTS_RECT, RING, SUPERSEDE_FADE,
  TARGET_X, cardWidth, endingStart, kick, ratingKey,
} from "../js/games/dance/render.js";
import { DANCE_TUTORIAL } from "../js/games/dance/tutorial.js";
import { keypointsFor } from "../js/games/dance/pose.js";

/** A 2D context that accepts every call and records a few. */
function fakeContext(log) {
  const target = {
    canvas: { width: 1280, height: 720 },
    filter: "none",
    createLinearGradient: () => ({ addColorStop() {} }),
    createRadialGradient: () => ({ addColorStop() {} }),
    measureText: (s) => ({ width: String(s).length * 12 }),
    drawImage: (img, ...args) => {
      if (!img || !(img.width > 0)) throw new Error("drawImage of nothing");
      log.push(["drawImage", img, args]);
    },
    fillText: (str) => log.push(["fillText", String(str)]),
  };
  return new Proxy(target, {
    get(t, k) {
      if (k in t) return t[k];
      return () => {};
    },
    set(t, k, v) {
      t[k] = v;
      return true;
    },
  });
}

function fakeCanvasFactory(log) {
  return (w, h) => {
    const c = { width: Math.max(1, Math.round(w)), height: Math.max(1, Math.round(h)) };
    c.getContext = () => fakeContext(log);
    return c;
  };
}

const MOVES = {
  arms_up: { label: "Arms Up", beats: 2, keyframes: [{ beat: 0, pose: { la: [170, 175], ra: [170, 175] } }] },
  t_pose: { label: "T Pose", beats: 2, keyframes: [{ beat: 0, pose: { la: [90, 90], ra: [90, 90] } }] },
  disco_r: { label: "Disco Point", beats: 2, keyframes: [
    { beat: 0, pose: { ra: [150, 165], la: [20, 10], lean: 8 } },
    { beat: 1, pose: { ra: [-30, -40], la: [40, 20], lean: -4 } }], judge: [0, 1] },
  disco_l: { mirror_of: "disco_r" },
  squat_clap: { label: "Squat Clap", beats: 2, keyframes: [
    { beat: 0, pose: { la: [60, -20], ra: [60, -20], squat: 0.9, ll: [30, 5], rl: [30, 5] } }] },
};
const CHART = ["disco_r", "disco_l", "t_pose", "arms_up", "squat_clap", "disco_r", "arms_up", "t_pose"];
const SONG = parseSong({
  id: "test_groove", title: "Test Groove", artist: "KP Arcade", bpm: 120, offset: 0, duration: 20, difficulty: 1,
  coach: { outfit: "neon" }, sections: [{ name: "intro", beat: 0 }, { name: "verse", beat: 16 }],
  moves: CHART.map((m, i) => ({ move: m, beat: 16 + 2 * i, ...(i === 3 ? { gold: true } : {}) })),
}, "test_groove", parseMoves(MOVES, "test"));

test("every phase renders, and the renderer never mutates the game", () => {
  const log = [];
  const clock = { t: 0 };
  const game = new DanceRaccoonGame({ songs: [SONG], player: new SilentPlayer(22, { clock: () => clock.t }) });
  const renderer = new DanceRenderer(1280, 720, stubArt(), { makeCanvas: fakeCanvasFactory(log) });
  const ctx = fakeContext(log);
  const frame = fakeCanvasFactory(log)(1280, 720);
  const phases = new Set();
  for (let i = 0; i < 60 * 30; i++) {
    clock.t += 1 / 60;
    const st = game.songTime;
    const pose = i % 200 < 150 ? keypointsFor(SONG.coachPose(SONG.beatOf(st))) : null;
    game.update({ ...emptyControls(pose), present: pose != null }, 1 / 60);
    if (i % 7 === 0 || game.phase === "game_over") {
      const before = JSON.stringify([game.phase, game.songTime, game.score, game.feedback, game.upcoming,
        game.coachPose, game.counts, game.goldFlash, game.notice]);
      renderer.draw(ctx, frame, game);
      const after = JSON.stringify([game.phase, game.songTime, game.score, game.feedback, game.upcoming,
        game.coachPose, game.counts, game.goldFlash, game.notice]);
      assert.equal(after, before, "the renderer only reads the game");
    }
    phases.add(game.phase);
  }
  assert.deepEqual([...phases], ["intro", "dancing", "outro", "game_over"]);
  assert.ok(log.length > 1000, "it drew");
});

test("the coach stays inside its box at the size the layout promises", () => {
  const p = new CoachPainter(stubArt(), "neon", { makeCanvas: fakeCanvasFactory([]) });
  for (const pose of [{}, { la: [170, 175], ra: [170, 175] }, { la: [90, 90], ra: [90, 90] },
    { lean: 16, head: 8, la: [120, 150], shift: 0.4 }, { la: [60, -20], ra: [60, -20], squat: 0.9 },
    { la: [200, 210], ra: [200, 210] }]) {
    const g = p.layout(pose);
    const pts = g.silhouette.flat();
    const xs = pts.map((q) => q[0]), ys = [...pts.map((q) => q[1]), g.headTop];
    assert.ok(Math.min(...xs) > COACH_ROI[0] && Math.max(...xs) < COACH_ROI[2], JSON.stringify(pose));
    assert.ok(Math.min(...ys) > COACH_ROI[1] && Math.max(...ys) < COACH_ROI[3], JSON.stringify(pose));
  }
  const g = p.layout({});
  const ys = g.silhouette.flat().map((q) => q[1]);
  const xs = g.silhouette.flat().map((q) => q[0]);
  assert.ok(Math.max(...ys) >= 640 && Math.max(...ys) <= 668, "feet at y ~660");
  assert.ok(Math.max(...ys) - g.headTop >= 470 && Math.max(...ys) - g.headTop <= 560, "about 520 px head to toe");
  assert.ok(Math.abs((Math.min(...xs) + Math.max(...xs)) / 2 - COACH_X) < 60, "centred near x 520");
});

test("mirror convention: the coach's screen-right arm is on the screen's right", () => {
  const p = new CoachPainter(stubArt(), "neon", { makeCanvas: fakeCanvasFactory([]) });
  const g = p.layout({ ra: [170, 175] });
  assert.ok(g.j.r_hand[0] > COACH_X && g.j.r_hand[1] < g.headC[1]);
  assert.ok(g.j.l_hand[1] > g.j.hip[1] - 60);
});

test("the tutorial has three cards that draw with and without art", () => {
  assert.deepEqual(DANCE_TUTORIAL.map((s) => s.title), ["STAND BACK", "COPY LIKE A MIRROR", "HIT THE POSES ON THE BEAT"]);
  const ctx = fakeContext([]);
  for (const step of DANCE_TUTORIAL) for (const t of [0, 0.5, 1.3, 2.9, 7.7]) step.draw(ctx, [40, 200, 400, 400], t);
  assert.equal(DanceRaccoonGame.tutorial, DANCE_TUTORIAL);
});

test("helpers", () => {
  assert.equal(ratingKey("PERFECT"), "perfect");
  assert.equal(ratingKey(null), null);
  assert.equal(ratingKey("unseen"), null);
  assert.ok(Number.isFinite(kick(3.5)));
});

// --- after QA (port of the newer tests in tests/test_dance_render.py) ---------

const ASSETS = new URL("../../assets/dance/", import.meta.url);
const readJSON = (p) => JSON.parse(readFileSync(new URL(p, ASSETS), "utf8"));

/** A plain object with the state docs/DANCE.md section 8 promises, at a song time. */
function fakeGame(song, songTime, phase = "dancing") {
  return {
    song, phase, songTime, beat: song.beatOf(songTime), coachPose: song.coachPose(song.beatOf(songTime)),
    upcoming: [], feedback: [], score: 0, displayScore: 0, stars: 0, combo: 0, goldFlash: 0, notice: null,
    counts: { perfect: 0, super: 0, good: 0, ok: 0, x: 0 }, player: null, lastRating: null, resultsT: 0,
  };
}

test("the coach's bow waits for the chart's own ending", () => {
  // Neon Nights ends finale (judged, beat 192) -> bow (unjudged, beat 196). The
  // renderer's bow starts with the chart's bow, so the finale is held on its
  // beat and the coach bows once.
  const song = parseSong(readJSON("songs/neon_nights/song.json"), "neon_nights", parseLibrary(readJSON("moves.json")));
  const jp = song.judgePoints();
  const trailing = song.chart.filter((p) => p.beat > jp[jp.length - 1].beat);
  assert.ok(trailing.length && !trailing.some((p) => p.move.judge.length));
  assert.equal(endingStart(song), song.timeOf(trailing[0].beat));
  const lib = parseMoves({ m: { beats: 2, keyframes: [{ beat: 0, pose: {} }] } }, "t");
  song.chart = [new Placed(lib.get("m"), 16.0)];
  assert.equal(endingStart(song), -Infinity, "no chart ending: bow when the outro starts");
});

test("the results have landed before the shell asks for a name", () => {
  // The shell covers the results with its name entry after ENTRY_DELAY and
  // stops updating the game, so the count-up must be done by then.
  const log = [];
  const g = fakeGame(SONG, 19.0, "game_over");
  Object.assign(g, { score: 9463, stars: 5.0, resultsT: ENTRY_DELAY - 0.1 });
  new DanceRenderer(1280, 720, stubArt(), { makeCanvas: fakeCanvasFactory(log) })
    .draw(fakeContext(log), fakeCanvasFactory(log)(1280, 720), g);
  const drawn = log.filter((e) => e[0] === "fillText").map((e) => e[1]);
  assert.ok(drawn.includes("9463"), drawn.join(" | "));
});

test("lane cards every beat grow smoothly and never overlap", () => {
  const step = 2.0 / 60 / 4; // a quarter frame at 120 BPM, in beats
  const ds = [];
  for (let d = -0.65; d < 6.0; d += step) ds.push(d);
  const widths = ds.map(cardWidth);
  for (let i = 1; i < widths.length; i++) assert.ok(Math.abs(widths[i] - widths[i - 1]) <= 4, "a step at a time");
  assert.ok(ds.filter((d) => d <= 0).every((d) => cardWidth(d) === CARD_BIG), "full size on and past the ring");
  assert.ok(ds.filter((d) => d >= GROW_BEATS).every((d) => cardWidth(d) === CARD_SMALL));
  for (const d of ds.filter((x) => x >= 0)) {
    const gap = PX_PER_BEAT - (cardWidth(d) + cardWidth(d + 1.0)) / 2;
    assert.ok(gap >= 8, `cards a beat apart overlap at d=${d.toFixed(2)}`);
  }
});

test("an older popup is gone once the newer one is a few frames old; none under a notice", () => {
  assert.ok(SUPERSEDE_FADE <= 0.06);
  const art = stubArt();
  const word = art.words.perfect;
  const count = (feedback, notice = null) => {
    const log = [];
    const g = fakeGame(SONG, 15.0);
    g.feedback = feedback;
    g.notice = notice;
    new DanceRenderer(1280, 720, art, { makeCanvas: fakeCanvasFactory(log) })
      .draw(fakeContext(log), fakeCanvasFactory(log)(1280, 720), g);
    return log.filter((e) => e[0] === "drawImage" && e[1] === word).length;
  };
  const two = [{ rating: "perfect", gold: false, age: 0.5 }, { rating: "perfect", gold: false, age: SUPERSEDE_FADE }];
  assert.equal(count(two), 1, "only the newer word is drawn");
  assert.equal(count([{ rating: "perfect", gold: false, age: 0.5 },
    { rating: "perfect", gold: false, age: SUPERSEDE_FADE / 2 }]), 2, "mid hand-over both show");
  assert.equal(count(two, "STEP BACK - SHOW YOUR ARMS"), 0, "the notice hides the words");
});

// --- difficulty badges, bigger cards and PiP ------------------------------------

/**
 * A 2D context that records where images and text land on the screen: the
 * destination rect of every drawImage and an estimate of every fillText,
 * clipped by the current clip path's bounding box (save/restore aware).
 * Transforms aren't tracked; callers record only the flat layers.
 */
function boundsContext() {
  const marks = [];
  let clip = [-Infinity, -Infinity, Infinity, Infinity];
  const stack = [];
  let box = null;
  const grow = (x0, y0, x1, y1) => {
    box = box ? [Math.min(box[0], x0), Math.min(box[1], y0), Math.max(box[2], x1), Math.max(box[3], y1)]
      : [x0, y0, x1, y1];
  };
  const mark = (img, r) => {
    const c = [Math.max(r[0], clip[0]), Math.max(r[1], clip[1]), Math.min(r[2], clip[2]), Math.min(r[3], clip[3])];
    if (c[2] > c[0] && c[3] > c[1]) marks.push({ img, rect: c });
  };
  const t = {
    canvas: { width: 1280, height: 720 }, filter: "none", font: "10px x", textAlign: "left", recording: false,
    marks,
    save: () => stack.push(clip),
    restore: () => { clip = stack.pop() ?? clip; },
    beginPath: () => { box = null; },
    moveTo: (x, y) => grow(x, y, x, y),
    lineTo: (x, y) => grow(x, y, x, y),
    arcTo: (x1, y1, x2, y2) => grow(Math.min(x1, x2), Math.min(y1, y2), Math.max(x1, x2), Math.max(y1, y2)),
    rect: (x, y, w, h) => grow(x, y, x + w, y + h),
    clip: () => {
      if (box) clip = [Math.max(clip[0], box[0]), Math.max(clip[1], box[1]), Math.min(clip[2], box[2]), Math.min(clip[3], box[3])];
    },
    createLinearGradient: () => ({ addColorStop() {} }),
    createRadialGradient: () => ({ addColorStop() {} }),
    measureText: (str) => ({ width: String(str).length * fontSize() * 0.62 }),
    drawImage: (img, ...a) => {
      if (!t.recording) return;
      const [x, y, w, h] = a.length >= 8 ? a.slice(4) : a.length >= 4 ? a : [a[0], a[1], img.width, img.height];
      mark(img, [x, y, x + w, y + h]);
    },
    fillText: (str, x, y) => {
      if (!t.recording) return;
      const w = String(str).length * fontSize() * 0.62, h = fontSize();
      const x0 = t.textAlign === "center" ? x - w / 2 : x;
      mark(String(str), [x0, y - h * 0.8, x0 + w, y + h * 0.25]);
    },
  };
  const fontSize = () => Number(/(\d+)px/.exec(t.font)?.[1] ?? 10);
  return new Proxy(t, { get: (o, k) => (k in o ? o[k] : () => {}), set: (o, k, v) => { o[k] = v; return true; } });
}

/** A renderer whose flat layers (lane, HUD, feedback, PiP, intro, results) record into the context. */
function recordingRenderer(ctx) {
  const r = new DanceRenderer(1280, 720, stubArt(), { makeCanvas: fakeCanvasFactory([]) });
  for (const name of ["_drawLane", "_drawHud", "_drawFeedback", "_drawPip", "_drawIntro", "_drawResults"]) {
    const real = r[name].bind(r);
    r[name] = (c, ...rest) => {
      ctx.recording = true;
      try {
        return real(c, ...rest);
      } finally {
        ctx.recording = false;
      }
    };
  }
  return r;
}

const overlaps = (a, b) => a[0] < b[2] && b[0] < a[2] && a[1] < b[3] && b[1] < a[3];

/** A dancing game on `SONG` at song time `st`, its lane and popups filled in. */
function dancingAt(game, st, difficulty) {
  const g = fakeGame(SONG, st, st < 7.5 ? "intro" : "dancing");
  g.difficulty = difficulty;
  g.upcoming = SONG.pictograms().filter((p) => p.beat >= g.beat - 0.5).slice(0, 8)
    .map((p) => ({ beat: p.beat, moveLabel: p.label, pose: p.pose, gold: p.gold }));
  g.feedback = [{ rating: "perfect", gold: true, age: (st * 7) % 0.8 }];
  g.combo = 12;
  g.score = 4321;
  g.displayScore = 4321;
  g.stars = 2.4;
  g.lastRating = "super";
  g.player = { shoulderPx: 150, lean: 4, centrePx: [640, 330], hasHips: true, legs: {},
    arms: { l: { upper: 160, fore: 170, reach: 165 }, r: { upper: 40, fore: 30, reach: 35 } } };
  return g;
}

test("the difficulty badge is drawn on the intro card, beside the score and on the results", () => {
  for (const level of ["easy", "normal", "hard"]) {
    const ctx = boundsContext();
    const r = recordingRenderer(ctx);
    const frame = fakeCanvasFactory([])(1280, 720);
    const badgeMarks = (g) => {
      ctx.marks.length = 0;
      r.draw(ctx, frame, g);
      const badges = new Set([0.6, 0.65, 0.8].map((s) => r.badge(level, s)));
      return ctx.marks.filter((m) => badges.has(m.img)).map((m) => m.rect);
    };
    const intro = badgeMarks(dancingAt(null, 1.0, level));
    assert.ok(intro.some(([x0, y0, x1, y1]) => Math.abs((x0 + x1) / 2 - INTRO_C[0]) < 2
      && Math.abs((y0 + y1) / 2 - (INTRO_C[1] + 122)) < 40), `${level}: under the title card`);
    const hud = badgeMarks(dancingAt(null, 15.0, level));
    assert.ok(hud.some(([x0, , x1, y1]) => x0 >= HUD_BADGE_ORG[0] - 2 && y1 < 100 && x1 < PIP_RECT[0]),
      `${level}: right of the score`);
    const over = { ...dancingAt(null, 19.0, level), phase: "game_over", resultsT: 2.0, score: 9000, stars: 4.5 };
    const res = badgeMarks(over);
    const [x0, y0, x1] = RESULTS_RECT;
    assert.ok(res.some((b) => b[0] >= x0 && b[2] <= x1 && b[3] < y0 + 60), `${level}: on the results card`);
  }
  const r = new DanceRenderer(1280, 720, stubArt(), { makeCanvas: fakeCanvasFactory([]) });
  assert.equal(r.badge("expert"), null);
  assert.equal(r.badge(null), null);
  assert.equal(r.badge("hard"), r.badge("hard"), "cached");
});

test("badge colours read easy green, normal cyan, hard magenta", () => {
  const r = new DanceRenderer(1280, 720, stubArt(), { makeCanvas: fakeCanvasFactory([]) });
  const [e, n, h] = ["easy", "normal", "hard"].map((k) => r.difficultyColours[k]); // RGB
  assert.ok(e[1] > Math.max(e[0], e[2]), "easy is green");
  assert.ok(n[2] > n[0] && n[1] > n[0], "normal is cyan");
  assert.ok(h[0] > h[1] && h[2] > h[1], "hard is magenta");
});

test("cards and PiP are bigger and stay in bounds", () => {
  assert.ok(CARD_SMALL / 76 >= 1.15 && CARD_SMALL / 76 <= 1.25 && CARD_BIG / 108 >= 1.15 && CARD_BIG / 108 <= 1.25);
  const [x0, y0, x1, y1] = PIP_RECT;
  assert.deepEqual([x1 - x0, y1 - y0], [384, 216]);
  assert.ok(x0 >= 0 && y0 >= 16 && x1 <= 1280 - 16);
  const ch = CARD_BIG * 208 / 192; // the card art's aspect
  assert.ok(LANE_Y + ch / 2 <= LANE_CLIP_BOTTOM && LANE_CLIP_BOTTOM < PAUSE_RECT[1], "a full-size card isn't clipped");
  assert.ok(LANE_Y + RING / 2 < LABEL_Y - 18, "the label sits under the ring");
  assert.ok(LABEL_Y + 8 < FOOTER_Y);
  assert.ok(LANE_RECT[3] < PAUSE_RECT[1]);
  assert.ok(TARGET_X + GROW_BEATS * PX_PER_BEAT + CARD_BIG / 2 < PAUSE_RECT[0]);
});

test("nothing of ours enters the PAUSE button or the footer; the intro stays clear of the PiP", () => {
  const ctx = boundsContext();
  const r = recordingRenderer(ctx);
  const frame = fakeCanvasFactory([])(1280, 720);
  let lane = 0;
  for (let st = 0.5; st < 19.5; st += 0.095) {
    ctx.marks.length = 0;
    const g = dancingAt(null, st, "hard");
    r.draw(ctx, frame, g);
    for (const m of ctx.marks) {
      assert.ok(!overlaps(m.rect, PAUSE_RECT), `at ${st.toFixed(2)} s something drew into PAUSE: ${JSON.stringify(m.rect)}`);
      assert.ok(m.rect[3] <= FOOTER_Y, `at ${st.toFixed(2)} s something drew into the footer: ${JSON.stringify(m.rect)}`);
      if (overlaps(m.rect, LANE_RECT)) lane += 1;
      if (g.phase === "intro" && m.img !== frame && !overlaps(m.rect, [0, 0, 300, 140])) {
        // Intro text (not the HUD, not the PiP's own camera image) stays under the PiP.
        const [px0, py0, px1, py1] = PIP_RECT;
        const isPip = m.rect[0] >= px0 && m.rect[2] <= px1 && m.rect[1] >= py0 && m.rect[3] <= py1;
        if (!isPip) assert.ok(!overlaps(m.rect, [px0 - 8, py0, px1, py1 + 8]), `intro over the PiP at ${st}: ${m.img}`);
      }
    }
  }
  assert.ok(lane > 50, "the lane draws");
});
