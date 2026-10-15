// Dance charts: the move library and every shipped song, read from disk and
// validated by the browser's parser (port of the chart half of
// tests/test_dance_chart.py; the importer and CLI are desktop-only).

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, statSync } from "node:fs";
import {
  BAR_BEATS, ChartError, DIFFICULTIES, EASE_BEATS, EASY_MIN_GAP, Placed, deriveEasy, difficultyId, parseLibrary,
  parseMoves, parseSong, parseSongIds,
} from "../js/games/dance/chart.js";
import { REST, keypointsFor, measure, normalizePose, similarity } from "../js/games/dance/pose.js";
import { DanceRaccoonGame, loadSongs, loadedSongs, songVariants } from "../js/games/dance/game.js";

const ASSETS = new URL("../../assets/dance/", import.meta.url);
const readJSON = (p) => JSON.parse(readFileSync(new URL(p, ASSETS), "utf8"));

const LIB = parseLibrary(readJSON("moves.json"), "moves.json");
const IDS = parseSongIds(readJSON("songs/index.json"));
const SONGS = IDS.map((id) => parseSong(readJSON(`songs/${id}/song.json`), id, LIB));
const NEON = SONGS.find((s) => s.id === "neon_nights");

const ELBOWS_HIDDEN = ["shoulder", "wrist", "hip"];
const FREEZE_MAX = 0.55; // a player frozen in the previous pose must score below this

/** How well a perfect performance of `performed` matches `target`. */
const sim = (target, performed, opts = {}) => similarity(target, measure(keypointsFor(performed, opts)))[0];

// --- every song --------------------------------------------------------------

test("every listed song loads and validates", () => {
  assert.ok(IDS.length && IDS[0] === "neon_nights");
  assert.deepEqual(SONGS.map((s) => s.id), IDS);
  for (const song of SONGS) {
    const jp = song.judgePoints();
    assert.ok(jp.length, `${song.id} has nothing to judge`);
    assert.ok(song.firstMoveBeat >= 16, `${song.id}: moves must wait for the intro`);
    assert.ok(jp.every((p) => p.beat < song.beatOf(song.duration)));
  }
});

test("every song's audio is delivered", () => {
  for (const song of SONGS) {
    let size = 0;
    try {
      size = statSync(new URL(`songs/${song.folder}/${song.audio}`, ASSETS)).size;
    } catch {
      continue; // not generated yet (tools/generate_dance_song.py)
    }
    assert.ok(size > 10_000, song.id);
  }
});

test("a frozen player cannot score", () => {
  // Consecutive judged poses differ enough that holding the previous one
  // earns less than 0.55 on the next, with or without elbows in view.
  for (const song of SONGS) {
    const jp = song.judgePoints();
    for (let i = 1; i < jp.length; i++) {
      const [a, b] = [jp[i - 1], jp[i]];
      const full = sim(b.pose, a.pose);
      const reach = sim(b.pose, a.pose, { visible: ELBOWS_HIDDEN });
      assert.ok(full < FREEZE_MAX, `${song.id}: '${a.label}' at beat ${a.beat} scores ${full.toFixed(2)} on beat ${b.beat}`);
      assert.ok(reach < FREEZE_MAX, `${song.id}: without elbows '${a.label}' at beat ${a.beat} scores ${reach.toFixed(2)}`);
    }
  }
});

test("a perfect player matches every judged pose", () => {
  for (const song of SONGS) {
    for (const p of song.judgePoints()) {
      assert.ok(sim(p.pose, p.pose) > 0.97, `${p.label} at ${p.beat}`);
      assert.ok(sim(p.pose, p.pose, { visible: ELBOWS_HIDDEN }) > 0.97, `${p.label} at ${p.beat}, elbows hidden`);
      const close = sim(p.pose, p.pose, { centrePx: [640, 480], shoulderPx: 200 }); // hips below the frame
      assert.ok(close > 0.97, `${p.label} at ${p.beat}, standing close`);
    }
  }
});

// --- Neon Nights (docs/DANCE.md section 6) ------------------------------------

const SECTIONS = [["intro", 0], ["verse 1", 16], ["chorus 1", 48], ["verse 2", 80], ["chorus 2", 112],
  ["bridge", 144], ["final chorus", 160], ["outro", 192]];

const sectionMoves = (song, lo, hi) => song.chart.filter((p) => lo <= p.beat && p.beat < hi);

test("Neon Nights metadata", () => {
  assert.equal(NEON.title, "Neon Nights");
  assert.equal(NEON.artist, "KP Arcade");
  assert.equal(NEON.bpm, 120);
  assert.equal(NEON.offset, 0);
  assert.equal(NEON.duration, 106);
  assert.equal(NEON.audio, "audio.m4a");
  assert.equal(NEON.difficulty, 1);
  assert.deepEqual(NEON.coach, { outfit: "neon" });
  assert.deepEqual(NEON.preview, [48, 64]);
  assert.deepEqual(NEON.sections.map((s) => [s.name, s.beat]), SECTIONS);
  assert.equal(NEON.beatOf(NEON.duration), 212, "52 bars plus a 2 s tail");
  assert.equal(NEON.sectionAt(50), "chorus 1");
});

test("Neon Nights structure", () => {
  assert.equal(sectionMoves(NEON, 0, 16).length, 0, "the intro has no moves");
  assert.equal(NEON.firstMoveBeat, 16);
  const choruses = [48, 112, 160].map((lo) => sectionMoves(NEON, lo, lo + 32));
  const signature = choruses[0].map((p) => p.move.name);
  for (const c of choruses) {
    assert.deepEqual(c.map((p) => p.move.name), signature, "every chorus dances the same signature");
    assert.ok(c[c.length - 1].end <= c[0].beat + 32);
  }
  const half = signature.length / 2;
  assert.deepEqual(signature.slice(0, half), signature.slice(half), "the phrase repeats within a chorus");
  assert.deepEqual(choruses.map((c) => c.filter((p) => p.gold).length), [1, 1, 2]);
  const inChorus = (p) => choruses.some((c) => c[0].beat <= p.beat && p.beat < c[0].beat + 32);
  assert.ok(!NEON.chart.some((p) => p.gold && !inChorus(p)), "gold only in choruses");
  const bridge = sectionMoves(NEON, 144, 160);
  assert.ok(bridge.length && bridge.every((p) => p.move.beats === 4), "the bridge slows down to 4-beat moves");
  const outro = sectionMoves(NEON, 192, 212);
  assert.ok(outro[0].beat === 192 && outro[0].move.judge.length, "a judged final pose on beat 192");
  const last = outro[outro.length - 1];
  assert.ok(last.move.name === "bow" && !last.move.judge.length, "then an unjudged bow");
  const jp = NEON.judgePoints();
  assert.equal(jp[jp.length - 1].beat, 192);
  assert.ok(NEON.lastJudgeTime < NEON.timeOf(208), "judging ends before the outro does");
});

test("verse 1 is the easiest", () => {
  const density = (lo, hi) => NEON.judgePoints().filter((p) => lo <= p.beat && p.beat < hi).length / (hi - lo);
  assert.ok(density(16, 48) <= 0.65);
  assert.ok(density(16, 48) < density(80, 112) && density(80, 112) <= density(48, 80));
  assert.ok(density(144, 160) <= 0.5);
});

test("sides alternate", () => {
  const names = NEON.chart.map((p) => p.move.name);
  names.forEach((n, i) => {
    const twin = `${n.slice(0, -2)}_l`;
    if (n.endsWith("_r") && names.includes(twin)) {
      assert.ok(names.slice(i + 1, i + 3).includes(twin) || names.slice(Math.max(0, i - 2), i).includes(twin), n);
    }
  });
});

// --- difficulties (docs/DANCE.md section 7a) ----------------------------------

test("difficulty ids fall back to normal", () => {
  assert.deepEqual(DIFFICULTIES, ["easy", "normal", "hard"]);
  assert.deepEqual(["easy", "HARD", "normal", null, "expert", ""].map(difficultyId),
    ["easy", "hard", "normal", "normal", "normal", "normal"]);
  assert.ok(NEON.level === "normal" && NEON.forDifficulty("normal") === NEON);
  assert.equal(parseSong(readJSON("songs/neon_nights/song.json"), "neon_nights", LIB, "nightmare").level, "normal");
});

test("hard dances the normal chart", () => {
  const hard = NEON.forDifficulty("hard");
  assert.equal(hard.level, "hard");
  const key = (p) => [p.beat, p.label, p.gold];
  assert.deepEqual(hard.judgePoints().map(key), NEON.judgePoints().map(key));
  assert.deepEqual(parseSong(readJSON("songs/neon_nights/song.json"), "neon_nights", LIB, "hard").judgePoints().map(key),
    hard.judgePoints().map(key));
});

test("easy judges about one pose per bar", () => {
  for (const song of SONGS) {
    const easy = song.forDifficulty("easy"), normal = song.judgePoints();
    const pts = easy.judgePoints();
    assert.ok(pts.length && pts.length <= 0.4 * normal.length, `${pts.length} vs ${normal.length}`);
    const beats = pts.map((p) => p.beat);
    for (let i = 1; i < beats.length; i++) assert.ok(beats[i] - beats[i - 1] >= EASY_MIN_GAP);
    const short = pts.filter((p) => easy.chart[p.placed].move.name.endsWith("~held")).map((p) => p.beat).sort((a, b) => a - b);
    const bars = short.map((b) => Math.floor(b / BAR_BEATS));
    assert.equal(bars.length, new Set(bars).size, "one held pose per bar");
    if (!Object.hasOwn(song.charts, "easy")) {
      assert.ok(pts.every((p) => normal.some((q) => Math.abs(p.beat - q.beat) < 1e-9 && q.pose === p.pose)),
        "easy only asks for poses the normal chart asks for, on the same beats");
    }
    assert.ok(easy.lastJudgeTime >= 25.0);
  }
});

test("the easy coach holds the kept pose", () => {
  for (const song of SONGS) {
    const easy = song.forDifficulty("easy");
    for (const p of easy.chart) {
      if (p.move.name.endsWith("~held")) assert.ok(p.move.keyframes.length === 1 && p.move.judge.join() === "0");
    }
    const pts = easy.judgePoints();
    for (let i = 1; i < pts.length; i++) {
      const [a, b] = [pts[i - 1], pts[i]];
      if (!easy.chart[a.placed].move.name.endsWith("~held")) continue;
      for (const u of [0.0, 0.5, 1.0]) {
        const t = a.beat + u * Math.max(0.0, b.beat - a.beat - EASE_BEATS);
        assert.ok(sim(a.pose, easy.coachPose(t)) > 0.97, `coach left ${a.label} at beat ${t}`);
      }
    }
  }
});

test("Neon Nights on easy", () => {
  const easy = NEON.forDifficulty("easy");
  const pts = easy.judgePoints();
  assert.equal(NEON.judgePoints().length, 145);
  assert.equal(pts.length, 49);
  SECTIONS.forEach(([name, lo], i) => {
    const hi = SECTIONS[i + 1]?.[1] ?? 212;
    if (NEON.judgePoints().some((p) => p.gold && lo <= p.beat && p.beat < hi)) {
      assert.ok(pts.some((p) => p.gold && lo <= p.beat && p.beat < hi), `${name} lost its gold`);
    }
  });
  assert.deepEqual(sectionMoves(easy, 144, 160).map((p) => p.move), sectionMoves(NEON, 144, 160).map((p) => p.move),
    "the bridge's 4-beat moves stay whole");
  assert.deepEqual(sectionMoves(easy, 192, 212).map((p) => [p.beat, p.move.name]), [[192, "finale"], [196, "bow"]]);
  assert.equal(pts[pts.length - 1].beat, 192);
  assert.equal(pts.filter((p) => p.beat >= 16 && p.beat < 48).length, 8, "verse 1: one pose a bar");
});

const handMade = (extra) => ({ title: "T", bpm: 120, duration: 40.0,
  moves: [{ move: "shine", beat: 16 }, { move: "rise", beat: 20 }, { move: "star", beat: 24 }], ...extra });

test("a song can hand-make its easy and hard charts", () => {
  const data = handMade({ charts: { easy: [{ move: "rise", beat: 20, gold: true }],
    hard: [{ move: "shine", beat: 16 }, { move: "star", beat: 18 }, { move: "rise", beat: 20 }] } });
  const easy = parseSong(data, "s", LIB, "easy");
  assert.deepEqual(easy.chart.map((p) => [p.move.name, p.beat, p.gold]), [["rise", 20, true]]);
  assert.deepEqual(parseSong(data, "s", LIB, "hard").chart.map((p) => p.move.name), ["shine", "star", "rise"]);
  const normal = parseSong(data, "s", LIB);
  assert.deepEqual(normal.chart.map((p) => p.move.name), ["shine", "rise", "star"]);
  assert.deepEqual(normal.forDifficulty("easy").chart, easy.chart, "the hand-made chart wins over the derived one");
});

test("hand-made charts are validated like the normal one", () => {
  for (const [charts, message] of [
    [{ easy: [{ move: "nope", beat: 16 }] }, "charts.easy: chart entry 0 at beat 16.0: unknown move 'nope'"],
    [{ hard: [{ move: "rise", beat: 16 }, { move: "shine", beat: 18 }] }, "charts.hard: move 'shine' at beat 18"],
    [{ hard: [{ move: "rise", beat: 78 }] }, "runs past the song's end"],
    [{ normal: [] }, "only 'easy' and 'hard' charts can be hand-made"],
    [{ expert: [] }, "only 'easy' and 'hard'"],
    [{ easy: { move: "rise" } }, "charts.easy: expected a list"],
    [[], "'charts' must be an object"],
  ]) {
    assert.throws(() => parseSong(handMade({ charts }), "s", LIB),
      (err) => err instanceof ChartError && err.message.includes(message), message);
  }
});

test("easy moves gold to the kept pose when the gold one would freeze", () => {
  const up = { la: [170, 175], ra: [170, 175] }, out = { la: [90, 90], ra: [90, 90] };
  const moves = parseMoves({ up: { beats: 2, keyframes: [{ beat: 0, pose: up }] },
    out: { beats: 2, keyframes: [{ beat: 0, pose: out }] } }, "t");
  const normal = [new Placed(moves.get("up"), 16), new Placed(moves.get("out"), 18),
    new Placed(moves.get("up"), 20, true), new Placed(moves.get("out"), 22)];
  const easy = deriveEasy(normal);
  assert.deepEqual(easy.map((p) => [p.beat, p.move.label, p.gold]), [[16, "Up", false], [22, "Out", true]]);
  assert.ok(easy[0].end === 20 && easy[1].end === 24, "held to the end of the bar's moves");
});

test("the freeze rule holds on every difficulty", () => {
  for (const song of SONGS) {
    for (const level of DIFFICULTIES) {
      const jp = song.forDifficulty(level).judgePoints();
      for (let i = 1; i < jp.length; i++) {
        assert.ok(sim(jp[i].pose, jp[i - 1].pose) < FREEZE_MAX, `${song.id} ${level} beat ${jp[i].beat}`);
        assert.ok(sim(jp[i].pose, jp[i - 1].pose, { visible: ELBOWS_HIDDEN }) < FREEZE_MAX, `${song.id} ${level}`);
      }
    }
  }
});

// --- the move library -------------------------------------------------------

test("library size, twins and labels", () => {
  assert.ok(LIB.size >= 22 && LIB.size <= 64);
  for (const [name, spec] of Object.entries(readJSON("moves.json").moves)) {
    if (name.endsWith("_l")) assert.equal(spec.mirror_of, `${name.slice(0, -2)}_r`, `${name}: author one side, mirror the other`);
  }
  for (const m of LIB.values()) assert.ok(m.label.length > 0 && m.label.length <= 15, `${m.name}: labels are short`);
  assert.ok(LIB.has("bow") && LIB.get("bow").judge.length === 0);
  assert.ok([...LIB.values()].some((m) => m.beats === 4 && m.judge.length >= 2), "slow moves for the bridge");
});

test("judged keyframes are clear arm shapes", () => {
  const rest = normalizePose(REST);
  for (const [name, move] of LIB) {
    move.keyframes.forEach((k, i) => {
      for (const s of ["l", "r"]) {
        const [upper, fore] = k.pose[`${s}a`];
        assert.ok(upper >= -110 && upper <= 260 && fore >= -110 && fore <= 260, `${name}[${i}] ${s}a out of range`);
        assert.ok(Math.abs(fore - upper) <= 160, `${name}[${i}] ${s}a folds the elbow too far`);
      }
      if (move.judge.includes(i)) assert.ok(sim(k.pose, rest) < FREEZE_MAX, `${name}[${i}] is too close to standing there`);
    });
    const judged = move.judge.map((i) => move.keyframes[i].pose);
    for (let i = 1; i < judged.length; i++) {
      assert.ok(sim(judged[i], judged[i - 1]) < FREEZE_MAX && sim(judged[i - 1], judged[i]) < FREEZE_MAX,
        `${name}: its judged keyframes look alike`);
    }
  }
});

// --- the preload path (what the browser runs) ---------------------------------

const diskFetch = async (href) => JSON.parse(readFileSync(new URL(href), "utf8"));

test("loadSongs fetches the library, the index and every song", async () => {
  const out = await loadSongs({ base: ASSETS, fetchJSON: diskFetch });
  assert.deepEqual(out.ids, IDS);
  assert.equal(out.library.size, LIB.size);
  assert.equal(out.songs[0].judgePoints().length, NEON.judgePoints().length);
  assert.deepEqual(songVariants(out.songs)[0], {
    id: "neon_nights", title: "Neon Nights", subtitle: "KP Arcade - 1:46", boardId: "dance-neon_nights",
  });
});

test("a broken song is skipped, not fatal; a missing library means no shared moves", async () => {
  const files = {
    "songs/index.json": ["good", "broken", "missing"],
    "songs/good/song.json": { title: "Good", bpm: 120, duration: 30, moves: [],
      custom_moves: { a: { beats: 1, keyframes: [{ beat: 0 }] } } },
    "songs/broken/song.json": { title: "Broken", bpm: 120, duration: 30, moves: [{ move: "nope", beat: 0 }] },
  };
  const base = new URL("https://example.test/assets/dance/");
  const fetchJSON = async (href) => {
    const key = href.slice(base.href.length);
    if (!(key in files)) throw new Error(`could not load ${href} (404)`);
    return structuredClone(files[key]);
  };
  const warn = console.warn;
  const warnings = [];
  console.warn = (msg) => warnings.push(msg);
  try {
    const out = await loadSongs({ base, fetchJSON });
    assert.deepEqual(out.songs.map((s) => s.title), ["Good"]);
    assert.equal(out.library.size, 0);
  } finally {
    console.warn = warn;
  }
  assert.ok(warnings.some((w) => /skipping song 'broken'.*unknown move 'nope'/.test(w)), warnings.join("\n"));
  assert.ok(warnings.some((w) => /skipping song 'missing'.*missing song.json/.test(w)), warnings.join("\n"));
});

test("variants() is empty before preload and lists the songs after", () => {
  if (loadedSongs() == null) assert.deepEqual(DanceRaccoonGame.variants(), []);
  assert.throws(() => new DanceRaccoonGame({}), /preload/);
});
