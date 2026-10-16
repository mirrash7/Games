// Dance charts: the move library and every shipped song, read from disk and
// validated by the browser's parser (port of the chart half of
// tests/test_dance_chart.py; the importer and CLI are desktop-only).

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, statSync } from "node:fs";
import {
  BAR_BEATS, ChartError, DIFFICULTIES, EASE_BEATS, EASY_MIN_GAP, Placed, deriveEasy, difficultyId, legLed, lint,
  parseLibrary, parseMoves, parseSong, parseSongIds,
} from "../js/games/dance/chart.js";
import { keyframeTravel } from "../js/games/dance/motion.js";
import { FULL_BODY, REST, forward, keypointsFor, measure, mirrorPose, normalizePose, similarity } from "../js/games/dance/pose.js";
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
// Legs the framing hides are left out (null) unless `unseenLegs` is given.
const sim = (target, performed, opts = {}, unseenLegs = null) =>
  similarity(target, measure(keypointsFor(performed, opts)), unseenLegs)[0];

// --- every song --------------------------------------------------------------

test("every listed song loads and validates", () => {
  assert.ok(IDS.length && IDS[0] === "spell_it_out", "the letters song is the first on the song list (and preselected)");
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
  // earns less than 0.55 on the next: seen whole (feet too), with the feet out
  // of frame, and without elbows.
  for (const song of SONGS) {
    const jp = song.judgePoints();
    for (let i = 1; i < jp.length; i++) {
      const [a, b] = [jp[i - 1], jp[i]];
      const body = sim(b.pose, a.pose, FULL_BODY);
      assert.ok(body < FREEZE_MAX, `${song.id}: '${a.label}' at beat ${a.beat} scores ${body.toFixed(2)} on beat ${b.beat}`);
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
      assert.ok(sim(p.pose, p.pose, FULL_BODY) > 0.99, `${p.label} at ${p.beat}, whole body in view`);
    }
  }
});

// --- the choreography lint (docs/DANCE.md section 5a) -------------------------

test("every song passes the choreography lint on every difficulty", () => {
  for (const song of SONGS) {
    for (const level of DIFFICULTIES) {
      const s = song.forDifficulty(level);
      const issues = lint(s, keyframeTravel, { phrases: level !== "easy" });
      assert.deepEqual(issues, [], `${song.id} ${level}: ${issues.map((i) => `beat ${i.beat}: [${i.kind}] ${i.message}`).join("; ")}`);
    }
  }
});

test("the legs set the rhythm", () => {
  // Most judged poses work the legs, and perfect arms with the legs standing
  // still fall short of PERFECT on every one of them.
  for (const song of SONGS) {
    const pts = song.judgePoints();
    const led = pts.filter((p) => legLed(p.pose));
    assert.ok(led.length >= 0.8 * pts.length, `${song.id}: ${led.length} of ${pts.length}`);
    for (const p of led) {
      const still = { ...normalizePose(p.pose), lf: [...REST.lf], rf: [...REST.rf], shift: 0.0, squat: 0.0 };
      assert.ok(sim(p.pose, still, FULL_BODY) < 0.9, `${song.id} ${p.label} at ${p.beat}: arms alone are a PERFECT`);
    }
  }
});

test("the lint flags the old charts' problems and passes a repeated phrase", () => {
  const clap = { beats: 2, keyframes: [{ beat: 0, pose: { la: [-30, -60], ra: [-30, -60], squat: 0.2 } },
    { beat: 1, pose: { la: [175, 180], ra: [175, 180] } }] };
  const song = (moves, custom) => parseSong({ title: "Lint", bpm: 120, duration: 60.0, moves, custom_moves: custom,
    sections: [{ name: "intro", beat: 0 }, { name: "verse", beat: 16 }] }, "lint", LIB);
  const kinds = (s) => new Set(lint(s, keyframeTravel).map((i) => i.kind));
  let k = kinds(song([0, 1, 2, 3].map((i) => ({ move: "clap", beat: 16 + 2 * i })), { clap }));
  for (const want of ["flow", "whiplash", "legs"]) assert.ok(k.has(want), `${want}: ${[...k]}`);
  const tiny = { beats: 2, keyframes: [{ beat: 0, pose: { la: [170, 175], ra: [90, 90], rf: [0.45, 0] } }] };
  const other = { beats: 2, keyframes: [{ beat: 0, pose: { la: [90, 90], ra: [170, 175] } }] };
  k = kinds(song([{ move: "tiny", beat: 16 }, { move: "other", beat: 18 }, { move: "hip_sway_r", beat: 20 },
    { move: "tiny", beat: 22 }], { tiny, other }));
  for (const want of ["unjudgeable", "phrase"]) assert.ok(k.has(want), `${want}: ${[...k]}`);
  const names = ["step_touch_r", "step_touch_l", "step_touch_r", "step_touch_l", "hip_sway_r", "hip_sway_l",
    "hip_sway_r", "hip_sway_l"];
  assert.deepEqual(lint(song(names.map((move, i) => ({ move, beat: 16 + 2 * i })), {}), keyframeTravel), []);
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
  const twin = (n) => (n.endsWith("_r") ? `${n.slice(0, -2)}_l` : n.endsWith("_l") ? `${n.slice(0, -2)}_r` : n);
  assert.deepEqual(signature.slice(half), signature.slice(0, half).map(twin),
    "a chorus dances its 16 counts, then the same 16 mirrored");
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
  // A one-sided move gets its twin in the same section (right away, or in the
  // phrase's mirrored half), so both sides get the work.
  SECTIONS.forEach(([name, lo], i) => {
    const hi = SECTIONS[i + 1]?.[1] ?? 212;
    const names = sectionMoves(NEON, lo, hi).map((p) => p.move.name);
    for (const n of names) if (n.endsWith("_r")) assert.ok(names.includes(`${n.slice(0, -2)}_l`), `${name}: ${n}`);
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
    const beats = pts.map((p) => p.beat);
    for (let i = 1; i < beats.length; i++) assert.ok(beats[i] - beats[i - 1] >= EASY_MIN_GAP);
    const short = pts.filter((p) => easy.chart[p.placed].move.name.endsWith("~held")).map((p) => p.beat).sort((a, b) => a - b);
    const bars = short.map((b) => Math.floor(b / BAR_BEATS));
    assert.equal(bars.length, new Set(bars).size, "one held pose per bar");
    // Slow moves stay whole (a bridge's 4-beat moves, Spell It Out's letters), so
    // the short moves are where the points go; and at most half overall.
    const whole = pts.length - short.length;
    assert.ok(pts.length && short.length <= 0.42 * (normal.length - whole), `${short.length} vs ${normal.length - whole}`);
    assert.ok(pts.length <= 0.5 * normal.length, `${pts.length} vs ${normal.length}`);
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
  assert.equal(NEON.judgePoints().length, 123);
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
  assert.equal(pts.filter((p) => p.beat >= 16 && p.beat < 48).length, 9,
    "verse 1: one pose a bar, plus the pre-chorus Squat & Rise in full");
  const vines = easy.chart.filter((p) => p.move.name.startsWith("vine"));
  assert.ok(vines.length && vines.every((p) => p.move.name.endsWith("~held") && p.move.judge.length === 1),
    "a grapevine (4 poses in a bar) becomes one held pose");
});

const handMade = (extra) => ({ title: "T", bpm: 120, duration: 40.0,
  moves: [{ move: "hip_sway_r", beat: 16 }, { move: "squat_rise", beat: 20 }, { move: "star_jump", beat: 24 }], ...extra });

test("a song can hand-make its easy and hard charts", () => {
  const data = handMade({ charts: { easy: [{ move: "squat_rise", beat: 20, gold: true }],
    hard: [{ move: "hip_sway_r", beat: 16 }, { move: "star_jump", beat: 18 }, { move: "squat_rise", beat: 20 }] } });
  const easy = parseSong(data, "s", LIB, "easy");
  assert.deepEqual(easy.chart.map((p) => [p.move.name, p.beat, p.gold]), [["squat_rise", 20, true]]);
  assert.deepEqual(parseSong(data, "s", LIB, "hard").chart.map((p) => p.move.name), ["hip_sway_r", "star_jump", "squat_rise"]);
  const normal = parseSong(data, "s", LIB);
  assert.deepEqual(normal.chart.map((p) => p.move.name), ["hip_sway_r", "squat_rise", "star_jump"]);
  assert.deepEqual(normal.forDifficulty("easy").chart, easy.chart, "the hand-made chart wins over the derived one");
});

test("hand-made charts are validated like the normal one", () => {
  for (const [charts, message] of [
    [{ easy: [{ move: "nope", beat: 16 }] }, "charts.easy: chart entry 0 at beat 16.0: unknown move 'nope'"],
    [{ hard: [{ move: "squat_rise", beat: 16 }, { move: "hip_sway_r", beat: 18 }] }, "charts.hard: move 'hip_sway_r' at beat 18"],
    [{ hard: [{ move: "squat_rise", beat: 78 }] }, "runs past the song's end"],
    [{ normal: [] }, "only 'easy' and 'hard' charts can be hand-made"],
    [{ expert: [] }, "only 'easy' and 'hard'"],
    [{ easy: { move: "squat_rise" } }, "charts.easy: expected a list"],
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
  assert.equal(out.songs[0].judgePoints().length, SONGS[0].judgePoints().length);
  assert.deepEqual(songVariants(out.songs)[0], {
    id: "spell_it_out", title: "Spell It Out", subtitle: "KP Arcade - 1:40", boardId: "dance-spell_it_out",
  });
  assert.deepEqual(songVariants(out.songs)[1], {
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

// --- Spell It Out: the letters everyone knows (tests/test_dance_chart.py) -------------

const SPELL = SONGS.find((s) => s.id === "spell_it_out");
const LETTERS_MOVE = LIB.get("letters");
const LETTERS = Object.fromEntries(LETTERS_MOVE.judge.map((i) => [LETTERS_MOVE.keyframes[i].label,
  LETTERS_MOVE.keyframes[i].pose]));

test("the letters are labelled one by one and spelled in every chorus, on every difficulty", () => {
  assert.equal(LETTERS_MOVE.label, "Letters");
  assert.deepEqual(LETTERS_MOVE.judge, [0, 2, 4, 6]);
  assert.deepEqual(LETTERS_MOVE.judge.map((i) => LETTERS_MOVE.keyframes[i].label), ["Y", "M", "C", "A"]);
  assert.ok([1, 3, 5, 7].every((i) => LETTERS_MOVE.keyframes[i].label == null));
  for (const level of DIFFICULTIES) {
    const lv = SPELL.forDifficulty(level);
    const sets = lv.chart.filter((p) => p.move.name === "letters").map((p) => p.beat);
    assert.deepEqual(sets, [48, 64, 112, 128, 160, 176, 192], level);
    const spelled = lv.judgePoints().filter((p) => p.label in LETTERS).map((p) => [p.beat, p.label]);
    assert.deepEqual(spelled, sets.flatMap((b) => ["Y", "M", "C", "A"].map((ch, k) => [b + 2 * k, ch])), level);
  }
  // Mirroring and EASY's held moves keep a keyframe's label; a bad one is refused.
  const moves = parseMoves({ a: { beats: 2, keyframes: [{ beat: 0, label: "Q", pose: { la: [90, 90] } }] },
    b: { mirror_of: "a" } }, "t");
  assert.equal(moves.get("b").keyframes[0].label, "Q");
  const held = deriveEasy([new Placed(moves.get("a"), 16), new Placed(moves.get("a"), 17)]);
  assert.deepEqual(held.map((p) => [p.beat, p.move.keyframes[0].label]), [[16, "Q"]]);
  for (const bad of ["", 7, "much too long a label"]) {
    assert.throws(() => parseMoves({ a: { beats: 1, keyframes: [{ beat: 0, label: bad }] } }, "t"),
      /'label' must be a short name/);
  }
});

test("consecutive letters are told apart (as tests/test_dance_chart.py)", () => {
  // The letters follow a player's reference poses: M and A are nearly the same
  // arms and Y and C share an upper arm, so the chart never puts them back to back.
  for (const [h, t] of [["Y", "M"], ["M", "C"], ["C", "A"]]) {
    assert.ok(sim(LETTERS[t], LETTERS[h], FULL_BODY) < FREEZE_MAX, `${h} held scores on ${t}`);
    assert.ok(sim(LETTERS[t], LETTERS[h]) < FREEZE_MAX, `${h} held scores on ${t}, feet out of frame`);
    assert.ok(sim(LETTERS[t], LETTERS[h], { visible: ELBOWS_HIDDEN }) < FREEZE_MAX, `${h} held scores on ${t} without elbows`);
  }
  assert.ok(sim(LETTERS.A, LETTERS.M, FULL_BODY) > FREEZE_MAX, "M and A alike: pinned so a change shows up");
  for (const level of DIFFICULTIES) {
    const labels = SPELL.forDifficulty(level).judgePoints().map((p) => p.label);
    for (let i = 1; i < labels.length; i++) {
      const pair = [labels[i - 1], labels[i]].sort().join("");
      assert.ok(pair !== "AM" && pair !== "CY", `${level}: ${pair}`);
    }
  }
});

test("the C reads the right way round on screen", () => {
  const c = normalizePose(LETTERS.C);
  const j = forward(c);
  // As the reference player struck it: a lean, both arms reaching right, the body the back of the C.
  assert.ok(c.lean > 30, "the body leans: head to the screen's right");
  assert.ok(j.l_hand[1] < j.head[1] && j.l_hand[0] > j.l_elbow[0], "top stroke: over the head, pointing right");
  assert.ok(j.r_elbow[0] > j.r_shoulder[0] && j.r_elbow[1] > j.r_shoulder[1], "bottom stroke: down and out");
  assert.ok(j.r_hand[0] > j.r_elbow[0], "bottom stroke: its hand points right");
  assert.ok(Math.min(j.l_hand[0], j.r_hand[0]) > j.hip[0] + 0.4, "the back (the body) on the left, the tips right");
  assert.ok(sim(c, c, FULL_BODY) > 0.99);
  assert.ok(sim(c, mirrorPose(c)) < 0.3, "a reversed C is a miss (OK starts at 0.40, even on EASY)");
});
