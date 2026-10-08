// Leaderboards: storage, ranking rules, and the shell's tutorial / name-entry / results flow.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  LocalBoard, SupabaseBoard, createBoard, qualifies, cleanName, nameAllowed, sortEntries, BOARD_SIZE,
} from "../js/leaderboard.js";
import { Shell, ENTRY_DELAY, RESULTS_DELAY, KEY_DWELL } from "../js/shell.js";
import { Game } from "../js/games/base.js";
import { KP, Pose } from "../js/core/pose.js";

function memoryStorage() {
  const m = new Map();
  return { getItem: (k) => (m.has(k) ? m.get(k) : null), setItem: (k, v) => m.set(k, String(v)), removeItem: (k) => m.delete(k) };
}

test("names: three characters, A-Z and 0-9, upper case", () => {
  assert.equal(cleanName("ab"), "AB");
  assert.equal(cleanName("a.b c-d"), "ABC");
  assert.equal(cleanName("x9!"), "X9");
  assert.ok(nameAllowed("ACE"));
  assert.ok(!nameAllowed(""));
  assert.ok(!nameAllowed("abcd"), "only already-clean names pass");
  assert.ok(!nameAllowed("ASS"), "obvious abuse is refused");
});

test("ranking: higher score first, ties go to whoever got there first", () => {
  const s = sortEntries([{ name: "B", score: 5, at: 2 }, { name: "A", score: 5, at: 1 }, { name: "C", score: 9, at: 3 }]);
  assert.deepEqual(s.map((e) => e.name), ["C", "A", "B"]);
});

test("qualifies: any positive score until the board is full, then beat the last", () => {
  assert.ok(!qualifies([], 0), "zero never makes the board");
  assert.ok(qualifies([], 1));
  const full = Array.from({ length: BOARD_SIZE }, (_, i) => ({ name: "X", score: 10 + i, at: i }));
  assert.ok(!qualifies(full, 10), "a tie with the last place does not displace it");
  assert.ok(qualifies(full, 11));
});

test("local board keeps the top ten per game, separately for each game", async () => {
  const b = new LocalBoard(memoryStorage());
  for (let i = 1; i <= 12; i++) await b.submit("snack", "AA", i);
  await b.submit("flappy", "ZZ", 3);
  const snack = await b.top("snack");
  assert.equal(snack.length, BOARD_SIZE);
  assert.equal(snack[0].score, 12);
  assert.equal(snack.at(-1).score, 3);
  assert.deepEqual((await b.top("flappy")).map((e) => e.name), ["ZZ"]);
});

test("local board survives a reload (same storage)", async () => {
  const store = memoryStorage();
  await new LocalBoard(store).submit("snack", "JOE", 42);
  assert.equal((await new LocalBoard(store).top("snack"))[0].name, "JOE");
});

test("local board works without storage (private mode) for the visit", async () => {
  const b = new LocalBoard(null);
  const { id } = await b.submit("snack", "ME", 7);
  assert.equal((await b.top("snack"))[0].id, id);
});

test("supabase board: reads the top ten, posts a cleaned name", async () => {
  const calls = [];
  const fetchImpl = async (url, opts = {}) => {
    calls.push({ url, opts });
    if (opts.method === "POST") return { ok: true, json: async () => [{ id: 7 }] };
    return { ok: true, json: async () => [{ id: 7, name: "ACE", score: 9, created_at: "2026-10-08T00:00:00Z" }] };
  };
  const b = new SupabaseBoard({ url: "https://x.supabase.co/", key: "pk", fetchImpl });
  const out = await b.submit("flappy", "ace!", 9.4);
  assert.equal(out.id, "7");
  const post = calls.find((c) => c.opts.method === "POST");
  assert.deepEqual(JSON.parse(post.opts.body), { game: "flappy", name: "ACE", score: 9 });
  assert.equal(post.opts.headers.apikey, "pk");
  const get = calls.find((c) => !c.opts.method);
  assert.match(get.url, /^https:\/\/x\.supabase\.co\/rest\/v1\/scores\?/);
  assert.match(get.url, /game=eq\.flappy/);
  assert.match(get.url, /order=score\.desc,created_at\.asc/);
});

test("no configuration: each browser keeps its own board", () => {
  assert.equal(createBoard({}).scope, "device");
});

// --- the shell flow ---

class Scored extends Game {
  static id = "scored";
  static title = "SCORED";
  static tutorial = [{ title: "ONE", text: "Do the thing.", draw() {} }];
  constructor(o) {
    super(o);
    this.score = 0;
  }
}

const none = { present: false, pose: null };
const step = (sh, seconds, controls = none) => {
  for (let i = 0; i < Math.round(seconds * 60); i++) sh.update(controls, 1 / 60);
};

async function shellWith(entries = []) {
  const board = new LocalBoard(memoryStorage());
  for (const [name, score] of entries) await board.submit("scored", name, score);
  const sh = new Shell({ games: [Scored], board });
  await sh.refreshScores();
  return sh;
}

function playTo(sh, score) {
  sh.handleKey("1");
  if (sh.screen === "tutorial") sh.handleKey("Enter");
  step(sh, 3.1);
  assert.equal(sh.screen, "playing");
  sh.game.score = score;
  sh.game.phase = "game_over";
}

test("a first round shows the how-to-play cards; a replay does not", async () => {
  const sh = await shellWith();
  sh.handleKey("1");
  assert.equal(sh.screen, "tutorial");
  step(sh, 5);
  assert.equal(sh.screen, "tutorial", "the cards wait for the player");
  sh.handleKey("Enter");
  assert.equal(sh.screen, "countdown");
  step(sh, 3.1);
  sh.handleKey("r");
  assert.equal(sh.screen, "countdown", "restart skips the cards");
  sh.handleKey("Escape");
  sh.handleKey("1");
  assert.equal(sh.screen, "countdown", "already seen this visit");
});

test("a score that makes the board asks for a name, then shows the board with it", async () => {
  const sh = await shellWith([["OLD", 3]]);
  playTo(sh, 8);
  step(sh, ENTRY_DELAY + 0.1);
  assert.equal(sh.screen, "entry");
  for (const k of ["h", "r", "x", "q"]) sh.handleKey(k); // H and R type, they don't swap hands or restart
  assert.equal(sh.entry.name, "HRX", "three characters at most");
  assert.equal(sh.hand, "right");
  sh.handleKey("Backspace");
  sh.handleKey("Enter");
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(sh.screen, "results");
  const rows = sh.scores.scored;
  assert.deepEqual(rows.map((e) => [e.name, e.score]), [["HR", 8], ["OLD", 3]]);
  assert.equal(rows[0].id, sh.lastEntryId, "the new row is highlighted");
  sh.handleKey("r");
  assert.equal(sh.screen, "countdown");
});

test("a score that misses the board goes straight to the leaderboard", async () => {
  const full = Array.from({ length: BOARD_SIZE }, (_, i) => [`P${i}`, 50 + i]);
  const sh = await shellWith(full);
  playTo(sh, 12);
  step(sh, RESULTS_DELAY + 0.1);
  assert.equal(sh.screen, "results");
  assert.equal(sh.lastEntryId, null);
});

test("an empty name is refused and so is abuse; SKIP saves nothing", async () => {
  const sh = await shellWith();
  playTo(sh, 5);
  step(sh, ENTRY_DELAY + 0.1);
  sh.handleKey("Enter");
  assert.equal(sh.screen, "entry");
  assert.ok(sh.entry.message);
  for (const k of "ass") sh.handleKey(k);
  sh.handleKey("Enter");
  assert.equal(sh.screen, "entry");
  sh.handleKey("Escape");
  assert.equal(sh.screen, "results");
  assert.deepEqual(sh.scores.scored ?? [], []);
});

test("letters can be typed by hovering the palm over the on-screen keys", async () => {
  const sh = await shellWith();
  playTo(sh, 5);
  step(sh, ENTRY_DELAY + 0.1);
  const key = sh._layout().find((b) => b.action === "key:K");
  const [x0, y0, x1, y1] = key.rect;
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  const xy = new Float32Array(34), c = new Float32Array(17);
  const put = (n, x, y) => { xy[2 * KP[n]] = x; xy[2 * KP[n] + 1] = y; c[KP[n]] = 0.9; };
  put("left_shoulder", 700, 500);
  put("right_shoulder", 560, 500);
  put("left_elbow", cx, cy + 138); // forearm straight up: the palm lands on the key
  put("left_wrist", cx, cy + 38);
  const controls = { present: true, pose: new Pose(xy, c) };
  step(sh, 0.5, none); // hand away first (the screen change blocks until it leaves)
  step(sh, KEY_DWELL + 0.6, controls);
  assert.equal(sh.entry.name, "K");
});
