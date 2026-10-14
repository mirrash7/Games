// The server's limits must never refuse a score the real games can produce.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { DANCE, GAMES, gameRules, nameError, scoreError, MIN_ROUND } from "../src/rules.js";
import { DEFAULT_RULES } from "../../web/js/games/flappy/game.js";
import { COMBO_MIN } from "../../web/js/games/snack/game.js";

test("flappy: the cap allows a pipe every intervalMin seconds", () => {
  assert.ok(GAMES.flappy.perSecond >= 1 / DEFAULT_RULES.intervalMin);
});

test("snack: the cap allows full waves of cookies, all eaten in one combo", () => {
  const perWave = 5 * 2 + 5 * 2; // five 2-point snacks plus a 2-per-snack combo bonus
  assert.ok(COMBO_MIN <= 5);
  assert.ok(GAMES.snack.perSecond >= perWave / 0.8);
});

test("names", () => {
  assert.equal(nameError("ACE"), null);
  assert.equal(nameError("A1"), null);
  for (const bad of ["", "ace", "ABCD", "A B", 7, null, "ASS"]) assert.ok(nameError(bad), String(bad));
});

test("scores", () => {
  assert.equal(scoreError("flappy", 10, 20), null);
  assert.ok(scoreError("flappy", 40, 20), "40 pipes in 20 s is impossible");
  assert.ok(scoreError("snack", 1, MIN_ROUND - 1), "before the countdown could end");
  assert.ok(scoreError("snack", 1.5, 60), "integers only");
  assert.ok(scoreError("snack", 0, 60));
  assert.ok(scoreError("pong", 1, 60));
});

test("dance: one rule set for every dance-<song> board, fixed games unchanged", () => {
  assert.equal(gameRules("dance-neon_nights"), DANCE);
  assert.equal(gameRules("dance-a"), DANCE);
  assert.equal(gameRules("snack"), GAMES.snack);
  assert.equal(gameRules("flappy"), GAMES.flappy);
  for (const bad of ["dance", "dance-", "dance-Neon", "dance-a b", "dance-../x", `dance-${"a".repeat(41)}`, "xdance-a", null, 7]) {
    assert.equal(gameRules(bad), null, String(bad));
    assert.equal(scoreError(bad, 100, 60), "unknown game");
  }
  assert.equal(DANCE.max, 10000);
  assert.equal(scoreError("dance-neon_nights", 9450, 104), null);
  assert.ok(scoreError("dance-neon_nights", 10001, 104), "above the maximum");
  assert.ok(scoreError("dance-neon_nights", 9000, 20), "9000 in 20 s is faster than any song allows");
});

test("dance: a perfect run on every shipped song is accepted when it ends", () => {
  const songs = new URL("../../assets/dance/songs/", import.meta.url);
  if (!existsSync(new URL("index.json", songs))) return;
  const ids = JSON.parse(readFileSync(new URL("index.json", songs)));
  assert.ok(ids.length > 0);
  const library = JSON.parse(readFileSync(new URL("../moves.json", songs))).moves;
  for (const id of ids) {
    const song = JSON.parse(readFileSync(new URL(`${id}/song.json`, songs)));
    const moves = { ...library, ...(song.custom_moves ?? {}) };
    const spec = (name) => (moves[name].mirror_of ? spec(moves[name].mirror_of) : moves[name]);
    // The round lasts at least until the last judged keyframe: 10000 needs it danced.
    let last = 0;
    for (const placed of song.moves) {
      const m = spec(placed.move);
      const judged = m.judge ?? m.keyframes.map((_, i) => i);
      for (const i of judged) last = Math.max(last, placed.beat + m.keyframes[i].beat);
    }
    const seconds = (last * 60) / song.bpm;
    assert.equal(scoreError(`dance-${id}`, 10000, seconds), null, `${id}: last judged move at ${seconds.toFixed(1)} s`);
  }
});
