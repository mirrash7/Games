// The server's limits must never refuse a score the real games can produce.
import { test } from "node:test";
import assert from "node:assert/strict";
import { GAMES, nameError, scoreError, MIN_ROUND } from "../src/rules.js";
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
