// Raccoon Dance parity: the browser judge replays the Python judge's recorded
// input (web/tests/fixtures/dance_parity.json, made by
// tools/make_dance_parity_fixture.py) on every difficulty and must reach the
// same verdicts: every rating identical, at the same frame, and the same score
// (within 1 point). deriveEasy must make exactly the Python's EASY charts.
// Regenerate the fixture after changing the Python judge, pose or chart code.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { Pose } from "../js/core/pose.js";
import { freezeScore, parseLibrary, parseMoves, parseSong } from "../js/games/dance/chart.js";
import { Judge } from "../js/games/dance/judge.js";
import { measure } from "../js/games/dance/pose.js";

const FIXTURE = JSON.parse(readFileSync(new URL("./fixtures/dance_parity.json", import.meta.url), "utf8"));
const ASSETS = new URL("../../assets/dance/", import.meta.url);
const readJSON = (p) => JSON.parse(readFileSync(new URL(p, ASSETS), "utf8"));

function songFor(name) {
  if (name === "groove") {
    const { library, song } = FIXTURE.groove;
    return parseSong(song, "parity", parseMoves(library, "parity library"));
  }
  return parseSong(readJSON(`songs/${name}/song.json`), name, parseLibrary(readJSON("moves.json")));
}

function toPose(p) {
  if (p == null) return null;
  const xy = Float32Array.from(p.xy, (v) => v / 10.0);
  const conf = new Float32Array(17);
  for (const i of p.v) conf[i] = 0.9;
  return new Pose(xy, conf, 0.9);
}

for (const stream of FIXTURE.streams) {
  for (const [level, want] of Object.entries(stream.levels)) {
    test(`parity with the Python judge: ${stream.name}, ${level}`, () => {
      const song = songFor(stream.name).forDifficulty(level);
      assert.deepEqual(song.judgePoints().map((p) => [p.index, p.time, p.weight]), want.points,
        "the same judge timeline");
      const j = new Judge(song, level);
      const measured = stream.poses.map((p) => measure(toPose(p)));
      const resumes = new Set(stream.resumes);
      const got = [];
      stream.frames.forEach(([st, idx], i) => {
        if (resumes.has(i)) j.onResume(st);
        for (const r of j.update(st, idx >= 0 ? measured[idx] : null)) got.push([i, r.point.index, r.name, r.similarity]);
      });
      for (const r of j.update(want.end_time, null)) got.push([stream.frames.length, r.point.index, r.name, r.similarity]);

      assert.equal(got.length, want.ratings.length);
      got.forEach((g, k) => {
        const w = want.ratings[k];
        assert.deepEqual(g.slice(0, 3), w.slice(0, 3), `rating ${k}`);
        if (w[3] == null) assert.equal(g[3], null, `rating ${k}`);
        else assert.ok(Math.abs(g[3] - w[3]) < 1e-9, `rating ${k}: similarity ${g[3]} vs ${w[3]}`);
      });
      assert.ok(Math.abs(j.score - want.score) <= 1, `score ${j.score} vs ${want.score}`);
      assert.deepEqual(j.counts, want.counts);
      assert.equal(j.unseen, want.unseen);
      assert.equal(j.bestCombo, want.best_combo);
      if (level === "normal") {
        const kinds = new Set(got.map((g) => g[2]));
        assert.ok(kinds.size >= 4, `the stream exercises several ratings: ${[...kinds]}`);
      }
    });
  }
}

for (const [name, want] of Object.entries(FIXTURE.easy)) {
  test(`deriveEasy makes the Python's EASY chart: ${name}`, () => {
    const song = songFor(name);
    const normal = song.judgePoints();
    // freezeScore decides which poses EASY keeps: it must match the Python's closely.
    for (const [a, b, score] of want.freeze) {
      const got = freezeScore(normal[b].pose, normal[a].pose);
      assert.ok(Math.abs(got - score) < 1e-9, `freeze ${a}->${b}: ${got} vs ${score}`);
    }
    const easy = song.forDifficulty("easy");
    assert.deepEqual(easy.chart.map((p) => [p.beat, p.move.name, p.move.beats, p.gold, [...p.move.judge]]), want.chart);
    assert.deepEqual(easy.judgePoints().map((p) => p.beat), want.judge_beats);
    if (name === "neon_nights") assert.equal(want.judge_beats.length, 49);
  });
}
