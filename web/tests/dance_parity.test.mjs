// Raccoon Dance parity: the browser judge replays the Python judge's recorded
// input (web/tests/fixtures/dance_parity.json, made by
// tools/make_dance_parity_fixture.py) and must reach the same verdicts: every
// rating identical, at the same frame, and the same score (within 1 point).
// Regenerate the fixture after changing the Python judge, pose or chart code.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { Pose } from "../js/core/pose.js";
import { parseLibrary, parseMoves, parseSong } from "../js/games/dance/chart.js";
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
  test(`parity with the Python judge: ${stream.name}`, () => {
    const song = songFor(stream.name);
    assert.deepEqual(song.judgePoints().map((p) => [p.index, p.time, p.weight]), stream.points,
      "the same judge timeline");
    const j = new Judge(song);
    const measured = stream.poses.map((p) => measure(toPose(p)));
    const resumes = new Set(stream.resumes);
    const got = [];
    stream.frames.forEach(([st, idx], i) => {
      if (resumes.has(i)) j.onResume(st);
      for (const r of j.update(st, idx >= 0 ? measured[idx] : null)) got.push([i, r.point.index, r.name, r.similarity]);
    });
    for (const r of j.update(stream.end_time, null)) got.push([stream.frames.length, r.point.index, r.name, r.similarity]);

    assert.equal(got.length, stream.ratings.length);
    got.forEach((g, k) => {
      const want = stream.ratings[k];
      assert.deepEqual(g.slice(0, 3), want.slice(0, 3), `rating ${k}`);
      if (want[3] == null) assert.equal(g[3], null, `rating ${k}`);
      else assert.ok(Math.abs(g[3] - want[3]) < 1e-9, `rating ${k}: similarity ${g[3]} vs ${want[3]}`);
    });
    assert.ok(Math.abs(j.score - stream.score) <= 1, `score ${j.score} vs ${stream.score}`);
    assert.deepEqual(j.counts, stream.counts);
    assert.equal(j.unseen, stream.unseen);
    assert.equal(j.bestCombo, stream.best_combo);
    const kinds = new Set(got.map((g) => g[2]));
    assert.ok(kinds.size >= 4, `the stream exercises several ratings: ${[...kinds]}`);
  });
}
