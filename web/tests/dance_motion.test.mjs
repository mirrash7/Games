// Raccoon Dance coach motion parity: motion.js must reproduce the Python's
// coach motion (src/kpapp/game/dance/motion.py) at every sampled beat of every
// song, and the painter's 3D layout (twist, sway, squash, foreshortening, draw
// order) at every fifth (web/tests/fixtures/dance_motion.json, made by
// tools/make_dance_motion_fixture.py). Regenerate the fixture after changing
// the Python motion or the painter's layout.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { parseLibrary, parseSong } from "../js/games/dance/chart.js";
import { coachMotion } from "../js/games/dance/motion.js";
import { CoachPainter } from "../js/games/dance/render.js";
import { stubArt } from "../js/games/dance/art.js";
import { angleDiff, forward, normalizePose } from "../js/games/dance/pose.js";

const FIXTURE = JSON.parse(readFileSync(new URL("./fixtures/dance_motion.json", import.meta.url), "utf8"));
const ASSETS = new URL("../../assets/dance/", import.meta.url);
const readJSON = (p) => JSON.parse(readFileSync(new URL(p, ASSETS), "utf8"));
const LIB = parseLibrary(readJSON("moves.json"));
const KEYS = { head_yaw: "headYaw", head_tilt: "headTilt", head_bob: "headBob" };
const TOL = 2e-6;

function close(got, want, where) {
  if (Array.isArray(want)) {
    assert.equal(got.length, want.length, where);
    want.forEach((w, i) => close(got[i], w, `${where}[${i}]`));
  } else if (want !== null && typeof want === "object") {
    for (const [k, w] of Object.entries(want)) close(got[KEYS[k] ?? k], w, `${where}.${k}`);
  } else if (typeof want === "number") {
    assert.ok(Math.abs(got - want) <= TOL * Math.max(1, Math.abs(want)), `${where}: ${got} vs ${want}`);
  } else {
    assert.equal(got, want, where);
  }
}

const fakeCanvas = (w, h) => ({ width: w, height: h, getContext: () => new Proxy({}, { get: () => () => ({ addColorStop() {} }) }) });

for (const { id, samples } of FIXTURE.songs) {
  test(`coach motion matches the Python: ${id}`, () => {
    const song = parseSong(readJSON(`songs/${id}/song.json`), id, LIB);
    const painter = new CoachPainter(stubArt(), "neon", { makeCanvas: fakeCanvas });
    for (const s of samples) {
      const m = coachMotion(song, s.beat);
      close(m, s.motion, `${id} beat ${s.beat}`);
      if (s.geometry) {
        const g = painter.layout(null, { motion: m });
        for (const [k, v] of Object.entries(s.geometry.j)) close(g.j[k], v, `${id} beat ${s.beat} ${k}`);
        close(g.headC, s.geometry.head_c, `${id} beat ${s.beat} head`);
        close(g.headAngle, s.geometry.head_angle, `${id} beat ${s.beat} head angle`);
        assert.deepEqual(g.behind, s.geometry.behind, `${id} beat ${s.beat} behind`);
        assert.deepEqual(g.frontFirst, s.geometry.front_first, `${id} beat ${s.beat} front`);
      }
    }
  });
}

test("idle groove motion matches the Python", () => {
  for (const s of FIXTURE.idle) close(coachMotion(null, s.beat, { energy: 0.5 }), s.motion, `idle ${s.beat}`);
});

test("the drawn coach shows the judged pose on its beat (browser port)", () => {
  // As tests/test_dance_motion.py: on every judge point of every song, the
  // painter's arm angles (torso frame, hip centre to shoulder centre) are the chart's.
  const painter = new CoachPainter(stubArt(), "neon", { makeCanvas: fakeCanvas });
  const deg = (v) => Math.atan2(v[0], v[1]) * 180 / Math.PI;
  for (const { id } of FIXTURE.songs) {
    const song = parseSong(readJSON(`songs/${id}/song.json`), id, LIB);
    for (const jp of song.judgePoints()) {
      const g = painter.layout(null, { motion: coachMotion(song, jp.beat) });
      const { j } = g;
      const mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
      const sc = mid(j.l_shoulder, j.r_shoulder), hc = mid(j.l_hip, j.r_hip);
      const lean = Math.atan2(sc[0] - hc[0], hc[1] - sc[1]);
      const torso = (v) => [v[0] * Math.cos(-lean) - v[1] * Math.sin(-lean), v[0] * Math.sin(-lean) + v[1] * Math.cos(-lean)];
      const t = normalizePose(jp.pose);
      for (const [s, sign] of [["l", -1], ["r", 1]]) {
        const ang = (a, b) => { const v = torso([j[b][0] - j[a][0], j[b][1] - j[a][1]]); return deg([sign * v[0], v[1]]); };
        assert.ok(angleDiff(ang(`${s}_shoulder`, `${s}_elbow`), t[`${s}a`][0]) <= 6, `${id} ${jp.label} ${jp.beat} ${s} upper`);
        assert.ok(angleDiff(ang(`${s}_elbow`, `${s}_wrist`), t[`${s}a`][1]) <= 6, `${id} ${jp.label} ${jp.beat} ${s} fore`);
      }
    }
  }
  assert.ok(forward); // the import the landing hero relies on still exists
});
