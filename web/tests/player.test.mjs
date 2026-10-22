// PlayerLock parity: the browser port replays what the desktop decided on
// fixed crowds (web/tests/fixtures/player_parity.json, made by
// tools/make_player_parity_fixture.py from tests/crowd_poses.py;
// tests/test_player_parity.py fails when the fixture is stale) and must agree
// on every result: the state, the locked track, which pose the games get, the
// switch count, `changed`, the hold time and who goes to `others`. That
// includes a dense crowd where the Python makes a mistake: the port has to
// make the same one. Plus the browser-side wiring: the extrapolator forgets
// motion across an identity change, and selection is cheap.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { Pose } from "../js/core/pose.js";
import { PoseExtrapolator } from "../js/core/extrapolate.js";
import { PlayerLock, bodyOf, likeness, mostLikelyPlayer } from "../js/core/player.js";

const FIXTURE = JSON.parse(readFileSync(new URL("./fixtures/player_parity.json", import.meta.url), "utf8"));
const [W, H] = FIXTURE.size;
const STATE = { searching: "S", locked: "L", holding: "H" };

function toPose(p) {
  const xy = new Float32Array(34);
  const conf = new Float32Array(17);
  p.v.forEach((k, j) => {
    xy[2 * k] = p.xy[2 * j] / 10.0;
    xy[2 * k + 1] = p.xy[2 * j + 1] / 10.0;
    conf[k] = 0.9;
  });
  // float32 like the Python's np.float32 box
  const box = p.b == null ? null : Array.from(Float32Array.from(p.b, (v) => v / 10.0));
  return new Pose(xy, conf, 0.9, box);
}

const close = (got, want, tol, what) => assert.ok(Math.abs(got - want) <= tol, `${what}: ${got} vs ${want}`);

for (const s of FIXTURE.scenarios) {
  test(`player lock parity: ${s.name}`, () => {
    const lock = new PlayerLock(W, H);
    const repick = new Set(s.repick);
    let b = 0;
    s.frames.forEach((f, i) => {
      const poses = f.poses.map(toPose);
      if (repick.has(i)) lock.repick();
      const got = lock.update(poses, f.t);
      const [state, id, idx, switches, changed, held, others] = s.expect[i];
      const at = `${s.name} result ${i} (t=${f.t})`;
      assert.equal(STATE[lock.state], state, `${at}: state`);
      assert.equal(lock.playerId ?? 0, id, `${at}: track`);
      assert.equal(poses.indexOf(got), idx, `${at}: pose handed out`);
      assert.equal(lock.switches, switches, `${at}: switches`);
      assert.equal(Number(lock.changed), changed, `${at}: changed`);
      close(lock.heldFor, held, 1e-9, `${at}: heldFor`);
      assert.deepEqual(lock.others.map((o) => poses.indexOf(o)), others, `${at}: others`);
      while (b < s.bodies.length && s.bodies[b][0] === i) {
        const [, ax, ay, scale, sure, quality, like, box] = s.bodies[b];
        const k = s.bodies.filter((r, j) => j < b && r[0] === i).length;
        const body = bodyOf(poses[k]);
        close(body.ax, ax, 1e-9, `${at} body ${k} ax`);
        close(body.ay, ay, 1e-9, `${at} body ${k} ay`);
        close(body.scale, scale, 1e-9, `${at} body ${k} scale`);
        assert.equal(Number(body.sure), sure, `${at} body ${k} sure`);
        close(body.quality, quality, 1e-12, `${at} body ${k} quality`);
        close(likeness(body, W, null), like, 1e-12, `${at} body ${k} likeness`);
        body.box.forEach((v, j) => close(v, box[j], 1e-9, `${at} body ${k} box`));
        b++;
      }
    });
    s.frames.filter((_, i) => i % 10 === 0).forEach((f, j) => {
      const poses = f.poses.map(toPose);
      assert.equal(poses.indexOf(mostLikelyPlayer(poses, W)), s.pick[j], `${s.name}: most likely player ${j}`);
    });
  });
}

test("the fixture covers the cases that matter", () => {
  const names = FIXTURE.scenarios.map((s) => s.name).join(" | ");
  for (const want of ["passers", "occluded 0.5 s", "occluded 2 s", "golf", "edges", "leaving", "dense", "repick", "no boxes", "no shoulders"]) {
    assert.ok(names.includes(want), want);
  }
  const passers = FIXTURE.scenarios.filter((s) => s.name.startsWith("passers"));
  for (const s of passers) assert.equal(s.expect.at(-1)[3], 0, `${s.name}: no switches`);
});

test("the extrapolator forgets motion when the identity changes", () => {
  const a = toPose(FIXTURE.scenarios[0].frames[5].poses[0]);
  const xy = a.xy.map((v, i) => (i % 2 === 0 ? v + 500 : v));
  const b = new Pose(xy, a.confidence, 0.9, null);
  const ex = new PoseExtrapolator();
  ex.update([a], 0.0, 0.0);
  ex.forget();
  ex.update([b], 0.05, 0.05);
  assert.deepEqual(Array.from(ex.posesAt(0.1)[0].xy), Array.from(b.xy));
});

test("selection costs a few hundredths of a millisecond per result", () => {
  const s = FIXTURE.scenarios.find((x) => x.name.startsWith("dense"));
  const frames = s.frames.map((f) => ({ t: f.t, poses: f.poses.map(toPose) }));
  const lock = new PlayerLock(W, H);
  const spent = [];
  for (let round = 0; round < 3; round++) {
    lock.reset();
    for (const f of frames) {
      const t0 = performance.now();
      lock.update(f.poses, f.t);
      spent.push(performance.now() - t0);
    }
  }
  spent.sort((x, y) => x - y);
  assert.ok(spent[spent.length >> 1] < 0.3, `median ${spent[spent.length >> 1]} ms`);
});
