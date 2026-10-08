// The browser decoder must reproduce rf-detr's own PostProcess exactly.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { decodePoses } from "../js/core/decode.js";

const fx = JSON.parse(readFileSync(new URL("./fixtures/decode_people_walking.json", import.meta.url)));
const [, Q, C] = fx.labels_shape;
const [, , slots, D] = fx.keypoints_shape;
const raw = { dets: Float32Array.from(fx.dets), labels: Float32Array.from(fx.labels),
              keypoints: Float32Array.from(fx.keypoints), Q, C, slots, D };

test("finds the same people as rf-detr", () => {
  const poses = decodePoses(raw, ...fx.size, fx.threshold);
  assert.equal(poses.length, fx.expected.length);
});

test("scores, boxes and keypoints match rf-detr", () => {
  const poses = decodePoses(raw, ...fx.size, fx.threshold);
  const want = [...fx.expected].sort((a, b) => b.score - a.score);
  poses.forEach((p, i) => {
    const e = want[i];
    assert.ok(Math.abs(p.score - e.score) < 1e-4, `score ${p.score} vs ${e.score}`);
    p.box.forEach((v, j) => assert.ok(Math.abs(v - e.box[j]) < 0.05, `box ${j}`));
    e.keypoints.forEach(([x, y, c], k) => {
      assert.ok(Math.abs(p.xy[2 * k] - x) < 0.05, `kp ${k} x ${p.xy[2 * k]} vs ${x}`);
      assert.ok(Math.abs(p.xy[2 * k + 1] - y) < 0.05, `kp ${k} y`);
      assert.ok(Math.abs(p.conf[k] - c) < 1e-4, `kp ${k} conf`);
    });
  });
});

test("background detections never pass the threshold", () => {
  const all = decodePoses(raw, ...fx.size, 0.0);
  assert.ok(all.every((p) => p.conf.some((c) => c > 0)), "only person detections are returned");
});
