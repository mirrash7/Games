// Music playback clocks (js/core/audio.js), with a fake AudioContext; the twin of tests/test_audio.py.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  SilentPlayer, WebAudioPlayer, loadTrack, openPlayer, unlockAudio, setAudioContext, audioContext,
} from "../js/core/audio.js";

const near = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) <= eps, `${a} != ${b} (±${eps})`);

class FakeSource {
  constructor(ctx) {
    this.ctx = ctx;
    this.started = null;
    this.stopped = false;
    this.connected = null;
  }
  connect(node) { this.connected = node; }
  disconnect() { this.connected = null; }
  start(when, offset) { this.started = { when, offset }; }
  stop() {
    if (this.stopped) throw new Error("InvalidStateError");
    this.stopped = true;
  }
}

class FakeContext {
  constructor({ outputLatency = 0.04, baseLatency = 0.01, state = "running" } = {}) {
    this.currentTime = 0;
    this.outputLatency = outputLatency;
    this.baseLatency = baseLatency;
    this.state = state;
    this.sampleRate = 48000;
    this.destination = { name: "speakers" };
    this.sources = [];
    this.resumed = 0;
  }
  createBufferSource() {
    const s = new FakeSource(this);
    this.sources.push(s);
    return s;
  }
  createBuffer(channels, length, rate) { return { duration: length / rate }; }
  resume() {
    this.resumed++;
    this.state = "running";
    return Promise.resolve();
  }
  async decodeAudioData(data) { return { duration: data.byteLength / 1000, decoded: true }; }
}

function fakeClock() {
  const c = () => c.t;
  c.t = 50;
  return c;
}

function rig(opts) {
  const ctx = new FakeContext(opts);
  const clock = fakeClock();
  const player = new WebAudioPlayer({ duration: 10 }, { context: ctx, clock });
  // Advance the audio clock in 10 ms callback steps, the wall clock smoothly.
  const advance = (seconds) => {
    const steps = Math.round(seconds / 0.01);
    for (let i = 0; i < steps; i++) {
      ctx.currentTime += 0.01;
      clock.t += 0.01;
    }
  };
  return { ctx, clock, player, advance };
}

test("SilentPlayer: play, pause, resume, stop, finished", () => {
  const clock = fakeClock();
  const p = new SilentPlayer(10, { clock });
  assert.equal(p.position(), 0);
  assert.equal(p.playing, false);
  p.play();
  clock.t += 1.5;
  near(p.position(), 1.5);
  p.pause();
  clock.t += 5;
  near(p.position(), 1.5);
  p.resume();
  near(p.position(), 1.5);
  clock.t += 2;
  near(p.position(), 3.5);
  p.play(8);
  clock.t += 7;
  assert.equal(p.position(), 10);
  assert.equal(p.finished, true);
  p.stop();
  assert.equal(p.position(), 0);
  assert.equal(p.finished, false);
});

test("SilentPlayer never goes backwards while playing and pausing", () => {
  const clock = fakeClock();
  const p = new SilentPlayer(30, { clock });
  p.play();
  let last = 0;
  for (let i = 0; i < 300; i++) {
    clock.t += ((i * 37) % 50) / 1000;
    if (i % 7 === 0) p.pause();
    if (i % 5 === 0) p.resume();
    const pos = p.position();
    assert.ok(pos >= last);
    last = pos;
  }
});

test("WebAudioPlayer: position is what the listener hears (output latency compensated)", () => {
  const { ctx, player, advance } = rig();
  ctx.currentTime = 3; // the context has been running a while
  player.play();
  const src = ctx.sources.at(-1);
  assert.deepEqual(src.started, { when: 3, offset: 0 });
  assert.equal(src.connected, ctx.destination);
  assert.equal(player.position(), 0);
  advance(0.03);
  assert.equal(player.position(), 0, "nothing audible before the output latency has passed");
  advance(0.47);
  near(player.position(), 0.5 - 0.04, 1e-6);
});

test("WebAudioPlayer: baseLatency stands in when outputLatency is missing", () => {
  const { player, advance } = rig({ outputLatency: 0, baseLatency: 0.01 });
  player.play(2);
  advance(1);
  near(player.position(), 2 + 1 - 0.01, 1e-6);
});

test("WebAudioPlayer: smooth between audio callbacks, monotonic", () => {
  const { ctx, clock, player, advance } = rig();
  player.play();
  advance(1);
  const a = player.position();
  clock.t += 0.004; // the audio clock has not ticked yet
  near(player.position(), a + 0.004, 1e-6);
  clock.t += 1; // a stalled context: extrapolation is capped
  assert.ok(player.position() <= a + 0.05 + 1e-9);
  const b = player.position();
  ctx.currentTime += 0.01; // the context catches up by less than we extrapolated
  assert.ok(player.position() >= b, "never backwards");
});

test("WebAudioPlayer: pause freezes, resume continues with a new source", () => {
  const { ctx, player, advance } = rig();
  player.play(1);
  advance(2);
  const heard = player.position();
  near(heard, 1 + 2 - 0.04, 1e-6);
  player.pause();
  assert.equal(ctx.sources[0].stopped, true);
  assert.equal(player.playing, false);
  advance(1);
  assert.equal(player.position(), heard);
  player.resume();
  assert.equal(ctx.sources.length, 2, "a source node plays once: resume makes a new one");
  near(ctx.sources[1].started.offset, heard, 1e-9);
  assert.equal(player.position(), heard);
  advance(0.02);
  assert.equal(player.position(), heard, "holds until the resumed audio is audible");
  let last = heard;
  for (let i = 0; i < 50; i++) {
    advance(0.01);
    const pos = player.position();
    assert.ok(pos >= last && pos <= last + 0.011);
    last = pos;
  }
  player.pause();
  player.pause(); // twice is harmless
  player.stop();
  assert.equal(player.position(), 0);
});

test("WebAudioPlayer: finished at the end, clamped to the duration", () => {
  const { player, advance } = rig();
  player.play(9.5);
  advance(1);
  assert.equal(player.position(), 10);
  assert.equal(player.finished, true);
  player.stop();
  assert.equal(player.finished, false);
});

test("play() resumes a suspended context (it was not unlocked by a gesture yet)", () => {
  const { ctx, player } = rig({ state: "suspended" });
  player.play();
  assert.equal(ctx.resumed, 1);
});

test("unlockAudio creates and resumes the shared context", () => {
  const ctx = new FakeContext({ state: "suspended" });
  setAudioContext(ctx);
  assert.equal(unlockAudio(), ctx);
  assert.equal(ctx.resumed, 1);
  assert.equal(audioContext(), ctx);
  setAudioContext(null);
});

test("loadTrack decodes, openPlayer falls back to silence", async () => {
  const ctx = new FakeContext();
  const fetchImpl = async (url) => ({ ok: url === "ok.m4a", status: 404, arrayBuffer: async () => new ArrayBuffer(4000) });
  const buffer = await loadTrack("ok.m4a", { context: ctx, fetchImpl });
  assert.equal(buffer.duration, 4);
  await assert.rejects(loadTrack("missing.m4a", { context: ctx, fetchImpl }), /404/);
  const real = await openPlayer("ok.m4a", 106, { context: ctx, fetchImpl });
  assert.ok(real instanceof WebAudioPlayer);
  const warn = console.warn;
  console.warn = () => {};
  try {
    const silent = await openPlayer("missing.m4a", 106, { context: ctx, fetchImpl });
    assert.ok(silent instanceof SilentPlayer);
    assert.equal(silent.duration, 106);
    const none = await openPlayer("ok.m4a", 5, { context: null, fetchImpl });
    assert.ok(none instanceof SilentPlayer);
  } finally {
    console.warn = warn;
  }
});
