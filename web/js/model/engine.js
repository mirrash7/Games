// Main-thread side of the pose pipeline: grabs the newest camera image, sends
// it to the inference worker when the worker is free, and keeps the latest
// result. The render loop reads `latest` and never waits (see AGENTS.md §2).

import { Pose } from "../core/pose.js";

export class RateMeter {
  constructor(window = 1.5) {
    this.window = window;
    this.ticks = [];
    this.ms = [];
  }
  tick(now, ms = null) {
    this.ticks.push(now);
    if (ms != null) this.ms.push(ms);
    while (this.ticks.length > 2 && this.ticks[0] < now - this.window) this.ticks.shift();
    while (this.ms.length > 30) this.ms.shift();
  }
  get fps() {
    const n = this.ticks.length;
    if (n < 2) return 0;
    return (n - 1) / Math.max(1e-6, this.ticks[n - 1] - this.ticks[0]);
  }
  get meanMs() {
    return this.ms.length ? this.ms.reduce((a, b) => a + b, 0) / this.ms.length : 0;
  }
}

export class PoseEngine {
  constructor({ resolution = 336, width = 1280, height = 720, people = 1 } = {}) {
    this.resolution = resolution;
    this.width = width;
    this.height = height;
    this.people = people;
    this.threshold = 0.5;
    this.worker = null;
    this.info = null;
    this.busy = false;
    this.latest = { poses: [], t0: 0, seq: 0 };
    this.meter = new RateMeter();
    this._seq = 0;
    this._small = document.createElement("canvas");
    this._small.width = this._small.height = resolution;
    this._sctx = this._small.getContext("2d", { willReadFrequently: true });
    this.error = null;
    this.onResult = null; // called when the worker becomes free again
  }

  /**
   * Start a worker with the model bytes, trying execution providers in order.
   * Resolves with {ep, adapter, warmupMs, threads}; on failure the worker is
   * shut down so the caller can retry with another model.
   */
  start(modelBuffer, eps = ["webgpu", "wasm"]) {
    this.stop();
    this.worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    return new Promise((resolve, reject) => {
      const failed = (err) => {
        this.stop();
        reject(err);
      };
      this.worker.onerror = (e) => failed(new Error(e.message || "the model worker failed to start"));
      this.worker.onmessage = (e) => {
        const m = e.data;
        if (m.type === "ready") {
          this.info = m;
          this.worker.onmessage = (ev) => this._onMessage(ev.data);
          resolve(m);
        } else if (m.type === "error") {
          failed(new Error(m.message));
        }
      };
      this.worker.postMessage({
        type: "init", model: modelBuffer, eps, resolution: this.resolution,
        width: this.width, height: this.height, people: this.people,
      }, [modelBuffer]);
    });
  }

  stop() {
    this.worker?.terminate();
    this.worker = null;
    this.busy = false;
  }

  _onMessage(m) {
    if (m.type === "poses") {
      this.busy = false;
      const now = performance.now() / 1000;
      this.meter.tick(now, m.inferMs);
      this.latest = { poses: m.poses.map((p) => Pose.fromObject(p)), t0: m.t0, seq: m.seq };
      this.error = null;
      this.onResult?.();
    } else if (m.type === "error") {
      this.busy = false;
      this.error = m.message;
      this.onResult?.();
    }
  }

  /**
   * Send `frame` (the mirrored 1280x720 camera canvas the games draw on) if
   * the worker is idle. The model sees the whole frame stretched to a square,
   * so its normalised outputs map straight back onto frame pixels.
   */
  submit(frame, t0) {
    if (!this.worker || this.busy) return false;
    this._sctx.drawImage(frame, 0, 0, this.resolution, this.resolution);
    const px = this._sctx.getImageData(0, 0, this.resolution, this.resolution).data;
    this.busy = true;
    this.worker.postMessage({ type: "infer", pixels: px.buffer, t0, seq: ++this._seq, threshold: this.threshold },
      [px.buffer]);
    return true;
  }

  /** Feed poses from somewhere else (demo mode, tests) through the same path. */
  inject(poses, t0) {
    this.latest = { poses, t0, seq: ++this._seq };
    this.meter.tick(t0);
  }
}
