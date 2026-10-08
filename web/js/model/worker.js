// Inference worker: owns the ONNX Runtime session so the render loop on the
// main thread never waits on the model (the browser twin of pipeline.py).
//
// Messages in:  {type: "init", model: ArrayBuffer, eps: ["webgpu", "wasm"], resolution, width, height}
//               {type: "infer", pixels: ArrayBuffer (RGBA, resolution^2), t0, seq, threshold?}
// Messages out: {type: "ready", ep, adapter, warmupMs, threads}
//               {type: "poses", poses: [{xy, conf, score, box}], t0, seq, inferMs}
//               {type: "error", message}

// Pinned: 1.22 compiled RF-DETR's GridSample to an invalid WebGPU pipeline and
// silently returned garbage (no detections). Check accuracy on any upgrade
// with web/dev/pipeline.html.
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/ort.webgpu.min.mjs";
import { decodePoses } from "../core/decode.js";

// ImageNet normalisation, as rf-detr's own preprocessing.
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

let session = null;
let input = null;
let size = 336;
let frameW = 1280;
let frameH = 720;
let maxPeople = 1;

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === "init") await init(msg);
    else if (msg.type === "infer") await infer(msg);
  } catch (err) {
    self.postMessage({ type: "error", message: String(err?.message ?? err), phase: msg.type });
  }
};

async function webgpuAdapter() {
  if (!("gpu" in navigator)) return null;
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) return null;
    const info = adapter.info ?? {};
    return [info.vendor, info.architecture, info.description].filter(Boolean).join(" ") || "WebGPU";
  } catch {
    return null;
  }
}

async function init({ model, eps = ["webgpu", "wasm"], resolution = 336, width = 1280, height = 720, people = 1 }) {
  size = resolution;
  frameW = width;
  frameH = height;
  maxPeople = people;
  input = new Float32Array(3 * size * size);

  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(navigator.hardwareConcurrency || 4, 16) : 1;
  ort.env.webgpu.powerPreference = "high-performance";
  ort.env.logLevel = "error"; // graph-optimiser warnings are expected for this model and only noise

  const adapter = eps.includes("webgpu") ? await webgpuAdapter() : null;
  const plans = eps.filter((ep) => ep !== "webgpu" || adapter);
  let lastErr = null;
  for (const provider of plans) {
    try {
      session = await ort.InferenceSession.create(model, {
        executionProviders: [provider],
        graphOptimizationLevel: "all",
      });
      // Warm up: the first runs compile shaders / allocate; time the last one.
      const t = await warmup();
      self.postMessage({
        type: "ready",
        ep: provider,
        adapter: provider === "webgpu" ? adapter : null,
        warmupMs: t,
        threads: ort.env.wasm.numThreads,
        isolated: self.crossOriginIsolated,
      });
      return;
    } catch (err) {
      lastErr = err;
      session = null;
    }
  }
  throw lastErr ?? new Error("no execution provider worked");
}

async function warmup() {
  input.fill(0);
  let ms = 0;
  for (let i = 0; i < 3; i++) {
    const t = performance.now();
    await run();
    ms = performance.now() - t;
  }
  return ms;
}

async function run() {
  const feeds = { input: new ort.Tensor("float32", input, [1, 3, size, size]) };
  const out = await session.run(feeds);
  const raw = {
    dets: out.dets.data,
    labels: out.labels.data,
    keypoints: out.keypoints.data,
    Q: out.labels.dims[1],
    C: out.labels.dims[2],
    slots: out.keypoints.dims[2],
    D: out.keypoints.dims[3],
  };
  return raw;
}

async function infer({ pixels, t0, seq, threshold = 0.5 }) {
  if (!session) return;
  const px = new Uint8ClampedArray(pixels);
  const n = size * size;
  for (let c = 0; c < 3; c++) {
    // (v / 255 - mean) / std, folded into one multiply-add.
    const k = 1 / (255 * STD[c]), b = -MEAN[c] / STD[c], base = c * n;
    for (let i = 0; i < n; i++) input[base + i] = px[4 * i + c] * k + b;
  }
  const t = performance.now();
  const raw = await run();
  const poses = decodePoses(raw, frameW, frameH, threshold).slice(0, maxPeople);
  const inferMs = performance.now() - t;
  self.postMessage({ type: "poses", poses, t0, seq, inferMs }, poses.flatMap((p) => [p.xy.buffer, p.conf.buffer]));
}
