// Boot and the 60 Hz render loop (the browser twin of app.py:_run_loop).
//
//   camera (rAF) --newest frame--> PoseEngine --worker--> ONNX Runtime (WebGPU or WASM)
//        |                              | latest {poses, t0}
//        v                              v
//   render loop: PoseExtrapolator -> ControlMapper -> Shell.update/render -> canvas
//
// The render loop never waits on the camera or the model.
//
// URL options: ?ep=wasm|webgpu (force a backend), ?game=snack|flappy (preselect),
// ?source=<image or video url> (stand-in for the webcam), ?debug=1 (skeleton + stats),
// ?leaderboard=<url> (a different leaderboard server, e.g. a local `wrangler dev`).

import { Camera, FileSource } from "./camera.js";
import { loadModel } from "./model/loader.js";
import { PoseEngine } from "./model/engine.js";
import { PoseExtrapolator } from "./core/extrapolate.js";
import { ControlMapper } from "./core/controls.js";
import { SKELETON } from "./core/pose.js";
import { banner, text, panel, WARN, css } from "./core/theme.js";
import { Shell } from "./shell.js";
import { GAMES } from "./games/index.js";
import { createBoard } from "./leaderboard.js";
import { LEADERBOARD } from "./config.js";

const W = 1280, H = 720;
const RESOLUTION = 336;
const params = new URLSearchParams(location.search);
const $ = (id) => document.getElementById(id);

const canvas = $("screen");
const ctx = canvas.getContext("2d", { alpha: false });
const ui = {
  intro: $("intro"), go: $("go"), device: $("device"), progress: $("progress"), bar: $("bar"),
  status: $("status"), failure: $("failure"), failureText: $("failure-text"), engine: $("engine"),
  camera: $("camera"), fullscreen: $("fullscreen"), debug: $("debug"),
};

let debug = params.get("debug") === "1";
let source = null;
let engine = null;
let variant = "";

// --- before start: tell the player which path their browser will take ---

/** {name, f16} for a usable WebGPU adapter, else null (the model runs on the CPU). */
async function probeGpu() {
  if (params.get("ep") === "wasm" || !("gpu" in navigator)) return null;
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) return null;
    return { name: adapter.info?.description || adapter.info?.vendor || "WebGPU", f16: adapter.features.has("shader-f16") };
  } catch {
    return null;
  }
}

const gpuP = probeGpu();
gpuP.then((gpu) => {
  const size = document.getElementById("model-size");
  if (size) size.textContent = gpu?.f16 ? "72 MB" : "143 MB";
  const cpuForced = params.get("ep") === "wasm";
  if (gpu) {
    ui.device.textContent = "Your graphics card will run the model (WebGPU). Expect a smooth 15-30 pose updates a second.";
  } else {
    ui.device.classList.add("cpu");
    ui.device.textContent = cpuForced
      ? "CPU only (you asked for it): about 6 pose updates a second on a fast laptop, slower elsewhere. Snack Attack is playable; Flappy Raccoon will feel sluggish."
      : "No WebGPU in this browser, so the model runs on your CPU: about 6 pose updates a second on a fast laptop, slower elsewhere. Snack Attack is playable; Flappy Raccoon will feel sluggish. Chrome or Edge on a recent computer uses the GPU.";
  }
  // A switch between the two paths, for comparing them.
  const other = new URL(location.href);
  if (cpuForced) other.searchParams.delete("ep");
  else other.searchParams.set("ep", "wasm");
  if (gpu || cpuForced) {
    const a = document.createElement("a");
    a.href = other.href;
    a.className = "switch";
    a.textContent = cpuForced ? "Use the GPU instead" : "Test on the CPU only";
    ui.device.append(" ", a);
  }
  ui.go.disabled = false;
  ui.go.focus();
  // Start the download now, so it is done (or nearly) by the time the player
  // presses play. Not on a data-saver connection: there it waits for the click.
  if (!navigator.connection?.saveData) beginModel();
});

// Art is small; fetch it straight away too.
const artP = Promise.all(GAMES.map((g) => g.preload()));
artP.catch(() => {});

/** Download and start the model once; every caller shares the same promise. */
let modelP = null;
function beginModel() {
  if (!modelP) {
    ui.progress.hidden = false;
    engine = new PoseEngine({ resolution: RESOLUTION, width: W, height: H });
    modelP = gpuP.then(startModel);
    modelP.then(
      (info) => setProgress(1, info.ep === "webgpu" ? "Pose model ready on your graphics card" : "Pose model ready on your CPU"),
      (err) => { if (!booting) fail(err); }, // shown at once, not on the next click
    );
  }
  return modelP;
}

ui.go.addEventListener("click", () => boot().catch(fail));

function setProgress(fraction, message) {
  ui.bar.style.width = `${Math.round(Math.min(1, Math.max(0, fraction)) * 100)}%`;
  if (message) ui.status.textContent = message;
}

function fail(err) {
  console.error(err);
  ui.intro.hidden = true;
  ui.failure.hidden = false;
  ui.failureText.textContent = String(err?.message ?? err);
}

let booting = false;
let running = false; // the arcade loop is up; until then keys and clicks do nothing

async function boot() {
  booting = true;
  ui.go.disabled = true;
  ui.go.textContent = "Waiting for the camera...";

  const sourceUrl = params.get("source");
  const camP = (sourceUrl ? new FileSource(sourceUrl, W, H) : new Camera(W, H)).open().catch((err) => {
    const name = err?.name ?? "";
    if (name === "NotAllowedError") throw new Error("Camera access was blocked. Allow the camera for this page (the camera icon in the address bar), then try again.");
    if (name === "NotFoundError" || name === "OverconstrainedError") throw new Error("No camera found. Plug one in, or close other apps using it, then try again.");
    if (name === "NotReadableError") throw new Error("The camera is busy in another app (Zoom, Teams, Photo Booth...). Close it and try again.");
    throw err;
  });
  const model = beginModel(); // usually already running since the page loaded
  source = await camP;
  ui.go.textContent = "Starting as soon as the model is ready...";
  const info = await model;
  await artP;
  console.info("[kp] model ready", info);
  variant = info.variant;

  ui.intro.hidden = true;
  ui.engine.className = `chip ${info.ep === "webgpu" ? "gpu" : "cpu"}`;
  if (info.ep !== "webgpu") {
    shell.notice = { text: "Running the model on your CPU, so tracking is slow. Chrome or Edge with WebGPU uses the GPU.", colour: WARN };
  }
  await fillCameraPicker();
  engine.onResult = () => sendFrame();
  running = true;
  requestAnimationFrame(tick);
}

/**
 * GPU with shader-f16: the fp16 model (72 MB, fastest). Otherwise, or if
 * that fails: fp32 (143 MB) on the GPU, then on the CPU. fp16 is GPU-only;
 * on the CPU backend it hangs.
 */
async function startModel(gpu) {
  const forced = params.get("ep");
  const attempts = [];
  if (gpu?.f16 && forced !== "wasm" && params.get("model") !== "fp32") attempts.push(["fp16", ["webgpu"]]);
  const fp32Eps = !gpu || forced === "wasm" ? ["wasm"] : forced === "webgpu" ? ["webgpu"] : ["webgpu", "wasm"];
  attempts.push(["fp32", fp32Eps]);
  let lastErr = null;
  for (const [variant, eps] of attempts) {
    try {
      const { buffer } = await loadModel(new URL("../models/", import.meta.url), variant, (f, cached, bytes) => {
        const mb = Math.round(bytes / 1e6);
        setProgress(f * 0.9, cached ? "Loading the pose model from the browser cache..."
          : `Downloading the pose model: ${Math.round(f * mb)} of ${mb} MB`);
      });
      setProgress(0.93, eps[0] === "webgpu" ? "Starting the model on your graphics card..." : "Starting the model on your CPU...");
      const info = await engine.start(buffer, eps);
      return { ...info, variant };
    } catch (err) {
      console.warn(`[kp] ${variant} on ${eps.join("/")} failed:`, err);
      lastErr = err;
    }
  }
  throw new Error(`The pose model could not start in this browser (${lastErr?.message ?? "unknown error"}).`);
}

// --- the loop ---

const shell = new Shell({ width: W, height: H, games: GAMES, mirrored: true, selected: params.get("game"),
  // ?leaderboard=<url> points this browser at another leaderboard server (local testing).
  board: createBoard(params.get("leaderboard") ? { url: params.get("leaderboard") } : LEADERBOARD) });
const extrapolator = new PoseExtrapolator({ gain: 0.7, maxLead: 0.12 });
const mapper = new ControlMapper(W, H);
let last = performance.now() / 1000;
let lastSeq = 0;
let unsent = false;
let renderTimes = [];
let chipAt = 0;

function sendFrame() {
  if (unsent && engine && !engine.busy) {
    engine.submit(source.frame, performance.now() / 1000);
    unsent = false;
  }
}

function tick() {
  const now = performance.now() / 1000;
  const dt = Math.min(now - last, 0.25);
  last = now;

  if (source.grab()) unsent = true;
  sendFrame();

  const latest = engine.latest;
  if (latest.seq !== lastSeq) {
    extrapolator.update(latest.poses, latest.t0, now);
    lastSeq = latest.seq;
  }
  const poses = extrapolator.posesAt(now);
  const controls = mapper.map(poses[0] ?? null);
  controls.stepPose = extrapolator.held[0] ?? null;

  const t0 = performance.now();
  shell.update(controls, dt);
  ctx.drawImage(source.frame, 0, 0);
  shell.render(ctx, source.frame);
  renderTimes.push(performance.now() - t0);
  if (renderTimes.length > 120) renderTimes.shift();

  if (debug) drawDebug(poses);
  if (source.age(now) > 2.5) banner(ctx, "CAMERA STOPPED", "Another app may be using it - or pick a different camera below");
  else if (engine.error) banner(ctx, "THE MODEL STOPPED", engine.error.slice(0, 90));

  if (now - chipAt > 0.5) {
    chipAt = now;
    const ep = engine.info.ep === "webgpu" ? "GPU" : "CPU";
    ui.engine.textContent = `${ep} · ${engine.meter.fps.toFixed(0)} poses/s · ${engine.meter.meanMs.toFixed(0)} ms`;
  }
  requestAnimationFrame(tick);
}

function drawDebug(poses) {
  ctx.save();
  ctx.lineWidth = 3;
  for (const p of poses) {
    for (const [a, b, colour] of SKELETON) {
      if (p.confidence[a] < 0.5 || p.confidence[b] < 0.5) continue;
      ctx.strokeStyle = colour;
      ctx.beginPath();
      ctx.moveTo(p.x(a), p.y(a));
      ctx.lineTo(p.x(b), p.y(b));
      ctx.stroke();
    }
    ctx.fillStyle = "#fff";
    for (let k = 0; k < 17; k++) {
      if (p.confidence[k] < 0.5) continue;
      ctx.beginPath();
      ctx.arc(p.x(k), p.y(k), 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.restore();
  const render = renderTimes.reduce((a, b) => a + b, 0) / Math.max(1, renderTimes.length);
  const lines = [
    `${engine.info.ep} ${variant}${engine.info.adapter ? ` (${engine.info.adapter})` : ""}  threads ${engine.info.threads}`,
    `model ${engine.meter.meanMs.toFixed(1)} ms   ${engine.meter.fps.toFixed(1)} poses/s`,
    `lead ${extrapolator.lastLeadMs.toFixed(0)} ms   render ${render.toFixed(2)} ms   people ${poses.length}`,
    `camera ${source.fps ? `${Math.round(source.fps)} fps` : "?"}  ${source.label.slice(0, 40)}`,
  ];
  panel(ctx, [W - 430, 12, W - 12, 24 + lines.length * 22], { alpha: 0.7, radius: 10 });
  lines.forEach((l, i) => text(ctx, l, W - 418, 34 + i * 22, 0.45, [235, 235, 235], { body: true, shadow: false }));
}

// --- input and chrome ---

window.addEventListener("keydown", (e) => {
  if (!running || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.target instanceof HTMLSelectElement) return;
  const k = e.key.toLowerCase();
  if (shell.screen === "entry") {
    if (shell.handleKey(e.key)) e.preventDefault(); // every letter is part of the name here
    return;
  }
  if (k === "f") toggleFullscreen();
  else if (k === "d") setDebug(!debug);
  else if (!shell.handleKey(e.key)) return;
  e.preventDefault();
});

canvas.addEventListener("click", (e) => {
  if (!running) return;
  const r = canvas.getBoundingClientRect();
  shell.handleClick(((e.clientX - r.left) / r.width) * W, ((e.clientY - r.top) / r.height) * H);
});

document.addEventListener("visibilitychange", () => {
  if (document.hidden) shell.pause(); // the player is not watching: never lose a life to that
});

function toggleFullscreen() {
  const stage = $("stage");
  if (document.fullscreenElement) document.exitFullscreen?.();
  else stage.requestFullscreen?.().catch(() => {});
}
ui.fullscreen.addEventListener("click", toggleFullscreen);

function setDebug(on) {
  debug = on;
  ui.debug.setAttribute("aria-pressed", String(on));
}
ui.debug.addEventListener("click", () => setDebug(!debug));
setDebug(debug);

async function fillCameraPicker() {
  if (!(source instanceof Camera)) {
    ui.camera.innerHTML = `<option>${source.label}</option>`;
    return;
  }
  const devices = await Camera.list();
  ui.camera.innerHTML = "";
  devices.forEach((d, i) => {
    const o = document.createElement("option");
    o.value = d.deviceId;
    o.textContent = d.label || `Camera ${i + 1}`;
    o.selected = d.deviceId === source.deviceId;
    ui.camera.append(o);
  });
  ui.camera.disabled = devices.length < 2;
  ui.camera.onchange = () => source.choose(ui.camera.value).catch((err) => console.error(err));
}

// Exposed for debugging in the console and for automated checks.
window.__kp = { shell, get engine() { return engine; }, get source() { return source; }, extrapolator, css };
