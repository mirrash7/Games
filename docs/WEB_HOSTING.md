# The browser build (GitHub Pages)

The arcade runs entirely in the browser: the same RF-DETR keypoint model,
exported to ONNX and run by onnxruntime-web, and a JavaScript port of
everything that runs per frame. No server sees the video. The site is static
and deploys to GitHub Pages from `web/` on every push to `main`.

URL once Pages is enabled: **https://mirrash7.github.io/Games/**

## GPU or CPU?

The page picks the backend itself, in this order:

1. **WebGPU + fp16 model (72 MB)** when the GPU supports `shader-f16`
   (Chrome/Edge on recent Macs and most recent PCs).
2. **WebGPU + fp32 model (143 MB)** on a WebGPU GPU without fp16.
3. **CPU (WASM) + fp32 model**, when there's no WebGPU (older browsers,
   some Firefox/Safari versions) or the GPU path fails.

The intro screen says which path the player will get, and the menu warns
when the model ended up on the CPU. The download starts as soon as the page
opens, while the player reads the intro and answers the camera prompt. It
waits for the click on a data-saver connection.

To test the CPU path, use **https://mirrash7.github.io/Games/?ep=wasm**, or
click "Test on the CPU only" on the intro card. `?model=fp32` forces the fp32
model on the GPU.

## Measured (2026-10-07, Apple M5 Max, Chromium, onnxruntime-web 1.29)

| path | model time / frame | pose updates / s |
|---|---|---|
| WebGPU, fp16 | ~20-25 ms | ~26 (camera-limited at 30 fps) |
| WebGPU, fp32 | ~29 ms | ~30 |
| CPU (WASM, 18 threads), fp32 | ~150 ms (measured on ORT 1.22, whose CPU kernels are correct) | ~6-7 |
| CPU (WASM, 1 thread: no cross-origin isolation) | ~1.2 s | ~1 |
| Desktop app, PyTorch MPS (for comparison) | ~33 ms plugged in | ~25 |

The browser's GPU path is as fast as the desktop app. The CPU path is
playable for Snack Attack (sluggish) but not for Flappy Raccoon, whose flap
detector wants 15+ updates a second. Ordinary laptops have fewer cores, so
expect worse than 6 per second there.

The CPU figures were measured while the browser tab was visible. A hidden tab
or pane throttles the model's worker threads by 10-20x, so the timings above
can't be reproduced from a background tab.

Accuracy: on a public sample image, WebGPU fp16, WebGPU fp32 and WASM fp32 all
find the same people. fp32 scores match the reference ONNX run to within 0.01,
and fp16 matches fp32 to within 0.01. (`web/dev/pipeline.html` reproduces
this; see below.)

### Things that cost a session to learn

- **onnxruntime-web 1.22 silently breaks this model on WebGPU.** RF-DETR's
  `GridSample` compiles to an invalid WebGPU pipeline; the session runs fast
  and returns no detections, with only console validation errors. 1.29 is
  correct. **Re-check accuracy with `web/dev/pipeline.html` on any ORT
  upgrade.** A speed benchmark on zero input will not catch this.
- **fp16 on the CPU backend hangs** (no fp16 kernels; casts everywhere), so
  fp16 is GPU-only and fp32 is the fallback.
- `onnxconverter_common` cannot convert this graph to fp16; ORT's
  `onnxruntime.transformers.float16.convert_float_to_float16(keep_io_types=True)`
  can (see `tools/export_web_model.py`).
- **Multithreaded CPU needs cross-origin isolation** (COOP/COEP headers for
  SharedArrayBuffer). GitHub Pages can't set headers, so `web/coi-sw.js`, a
  service worker, adds them and the page reloads once. If the browser refuses
  the service worker, the CPU path runs on one thread; WebGPU doesn't care.
- `img.decode()` never resolves while a tab is hidden; `gfx.loadImage` waits on
  `onload` instead.
- GitHub blocks files over 100 MB, so the models ship as equal chunks under
  48 MiB (`*.onnx.partN` + `manifest_<variant>.json`). The page downloads them,
  checks the SHA-256, and keeps them in Cache Storage per variant and hash, so
  a return visit starts in about a second.

## Layout

```
web/
  index.html, style.css     intro card, canvas, toolbar (camera picker, fullscreen, skeleton)
  coi-sw.js                 cross-origin isolation for static hosting
  js/main.js                boot (backend choice, downloads) and the render loop (app.py's twin)
  js/camera.js              getUserMedia -> mirrored 1280x720 frame; built-in camera over iPhone
  js/shell.js               welcome screen, dwell buttons, countdown, pause, game over (shell.py)
                            + browser-only: how-to-play cards, name entry, results table
  js/leaderboard.js         per-game top-10 tables: this browser, or worldwide (docs/LEADERBOARD.md)
  js/config.js              site settings (the worldwide leaderboard server's URL)
  js/model/                 loader.js (chunks + cache), engine.js (main thread), worker.js (ORT)
  js/core/                  decode (rf-detr PostProcess), pose, hand, controls, extrapolate,
                            random, theme, gfx
  js/games/                 base.js (the Game interface), index.js (registry), snack/, flappy/
  models/                   model chunks + manifests (the unsplit .onnx files are gitignored)
  tests/                    node:test suites (also run by pytest via tests/test_web.py)
  dev/                      dev-only harness pages (not published)
assets/<game>/generated/    game art, mounted into the site at assets/snack, assets/flappy
```

The render loop never waits on the model. Inference runs in a module Web
Worker; the main thread sends the newest frame, 336x336 RGBA, whenever the
worker is free, and projects poses to the current time with the same
`PoseExtrapolator` as the desktop app.

## Develop

```bash
uv run python tools/serve_web.py           # http://localhost:8765 with COOP/COEP set
cd web && node --test tests/*.test.mjs     # or: uv run --with pytest pytest tests/ -q
```

Open `http://localhost:8765/`. Useful URL options:

- `?debug=1` shows the skeleton and the timings (or press D).
- `?game=snack` or `?game=flappy` preselects a game.
- `?source=dev/people-walking.jpg` uses an image or video in place of the
  webcam. Sample media is gitignored; this one is Roboflow's public
  people-walking image.

Dev pages:

- `dev/pipeline.html?model=fp16&ep=webgpu&threshold=0.2` runs the model path
  on a still and reports backend, timing and scores.
- `dev/snack.html` and `dev/flappy.html` render contact sheets of key moments
  from scripted poses.
- `dev/tutorial.html?game=snack&t=1` freezes the how-to-play cards at a
  time, and `&screen=entry` or `&screen=results` shows the leaderboard
  screens. `&crop=x0,y0,x1,y1` enlarges part of the screen.

## Re-export the model

```bash
uv run --with "onnx>=1.17" --with onnxscript --with onnxruntime --with sympy --with packaging \
    python tools/export_web_model.py
```

This exports fp32 at 336, converts fp16, then splits both and writes the
manifests. Commit the new `web/models/*.part*` and `manifest_*.json`. The
SHA-256 in the manifest invalidates players' cached copies.

## Deploy

`.github/workflows/pages.yml` runs the browser tests and then
`tools/build_web.py`, which copies `web/` without tests, dev pages or unsplit
models, mounts the art and adds `.nojekyll`. It then publishes to Pages.

Pages has to be switched on once by a repo **admin**: Settings → Pages →
Build and deployment → Source: **GitHub Actions**. After that, every push to
`main` that touches `web/`, `assets/` or the workflow deploys within a couple
of minutes. Re-run it by hand from the Actions tab ("Deploy browser arcade" →
Run workflow).

## Keeping the two builds in step

The Python package is the reference and the JS is a port. Behaviour changes
(tuning numbers, rules, gesture thresholds) go into both, along with the
matching tests in `tests/` and `web/tests/`. The art lives once in `assets/`.
