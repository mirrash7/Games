# Hosting the games on a website

**Short version:** this repo is a desktop Python app (PyTorch on the GPU via MPS,
an OpenCV window, direct camera access). A web page cannot run it as-is. There are
three ways to get the games onto a website. They differ a lot in effort and in
how the games feel, so choose one before building toward it.

## What we know (measured 2026-10-06, M5 Max)

- **The model leaves PyTorch cleanly.** `RFDETRKeypointPreview(...).export(shape=(336, 336))`
  produces one ONNX file of **143 MB** (fp32; fp16 should be about half). It needs
  `onnx>=1.17` and `onnxscript` installed, because the environment's own `onnx`
  is missing or too old for the exporter.
- **ONNX Runtime runs it correctly:** input `[1, 3, 336, 336]`, outputs `dets
  [1,100,4]`, `labels [1,100,2]` and `keypoints [1,100,34,8]`. Post-processing
  (decoding those to 17 keypoints) currently lives in rfdetr's Python and would
  need reimplementing.
- **CPU speed is far too slow for games:** 232 ms per frame (~4 FPS) in
  onnxruntime on CPU. A browser would have to use the GPU (WebGPU via
  onnxruntime-web). **That speed is not yet measured**, and it is the deciding number.
- Games currently need ~20-30 model updates per second with ~100 ms end-to-end
  latency (see AGENTS.md §6).

## Options

| | A. Runs in the browser | B. Server runs the model | C. Download the app |
|---|---|---|---|
| How | Port the runtime and games to TypeScript; RF-DETR via onnxruntime-web on WebGPU | Browser sends webcam frames to a GPU server running this Python model; games in the browser | Website is a landing page; players download this app |
| Player experience | Open a link and play; no install | Open a link; extra network lag | Install first; macOS-first |
| Camera privacy | Video never leaves the device | **Video is streamed to your server** | Video stays local |
| Latency | Depends on WebGPU speed (unmeasured) | +50-150 ms network round trip on top of ~100 ms: bad for Flappy/Fruit Ninja | Same as today |
| Running cost | Static hosting only | A GPU server per concurrent player group | Static hosting only |
| Effort | Highest: games rewritten in TS/Canvas; 70-143 MB model download | Medium: a frame-streaming server + browser games | Lowest: packaging, code signing |
| Same RF-DETR backbone | Yes (the ONNX export above) | Yes (unchanged) | Yes (unchanged) |

## Recommendation

1. **First, measure the deciding number:** onnxruntime-web + WebGPU running the
   exported model in Chrome on a typical laptop. At ≤ 50 ms per frame, option A
   is viable and is the best experience: a link, no install, private. At much
   more than that, the RF-DETR checkpoint is too heavy for in-browser real time,
   and the choice becomes C (or a lighter browser-native pose model, which would
   give up the "same backbone" requirement).
2. Keep this Python repo as the **reference implementation**: its tests, tuning
   numbers and measured lessons carry straight into a port. The game rules
   (`update` logic, physics constants, fairness proofs) are plain arithmetic
   and port mechanically. The OpenCV drawing does not; a web version would
   redraw with Canvas/WebGL using the same procedural art (the PNGs in
   `assets/*/generated/` can be served as-is).
3. Avoid B for these games: the network round trip lands on top of the
   existing latency, exactly where Flappy Bird and Fruit Ninja are least forgiving.

## Reproducing the export

```bash
uv run --with "onnx>=1.17" --with onnxruntime --with onnxscript python - <<'PY'
from rfdetr import RFDETRKeypointPreview
RFDETRKeypointPreview(device="cpu", resolution=336).export(
    output_dir="build/onnx", shape=(336, 336), batch_size=1)
PY
```

`build/` is git-ignored. The model file is too large for a normal git repo; host
it alongside the website or on a CDN.
