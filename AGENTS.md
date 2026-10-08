# Building games on KP — guide for agents

KP is a platform for **body-controlled games**: a webcam feeds the RF-DETR
keypoint model (17 COCO body keypoints per person, run locally on Apple MPS),
and games read the player's pose to drive play. One shared backbone — camera,
model, pose tracking, arcade menu — and any number of games on top.

This file is the handoff: how the platform works, how to add a game, and the
measured facts that took real sessions to learn. `README.md` holds the
player-facing docs and longer design rationale; read the relevant section
before changing a subsystem.

---

## 1. Run it

```bash
uv sync                                   # Python 3.12, deps incl. torch, rfdetr, opencv
uv run kp play --resolution 336           # the arcade (welcome screen, pick a game)
uv run kp game --game fruitninja          # arcade with one game preselected (Snack Attack)
uv run --with pytest pytest tests/ -q     # full test suite (fast, no camera needed)
```

Other commands: `kp cameras` (which camera sends real images, by name),
`kp bench` (model-only latency), `kp pose` (camera + skeleton, no game).
Useful flags: `--synthetic` (generated frames, no camera), `--headless`
(no window), `--duration N`, `--record DIR` (frames + metrics for review),
`--camera iphone|macbook|N`, `--verbose` (library logs), `--fps N`.

`--resolution` must be a **multiple of 24** (336, 504, 576). 336 is the
working default: fastest, and accurate enough for one close player.

---

## 2. How a frame flows

```
camera thread ──newest frame──► inference worker (RF-DETR, 15-30 Hz)
     │                                   │ Result(poses, timestamp)
     │                                   ▼
     └──────────► render loop, own 60 Hz clock (app.py:_run_loop)
                    PoseExtrapolator  → poses projected to *now*
                    ControlMapper     → ControlState (signals + raw pose)
                    Shell.update/render → your Game.update / Game.render
                    cv2.imshow
```

- The **render loop never waits on the camera or the model**. A stalled camera
  or slow inference must not freeze the game; poses update at 15-30 Hz while
  your game runs at 60 Hz.
- `PoseExtrapolator` (velocity projection, gain 0.7) runs before your game
  sees the pose. Motion therefore arrives in **steps**: a few near-identical
  frames, then a jump. See §6.
- **Global smoothing is off** (`Config.smoothing = 0`). Simulated through the
  real pipeline, the old EMA of 0.5 added ~36 ms of lag on battery. Filter in
  your game instead, where you can choose the trade-off: `HandTracker` (One
  Euro) for cursors, a time window for gestures. Raw poses wobble ~4-5 px held still.
- End-to-end latency, camera to screen, is roughly **85-150 ms**: ~40 ms camera,
  35 ms (plugged in) or 50 ms (battery, Low Power Mode) of model time, plus
  waiting for the next frame. On battery it's about 30 ms worse, and the app
  says so at startup. Tune every timing-sensitive mechanic for this.

---

## 3. Repo map

```
src/kpapp/
  app.py         CLI, camera selection, the 60 Hz render loop, recording hooks
  camera.py      threaded capture, health checks, choose-by-name, probe
  inference.py   RF-DETR wrapper -> Pose; PoseSmoother (off by default); PoseExtrapolator
  pipeline.py    inference worker thread
  controls.py    Pose -> ControlState (steer, throttle, gestures, raw pose)
  hand.py        HandTracker: stable palm cursor (use this for any pointing)
  shell.py       arcade: welcome screen, countdown, pause, game over, hover-to-press
  gfx.py         sprite drawing: alpha_composite, additive_composite, blit
  theme.py       the house look: palette, fonts, panel(), centred_text(), wrap()
  overlay.py     outlined_text, draw_banner, debug skeleton/HUD
  recorder.py    --record: frames + metrics.jsonl
  config.py      Config, KP (keypoint name -> index), SKELETON
  game/
    base.py        Game interface
    __init__.py    REGISTRY (name -> class) - register games here
    template/      Touch Targets: the reference game to copy (tested, off-menu)
    fruitninja/    Snack Attack: a raccoon on the palm eats snacks, avoids trash (Fruit Ninja rules; Blade swept-segment cuts)
    flappy/        Flappy Raccoon: FlapDetector (gesture.py), physics, picture-in-picture camera
    holeinwall/    full-body pose matching (off the menu, still runnable)
    demo.py        "catch": the first control-path demo
tools/generate_<game>_*.py   procedural, deterministic asset generators
assets/<game>/generated/     their output (+ manifest.json)
tests/test_<area>.py         one file per area; all run without a camera
```

---

## 4. Add a new game

1. **Copy the template.** `cp -r src/kpapp/game/template src/kpapp/game/<name>`,
   then rename the class, set `name`, `title`, `blurb`, delete `menu = False`.
   It already shows every contract below in about 100 lines.
2. **Satisfy the contract** (`game/base.py`, `shell.py`):
   - Constructor `(size, mirrored=True, seed=None, **options)`. The shell builds
     a **fresh instance** for every start and replay, passing `mirrored`.
   - `update(controls, dt)`: clamp `dt` (e.g. `min(dt, 0.1)`) so a stall never
     teleports objects through each other.
   - `render(frame)`: `frame` is the live 1280x720 **mirrored BGR camera image**.
     Draw on it in place: cover it fully, blend over it (Fruit Ninja), or show
     it picture-in-picture (Flappy).
   - `phase` with `.value == "game_over"` when the game ends. That's how the
     shell knows to offer PLAY AGAIN / MENU. Stop scoring once over.
   - `reset()`, and `on_resume()`, which resets any trackers or gesture history,
     because the player moved during the pause.
   - Optional: `swap_hand()` if your game has a dominant hand (the `h` key).
   - Keep the **bottom-left footer** (~y 700+) and the **bottom-right PAUSE
     button** (x 1090-1258, y 646-698) clear of important art. Score goes top-left
     or top-centre.
3. **Choose your input** from §5. Don't read raw keypoints when a primitive fits.
4. **Art** (if any): write `tools/generate_<name>_assets.py`, which outputs to
   `assets/<name>/generated/` plus a `manifest.json`, and an `art.py` loader that
   caches and fails with a "run the generator" message. Make art **injectable**
   (`art=None` constructor arg) so tests can pass stubs. See §7.
5. **Register** it in `game/__init__.py` `REGISTRY`. Games with a `title` and
   `menu = True` appear on the welcome screen. Add a countdown tip for it in
   `Shell._draw_countdown` (what to have in frame).
6. **Test** in `tests/test_<name>.py` (§8), then **verify visually and live** (§8).

---

## 5. Reading the player — input primitives

`ControlState` (from `controls.py`) arrives every frame:

| field | meaning |
|---|---|
| `pose` | the raw `Pose` (or None): `xy` (17,2) px, `confidence` (17,), `point(i, min_conf)` |
| `present` | a player with shoulders visible |
| `steer` | -1..1, hands' horizontal offset from the torso (falls back to torso lean) |
| `throttle` | 0..1, how far the higher hand is above the shoulders |
| `hands_up`, `arms_out`, `crouching` | latched booleans with on/off hysteresis |
| `left_hand`, `right_hand` | normalised 0..1 wrist positions |

Reusable primitives (prefer these over hand-rolled keypoint logic):

| need | use | notes |
|---|---|---|
| point / cursor with the hand | `hand.HandTracker(hand, mirrored).update(pose, t)` | palm (not wrist), survives elbow drop-outs, One Euro smoothed |
| one-off palm position | `hand.hand_point(pose, hand, mirrored)` | stateless version |
| which keypoint is the player's hand | `hand.hand_keypoint(hand, mirrored)` | mirrored feed: real right hand = `left_wrist` |
| swipes / cuts / hit-tests along motion | `game.fruitninja.blade.Blade` | swept segments, windowed speed, blade width, glitch guard |
| arm-flap (wing-beat) events | `game.flappy.gesture.FlapDetector` | one event per downstroke, hysteresis + refractory |
| match a whole-body shape | `game.holeinwall.matching.evaluate` + `HoleCache` | torso-normalised, joint + silhouette blend |
| smoothing any 2D signal | `hand.OneEuroFilter` | low jitter when slow, low lag when fast |
| hover-to-press buttons | `shell.Dwell` | blocks firing until the hand leaves after a screen change |

Use your own clock (sum of clamped `dt`) for timestamps you pass to these.

---

## 6. Hard-won facts — measured on real sessions; don't re-learn them

**Input**
- **Mirroring:** the feed is a selfie view, and the model labels limbs as they
  *appear*, so the player's real **right** hand is `left_wrist`. Use
  `hand_keypoint`. Getting this wrong is invisible in code and maddening in play.
- **Wrist ≠ hand.** Players aim with the palm. `HandTracker` projects 0.38 of a
  forearm past the wrist (`PALM_REACH`).
- **Elbows go missing:** in 54% of frames where the wrist was visible, the
  elbow wasn't. Usually it was below the frame with the hand raised. Snapping
  back to the wrist then made the cursor jump ~39 px/frame. `HandTracker`
  remembers the forearm and otherwise assumes it points up.
- **Hips are often out of frame** (players stand close). Normalise by
  **shoulder width** and measure heights from the **shoulder line**, not the hips.
- **Stepped motion:** per-frame velocity flickers between ~0 and huge. Measure
  speed over a **window** (~80 ms) and test motion as **swept segments**, never
  "is it inside the target right now". A fast hand jumps 50-250 px between pose updates.
- **Gesture detectors read `step_pose`, not `pose`.** `controls.pose` is
  extrapolated to every render frame, so each new model result arrives as a jump
  between two frames 8-16 ms apart. Anything that times motion between model
  updates (FlapDetector) then sees "teleports" and drops its history. Simulated at
  120 Hz render, Flappy caught 0 of 30 flaps; live it felt like "a few flaps work,
  then the bird falls". `controls.step_pose` (JS: `stepPose`) is the newest result
  projected to when it arrived, held until the next one: 30/30, and up to 40 ms
  quicker than the raw result. Cursors (HandTracker) still use `pose`.
- **Teleports happen:** a tracking glitch can jump a keypoint across the
  screen. Treat implausible jumps (> ~35% of the width between samples) as
  "lost, then found", not as motion.
- **Never punish tracking loss.** If you can't see the player, say what to
  fix on screen ("STEP BACK — HIPS MUST BE IN FRAME") and don't take a life.
- Feeding the model RGB vs BGR made no measurable difference; it's RGB now,
  per the docs.

**Game design under ~100 ms latency**
- Difficulty should **ease in and keep rising**. Flappy Raccoon's knobs follow
  `limit + (start - limit) * exp(-score / tau)`: generous at first, harder at
  every score, never flat. Prove fairness at the limit, not just the start;
  its test flies 120 pipes with 150 ms of lag.
- Original arcade tunings are too tight. Widen hit areas (Fruit Ninja's blade
  is a 34 px half-width stroke, and its trail stays sharp for 0.15 s), start
  gentle (one or two fruit, no bombs in the first waves), and require a **hold**
  or **speed** so passing hands don't trigger things.
- Random by default (`random.Random(seed)` with `seed=None`). Two games once
  shipped with fixed seeds and played identically every round. Keep a `seed`
  argument for tests.
- Fairness should be **provable**. Fruit Ninja solves bomb-vs-fruit closest
  approach exactly at launch (same gravity, so relative motion is linear).
  Sampling every 1/30 s missed a real crossing.

**Rendering (60 FPS budget: aim for under 4 ms per game frame)**
- Pre-scale and pre-render at init. Per frame, blend only the region you touch
  (a full-frame float blend is ~7 ms). Use `gfx.blit` / `alpha_composite`.
- Never use `cv2.dilate` with a big kernel per frame: 22 ms at 65 px, 363 ms
  at 259 px. Bake padding into the drawing instead.
- **OpenCV 5 text:** thickness changes font *weight and width*, so the
  "thick black text, then thin colour on top" outline drifts into ghost
  letters. Use `overlay.outlined_text`.
- Cache rotated sprites in angle buckets (see `FruitNinjaRenderer._rotated`).

---

## 7. Look and feel, and art

**Every game should look like part of one arcade.** Use `kpapp.theme`, never ad-hoc colours:
- `theme.ACCENT` (gold) for titles and the score; `theme.CYAN` for active,
  hovered or positive feedback; `theme.WARN` for guidance the player must act
  on; `theme.DANGER` for loss; `theme.DIM` for hints. All colours are BGR.
- Titles and numbers in `theme.TITLE_FONT` (DUPLEX) with an outline: use
  `theme.centred_text`, which draws through `outlined_text`. Body text in
  `theme.BODY_FONT`.
- Overlays (scoreboards, prompts) sit on `theme.panel(...)`: a dark,
  translucent, rounded panel with an optional accent border.
- Juice that the existing games share: a "+N" popup at the point of scoring that
  pops in, rises and fades (~0.9 s); a score that pulses when it changes; a short
  flash or shake on failure. Copy the pattern from Fruit Ninja's or Flappy's `render.py`.
- Score top-left or top-centre, never under the shell's PAUSE button (bottom-right)
  or footer (bottom-left).


- **Original and procedural only:** numpy + OpenCV drawing in
  `tools/generate_<name>_*.py`. Never download or imitate a commercial game's
  sprites. Matching a game's *mechanics* is fine.
- **Deterministic:** seed all randomness and verify byte-identical output
  across two runs (md5).
- Gameplay numbers that belong to the art (hit radius, ground height, juice
  colour, points) go in `manifest.json`, so they travel with the sprites.
- BGRA for sprites. Draw **tintable** sprites in pure white with alpha.
  Dither dark gradients (720p bands badly otherwise). Leave padding for rotation.
- The player stands ~2 m back: size everything to read at that distance.

---

## 8. Testing and verification

**Unit tests** (no camera, no model, all under ~2 s each):
- Inject stub art and a seed. For pose input, build `Pose(xy, confidence)` by
  hand: see `hand_pose` in `tests/test_fruitninja.py` and `palm_at` in
  `tests/test_template.py`.
- Simulate **realistic** input: multi-step swipes (no 800 px single-frame
  jumps; real play never exceeded ~245 px), stepped updates, noise, drop-outs.
- Pin the *reason* for a design in a test (e.g. "containment alone would not
  discriminate"), so it can't quietly regress.
- Tests through the real shell: `Shell(size, {name: Cls}, selected=name)`,
  then `handle_key(ord("1"))` and step through the 3 s countdown.

**Headless performance** (synthetic frames, no camera):
```bash
uv run kp game --game <name> --synthetic --headless --duration 10 --resolution 336
```
Expect `render 60.0 FPS`. Profile render pieces individually before optimising.
**Benchmark by interleaving A/B runs.** On battery, Low Power Mode swings
model latency by ~10 ms between runs.

**Look at it.** Render a contact sheet of key moments with the real art, then
open the image and actually inspect it. Most bugs found this way were invisible
to tests: ghost text, a cursor off the palm, popups too small.

**Live sessions** with a real player:
```bash
uv run kp play --resolution 336 --record <dir>
```
`<dir>/metrics.jsonl` has one line per frame: `t`, `screen`, `phase`, `score`,
`lives`, `render_ms`, `frame_ms`, `infer_ms`, `infer_fps`, `render_fps`,
`camera_age_ms`, `people`, wrist/hip confidences, blade speed/slicing. You also
get periodic and event JPEGs plus raw camera PNGs every ~3 s, which you can
re-run the model on offline. Analyse these before tuning anything.

---

## 9. Environment notes (this Mac)

- **Camera access:** an agent's sandboxed shell may be denied the camera by
  macOS. The user's own terminal has permission. Launch live runs there (in
  Claude Code: `run_in_terminal`), not from a sandboxed Bash.
- **Camera selection is by name.** The built-in camera is the default; the
  iPhone (Continuity Camera) is never chosen automatically. Camera *numbers*
  swap between devices.
- **A closed lid gives black frames:** the built-in camera stays listed but
  sends exact-zero frames. After opening the lid it takes ~3 s to wake. The app
  detects blank streams within ~2.5 s, reopens the camera twice, then stops
  with a message.
- Two `objc[...] implemented in both` lines at startup are expected. PyAV and
  OpenCV both bundle FFmpeg's camera classes. Capture is pinned to the native
  AVFoundation backend, so our code never uses the duplicates.
- Model speed: ~33 ms/frame on AC power, ~50-65 ms on battery in Low Power Mode.

---

## 10. Current state and open work

- **On the welcome screen:** Snack Attack (`fruitninja`: Fruit Ninja re-themed with
  a purple raccoon cursor, cotton candy and other snacks, and trash bags for
  bombs) and Flappy Raccoon (`flappy`: the same raccoon with wings). Flappy lives in
  `game/flappy/`: original art from `tools/generate_flappy_assets.py`,
  `FlapDetector` in `gesture.py`, and the game in `game.py`/`render.py`. Its
  physics are tuned so a bot with 150 ms input lag passes every generated course.
  It shows STEP BACK when the player stands too close: wrists then leave the
  frame on every downstroke, and at a 15-20 Hz model rate up to ~30% of flaps
  are missed. Not yet tested with a live player.
- **Off the menu:** Hole in the Wall (`--game holeinwall`), Catch (`--game catch`),
  Template (`--game template`).
- Ideas the platform already supports: dodge games (`steer` / torso lean),
  rhythm games (`hands_up` / `arms_out` on beat), boxing (swept-segment punches
  via `Blade` on both hands), drawing (palm cursor + trail), two-player
  (`--max-people 2`; you'd need identity tracking, because pose order is by
  score and can swap between frames).
- **Website:** the browser build in `web/` (GitHub Pages) runs the same model
  with onnxruntime-web and a JavaScript port of the runtime, shell and both
  games. See §11 and `docs/WEB_HOSTING.md`.
- Known limits: one player is the tuned case. Latency is ~100 ms+. The preview
  keypoint model is the only checkpoint available (no smaller variant).

---

## 11. The browser build (`web/`)

The arcade also runs in the browser: https://mirrash7.github.io/Games/, deployed
by `.github/workflows/pages.yml` from `web/` on every push to `main`. Full notes,
measurements and deployment are in `docs/WEB_HOSTING.md`. In short:

- **Same model, same rules.** RF-DETR is exported to ONNX (`tools/export_web_model.py`)
  and run by onnxruntime-web in a Web Worker: WebGPU + fp16 when the GPU has
  `shader-f16`, else fp32 on WebGPU, else fp32 on the CPU (WASM, ~6 updates/s).
  `web/js/core/decode.js` is a tested port of rf-detr's PostProcess.
- **The Python code is the reference; the JS is a port.** Module for module:
  `core/hand.js`, `core/controls.js`, `core/extrapolate.js`, `shell.js`,
  `games/snack/*` (fruitninja), `games/flappy/*` (incl. `gesture.js`). A behaviour
  change goes into both, with tests in `tests/` and `web/tests/`. Colours in JS
  are **RGB** (`core/theme.js`), not OpenCV's BGR: convert, or purple turns orange.
- **A new browser game:** a class extending `games/base.js` (`static id/title/blurb/tip`,
  `static async preload()` for art, `update(controls, dt)`, `render(ctx, frame)`,
  `phase`), registered in `games/index.js`. Logic modules must import under Node
  with no DOM, so tests run without a browser; build the renderer lazily.
- **Art** stays in `assets/<game>/generated/`. `tools/build_web.py` (Pages) and
  `tools/serve_web.py` (local) mount it into the site; add a line to `MOUNTS`.
- **Verify:** `cd web && node --test tests/*.test.mjs` (pytest runs it too), then
  look at it: `uv run python tools/serve_web.py`, open `http://localhost:8765/?debug=1`
  (`?source=<image/video>` stands in for a webcam), plus `dev/<game>.html` contact
  sheets and `dev/pipeline.html` for model accuracy and speed.
- **Browser-only screens:** each game's `static tutorial` (cards drawn with
  `core/figure.js`, shown before a player's first round per visit) and the
  leaderboards (`leaderboard.js`, `docs/LEADERBOARD.md`: name entry by hovering
  over letters, per-game top 10, per browser or worldwide via the Cloudflare
  Worker in `leaderboard/`). A new game needs a `score` property and an entry in
  `leaderboard/src/rules.js` (its fastest possible scoring rate, with a test).
- **Pinned onnxruntime-web 1.29.** 1.22 silently returned no detections on WebGPU
  (broken `GridSample`). Check accuracy with `dev/pipeline.html` after any upgrade.
