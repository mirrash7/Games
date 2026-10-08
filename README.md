# KP — real-time RF-DETR keypoint control

> **Building a new game or picking up this work?** Start with [AGENTS.md](AGENTS.md):
> how the platform works, the game contract, input primitives, and measured pitfalls.

Runs the RF-DETR keypoint model locally on live webcam video and turns body pose
into game control input.

## Play in the browser

**https://mirrash7.github.io/Games/**: no install. The same model runs inside
the browser (onnxruntime-web), on the GPU via WebGPU when available (~20-30 ms a
frame on an M5 Max, as fast as this desktop app) and on the CPU otherwise
(~150 ms, ~6 pose updates a second: Snack Attack is playable, Flappy Raccoon
is not). Video never leaves the computer. Each game has a how-to-play screen and its
own leaderboard ([docs/LEADERBOARD.md](docs/LEADERBOARD.md)). How it's built,
measured and deployed: [docs/WEB_HOSTING.md](docs/WEB_HOSTING.md). The rest of this README
covers the desktop Python app, which is the reference implementation.

## Measured on this machine (Apple M5 Max, MPS)

| | |
|---|---|
| Model inference | **44 ms — 22.6 FPS** (576px, graph-optimized) |
| Eager, unoptimized | 55 ms — 18 FPS |
| Full loop (capture + infer + game + overlay) | **37 FPS render / 22 FPS inference** |

Inference and rendering run on separate threads, so the picture stays smooth at
camera rate while poses update at model rate. Keypoints are then projected
forward in time (see *Latency* below) so controls track the player rather than
lagging one inference behind.

### Model resolution (measured, 1280x720 input)

Resolution must be a multiple of **24** (patch_size 12 x 2 windows). 392, 448
and 560 are rejected outright.

| `--resolution` | Latency | FPS | Accuracy on a hard crowd photo |
|---|---|---|---|
| 336 | 29.9 ms | 33.4 | 2/4 people, 19 keypoints — big loss |
| 504 | 38.9 ms | 25.7 | 4/4 people, 49 keypoints — no loss |
| **576** (default) | 44.5 ms | 22.5 | 4/4 people, 47 keypoints |
| 672 | 53.9 ms | 18.6 | — |

`--resolution 504` is close to free: ~14% faster with no measured accuracy cost.
The 336 numbers come from distant, tiny subjects; a single close webcam subject
should degrade far more gracefully, so it is worth trying for your own scene.

### Camera resolution barely matters

The model resizes every frame to a square `--resolution` internally, so capture
size only changes preprocessing cost:

| Capture | Inference | Full-loop render |
|---|---|---|
| 1920x1080 | 45.6 ms | 38.1 FPS |
| 1280x720 | 44.1 ms | 37.3 FPS |
| 640x480 | 43.5 ms | 36.8 FPS |

Dropping 1080p to 480p buys ~2 ms and no render throughput, while feeding the
model an upscaled image. Not a useful lever — change `--resolution` instead.

## Setup

```bash
uv sync
```

Weights (~1 GB) download automatically on first run into `~/.roboflow/models`.

**macOS camera permission:** the first live run triggers a camera prompt. If you
launch from a terminal, grant camera access to that terminal under
System Settings → Privacy & Security → Camera, then rerun.

## Run it

Live skeleton overlay — the real-time sanity check:

```bash
uv run kp pose
```

The pose-controlled demo game:

```bash
uv run kp game
```

### Controls

| key | action |
|---|---|
| `1`-`2`, `Enter` | start a game from the welcome screen |
| `Space` / `p` | pause / resume |
| `r` | restart the current game |
| `Esc` | back one screen (quits from the welcome screen) |
| `h` | switch which hand points and slices |
| `s` / `d` | skeleton overlay / debug stats (off by default) |
| `q` | quit |

Closing the window also quits. These are shown bottom-right in game.

**Framing matters.** The shapes are full-body, so the camera must see you from
head to hips at minimum — knees and ankles may be out of frame without penalty.
Stand back far enough that your hips are visible with room to throw your arms
out. If the game cannot see enough of you it tells you what is missing, and
replays the wall rather than charging you a life.

### Camera trouble

```bash
uv run kp cameras
```

Probes every camera and reports, by name, which ones deliver real images.

**The app uses the built-in Mac camera by default and never switches to a phone
on its own.** Camera *numbers* are not stable on a Mac - with Continuity Camera
the built-in camera and an iPhone have swapped between 0 and 1 on this machine -
so the camera is chosen by name and type instead. OpenCV numbers devices by
their position in AVFoundation's device list, so reading that list (via PyObjC)
gives the name behind each number. To use another camera deliberately, name it:

```bash
uv run kp play --camera iphone
```

If the built-in camera sends only black frames, the app reopens it (that has
revived it before), then stops with a message rather than falling back to the
phone. A closed MacBook lid is the usual cause: macOS still lists the camera,
but it can only send black.

The app handles this itself now. The camera starts warming up before the
model loads, so a slow cold start (no frames for several seconds) is waited out
in time you are already waiting through. A camera streaming *solid blank
frames* at full rate is a different failure - it will not come good by waiting
- so it is abandoned as soon as that is clear, not re-probed, and the app
moves on. On this Mac that took launch from ~15 s of waiting to ~2 s after the
model loads. Mid-game, a stall
or blank feed puts a warning on screen and the app tries to reopen the device.
The game keeps running throughout, because the render loop has its own 60 FPS
clock instead of waiting on camera frames (`--fps` changes the rate).

Camera access is pinned to OpenCV's native AVFoundation backend (which "auto"
picks anyway), so it can never fall back to its FFmpeg path. That matters
because the model's first prediction loads PyAV, which bundles a second copy of
FFmpeg's macOS camera classes; the two `objc[...] ... implemented in both`
lines at startup are macOS noticing the duplicate. With the native backend our
capture never touches those classes. Other startup chatter (rf-detr checkpoint
notes, ~20 graph-tracing warnings) is silenced during model setup; add
`--verbose` to see it.

### Recording a session

```bash
uv run kp game --game fruitninja --duration 45 --record out/
```

Saves a frame every second plus one on every scoring event and phase change,
raw camera frames every few seconds, and `metrics.jsonl` with per-frame timings,
camera health, tracking confidence and game state.

**Performance on battery:** Low Power Mode throttles the GPU. The same 336px
model measured 30 ms on power and 38-47 ms on battery, swinging ~10 ms run to
run, so compare settings with interleaved runs rather than back-to-back.

### Verify speed without a webcam

Model throughput only:

```bash
uv run kp bench --iters 40
```

Whole pipeline on generated frames, no camera or window needed:

```bash
uv run kp game --synthetic --headless --duration 12
```

### Useful flags

`--resolution 504` (multiples of 24 only) · `--threshold 0.5` ·
`--device mps|cpu` · `--no-compile` · `--max-people 3` · `--smoothing 0.3` (extra keypoint smoothing, off by default) ·
`--extrapolation 0` (disable velocity projection) ·
`--infer-every 3` (cap model rate; only bites below ~22 FPS) · `--camera 1` · `--no-mirror`

## Games

```bash
uv run kp play --resolution 336
```

Opens the **Motion Arcade** welcome screen. Nothing starts on its own: point
with your right hand and hold it over a game for about a second (or press 1/2),
then a 3-2-1 countdown gives you time to step back before play begins.
`uv run kp game --game fruitninja` opens the same screen with that game picked.

Everything works standing back from the keyboard:

| on screen | hover your hand | or press |
|---|---|---|
| welcome | over a game tile (1 s) | `1`/`2`, `Enter` |
| playing | on **PAUSE**, bottom-right (1.5 s, so passing hands don't pause) | `Space` |
| paused | **RESUME** / **MENU** | `Space` / `Esc` |
| game over | **PLAY AGAIN** / **MENU** | `R` / `Esc` |

After any screen change a button won't fire until your hand has moved off it,
so a hand resting where a new button appears can't click through by accident.

### Hole in the Wall

A wall advances on you with a body-shaped opening cut through it. Match the
shape before it arrives. 16 shapes across three difficulty tiers, ramping as you
clear walls; three lives; streak multiplier capped at 10x.

Runs at **30 FPS render / 30 FPS inference** at `--resolution 336`, with the
scene costing ~4.8 ms per frame (p95 8.6 ms).

**The opening is locked to your body**, not to the middle of the screen — it
tracks your torso centre and your apparent size. Scoring is already invariant to
where you stand and how far you are from the camera, so pinning the hole to the
screen would let a perfectly matched pose look like a miss. This way the picture
and the score always agree, and you never have to hunt for a sweet spot in frame
before you can play.

**Only visible joints are judged.** The target silhouette is drawn with the same
visibility as the player, so limbs the camera cannot see are excluded from both
sides. Without this, a flawless pose with knees out of frame scored 0.63 overlap
and failed — punishing the player for their room, not their pose.

**How the fit is judged.** Two metrics, blended 60/40:

| | what it catches |
|---|---|
| per-joint agreement | a single wrong limb — every joint weighted equally |
| silhouette IoU | overall shape, including limbs the joints miss |

Neither works alone. Containment — "is your body inside the opening?" — is the
obvious test and it is *useless*: any compact pose sits entirely inside a
spread-eagle hole, so arms-down scores a perfect 1.000 against a STAR. Silhouette
IoU alone is dominated by torso pixels, so one wrong arm barely moves it (0.89
for a clearly wrong pose). The blend separates cleanly — identical 1.00, small
error 0.95, one wrong arm 0.85, arms-down-instead-of-out 0.71 — with the pass
line at 0.82 and PERFECT at 0.93. `tests/test_holeinwall.py` pins this down,
including a regression test asserting containment alone would *not* discriminate.

### Snack Attack (Fruit Ninja rules, raccoon theme)

Your hand is a **purple-and-white raccoon** (an original character inspired by
Roboflow's mascot, Lenny) that leaves a purple trail. Swipe it through the
flying snacks to gobble them: cotton candy (the most common), plus other
raccoon favourites. Avoid the **bags of trash**: grabbing one ends the run. The
raccoon opens its mouth on every bite. Its hit area matches the drawn head
(44 px half-width).

The code keeps its Fruit Ninja names (`game/fruitninja/`, `--game fruitninja`):
a "fruit" is a snack and a "bomb" is the bag of trash. Art comes from
`tools/generate_fruitninja_snacks.py` and `tools/generate_fruitninja_scene.py`.

The rules and tuning below are unchanged from Fruit Ninja.

#### Fruit Ninja mechanics

Classic rules: a point per fruit, a bonus for cutting several in one swing,
three dropped fruit and you are out, and slicing a bomb ends the run
immediately. Your **right hand is the blade** — the trail follows your wrist.

Renders in ~1 ms in normal play (7.3 ms on a deliberately overloaded board).

**The blade sits in the middle of your palm, and holds steady.** The model
tracks the wrist joint, but you aim with your hand, so the cursor is pushed
0.38 of a forearm past the wrist (`PALM_REACH` in `hand.py`).

On recorded play the elbow was missing in 54% of frames where the wrist was
visible - usually below the bottom of the frame with the hand raised. The old
cursor fell back to the bare wrist each time, which put the dot at the base of
the hand *and* made it jump ~39 px per frame between palm and wrist as the
elbow flickered in and out: that was the jittery path. `HandTracker` now
remembers the forearm across short drop-outs and otherwise assumes it points
straight up (mid-palm in 8 of 9 such recorded frames). A One Euro filter
removes the remaining tracking noise - heavy smoothing when the hand is slow,
almost none during a swipe - taking a still hand from 4.6 to 2.0 px of shimmer
for about 4 ms of added lag. With the elbow flickering, the dot now moves
1.6 px per frame instead of 39. The drawn streak is also rounded with Chaikin
subdivision; cuts are still tested on the real samples.

**Hitting is deliberately forgiving.** The ring around the blade tip is its
real hit area. Three knobs in `fruitninja.game.Rules` set how generous it is:

| setting | default | effect |
|---|---|---|
| `hit_radius` | 34 px | blade half-width, added to each fruit's radius |
| `cut_window` | 0.15 s | how long the trail stays sharp - fruit flying into it is cut |
| `min_speed` | 260 px/s | how fast a swing must be to cut (was 380) |

Speed is averaged over 80 ms, because tracking updates the hand in steps at
15-30 Hz while the game runs at 60 Hz - frame-to-frame speed flickers between
zero and huge and would randomly drop real swings. A jump of more than 35% of
the screen between samples is treated as the tracker losing the hand: it
starts a fresh stroke instead of cutting a line across everything.

**Every game is different, and bombs are always avoidable.** The spawner used
to run on a fixed seed, so every round threw the same fruit to the same places
(Hole in the Wall had the same bug with its shape order). Both are now unseeded;
pass a seed only for reproducible tests.

Fruit and bombs fall under the same gravity, so their separation moves in a
straight line and each object's time on screen is an exact interval (linear x,
parabolic y). At launch the spawner solves the true closest approach of every
bomb to every fruit while both are on screen, and only throws a bomb on a path
that stays at least fruit radius + bomb radius + blade half-width + 30 px clear;
a bomb with no safe path is not thrown. A test checks this independently with
real game physics, frame by frame, over 90 seconds at the hardest difficulty.
An earlier version that *sampled* the paths every 1/30 s missed a fruit rising
past a falling bomb at the bottom edge (closing at ~2700 px/s); solving it
exactly is both correct and cheap - 1.6 ms per wave, versus up to 58 ms for
sampling fine enough to be safe.

**Every cut pays out visibly:** a "+1" (or a larger "+2" for pineapple and
coconut) pops up at the cut in the fruit's juice colour - overshooting in, then
rising and fading over 0.9 s - and the score counter pulses.

Opening waves are one or two fruit with no bombs for the first four waves. On
recorded play the old two-to-five-fruit openings ended rounds in about three
seconds.

**Slicing tests the swept path, not the current position.** Poses arrive at
~30 Hz, so a hand moving 1500 px/s jumps ~50 px between frames. Asking "is the
blade inside a fruit right now" would miss constantly — and would miss *hardest
on the fastest swings*, which is backwards for a slicing game. Every frame tests
the segment swept since the last sample, which cannot tunnel at any hand speed.
A slice also requires real speed, otherwise resting a hand on a fruit would
quietly dissolve the board.

**Which wrist is your right hand depends on mirroring.** The feed is mirrored by
default, and the model labels limbs by how they *appear* — a horizontally
flipped person looks like someone with swapped anatomy — so on a mirrored feed
your real right hand is annotated `left_wrist`. That resolution lives in one
named function (`blade_keypoint`) with tests pinning both cases, because getting
it backwards makes the blade follow the wrong arm in a way that is very hard to
diagnose from the inside. If it still feels wrong, **press `h` to swap hands
live**.

Original art, not Halfbrick's: the sprites are generated procedurally like the
rest of the project.

```bash
uv run python tools/generate_fruitninja_snacks.py
uv run python tools/generate_fruitninja_scene.py
```

### Flappy Raccoon (Flappy Bird rules)

Flap your arms like wings: raise both arms, then beat them down. Each flap is
one hop. The first flap starts the round. A camera window in the corner shows
your arms, a wing meter and a FLAP! flash, so you can see each flap register.
Stand back far enough for your **waist** to be in view, or the game prompts
you to: when you're close, your wrists leave the frame on every downstroke,
and flaps get missed.

The physics are made for webcam latency. Gravity is about half the original's,
and difficulty eases in, then keeps rising. Gaps start at 300 px (48% of the
play area), 247 px by score 10 and 216 px by 20, approaching 175 px without
ever flattening out. Scroll speed rises from 200 toward 270 px/s, and the time
between pipes falls from 2.0 toward 1.55 s, on the same kind of curve. Gap
placement limits how far one gap can be from the next. A test flies a bot with
150 ms of input lag through many random courses and requires it to finish every
one, including 120-pipe runs that reach the hardest settings. The arm-flap detector fires once per downstroke, ignores slow arm lowering
and jitter, and works with one arm if only one is visible. The hero is the
Snack Attack raccoon with feathered wings and a ringed tail. Its head is drawn by
the Snack Attack generator, so both games share one character. It only tilts
gently (-15 to +30 degrees), because a front-facing face tipped further just looks
sideways. The fairness proof flies the shipped collision circle, read from the
art manifest. All art is original and procedural:
`uv run python tools/generate_flappy_assets.py`.

### Hole in the Wall (off the menu)

Still in the code: `uv run kp game --game holeinwall`.

### Catch (demo)

Steer the paddle by moving both hands left/right; raise your hands to widen it;
catch the falling blocks. It exists to prove the control path end to end — pose
→ control → simulation → render.

## Assets

Every visual is generated procedurally with numpy and OpenCV — no binary source
art, no extra dependencies. Regenerate (deterministic, byte-identical output):

```bash
uv run python tools/generate_holeinwall_assets.py
```

Written to `assets/holeinwall/generated/`: the studio background, the yellow wall
panel, chrome rim, hole glow, spark, PERFECT!/SPLASH! badges, HUD panel and
countdown ring.

One performance trap worth recording: the hole needs clearance around the pose,
and the obvious way to get it is `cv2.dilate`. Don't. A non-separable ellipse
kernel scales with its own *area* — 22 ms at 65px and **363 ms** at 259px on a
720p frame. Baking the padding into the drawing instead (thicker limb strokes, a
stroked torso outline) is geometrically equivalent within 0.94 IoU and costs
0.13 ms — a 100-1500x difference, and it is why the game renders in single-digit
milliseconds. The hole is also punched with a *binary* mask rather than a
per-pixel alpha blend (~7 ms full-frame) and the hard edge hidden under an
anti-aliased glow rim.

## Architecture

```
camera.py     capture thread, always hands back the newest frame
inference.py  RF-DETR wrapper -> Pose records; optional EMA; extrapolation
pipeline.py   inference worker thread + rate meters
controls.py   Pose -> ControlState (steer/throttle/gestures/raw pose)
overlay.py    skeleton and HUD drawing
app.py        CLI: bench | pose | game
game/
  base.py         Game interface
  demo.py         catch demo
  fruitninja/
    art.py        sprite schema + manifest loading
    blade.py      wrist -> blade trail, swept-segment slicing
    entities.py   physics, halves, splatter, wave spawner
    game.py       classic rules: score, combos, lives, bombs
    render.py     scene compositing
  holeinwall/
    core.py       canonical + torso pose spaces, TargetPose, FitResult
    silhouette.py keypoints -> filled body mask (with free padding)
    poses.py      16 target shapes, built by forward kinematics
    matching.py   joint + IoU fit scoring
    render.py     wall, hole, ghost guide, HUD, particles
    game.py       round state machine and scoring
    assets.py     asset loading + clipped alpha compositing
```

Three design points that make it feel real-time:

**Newest-frame-wins capture.** A background thread drains the camera constantly
and keeps only the latest frame. Any downstream hitch drops frames instead of
accumulating latency.

**Decoupled inference.** The model worker always grabs the newest frame and
publishes results; the render loop draws every camera frame with the most recent
pose. The overlay lags by one inference period (~45 ms) but motion stays fluid.
The HUD's `pose age` shows this staleness in frames.

**Velocity extrapolation.** Each result describes a frame that is already one
model-latency old, so drawing the newest result puts the player 45-90 ms in the
past. `PoseExtrapolator` estimates per-keypoint velocity between results and
projects to render time, cancelling most of that lag. Gain defaults to 0.7
rather than 1.0 because overshoot at a direction reversal reads as a visible
snap-back, which feels worse than slight lag. The projection is clamped
(`max_lead`, 120 ms) so a stalled worker cannot fling the skeleton off-screen,
and velocity is only taken from keypoints confident in *both* frames so a
reappearing limb snaps in instead of rocketing across the frame. The HUD's
`lead` line shows how far poses are being projected.

**Body-relative controls.** Signals are normalized against the player's own
shoulder width, so a gesture means the same thing near or far from the camera.
Boolean gestures use on/off hysteresis so a hand at the threshold doesn't chatter.

## Adding a game

Subclass `Game`, implement `update(controls, dt)` and `render(frame)`, register
it in `game/__init__.py`. Most games should use the derived signals on
`ControlState` (steer, throttle, gestures) and stay ignorant of keypoints. Pose
matching genuinely needs the raw joints, so `ControlState.pose` carries them for
the games that do.

## Tests

```bash
uv run --with pytest pytest tests/ -q   # also runs the browser suite (web/tests) if Node is installed
```

204 tests. The Hole in the Wall suite covers fit scoring (distance and position
invariance, wrong-pose rejection, every authored hole being passable by its own
pose and by no other) and the round state machine (phase transitions, scoring,
lives, difficulty ramp, and the wall never overshooting on a stalled frame). The
pose library is checked for human proportions, consistent bone lengths across all
16 shapes, and pairwise distinctness. The control suite drives mapping and
extrapolation with synthetic skeletons:
deadzone, hysteresis, scale invariance, the lean fallback when hands are hidden,
and the extrapolator's linear projection, gain, lead clamping, stale-gap reset,
and refusal to fling a reappearing keypoint.

## Notes

- `RFDETRKeypointPreview` is a preview model (COCO 17 keypoints, trained at 576px).
- `--half` is exposed but fp16 support on MPS is uneven; fp32 is the default.
- Crowded, far-away subjects detect poorly; a single close subject is the
  intended case here.
