# Raccoon Dance: design spec

A Just Dance-style game. A coach (the arcade's purple raccoon) dances to a
song, and the player copies it **like a mirror**. RF-DETR keypoints judge each
move as PERFECT / SUPER / GOOD / OK / X, the score fills a star bar, and every
song has its own leaderboard.

Songs are data: a folder with an audio file and a `song.json` chart. Moves
come from a shared move library or are defined in the song itself. Adding a
song means adding a folder, by hand or with the importer, which turns a
recording of someone dancing to the track into a chart.

This file is the contract between the modules. Change it here first, then in
code.

**Everything is original.** No commercial songs or choreography: the songs
are synthesised by `tools/generate_dance_song.py`, and the moves are our own.
Matching Just Dance's *mechanics* is fine.

---

## 1. Files

```
assets/dance/
  moves.json                 shared move library (section 4)
  songs/index.json           ["spell_it_out", "neon_nights", ...]: the song list, in menu order (the first is preselected)
  songs/<id>/song.json       metadata + chart (section 5)
  songs/<id>/audio.m4a       AAC, 44.1 kHz stereo; read by both builds
  generated/                 art from tools/generate_dance_assets.py (+ manifest.json)

src/kpapp/game/dance/
  pose.py        pose angles and feet, forward kinematics (leg IK), measuring a player, similarity  (core)
  chart.py       load/validate songs and moves, beat <-> time, judge timeline, lint (core)
  judge.py       timing windows, ratings, score, stars
  motion.py      the drawn coach's performance: eased hits, steps, groove, follow-through, depth (section 4a)
  game.py        DanceGame: phases, audio sync, judging, juice state
  render.py      DanceRenderer: stage, coach, pictograms, feedback, HUD, results
  art.py         asset loader (lazy, cached, "run the generator" message)
  tutorial.py    how-to-play cards
  importer.py    recording -> chart (used by `kp dance-import`)
  preview.py     contact sheets of a chart or the move library (`kp dance-preview`)
src/kpapp/audio.py            open_player() + SilentPlayer (the clock without sound), shared by any game
src/kpapp/audio_device.py     AudioPlayer: the real thing (sounddevice + PyAV decode)
tools/generate_dance_song.py  procedural, deterministic song -> audio.m4a only (song.json is hand-written)
tools/generate_dance_assets.py procedural, deterministic stage/pictogram/FX art
web/js/games/dance/...        the browser port, module for module
web/js/core/audio.js          Web Audio player with the same surface as audio.py
```

## 2. Coordinates and mirroring

Poses live in **screen space as the player sees it**. "Left" and "right" mean
the **viewer's** left and right. The coach faces the player, and the player
copies it like a mirror: when the coach's arm on the screen's right goes up,
the player raises the arm on the screen's right in their own mirrored camera
image. That is their real right arm.

Measuring the player happens in the same screen space, on the mirrored frame
the game sees. On a mirrored feed the model labels limbs as they appear, so the
**screen-right** limbs are COCO `left_*` and the **screen-left** limbs are COCO
`right_*` (the same fact as `hand_keypoint`). With `mirrored=False` the mapping
flips. `pose.py` owns this mapping, and nothing else should re-derive it.

## 3. Pose representation (`pose.py`)

A pose is the arms as angles in degrees in the **torso frame** (as if the
torso were upright, before applying `lean`), and the lower body as a dancer
thinks of it: where each foot is on the floor, where the hips are over the
feet, and how much the knees bend.

```json
{"la": [upper, fore], "ra": [upper, fore],   // arms: screen-left, screen-right
 "lf": [out, up], "rf": [out, up],           // feet: screen-left, screen-right
 "lean": 0, "head": 0, "shift": 0, "squat": 0}
```

- **Limb angle (arms):** 0 = pointing straight down. Positive rotates outward
  and up, away from the body's midline: towards the screen-left for `la`, and
  towards the screen-right for `ra`. 90 = horizontal outward, 180 = straight
  up, above 180 = past vertical over the head towards the other side,
  negative = across the front of the body towards the other side. The
  forearm angle is **absolute in the torso frame**, not relative to the upper
  arm: "forearm straight up" is 180 whatever the upper arm does.
- **Feet:** `[out, up]` in shoulder widths (SW). `out` is how far the ankle is
  from the body's floor centre line, positive outward (like the arm angles,
  so mirroring just swaps `lf` and `rf`); negative crosses past the centre
  line (a cross-behind). `up` is the height off the floor (a knee lift, a
  kick). Rest is `[0.37, 0]`: feet under the hips. Feet are **relative to the
  body**: the coach travels across the stage on its own (section 4a), so a
  step touch right and a step touch left are mirror images of each other.
- **shift:** the hips across, in SW, + = screen-right. The judge reads the
  *weight*: `shift` minus the feet's midpoint (`(rf.out - lf.out) / 2`).
  Over the screen-right foot of a 1.4 SW step, that is about +0.3.
- **squat:** 0..1, the supporting knee's bend: `SQUAT_BEND` (90) degrees at 1.
  The supporting leg is the planted foot nearest under the hips.
- **lean:** torso tilt from vertical. Positive tilts the head towards
  screen-right. **head:** head tilt relative to the torso, the same sign. Cosmetic.
- **Old charts:** `ll` / `rl` (`[thigh, shin]` leg angles, the format before
  feet) still load: `normalize_pose` turns them into feet by forward
  kinematics from the rest hips. A pose can't give both.
- **Ranges:** feet out -1.0..2.2, up 0..1.4; shift -1.5..1.5; squat 0..1.
- **Missing keys default to the rest pose:**
  `la=[10,5]`, `ra=[10,5]`, `lf=[0.37,0]`, `rf=[0.37,0]`, lean/head/shift/squat 0.

**Body proportions** are in shoulder widths (SW = 1). Shoulder centre to hip
centre 1.25, hip width 0.6, upper arm 0.62, forearm 0.58, hand 0.16, thigh
0.85, shin 0.82 (a leg is 1.67), neck 0.2, head radius 0.3.

**`forward(pose) -> joints`** gives a dict of named 2D points in SW units,
with y down and the origin at the *rest* hip centre:
`hip, head, neck, l_shoulder, r_shoulder, l_elbow, r_elbow, l_wrist, r_wrist,
l_hand, r_hand, l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle`.
The ankles stand where the feet say, on the floor (`FLOOR`, 1.6685 SW below
the rest hips) minus `up`. The hip centre (`pelvis`) is at `shift` across and
as low as the squat and the feet need: the supporting leg bends to the squat,
and a foot planted wide pulls the hips down until the leg reaches it (a wide
stance or a lunge sits lower). The knees join hips and ankles by two-bone IK,
bending **outward** (a knee lift comes up and out to the side; a squat is a
plié). Arms hang from the leaning torso; legs stay in the world frame. The
coach renderer and the pictograms draw from this.

**`measure(keypoint_pose, mirrored=True) -> Measured`** reads a player's
`Pose` (COCO keypoints) into the same angles:
- The torso up-axis runs from the hip centre to the shoulder centre when
  both hips are visible (conf >= 0.4). Otherwise it is perpendicular to the
  shoulder line, pointing up. No shoulders means no measurement (None).
- **Per arm:** with the elbow and wrist visible, the upper arm and forearm
  angles. Without the elbow, only the shoulder-to-wrist "reach" direction
  (elbows are missing in about 54% of frames; AGENTS.md section 6). With the
  elbow but not the wrist (a close player's hands above the top of the
  frame), only the upper arm.
- **Lower body** (`Measured.lower`, section 3a): only with both legs whole in
  view (hips, knees and ankles). `Measured.legs` keeps the thigh and shin
  angles too (the importer, the PiP).
- `Measured` also carries the shoulder width in px and which parts were seen.

**`similarity(target_pose, measured) -> (score 0..1, parts)`**:
- **Per arm segment:** the angle difference d in degrees (wrapped to 0..180)
  gives `s = clamp(1 - (d - 20) / 70, 0, 1)`. Up to 20 degrees is perfect;
  90 degrees scores 0.
- **Arm with elbow:** the mean of the upper arm and forearm. **Arm without
  elbow:** the reach direction compared with the target's shoulder-to-wrist
  direction from `forward()`. **Elbow without wrist:** the upper arm alone.
  Ignoring it made a perfect but close player's Clap Up an X: the window only
  ever saw the arms on their way up.
- **Weights:** arms 1.0 each. **Legs 2.0** (as much as both arms: a leg-led
  move is half the score), when the target's lower body is active and the
  player's is in view (section 3a). Lean 0.3, only when hips are visible and
  |target lean| > 8; it scores `clamp(1 - (|d| - 6) / 24, 0, 1)`.
- **Score:** the weighted mean over the parts that count. If no arm is
  measured, the score is None: "can't see you", which is never a miss.

## 3a. Judging the lower body

What a webcam sees of a player 2.5-3 m back, facing it, is the legs'
outline: depth is lost (a knee bent *forward* is foreshortened, a box step's
forward step is invisible). So the lower body is judged on four features that
read from the front, in shoulder widths scaled by the player's longer leg
(their own proportions and a twisted torso don't matter):

| feature | what | rest |
|---|---|---|
| `stance` | screen-right ankle x minus screen-left ankle x (negative = crossed) | 0.74 |
| `weight` | hip centre x minus the ankles' midpoint (+ = over the screen-right foot) | 0 |
| `lift` | screen-left ankle y minus screen-right ankle y (+ = the screen-right foot is higher) | 0 |
| `bend` | each knee's bend in degrees (0 = straight) | 0 |

**One function, both sides.** `read_lower(get)` computes them from joint
positions: from the coach's `forward()` joints for the target (`leg_target`,
cached) and from the player's keypoints in `measure()`. So `shift` and
`squat` mean exactly what the camera reads off a player; round-tripping
forward -> keypoints -> measure reads the coach back within 0.01 SW and 0.5
degrees (`tests/test_dance_pose.py`, `web/tests/dance_pose.test.mjs`).

**What is judged.** A target's lower body counts only when it is *active*,
away from standing: a **step** (stance more than 0.30 from rest, or weight
over 0.15), a **lift** (over 0.15) or a **bend** (a knee over 25 degrees).
An arms move leaves the legs free: a player bouncing to the music is never
marked down. When active, only the active features are compared: stance and
weight together for a step, lift for a lift, each bent knee for a bend (each
knee counts half). Each feature scores `clamp(1 - (|d| - free) / span, 0, 1)`:

| feature | free | span (zero at free + span) |
|---|---|---|
| stance | 0.20 SW | 0.45 |
| weight | 0.10 SW | 0.25 |
| lift | 0.15 SW | 0.30 |
| bend | 20 degrees | 35 |

The free zones absorb keypoint noise (±4 px is ±0.04 SW at 110 px a shoulder
width), the extrapolator's overshoot as a foot lands (lift went 0.19 past the
target in simulation: `LIFT_TOL` and `BEND_TOL` started at 0.10 and 15 and
cost a perfect dancer two SUPERs on HARD) and how wide people naturally step.
A player standing still gets nothing for a step (stance 0.75 off, weight 0.3
off), so perfect arms with still legs score about 0.55 (OK) on a step.

**Feet are optional.** Without both whole legs in view (a crowd at a booth,
or a player standing close, feet below the frame) the legs aren't judged
(`UNSEEN_LEGS` = None) and the arms carry the point: never an X, and the game
doesn't ask for the feet. One leg alone (a lifted foot in view, the other below
the frame) is not judged either. Legs count whenever the camera sees them, so a
player in full view who keeps their legs still scores less than one hiding
them; that is the accepted cost of not requiring feet. (Before, unseen legs
earned half credit, 0.5, and the game said STEP BACK - SHOW YOUR FEET: hidden
feet then scored 8000, between dancing them and standing still.) The tutorial
and the countdown tip ask for the head and both arms in the picture.

**Known limits** (not yet tested with a live player): a squat with the knees
pushed *forward* reads as less bend than one with the knees out over the toes
(the coach shows a plié, knees out); a forward knee lift reads as a lift but
a thinner bend. The tolerances are tuned in simulation only.

## 4. Move library (`assets/dance/moves.json`)

```json
{"version": 2,
 "moves": {
   "step_touch_r": {
     "label": "Step Touch",
     "beats": 2,                       // how long the move occupies the chart
     "keyframes": [                    // beat offsets within the move, ascending
       {"beat": 0, "pose": {"ra": [95, 100], "la": [20, -88],          // step right, right arm out
                            "lf": [0.7, 0], "rf": [0.7, 0], "shift": 0.32, "squat": 0.12}},
       {"beat": 1, "pose": {"ra": [80, -80], "la": [80, -80],          // touch: fists in, left toe taps
                            "lf": [0.22, 0.08], "rf": [0.22, 0], "shift": 0.1, "squat": 0.3}}
     ],                                // a keyframe may carry its own "label" (below)
     "judge": [0],                     // keyframe indexes that are judged (default: all)
     "pictogram": 0                    // the move's thumbnail in kp dance-preview (default: first judged)
   },
   "step_touch_l": {"mirror_of": "step_touch_r"}}}
```

49 moves, 48 of them judge the legs (all but the bow). Version 2 is the feet
format (section 3); version 1 files (`ll`/`rl` leg angles) still load.

- **Keyframe labels:** a keyframe may name itself (`"label": "Y"`, up to 15
  characters). Its judge point, card, the name under the ring and the rating
  popup then say that instead of the move's label. Spell It Out's `letters` is
  one 8-beat move whose four judged keyframes are labelled Y, M, C and A: one
  move keeps the four letters together (EASY keeps slow moves whole, so it
  spells them too, section 7a), and each card still says its letter. Mirrored
  twins and EASY's held moves keep a keyframe's label.

- **Coach animation** eases between keyframes (smoothstep in beat time).
  It reaches each keyframe exactly on its beat and holds it until the next
  keyframe, easing into that one over its last 0.75 beat (`EASE_BEATS`, or
  the whole gap if shorter). Before the first move it grooves (`idle_groove`)
  and eases into the first keyframe the same way.
- **Pictograms:** one card per *judge point* (`Song.pictograms()`), showing
  the pose judged on that beat under its move's label and reaching the target
  ring on that beat. A two-pose move shows two cards, so card, label and judge
  always agree; cards can come every beat. (Earlier, one card per move left
  60 of Neon Nights' 145 points without a card, and "Point Up" was labelled
  over a horizontal point.) `pictogram` now only picks the move's thumbnail
  in `kp dance-preview`. A one-character label (the letters) is also drawn
  on the card, as a hot-pink badge (gold on gold cards) in its bottom-right
  corner, the one the letters' arms and legs leave free: the letter reads from
  2 m before the card reaches the ring (`render.glyph_label`, `draw_glyph_badge`).
- **Mirrored twins:** a move named `x_l` / `x_r` can be generated from the
  other side (`chart.mirror_pose`: swap `la`/`ra` and `lf`/`rf`, negate
  `lean`, `head` and `shift`). Prefer authoring one side and declaring
  `"mirror_of": "x_r"`.
- **Leg cards:** a pose whose lower body the judge reads (section 3a) is drawn
  with its legs in the arm colour on a floor line, with an arrow on the floor
  towards the foot the weight goes over (a step, a sway) and a chevron over a
  lifted foot (a knee lift, a kick). `kp dance-preview` draws judged legs
  bright on a floor line with the weight arrow.

## 4a. The coach's performance (`motion.py`) and its 3D look

`Song.coach_pose(beat)` above is the *target* timeline: bots, tests and the
parity fixture copy it, and it never changes for looks. The drawn coach
dances `motion.coach_motion(song, beat)` on top of it, a pure function of the
song's keyframes and the beat (no clock, no state):

- **Hits:** each transition keeps the chart's timing (it starts `EASE_BEATS`
  before the keyframe) but winds up a little the other way, arrives early with
  a small overshoot (10% of the move, at most 8 degrees) and settles onto the
  keyframe exactly on its beat. The upper arm leads, the forearm and legs
  follow, so limbs swing in soft arcs.
- **Groove**, scaled by section (chorus 1.0 ... outro 0.5): a dip on every beat
  (knees bend, the torso squashes; it stretches on the off-beat), weight
  shifting from foot to foot, alternate shoulder rolls, a 2-degree lean sway
  that is zero on every beat, a torso twist, a head bob that lags the body.
- **Follow-through** from lagged samples of the same timeline: the head tilt
  trails the torso, the tail swings against the body, the gloves trail their
  forearms (up to 30 degrees), and a jump hops and lands before its beat.
- **Steps** (`travel`): feet are relative to the body, so the coach travels.
  For each transition the *supporting* foot is the one under the hips at its
  start (stepping right, the left foot stays; the weight follows the step);
  it stays planted on the stage while the body moves over it, and the free
  foot lifts in an arc (up to 0.16 SW, `STEP_LIFT`) on its way. A **jump** is
  both feet spreading or closing while the weight stays centred (a star
  jump): the stance centre stays put and the coach hops. Every transition
  also drifts the coach back towards its spot (15% of the way, at most 0.06
  SW, `RECENTRE`), so R-then-L phrases that start from different places don't
  walk it off the stage; it stays within 1.3 SW of its spot on both songs
  (`TRAVEL_MAX` 1.8 clamps; the lint warns past 1.5).
- **Depth:** the chest turns towards an arm reaching across it (up to ~24
  degrees); the head picks the nearest pre-rendered turn
  (`coach_head_turns.png`, -20..20 degrees); the far arm of a big twist goes
  behind the jacket, an arm crossing the body comes in front, nearer limbs are
  drawn thicker and a forearm pointing at the viewer shorter with a bigger glove.

**The guarantee:** on every judged beat a camera looking at the drawn coach
reads the chart's pose, however far it has stepped. Arm angles in the torso
frame are exact (everything else moves whole parts: shoulders, head, squash,
twist); the dip bends the knees a few degrees on the beat (`DIP_SQUAT`, 4
degrees at full energy) and the groove's sway moves the hips 0.035 SW.
`tests/test_dance_motion.py` measures the painter's own joints with the
judge's `measure` on every judge point of every song and difficulty: arms
and lean within 6 degrees (worst today 0.0), stance, weight and lift within
0.08 SW (worst 0.035, the sway), knee bends within 8 degrees (worst 3.6).

**Painting:** the stage coach is lit from the upper left. Each part is filled
in its shadow colour (the base darkened, plus a little of the outfit's glow),
then in its own colour shrunk and moved towards the light, leaving a
core-shadow crescent on the lower right; limbs get a highlight streak, the
jacket a sheen, and the jacket's hem and the head shade what is under them.
The neon rim sits a little to the lower right (a back light), the belt sags
round the waist, and a soft contact shadow under the feet shrinks during a
hop. The lower foot is on the floor unless the coach hops. The head turns and their shading are
baked by the generator. The pictogram cards and the tutorial's small coaches
stay flat (`CoachPainter(shade=False)`): they are instructions.

`web/js/games/dance/motion.js` is a port; `web/tests/fixtures/dance_motion.json`
(`tools/make_dance_motion_fixture.py`) pins the motion and the painter's
layout to the Python's, and `tests/test_dance_motion.py` checks the fixture is
current.

## 5. Song chart (`songs/<id>/song.json`)

```json
{"version": 1,
 "id": "neon_nights", "title": "Neon Nights", "artist": "KP Arcade",
 "bpm": 120, "offset": 0.0,            // seconds into the audio of beat 0
 "duration": 106.0, "audio": "audio.m4a",
 "difficulty": 1,                      // 1-3, shown on the song card
 "coach": {"outfit": "neon"},          // the renderer's palette key
 "preview": [32, 48],                  // beats looped on the song-select card (optional)
 "sections": [{"name": "intro", "beat": 0}, {"name": "verse", "beat": 16}, ...],
 "moves": [{"move": "disco_point_r", "beat": 16}, {"move": "clap_up", "beat": 18, "gold": true}, ...],
 "custom_moves": {},                   // same schema as the library; overrides by name
 "charts": {"easy": [...], "hard": [...]}   // optional hand-made charts per difficulty (section 7a)
}
```

`moves` is the NORMAL chart. `charts` is optional: a list under `easy` or
`hard`, in the same format as `moves`, replaces that difficulty's chart
(otherwise EASY is derived from `moves` and HARD dances `moves`). Only `easy`
and `hard` are allowed there.

- **Validation:** `chart.load_song` rejects overlapping moves, unknown move
  names, keyframes outside a move, non-ascending beats, a chart running past
  the duration, and bad pose keys, with a clear message naming the song,
  move and beat. Hand-made `charts` are validated the same way, whichever
  difficulty is being loaded. `tests/test_dance_chart.py` validates every song
  in `assets/dance/songs/`, on every difficulty (including the freeze rule
  of section 7a).
- **Gold moves:** a judged point counts double, flashes gold, and earns a
  gold burst when rated SUPER or better. Use one or two per chorus.

## 5a. Choreography rules and the lint

The first charts were mostly arm poses picked one after another: a new
direction every beat (hands travelling a median 2 shoulder widths a beat,
whiplash from chest to overhead), legs straight, and phrases that never came
back. The rules below come from how dance-fitness classes (aerobics, Zumba),
line dances and party dances are built, and from what a webcam can judge.
They are generic patterns, not anyone's routine.

1. **8-counts.** Choreography is built in 8-count phrases from each
   section's start; four make a 32-count block (verse, chorus). No move
   straddles an 8-count boundary.
2. **Repetition.** Every 8-count comes back: danced again elsewhere, or
   mirrored (right then left: the chorus dances its 16 counts, then the same
   16 mirrored), or its first 4 counts repeat (step touch R, L, R, L). Verses
   share their last two phrases (hip sways, then the same pre-chorus), so a
   player learns them once. Line dances and class routines work the same way:
   a short block, repeated and mirrored.
3. **Legs set the rhythm, arms decorate.** The base steps carry the beat: step
   touch, grapevine (step, cross behind, step, touch), hip sway, knee lift,
   side kick, toe tap out, side lunge, plié squat and rise, jumps. The arms go
   *with* the step: the arm on the stepping side reaches or points that way,
   hands meet or pull in on the touch, a knee lift goes with the opposite arm
   up (as in marching), a kick with a punch to the same side. Every section
   works the legs (at least a quarter of its judged poses; in practice 95%+).
4. **Weight transfers are visible.** On a step the hips go over the stepping
   foot (`shift`); on a touch the feet come together with the weight still on
   the standing foot and the free toe tapping. One foot moves at a time (at
   most two jumps an 8-count).
5. **Flow.** The end of one move is near the start of the next. Hands travel
   at most 2.35 SW a beat (relative to the shoulder; a full overhead-to-hip
   swing in one beat is whiplash) and not more than 1.9 SW three transitions
   running. Feet move at most 1.2 SW a beat, the hips 0.9, the squat 0.7.
   Big arm changes alternate both arms between "out" and "up" (T, arms up,
   clap overhead, heart overhead), the moves that flow; "low" arm shapes
   (hands on hips, at the chest) look like standing to a camera that loses
   the elbows, so they are never the whole of a judged pose.
6. **One new idea per phrase**, difficulty from the tempo of changes, not
   from randomness: verses judge a pose every 2 beats (0.5 a beat, verse 2
   0.625), choruses most beats (0.81), the bridge every 2 beats. Bigger moves
   (grapevines, jumps, lunges) and gold go on the chorus.
7. **The camera must be able to judge it.** A leg move must move the legs past
   the judge's activity margins (section 3a), feet stay within 1.6 SW out and
   1.0 SW up, and depth-only moves (a box step's forward step) are expressed
   by what shows from the front.
8. **The freeze rule** (section 7a): consecutive judged poses differ enough
   that holding the previous one scores below 0.55 on the next, with the whole
   body seen, with the feet out of frame (arms only: a close player) and
   without elbows. With the feet hidden the arms must carry the difference,
   so every judged beat changes the arms too.

**`chart.lint(song, phrases=True)`** (`chart.js`: `lint(song, keyframeTravel,
{phrases})`) checks them and returns `LintIssue(kind, beat, message)`s:

| kind | flags |
|---|---|
| `flow` | a hand over 2.35 SW a beat, a foot over 1.2, the hips over 0.9, the squat over 0.7 |
| `whiplash` | three transitions in a row with a hand over 1.9 SW a beat |
| `jumps` | more than two jumps in an 8-count |
| `phrase` | a move across an 8-count boundary; an 8-count that never repeats |
| `legs` | a section with fewer than 25% of its judged poses working the legs |
| `unjudgeable` | a judged pose whose legs move but too little for the camera; a foot out of range |
| `freeze` / `rest` | the freeze rule; a pose that standing still with arms down scores on |
| `wander` | the coach travelling more than 1.5 SW from its spot |

Every song passes on every difficulty (EASY without the phrase checks: it
keeps one pose a bar); the charts before this had 110 (Neon Nights) and 63
(Pumpkin Patch Bounce) issues, mostly flow and whiplash. Hand travel now:
median 1.52 SW a beat on Neon Nights (was 2.03), 1.70 on Pumpkin Patch Bounce
(was 1.97), 1.52 on Spell It Out (written against the lint from the start),
none over 2.3. `kp dance-preview` and `kp dance-import` print it;
`tests/test_dance_chart.py` and `web/tests/dance_chart.test.mjs` run it on
every shipped song.

## 6. The test song: "Neon Nights"

- **Tempo:** 120 BPM, 4/4, so one beat is 0.5 s and one bar is 2 s.
- **Length:** 52 bars = 208 beats = 104 s of music plus 2 s of ring-out;
  `duration` in song.json is 106.0, the audio file's own length.
- **Key and style:** A minor, upbeat synth-pop or electro.

An original routine with an 80s-workout flavour (it's synth-pop): base steps
from aerobics with the arms layered on top. 123 judge points (EASY 49).

| beats | section | phrases (8-counts) | judged |
|---|---|---|---|
| 0-15 | intro (4 bars) | none; the coach grooves, "GET READY" | |
| 16-47 | verse 1 | Step Touch R/L x2 (arm opens to the step, fists in on the touch) / Step Press R/L x2 (one fist presses up) / Hip Sway R/L x2 (feet planted, hips over a foot, arm up) / pre-chorus: Step Touch R, L, Squat & Rise | a pose every 2 beats |
| 48-79 | chorus 1 | Grapevine R (step, cross behind, step, touch: open, up, open, clap up) / Grapevine L / Disco Point R, L (step and point up), Knee Pump R (arms up, then the knee with arms out), Star Jump (feet out, clap overhead); then the same 16 counts mirrored. 1 gold: the last Star Jump | most beats (0.81) |
| 80-111 | verse 2 | Knee Up R/L x2 (knee with the opposite arm up) / Side Kick R/L x2 (kick and punch, land in a plié, hands overhead) / Hip Sway R/L x2 / the pre-chorus again | 0.625 a beat |
| 112-143 | chorus 2 | as chorus 1, 1 gold | |
| 144-159 | bridge | Rainbow R, L (a side lunge, arms arc over to the top) / Wide Squat R, L (plié with arms out, rise into a hip sway, hands overhead): 4-beat moves | every 2 beats |
| 160-191 | final chorus | as chorus 1, 2 gold (both Star Jumps) | |
| 192-207 | outro | Superstar at 192 (point up, toe tapped out), then a bow | |

Beat 0 is at `offset` seconds into the audio. The generator must put the first downbeat exactly there, and the
chart is authored against these beats.

## 6b. The second song: "Pumpkin Patch Bounce"

116 BPM, 4/4, offset 0.0, 93 s (176 beats of music plus a tail), coach outfit
`pumpkin`. An original routine in the *style* of short-video dances: a
16-beat hook that keeps coming back, upper-body moves that read close to the
camera, and some October flavour. No creator's routine is reproduced.

| beats | section | moves |
|---|---|---|
| 0-15 | intro | none (count-in on 12-15) |
| 16-47 | hook 1 | the hook, twice; gold on the second Woah (32) |
| 48-79 | verse | Step Clap R/L, Hair Flip R/L, Howl R/L, Scarecrow, Hands Up; twice |
| 80-111 | hook 2 | the hook, twice; gold on 96 |
| 112-127 | break | half time, 4-beat moves: Ghost Arms R/L (side lunges, floaty arms), Spooky Claws R/L (lean, then a deep creeping squat) |
| 128-159 | final hook | the hook, twice; gold on both Woahs (128, 144) |
| 160-175 | outro | Boo! on 160 (a wide squat, judged), then the bow |

**The hook** is two 8-counts, danced A B A B in every hook, with a bounce in
the knees throughout:
A = Woah (knees bend, hips and fists snap right, then left) / Frame (hands
frame the face, knees bent together) / Point Over R (step right, point up and
over; touch, hands pull in) / Point Over L;
B = Heart Hands (knees bent) / Brush Off R (brush the shoulder, toe taps out) /
Brush Off L / Cheer (knee up, arms up).
**The verse:** Step Clap (step, clap overhead on the touch), Hair Flip (hip
bump, then knees together), Howl (knee up, arm to the moon), Scarecrow (plié,
dangling forearms), Hands Up (jump out, arms up).
`tests/test_dance_chart.py` pins the structure, that the hook's poses repeat
identically, and that standing still scores on no judged pose.
Judge points: EASY 40, NORMAL 99, HARD 99.

## 6c. The letters song: "Spell It Out" (first on the song list)

127 BPM, 4/4, B-flat major, offset 0.0, 100.5 s (208 beats = 98.27 s of
music plus a ring-out), coach outfit `disco` (a gold jacket, white flares,
hot-pink cuffs, belt and stripes). It is the first id in `songs/index.json`,
so it heads the song list and is preselected in both builds (the landing
page's hero still dances Neon Nights, by id).

The dance everyone knows: spelling four letters with the arms on every
chorus. The letter gesture is a generic crowd move; the song, the routine
around it and every pose are our own, and no names from elsewhere appear
(song, ids, files: the moves are just "Y", "M", "C", "A"). The music borrows
only the cadence of late-70s party disco: four-on-the-floor at 127, an
octave-bouncing bass, claps on 2 and 4, an open hat on every off-beat, string
swoops and brass stabs. The chorus leaves the crowd room to spell: four tutti
brass-and-string stabs two beats apart, each a step higher than the last
(B-flat, C, D, F), with only the beat and the claps under them, and the
letters land on them. Melody, harmony (verses I-iii-IV-V, I-iii-ii-V;
choruses IV-V-iii-vi, IV-V-ii-V), bass line and arrangement are original, and
it is instrumental like the others (`tools/dance_songs/spell_it_out.py`).

| beats | section | moves | judged |
|---|---|---|---|
| 0-15 | intro | none; snare count-in on 12, 13, 14 under a riser and a string swoop | |
| 16-47 | verse 1 | Step Touch R/L x2 / Step Clap R/L x2 (a clap overhead on the touch) / Disco Point R/L x2 / Hip Sway R, L, Step Touch R, L | 0.625 a beat |
| 48-79 | chorus 1 | **Letters** (Y M C A on 48, 50, 52, 54) / Grapevine R, L (open, clap overhead, open, touch and point up) / **Letters** (gold, 64) / Grapevine R, L | 0.75 |
| 80-111 | verse 2 | Side Kick R/L x2 / Knee Up L/R x2 (marching, the opposite arm up) / the verse's last two 8-counts again | 0.625 |
| 112-143 | chorus 2 | as chorus 1, gold on 128 | |
| 144-159 | bridge | the kick drops: Wave x4 (arms overhead swaying right, left, with the hips) / Step Touch R/L x2 | 0.75 |
| 160-191 | final chorus | as chorus 1, gold on both sets (160, 176) | |
| 192-207 | outro | Letters on 192 over the last four stabs, the final hit on 200, the bow on 202 | |

**The letters** are one 8-beat move (`letters`), judged on beats 0, 2, 4, 6
and labelled per keyframe (section 4), with a bounce on each off-beat between.
They follow **reference poses a player struck in front of the camera**
(`tools/capture_poses.py`: a countdown per pose, then a second of keypoints
read by the judge's own `measure()`). The captures' arm angles became the
targets, nudged at most ~20 degrees (inside the judge's no-penalty band, so
the reference poses still score PERFECT) where the coach's big head and short
arms needed it to read, or where consecutive letters had to be told apart:

| letter | reference (upper / forearm) | chart | the shape |
|---|---|---|---|
| Y | 135 / 145 | 135 / 145 | arms up in a wide V, feet apart |
| M | 137 / 237 | 150 / 250 | hands on top of the head, elbows up and out, a step right |
| C | top 149 / 216, bottom 87 / 151, lean 44 | top 150 / 232, bottom 70 / 135, lean 44 | the body leans (head to the screen's right) and both arms reach right: the screen-left arm up and over the head, the screen-right arm down and out; the leaning body is the C's back |
| A | 154 / 215 | 172 / 230 | hands meeting on top of the head, elbows bent out, feet wide |

The C opens to the screen's right: the coach shows a C, and a player copying
like a mirror makes the same C in their own mirrored camera image; a reversed
C is a miss (`test_the_c_reads_the_right_way_round_on_screen`). The
reference first C (the bottom arm curled across the chest) didn't read on the
coach; capture the poses again to revise the letters.

Consecutive letters (Y-M, M-C, C-A) score below the freeze line (0.55) when
the previous one is held, seen whole, with the feet out of frame and without
elbows (`test_consecutive_letters_are_told_apart`). Letters that never
follow each other needn't be told apart, and in the reference poses two
pairs aren't: **M and A** are nearly the same arms (hands on the head, hands
together just above it) and **Y and C** share an upper arm. The chart never
puts them back to back, which the same test checks on every difficulty. Each
letter card carries its letter as a badge (section 4).

Judge points: EASY 58, NORMAL 128, HARD 128. EASY keeps every letter (the
`letters` move is slow: a bar or longer, judged poses 2 beats apart) and
holds one pose a bar elsewhere. Results (section 7's input path, whole body
in view; `tests/test_dance_game.py`):

| player | EASY | NORMAL | HARD |
|---|---|---|---|
| perfect | 10000 | 10000 | 10000 |
| perfect, legs out of frame | 10000, 0 X | 10000, 0 X | 10000, 0 X |
| perfect, 250 ms behind the coach | 10000 | 10000 | 9972 |
| precise (10 degrees, 0.06 SW, 200 ms late) | 10000 | 10000 | 9764 |
| 10-20 degrees off, 200 ms late, 15 Hz / 200 ms | 9703 | 9347 | 8472 (4 stars) |
| sloppy | 8203 (4 stars) | 7997 (3 stars) | 5639 (2 stars) |
| perfect arms, legs standing still | 5973 (2 stars) | 5830 (2 stars) | 4309 (2 stars) |
| frozen on the first pose | 3115 (1 star) | 3170 (1 star) | 1927 |
| idle, arms down | 0 | 250 | 83 |
| random arms | 1534 | 1566 | 667 |
| wrong side (mirror image) | 6892 | 5444 | 4410 |

The sloppy dancer, 250 ms late, scores a little less than on the slower songs:
at 127 BPM a beat is 0.47 s, and lateness costs more of each pose's window.

## 7. Judging (`judge.py`)

The numbers in this section are NORMAL's. EASY and HARD change the window,
the hold and the rating thresholds, and EASY judges a derived chart
(section 7a).

- **Judge points:** each judged keyframe becomes a judge point at
  `t = time_of(move.beat + keyframe.beat)`, with weight 2 if gold, else 1.
- **Window:** for each judge point, the song time range `[t - 0.20, t + 0.50]`.
  It is skewed late because a pose reaches the game about 100-150 ms after the
  player makes it (AGENTS.md section 2), and players copying a coach run late
  on top of that. The point is resolved when the window closes.
- **Hold:** a match counts only once it has been held for `HOLD` = 0.12 s.
  The window's score is the best, over the window, of the *worst* similarity
  in the last 0.12 s. Without it, a limb sweeping through the target on its
  way elsewhere scores a momentary PERFECT. Frames where the player can't be
  measured neither break nor extend a hold. A window seen too briefly to hold
  anything uses its worst sample.
- **Rating:** from that score. PERFECT >= 0.90, SUPER >= 0.74,
  GOOD >= 0.60, OK >= 0.42, else X. Values are 1.0 / 0.8 / 0.55 / 0.3 / 0.
- **Not seen:** if the player was never measurable during the window, the
  point is "unseen". It shows no rating, and the screen says what to fix
  ("STEP BACK - SHOW YOUR ARMS"). It is never a miss, but earns no points.
  It doesn't break the combo either. A stall that skips a whole window is
  unseen too. So is a window that loses the player before `LOST_BEFORE`
  (0.25 s) past its point's time and never sees them again, when what it saw
  would rate X: it only saw them on their way into the pose.
- **Pause:** on resume, windows that were open forget what they saw before
  the pause (no free PERFECT for a pose held while pausing). Points whose
  window is open at the resume or opens within `RESUME_GRACE` = 1.0 s of it
  are excused: a rating that would be X is unseen instead, because the
  player's hand was just on the RESUME button.
- **Score:** `round(10000 * sum(value * weight) / sum(weight))` over all
  judge points in the song. The maximum is always 10000.
- **Stars:** 1-5 at 2000 / 4000 / 6000 / 8000 / 9200. "SUPERSTAR" at 9600.
  The star bar (`game.stars`) fills each star's segment linearly between
  thresholds.
- **Combo:** consecutive SUPER-or-better ratings.

**Tuned in simulation** (`tests/test_dance_game.py`). Every simulated player
goes through the app's input path: `keypoints_for` with ±4 px noise, model
results at 20 Hz arriving 150 ms late, and `PoseExtrapolator`. Changes from
the starting numbers (window `[t - 0.20, t + 0.40]`, PERFECT 0.86):

| change | why |
|---|---|
| `HOLD` 0.12 s added | Judged on its best single frame, a dancer holding every limb 25-30 degrees off and 250 ms late scored 9870 (5 stars), because its limbs swept through the targets between poses. |
| window end 0.40 -> 0.50 | A flawless dancer just 250 ms behind the coach took 15 X on Neon Nights (7138). The late edge minus the lag minus the hold left it under 0.2 s. At 0.50 it is all PERFECT. The frozen, idle, random and wrong-side players gained at most a few hundred points. |
| PERFECT 0.86 -> 0.90 | A segment scores `1 - (d - 20) / 70`, so 0.86 meant "within 30 degrees", and the 25-30 degree dancer took PERFECT on 38% of moves. At 0.90 (about 27 degrees) it is 88% SUPER. 0.92 was too far: a dancer 10-20 degrees off and 200 ms late on a 15 Hz model dropped to 4 stars. |

Results with these numbers on NORMAL (Spell It Out's are in section 6c;
`tests/test_dance_game.py`; players
2.5-3 m back with the whole body in view, `FULL_BODY`, unless stated; the
sloppy players are 25-30 degrees off on every arm segment, 0.12-0.2 SW off on
every foot, the hips and squat sloppy too, 250 ms late):

| player | test chart (42, arms only) | Neon Nights (123) | Pumpkin Patch Bounce (99) |
|---|---|---|---|
| perfect, 20 Hz, 150 ms lag | 10000 | 10000 | 10000 |
| perfect, legs out of frame (the arms carry it) | 10000 | 10000, 0 X | 10000, 0 X |
| perfect, half the elbows missing, no hips | 10000 | 10000 | 10000 |
| perfect, 250 ms behind the coach | 10000 | 10000 | 10000 |
| 10-20 degrees off (feet 0.05-0.1), 200 ms late, 15 Hz / 200 ms | 9435 (close framing) | 9559 (5 stars) | 9402 (5 stars) |
| sloppy | 8217 (4 stars) | 8177 (4 stars) | 8201 (4 stars) |
| **perfect arms, legs standing still** | - | **6075 (3 stars)** | **5944 (2 stars)** |
| frozen on the first pose | 1 star or less | 2390 (1 star) | 2780 (1 star) |
| idle, arms down | | 0 | 1150 |
| random arms | | 1689 | 1944 |
| wrong side (mirror image) | 1 star or less | 5811 (2 stars) | 4519 (2 stars) |

The wrong-side player is caught only on asymmetric poses; plenty of the
routines' poses (grapevine arms, star jumps, squats) are symmetric, where the
mirror image *is* the move. The legs weigh half of a leg move, so dancing
from the waist up loses a star or two: on a step it is an OK at best.

**Through the whole app path** (QA before the legs were judged, on the old,
arms-only Neon Nights chart: the real Shell, `ControlMapper`,
`PoseExtrapolator`, ±4 px noise; kept for the history of the arm tuning):

| player | score | stars |
|---|---|---|
| perfect, 20 Hz / 150 ms or 12 Hz / 220 ms (battery) | 10000 | 5 |
| perfect, standing close (hips, elbows or raised wrists out of frame) | 10000 | 5 |
| 10-20 degrees off, 200 ms late, 15 Hz / 200 ms | 9554 | 5 |
| 25-30 degrees off, 250 ms late | 7681 | 3 |
| perfect, but out of the frame for 10 s | 8846 (17 unseen, 0 X) | 4 |
| arms at 60% of every angle | 5977 | 2 |
| only one arm right, the other hanging | 5235 / 5188 | 2 |
| wrong side | 5124 | 2 |
| arms straight up / T-pose all song | 2879 / 1977 | 1 / 0 |
| frozen, idle, random | 1416, 906, 1950 | 0 |

## 7a. Difficulty: EASY / NORMAL / HARD

The shell offers a difficulty picker for games that declare
`difficulties` (section 9). Every song plays on all three:

| | chart | judging (`judge.DIFFICULTY`) |
|---|---|---|
| **EASY** "fewer moves, held longer" | derived from the normal chart: about one judged pose per bar, each held (Neon Nights: 49 points, Pumpkin Patch Bounce 40) | window [-0.25, +0.60], hold 0.12 s, PERFECT 0.86 / SUPER 0.72 / GOOD 0.56 / OK 0.40 |
| **NORMAL** "the full routine" | `moves` (Neon Nights: 123 points, Pumpkin Patch Bounce 99) | window [-0.20, +0.50], hold 0.12 s, 0.90 / 0.74 / 0.60 / 0.42 (section 7, unchanged) |
| **HARD** "every move, judged strictly" | `moves` | window [-0.15, +0.50], hold 0.15 s, 0.93 / 0.84 / 0.68 / 0.50 |

Rating values (1.0 / 0.8 / 0.55 / 0.3 / 0), gold weights, star thresholds and
the 0..10000 score are the same on every difficulty. All judging numbers live
in one table, `judge.DIFFICULTY` (`Rules(early, late, hold, ratings)`); the
old module constants (`WINDOW_LATE`, `HOLD`, `RATINGS`...) are NORMAL's.

**Deriving EASY** (`chart.derive_easy`, `Song.for_difficulty("easy")`), so
every song gets it for free:

- **Short moves** (under a bar, `BAR_BEATS` = 4) are grouped by the bar they
  start in. Each group becomes one *held* move (`<name>~held`): one keyframe,
  the kept pose, from its beat to the group's end. The coach holds it and
  eases into the next kept pose (`EASE_BEATS`), so it does no in-between
  poses the judge doesn't ask for.
- **Which pose:** a gold one first, else the earliest. A candidate is skipped
  if it is under `EASY_MIN_GAP` (2) beats after the previous kept pose, or if
  a player frozen in the previous kept pose would score on it
  (`chart.freeze_score >= FREEZE_MAX`, 0.55: whole body, feet out of frame,
  elbows hidden). A
  group whose candidates all fail adds nothing; the coach holds on.
- **Gold:** a group that had a gold point keeps a gold point (the kept pose
  turns gold if the gold one had to be skipped), so every section with gold
  keeps gold.
- **Slow moves** (a bar or longer with their judged poses at least 2 beats
  apart: the bridge's 4-beat moves, Squat & Rise, the final pose) and
  **unjudged moves** (the bow) stay as they are, judges included: they are
  slow already, and judging only one pose of them left the coach visibly
  hitting poses that no longer counted. The same gap and freeze checks apply.
  A bar-long move that judges every beat (the Grapevine) is grouped like the
  short moves: one held pose for its bar.
- On Neon Nights: verse 1 goes from 16 poses to 9 (one a bar, the pre-chorus
  Squat & Rise in full), each chorus from 26 to 8, the bridge keeps its 8,
  plus the final pose: 49 points. On Spell It Out every set of letters stays
  whole (28 points): 58 of 128. The short moves become one pose a bar (at most
  0.42 of their normal points), and EASY has at most half the points overall. The last one is at 96 s, well past the
  leaderboard's 400-points-a-second cap (`leaderboard/src/rules.js`: a dance
  run must end its last judged move at least 25 s in; EASY's last points are
  at 96 s and 82.8 s).

**The freeze rule on every difficulty:** consecutive judged poses must differ
enough that holding the previous one scores below 0.55 on the next
(`tests/test_dance_chart.py::test_a_frozen_player_cannot_score`, run for
every song on every difficulty).

**Hand-made charts** (`"charts": {"easy": [...]}` in song.json, section 5)
replace the derivation. Give EASY at least two beats between judged poses;
the tests check the freeze rule and perfect-match rules on them too.

**Leaderboards:** NORMAL keeps each song's board, `dance-<song id>`, so
scores from before difficulties stay put. EASY and HARD get
`dance-<song id>_easy` / `dance-<song id>_hard` (`game.board_id_for`),
inside the server's `/^dance-[a-z0-9_]{1,40}$/`.

**Tuned in simulation** on both songs, through the same input path as
section 7 (whole body in view, keypoints with ±4 px noise, 20 Hz, 150 ms
lag, `PoseExtrapolator`;
`tests/test_dance_game.py::test_fairness_on_every_song_and_difficulty`).
Neon Nights / Pumpkin Patch Bounce:

| player | EASY (49 / 40) | NORMAL (123 / 99) | HARD (123 / 99) |
|---|---|---|---|
| perfect | 10000 / 10000 | 10000 / 10000 | 10000 / 10000 |
| perfect, legs out of frame | 10000 / 10000 | 10000 / 10000 | 10000 / 10000 |
| perfect, 250 ms behind the coach | 10000 / 10000 | 10000 / 10000 | 9937 / 9925 |
| precise: within 10 degrees and 0.06 SW, 200 ms late | 10000 / 10000 | 10000 / 9981 | 9921 / 9944 |
| 10-20 degrees off, 200 ms late, 15 Hz / 200 ms | 9774 / 9955 | 9559 / 9402 | 8531 / 8626 (4 stars) |
| sloppy (arms 25-30 degrees, feet 0.12-0.2 SW, 250 ms late) | 8604 / 8489 (4 stars) | 8177 / 8201 (4 stars) | 6126 / 6014 (3 stars) |
| perfect arms, legs standing still | 6311 / 6273 (3 stars) | 6075 / 5944 (3 / 2) | 4614 / 4589 (2 stars) |
| frozen on the first pose | 2840 / 3602 (1 star) | 2390 / 2780 (1 star) | 1461 / 2112 |
| idle, arms down | 0 / 955 | 0 / 1150 | 0 / 477 |
| random arms | 1642 / 2080 | 1689 / 1944 | 630 / 953 |
| wrong side (mirror image) | 6764 / 4830 | 5811 / 4519 | 4839 / 3463 |

The sloppy dancer gained a star on NORMAL and HARD when the legs started
counting: its feet, 0.12-0.2 SW off, are mostly inside the legs' free zones.
It still sits well below a precise dancer, and perfect arms with still legs
sit 2000 points below it.

Why these numbers:

- **EASY's thresholds and late edge:** a beginner 30-40 degrees off, feet
  0.15-0.25 SW off and 400 ms behind the coach scores 7151 (3 stars) on EASY;
  the normal rules on the same chart gave it 4660 (2 stars).
  EASY points are at least 2 beats (1 s) apart, so the wider window never
  reaches into the next point's pose. EASY is not free: frozen, idle and random
  players stay at a star or less. (Without the freeze rule the frozen player
  made 2415.)
- **HARD's hold 0.15 s** is the main lever: at 0.12 the sloppy dancer scores
  6988 on HARD instead of 6126. The thresholds (PERFECT "every limb within ~25
  degrees") put that dancer in SUPER and GOOD. (Before the legs counted it
  made 5150-5380.)
- **HARD keeps the late edge at 0.50.** At the spec's first 0.40, separating
  precise from sloppy came down to 50 ms of lateness: the precise dancer fell
  to 3 stars (6020) and a flawless one 250 ms late to 1 star. HARD judges the
  pose, not the camera's lag. On a slow model (15 Hz, 200 ms) it does cost a
  star or two.

## 8. Game flow (`game.py`)

The shell starts the game with a chosen song (section 9). The game's phases,
`phase.value`, are:

1. **`intro`:** song title card, audio starts, the coach grooves. "GET
   READY" shows 8 beats before the first move, then "3, 2, 1, GO" on the 4
   beats before it (beats 12-15 of Neon Nights, on its count-in snares).
2. **`dancing`:** moves are judged; the pictogram lane scrolls.
3. **`outro`:** the last judge point has resolved. The coach bows and the
   audio plays out. If the chart ends with unjudged moves (Neon Nights:
   finale, then a `bow` move), the renderer's bow waits for the first of
   them, so the last judged pose is held and the coach bows once.
4. **`game_over`:** after the audio ends (or 3 s after the end of the last
   chart move, `OUTRO_HOLD`). The
   results screen shows the score, stars and rating counts. The shell
   overlays its leaderboard and buttons as for other games.

- **Audio sync:** song time = the `AudioPlayer.position()` minus the song
  offset. The game never advances song time from `dt` while audio is playing.
  `position()` is the audible position, with output latency compensated.
- **Pause:** `on_pause()` pauses audio. `on_resume()` resumes it and resets
  any per-window state. `close()` stops audio; the shell calls it whenever
  it drops the game, and the app calls it when it quits.
- **`preload()`:** loads the art, builds the renderer and renders one
  throwaway frame. The shell calls it when it builds the game (on the song
  pick), so the countdown's first frame takes ~3 ms instead of ~30-60 ms.
- **Headless runs** (`--headless`) set `audio.SILENT`: the same clock, no sound.
- **`score`:** the property the shell's leaderboard reads, an int from 0 to
  10000.

**State the renderer reads.** Read-only; the renderer never mutates the game:

```
game.song             Song (title, artist, bpm, sections, coach outfit, duration; .level, the difficulty's chart)
game.difficulty       "easy" | "normal" | "hard" (the badges)
game.phase            Phase
game.song_time        seconds since beat 0 (negative before it)
game.beat             fractional beat (song_time * bpm / 60)
game.coach_pose       the current interpolated pose dict (section 3) for the coach
game.upcoming         list of PictogramItem(beat, move_label, pose, gold): the next 8 judge points, one card
                      each (a card stays 0.5 beat past its beat to slide through the ring)
game.feedback         list of Feedback(rating, gold, age, x?, ...), recent ratings for the popup (newest last)
game.score, game.display_score (animated), game.stars (0-5 float for the bar fill)
game.counts           {"perfect": n, "super": n, "good": n, "ok": n, "x": n}
game.player           latest Measured (or None) for the PiP skeleton tint
game.last_rating      rating of the last resolved point (for the PiP tint)
game.notice           None, or "STEP INTO THE FRAME" / "STEP BACK - SHOW YOUR ARMS" (after 0.6 s unseen);
                      never about the feet, which are optional
game.gold_flash       0..1, decaying, after a gold move rated SUPER or better
game.combo            consecutive SUPER-or-better count
game.results_t        seconds since game_over (for the results animation)
```

## 9. Shell support: song select

A game class can offer **variants**:
`variants() -> [{"id", "title", "subtitle", "board_id"}]`. Choosing that
game's tile on the welcome screen opens a **variant picker**: hover-to-press
song cards, using the same `Dwell` and keys 1-9. The game is then built with
`variant=<id>`.

- **Leaderboard:** one board per variant and difficulty. `variants()` gives
  the normal board, `"dance-<song id>"`; the shell appends `_easy` / `_hard`
  for the other difficulties, and the game's own `board_id` agrees. The
  server accepts any `dance-*` id under one rule set (`leaderboard/src/rules.js`).
- **Difficulty:** `DanceGame.difficulties` = `[{"id", "label", "hint"}]` for
  easy / normal / hard, and `default_difficulty = "normal"`. The shell builds
  the game with `difficulty=<id>`; unknown ids fall back to normal
  (section 7a).
- **New hooks:** the shell calls `game.on_pause()`, `game.close()` and
  `game.preload()` when present, so audio stops and starts with the shell's
  screens and the art loads on the song pick.
- **No song list:** `--autostart` (and headless runs) start the first
  variant, on its own board.

## 10. Layout (1280 x 720)

Numbers from `render.py`:

- **Coach:** centre-left, standing in a spotlight. About 520 px tall (head
  to feet), shoulder width 114 px (`COACH_SCALE`), centred on x = 520, feet
  at y = 660, never drawn outside `COACH_ROI` (110, 56)-(930, 704).
- **Pictogram lane:** bottom-right, `LANE_RECT` x 740-1262, y 513-631. Cards
  scroll right to left at 128 px per beat onto a 146 px target ring at
  (812, 572); the move's name sits under the ring at y 670 (scale 0.72).
  Cards come as often as every beat; a card's size goes by distance alone
  (`card_width`): 92 px until a beat out, growing to 132 px on the ring, full
  size while it slides through and fades (both about 20% bigger than the
  first 76 / 108). Two cards a beat apart keep a gap of at least 16 px. Gold
  cards have gold frames. Cards are clipped above y 645: the shell's PAUSE
  button is x 1090-1258, y 646-698 (`PAUSE_RECT`), and only small cards
  (bottom y 622) ever reach x 1090. The footer starts at y 700.
- **Player PiP:** top-right, (880, 16)-(1264, 232), 384 x 216 (16:9): the
  camera frame as the game received it (already mirrored), with the measured
  skeleton tinted by the last rating. The frame is shrunk by a 2x box filter
  then a bilinear step: `INTER_AREA` straight to 384 px took 4.5 ms.
- **Intro:** the title card ("NOW DANCING", title, artist, difficulty badge)
  and GET READY centre on (958, 352), the 3-2-1-GO words on y 366, all under
  the PiP once popped in.
- **Difficulty badge:** a pill in the difficulty's colour (EASY green
  `theme.GOOD`, NORMAL the manifest's cyan, HARD its magenta; overridable in
  the art manifest as `"difficulty"`): on the intro card under the artist, in
  the HUD right of the score (left edge x 262, centred on y 58), and beside
  "RESULTS" on the results card.
- **Feedback:** the rating word pops just above the coach's ears, following
  its head (x clamped to 260-660, so the widest word stays left of the PiP;
  y at least 108). A newer word whisks the
  older one up and out within 0.04 s (`SUPERSEDE_FADE`; 0.12 s read as
  "PERFECT" stacked over "PERFECT"). Words are hidden while a notice shows (the feet notice too: it sits where they pop).
- **Notice:** a panel centred at (600, 96).
- **Score:** top-left at (78, 76), the combo under it. The star bar runs down
  the left edge, x 30-50, y 128-552, with five star markers.
- **Results:** a panel on the right, (700, 34)-(1250, 528), above the
  shell's buttons. Its count-up lands by ~1.25 s, before the shell's name
  entry covers it (`ENTRY_DELAY` 1.6 s).
- **Footer:** bottom-left belongs to the shell.

## 11. Adding a song

1. **Make a folder** `assets/dance/songs/<id>/` with `audio.m4a` (or .ogg/.mp3)
   and a `song.json` with at least `title`, `bpm`, `offset` and `duration`
   (`load_song` refuses a song without a title).
2. **Write the chart** (`moves` / `custom_moves`) by hand, or import it:
   `uv run kp dance-import --song <id> --video take.mp4 [--start SECONDS]`
   runs the model over a recording of someone dancing to the track and
   writes one keyframe per beat (`--merge` joins held poses; legs come out as
   `ll`/`rl` angles, which load as feet), keeping a
   `song.json.bak`. `--live` records from the camera while the song plays.
   Code can call `importer.chart_from_poses(samples, ...)` directly with
   (audio time, Pose) samples. The import is a draft: review it with
   `uv run kp dance-preview --song <id>` (writes `<id>_chart.png`), trim
   it, rename the moves and mark gold ones.
3. **Add the id** to `songs/index.json`.
4. **Check it:** `kp dance-preview` prints the choreography lint (section
   5a); fix what it flags. Then run `uv run --with pytest pytest tests/test_dance_chart.py`
   (it validates and lints every song in `index.json`, on every difficulty). Look at
   the derived EASY chart with `uv run kp dance-preview --song <id> --difficulty easy`;
   if it reads wrong, hand-make one under `"charts": {"easy": [...]}`. The browser picks the song up
   from the same folder. Followed end to end in QA with a 20 s cut of Neon
   Nights and a synthetic dancer: 15 imported moves, the song on the song
   list with its own board (`dance-<id>`), and a perfect bot at 10000.
