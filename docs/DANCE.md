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
  songs/index.json           ["neon_nights", ...]: the song list, in menu order
  songs/<id>/song.json       metadata + chart (section 5)
  songs/<id>/audio.m4a       AAC, 44.1 kHz stereo; read by both builds
  generated/                 art from tools/generate_dance_assets.py (+ manifest.json)

src/kpapp/game/dance/
  pose.py        pose angles, forward kinematics, measuring a player, similarity  (core)
  chart.py       load/validate songs and moves, beat <-> time, judge timeline      (core)
  judge.py       timing windows, ratings, score, stars
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

A pose is a set of angles in degrees, in the **torso frame**: as if the torso
were upright, before applying `lean`.

```json
{"la": [upper, fore], "ra": [upper, fore],   // arms: screen-left, screen-right
 "ll": [thigh, shin], "rl": [thigh, shin],   // legs
 "lean": 0, "head": 0, "shift": 0, "squat": 0}
```

- **Limb angle:** 0 = pointing straight down. Positive rotates outward and up,
  away from the body's midline: towards the screen-left for `la`/`ll`, and
  towards the screen-right for `ra`/`rl`. 90 = horizontal outward, 180 =
  straight up, above 180 = past vertical over the head towards the other
  side, negative = across the front of the body towards the other side.
  The forearm and shin angles are **absolute in the torso frame**, not
  relative to the parent segment. That keeps authoring simple: "forearm
  straight up" is 180 whatever the upper arm does.
- **lean:** torso tilt from vertical. Positive tilts the head towards
  screen-right.
- **head:** head tilt relative to the torso, the same sign. Cosmetic.
- **shift:** sideways step of the whole body, in shoulder widths; positive is
  screen-right.
- **squat:** 0..1, lowers the hips; the coach bends its knees to match.
  Cosmetic for judging.
- **Missing keys default to the rest pose:**
  `la=[10,5]`, `ra=[10,5]`, `ll=[3,2]`, `rl=[3,2]`, lean/head/shift/squat 0.

**Body proportions** are in shoulder widths (SW = 1). Shoulder centre to hip
centre 1.25, hip width 0.6, upper arm 0.62, forearm 0.58, hand 0.16, thigh
0.85, shin 0.82, neck 0.2, head radius 0.3.

**`forward(pose) -> joints`** gives a dict of named 2D points in SW units,
with y down and the origin at the *rest* hip centre (`shift` and `squat` move
the `hip` point away from it, so the feet stay planted):
`hip, head, neck, l_shoulder, r_shoulder, l_elbow, r_elbow, l_wrist, r_wrist,
l_hand, r_hand, l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle`. The coach
renderer and pictograms draw from this.

**`measure(keypoint_pose, mirrored=True) -> Measured`** reads a player's
`Pose` (COCO keypoints) into the same torso-frame angles:
- The torso up-axis runs from the hip centre to the shoulder centre when
  both hips are visible (conf >= 0.4). Otherwise it is perpendicular to the
  shoulder line, pointing up. No shoulders means no measurement (None).
- **Per arm:** with the elbow and wrist visible, the upper arm and forearm
  angles. Without the elbow, only the shoulder-to-wrist "reach" direction
  (elbows are missing in about 54% of frames; AGENTS.md section 6). With the
  elbow but not the wrist (a close player's hands above the top of the
  frame), only the upper arm.
- **Per leg:** only with the hip, knee and ankle visible.
- `Measured` also carries the shoulder width in px and which parts were seen.

**`similarity(target_pose, measured) -> (score 0..1, parts)`**:
- **Per segment:** the angle difference d in degrees (wrapped to 0..180)
  gives `s = clamp(1 - (d - 20) / 70, 0, 1)`. Up to 20 degrees is perfect;
  90 degrees scores 0.
- **Arm with elbow:** the mean of the upper arm and forearm. **Arm without
  elbow:** the reach direction compared with the target's shoulder-to-wrist
  direction from `forward()`. **Elbow without wrist:** the upper arm alone.
  Ignoring it made a perfect but close player's Clap Up an X: the window only
  ever saw the arms on their way up.
- **Weights:** arms 1.0 each. Legs 0.35 each, and only when measured *and*
  the target leg is not near rest (more than 25 degrees from rest in either
  angle). Lean 0.3, only when hips are visible and |target lean| > 8; it
  scores `clamp(1 - (|d| - 6) / 24, 0, 1)`.
- **Score:** the weighted mean over the parts that count. If no arm is
  measured, the score is None: "can't see you", which is never a miss.

## 4. Move library (`assets/dance/moves.json`)

```json
{"version": 1,
 "moves": {
   "disco_point_r": {
     "label": "Disco Point",
     "beats": 2,                       // how long the move occupies the chart
     "keyframes": [                    // beat offsets within the move, ascending
       {"beat": 0, "pose": {"ra": [150, 165], "la": [20, 10], "lean": 6}},
       {"beat": 1, "pose": {"ra": [-30, -40], "la": [40, 20]}}
     ],
     "judge": [0, 1],                  // keyframe indexes that are judged (default: all)
     "pictogram": 0                    // the move's thumbnail in kp dance-preview (default: first judged)
   }}}
```

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
  in `kp dance-preview`.
- **Mirrored twins:** a move named `x_l` / `x_r` can be generated from the
  other side (`chart.mirror_pose`: swap `la`/`ra` and `ll`/`rl`, negate
  `lean`, `head` and `shift`). Prefer authoring one side and declaring
  `"mirror_of": "x_r"`.

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
 "custom_moves": {}                    // same schema as the library; overrides by name
}
```

- **Validation:** `chart.load_song` rejects overlapping moves, unknown move
  names, keyframes outside a move, non-ascending beats, a chart running past
  the duration, and bad pose keys, with a clear message naming the song,
  move and beat. `tests/test_dance_chart.py` validates every song in
  `assets/dance/songs/`.
- **Gold moves:** a judged point counts double, flashes gold, and earns a
  gold burst when rated SUPER or better. Use one or two per chorus.

## 6. The test song: "Neon Nights"

- **Tempo:** 120 BPM, 4/4, so one beat is 0.5 s and one bar is 2 s.
- **Length:** 52 bars = 208 beats = 104 s of music plus 2 s of ring-out;
  `duration` in song.json is 106.0, the audio file's own length.
- **Key and style:** A minor, upbeat synth-pop or electro.

| beats | section | moves? |
|---|---|---|
| 0-15 | intro (4 bars) | none; the coach grooves, "GET READY" |
| 16-47 | verse 1 (8 bars) | simple, mostly 2-beat moves |
| 48-79 | chorus 1 (8 bars) | signature moves, 1 gold |
| 80-111 | verse 2 (8 bars) | variations |
| 112-143 | chorus 2 (8 bars) | as chorus 1, 1 gold |
| 144-159 | bridge (4 bars) | slower, 4-beat moves |
| 160-191 | final chorus (8 bars) | as chorus, 2 gold |
| 192-207 | outro (4 bars) | a final pose at beat 192, then a bow |

Beat 0 is at `offset` seconds into the audio. The generator must put the first downbeat exactly there, and the
chart is authored against these beats.

## 7. Judging (`judge.py`)

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

Results with these numbers, on the test chart (42 points) / Neon Nights (145):

| player | test chart | Neon Nights |
|---|---|---|
| perfect, 20 Hz, 150 ms lag | 10000, all PERFECT | 10000, all PERFECT |
| perfect, half the elbows missing, no hips | 10000 | 10000 |
| perfect, 250 ms behind the coach | 10000 | 10000 |
| 10-20 degrees off, 200 ms late, 15 Hz with 200 ms lag | 9478 (5 stars) | 9379 (5 stars) |
| 25-30 degrees off, 250 ms late | 8217 (4 stars: 5 P, 37 S) | 7919 (3 stars: 9 P, 124 S, 12 G) |
| frozen on the first pose | 3533 (1 star) | 1268 |
| idle, arms down | 1435 | 1289 |
| random arms | 2402 (1 star) | 2117 (1 star) |
| wrong side (mirror image) | 1565 | 5060 (2 stars) |

The wrong-side player is caught only on asymmetric moves. A third of
Neon Nights' points are symmetric poses (Clap Up, Shine, Star Jump...), where
the mirror image *is* the move.

**Through the whole app path** (QA: the real Shell, `ControlMapper`,
`PoseExtrapolator`, ±4 px noise, Neon Nights):

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
game.song             Song (title, artist, bpm, sections, coach outfit, duration)
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
game.notice           None, "STEP BACK - SHOW YOUR ARMS" or "STEP INTO THE FRAME" (after 0.6 s unseen)
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

- **Leaderboard:** one board per variant, `board_id` = `"dance-<song id>"`.
  The server accepts any `dance-*` id under one rule set
  (`leaderboard/src/rules.js`).
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
- **Pictogram lane:** bottom-right, `LANE_RECT` x 748-1262, y 544-642. Cards
  scroll right to left at 112 px per beat onto a target ring at (806, 593);
  the move's name sits under the ring at y 672. Cards come as often as every
  beat; a card's size goes by distance alone (`card_width`): 76 px until a
  beat out, growing to 108 px on the ring, full size while it slides through
  and fades. Two cards a beat apart keep a gap of at least 8 px. Gold cards
  have gold frames. x 1090+ below y 646 is the shell's
  PAUSE button.
- **Player PiP:** top-right, (944, 16)-(1264, 196), 320 x 180: the camera
  frame as the game received it (already mirrored), with the measured
  skeleton tinted by the last rating.
- **Feedback:** the rating word pops just above the coach's ears, following
  its head (x clamped to 260-780, y at least 108). A newer word whisks the
  older one up and out within 0.04 s (`SUPERSEDE_FADE`; 0.12 s read as
  "PERFECT" stacked over "PERFECT"). Words are hidden while a notice shows.
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
   writes one keyframe per beat (`--merge` joins held poses), keeping a
   `song.json.bak`. `--live` records from the camera while the song plays.
   Code can call `importer.chart_from_poses(samples, ...)` directly with
   (audio time, Pose) samples. The import is a draft: review it with
   `uv run kp dance-preview --song <id>` (writes `<id>_chart.png`), trim
   it, rename the moves and mark gold ones.
3. **Add the id** to `songs/index.json`.
4. **Check it:** run `uv run --with pytest pytest tests/test_dance_chart.py`
   (it validates every song in `index.json`). The browser picks the song up
   from the same folder. Followed end to end in QA with a 20 s cut of Neon
   Nights and a synthetic dancer: 15 imported moves, the song on the song
   list with its own board (`dance-<id>`), and a perfect bot at 10000.
