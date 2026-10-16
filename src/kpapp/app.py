"""CLI entry point.

Modes:
  bench  headless model throughput, no camera or window
  pose   camera + skeleton overlay, the real-time sanity check
  game   pose-driven demo game
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np

from .camera import CameraStream, SyntheticCamera
from .config import Config
from .controls import ControlMapper
from .game import REGISTRY
from .inference import KeypointDetector, PoseExtrapolator
from .overlay import draw_banner, draw_box, draw_hud, draw_pose
from .pipeline import InferenceWorker, RateMeter
from .recorder import Recorder
from .leaderboard import create_board
from .shell import Screen, Shell


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    for name in (
        "resolution", "threshold", "device", "half",
        "smoothing", "max_people", "infer_every", "extrapolation",
    ):
        value = getattr(args, name, None)
        if value is not None:
            setattr(cfg, name, value)
    if getattr(args, "width", None):
        cfg.capture_width = args.width
        cfg.capture_height = args.height
    if getattr(args, "no_compile", False):
        cfg.compile_model = False
    if getattr(args, "no_mirror", False):
        cfg.mirror = False
    if getattr(args, "fps", None):
        cfg.display_fps = args.fps
    if getattr(args, "verbose", False):
        cfg.verbose = True
    return cfg


def _power_tip() -> str | None:
    """Warn when macOS power settings slow the model.

    On this Mac the 336 px model takes ~35 ms a frame plugged in and ~50 ms
    on battery in Low Power Mode - about 30 ms more lag between a movement and
    the game seeing it. Only the player can fix that, so tell them.
    """
    if sys.platform != "darwin":
        return None
    import subprocess

    try:
        batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=2).stdout
        settings = subprocess.run(["pmset", "-g"], capture_output=True, text=True, timeout=2).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    on_battery = "Battery Power" in batt
    low_power = any(line.split()[:2] == ["lowpowermode", "1"] or line.split()[:2] == ["powermode", "1"]
                    for line in settings.splitlines())
    if low_power:
        return ("Low Power Mode is on: pose tracking will lag ~30 ms more. "
                "Plug in and turn it off for the most responsive play.")
    if on_battery:
        return "On battery power: plugging in can make pose tracking more responsive."
    return None


def _prepare(cfg: Config, size: tuple[int, int], optimize: bool) -> KeypointDetector:
    print(f"[kp] loading RF-DETR keypoint model at {cfg.resolution}px ...")
    t0 = time.perf_counter()
    detector = KeypointDetector(cfg)
    print(f"[kp] device: {detector.device}  ({time.perf_counter() - t0:.1f}s)")
    if optimize:
        print("[kp] optimizing graph (one-time, can take up to a minute) ...")
        t0 = time.perf_counter()
        detector.optimize()
        print(f"[kp] optimized in {time.perf_counter() - t0:.1f}s")
    print("[kp] warming up ...")
    detector.warmup(size)
    return detector


def cmd_bench(args: argparse.Namespace) -> int:
    """Measure model-only throughput on synthetic frames."""
    cfg = build_config(args)
    size = (cfg.capture_width, cfg.capture_height)
    detector = _prepare(cfg, size, optimize=cfg.compile_model)

    rng = np.random.default_rng(0)
    frames = [
        (rng.random((size[1], size[0], 3)) * 255).astype(np.uint8)
        for _ in range(8)
    ]

    times: list[float] = []
    for i in range(args.iters):
        t0 = time.perf_counter()
        detector.infer(frames[i % len(frames)])
        times.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(times)
    print("\n--- benchmark ---")
    print(f"device      : {detector.device}")
    print(f"resolution  : {cfg.resolution}px   input {size[0]}x{size[1]}")
    print(f"iterations  : {args.iters}")
    print(f"median      : {np.median(arr):6.1f} ms  ({1000 / np.median(arr):5.1f} FPS)")
    print(f"mean        : {arr.mean():6.1f} ms")
    print(f"p95         : {np.percentile(arr, 95):6.1f} ms")
    print(f"min / max   : {arr.min():6.1f} / {arr.max():6.1f} ms")
    return 0


def _open_stream(index: int, cfg: Config):
    return CameraStream(index=index, width=cfg.capture_width, height=cfg.capture_height,
                        fps=cfg.capture_fps, mirror=cfg.mirror).start(wait=0)


def _begin_camera(args: argparse.Namespace, cfg: Config):
    """Pick the camera and start it warming up, without waiting for frames.

    A camera that has sat idle can take several seconds to start streaming.
    Starting it here, before the 5-10 s model load, lets that warm-up happen in
    time the player is already waiting through. Returns (camera or None, choice).
    """
    from .camera import list_cameras, resolve_camera

    if getattr(args, "synthetic", False):
        return SyntheticCamera(width=cfg.capture_width, height=cfg.capture_height, fps=30), None
    choice = resolve_camera(getattr(args, "camera", None), list_cameras())
    cfg.camera_index = choice.index
    print(f"[kp] using {choice.label}")
    try:
        return _open_stream(choice.index, cfg), choice
    except RuntimeError as exc:
        if choice.explicit:
            raise
        print(f"[kp] {choice.label}: {exc}")
        return None, choice


def _await_real(camera, timeout: float) -> str:
    """Wait for a real frame: "real", "blank" (streaming solid blank), or "timeout".

    A cold start sends nothing for a while and deserves the wait; a camera
    streaming solid blank frames at full rate will not come good by waiting.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if camera.last_real_time > 0.0:
            return "real"
        if getattr(camera, "is_blank_stream", lambda: False)():
            return "blank"
        time.sleep(0.05)
    return "real" if camera.last_real_time > 0.0 else "timeout"


def _ensure_streaming(camera, choice, cfg: Config, timeout: float = 8.0,
                      opener=None, revive: int = 2):
    """Confirm the camera is sending real images, recovering it if not.

    The default is the built-in camera, and a phone is never switched to
    automatically. On this Mac the built-in camera has intermittently sent
    solid black, and a reopen has brought it back, so it is revived before
    anything else is tried. A camera the player named is respected as-is.
    """
    from . import camera as camera_mod

    if choice is None:  # synthetic
        return camera
    opener = opener or (lambda idx: _open_stream(idx, cfg))

    if camera is not None:
        if camera.last_real_time == 0.0:
            print("[kp] waiting for the camera to start ...")
        state = _await_real(camera, timeout)
        if state == "real":
            return camera
        if state == "blank":
            print(f"[kp] {choice.label} is sending only blank frames")

    if choice.explicit:
        print(f"[kp] WARNING: {choice.label} is not sending real images.")
        return camera

    for attempt in range(1, revive + 1):
        print(f"[kp] reopening {choice.label} ({attempt}/{revive}) ...")
        if camera is not None:
            camera.stop()  # release, or the reopen fights this handle
        try:
            camera = opener(choice.index)
        except RuntimeError:
            camera = None
            continue
        if _await_real(camera, 5.0) == "real":
            print(f"[kp] {choice.label} recovered")
            return camera

    if camera is not None:
        camera.stop()
    idx = camera_mod.find_working_camera(preferred=choice.index, seconds=4.0,
                                         skip=(choice.index,) + choice.avoid)
    if idx is None:
        phone_hint = " Or use your phone deliberately: --camera iphone." if choice.avoid else ""
        raise RuntimeError(
            f"{choice.label} is not streaming real images and no other non-phone camera "
            "is available. Close apps that may hold it (Zoom, Teams, Loom, FaceTime)."
            + phone_hint
        )
    print(f"[kp] switched to camera {idx}")
    fallback = opener(idx)
    if _await_real(fallback, timeout) != "real":
        fallback.stop()
        raise RuntimeError(f"camera {idx} passed the probe but then stopped streaming.")
    return fallback


def _camera_problem(camera, now: float) -> str | None:
    """A message for the player if the camera has stopped working, else None."""
    if now - camera.last_frame_time > 1.5:
        return "CAMERA STALLED - NO NEW FRAMES"
    if now - camera.last_real_time > 2.0:
        return "CAMERA SENDING BLANK FRAMES"
    return None


def _run_loop(args: argparse.Namespace, game_name: str | None, use_shell: bool = True) -> int:
    cfg = build_config(args)

    headless = getattr(args, "headless", False)
    duration = getattr(args, "duration", None)
    if headless and not duration:
        duration = 10.0
    if headless:  # a benchmark run makes no sound (games still get a music clock)
        from . import audio

        audio.SILENT = True

    try:
        camera, choice = _begin_camera(args, cfg)
    except (RuntimeError, ValueError) as exc:
        print(f"[kp] {exc}", file=sys.stderr)
        return 1

    size = camera.size if camera is not None else (cfg.capture_width, cfg.capture_height)
    # The model loads while the camera warms up; only then is the camera checked.
    detector = _prepare(cfg, size, optimize=cfg.compile_model)
    try:
        camera = _ensure_streaming(camera, choice, cfg)
    except RuntimeError as exc:
        print(f"[kp] {exc}", file=sys.stderr)
        return 1
    size = camera.size
    print(f"[kp] camera {size[0]}x{size[1]}")
    tip = _power_tip()
    if tip:
        print(f"[kp] tip: {tip}")
    worker = InferenceWorker(camera, detector, cfg).start()
    mapper = ControlMapper(size)
    extrapolator = PoseExtrapolator(gain=cfg.extrapolation, max_lead=cfg.max_lead)
    shell = None
    if use_shell:
        # Headless runs are benchmarks: there is no one to press start.
        autostart = bool(getattr(args, "autostart", False) or headless) and game_name is not None
        board = None if headless else create_board(getattr(args, "leaderboard", None))
        shell = Shell(size, REGISTRY, mirrored=cfg.mirror, selected=game_name, autostart=autostart, board=board)
        if board is not None:
            where = "worldwide" if getattr(board, "scope", "") == "world" else f"this computer ({board.path})"
            print(f"[kp] leaderboard: {where}")

    render_meter = RateMeter()
    recorder = Recorder(args.record) if getattr(args, "record", None) else None
    last_phase = None
    show_skeleton = shell is None  # skeleton is a debug view; games have their own art
    show_debug = shell is None
    last_result_ts = -1.0
    last_t = time.perf_counter()

    # The render loop runs on its own clock rather than waiting for camera
    # frames. Tying them together meant a stalled camera froze the whole game,
    # and it capped animation at the camera's rate even when healthy.
    period = 1.0 / max(1.0, cfg.display_fps)
    next_tick = time.perf_counter()
    camera_warned = False
    last_reopen = 0.0

    if headless:
        print(f"[kp] running headless for {duration:.0f}s ...")
    else:
        if shell is not None:
            print("[kp] arcade open - hover a game or press 1/2 to start; q quits")
        else:
            print("[kp] running - q quit, s skeleton, d debug")
        cv2.namedWindow(cfg.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(cfg.window_name, size[0], size[1])
        if shell is not None:
            # Clicks work like hovering does (OpenCV reports image coordinates).
            def on_mouse(event, x, y, _flags, _param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    shell.handle_click(x, y)
            cv2.setMouseCallback(cfg.window_name, on_mouse)
    started = time.perf_counter()

    try:
        while True:
            t_start = time.perf_counter()
            frame, seq = camera.read()
            if frame is None:
                time.sleep(0.01)
                continue
            canvas = frame.copy()

            result = worker.result
            now = time.perf_counter()
            dt, last_t = now - last_t, now

            # Feed each result to the extrapolator exactly once, then ask for
            # poses projected to *now* rather than to the frame the model saw.
            if result.timestamp != last_result_ts:
                extrapolator.update(result, arrived=now)
                last_result_ts = result.timestamp
            poses = extrapolator.poses_at(now)
            pose = poses[0] if poses else None

            controls = mapper.map(pose)
            controls.step_pose = extrapolator.held[0] if extrapolator.held else None
            game = shell.game if shell is not None else None
            score_before = getattr(game, "score", None)
            t_update = time.perf_counter()
            if shell is not None:
                shell.update(controls, dt)
            t_render = time.perf_counter()
            if shell is not None:
                shell.render(canvas)
            t_rendered = time.perf_counter()
            game = shell.game if shell is not None else None

            if show_skeleton:
                for p in poses:
                    draw_box(canvas, p)
                    draw_pose(canvas, p)

            render_meter.tick()
            if show_debug:
                # Age tells you how stale the drawn pose is - the real
                # end-to-end latency the player feels.
                age = (seq - result.frame_seq) if result.frame_seq else 0
                draw_hud(
                    canvas,
                    [
                        f"render {render_meter.fps:5.1f} FPS   infer {worker.meter.fps:5.1f} FPS",
                        f"model  {worker.meter.mean:5.1f} ms   pose age {age:2d} frames",
                        f"lead   {extrapolator.last_lead_ms:5.1f} ms   people {len(poses)}",
                    ],
                    origin="bottom-left" if shell is not None else "top-left",
                )

            problem = _camera_problem(camera, time.monotonic())
            if problem:
                draw_banner(canvas, problem, "close Zoom / Teams / Loom, or run: uv run kp cameras")
                if not camera_warned:
                    print(f"[kp] {problem.lower()}")
                    camera_warned = True
                # Try to recover a stalled device, but not in a tight loop. On
                # this Mac a stall has needed two reopens; these timings bring
                # that from ~10 s down to ~5 s.
                stalled_for = time.monotonic() - camera.last_frame_time
                if stalled_for > 2.0 and time.monotonic() - last_reopen > 3.0:
                    last_reopen = time.monotonic()
                    print("[kp] reopening camera ...")
                    camera.reopen()
            elif camera_warned:
                print("[kp] camera recovered")
                camera_warned = False

            if recorder is not None:
                score_after = getattr(game, "score", None)
                phase = getattr(getattr(game, "phase", None), "value", None)
                event = None
                if score_before is not None and score_after != score_before:
                    event = "score"
                elif phase != last_phase and phase is not None:
                    # Transitions only: logging every frame of a phase floods
                    # the event budget with near-identical frames.
                    event = phase
                last_phase = phase
                recorder.frame(canvas, _metrics(
                    game, pose, result, worker, render_meter,
                    screen=getattr(getattr(shell, "screen", None), "value", None),
                    update_ms=(t_render - t_update) * 1000,
                    render_ms=(t_rendered - t_render) * 1000,
                    frame_ms=(time.perf_counter() - t_start) * 1000,
                    camera_age_ms=(time.monotonic() - camera.last_frame_time) * 1000,
                ), event=event, raw=frame)

            if duration and time.perf_counter() - started >= duration:
                break

            # Pace to the display rate. If we fell badly behind, resync rather
            # than sprinting to catch up.
            next_tick += period
            delay = next_tick - time.perf_counter()
            if delay < -period:
                next_tick = time.perf_counter()
                delay = 0.0

            if headless:
                if delay > 0:
                    time.sleep(delay)
                continue

            cv2.imshow(cfg.window_name, canvas)
            key = normalise_key(cv2.waitKeyEx(max(1, int(delay * 1000))))
            if key == 255:
                pass
            elif shell is not None and shell.screen is Screen.ENTRY:
                shell.handle_key(key)  # every key is part of the name here, q included
            elif key == ord("q"):
                break
            elif key == ord("s"):
                show_skeleton = not show_skeleton
            elif key == ord("d"):
                show_debug = not show_debug
            elif shell is not None:
                # Esc steps back a screen; it only quits from the welcome screen.
                if shell.handle_key(key):
                    break
            elif key == 27:
                break
            if cv2.getWindowProperty(cfg.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if shell is not None and shell.game is not None:
            shell._close_game()  # stop the music (Raccoon Dance) before the window goes
        worker.stop()
        camera.stop()
        cv2.destroyAllWindows()
        if recorder is not None:
            recorder.close()
            print(f"[kp] recording saved to {recorder.dir}")

    print(
        f"\n[kp] render {render_meter.fps:.1f} FPS | "
        f"inference {worker.meter.fps:.1f} FPS ({worker.meter.mean:.1f} ms) | "
        f"frames skipped by model {worker.skipped}"
    )
    return 0


# cv2.waitKeyEx's arrow-key codes per platform (macOS Cocoa, GTK/X11, Windows).
_ARROWS = {
    63234: "left", 63232: "up", 63235: "right", 63233: "down",
    65361: "left", 65362: "up", 65363: "right", 65364: "down",
    2424832: "left", 2490368: "up", 2555904: "right", 2621440: "down",
}


def normalise_key(raw: int) -> int:
    """A waitKeyEx code -> 255 for none, the shell's KEY_* for arrows, else the low byte."""
    from .shell import KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_UP

    if raw < 0:
        return 255
    arrow = _ARROWS.get(raw)
    if arrow is not None:
        return {"left": KEY_LEFT, "up": KEY_UP, "right": KEY_RIGHT, "down": KEY_DOWN}[arrow]
    return raw & 0xFF


def cmd_cameras(args: argparse.Namespace) -> int:
    """Report which camera indices actually stream real images."""
    from .camera import list_cameras, probe

    names = {c.index: c for c in list_cameras()}
    print("[kp] probing cameras (a few seconds each) ...")
    print(f"{'index':>5}  {'status':<10} {'fps':>5}  {'real/total':>11}  size")
    working = []
    for idx in range(args.max_index + 1):
        r = probe(idx, seconds=args.seconds)
        if not r["opened"]:
            print(f"{idx:>5}  {'absent':<10}")
            continue
        status = "OK" if r["healthy"] else ("BLACK" if r["frames"] else "NO FRAMES")
        size = f"{r['size'][0]}x{r['size'][1]}" if r["size"] else "-"
        info = names.get(idx)
        tag = (f"  {info.name}" + ("  [built-in, default]" if info.builtin else "")
               + ("  [phone]" if info.phone else "")) if info else ""
        print(f"{idx:>5}  {status:<10} {r['fps']:>5}  {r['real_frames']:>5}/{r['frames']:<5}  {size}{tag}")
        if r["healthy"]:
            working.append(idx)
    if working:
        def describe(i):
            info = names.get(i)
            return f"{info.name} ({i})" if info else str(i)
        print(f"[kp] working: {', '.join(describe(i) for i in working)}")
        builtin = next((c for c in names.values() if c.builtin), None)
        if builtin and builtin.index not in working:
            print(f"[kp] the built-in {builtin.name} is not sending images - if the lid is "
                  "closed or the camera covered, open it; it can take a few seconds to wake.")
    else:
        print("[kp] no camera delivered real frames. Close apps that may hold it "
              "(Zoom, Teams, Loom, FaceTime), or check the iPhone if Continuity Camera is on.")
    return 0 if working else 1


def _metrics(game, pose, result, worker, render_meter, **timings) -> dict:
    """One line of the recording: timings, tracking quality, game state."""
    from .config import KP

    screen = timings.pop("screen", None)
    m = {k: round(v, 3) for k, v in timings.items()}
    if screen:
        m["screen"] = screen
    m["infer_ms"] = round(result.infer_ms, 2)
    m["render_fps"] = round(render_meter.fps, 1)
    m["infer_fps"] = round(worker.meter.fps, 1)
    m["people"] = len(result.poses)
    if pose is not None:
        m["lw_conf"] = round(float(pose.confidence[KP["left_wrist"]]), 3)
        m["rw_conf"] = round(float(pose.confidence[KP["right_wrist"]]), 3)
        m["hip_conf"] = round(float(min(pose.confidence[KP["left_hip"]],
                                        pose.confidence[KP["right_hip"]])), 3)
    blade = getattr(game, "blade", None)
    if blade is not None:
        m["blade_active"] = blade.active
        m["blade_speed"] = round(blade.speed, 1)
        m["slicing"] = blade.slicing
    for attr in ("score", "lives"):
        if hasattr(game, attr):
            m[attr] = getattr(game, attr)
    phase = getattr(game, "phase", None)
    if phase is not None:
        m["phase"] = getattr(phase, "value", str(phase))
    return m


def cmd_pose(args: argparse.Namespace) -> int:
    return _run_loop(args, game_name=None, use_shell=False)


def cmd_play(args: argparse.Namespace) -> int:
    return _run_loop(args, game_name=getattr(args, "game", None), use_shell=True)


def cmd_game(args: argparse.Namespace) -> int:
    return _run_loop(args, game_name=args.game)


def cmd_dance_preview(args: argparse.Namespace) -> int:
    """Contact sheet of a song's chart (or the move library) for review."""
    from pathlib import Path

    from .game.dance import chart as dance_chart
    from .game.dance.preview import chart_sheet, library_sheet

    try:
        if args.library:
            img = library_sheet(dance_chart.load_library())
        else:
            img = chart_sheet(dance_chart.load_song(dance_chart.SONGS / args.song, difficulty=args.difficulty))
    except dance_chart.ChartError as exc:
        print(f"[kp] {exc}", file=sys.stderr)
        return 1
    name = "moves" if args.library else (args.song if args.difficulty == "normal" else f"{args.song}_{args.difficulty}")
    out = Path(args.out or f"{name}_chart.png")
    cv2.imwrite(str(out), img)
    print(f"[kp] wrote {out} ({img.shape[1]}x{img.shape[0]})")
    return 0


def _live_dance_samples(args: argparse.Namespace, cfg: Config, detector, song, end_time: float):
    """Play the song and record the camera: (audio time, Pose) samples (beat 0 at song.offset)."""
    from .audio import open_player
    from .overlay import outlined_text

    camera, choice = _begin_camera(args, cfg)
    camera = _ensure_streaming(camera, choice, cfg)
    player = open_player(song.audio, song.duration)
    camera_lag = 0.04  # a frame reaches us ~40 ms after it was captured (AGENTS.md section 2)
    samples, seq = [], -1
    player.play(0.0)
    try:
        while not player.finished:
            frame, seq = camera.wait_for_frame(seq, timeout=1.0)
            if frame is None:
                continue
            # On the audio's clock (beat 0 at song.offset), like video_samples:
            # chart_from_poses and end_time both count from the start of the audio.
            t = player.position() - camera_lag
            if t > end_time:
                break
            result = detector.infer(frame)
            pose = result.poses[0] if result.poses else None
            samples.append((t, pose))
            view = frame.copy()
            if pose is not None:
                draw_pose(view, pose)
            beat = song.beat_of(t - song.offset)
            outlined_text(view, f"{song.title}  beat {beat:5.1f}  {song.section_at(beat)}   q stops", (24, 44),
                          cv2.FONT_HERSHEY_DUPLEX, 0.9, (70, 210, 255))
            cv2.imshow(cfg.window_name, view)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        player.stop()
        camera.stop()
        cv2.destroyAllWindows()
    return samples


def cmd_dance_import(args: argparse.Namespace) -> int:
    """Recording of someone dancing to a song -> draft chart (docs/DANCE.md section 11)."""
    import json
    from pathlib import Path

    from .game.dance import chart as dance_chart
    from .game.dance import importer

    if not args.video and not args.live:
        print("[kp] give --video take.mp4 or --live", file=sys.stderr)
        return 2
    path, data = importer.load_song_data(args.song)
    bpm, offset, duration = float(data["bpm"]), float(data.get("offset", 0.0)), float(data["duration"])
    start_beat = args.from_beat
    end_beat = args.to_beat if args.to_beat is not None else float(int(duration * bpm / 60.0))
    cfg = build_config(args)
    detector = _prepare(cfg, (cfg.capture_width, cfg.capture_height), optimize=False)

    if args.live:
        song = dance_chart.load_song(path.parent)
        samples = _live_dance_samples(args, cfg, detector, song, offset + end_beat * 60.0 / bpm)
        mirrored = cfg.mirror
    else:
        wanted = importer.beat_times(bpm, offset, start_beat, end_beat, args.every)
        video_start = args.start if args.start is not None else offset

        def detect(frame):
            result = detector.infer(frame)
            return result.poses[0] if result.poses else None

        def progress(i, n):
            print(f"\r[kp] analysing beat {i + 1}/{n}", end="", flush=True)

        samples = importer.video_samples(args.video, detect, wanted, video_start, offset,
                                         mirror=not args.no_mirror, progress=progress)
        print()
        mirrored = not args.no_mirror
    imported = importer.chart_from_poses(samples, bpm, offset, start_beat, end_beat, every=args.every,
                                         mirrored=mirrored, merge=args.merge)
    print(f"[kp] {len(samples)} poses -> {len(imported['moves'])} moves")
    out_path = Path(args.out) if args.out else path
    if args.out and out_path.exists():
        base = json.loads(out_path.read_text())
    else:
        base = data
    merged = importer.merge_into_song(base, imported, start_beat, end_beat)
    importer.write_song_json(out_path, merged)
    if out_path.name == "song.json":
        try:
            dance_chart.load_song(out_path.parent)
        except dance_chart.ChartError as exc:
            print(f"[kp] warning: the imported chart doesn't validate: {exc}", file=sys.stderr)
    print(f"[kp] wrote {out_path}" + (f" (backup {out_path.with_suffix('.json.bak')})" if out_path == path else ""))
    print(f"[kp] review it: uv run kp dance-preview --song {args.song}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="kp", description="Real-time RF-DETR keypoint app")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--resolution", type=int, help="model input size, must be a multiple of 24 (default 576)")
        p.add_argument("--threshold", type=float, help="detection confidence threshold")
        p.add_argument("--device", type=str, help="mps | cuda | cpu (default: auto)")
        p.add_argument("--half", action="store_true", help="run in fp16")
        p.add_argument("--no-compile", action="store_true", help="skip graph optimization")
        p.add_argument("--width", type=int, help="capture width")
        p.add_argument("--height", type=int, default=720, help="capture height")
        p.add_argument("--max-people", type=int, dest="max_people", help="max tracked poses")
        p.add_argument("--infer-every", type=int, dest="infer_every", help="run model on at most every Nth camera frame")
        p.add_argument("--smoothing", type=float, help="keypoint EMA, 0 disables")
        p.add_argument("--verbose", action="store_true", help="show library logs during model setup")
        p.add_argument("--extrapolation", type=float,
                       help="velocity projection gain 0-1, 0 disables (default 0.7)")

    def live(p: argparse.ArgumentParser) -> None:
        p.add_argument("--camera", type=str,
                       help="camera index or name, e.g. 0, macbook, iphone (default: built-in)")
        p.add_argument("--no-mirror", action="store_true")
        p.add_argument("--synthetic", action="store_true", help="generated frames, no webcam needed")
        p.add_argument("--headless", action="store_true", help="no window; print timings and exit")
        p.add_argument("--fps", type=float, help="display/game rate, independent of the camera (default 60)")
        p.add_argument("--duration", type=float,
                       help="stop after this many seconds (headless default 10)")
        p.add_argument("--leaderboard", type=str, metavar="local|URL",
                       help="where high scores go: 'local' (this computer), a leaderboard server URL, "
                            "or by default the same server as the website (web/js/config.js), else local")
        p.add_argument("--record", type=str, metavar="DIR",
                       help="save frames and per-frame timings to DIR for review")

    p_cams = sub.add_parser("cameras", help="find which camera indices stream real images")
    p_cams.add_argument("--max-index", type=int, default=3)
    p_cams.add_argument("--seconds", type=float, default=6.0,
                        help="max wait per camera; a waking camera needs a few seconds")
    p_cams.set_defaults(func=cmd_cameras)

    p_bench = sub.add_parser("bench", help="headless throughput benchmark")
    common(p_bench)
    p_bench.add_argument("--iters", type=int, default=50)
    p_bench.set_defaults(func=cmd_bench, camera_index=None)

    p_pose = sub.add_parser("pose", help="live camera with skeleton overlay")
    common(p_pose)
    live(p_pose)
    p_pose.set_defaults(func=cmd_pose)

    p_play = sub.add_parser("play", help="open the arcade: welcome screen, pick a game")
    common(p_play)
    live(p_play)
    p_play.set_defaults(func=cmd_play)

    p_game = sub.add_parser("game", help="open the arcade with one game selected")
    common(p_game)
    live(p_game)
    p_game.add_argument("--autostart", action="store_true",
                        help="skip the welcome screen and countdown")
    p_game.add_argument("--game", choices=sorted(REGISTRY), default="catch")
    p_game.set_defaults(func=cmd_game)

    p_dimp = sub.add_parser("dance-import", help="turn a recording of someone dancing to a song into a chart")
    common(p_dimp)
    p_dimp.add_argument("--song", required=True, help="song id: a folder in assets/dance/songs with a song.json")
    p_dimp.add_argument("--video", type=str, help="recording of the dance (anything OpenCV can read)")
    p_dimp.add_argument("--live", action="store_true", help="play the song and record from the camera instead")
    p_dimp.add_argument("--camera", type=str, help="for --live: camera index or name")
    p_dimp.add_argument("--start", type=float, metavar="SECONDS",
                        help="where beat 0 falls in the video (default: the song's offset)")
    p_dimp.add_argument("--every", type=float, choices=[1.0, 0.5], default=1.0, help="beats per keyframe")
    p_dimp.add_argument("--from-beat", type=float, default=16.0, dest="from_beat",
                        help="first beat to import (default 16, after a 4-bar intro)")
    p_dimp.add_argument("--to-beat", type=float, dest="to_beat", help="import up to this beat (default: the end)")
    p_dimp.add_argument("--merge", action="store_true", help="merge held poses into longer moves")
    p_dimp.add_argument("--no-mirror", action="store_true", help="the recording is already a mirror view")
    p_dimp.add_argument("--out", type=str, help="write here instead of the song's song.json")
    p_dimp.set_defaults(func=cmd_dance_import, resolution=336)

    p_dprev = sub.add_parser("dance-preview", help="contact sheet of a song's chart, for review")
    p_dprev.add_argument("--song", default="neon_nights", help="song id (default neon_nights)")
    p_dprev.add_argument("--library", action="store_true", help="draw the whole move library instead")
    p_dprev.add_argument("--difficulty", choices=["easy", "normal", "hard"], default="normal",
                         help="which difficulty's chart to draw (default normal; easy is derived from it)")
    p_dprev.add_argument("--out", type=str, help="image path (default <song>_chart.png)")
    p_dprev.set_defaults(func=cmd_dance_preview)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
