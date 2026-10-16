"""Capture reference poses from the camera, for authoring dance moves.

    uv run python tools/capture_poses.py --out DIR [--poses Y,M,C,A] [--seconds 6]

For each pose name it shows the mirrored camera (as the game does) with a
countdown, then records the model's keypoints for about a second while the
player holds still. DIR gets, per pose, the mirrored frame with the skeleton
drawn (`<name>.jpg`) and every keypoint reading of the hold (`<name>.json`),
plus the dance pose measured from the steadiest reading (`poses.json`: arm
angles etc., the same `measure()` the judge uses). Press ESC to stop, SPACE
to start the next countdown early.

The frames are pictures of a real person: write them outside the repo (the
scratchpad) and never commit them.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from kpapp import theme
from kpapp.app import _begin_camera, _ensure_streaming, _prepare, build_config
from kpapp.game.dance.pose import measure
from kpapp.overlay import draw_banner, draw_pose

HOLD = 1.2  # seconds of keypoints recorded per pose


def _measured(m) -> dict | None:
    if m is None:
        return None
    out = {"arms": {s: {"upper": a.upper, "fore": a.fore, "reach": a.reach} for s, a in m.arms.items()},
           "lean": m.lean if m.has_hips else None}
    lower = getattr(m, "lower", None)
    if lower is not None:
        out["lower"] = {k: getattr(lower, k) for k in ("stance", "weight", "lift", "bend") if hasattr(lower, k)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="folder for the captures (keep it outside the repo)")
    ap.add_argument("--poses", default="Y,M,C,A", help="comma-separated pose names, captured in order")
    ap.add_argument("--seconds", type=float, default=6.0, help="countdown before each capture")
    ap.add_argument("--camera", type=str, help="camera index or name (default: built-in)")
    ap.add_argument("--resolution", type=int, default=336)
    args = ap.parse_args()
    run = argparse.Namespace(camera=args.camera, no_mirror=False, synthetic=False, resolution=args.resolution,
                             threshold=None, device=None, half=False, no_compile=True, width=None, height=720,
                             max_people=1, infer_every=None, smoothing=None, verbose=False, extrapolation=None)
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    names = [n.strip() for n in args.poses.split(",") if n.strip()]

    cfg = build_config(run)
    camera, choice = _begin_camera(run, cfg)
    detector = _prepare(cfg, camera.size, optimize=False)
    camera = _ensure_streaming(camera, choice, cfg)
    win = "capture poses"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    results: dict[str, dict] = {}
    seq = 0
    try:
        for name in names:
            start = time.monotonic()
            hold: list[tuple[float, np.ndarray, np.ndarray]] = []
            best_frame = None
            while True:
                frame, seq = camera.wait_for_frame(seq)
                if frame is None:
                    continue
                frame = frame.copy()
                poses = detector.infer(frame, seq).poses
                pose = poses[0] if poses else None
                left = args.seconds - (time.monotonic() - start)
                if pose is not None:
                    draw_pose(frame, pose)
                if left > 0:
                    draw_banner(frame, f"STRIKE  {name}", f"hold it in {int(left) + 1}...")
                elif left > -HOLD:
                    draw_banner(frame, f"HOLD  {name}", "recording")
                    if pose is not None:
                        hold.append((time.monotonic(), pose.xy.copy(), pose.confidence.copy()))
                        best_frame = frame.copy()
                else:
                    break
                theme.centred_text(frame, f"{names.index(name) + 1} / {len(names)}   ESC stop   SPACE skip wait",
                                   frame.shape[1] // 2, frame.shape[0] - 24, 0.6, theme.DIM, 1)
                cv2.imshow(win, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    return 1
                if key == 32 and left > 0:
                    start -= left
            if not hold:
                print(f"[capture] {name}: nobody seen during the hold, skipped", file=sys.stderr)
                continue
            # The steadiest reading: the one closest to the median of the hold.
            stack = np.stack([xy for _, xy, _ in hold])
            med = np.median(stack, axis=0)
            i = int(np.argmin(np.abs(stack - med).sum(axis=(1, 2))))
            _, xy, conf = hold[i]
            from kpapp.inference import Pose

            m = measure(Pose(xy=xy, confidence=conf, score=1.0), mirrored=True)
            results[name] = {"measured": _measured(m), "frame_size": list(camera.size)}
            (out / f"{name}.json").write_text(json.dumps({
                "name": name, "frame_size": list(camera.size),
                "hold": [{"t": round(t - hold[0][0], 3), "xy": x.round(1).tolist(), "conf": c.round(3).tolist()}
                         for t, x, c in hold],
                "steadiest": i,
            }))
            cv2.imwrite(str(out / f"{name}.jpg"), best_frame)
            print(f"[capture] {name}: {len(hold)} readings -> {out / (name + '.jpg')}")
    finally:
        camera.stop()
        cv2.destroyAllWindows()
    (out / "poses.json").write_text(json.dumps(results, indent=1))
    print(f"[capture] done: {out / 'poses.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
