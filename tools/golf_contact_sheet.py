#!/usr/bin/env python3
"""Render Raccoon Golf's key moments to one contact sheet, with the real art.

    uv run python tools/golf_contact_sheet.py [--out golf_sheet.png] [--full DIR]

Seeded and scripted (synthetic poses, scripted shots), so the same command
draws the same sheet: use it to review the look after a change, and to compare
the browser port frame by frame. --full also writes each moment at 1280x720.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from kpapp import theme
from kpapp.config import KP
from kpapp.controls import ControlState
from kpapp.game.golf import physics as P
from kpapp.game.golf.game import GolfGame, Phase
from kpapp.game.golf.swing import SwingEvent
from kpapp.inference import Pose
from kpapp.shell import Shell

W, H = 1280, 720
DT = 1 / 60
SW, CX, CY = 200.0, 640.0, 330.0  # the synthetic player's shoulder width and shoulder midpoint (px)


def camera_frame() -> np.ndarray:
    """A stand-in camera picture: a dim room, so the picture-in-picture has something in it."""
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    img = (np.array([70, 60, 55], np.float32) * (1 - y) + np.array([40, 35, 32], np.float32) * y)
    img = np.repeat(img, W, axis=1)
    out = img.astype(np.uint8)
    cv2.ellipse(out, (int(CX), int(CY + 260)), (150, 300), 0, 0, 360, (90, 80, 120), -1, cv2.LINE_AA)
    cv2.circle(out, (int(CX), int(CY - 140)), 80, (120, 140, 180), -1, cv2.LINE_AA)
    return out


def grip(u, h, apart=0.15) -> ControlState:
    """Controls for a player whose grip is at (u, h) shoulder widths from the shoulder midpoint."""
    xy = np.zeros((17, 2), np.float32)
    c = np.zeros(17, np.float32)
    pts = {"left_shoulder": (CX + SW / 2, CY), "right_shoulder": (CX - SW / 2, CY), "nose": (CX, CY - 0.7 * SW)}
    if u is not None:
        gx, gy = CX + u * SW, CY - h * SW
        pts["left_wrist"] = (gx + apart * SW / 2, gy)
        pts["right_wrist"] = (gx - apart * SW / 2, gy)
    for n, p in pts.items():
        xy[KP[n]] = p
        c[KP[n]] = 0.9
    pose = Pose(xy=xy, confidence=c, score=0.9)
    return ControlState(present=True, pose=pose, step_pose=pose)


def arc(a: float, r: float = 1.35):
    return r * math.sin(a), -r * math.cos(a)


def run(game, controls, seconds):
    for _ in range(int(round(seconds / DT))):
        game.update(controls(game.clock) if callable(controls) else controls, DT)


def snap(game, frame) -> np.ndarray:
    img = frame.copy()
    game.render(img)
    return img


def swing(game, top, frame, shots):
    """Address, backswing to angle `top`, swing through. Grabs the backswing frame."""
    run(game, grip(*arc(0)), 0.7)
    t0 = game.clock
    run(game, lambda t: grip(*arc(top * min(1.0, (t - t0) / 0.7))), 0.85)
    shots.append(snap(game, frame))
    t1 = game.clock
    run(game, lambda t: grip(*arc(top - (top + 2.0) * min(1.0, (t - t1) / 0.45))), 0.5)


def build_sheet(seed: int = 3) -> tuple[list[np.ndarray], list[str]]:
    frame = camera_frame()
    shots, labels = [], []
    game = GolfGame((W, H), seed=seed)
    run(game, grip(None, None), 1.1)
    shots.append(snap(game, frame))
    labels.append("1. Hole intro")
    run(game, grip(None, None), 2.2)
    run(game, grip(*arc(0)), 0.7)
    shots.append(snap(game, frame))
    labels.append("2. Tee shot: READY at address")
    # Point at the aim buttons with one hand (hands apart).
    run(game, grip(None, None), 0.3)
    swing(game, 2.45, frame, shots)
    labels.append("3. Backswing into the overswing zone")
    run(game, grip(None, None), 1.3)
    shots.append(snap(game, frame))
    labels.append("4. Ball in flight")
    while game.phase is Phase.FLIGHT:
        run(game, grip(None, None), 0.1)
    run(game, grip(None, None), 0.25)
    shots.append(snap(game, frame))
    labels.append(f"5. Result: {game.message.text if game.message else '-'}")

    # Hole 2: into the water.
    w = GolfGame((W, H), seed=seed)
    w._start_hole(1)
    w.phase, w.phase_t = Phase.AIM, 0.0
    w._hit(SwingEvent(t=0, power=0.55, quality=1.0, side=1, speed=9, practice=False))
    while w.phase is Phase.FLIGHT:
        run(w, grip(None, None), 0.1)
    run(w, grip(None, None), 0.4)
    shots.append(snap(w, frame))
    labels.append("6. Water hazard")

    # Hole 1, on the green: putter, slope grid, READY.
    g = GolfGame((W, H), seed=seed)
    g.phase = Phase.AIM
    px, py = g.hole.pin
    g.ball, g.lie = (px - 3.0, py - 6.0), "green"
    g._setup_shot()
    run(g, grip(*arc(0)), 0.7)
    shots.append(snap(g, frame))
    labels.append("7. On the green: putter, slope grid")
    # Sink it: find the power that holes out from here.
    best = None
    for off in range(0, 41):
        for sign in (1, -1):
            g.aim_offset = sign * math.radians(off * 0.5)
            for k in range(1, 200):
                f = P.simulate(g.hole, g.ball, P.Shot(P.PUTTER, k / 200, g.heading, "green"))
                if f.state == "holed":
                    best = k / 200
                    break
            if best is not None:
                break
        if best is not None:
            break
    if best is None:
        best = 0.4
    g.strokes = 2
    g._hit(SwingEvent(t=0, power=best, quality=1.0, side=1, speed=2, practice=False))
    while g.phase is Phase.FLIGHT:
        run(g, grip(None, None), 0.1)
    run(g, grip(None, None), 0.5)
    shots.append(snap(g, frame))
    labels.append(f"8. Hole out: {g.message.text if g.message else '-'}")
    run(g, grip(None, None), 2.2)
    shots.append(snap(g, frame))
    labels.append("9. Scorecard")

    o = GolfGame((W, H), seed=seed)
    for row, s in zip(o.card, (4, 2, 6)):
        row.strokes, row.done = s, True
    from kpapp.game.golf.game import hole_points
    for row in o.card:
        row.points = hole_points(row.strokes, row.par)
    o.phase = Phase.GAME_OVER
    shots.append(snap(o, frame))
    labels.append("10. Game over")

    # Aim / club buttons (hands apart, pointing) and the welcome screen.
    a = GolfGame((W, H), seed=seed)
    a.phase = Phase.AIM
    xy = np.zeros((17, 2), np.float32)
    c = np.zeros(17, np.float32)
    for n, p in {"left_shoulder": (CX + SW / 2, CY), "right_shoulder": (CX - SW / 2, CY),
                 "left_wrist": (1010, 455), "left_elbow": (900, 470), "right_wrist": (CX - 140, CY + 280)}.items():
        xy[KP[n]] = p
        c[KP[n]] = 0.9
    pose = Pose(xy=xy, confidence=c, score=0.9)
    run(a, ControlState(present=True, pose=pose, step_pose=pose), 0.45)
    shots.append(snap(a, frame))
    labels.append("11. Adjusting aim (hands apart)")

    from kpapp.game import REGISTRY
    sh = Shell((W, H), REGISTRY)
    menu = frame.copy()
    sh.render(menu)
    shots.append(menu)
    labels.append("12. Welcome screen, four games")
    tut = Shell((W, H), REGISTRY, selected="golf")
    tut.choose("golf")
    tut.tutorial_t = 1.6
    card = frame.copy()
    tut.render(card)
    shots.append(card)
    labels.append("13. How to play")
    return shots, labels


def compose(shots, labels, cols: int = 2, scale: float = 0.5) -> np.ndarray:
    tw, th = int(W * scale), int(H * scale)
    rows = -(-len(shots) // cols)
    pad, lab = 12, 34
    sheet = np.full((rows * (th + lab + pad) + pad, cols * (tw + pad) + pad, 3), (34, 28, 30), np.uint8)
    for i, (img, text) in enumerate(zip(shots, labels)):
        r, cc = divmod(i, cols)
        x, y = pad + cc * (tw + pad), pad + r * (th + lab + pad)
        sheet[y + lab:y + lab + th, x:x + tw] = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        cv2.putText(sheet, text, (x + 4, y + 24), theme.BODY_FONT, 0.7, (235, 235, 235), 1, cv2.LINE_AA)
    return sheet


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("golf_sheet.png"))
    ap.add_argument("--full", type=Path, default=None, help="also write each moment full size here")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    shots, labels = build_sheet(args.seed)
    cv2.imwrite(str(args.out), compose(shots, labels))
    if args.full:
        args.full.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(shots):
            cv2.imwrite(str(args.full / f"golf_{i + 1:02d}.png"), img)
    print(f"wrote {args.out} ({len(shots)} moments)")


if __name__ == "__main__":
    main()
