"""Skeleton and HUD drawing."""

from __future__ import annotations

import cv2
import numpy as np

from .config import SKELETON
from .inference import Pose

_HUD_BG = (24, 24, 24)
_HUD_FG = (235, 235, 235)


def draw_pose(frame: np.ndarray, pose: Pose, min_conf: float = 0.5) -> None:
    xy = pose.xy
    vis = pose.confidence >= min_conf

    for a, b, colour in SKELETON:
        if vis[a] and vis[b]:
            cv2.line(
                frame,
                (int(xy[a][0]), int(xy[a][1])),
                (int(xy[b][0]), int(xy[b][1])),
                colour,
                2,
                cv2.LINE_AA,
            )

    for i in range(len(xy)):
        if not vis[i]:
            continue
        centre = (int(xy[i][0]), int(xy[i][1]))
        cv2.circle(frame, centre, 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, centre, 5, (0, 0, 0), 1, cv2.LINE_AA)


def draw_box(frame: np.ndarray, pose: Pose) -> None:
    if pose.box is None:
        return
    x1, y1, x2, y2 = (int(v) for v in pose.box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"{pose.score:.2f}",
        (x1, max(14, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )


def draw_hud(frame: np.ndarray, lines: list[str], origin: str = "top-left") -> None:
    """Stats panel. Games draw their own HUD top-left, so they ask for bottom-left."""
    if not lines:
        return
    pad, lh = 8, 20
    width = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] for t in lines)
    height = pad * 2 + lh * len(lines)
    y0 = frame.shape[0] - height - 34 if origin == "bottom-left" else 0
    panel = frame[y0 : y0 + height, 0 : width + pad * 2]
    cv2.addWeighted(panel, 0.35, np.full_like(panel, _HUD_BG), 0.65, 0, panel)
    for i, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (pad, y0 + pad + lh * (i + 1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            _HUD_FG,
            1,
            cv2.LINE_AA,
        )


def draw_banner(frame: np.ndarray, text: str, sub: str | None = None) -> None:
    """Centred warning that must be seen - e.g. the camera has stopped."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 1.0, 2)
    sw = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0] if sub else 0
    bw = max(tw, sw) + 48
    bh = th + (40 if sub else 24) + 8
    x0, y0 = w // 2 - bw // 2, h // 2 - bh // 2
    cv2.rectangle(frame, (x0, y0), (x0 + bw, y0 + bh), (20, 20, 26), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + bw, y0 + bh), (60, 160, 255), 2)
    cv2.putText(frame, text, (w // 2 - tw // 2, y0 + th + 14), font, 1.0,
                (90, 190, 255), 2, cv2.LINE_AA)
    if sub:
        cv2.putText(frame, sub, (w // 2 - sw // 2, y0 + th + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 210), 1, cv2.LINE_AA)


def outlined_text(
    frame: np.ndarray,
    text: str,
    org: tuple[int, int],
    font: int,
    scale: float,
    colour: tuple[int, int, int],
    thickness: int = 2,
    outline: tuple[int, int, int] = (12, 12, 12),
    width: int = 2,
) -> None:
    """Text with a dark outline that stays aligned to the glyphs.

    The usual trick - draw the same text thicker in black, then thinner in
    colour - breaks on OpenCV 5, where thickness sets font *weight* and heavier
    text is wider. The outline then drifts right along the string and shows as
    ghost letters past the end. Stamping the outline at the same weight in
    eight directions keeps every glyph registered.
    """
    x, y = org
    for dx in (-width, 0, width):
        for dy in (-width, 0, width):
            if dx or dy:
                cv2.putText(frame, text, (x + dx, y + dy), font, scale, outline, thickness, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale, colour, thickness, cv2.LINE_AA)
