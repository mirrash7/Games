"""The arcade's look, shared by every game so they feel like one product.

Use these instead of picking colours and fonts ad hoc. The style in short: a
dark, translucent rounded panel; bold DUPLEX titles in the gold accent with a
dark outline; light SIMPLEX body text; cyan for "active"/positive feedback;
everything sized to read from ~2 m. Colours are BGR (OpenCV order).
"""

from __future__ import annotations

import cv2
import numpy as np

from .overlay import outlined_text

# --- palette (BGR) ---
WHITE = (255, 255, 255)
DIM = (175, 175, 190)  # secondary text, hints
ACCENT = (70, 210, 255)  # gold: titles, highlights, the score
CYAN = (255, 220, 110)  # active / hovered / positive feedback
PANEL = (28, 24, 22)  # translucent panel fill
GOOD = (140, 235, 150)  # success, "hand found"
WARN = (90, 190, 255)  # guidance the player must act on ("STEP BACK")
DANGER = (70, 70, 255)  # game over, lost life

# --- type ---
TITLE_FONT = cv2.FONT_HERSHEY_DUPLEX
BODY_FONT = cv2.FONT_HERSHEY_SIMPLEX


def panel(img, rect, colour=PANEL, alpha=0.78, border=None, radius=18, thickness=2) -> None:
    """Translucent rounded rectangle, blended only inside its own bounds."""
    x0, y0, x1, y1 = (int(v) for v in rect)
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    over = roi.copy()
    r = min(radius, (x1 - x0) // 2, (y1 - y0) // 2)
    bw, bh = x1 - x0, y1 - y0
    cv2.rectangle(over, (r, 0), (bw - r, bh), colour, -1)
    cv2.rectangle(over, (0, r), (bw, bh - r), colour, -1)
    for cx, cy in ((r, r), (bw - r, r), (r, bh - r), (bw - r, bh - r)):
        cv2.circle(over, (cx, cy), r, colour, -1, cv2.LINE_AA)
    cv2.addWeighted(over, alpha, roi, 1.0 - alpha, 0, roi)
    if border is not None:
        pts = [((x0 + r, y0), (x1 - r, y0)), ((x0 + r, y1), (x1 - r, y1)),
               ((x0, y0 + r), (x0, y1 - r)), ((x1, y0 + r), (x1, y1 - r))]
        for a, b in pts:
            cv2.line(img, a, b, border, thickness, cv2.LINE_AA)
        for (cx, cy), ang in (((x0 + r, y0 + r), 180), ((x1 - r, y0 + r), 270),
                              ((x1 - r, y1 - r), 0), ((x0 + r, y1 - r), 90)):
            cv2.ellipse(img, (cx, cy), (r, r), 0, ang, ang + 90, border, thickness, cv2.LINE_AA)


def centred_text(img, text, centre_x, y, scale, colour, thick=2, font=cv2.FONT_HERSHEY_DUPLEX, shadow=True):
    (tw, _), _ = cv2.getTextSize(text, font, scale, thick)
    x = int(centre_x - tw / 2)
    if shadow:
        outlined_text(img, text, (x, y), font, scale, colour, thick, width=2 if scale >= 1.0 else 1)
    else:
        cv2.putText(img, text, (x, y), font, scale, colour, thick, cv2.LINE_AA)


def wrap(text: str, width_px: int, scale: float, font=cv2.FONT_HERSHEY_SIMPLEX) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if cv2.getTextSize(trial, font, scale, 1)[0][0] <= width_px or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines
