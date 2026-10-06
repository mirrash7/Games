"""Shared sprite drawing for every game. All functions draw in place on a BGR frame.

Everything clips at the frame edges (a sprite half off-screen draws its visible
part) and blends only the overlapping region, never the whole frame - at 60 FPS
a full-frame float blend costs ~7 ms, a third of the frame budget.
"""

from __future__ import annotations

import numpy as np


def alpha_composite(dst_bgr: np.ndarray, src_bgra: np.ndarray, x: int, y: int) -> None:
    """Composite a BGRA sprite onto a BGR frame in place at (x, y).

    Clipped at all four edges, so a sprite hanging off the frame draws its
    visible part instead of crashing or wrapping. Called every frame, so the
    blend is a single vectorised pass over the overlapping region only.
    """
    fh, fw = dst_bgr.shape[:2]
    sh, sw = src_bgra.shape[:2]

    x, y = int(x), int(y)
    # Overlap in frame coords, then the matching window in sprite coords.
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + sw, fw), min(y + sh, fh)
    if x0 >= x1 or y0 >= y1:
        return

    src = src_bgra[y0 - y : y1 - y, x0 - x : x1 - x]
    dst = dst_bgr[y0:y1, x0:x1]

    if src.shape[2] == 3:  # tolerate an opaque sprite
        dst[:] = src
        return

    # uint16 fixed point rather than float32: same result to within 1 LSB and
    # measurably cheaper, because this is memory bound and uint16 moves half
    # the bytes float32 does.
    a = src[:, :, 3].astype(np.uint16)[:, :, None]
    dst[:] = (
        (dst.astype(np.uint16) * (255 - a) + src[:, :, :3].astype(np.uint16) * a + 127) // 255
    ).astype(np.uint8)


def additive_composite(
    dst_bgr: np.ndarray, src_bgra: np.ndarray, x: int, y: int, gain: float = 1.0
) -> None:
    """Add a BGRA sprite's premultiplied colour onto a BGR frame (glows, sparks)."""
    fh, fw = dst_bgr.shape[:2]
    sh, sw = src_bgra.shape[:2]

    x, y = int(x), int(y)
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + sw, fw), min(y + sh, fh)
    if x0 >= x1 or y0 >= y1:
        return

    src = src_bgra[y0 - y : y1 - y, x0 - x : x1 - x]
    dst = dst_bgr[y0:y1, x0:x1]
    alpha = src[:, :, 3].astype(np.float32) * (gain / 255.0)
    add = src[:, :, :3].astype(np.float32) * alpha[:, :, None]
    dst[:] = np.clip(dst.astype(np.float32) + add, 0, 255).astype(dst.dtype)


def blit(
    frame: np.ndarray,
    sprite: np.ndarray,
    centre,
    alpha: float = 1.0,
    tint: tuple[int, int, int] | None = None,
) -> None:
    """Draw a BGRA sprite centred on `centre`, with optional fade and flat tint.

    Fade and tint are applied inside the blend instead of on a copy of the
    sprite: a busy Fruit Ninja board has ~35 fading or tinted sprites a frame,
    and copying each one cost more than everything else combined. `tint`
    replaces the sprite's colour entirely - draw tintable sprites in white.
    """
    h, w = sprite.shape[:2]
    x = int(centre[0] - w / 2)
    y = int(centre[1] - h / 2)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return
    src = sprite[y0 - y : y1 - y, x0 - x : x1 - x]
    dst = frame[y0:y1, x0:x1]
    a = src[:, :, 3].astype(np.uint16)
    if alpha < 0.999:
        # Clamped: the blend relies on a <= 255 so (255 - a) cannot wrap.
        a = (a * min(1.0, max(0.0, alpha))).astype(np.uint16)
    a = a[:, :, None]
    colour = (np.array(tint, np.uint16)[None, None, :] if tint is not None
              else src[:, :, :3].astype(np.uint16))
    dst[:] = ((dst.astype(np.uint16) * (255 - a) + colour * a) // 255).astype(np.uint8)
