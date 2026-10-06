"""Load the procedurally generated Hole in the Wall art.

The PNGs under ``assets/holeinwall/generated/`` are produced by
``tools/generate_holeinwall_assets.py``. Nothing here draws anything; it just
decodes the files once, caches them, and offers the one per-frame primitive
every caller needs - :func:`alpha_composite`.

Decoding a megabyte of PNG mid-game would blow the frame budget, so
:func:`load_assets` is called at startup and the result is held in a
module-level cache.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import cv2
import numpy as np

# .../src/kpapp/game/holeinwall/assets.py -> repo root
ASSET_DIR = Path(__file__).resolve().parents[4] / "assets" / "holeinwall" / "generated"

_GENERATE_HINT = "uv run python tools/generate_holeinwall_assets.py"


@dataclass(frozen=True)
class Assets:
    """Every generated image, decoded. BGR where opaque, BGRA where not."""

    arena_bg: np.ndarray  # 1280x720 BGR - studio behind the wall
    wall_panel: np.ndarray  # 1024x1024 BGR - yellow surface, scale to taste
    wall_edge: np.ndarray  # BGRA chrome strip for the wall's outer rim
    hole_glow: np.ndarray  # 256x256 BGRA radial glow, composite additively
    particle: np.ndarray  # 64x64 BGRA white spark, tint at runtime
    badge_pass: np.ndarray  # BGRA "PERFECT!"
    badge_fail: np.ndarray  # BGRA "SPLASH!"
    hud_panel: np.ndarray  # BGRA rounded glossy panel
    countdown_ring: np.ndarray  # 256x256 BGRA ring


# field name -> file name
_FILES = {f.name: f"{f.name}.png" for f in fields(Assets)}

_CACHE: Assets | None = None


class AssetsMissingError(RuntimeError):
    """Raised when the generated art is not on disk."""


def _load_one(name: str, directory: Path) -> np.ndarray:
    path = directory / _FILES[name]
    if not path.is_file():
        raise AssetsMissingError(
            f"missing Hole in the Wall asset: {path}\n"
            f"Generate the art first:\n    {_GENERATE_HINT}"
        )
    # IMREAD_UNCHANGED keeps the alpha channel; without it OpenCV drops it.
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise AssetsMissingError(
            f"could not decode Hole in the Wall asset: {path}\n"
            f"The file may be truncated - regenerate it:\n    {_GENERATE_HINT}"
        )
    return img


def load_assets(directory: Path | None = None, *, reload: bool = False) -> Assets:
    """Decode every asset once and cache it.

    Raises :class:`AssetsMissingError` naming the generator script if a file is
    absent, which is the common first-run failure.
    """
    global _CACHE
    if _CACHE is not None and not reload and directory is None:
        return _CACHE

    src = ASSET_DIR if directory is None else Path(directory)
    if not src.is_dir():
        raise AssetsMissingError(
            f"Hole in the Wall asset directory not found: {src}\n"
            f"Generate the art first:\n    {_GENERATE_HINT}"
        )

    loaded = Assets(**{name: _load_one(name, src) for name in _FILES})
    if directory is None:
        _CACHE = loaded
    return loaded


# Moved to kpapp.gfx so games do not depend on each other; re-exported here so
# existing imports keep working.
from ...gfx import additive_composite, alpha_composite  # noqa: E402,F401
