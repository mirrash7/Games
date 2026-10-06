"""Asset schema for Fruit Ninja.

Sprites are generated procedurally (see tools/generate_fruitninja_*.py) and
described by a manifest so gameplay numbers - collision radius, juice colour,
score value - travel with the art instead of being duplicated in code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ASSET_DIR = Path(__file__).resolve().parents[4] / "assets" / "fruitninja" / "generated"
MANIFEST = ASSET_DIR / "manifest.json"


class AssetsMissingError(RuntimeError):
    pass


@dataclass(frozen=True)
class FruitArt:
    """One fruit: the whole sprite plus the two halves it becomes."""

    name: str
    whole: np.ndarray  # BGRA
    half_a: np.ndarray  # BGRA, the half above the cut
    half_b: np.ndarray  # BGRA, the half below the cut
    juice: tuple[int, int, int]  # BGR, for splatter and the trail flash
    radius: int  # collision radius in pixels at native sprite size
    score: int = 1


@dataclass
class Art:
    fruits: list[FruitArt]
    bomb: np.ndarray
    background: np.ndarray
    splat: np.ndarray  # white, tinted per fruit at runtime
    flash: np.ndarray
    life_full: np.ndarray
    life_lost: np.ndarray


_cache: Art | None = None


def _read(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    if not path.exists():
        raise AssetsMissingError(
            f"missing asset {path.name}. Run:\n"
            f"  uv run python tools/generate_fruitninja_fruit.py\n"
            f"  uv run python tools/generate_fruitninja_scene.py"
        )
    img = cv2.imread(str(path), flags)
    if img is None:
        raise AssetsMissingError(f"could not decode {path}")
    return img


def load_art(*, reload: bool = False) -> Art:
    global _cache
    if _cache is not None and not reload:
        return _cache

    if not MANIFEST.exists():
        raise AssetsMissingError(
            f"missing {MANIFEST}. Run tools/generate_fruitninja_fruit.py first."
        )
    spec = json.loads(MANIFEST.read_text())

    fruits = [
        FruitArt(
            name=f["name"],
            whole=_read(ASSET_DIR / f"fruit_{f['name']}_whole.png"),
            half_a=_read(ASSET_DIR / f"fruit_{f['name']}_half_a.png"),
            half_b=_read(ASSET_DIR / f"fruit_{f['name']}_half_b.png"),
            juice=tuple(int(c) for c in f["juice"]),
            radius=int(f["radius"]),
            score=int(f.get("score", 1)),
        )
        for f in spec["fruits"]
    ]
    _cache = Art(
        fruits=fruits,
        bomb=_read(ASSET_DIR / "bomb.png"),
        background=_read(ASSET_DIR / "background.png", cv2.IMREAD_COLOR),
        splat=_read(ASSET_DIR / "splat.png"),
        flash=_read(ASSET_DIR / "flash.png"),
        life_full=_read(ASSET_DIR / "life_full.png"),
        life_lost=_read(ASSET_DIR / "life_lost.png"),
    )
    return _cache
