"""Game interface.

Games get a fixed contract: a ControlState per frame plus real elapsed time,
and they draw straight onto the camera frame. Keeping games ignorant of poses
and models means a new game is one file with two methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..controls import ControlState


class Game(ABC):
    name: str = "game"
    # Shown on the welcome screen. Games without a title stay off the menu
    # (they can still be launched directly with --game).
    title: str = ""
    blurb: str = ""
    menu: bool = True  # False keeps a game off the welcome screen (still runnable via --game)

    def __init__(self, size: tuple[int, int], **options: object) -> None:
        self.width, self.height = size
        # Context from the app (e.g. mirrored=True). Games take what they need
        # and ignore the rest, so adding context does not break every game.
        self.options = options

    @abstractmethod
    def update(self, controls: ControlState, dt: float) -> None:
        """Advance simulation by `dt` seconds using this frame's input."""

    @abstractmethod
    def render(self, frame: np.ndarray) -> None:
        """Draw the current state onto `frame` in place."""

    def reset(self) -> None:
        pass

    def on_resume(self) -> None:
        """Called when play resumes after a pause."""
