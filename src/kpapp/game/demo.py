"""A minimal game that exercises every part of the control path.

Steer a paddle with your hands, raise them to widen it, and catch falling
blocks. Deliberately simple: its job is to prove that pose -> control ->
simulation -> render holds together in real time.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import cv2
import numpy as np

from ..controls import ControlState
from .base import Game


@dataclass
class Block:
    x: float
    y: float
    speed: float
    size: int


class CatchGame(Game):
    name = "catch"

    def __init__(self, size: tuple[int, int], **options: object) -> None:
        super().__init__(size, **options)
        self.reset()

    def reset(self) -> None:
        self.paddle_x = self.width / 2
        self.paddle_w = self.width * 0.12
        self.paddle_y = self.height - 60
        self.blocks: list[Block] = []
        self.score = 0
        self.missed = 0
        self._spawn_timer = 0.0
        self._flash = 0.0

    def update(self, controls: ControlState, dt: float) -> None:
        dt = min(dt, 0.1)  # clamp so a stall doesn't teleport everything

        if controls.present:
            # Steer is a velocity, not a position: holding a lean keeps moving.
            self.paddle_x += controls.steer * self.width * 1.6 * dt
            target_w = self.width * (0.12 + 0.10 * controls.throttle)
            self.paddle_w += (target_w - self.paddle_w) * min(1.0, 8.0 * dt)
        self.paddle_x = float(np.clip(self.paddle_x, self.paddle_w / 2, self.width - self.paddle_w / 2))

        self._spawn_timer -= dt
        if self._spawn_timer <= 0:
            self._spawn_timer = random.uniform(0.6, 1.3)
            self.blocks.append(
                Block(
                    x=random.uniform(40, self.width - 40),
                    y=-20.0,
                    speed=random.uniform(0.25, 0.45) * self.height,
                    size=random.randint(18, 30),
                )
            )

        left, right = self.paddle_x - self.paddle_w / 2, self.paddle_x + self.paddle_w / 2
        alive: list[Block] = []
        for b in self.blocks:
            b.y += b.speed * dt
            if b.y >= self.paddle_y - b.size / 2 and b.y <= self.paddle_y + 24:
                if left <= b.x <= right:
                    self.score += 1
                    self._flash = 0.25
                    continue
            if b.y > self.height + 40:
                self.missed += 1
                continue
            alive.append(b)
        self.blocks = alive
        self._flash = max(0.0, self._flash - dt)

    def render(self, frame: np.ndarray) -> None:
        for b in self.blocks:
            top_left = (int(b.x - b.size / 2), int(b.y - b.size / 2))
            bottom_right = (int(b.x + b.size / 2), int(b.y + b.size / 2))
            cv2.rectangle(frame, top_left, bottom_right, (80, 220, 255), -1, cv2.LINE_AA)
            cv2.rectangle(frame, top_left, bottom_right, (20, 20, 20), 1, cv2.LINE_AA)

        colour = (120, 255, 160) if self._flash > 0 else (255, 170, 60)
        cv2.rectangle(
            frame,
            (int(self.paddle_x - self.paddle_w / 2), int(self.paddle_y)),
            (int(self.paddle_x + self.paddle_w / 2), int(self.paddle_y + 18)),
            colour,
            -1,
            cv2.LINE_AA,
        )

        text = f"score {self.score}   missed {self.missed}"
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(
            frame,
            text,
            (self.width - tw - 16, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
