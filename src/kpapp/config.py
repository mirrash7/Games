"""Tunable settings for the real-time loop.

Everything that affects latency lives here so it can be swept from the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    # --- camera ---
    camera_index: int = 0
    capture_width: int = 1280
    capture_height: int = 720
    capture_fps: int = 60
    mirror: bool = True  # selfie view: user's right hand appears on the right

    # --- model ---
    # Trained at 576x576. Must be a multiple of 24 (patch_size 12 * 2 windows);
    # 504 and 336 are the useful faster steps, 392/448/560 are rejected.
    resolution: int = 576
    threshold: float = 0.5
    device: str | None = None  # None -> auto (mps > cuda > cpu)
    half: bool = False  # fp16; MPS support for this varies, off by default
    compile_model: bool = True  # run optimize_for_inference() when available

    # --- pipeline ---
    infer_every: int = 1  # run inference on at most every Nth captured frame
    # EMA on every keypoint before any game sees it (0 = off, ->1 = heavy).
    # Off by default: simulated through the real pipeline, 0.5 added ~36 ms of
    # lag on battery (~28 ms plugged in) - the single biggest software delay -
    # while every consumer already filters its own input (HandTracker's One
    # Euro filter, FlapDetector's time window). Raw poses wobble ~4-5 px when
    # held still instead of ~2 px. Raise it only for a game reading raw
    # keypoints directly that needs them steadier.
    smoothing: float = 0.0
    # Velocity projection toward render time. 0 disables; ~0.7 cancels most of
    # the inference lag without overshooting on direction changes.
    extrapolation: float = 0.7
    max_lead: float = 0.12  # seconds; ceiling on how far poses may be projected
    max_people: int = 1  # only track the N highest-scoring poses

    verbose: bool = False  # show library logs during model setup

    # --- display ---
    show_fps: bool = True
    # Game and render rate. Runs on its own clock, decoupled from the camera,
    # so animation stays smooth and a stalled camera cannot freeze the game.
    display_fps: float = 60.0
    window_name: str = "KP"


# COCO-17 keypoint layout, the standard RF-DETR keypoint output ordering.
KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

KP = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

# (a, b, colour) edges for the skeleton overlay, BGR.
SKELETON: tuple[tuple[int, int, tuple[int, int, int]], ...] = (
    (KP["left_shoulder"], KP["right_shoulder"], (255, 200, 0)),
    (KP["left_shoulder"], KP["left_elbow"], (0, 220, 255)),
    (KP["left_elbow"], KP["left_wrist"], (0, 220, 255)),
    (KP["right_shoulder"], KP["right_elbow"], (0, 140, 255)),
    (KP["right_elbow"], KP["right_wrist"], (0, 140, 255)),
    (KP["left_shoulder"], KP["left_hip"], (255, 200, 0)),
    (KP["right_shoulder"], KP["right_hip"], (255, 200, 0)),
    (KP["left_hip"], KP["right_hip"], (255, 200, 0)),
    (KP["left_hip"], KP["left_knee"], (120, 255, 120)),
    (KP["left_knee"], KP["left_ankle"], (120, 255, 120)),
    (KP["right_hip"], KP["right_knee"], (60, 200, 60)),
    (KP["right_knee"], KP["right_ankle"], (60, 200, 60)),
    (KP["nose"], KP["left_eye"], (200, 160, 255)),
    (KP["nose"], KP["right_eye"], (200, 160, 255)),
    (KP["left_eye"], KP["left_ear"], (200, 160, 255)),
    (KP["right_eye"], KP["right_ear"], (200, 160, 255)),
)
