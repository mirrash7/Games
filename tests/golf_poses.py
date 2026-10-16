"""Synthetic golfers for the Raccoon Golf tests (shared by test_golf_swing.py and test_golf.py).

A golfer faces the camera. Their grip moves on keyframed paths in the body
frame: `u` sideways from the shoulder midpoint and `h` above the shoulder line,
both in shoulder widths (sw). Poses come out the way the real pipeline
delivers them: from a model running at 15-30 Hz (each pose held until the
next, as `controls.step_pose` is), late by a fixed lag, with ~5 px of keypoint
noise, and elbows mostly missing (on recorded play the elbow was gone in 54%
of frames with a visible wrist).
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np

from kpapp.config import KP
from kpapp.inference import Pose

GAME_HZ = 60.0
SW = 200.0  # px: a player ~2 m from a 1280x720 camera
CENTRE = (640.0, 330.0)  # shoulder midpoint, px
RADIUS = 1.35  # sw: shoulder line to the grip with the arms hanging


def grip_pose(u, h, sw=SW, centre=CENTRE, noise=0.0, rng=None, gap=0.15, elbows=False,
              hands=2, shoulders=True) -> Pose:
    """A pose whose grip (midpoint of the wrists) is at (u, h) sw; None u = no hands."""
    rng = rng if rng is not None else np.random.default_rng(0)
    xy = np.zeros((17, 2), np.float32)
    c = np.zeros(17, np.float32)
    cx, cy = centre
    pts = {"nose": (cx, cy - 0.7 * sw)}
    if shoulders:
        pts["left_shoulder"] = (cx + sw / 2, cy)
        pts["right_shoulder"] = (cx - sw / 2, cy)
    if u is not None:
        gx, gy = cx + u * sw, cy - h * sw
        if hands >= 1:
            pts["left_wrist"] = (gx + gap * sw / 2, gy)
        if hands >= 2:
            pts["right_wrist"] = (gx - gap * sw / 2, gy)
        if elbows:
            # Elbows above and outside the wrists, as with a relaxed grip.
            pts["left_elbow"] = (gx + 0.35 * sw, gy - 0.55 * sw)
            pts["right_elbow"] = (gx - 0.35 * sw, gy - 0.55 * sw)
    for name, p in pts.items():
        xy[KP[name]] = np.asarray(p, np.float32) + (rng.normal(0, noise, 2) if noise else 0.0)
        c[KP[name]] = 0.9
    return Pose(xy=xy, confidence=c, score=0.9)


def path(*kf):
    """(u, h)(t) through (t, u, h) keyframes with cosine easing between them."""

    def at(t):
        if t <= kf[0][0]:
            return kf[0][1], kf[0][2]
        for (t0, u0, h0), (t1, u1, h1) in pairwise(kf):
            if t <= t1:
                e = (1 - math.cos(math.pi * (t - t0) / (t1 - t0))) / 2
                return u0 + (u1 - u0) * e, h0 + (h1 - h0) * e
        return kf[-1][1], kf[-1][2]

    return at


def arc(angle: float, radius: float = RADIUS, side: int = 1) -> tuple[float, float]:
    """The grip on a circle round the shoulders: 0 = hanging straight down,
    +angle swings out to `side` and up (radians)."""
    return side * radius * math.sin(angle), -radius * math.cos(angle)


def top_angle_for(power: float, radius: float = RADIUS) -> float:
    """The backswing angle whose top height gives `power` (swing.py's formula)."""
    from kpapp.game.golf.swing import FULL_H

    h_addr = -radius
    h = h_addr + power * (FULL_H - h_addr)
    return math.acos(max(-1.0, min(1.0, -h / radius)))


def swing_path(power=1.0, side=1, start=0.8, back=0.8, pause=0.15, down=0.30, follow=0.35,
               address_hold=0.8, radius=RADIUS, low_cross=0.0):
    """A full swing: address, backswing to `power`, a fast downswing through
    the bottom, a follow-through to the other side. Returns ((u, h)(t), t_impact).

    low_cross raises the arc's bottom (sw) for a thin strike.
    """
    top = top_angle_for(power, radius)
    t0 = start + address_hold
    t_top = t0 + back
    t_down = t_top + pause
    t_end = t_down + down + follow

    def at(t):
        if t < t0:
            u, h = arc(0.0, radius, side)
        elif t < t_top:
            e = (1 - math.cos(math.pi * (t - t0) / back)) / 2
            u, h = arc(top * e, radius, side)
        elif t < t_down:
            u, h = arc(top, radius, side)
        elif t < t_end:
            # Accelerating through the ball: angle goes top -> -top*0.9.
            k = (t - t_down) / (down + follow)
            e = k * k * (3 - 2 * k)
            a = top + (-0.9 * top - top) * e
            u, h = arc(a, radius, side)
            h += low_cross * max(0.0, math.cos(a)) ** 4
        else:
            u, h = arc(-0.9 * top, radius, side)
        return u, h

    # Impact: the angle passes 0 when e = top / (1.9 top) = 0.526.
    e = 1 / 1.9
    # Solve k*k*(3-2k) = e by bisection.
    lo, hi = 0.0, 1.0
    for _ in range(40):
        m = (lo + hi) / 2
        if m * m * (3 - 2 * m) < e:
            lo = m
        else:
            hi = m
    return at, t_down + lo * (down + follow)


def model_clock(model_hz, duration, rng, jitter=0.25, phase=0.0):
    times, t = [], phase - 1.0 / model_hz
    while t < duration + 1.0:
        times.append(t)
        t += (1.0 / model_hz) * (1.0 + rng.uniform(-jitter, jitter))
    return np.asarray(times)


def stream(motion, duration, model_hz=20.0, lag=0.0, noise=0.0, seed=0, elbows_every=0, gap=0.15,
           hands=None, lost=None, sw=SW, centre=CENTRE):
    """Yield (t, pose) at GAME_HZ for a grip moving along motion(t) -> (u, h) or None.

    The model publishes at model_hz (jittered); each pose shows the player as
    they were `lag` seconds before it is delivered, and is held until the next.
    elbows_every=3 shows the elbows on one model frame in three.
    hands(t) -> 0/1/2 visible wrists; lost(t) -> True feeds no pose at all.
    """
    rng = np.random.default_rng(seed)
    clock = model_clock(model_hz, duration, rng) if model_hz else None
    held_key, held = None, None
    n = 0
    for i in range(int(duration * GAME_HZ)):
        t = i / GAME_HZ
        ts = t if clock is None else float(clock[np.searchsorted(clock, t + 1e-9) - 1])
        if lost is not None and lost(ts):
            yield t, None
            continue
        if ts != held_key:
            held_key = ts
            n += 1
            m = motion(ts - lag)
            u, h = (None, None) if m is None else m
            held = grip_pose(u, h, sw=sw, centre=centre, noise=noise, rng=rng, gap=gap,
                             elbows=bool(elbows_every) and n % elbows_every == 0,
                             hands=hands(ts) if hands is not None else 2)
        yield t, held


def run_detector(det, motion, duration, **kw):
    """Feed a detector; return the events."""
    events = []
    for t, pose in stream(motion, duration, **kw):
        ev = det.update(pose, t)
        if ev is not None:
            events.append(ev)
    return events
