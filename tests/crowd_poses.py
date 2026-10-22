"""Synthetic crowds for the PlayerLock tests (tests/test_player.py) and the
browser parity fixture (tools/make_player_parity_fixture.py).

A person is a standing body drawn in the 1280x720 mirrored frame from a few
numbers: where their shoulder midpoint is, their shoulder width in px (sw,
which says how near the camera they stand: ~130 px at 2.5 m, ~200 px at
1.8 m, ~70 px at 5 m), how far their shoulders have turned, and their arms.
Proportions follow COCO keypoints on a real body: head 0.55 sw above the
shoulders, hips 1.45 sw below, ankles 3.4 sw below. Keypoints outside the
frame are invisible, as the model reports them.

Streams come out the way the real pipeline delivers them: results at 15-30 Hz
with jitter, ~4 px keypoint noise, elbows often missing, now and then a missed
detection, sorted by detection score (passers-by often score higher than the
player). A nearer person whose body covers someone's torso hides them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from kpapp.config import KP
from kpapp.inference import Pose

W, H = 1280, 720


@dataclass
class Look:
    """One person at one moment."""

    x: float  # shoulder midpoint, px
    y: float
    sw: float  # shoulder width (front-on), px
    turn: float = 0.0  # radians the shoulders have turned away (golf backswing ~1.2)
    arms: tuple[float, float] = (0.15, 0.15)  # each arm's angle up from hanging, radians
    score: float = 0.8
    visible: bool = True  # False: the model missed them this frame


def body_pose(look: Look, rng=None, noise: float = 0.0, elbow_drop: float = 0.0) -> Pose | None:
    """The 17 keypoints and box of `look`, or None when nothing is in frame."""
    rng = rng if rng is not None else np.random.default_rng(0)
    sw, cx, cy = look.sw, look.x, look.y
    half = 0.5 * sw * max(0.25, math.cos(look.turn))
    pts = {
        "nose": (cx, cy - 0.55 * sw),
        "left_eye": (cx + 0.1 * sw, cy - 0.65 * sw),
        "right_eye": (cx - 0.1 * sw, cy - 0.65 * sw),
        "left_ear": (cx + 0.22 * sw, cy - 0.6 * sw),
        "right_ear": (cx - 0.22 * sw, cy - 0.6 * sw),
        "left_shoulder": (cx + half, cy),
        "right_shoulder": (cx - half, cy),
        "left_hip": (cx + 0.3 * sw, cy + 1.45 * sw),
        "right_hip": (cx - 0.3 * sw, cy + 1.45 * sw),
        "left_knee": (cx + 0.32 * sw, cy + 2.45 * sw),
        "right_knee": (cx - 0.32 * sw, cy + 2.45 * sw),
        "left_ankle": (cx + 0.34 * sw, cy + 3.4 * sw),
        "right_ankle": (cx - 0.34 * sw, cy + 3.4 * sw),
    }
    for side, sgn, ang in (("left", 1.0, look.arms[0]), ("right", -1.0, look.arms[1])):
        sx, sy = pts[f"{side}_shoulder"]
        ex, ey = sx + sgn * 0.75 * sw * math.sin(ang), sy + 0.75 * sw * math.cos(ang)
        a2 = ang * 1.15
        pts[f"{side}_elbow"] = (ex, ey)
        pts[f"{side}_wrist"] = (ex + sgn * 0.7 * sw * math.sin(a2), ey + 0.7 * sw * math.cos(a2))
    xy = np.zeros((17, 2), np.float32)
    conf = np.zeros(17, np.float32)
    for name, (px, py) in pts.items():
        if noise:
            px, py = px + rng.normal(0, noise), py + rng.normal(0, noise)
        i = KP[name]
        xy[i] = (px, py)
        inside = 0.0 <= px < W and 0.0 <= py < H
        dropped = "elbow" in name and elbow_drop and rng.random() < elbow_drop
        conf[i] = 0.9 if inside and not dropped else 0.05
    if not (conf >= 0.5).any():
        return None
    vis = conf >= 0.5
    x0, y0 = xy[vis].min(axis=0)
    x1, y1 = xy[vis].max(axis=0)
    pad = 0.15 * sw
    box = np.array([max(0.0, x0 - pad), max(0.0, y0 - 0.35 * sw), min(W, x1 + pad), min(H, y1 + 0.1 * sw)],
                   np.float32)
    return Pose(xy=xy, confidence=conf, score=look.score, box=box)


def covers(front: Look, back: Look) -> bool:
    """`front` (nearer, so bigger) hides `back`'s torso."""
    if front.sw <= back.sw:
        return False
    return abs(front.x - back.x) < 0.75 * front.sw and front.y - 0.6 * front.sw < back.y + 0.5 * back.sw


def model_clock(rate_hz: float, duration: float, rng, jitter: float = 0.25) -> list[float]:
    """Result timestamps at ~rate_hz with +-jitter of the period."""
    period = 1.0 / rate_hz
    t, out = 0.0, []
    while t < duration:
        out.append(round(t, 6))
        t += period * (1.0 + jitter * (2.0 * rng.random() - 1.0))
    return out


# --- people --------------------------------------------------------------------

Person = Callable[[float], Look | None]


def dancer(x=640.0, sw=130.0, step=1.0, bpm=120.0, score=0.8, y=150.0) -> Person:
    """Steps `step` sw side to side on every beat (quick, eased), arms waving."""
    beat = 60.0 / bpm

    def at(t: float) -> Look:
        b = t / beat
        k, f = math.floor(b), b - math.floor(b)
        e = min(1.0, f / 0.35)  # the step lands in the first third of the beat
        e = e * e * (3 - 2 * e)
        side = lambda n: (n % 2) * 2 - 1  # noqa: E731
        off = side(k - 1) + (side(k) - side(k - 1)) * e
        arms = (1.2 + 1.0 * math.sin(2.1 * t), 1.0 + 1.1 * math.sin(1.7 * t + 1.0))
        return Look(x + 0.5 * step * sw * off, y, sw, arms=arms, score=score)

    return at


def golfer(x=640.0, sw=150.0, score=0.8, period=3.0) -> Person:
    """Address, backswing (shoulders turn ~70 degrees), a fast downswing, repeat."""

    def at(t: float) -> Look:
        f = (t % period) / period
        if f < 0.4:
            turn, arm = 0.0, 0.1
        elif f < 0.7:
            e = (f - 0.4) / 0.3
            turn, arm = 1.2 * e, 0.1 + 2.4 * e
        elif f < 0.78:
            e = (f - 0.7) / 0.08
            turn, arm = 1.2 * (1 - e), 2.5 - 2.4 * e
        else:
            e = min(1.0, (f - 0.78) / 0.1)
            turn, arm = 0.9 * e * (1 - e) * 4 * 0.3, 0.1 + 2.2 * e
        return Look(x, 170.0, sw, turn=turn, arms=(arm, arm), score=score)

    return at


def walker(start_x: float, speed_sw: float, sw: float, t0: float = 0.0, score=0.9, y=None,
           stop_x: float | None = None) -> Person:
    """Walks across at speed_sw own shoulder widths per second (1.3 m/s ~ 3.7),
    from t0, optionally stopping at stop_x. None while out of frame or not yet in."""
    y = y if y is not None else 150.0 + (130.0 - sw) * 0.6

    def at(t: float) -> Look | None:
        if t < t0:
            return None
        x = start_x + speed_sw * sw * (t - t0)
        if stop_x is not None and (x - stop_x) * speed_sw > 0:
            x = stop_x
        if x < -sw or x > W + sw:
            return None
        bob = 0.35 * math.sin(2 * math.pi * 1.8 * (t - t0))
        return Look(x, y, sw, arms=(0.25 + bob, 0.25 - bob), score=score)

    return at


def stander(x: float, sw: float, score=0.85, y=None, t0=0.0, t1=1e9) -> Person:
    """Stands still (watching, queueing) from t0 to t1."""
    y = y if y is not None else 150.0 + (130.0 - sw) * 0.6

    def at(t: float) -> Look | None:
        if not (t0 <= t < t1):
            return None
        return Look(x, y, sw, arms=(0.15, 0.2), score=score)

    return at


def leaving(person: Person, t_leave: float, speed_sw=3.5, direction=1.0) -> Person:
    """`person` until t_leave, then walking out of frame sideways."""

    def at(t: float) -> Look | None:
        if t < t_leave:
            return person(t)
        base = person(t_leave)
        if base is None:
            return None
        x = base.x + direction * speed_sw * base.sw * (t - t_leave)
        if x < -base.sw or x > W + base.sw:
            return None
        return Look(x, base.y, base.sw, arms=(0.3, 0.3), score=base.score)

    return at


def hidden(person: Person, t0: float, t1: float) -> Person:
    """`person`, but missed by the model from t0 to t1 (fully occluded)."""

    def at(t: float) -> Look | None:
        look = person(t)
        if look is not None and t0 <= t < t1:
            look.visible = False
        return look

    return at


# --- streams -------------------------------------------------------------------


def stream(people: dict[str, Person], duration: float, rate_hz: float = 20.0, seed: int = 0,
           noise: float = 4.0, elbow_drop: float = 0.4, miss: float = 0.03, occlude: bool = True):
    """[(t, [(name, Pose)])] sorted by score per result, as the model delivers them."""
    rng = np.random.default_rng(seed)
    out = []
    for t in model_clock(rate_hz, duration, rng):
        looks = {n: p(t) for n, p in people.items()}
        looks = {n: lk for n, lk in looks.items() if lk is not None}
        dets = []
        for name, lk in looks.items():
            if not lk.visible or rng.random() < miss:
                continue
            if occlude and any(covers(o, lk) for m, o in looks.items() if m != name):
                continue
            pose = body_pose(lk, rng, noise=noise, elbow_drop=elbow_drop)
            if pose is not None and (pose.confidence[[KP["left_shoulder"], KP["right_shoulder"]]] >= 0.5).any():
                pose.score = float(np.clip(lk.score + rng.normal(0, 0.03), 0.51, 0.99))
                dets.append((name, pose))
        dets.sort(key=lambda d: -d[1].score)
        out.append((t, dets))
    return out


def run(lock, frames):
    """Feed `frames` to `lock`: [(t, name of the pose returned or None, state)]."""
    log = []
    for t, dets in frames:
        got = lock.update([p for _, p in dets], t)
        name = next((n for n, p in dets if p is got), None)
        log.append((t, name, lock.state))
    return log
