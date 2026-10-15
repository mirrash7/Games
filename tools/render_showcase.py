"""Render the landing page's looping showcase clips from the real games.

    uv run python tools/render_showcase.py                 # all three clips + the hero montage
    uv run python tools/render_showcase.py --only snack    # one clip (snack | flappy | dance | hero)
    uv run python tools/render_showcase.py --sheet out.png # also write a contact sheet (6 frames a clip)

Writes, for each live game (`snack`, `flappy`, `dance`), into `web/media/`:

- `<id>.mp4`  H.264 (libx264, yuv420p, crf 28, +faststart), 960x540, 30 fps, no audio
- `<id>.webm` VP9 (libvpx-vp9, crf 38), same frames
- `<id>.jpg`  the poster: the clip's most striking frame, 1280x720, at most 120 KB

and `hero.mp4/.webm/.jpg`: a ~10 s montage cutting between the three games with
quick crossfades, for the landing hero when WebGL isn't available.

**Real games, no people.** Every frame is the real Python game (`update` at
60 Hz, `render` at 1280x720, the shipped art) with no shell around it, so there
is no PAUSE button, footer or menu. Nobody is filmed: the camera layer the
games draw over is a synthetic player, a glowing COCO-17 keypoint skeleton on a
dark gradient with a Roboflow-style detection box ("person 0.97"). The skeleton
is drawn from exactly the pose the game reads (`controls.pose`), so the raccoon
sits on its palm in Snack Attack, its arms are the flaps in Flappy Raccoon's
picture-in-picture, and it copies the coach in Raccoon Dance.

**Bots, through the real input path.** Each bot moves a body; its keypoints go
through a simulated model (results at 30 Hz, 90 ms late, a few px of noise)
and the app's `PoseExtrapolator`, exactly as `app.py` feeds a game
(`pose` = projected to now, `step_pose` = held). So the games see realistic
stepped, lagged input:

- Snack Attack: plans swipes through the snacks' predicted positions (lag
  included), chains several into one swing for combos, and never sweeps near
  a trash bag. The body steps sideways to keep the hand in reach.
- Flappy Raccoon: the fairness bot's policy (tests/test_flappy.py) with real
  arm downstrokes read by the real `FlapDetector`.
- Raccoon Dance: a near-perfect dancer (the coach's pose, 0.1 s late, as in
  tests/test_dance_game.py) on Neon Nights, NORMAL, around the gold move.

**Deterministic.** Games are seeded; the few unseeded `np.random.default_rng()`
calls in the games (sparks, feathers) are routed to a seeded stream while a
showcase runs; timings are fixed below. Same code, same frames. The video
encoders themselves are not guaranteed byte-identical across library versions.

**Loops.** Each clip's last 0.4 s crossfades into the 0.4 s that precede its
first frame, so the cut back to the start is continuous rather than a jump.

Tests: tests/test_showcase.py (media limits, and frame determinism through
`clip_frames`).
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kpapp.config import KP  # noqa: E402
from kpapp.controls import ControlState  # noqa: E402
from kpapp.inference import Pose, PoseExtrapolator, Result  # noqa: E402

MEDIA = ROOT / "web" / "media"
W, H = 1280, 720
SIM_HZ = 60  # game updates per second (the app's render rate)
FPS = 30  # output frame rate: one rendered frame every other update
OUT_SIZE = (960, 540)
LOOP_FADE = 0.4  # s: the end of a clip crossfades into what precedes its start
POSTER_MAX = 120 * 1024
CLIP_MAX = 1_500_000
IDS = ("snack", "flappy", "dance")

# --- the look of the synthetic player (BGR) ---------------------------------
CV_PURPLE = (235, 51, 128)  # #8033EB: the detection box and its tag
BONE = (250, 190, 225)  # lavender core
BONE_GLOW = (235, 60, 150)  # purple glow under it
JOINT = (255, 240, 250)

# COCO's skeleton (0-based), less its ear-to-shoulder edges: drawn on a synthetic
# body those boxed the head and torso into one coffin-like outline.
COCO_BONES = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
              (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4)]


# --- determinism -------------------------------------------------------------

class SeededNumpy:
    """Routes unseeded `np.random.default_rng()` calls to one seeded stream.

    The games draw cosmetic randomness (sparks, feathers, shake) that way; it
    is patched only around a showcase's own update/render calls, so two
    showcases (or two runs) never share or perturb each other's stream."""

    _orig = staticmethod(np.random.default_rng)

    def __init__(self, seed: int) -> None:
        self.stream = self._orig(seed)

    @contextmanager
    def active(self):
        orig, stream = self._orig, self.stream

        def rng(seed=None, *args, **kw):
            return stream if seed is None and not args and not kw else orig(seed, *args, **kw)

        np.random.default_rng = rng
        try:
            yield
        finally:
            np.random.default_rng = orig


# --- the synthetic camera ----------------------------------------------------

def _mono_font(px: int):
    """A monospace TrueType font for the detection tag, or None (Hershey fallback)."""
    try:
        from PIL import ImageFont
    except ImportError:  # pragma: no cover
        return None
    for path in ("/System/Library/Fonts/SFNSMono.ttf", "/System/Library/Fonts/Menlo.ttc",
                 "/Library/Fonts/Menlo.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, px)
            except OSError:
                continue
    return None


class SyntheticCamera:
    """The "camera image" a game draws over: a dark backdrop and a keypoint skeleton.

    `scale` thickens everything for games that shrink the camera into a
    picture-in-picture (Flappy shows it at 1/4 size, Dance at 0.3)."""

    def __init__(self, scale: float = 1.0, seed: int = 0, glow: float = 1.0,
                 keep_out: tuple = ()) -> None:
        self.scale = scale
        self.glow = glow
        self.keep_out = keep_out  # HUD rects (x0, y0, x1, y1) the tag must not cover
        self.bg = self._backdrop(seed)
        self._tags: dict[str, np.ndarray] = {}

    @staticmethod
    def _backdrop(seed: int) -> np.ndarray:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        v = yy / H
        top = np.array([46, 20, 30], np.float32)  # deep indigo-violet
        bottom = np.array([14, 8, 12], np.float32)
        img = top[None, None] * (1 - v[..., None]) + bottom[None, None] * v[..., None]
        # A soft light behind the player, and a vignette.
        d = np.hypot((xx - W * 0.5) / (W * 0.55), (yy - H * 0.42) / (H * 0.75))
        img += np.array([34, 12, 26], np.float32)[None, None] * np.clip(1 - d, 0, 1)[..., None] ** 2
        img *= (1 - 0.35 * np.clip(d - 0.6, 0, 1))[..., None]
        # A faint floor grid: reads as "a room the camera sees", not a void.
        floor = H * 0.78
        grid = np.zeros((H, W), np.float32)
        for k in range(1, 9):
            y = floor + (H - floor) * (k / 8) ** 1.8
            cv2.line(grid, (0, int(y)), (W, int(y)), 1.0, 1, cv2.LINE_AA)
        for k in range(-14, 15):
            cv2.line(grid, (int(W / 2 + k * 18), int(floor)), (int(W / 2 + k * 150), H), 1.0, 1, cv2.LINE_AA)
        grid *= np.clip((yy - floor) / (H - floor), 0, 1) * 0.6
        img += grid[..., None] * np.array([70, 30, 55], np.float32)[None, None]
        # Ordered dither, so the dark gradient does not band at 720p.
        rng = np.random.default_rng(seed)
        img += rng.uniform(-1.5, 1.5, img.shape[:2]).astype(np.float32)[..., None]
        return np.clip(img, 0, 255).astype(np.uint8)

    def _tag(self, text: str) -> np.ndarray:
        img = self._tags.get(text)
        if img is not None:
            return img
        s = self.scale
        px = int(round(18 * s))
        pad_x, pad_y = int(round(7 * s)), int(round(4 * s))
        font = _mono_font(px)
        if font is not None:
            from PIL import Image, ImageDraw

            x0, y0, x1, y1 = font.getbbox(text)
            tw, th = x1 - x0, int(px * 0.78)
            pil = Image.new("RGB", (tw + 2 * pad_x, th + 2 * pad_y), CV_PURPLE[::-1])
            ImageDraw.Draw(pil).text((pad_x - x0, pad_y + th - px * 0.82 - 1), text, font=font,
                                     fill=(255, 255, 255))
            img = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        else:  # pragma: no cover - no TrueType mono font on this machine
            fs = 0.5 * s
            (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, fs * 2, 1)
            img = np.full((th + 2 * pad_y + base, tw + 2 * pad_x, 3), CV_PURPLE, np.uint8)
            cv2.putText(img, text, (pad_x, pad_y + th), cv2.FONT_HERSHEY_PLAIN, fs * 2,
                        (255, 255, 255), max(1, int(s)), cv2.LINE_AA)
        self._tags[text] = img
        return img

    def frame(self, pose: Pose | None) -> np.ndarray:
        img = self.bg.copy()
        if pose is None:
            return img
        s = self.scale
        vis = pose.confidence >= 0.4
        pts = {i: (float(pose.xy[i][0]), float(pose.xy[i][1])) for i in range(17) if vis[i]}

        def p(i):
            return int(round(pts[i][0] * 16)), int(round(pts[i][1] * 16))

        # Glow: the bones drawn thick on a quarter-size layer, blurred, added.
        q = 4
        glow = np.zeros((H // q, W // q, 3), np.uint8)
        for a, b in COCO_BONES:
            if a in pts and b in pts:
                cv2.line(glow, (int(pts[a][0] / q), int(pts[a][1] / q)),
                         (int(pts[b][0] / q), int(pts[b][1] / q)), BONE_GLOW,
                         max(2, int(round(5 * s))), cv2.LINE_AA)
        glow = cv2.GaussianBlur(glow, (0, 0), 3.0 * max(1.0, s * 0.8))
        glow = cv2.resize(glow, (W, H), interpolation=cv2.INTER_LINEAR)
        if self.glow != 1.0:
            glow = cv2.convertScaleAbs(glow, alpha=self.glow)
        cv2.add(img, glow, dst=img)

        lw = max(2, int(round(4 * s)))
        for a, b in COCO_BONES:
            if a in pts and b in pts:
                cv2.line(img, p(a), p(b), BONE_GLOW, lw + max(2, int(round(3 * s))), cv2.LINE_AA, shift=4)
                cv2.line(img, p(a), p(b), BONE, lw, cv2.LINE_AA, shift=4)
        jr = max(3, int(round(6 * s)))
        for i in pts:
            r = jr - 2 if i <= 4 else jr  # face points a touch smaller
            cv2.circle(img, p(i), (r + max(1, int(round(2 * s)))) * 16, CV_PURPLE, -1, cv2.LINE_AA, shift=4)
            cv2.circle(img, p(i), r * 16, JOINT, -1, cv2.LINE_AA, shift=4)

        self._draw_box(img, pose, pts)
        return img

    def _draw_box(self, img, pose: Pose, pts: dict) -> None:
        if KP["left_shoulder"] not in pts or KP["right_shoulder"] not in pts:
            return
        s = self.scale
        sw = math.dist(pts[KP["left_shoulder"]], pts[KP["right_shoulder"]])
        xs = [x for x, _ in pts.values()]
        ys = [y for _, y in pts.values()]
        head_top = min(ys) - 0.5 * sw if any(i <= 4 for i in pts) else min(ys) - 0.2 * sw
        x0, x1 = min(xs) - 0.22 * sw, max(xs) + 0.22 * sw
        y0, y1 = head_top, max(ys) + 0.18 * sw
        t = max(1, int(round(2 * s)))
        x0, y0 = int(max(t, x0)), int(max(t, y0))
        x1, y1 = int(min(W - 1 - t, x1)), int(min(H - 1 - t, y1))
        cv2.rectangle(img, (x0, y0), (x1, y1), CV_PURPLE, t, cv2.LINE_AA)
        tag = self._tag(f"person {pose.score:.2f}")
        th, tw = tag.shape[:2]
        ty = y0 - th if y0 - th >= 0 else y0  # above the box, or inside it at the top edge
        tx = min(max(0, x0 - t // 2), W - tw)

        def covers(x):
            return any(x < r[2] and x + tw > r[0] and ty < r[3] and ty + th > r[1] for r in self.keep_out)

        if covers(tx):  # slide along the box's top edge, clear of the game's HUD
            for x in range(tx, max(tx, x1 - tw) + 1, 8):
                if not covers(x):
                    tx = x
                    break
        img[ty:ty + th, tx:tx + tw] = tag


# --- the simulated model -----------------------------------------------------

class PosePipeline:
    """The app's input path, simulated: the body is captured at `model_hz`,
    published `latency` s later with a little keypoint noise, and projected
    to each render frame by the app's PoseExtrapolator (`app.py:_run_loop`)."""

    def __init__(self, model_hz: float = 30.0, latency: float = 0.09, noise: float = 1.0,
                 seed: int = 0) -> None:
        self.period = 1.0 / model_hz
        self.latency = latency
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.extrap = PoseExtrapolator()
        self.pending: deque = deque()
        self.next_capture = 0.0

    def step(self, t: float, body) -> ControlState:
        """`body()` returns the true Pose now (or None). Call once per update."""
        if t >= self.next_capture - 1e-9:
            self.next_capture += self.period
            truth = body()
            poses = []
            if truth is not None:
                xy = truth.xy + self.rng.normal(0, self.noise, truth.xy.shape)
                score = float(np.clip(0.97 + self.rng.normal(0, 0.008), 0.94, 0.99))
                poses = [Pose(xy=xy.astype(np.float32), confidence=truth.confidence.copy(), score=score)]
            # The frame enters the model ~40 ms after the moment it shows; the
            # result reaches the render loop `latency` after that moment.
            self.pending.append((t + self.latency, Result(poses=poses, timestamp=t + 0.04)))
        while self.pending and self.pending[0][0] <= t + 1e-9:
            self.extrap.update(self.pending.popleft()[1], arrived=t)
        poses = self.extrap.poses_at(t)
        pose = poses[0] if poses else None
        held = self.extrap.held[0] if self.extrap.held else None
        return ControlState(present=pose is not None, pose=pose, step_pose=held)


# --- body helpers --------------------------------------------------------------

def two_link(shoulder, target, a: float, b: float, bend: float) -> tuple[np.ndarray, np.ndarray]:
    """Elbow and end point of a two-segment limb from `shoulder` toward `target`.

    The end point is clamped to the limb's reach; `bend` (+1/-1) picks which
    side of the shoulder-target line the elbow sits on."""
    s = np.asarray(shoulder, np.float64)
    d = np.asarray(target, np.float64) - s
    L = float(np.hypot(*d))
    reach = (a + b) * 0.995
    if L > reach:
        d *= reach / L
        L = reach
    L = max(L, abs(a - b) + 1e-3)
    u = d / L
    x = (a * a - b * b + L * L) / (2 * L)
    y = math.sqrt(max(0.0, a * a - x * x))
    n = np.array([-u[1], u[0]]) * bend
    elbow = s + u * x + n * y
    return elbow, s + d


class Body:
    """A front-facing body in screen pixels, written as COCO keypoints.

    Placed by its hip centre `hx` and a torso `lean` (degrees, + tips the
    head to screen-right). Screen-right limbs are COCO `left_*` on a mirrored
    feed (AGENTS.md section 6)."""

    def __init__(self, shoulder_px: float, shoulder_y: float) -> None:
        self.sw = shoulder_px
        self.sy = shoulder_y  # shoulder line when upright
        self.hip_y = shoulder_y + 1.25 * shoulder_px

    def frame(self, hx: float, lean: float = 0.0) -> dict[str, np.ndarray]:
        """Shoulders, head and hips for a body at `hx`, leaning `lean` degrees."""
        sw, a = self.sw, math.radians(lean)
        up = np.array([math.sin(a), -math.cos(a)])
        across = np.array([math.cos(a), math.sin(a)])  # toward screen-right
        hip = np.array([hx, self.hip_y])
        neck = hip + up * 1.25 * sw
        return {"r_shoulder": neck + across * sw / 2, "l_shoulder": neck - across * sw / 2,
                "head": neck + up * 0.66 * sw,
                "r_hip": hip + np.array([0.3 * sw, 0.0]), "l_hip": hip - np.array([0.3 * sw, 0.0])}

    def pose(self, fr: dict, right_arm, left_arm, feet=None) -> Pose:
        """`right_arm`/`left_arm`: (elbow, wrist) of the screen-right/-left arm.
        `feet`: (screen-right, screen-left) ankle x, default under the hips."""
        sw = self.sw
        xy = np.zeros((17, 2), np.float32)
        c = np.zeros(17, np.float32)

        def put(name, p, conf=0.9):
            xy[KP[name]] = p
            c[KP[name]] = conf if -4 <= p[1] <= H + 4 and -4 <= p[0] <= W + 4 else 0.0

        put("left_shoulder", fr["r_shoulder"])
        put("right_shoulder", fr["l_shoulder"])
        for name, p in face_points(fr["head"], sw).items():
            put(name, p)
        put("left_elbow", right_arm[0])
        put("left_wrist", right_arm[1])
        put("right_elbow", left_arm[0])
        put("right_wrist", left_arm[1])
        if feet is None:
            feet = (fr["r_hip"][0] + 0.06 * sw, fr["l_hip"][0] - 0.06 * sw)
        for coco, hip, fx, out in (("left", fr["r_hip"], feet[0], 1), ("right", fr["l_hip"], feet[1], -1)):
            ankle = np.array([fx, hip[1] + 1.67 * sw])
            knee = (hip + ankle) / 2 + np.array([out * 0.05 * sw, 0.0])
            put(f"{coco}_hip", hip)
            put(f"{coco}_knee", knee)
            put(f"{coco}_ankle", ankle)
        return Pose(xy=xy, confidence=c, score=0.97)


def face_points(head, sw: float) -> dict[str, tuple[float, float]]:
    """Nose, eyes and ears around a head centre, spread like a real detection."""
    hx, hy = float(head[0]), float(head[1])
    return {"nose": (hx, hy + 0.12 * sw),
            "left_eye": (hx + 0.08 * sw, hy - 0.02 * sw), "right_eye": (hx - 0.08 * sw, hy - 0.02 * sw),
            "left_ear": (hx + 0.19 * sw, hy + 0.04 * sw), "right_ear": (hx - 0.19 * sw, hy + 0.04 * sw)}


def _ease(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


# --- showcases -------------------------------------------------------------------

@dataclass
class Timing:
    start: float  # sim seconds at which the loop begins
    seconds: float  # loop length
    poster: float  # s into the loop: the poster frame
    hero: float  # s into the loop where its hero-montage segment starts


class Showcase:
    id = ""
    timing: Timing
    seed = 0

    def __init__(self) -> None:
        self.t = 0.0
        self.ticks = 0
        self.numpy = SeededNumpy(self.seed)
        self.controls = ControlState()

    def step(self) -> None:
        raise NotImplementedError

    def camera(self) -> SyntheticCamera:
        raise NotImplementedError

    def render(self) -> np.ndarray:
        frame = self.cam.frame(self.controls.pose)
        with self.numpy.active():
            self.game.render(frame)
        return frame


# --- Snack Attack ---

class SnackBot:
    """Swipes the raccoon through the snacks, several per swing when it can.

    Plans in true time against where each snack *will* be when the game sees
    the hand there (`LAG` later), and rejects any swing that passes near a
    trash bag. Between swings the hand drifts slowly (below the cutting speed)."""

    LAG = 0.07  # how far behind the game sees the true hand, after extrapolation
    SWIPE = 1900.0  # px/s through the snacks
    APPROACH = 1250.0  # px/s getting into position
    DRIFT = 140.0  # px/s between swings: well under the 260 px/s cutting speed
    BOMB_CLEAR = 44.0 + 70.0  # blade half-width + margin, on top of the bag's radius
    G = 1500.0

    def __init__(self, body: Body) -> None:
        self.body = body
        self.hand = np.array([700.0, 330.0])
        self.path: list[tuple[float, np.ndarray]] = []  # (time, palm) waypoints
        self.cx = 560.0  # hip centre
        self.lean = 0.0
        self.feet = [560.0 + 0.36 * body.sw, 560.0 - 0.36 * body.sw]
        self.step: list = [None, None]

    def _pred(self, f, tau: float) -> np.ndarray:
        return np.array([f.pos[0] + f.vel[0] * tau, f.pos[1] + f.vel[1] * tau + 0.5 * self.G * tau * tau])

    def _plan(self, t: float, game) -> None:
        live = [f for f in game.fruits if not f.sliced]
        bombs = [f for f in live if f.is_bomb]
        snacks = []
        for f in live:
            if f.is_bomb:
                continue
            p = self._pred(f, self.LAG + 0.3)
            if 90 < p[0] < W - 90 and 60 < p[1] < H - 150 and f.pos[1] < H + 20:
                snacks.append(f)
        while snacks:
            path = self._swing(t, snacks)
            bad = self._conflict(t, path, bombs)
            if bad is None:
                self.path = path
                return
            worst = min(snacks, key=lambda f: float(np.hypot(*(self._pred(f, 0.3) - bad))))
            snacks = [f for f in snacks if f is not worst]
        # Nothing safe to cut: drift toward a resting spot, slowly (no cutting),
        # unless a bag is coming: then step out of its way along the safest line.
        rest = np.array([self.cx + self.body.sw * 0.5 + 70, self.body.sy - 20.0])
        d = rest - self.hand
        step = min(1.0, self.DRIFT * 0.25 / max(1e-6, float(np.hypot(*d))))
        stay = [(t, self.hand.copy()), (t + 0.25, self.hand + d * step)]
        if self._clearance(t, stay, bombs) > 60.0:
            self.path = stay
            return
        best, best_c = stay, self._clearance(t, stay, bombs)
        for k in range(12):
            ang = k * math.pi / 6
            to = np.clip(self.hand + 110.0 * np.array([math.cos(ang), math.sin(ang)]), [60, 60], [W - 60, H - 160])
            cand = [(t, self.hand.copy()), (t + 0.25, to)]
            c = self._clearance(t, cand, bombs)
            if c > best_c:
                best, best_c = cand, c
        self.path = best

    def _clearance(self, t: float, path, bombs, horizon: float = 0.6) -> float:
        """Smallest gap between the hand (following `path`, then holding) and any
        bag's edge plus the blade margin, over the next `horizon` s."""
        worst = float("inf")
        for k in range(int(horizon * 60) + 1):
            s = t + k / 60
            p = self._curve(path, s)
            for bomb in bombs:
                bp = self._pred(bomb, s - t + self.LAG)
                worst = min(worst, float(np.hypot(*(p - bp))) - bomb.radius - self.BOMB_CLEAR)
        return worst

    def _swing(self, t: float, snacks) -> list[tuple[float, np.ndarray]]:
        first_guess = [self._pred(f, 0.35) for f in snacks]
        order = sorted(range(len(snacks)), key=lambda i: first_guess[i][0])
        if abs(first_guess[order[-1]][0] - self.hand[0]) < abs(first_guess[order[0]][0] - self.hand[0]):
            order.reverse()
        targets = [snacks[i] for i in order]
        direction = 1.0 if first_guess[order[-1]][0] >= first_guess[order[0]][0] else -1.0
        if len(targets) == 1:  # a lone snack: swing toward the side with more room
            direction = 1.0 if first_guess[order[0]][0] > self.hand[0] else -1.0
        path = [(t, self.hand.copy())]
        tt, here = t, self.hand.copy()
        # Approach: a point before the first snack, a little below its line.
        p0 = self._pred(targets[0], (tt - t) + self.LAG + 0.25)
        start = p0 + np.array([-direction * 150.0, 50.0])
        start = np.clip(start, [40, 60], [W - 40, H - 120])
        dt = max(0.12, float(np.hypot(*(start - here))) / self.APPROACH)
        tt += dt
        here = start
        path.append((tt, here.copy()))
        for f in targets:
            arrive = tt
            p = here
            for _ in range(3):  # where it will be when we get there
                p = self._pred(f, arrive - t + self.LAG)
                arrive = tt + max(0.05, float(np.hypot(*(p - here))) / self.SWIPE)
            tt, here = arrive, p
            path.append((tt, here.copy()))
        end = here + np.array([direction * 180.0, -40.0])
        end = np.clip(end, [40, 60], [W - 40, H - 120])
        tt += max(0.08, float(np.hypot(*(end - here))) / self.SWIPE)
        path.append((tt, end.copy()))
        tt += 0.18
        path.append((tt, end + np.array([direction * 25.0, 10.0])))  # ease out
        return path

    def _conflict(self, t: float, path, bombs) -> np.ndarray | None:
        """Where the swing (the curve actually flown, sampled) passes a bag, if it does."""
        if not bombs:
            return None
        t0, t1 = path[0][0], path[-1][0] + 0.1
        for k in range(int((t1 - t0) * 120) + 1):
            s = t0 + k / 120
            p = self._curve(path, s)
            for bomb in bombs:
                bp = self._pred(bomb, s - t + self.LAG)
                if np.hypot(*(p - bp)) < bomb.radius + self.BOMB_CLEAR:
                    return bp
        return None

    def _palm_at(self, t: float) -> np.ndarray:
        return self._curve(self.path, t) if self.path else self.hand

    @staticmethod
    def _curve(pts, t: float) -> np.ndarray:
        """Catmull-Rom through the waypoints (passes through each of them)."""
        if t <= pts[0][0]:
            return pts[0][1]
        if t >= pts[-1][0]:
            return pts[-1][1]
        for i in range(len(pts) - 1):
            if pts[i][0] <= t <= pts[i + 1][0]:
                break
        p0 = pts[max(0, i - 1)][1]
        p1, p2 = pts[i][1], pts[i + 1][1]
        p3 = pts[min(len(pts) - 1, i + 2)][1]
        u = (t - pts[i][0]) / max(1e-9, pts[i + 1][0] - pts[i][0])
        u2, u3 = u * u, u * u * u
        return 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2
                      + (-p0 + 3 * p1 - 3 * p2 + p3) * u3)

    def tick(self, t: float, dt: float, game) -> None:
        if not self.path or t >= self.path[-1][0]:
            self._plan(t, game)
        self.hand = np.asarray(self._palm_at(t), np.float64)
        # The body steps toward where the hand is going (looking ahead along
        # the plan) and leans into the reach, so the arm stays in reach
        # without the figure sliding about like a robot.
        b, sw = self.body, self.body.sw
        ahead = float(np.mean([self._palm_at(t + k * 0.06)[0] for k in range(-2, 7)]))
        want = ahead - 70.0 - sw / 2  # where the shoulder centre would like to be
        self.cx += (float(np.clip(want, 170, W - 300)) - self.cx) * (1 - math.exp(-dt / 0.30))
        lean = math.degrees(math.asin(float(np.clip((want - self.cx) / (1.25 * sw), -0.25, 0.25))))
        self.lean += (lean - self.lean) * (1 - math.exp(-dt / 0.08))
        for i, side in enumerate((1, -1)):
            stance = self.cx + side * 0.36 * sw
            if abs(self.feet[i] - stance) > 0.25 * sw or self.step[i] is not None:  # take a step
                if self.step[i] is None:
                    self.step[i] = (t, self.feet[i])
                t0, x0 = self.step[i]
                u = (t - t0) / 0.22
                self.feet[i] = x0 + (stance - x0) * _ease(u)
                if u >= 1:
                    self.step[i] = None

    def pose(self, t: float) -> Pose:
        b = self.body
        sw = b.sw
        fr = b.frame(self.cx, self.lean)
        # The palm is the wrist pushed 0.38 of a forearm on (hand.PALM_REACH), so
        # the elbow-to-palm segment is 1.38 forearms long.
        fore = 0.58 * sw
        elbow, palm = two_link(fr["r_shoulder"], self.hand, 0.62 * sw, fore * 1.38, bend=-1.0)
        wrist = elbow + (palm - elbow) / 1.38
        # The other arm hangs loose and sways a little.
        sh2 = fr["l_shoulder"]
        sway = math.sin(t * 2.1) * 0.06 + math.radians(self.lean) * 0.5
        e2 = sh2 + 0.62 * sw * np.array([-math.sin(0.30 + sway), math.cos(0.30 + sway)])
        w2 = e2 + 0.58 * sw * np.array([-math.sin(-0.55 + sway), math.cos(-0.55 + sway)])
        return b.pose(fr, (elbow, wrist), (e2, w2), feet=tuple(self.feet))


class SnackShowcase(Showcase):
    id = "snack"
    seed = 6
    timing = Timing(start=40.0, seconds=7.5, poster=3.0, hero=1.0)

    def __init__(self) -> None:
        super().__init__()
        from kpapp.game.fruitninja.game import FruitNinjaGame

        self.game = FruitNinjaGame((W, H), seed=self.seed)
        self.bot = SnackBot(Body(shoulder_px=160.0, shoulder_y=245.0))
        self.pipe = PosePipeline(seed=self.seed)
        self.cam = SyntheticCamera(scale=1.4, seed=self.seed, glow=1.6,
                                   keep_out=((0, 0, 240, 100), (1050, 0, W, 80)))

    def step(self) -> None:
        dt = 1.0 / SIM_HZ
        self.bot.tick(self.t, dt, self.game)
        self.controls = self.pipe.step(self.t, lambda: self.bot.pose(self.t))
        with self.numpy.active():
            self.game.update(self.controls, dt)
        self.t += dt
        self.ticks += 1


# --- Flappy Raccoon ---

class FlappyBot:
    """The fairness bot's policy (tests/test_flappy.py LaggedBot), but with arms.

    Decides from the screen when a flap must land, allowing for the time the
    flap takes to reach the game (`DELAY`: the downstroke until the detector
    fires, plus the pipeline), then performs a real downstroke for the real
    FlapDetector to read."""

    DELAY = 0.15
    DOWN = 0.16  # s: a brisk downstroke
    UP = 0.30  # s: back up
    TOP, BOTTOM = 128.0, 12.0  # arm angle from straight down, degrees

    def __init__(self, body: Body) -> None:
        self.body = body
        self.stroke_t0: float | None = None
        self.pending: list[float] = []  # when started flaps should reach the game
        self.started = False

    def angle(self, t: float) -> float:
        if self.stroke_t0 is None:
            return self.TOP + 3.0 * math.sin(t * 5.0)
        u = t - self.stroke_t0
        if u < self.DOWN:
            k = (1 - math.cos(math.pi * u / self.DOWN)) / 2
            return self.TOP + (self.BOTTOM - self.TOP) * k
        k = (1 - math.cos(math.pi * min(1.0, (u - self.DOWN) / self.UP))) / 2
        return self.BOTTOM + (self.TOP - self.BOTTOM) * k

    def _predict(self, game, t: float) -> float:
        r = game.rules
        y, vy = game.bird.y, game.bird.vy
        flaps = sorted(f for f in self.pending if f > t)
        tt, end, dt = t, t + self.DELAY, 1.0 / SIM_HZ
        while tt < end - 1e-9:
            while flaps and flaps[0] <= tt + 1e-9:
                vy = -r.flap_velocity
                flaps.pop(0)
            vy = min(vy + r.gravity * dt, r.terminal_velocity)
            y += vy * dt
            tt += dt
        return y

    def tick(self, t: float, game) -> None:
        if self.stroke_t0 is not None and t - self.stroke_t0 >= self.DOWN + self.UP:
            self.stroke_t0 = None
        if self.stroke_t0 is not None:
            return
        if game.phase.value == "ready":
            if t > 0.6 and not self.started:
                self._flap(t)
                self.started = True
            return
        r = game.rules
        y = self._predict(game, t)
        reach = game.pipe_width / 2 + game.hit_radius
        target = next((p for p in game.pipes
                       if p.x - game.pipe_speed() * self.DELAY + reach > game.bird.x), None)
        if target is not None:
            rise = r.flap_velocity ** 2 / (2 * r.gravity)
            lo = target.gap_top + game.hit_radius + rise
            hi = target.gap_bottom - game.hit_radius - r.terminal_velocity / SIM_HZ - 2
            line = (lo + hi) / 2
        else:
            line = game.ground_y * 0.5
        if y > line:
            self._flap(t)

    def _flap(self, t: float) -> None:
        self.stroke_t0 = t
        self.pending = [f for f in self.pending if f > t - 1.0] + [t + self.DELAY]

    def pose(self, t: float, cx: float) -> Pose:
        b = self.body
        sw = b.sw
        th = math.radians(self.angle(t))
        fr = b.frame(cx)
        arms = []
        for side in (1, -1):  # screen-right, screen-left
            sh = fr["r_shoulder"] if side > 0 else fr["l_shoulder"]
            wrist = sh + 1.2 * sw * np.array([side * math.sin(th), math.cos(th)])
            elbow, wrist = two_link(sh, wrist, 0.62 * sw, 0.62 * sw, bend=-side if th > 1.2 else side)
            arms.append((elbow, wrist))
        return b.pose(fr, arms[0], arms[1])


class FlappyShowcase(Showcase):
    id = "flappy"
    seed = 11
    # 7.25 s is exactly four pipe passes (scores 11 -> 15), so at the loop
    # point the pipes stand where they started and only the bird crossfades.
    timing = Timing(start=24.0, seconds=7.25, poster=1.5, hero=0.5)

    def __init__(self) -> None:
        super().__init__()
        from kpapp.game.flappy.game import FlappyGame

        FlappyGame.session_best = 0  # class-level session best: start every run the same
        self.game = FlappyGame((W, H), seed=self.seed)
        self.bot = FlappyBot(Body(shoulder_px=165.0, shoulder_y=300.0))
        self.pipe = PosePipeline(seed=self.seed)
        self.cam = SyntheticCamera(scale=3.2, seed=self.seed, glow=1.2)

    def step(self) -> None:
        dt = 1.0 / SIM_HZ
        self.bot.tick(self.t, self.game)
        self.controls = self.pipe.step(self.t, lambda: self.bot.pose(self.t, W / 2))
        with self.numpy.active():
            self.game.update(self.controls, dt)
        self.t += dt
        self.ticks += 1


# --- Raccoon Dance ---

class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class DanceShowcase(Showcase):
    id = "dance"
    seed = 5
    LATE = 0.10  # the dancer's own reaction time, on top of the pipeline's lag
    timing = Timing(start=33.6, seconds=7.4, poster=6.1, hero=3.4)

    def __init__(self) -> None:
        super().__init__()
        from kpapp.audio import SilentPlayer
        from kpapp.game.dance import chart
        from kpapp.game.dance.game import DanceGame

        self.clock = _Clock()
        song = chart.load_song(chart.SONGS / "neon_nights", chart.load_library(chart.LIBRARY))
        player = SilentPlayer(song.duration + 2.0, clock=self.clock)
        self.game = DanceGame((W, H), variant="neon_nights", difficulty="normal", player=player,
                              seed=self.seed)
        self.pipe = PosePipeline(seed=self.seed)
        self.cam = SyntheticCamera(scale=2.6, seed=self.seed, glow=1.2)

    def _body(self) -> Pose:
        from kpapp.game.dance.pose import keypoints_for

        g = self.game
        st = g.audio.position() - g.song.offset if g._started else -g.song.offset
        target = g.song.coach_pose(g.song.beat_of(st - self.LATE))
        kp = keypoints_for(target, centre_px=(W / 2, 250.0), shoulder_px=150.0)
        # keypoints_for puts the whole face on one point; spread it so it reads as a head.
        for name, p in face_points(kp.xy[KP["nose"]], 150.0).items():
            kp.xy[KP[name]] = p
            kp.confidence[KP[name]] = 0.9
        return kp

    def step(self) -> None:
        dt = 1.0 / SIM_HZ
        self.controls = self.pipe.step(self.t, self._body)
        with self.numpy.active():
            self.game.update(self.controls, dt)
        self.t += dt
        self.clock.t = self.t
        self.ticks += 1


SHOWCASES = {"snack": SnackShowcase, "flappy": FlappyShowcase, "dance": DanceShowcase}


def make(clip_id: str) -> Showcase:
    return SHOWCASES[clip_id]()


# --- frames ---------------------------------------------------------------------

def clip_frames(sc: Showcase):
    """The loop's frames (1280x720 BGR, at FPS), its end crossfaded into what
    precedes its start. Deterministic: same showcase, same frames."""
    per = SIM_HZ // FPS
    fade = int(round(LOOP_FADE * FPS))
    total = int(round(sc.timing.seconds * FPS))
    pre_ticks = int(round(sc.timing.start * SIM_HZ)) - fade * per
    while sc.ticks < pre_ticks:
        sc.step()
    before = []
    for _ in range(fade):
        before.append(sc.render())
        for _ in range(per):
            sc.step()
    for i in range(total):
        frame = sc.render()
        k = i - (total - fade)
        if k >= 0:
            w = (k + 1) / (fade + 1)
            frame = cv2.addWeighted(frame, 1 - w, before[k], w, 0)
        yield frame
        for _ in range(per):
            sc.step()


# --- encoding ------------------------------------------------------------------

class Encoder:
    """mp4 (H.264) and webm (VP9) side by side, fed the same frames."""

    def __init__(self, stem: Path) -> None:
        import av

        self.av = av
        self.outs = []
        for ext, codec, opts in (
            ("mp4", "libx264", {"crf": "28", "preset": "slow", "profile": "high",
                                "x264-params": "keyint=60:min-keyint=30"}),
            ("webm", "libvpx-vp9", {"crf": "38", "b": "0", "deadline": "good", "cpu-used": "1",
                                    "row-mt": "1", "g": "60"}),
        ):
            path = stem.with_suffix("." + ext)
            options = {"movflags": "+faststart"} if ext == "mp4" else {}
            box = av.open(str(path), mode="w", options=options)
            st = box.add_stream(codec, rate=FPS, options=opts)
            st.width, st.height = OUT_SIZE
            st.pix_fmt = "yuv420p"
            st.codec_context.time_base = Fraction(1, FPS)
            self.outs.append((box, st))
        self.n = 0

    def add(self, bgr_small: np.ndarray) -> None:
        frame = self.av.VideoFrame.from_ndarray(bgr_small, format="bgr24")
        frame.pts = self.n
        frame.time_base = Fraction(1, FPS)
        for box, st in self.outs:
            for packet in st.encode(frame):
                box.mux(packet)
        self.n += 1

    def close(self) -> None:
        for box, st in self.outs:
            for packet in st.encode():
                box.mux(packet)
            box.close()


def small(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, OUT_SIZE, interpolation=cv2.INTER_AREA)


def write_poster(frame: np.ndarray, path: Path) -> int:
    for q in range(88, 50, -4):
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, q, cv2.IMWRITE_JPEG_OPTIMIZE, 1,
                                               cv2.IMWRITE_JPEG_PROGRESSIVE, 1])
        if ok and len(buf) <= POSTER_MAX:
            path.write_bytes(buf.tobytes())
            return len(buf)
    return write_poster(small(frame), path)


HERO_SEGMENT = 3.6  # s of each game in the montage
HERO_FADE = 0.4


def render_clip(clip_id: str, write: bool = True, sheet: list | None = None):
    """Render one clip; returns (hero segment frames at OUT_SIZE, poster frame)."""
    sc = make(clip_id)
    tm = sc.timing
    enc = Encoder(MEDIA / clip_id) if write else None
    hero0 = int(round(tm.hero * FPS))
    hero1 = hero0 + int(round(HERO_SEGMENT * FPS))
    poster_i = int(round(tm.poster * FPS))
    total = int(round(tm.seconds * FPS))
    picks = {int(round(k * (total - 1) / 5)) for k in range(6)}
    hero, poster = [], None
    for i, frame in enumerate(clip_frames(sc)):
        lo = small(frame)
        if enc is not None:
            enc.add(lo)
        if hero0 <= i < hero1:
            hero.append(lo)
        if i == poster_i:
            poster = frame.copy()
        if sheet is not None and i in picks:
            sheet.append((clip_id, i, lo.copy()))
    if enc is not None:
        enc.close()
        write_poster(poster, MEDIA / f"{clip_id}.jpg")
    return hero, poster


def render_hero(segments: list[list[np.ndarray]], posters: list[np.ndarray]) -> None:
    """Cut between the games with quick crossfades; the last fades into the first."""
    o = int(round(HERO_FADE * FPS))
    enc = Encoder(MEDIA / "hero")
    for k, seg in enumerate(segments):
        prev = segments[k - 1]
        for i in range(len(seg) - o):
            if i < o:
                w = (i + 1) / (o + 1)
                enc.add(cv2.addWeighted(prev[len(prev) - o + i], 1 - w, seg[i], w, 0))
            else:
                enc.add(seg[i])
    enc.close()
    write_poster(posters[0], MEDIA / "hero.jpg")


def contact_sheet(rows: list, path: Path) -> None:
    thumb = (480, 270)
    by = {}
    for cid, i, img in rows:
        by.setdefault(cid, []).append((i, img))
    lines = []
    for cid, items in by.items():
        items.sort(key=lambda x: x[0])
        cells = []
        for i, img in items[:6]:
            t = cv2.resize(img, thumb, interpolation=cv2.INTER_AREA)
            cv2.putText(t, f"{cid} {i / FPS:.2f}s", (8, 262), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cells.append(t)
        while len(cells) < 6:
            cells.append(np.zeros((thumb[1], thumb[0], 3), np.uint8))
        lines.append(np.hstack(cells))
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack(lines))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", choices=(*IDS, "hero"), help="one clip (hero renders all three, writes only the montage)")
    ap.add_argument("--sheet", type=Path, help="also write a contact sheet PNG here")
    args = ap.parse_args(argv)
    MEDIA.mkdir(parents=True, exist_ok=True)
    ids = IDS if args.only in (None, "hero") else (args.only,)
    sheet: list | None = [] if args.sheet else None
    segments, posters = [], []
    for cid in ids:
        print(f"[showcase] {cid} ...", flush=True)
        hero, poster = render_clip(cid, write=args.only != "hero", sheet=sheet)
        segments.append(hero)
        posters.append(poster)
    if len(ids) == len(IDS):
        print("[showcase] hero ...", flush=True)
        render_hero(segments, posters)
    for p in sorted(MEDIA.glob("*")):
        print(f"  {p.relative_to(ROOT)}  {p.stat().st_size / 1024:.0f} KB")
    if sheet is not None:
        contact_sheet(sheet, args.sheet)
        print(f"[showcase] sheet -> {args.sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
