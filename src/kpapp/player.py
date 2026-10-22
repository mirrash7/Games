"""PlayerLock: follow one main player through a crowd.

The model sees everyone in frame. At a booth that means passers-by, people
queueing behind and friends watching from the side. Handing the games the
highest-scoring detection lets any of them take over: the cursor jumps to a
stranger's hand, a flap fires from someone walking past, the dance judge
scores the wrong body. PlayerLock picks the main player once and then follows
that identity, result by result, so every game sees only them.

It runs on each new model result, before PoseExtrapolator, so the
extrapolated pose and `step_pose` also follow one identity. It costs ~0.05 ms
per result for a handful of people (tests/test_player.py pins the budget);
the model's cost is per frame, not per person, so detecting everyone is free.

How it works:

- **Everyone is tracked.** A light tracker keeps a track per person, matched
  to new detections by the torso anchor (shoulder midpoint) normalised by the
  person's own size, a size ratio, box overlap and how long the track has been
  unseen, greedily by lowest cost over all tracks at once. A passer-by keeps
  their own identity while they cross the player, even hidden behind them
  (their walk is predicted), so when the player is momentarily missed the
  passer's detection is already explained and is not handed to the player.
- **Gates are generous for motion, strict for size.** The anchor may move
  0.35 SW plus 6 SW per second since the track was last seen (a quick dance
  step; poses arrive 33-70 ms apart), up to 3 SW after a gap. Size may change
  only 0.8-1.25x between sightings: anyone who can hide the player is nearer
  the camera, so bigger. The locked player is also expected back at "home",
  where they have been standing, and wins ties (PLAYER_PRIOR).
- **Acquire** the most likely player: big (shoulder width, or torso length on
  a turned body), central, much of the body visible, and not walking (net
  movement over a second: walkers cover 3-4 SW, dancers stay within a step).
  A lone central candidate locks at once; otherwise the best must stay best
  for ACQUIRE_TIME, so somebody crossing the frame doesn't grab it.
- **Hold** through occlusion: the locked player out of sight mid-frame gives
  no pose (the games say what to fix) for up to HOLD_TIME, never the
  passer-by. Last seen at a side edge, the lock lets go after EDGE_HOLD; seen
  walking off (over WALK_SPEED for a second), at once.
- **Never switch for score or a brief size.** The only hand-over while the
  player is held is a sustained TAKEOVER_TIME by a still, central person
  who is much bigger (standing much nearer), or the lock being on someone off
  to the side. `repick()` (the L key) passes the lock to the next best now.
- After a release, the same person coming back within RECALL_TIME (same
  place, same size) is recognised as the player again, not as a switch.

The browser port is web/js/core/player.js, checked against this module by
tools/make_player_parity_fixture.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .config import KP
from .inference import Pose

MIN_CONF = 0.5
BODY = (0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)  # nose and the 12 limb joints

# Who can be the player.
MIN_SCALE = 0.015  # of the frame width: tiny background detections never qualify
WALK_SPEED = 2.2  # own shoulder widths per second: faster is walking past (1.3 m/s is ~3.7)
# Net displacement over a second: a walker covers 3-4 SW, a dancer stepping side
# to side ends up within a step (~1 SW) of where they were.
SPEED_WINDOW = 1.0  # s of anchor history to measure that speed over
SPEED_MIN_SPAN = 0.5  # s: shorter history leaves the speed unknown
WALK_SPAN = 0.8  # s of history before the locked player counts as walking off
# Horizontal places are |x - centre| / frame width (0 = centre, 0.5 = the edge).
CENTRAL = 0.35  # a lone candidate inside the middle 70% locks at once
# Acquisition and hand-over.
ACQUIRE_TIME = 0.3  # s the best candidate must stay best to lock
KEEP_CLAIM = 0.85  # a waiting candidate keeps its claim while within this of the best
TAKEOVER_TIME = 1.0  # s a challenger must qualify to take the lock
TAKEOVER_CENTRAL = 0.2  # a challenger must be inside the middle 40%
TAKEOVER_STILL = 1.5  # SW/s a challenger must be under
TAKEOVER_BIGGER = 1.6  # x the locked player's size: standing much nearer
TAKEOVER_ASIDE = 0.3  # the locked player in the outer 20% each side is "off to the side"
# Keeping the lock.
HOLD_TIME = 1.2  # s the lock holds an unseen player mid-frame
EDGE_HOLD = 0.3  # s when they were last seen at a side edge (walked out)
EDGE = 0.38  # the outer 12% each side counts as "at the edge"
RECALL_TIME = 4.0  # s after a release that the same player is recognised
FORGET_TIME = 1.5  # s an unmatched non-player track is kept (someone walking behind the player)
MAX_TRACKS = 16
DUPLICATE_IOU = 0.8  # boxes overlapping this much are one person reported twice
# Matching.
# A torso moves at most ~7 SW/s (a quick dance step); what it can't do is jump
# 0.8 SW in the 33 ms between two results, so the gate grows with the time
# since the track was last seen: 0.55 SW at 30 Hz, 0.75 at 15 Hz, 3 after 0.45 s.
GATE_BASE = 0.35  # SW
GATE_RATE = 6.0  # SW per second since last seen
GATE_MAX = 3.0  # SW
SIZE_MIN = 0.8  # size ratio new/old between consecutive results
SIZE_MAX = 1.25  # an occluder is nearer, so bigger: it fails this unless it is a near-twin
PRED_MAX = 0.15  # s of recent velocity used to predict the anchor (then the drift)
PRED_CAP = 0.5  # SW: a prediction never moves further than this
W_SIZE = 3.0  # cost weight of |ln size ratio|: 10% bigger costs like 0.3 SW further
W_IOU = 0.5  # cost weight of 1 - box IoU
HOME_TAU = 1.5  # s: how slowly a person's home follows them
UNSURE_SIZE = 0.3  # cost added when either size is a guess (no shoulders)
PLAYER_PRIOR = 0.15  # the person at the player's spot is the player unless clearly not
W_GAP = 0.75  # cost per second unseen: a track seen a moment ago beats one hidden for a while


@dataclass
class Body:
    """What the lock reads from one detection."""

    ax: float  # anchor: shoulder midpoint (or a fallback)
    ay: float
    scale: float  # px: max(shoulder width, 0.7 x torso length)
    sure: bool  # the scale came from the shoulders, not a guess
    shoulders: bool
    box: tuple[float, float, float, float]
    quality: float  # 0.4..1: how much of the body is visible


# Plain sqrt and left-to-right sums rather than math.hypot and sum() (which
# compensates in 3.12): the browser port then computes the very same doubles.
def _dist(dx: float, dy: float) -> float:
    return math.sqrt(dx * dx + dy * dy)


def _mean(pts: list[tuple[float, float]]) -> tuple[float, float]:
    sx = sy = 0.0
    for x, y in pts:
        sx += x
        sy += y
    return sx / len(pts), sy / len(pts)


def _vis(pose: Pose, i: int) -> bool:
    return float(pose.confidence[i]) >= MIN_CONF


def _pt(pose: Pose, i: int) -> tuple[float, float]:
    return float(pose.xy[i][0]), float(pose.xy[i][1])


def pose_box(pose: Pose) -> tuple[float, float, float, float]:
    """The detection box, or the bounding box of the visible keypoints."""
    if pose.box is not None:
        b = pose.box
        return float(b[0]), float(b[1]), float(b[2]), float(b[3])
    xs = [float(pose.xy[i][0]) for i in range(len(pose.confidence)) if _vis(pose, i)]
    ys = [float(pose.xy[i][1]) for i in range(len(pose.confidence)) if _vis(pose, i)]
    if not xs:
        return 0.0, 0.0, 0.0, 0.0
    return min(xs), min(ys), max(xs), max(ys)


def body_of(pose: Pose) -> Body:
    box = pose_box(pose)
    ls, rs = KP["left_shoulder"], KP["right_shoulder"]
    lh, rh = KP["left_hip"], KP["right_hip"]
    shoulders = _vis(pose, ls) and _vis(pose, rs)
    if shoulders:
        (x1, y1), (x2, y2) = _pt(pose, ls), _pt(pose, rs)
        ax, ay = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        sw = _dist(x1 - x2, y1 - y2)
        hips = [_pt(pose, i) for i in (lh, rh) if _vis(pose, i)]
        torso = 0.0
        if hips:
            hx, hy = _mean(hips)
            torso = _dist(hx - ax, hy - ay)
        scale, sure = max(sw, 0.7 * torso), True
    else:
        # A guess at where the shoulder midpoint is, from what is visible.
        scale, sure = 0.2 * (box[3] - box[1]), False
        hips = [_pt(pose, i) for i in (lh, rh) if _vis(pose, i)]
        if _vis(pose, ls) or _vis(pose, rs):
            ax, ay = _pt(pose, ls if _vis(pose, ls) else rs)
        elif _vis(pose, KP["nose"]):
            ax, ay = _pt(pose, KP["nose"])
            ay += 0.55 * scale
        elif hips:
            ax, ay = _mean(hips)
            ay -= 1.45 * scale
        else:
            ax, ay = (box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5
    seen = sum(1 for i in BODY if _vis(pose, i))
    return Body(ax=ax, ay=ay, scale=max(scale, 1.0), sure=sure, shoulders=shoulders, box=box,
                quality=0.4 + 0.6 * seen / len(BODY))


def iou(a, b) -> float:
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0.0 or iy <= 0.0:
        return 0.0
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0.0 else 0.0


def likeness(body: Body, width: float, speed: float | None) -> float:
    """How much this looks like the person playing: big, central, whole, not walking."""
    dx = min(1.0, abs(body.ax - width * 0.5) / (width * 0.5))
    centre = 1.0 - 0.75 * dx * dx
    motion = 1.0 if speed is None else 1.0 / (1.0 + (speed / 1.5) * (speed / 1.5))
    return (body.scale / width) * centre * body.quality * motion


def most_likely_player(poses: list[Pose], width: float) -> Pose | None:
    """The single most player-like pose, with no history (for one-off frames:
    importing a dance from a recording). Live play uses PlayerLock."""
    best, best_l = None, -1.0
    for p in poses:
        b = body_of(p)
        if not b.shoulders:
            continue
        lk = likeness(b, width, None)
        if lk > best_l:
            best, best_l = p, lk
    return best


@dataclass
class Track:
    id: int
    body: Body
    pose: Pose | None
    first_seen: float
    last_seen: float
    vx: float = 0.0
    vy: float = 0.0
    hist: list[tuple[float, float, float]] = field(default_factory=list)
    seen: bool = True
    hx: float = 0.0  # home: where this person has been standing (slow average)
    hy: float = 0.0
    x0: float = 0.0  # where they were first seen

    def drift(self, min_span: float = SPEED_MIN_SPAN) -> tuple[float, float] | None:
        """Net anchor velocity (px/s) over the last SPEED_WINDOW, None if too little history.

        A walker's drift is their walking speed; a dancer stepping side to side
        drifts little however fast each step is."""
        if len(self.hist) < 2:
            return None
        t0, x0, y0 = self.hist[0]
        t1, x1, y1 = self.hist[-1]
        span = t1 - t0
        if span < min_span:
            return None
        return (x1 - x0) / span, (y1 - y0) / span

    def speed(self, min_span: float = SPEED_MIN_SPAN) -> float | None:
        """The drift in own shoulder widths per second."""
        v = self.drift(min_span)
        return None if v is None else _dist(v[0], v[1]) / self.body.scale

    def predict(self, t: float, walking: bool = True) -> tuple[float, float]:
        """Where the anchor should be at `t`: the recent velocity for the first
        PRED_MAX (capped), then (`walking`) the drift, so somebody walking behind
        the player is expected where they come out, not where they went in."""
        gap = max(t - self.last_seen, 0.0)
        lead = min(gap, PRED_MAX)
        dx, dy = self.vx * lead, self.vy * lead
        d = _dist(dx, dy)
        cap = PRED_CAP * self.body.scale
        if d > cap:
            dx, dy = dx * cap / d, dy * cap / d
        if walking and gap > PRED_MAX:
            v = self.drift()
            if v is not None:
                dx += v[0] * (gap - PRED_MAX)
                dy += v[1] * (gap - PRED_MAX)
        return self.body.ax + dx, self.body.ay + dy

    def absorb(self, body: Body, pose: Pose, t: float) -> None:
        dt = t - self.last_seen
        if 0.0 < dt <= 0.3:
            self.vx = 0.5 * self.vx + 0.5 * (body.ax - self.body.ax) / dt
            self.vy = 0.5 * self.vy + 0.5 * (body.ay - self.body.ay) / dt
        else:
            self.vx = self.vy = 0.0
        a = min(1.0, max(dt, 0.0) / HOME_TAU)
        self.hx += a * (body.ax - self.hx)
        self.hy += a * (body.ay - self.hy)
        if body.sure and self.body.sure:
            # Follow gradual changes (turning, stepping nearer), not single-frame ones.
            body.scale = 0.5 * self.body.scale + 0.5 * body.scale
        self.body = body
        self.pose = pose
        self.last_seen = t
        self.seen = True
        if body.shoulders:
            # Speed from true shoulder midpoints only: half a body stepping in at
            # the edge has a guessed anchor that jumps, which hides a walk.
            self.hist.append((t, body.ax, body.ay))
        while self.hist and self.hist[0][0] < t - SPEED_WINDOW:
            self.hist.pop(0)


class PlayerLock:
    """Pick the main player among everyone detected, and keep following them.

    Call `update(poses, t)` once per model result (`t` = the result's
    timestamp, s). It returns the locked player's pose, or None when nobody
    is locked or the locked player is momentarily out of sight. `others` are
    everyone else (for the debug overlay); `state` is "searching", "locked"
    or "holding"; `changed` is True when the identity handed to the games
    differs from the previous result's (reset any motion history then).
    """

    def __init__(self, frame_size: tuple[int, int]) -> None:
        self.width = float(frame_size[0])
        self.height = float(frame_size[1])
        self.reset()

    def reset(self) -> None:
        self.tracks: list[Track] = []
        self._next_id = 1
        self.player_id: int | None = None
        self.player: Pose | None = None
        self.others: list[Pose] = []
        self.state = "searching"
        self.switches = 0  # times the lock moved to a different person
        self.changed = False
        self.held_for = 0.0  # s the locked player has been out of sight
        self._challenger: tuple[int, float] | None = None  # (track id, since)
        self._lost: tuple[Body, float] | None = None  # the released player's last body, last seen
        self._last_id: int | None = None  # the previous person locked
        self._repick = False
        self._given: int | None = None  # identity handed out on the previous result
        self._walked: int | None = None  # a track released for walking off

    def repick(self) -> None:
        """Pass the lock to the next most likely player on the next result (the L key)."""
        self._repick = True

    def _track(self, tid: int | None) -> Track | None:
        for tr in self.tracks:
            if tr.id == tid:
                return tr
        return None

    @property
    def locked(self) -> Track | None:
        return self._track(self.player_id)

    @property
    def player_box(self) -> tuple[float, float, float, float] | None:
        tr = self.locked
        return tr.body.box if tr is not None else None

    # --- per result -------------------------------------------------------

    def update(self, poses: list[Pose], t: float) -> Pose | None:
        bodies = [body_of(p) for p in poses]
        # The model sometimes reports one person twice (no NMS in DETR): keep
        # the first (highest-scoring) of near-identical boxes, show the rest
        # as `others`.
        keep = []
        for i, b in enumerate(bodies):
            if all(iou(bodies[k].box, b.box) < DUPLICATE_IOU for k in keep):
                keep.append(i)
        self._match([poses[i] for i in keep], [bodies[i] for i in keep], t)
        owner: list[int | None] = [None] * len(poses)
        for k, i in enumerate(keep):
            owner[i] = self._owner[k]
        self._owner = owner

        if self._repick:
            self._repick = False
            self._do_repick(t)
        tr = self.locked
        if tr is not None and tr.seen:
            # Players play on the spot. Walking for a second is leaving, or the
            # lock following somebody who walked through the player's spot
            # while the player was hidden: either way, let go.
            speed = tr.speed(WALK_SPAN)
            if speed is not None and speed > WALK_SPEED:
                self._release(tr, recall=False)
                self._walked = tr.id  # not to be locked again on the way out
                tr = None
        if tr is not None and not tr.seen:
            hold = EDGE_HOLD if abs(tr.body.ax - self.width * 0.5) >= EDGE * self.width else HOLD_TIME
            if t - tr.last_seen > hold:
                self._release(tr)
                tr = None
        if tr is None:
            self._acquire(t)
        else:
            self._consider_takeover(t)

        tr = self.locked
        seen = tr is not None and tr.seen
        self.player = tr.pose if seen else None
        self.state = "searching" if tr is None else ("locked" if seen else "holding")
        self.held_for = 0.0 if tr is None or seen else t - tr.last_seen
        self.others = [p for p, tk in zip(poses, self._owner) if tk != self.player_id or not seen]
        given = self.player_id if seen else None
        self.changed = given is not None and self._given is not None and given != self._given
        if given is not None:
            self._given = given
        return self.player

    def _match(self, poses: list[Pose], bodies: list[Body], t: float) -> None:
        pairs: list[tuple[float, int, int]] = []
        for tr in self.tracks:
            tr.seen = False
            is_player = tr.id == self.player_id
            # The player plays on the spot: predicted from where they are, never by
            # drift, and also expected back home (where they have been standing).
            px, py = tr.predict(t, walking=not is_player)
            gap = max(t - tr.last_seen, 0.0)
            gate = min(GATE_MAX, GATE_BASE + GATE_RATE * gap) * tr.body.scale
            for di, b in enumerate(bodies):
                ex, ey = px, py
                d = _dist(b.ax - px, b.ay - py)
                if is_player:
                    dh = _dist(b.ax - tr.hx, b.ay - tr.hy)
                    if dh < d:
                        ex, ey, d = tr.hx, tr.hy, dh
                if d > gate:
                    continue
                if b.sure and tr.body.sure:
                    r = b.scale / tr.body.scale
                    if r < SIZE_MIN or r > SIZE_MAX:
                        continue
                    size_cost = W_SIZE * abs(math.log(r))
                else:
                    size_cost = UNSURE_SIZE
                # The box where we expect them, against the box we got.
                sx, sy = ex - tr.body.ax, ey - tr.body.ay
                pbox = (tr.body.box[0] + sx, tr.body.box[1] + sy, tr.body.box[2] + sx, tr.body.box[3] + sy)
                cost = d / tr.body.scale + size_cost + W_IOU * (1.0 - iou(pbox, b.box)) + W_GAP * gap
                if is_player:
                    cost -= PLAYER_PRIOR
                pairs.append((cost, tr.id, di))
        pairs.sort()
        by_id = {tr.id: tr for tr in self.tracks}
        used_t: set[int] = set()
        self._owner: list[int | None] = [None] * len(bodies)
        for _cost, tid, di in pairs:
            if tid in used_t or self._owner[di] is not None:
                continue
            used_t.add(tid)
            self._owner[di] = tid
            by_id[tid].absorb(bodies[di], poses[di], t)
        for di, b in enumerate(bodies):
            if self._owner[di] is None:
                tr = Track(id=self._next_id, body=b, pose=poses[di], first_seen=t, last_seen=t,
                           hist=[(t, b.ax, b.ay)] if b.shoulders else [], hx=b.ax, hy=b.ay, x0=b.ax)
                self._next_id += 1
                self.tracks.append(tr)
                self._owner[di] = tr.id
        self.tracks = [tr for tr in self.tracks
                       if tr.seen or tr.id == self.player_id or t - tr.last_seen <= FORGET_TIME]
        if len(self.tracks) > MAX_TRACKS:
            self.tracks.sort(key=lambda tr: (tr.id != self.player_id, -tr.last_seen, tr.id))
            self.tracks = sorted(self.tracks[:MAX_TRACKS], key=lambda tr: tr.id)

    # --- choosing ----------------------------------------------------------

    def _eligible(self, tr: Track) -> bool:
        if not tr.seen or not tr.body.shoulders or tr.body.scale < MIN_SCALE * self.width:
            return False
        speed = tr.speed()
        return speed is None or speed <= WALK_SPEED

    def _dx(self, tr: Track) -> float:
        return abs(tr.body.ax - self.width * 0.5) / self.width

    def _like(self, tr: Track) -> float:
        return likeness(tr.body, self.width, tr.speed())

    def _ranked(self, exclude: int | None = None) -> list[Track]:
        cands = [tr for tr in self.tracks if self._eligible(tr) and tr.id != exclude]
        cands.sort(key=lambda tr: (-self._like(tr), tr.id))
        return cands

    def _lock(self, tr: Track, recall: bool = False) -> None:
        if not recall and self._last_id is not None and tr.id != self._last_id:
            self.switches += 1
        self.player_id = tr.id
        self._last_id = tr.id
        self._challenger = None
        self._lost = None

    def _release(self, tr: Track, recall: bool = True) -> None:
        self._lost = (tr.body, tr.last_seen) if recall else None
        self.player_id = None
        self._challenger = None

    def _acquire(self, t: float) -> None:
        if self._walked is not None:
            # Banned while walking out; welcome again once they stop.
            w = self._track(self._walked)
            gone = w is None
            speed = w.speed() if w is not None else None
            if gone or (speed is not None and speed <= TAKEOVER_STILL):
                self._walked = None
        cands = self._ranked(exclude=self._walked)
        if not cands:
            self._challenger = None
            return
        if self._lost is not None:
            lost, when = self._lost
            if t - when > RECALL_TIME:
                self._lost = None
            else:
                # The player we let go of, back where they were: the same person.
                for tr in cands:
                    d = _dist(tr.body.ax - lost.ax, tr.body.ay - lost.ay)
                    r = tr.body.scale / lost.scale
                    if d <= 1.5 * lost.scale and SIZE_MIN <= r <= SIZE_MAX:
                        self._lock(tr, recall=True)
                        return
        if len(cands) == 1 and self._dx(cands[0]) <= CENTRAL:
            # Alone and central: no contest. Unless they walked in from the
            # side and might be walking on: then once their speed is known.
            lone = cands[0]
            if lone.speed() is not None or abs(lone.x0 - self.width * 0.5) <= CENTRAL * self.width:
                self._lock(lone)
                return
        # A contest is only between people known not to be walking past: a
        # newcomer's speed is known after SPEED_MIN_SPAN in view.
        cands = [tr for tr in cands if tr.speed() is not None]
        if not cands:
            self._challenger = None
            return
        best = cands[0]
        # The candidate in waiting keeps its claim unless clearly beaten, or two
        # similar people flip-flopping for first place would never lock.
        pick = None
        if self._challenger is not None:
            pick = next((tr for tr in cands if tr.id == self._challenger[0]), None)
            if pick is not None and self._like(pick) < KEEP_CLAIM * self._like(best):
                pick = None
        if pick is None:
            # The first claim locks at once (they have been watched for
            # SPEED_MIN_SPAN already); a new leader must hold it ACQUIRE_TIME.
            first = self._challenger is None
            self._challenger = (best.id, t - ACQUIRE_TIME if first else t)
            pick = best
        if t - self._challenger[1] >= ACQUIRE_TIME:
            self._lock(pick)

    def _consider_takeover(self, t: float) -> None:
        cur = self.locked
        best = None
        for tr in self._ranked(exclude=self.player_id):
            speed = tr.speed()
            if self._dx(tr) > TAKEOVER_CENTRAL or speed is None or speed > TAKEOVER_STILL:
                continue
            bigger = tr.body.scale >= TAKEOVER_BIGGER * cur.body.scale
            aside = self._dx(cur) >= TAKEOVER_ASIDE and tr.body.scale >= 0.8 * cur.body.scale
            if bigger or aside:
                best = tr
                break
        if best is None:
            self._challenger = None
            return
        if self._challenger is None or self._challenger[0] != best.id:
            self._challenger = (best.id, t)
        elif t - self._challenger[1] >= TAKEOVER_TIME:
            self._lock(best)

    def _do_repick(self, t: float) -> None:
        cands = self._ranked(exclude=self.player_id)
        if cands:
            self._lock(cands[0])
        elif self.locked is None or not self.locked.seen:
            self.player_id = None
            self._challenger = None
            self._lost = None
