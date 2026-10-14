"""Judging a dance: timing windows, ratings, score and stars (docs/DANCE.md section 7).

Every judged keyframe in a song is a judge point at time `t`. While the song
time is inside `[t - WINDOW_EARLY, t + WINDOW_LATE]`, the judge keeps the best
similarity between the point's pose and the player that was *held* for HOLD
seconds. When the window closes, the point resolves into a rating.

Why "best in a window" rather than "the pose at t": poses reach the game in
steps (15-30 Hz) and 100-150 ms late (AGENTS.md sections 2 and 6), so the
frame that shows the player hitting a pose lands at some unpredictable moment
after the beat. The window is skewed late for the same reason.

A point the judge never got a usable look at (no shoulders, no arms) is
UNSEEN: no rating on screen, no points, and never a miss. Tracking loss is
the camera's fault, not the player's.
"""

from __future__ import annotations

from dataclasses import dataclass

from .pose import Measured, similarity

# --- the numbers (docs/DANCE.md section 7; pinned by tests/test_dance_game.py) ---

WINDOW_EARLY = 0.20  # s before the point's time that a pose already counts
# s after it. The spec started at 0.40; simulated through the real input path
# (20 Hz, 150 ms lag, extrapolation) a flawless dancer just 250 ms behind the
# coach then took 15 X on Neon Nights (7138), because the late edge minus
# the lag minus HOLD left under 0.2 s for the player. At 0.50 that dancer is
# PERFECT throughout, and the frozen, idle, random and wrong-side players
# gain almost nothing (docs/DANCE.md section 7).
WINDOW_LATE = 0.50
# A match only counts once it has been *held* this long (song time): the
# window's score is the best, over the window, of the worst similarity in the
# last HOLD seconds. Without it, a limb sweeping through the target angle on
# its way somewhere else scores a momentary PERFECT ("require a hold so
# passing hands don't trigger things", AGENTS.md section 6).
HOLD = 0.12

# (name, minimum best similarity, value). Checked in order; below OK is X.
# PERFECT is 0.90 (the spec started at 0.86). A segment scores
# 1 - (d - 20) / 70, so 0.86 meant "every limb within 30 degrees": a dancer
# holding every limb 25-30 degrees off took PERFECT on 38% of the moves. At
# 0.90 (about 27 degrees) that dancer is SUPER (88%), while one 10-20 degrees
# off and 200 ms late at 15 Hz stays a 5-star PERFECT dancer. 0.92 cost that
# dancer its fifth star. The other numbers are the spec's.
RATINGS = (
    ("perfect", 0.90, 1.0),
    ("super", 0.74, 0.8),
    ("good", 0.60, 0.55),
    ("ok", 0.42, 0.3),
)
MISS = ("x", 0.0)
UNSEEN = "unseen"
RATING_NAMES = tuple(r[0] for r in RATINGS) + (MISS[0],)
COMBO_FROM = ("perfect", "super")  # ratings that extend the combo

SCORE_MAX = 10000
STAR_SCORES = (2000, 4000, 6000, 8000, 9200)  # 1..5 stars
SUPERSTAR = 9600

# After a pause the player's hand is on the shell's RESUME button, not in the
# move. Points whose window is open at the resume, or opens within this many
# seconds after it, are "excused": what was seen before the pause is
# forgotten (no free PERFECT from a pose held before pausing), and if what is
# seen after it would be an X, the point is unseen instead.
RESUME_GRACE = 1.0

# Tracking loss is never a miss (AGENTS.md section 6). A window that loses the
# player before this many seconds past its point's time, and never sees them
# again, saw at most the player on their way into the pose (they reach the game
# 100-150 ms late): if that would be an X, the point is unseen instead.
LOST_BEFORE = WINDOW_LATE / 2


def rate(sim: float) -> tuple[str, float]:
    """A best similarity -> (rating name, value)."""
    for name, need, value in RATINGS:
        if sim >= need:
            return name, value
    return MISS


def stars_for(score: float) -> int:
    return sum(1 for s in STAR_SCORES if score >= s)


def star_fill(score: float) -> float:
    """0..5, continuous: the star bar's fill. Each star's segment fills
    linearly from the previous threshold to its own."""
    lo, out = 0.0, 0.0
    for need in STAR_SCORES:
        if score >= need:
            out += 1.0
            lo = need
            continue
        return out + max(0.0, score - lo) / (need - lo)
    return out


@dataclass(frozen=True)
class Rating:
    name: str  # "perfect" | "super" | "good" | "ok" | "x" | "unseen"
    value: float  # 1.0 / 0.8 / 0.55 / 0.3 / 0, and 0 for unseen
    weight: float  # 2 for a gold point, else 1
    similarity: float | None  # the best seen; None if unseen
    point: object  # the chart.JudgePoint
    time: float  # song time it resolved at

    @property
    def gold(self) -> bool:
        return bool(getattr(self.point, "gold", False))

    @property
    def seen(self) -> bool:
        return self.name != UNSEEN

    @property
    def combo(self) -> bool:
        return self.name in COMBO_FROM


class _Window:
    __slots__ = ("best", "excused", "floor", "last_seen", "point", "recent")

    def __init__(self, point) -> None:
        self.point = point
        self.best: float | None = None  # best similarity held for HOLD seconds
        self.floor: float | None = None  # worst sample seen, for a window seen too briefly to hold
        self.recent: list[tuple[float, float]] = []  # (song time, similarity), the last HOLD s
        self.excused = False
        self.last_seen: float | None = None  # song time of the last measurable sample

    def add(self, t: float, sim: float) -> None:
        self.floor = sim if self.floor is None else min(self.floor, sim)
        self.last_seen = t
        r = self.recent
        r.append((t, sim))
        # Keep the newest sample at or before t - HOLD, so the kept run spans HOLD.
        while len(r) > 1 and r[1][0] <= t - HOLD:
            del r[0]
        if t - r[0][0] >= HOLD - 1e-9:
            held = min(s for _, s in r)
            if self.best is None or held > self.best:
                self.best = held

    def forget(self) -> None:
        self.best = self.floor = self.last_seen = None
        self.recent.clear()

    def result(self) -> float | None:
        return self.best if self.best is not None else self.floor


class Judge:
    """Feed it `update(song_time, measured)` every frame; it returns the
    ratings that resolved on that frame (for juice)."""

    def __init__(self, song) -> None:
        self.song = song
        self.points = list(song.judge_points())
        self.total_weight = sum(p.weight for p in self.points)
        self.reset()

    def reset(self) -> None:
        self._next = 0  # index of the first point not yet opened
        self._open: list[_Window] = []
        self.results: list[Rating] = []
        self.counts = {n: 0 for n in RATING_NAMES}
        self.unseen = 0
        self.earned = 0.0  # sum of value * weight
        self.combo = 0
        self.best_combo = 0
        self.last_rating: str | None = None  # the last *seen* rating
        self.song_time = -1e9
        self._excuse_until = -1e9

    # --- state ---

    @property
    def score(self) -> int:
        if self.total_weight <= 0:
            return 0
        return round(SCORE_MAX * self.earned / self.total_weight)

    @property
    def stars(self) -> int:
        return stars_for(self.score)

    @property
    def star_fill(self) -> float:
        return star_fill(self.score)

    @property
    def superstar(self) -> bool:
        return self.score >= SUPERSTAR

    @property
    def done(self) -> bool:
        return len(self.results) == len(self.points)

    @property
    def open_points(self) -> list:
        """Points whose window contains the current song time."""
        return [w.point for w in self._open]

    def judging(self, song_time: float | None = None) -> bool:
        """Is any window open at `song_time` (default: the last update)?"""
        t = self.song_time if song_time is None else song_time
        return any(p.time - WINDOW_EARLY <= t <= p.time + WINDOW_LATE for p in self.points[:self._next + 4])

    # --- feeding ---

    def update(self, song_time: float, measured: Measured | None) -> list[Rating]:
        self.song_time = song_time
        # Open every window that has started.
        while self._next < len(self.points) and song_time >= self.points[self._next].time - WINDOW_EARLY:
            w = _Window(self.points[self._next])
            w.excused = w.point.time - WINDOW_EARLY <= self._excuse_until
            self._open.append(w)
            self._next += 1
        resolved: list[Rating] = []
        still: list[_Window] = []
        for w in self._open:
            if song_time > w.point.time + WINDOW_LATE:
                # Closed. A frame that jumped past the whole window (a stall)
                # saw nothing of it: unseen, never a miss.
                resolved.append(self._resolve(w, song_time))
                continue
            if measured is not None:
                sim, _ = similarity(w.point.pose, measured)
                if sim is not None:  # an unmeasurable frame neither breaks nor extends a hold
                    w.add(song_time, sim)
            still.append(w)
        self._open = still
        return resolved

    def on_resume(self, song_time: float) -> None:
        """Forget what open windows saw before a pause, and excuse the points
        near the resume from X ratings (see RESUME_GRACE)."""
        self._excuse_until = song_time + RESUME_GRACE
        for w in self._open:
            w.forget()
            w.excused = True

    def _resolve(self, w: _Window, song_time: float) -> Rating:
        p = w.point
        best = w.result()
        if best is None:
            name, value = UNSEEN, 0.0
        else:
            name, value = rate(best)
            lost = w.last_seen is not None and w.last_seen < p.time + LOST_BEFORE
            if name == MISS[0] and (w.excused or lost):
                name, value = UNSEEN, 0.0
        r = Rating(name, value, p.weight, best if name != UNSEEN else None, p, song_time)
        self.results.append(r)
        if name == UNSEEN:
            self.unseen += 1  # the combo carries on: tracking loss is never punished
        else:
            self.counts[name] += 1
            self.earned += value * p.weight
            self.last_rating = name
            self.combo = self.combo + 1 if r.combo else 0
            self.best_combo = max(self.best_combo, self.combo)
        return r
