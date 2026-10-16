"""Raccoon Golf's course: holes as data, and the ground questions physics asks.

The course lives in `assets/golf/course.json` (hand-authored, original): one
entry per hole with polygons in metres. Each hole has its own frame: the tee
near the origin, +y running down the hole, +x to the right as seen from the
tee, z up. The ground is flat (z = 0) everywhere; only the green has a height
field, and it is used for roll, not for drawing.

    land     the island; anything outside it is the sea, i.e. out of bounds
    fairway  one or more polygons, mown in stripes
    green, tee
    bunkers, water (ponds and creeks: a water hazard, not OB)
    trees    {x, y, height, radius, kind}: radius is the canopy's
    path     the hole's centre line, tee -> pin (auto-aim follows it on doglegs)
    slope    green height h(x, y) = a (x - pin x) + b (y - pin y)
                                    + sum A exp(-d^2 / (2 s^2)) over bumps
    wind     [min, max] mph; each hole draws its own wind from this range

Everything here is plain floats and lists so the browser port can mirror it
line for line.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

COURSE_FILE = Path(__file__).resolve().parents[4] / "assets" / "golf" / "course.json"

YARDS_PER_M = 1.0936133
FEET_PER_M = 3.2808399
MPH = 0.44704  # m/s per mph

# Lies, in the order they win when polygons overlap (water drawn over a
# fairway is water). Anything on the island and in none of them is rough.
LIES = ("ob", "water", "bunker", "green", "tee", "fairway", "rough")


def point_in_poly(x: float, y: float, poly) -> bool:
    """Even-odd ray cast. `poly` is a list of [x, y]."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


@dataclass
class Area:
    """A polygon with a cached bounding box (most lookups miss the box)."""

    pts: list
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0

    def __post_init__(self) -> None:
        xs = [p[0] for p in self.pts]
        ys = [p[1] for p in self.pts]
        self.x0, self.x1, self.y0, self.y1 = min(xs), max(xs), min(ys), max(ys)

    def contains(self, x: float, y: float) -> bool:
        if x < self.x0 or x > self.x1 or y < self.y0 or y > self.y1:
            return False
        return point_in_poly(x, y, self.pts)


@dataclass
class Tree:
    x: float
    y: float
    height: float  # m, top of the canopy
    radius: float  # m, canopy radius (the part a ball can hit)
    kind: str  # "palm" or "round": which sprite


@dataclass
class Hole:
    number: int
    name: str
    par: int
    tee: tuple[float, float]
    pin: tuple[float, float]
    land: Area
    fairway: list[Area]
    green: Area
    tee_box: Area
    bunkers: list[Area]
    water: list[Area]
    trees: list[Tree]
    path: list[tuple[float, float]]
    slope_plane: tuple[float, float] = (0.0, 0.0)
    bumps: list[tuple[float, float, float, float]] = field(default_factory=list)  # x, y, sigma, amplitude
    wind_range: tuple[float, float] = (0.0, 8.0)  # mph

    # --- where is the ball? ---

    def lie_at(self, x: float, y: float) -> str:
        if not self.land.contains(x, y):
            return "ob"
        for a in self.water:
            if a.contains(x, y):
                return "water"
        for a in self.bunkers:
            if a.contains(x, y):
                return "bunker"
        if self.green.contains(x, y):
            return "green"
        if self.tee_box.contains(x, y):
            return "tee"
        for a in self.fairway:
            if a.contains(x, y):
                return "fairway"
        return "rough"

    def distance_to_pin(self, x: float, y: float) -> float:
        return math.hypot(self.pin[0] - x, self.pin[1] - y)

    # --- the green's height field ---

    def green_height(self, x: float, y: float) -> float:
        a, b = self.slope_plane
        h = a * (x - self.pin[0]) + b * (y - self.pin[1])
        for bx, by, s, amp in self.bumps:
            h += amp * math.exp(-((x - bx) ** 2 + (y - by) ** 2) / (2 * s * s))
        return h

    def green_gradient(self, x: float, y: float) -> tuple[float, float]:
        """d(height)/dx, d(height)/dy: a ball rolls toward minus this."""
        gx, gy = self.slope_plane
        for bx, by, s, amp in self.bumps:
            e = amp * math.exp(-((x - bx) ** 2 + (y - by) ** 2) / (2 * s * s))
            gx -= e * (x - bx) / (s * s)
            gy -= e * (y - by) / (s * s)
        return gx, gy

    # --- aiming ---

    def aim_point(self, x: float, y: float, reach: float) -> tuple[float, float]:
        """Where auto-aim points from (x, y) with a club that carries `reach` m.

        The pin if it is in range; otherwise the point on the hole's centre
        line `reach` metres away (ahead of the ball), so a tee shot on a dogleg
        aims down the fairway rather than over the trees at the green.
        """
        if self.distance_to_pin(x, y) <= reach * 1.05:
            return self.pin
        # Nearest point of the path to the ball, then walk forward from there.
        best_i, best_u, best_d = 0, 0.0, math.inf
        for i in range(len(self.path) - 1):
            (ax, ay), (bx, by) = self.path[i], self.path[i + 1]
            dx, dy = bx - ax, by - ay
            seg = dx * dx + dy * dy
            u = 0.0 if seg <= 0 else min(1.0, max(0.0, ((x - ax) * dx + (y - ay) * dy) / seg))
            d = math.hypot(ax + dx * u - x, ay + dy * u - y)
            if d < best_d:
                best_i, best_u, best_d = i, u, d
        for i in range(best_i, len(self.path) - 1):
            (ax, ay), (bx, by) = self.path[i], self.path[i + 1]
            u0 = best_u if i == best_i else 0.0
            # First point along this segment at distance `reach` from the ball.
            steps = 40
            for k in range(steps + 1):
                u = u0 + (1.0 - u0) * k / steps
                px, py = ax + (bx - ax) * u, ay + (by - ay) * u
                if math.hypot(px - x, py - y) >= reach:
                    return (px, py)
        return self.pin

    @property
    def length(self) -> float:
        """Tee to pin along the centre line, in metres (the card's yardage)."""
        return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(self.path, self.path[1:]))


def _area(pts) -> Area:
    return Area([[float(p[0]), float(p[1])] for p in pts])


def hole_from_dict(d: dict, number: int) -> Hole:
    slope = d.get("slope", {})
    return Hole(
        number=number,
        name=d.get("name", f"Hole {number}"),
        par=int(d["par"]),
        tee=(float(d["tee"][0]), float(d["tee"][1])),
        pin=(float(d["pin"][0]), float(d["pin"][1])),
        land=_area(d["land"]),
        fairway=[_area(p) for p in d.get("fairway", [])],
        green=_area(d["green"]),
        tee_box=_area(d["tee_box"]),
        bunkers=[_area(p) for p in d.get("bunkers", [])],
        water=[_area(p) for p in d.get("water", [])],
        trees=[Tree(float(t["x"]), float(t["y"]), float(t["height"]), float(t["radius"]), t.get("kind", "palm"))
               for t in d.get("trees", [])],
        path=[(float(p[0]), float(p[1])) for p in d["path"]],
        slope_plane=(float(slope.get("a", 0.0)), float(slope.get("b", 0.0))),
        bumps=[tuple(float(v) for v in b) for b in slope.get("bumps", [])],
        wind_range=(float(d.get("wind", [0, 8])[0]), float(d.get("wind", [0, 8])[1])),
    )


@dataclass
class Course:
    name: str
    holes: list[Hole]

    @property
    def par(self) -> int:
        return sum(h.par for h in self.holes)


_cache: Course | None = None


def load_course(path: Path | str | None = None) -> Course:
    """The course from its JSON (cached for the default file)."""
    global _cache
    if path is None and _cache is not None:
        return _cache
    data = json.loads(Path(path or COURSE_FILE).read_text())
    course = Course(data.get("name", "Course"), [hole_from_dict(h, i + 1) for i, h in enumerate(data["holes"])])
    if path is None:
        _cache = course
    return course
