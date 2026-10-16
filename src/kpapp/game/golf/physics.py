"""Ball flight, bounce and roll: pure, fixed-step and deterministic.

One `Flight` per stroke. It advances in fixed SUBSTEP steps whatever the frame
rate, so the same shot lands on the same spot at 30 or 120 FPS and in the
browser port (plain floats, no numpy, no randomness: any randomness - the
overswing's hook or slice - is decided by the game before launch).

The model, per substep, in metres and seconds:

    air:  a = gravity
              - DRAG |v_r| v_r                     (v_r: velocity relative to the wind)
              + LIFT spin |v_r|^2 n_up             (backspin: n_up is v_r turned 90 degrees up)
              + SIDE curve spin |v_r|^2 n_right    (hook/slice: curve -1 hook .. +1 slice)
          spin decays with SPIN_TAU.
    land: a bounce keeps RESTITUTION of the vertical speed and RETAIN of the
          horizontal (times the club's `bite` on the first bounce: a wedge's
          backspin stops it, a driver releases), until the vertical speed is
          under ROLL_BELOW, then it rolls.
    roll: constant deceleration per surface (FRICTION), plus the green's slope
          (5/7 g grad h, a rolling ball) on the green.

Clubs carry about their nominal yardage at full power with no wind (pinned in
tests/test_golf.py). Power scales the launch speed by power^POWER_EXP, chosen
so carry is close to proportional to power: the gauge reads as a percentage of
the club's distance, as in Wii golf.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .course import YARDS_PER_M, Hole

GRAVITY = 9.81
SUBSTEP = 1.0 / 120.0  # s; at 70 m/s a step is 0.6 m, finer than any tree or cup test needs
MAX_TIME = 40.0  # s of simulated time before a shot is called at rest

# Air. DRAG = rho Cd A / 2m for a golf ball is ~0.005 /m; LIFT gives a driver
# its climbing flight instead of a cannonball's arc.
DRAG = 0.0050
LIFT = 0.0042
SIDE = 0.0026
SPIN_TAU = 5.0  # s

# Ground. Per surface: vertical restitution, horizontal speed kept on the
# first bounce (the impact, before the club's bite), kept on later bounces,
# and rolling deceleration (m/s^2). Tuned so a full driver rolls ~25 m on the
# fairway and a full 7 iron stops ~4 m after pitching on the green. The
# green's 0.65 m/s^2 is a medium-fast green: a 4.6 m/s putt rolls 16 m.
SURFACES = {
    "tee": (0.30, 0.45, 0.75, 2.4),
    "fairway": (0.30, 0.45, 0.75, 2.4),
    "rough": (0.18, 0.25, 0.50, 5.5),
    "green": (0.22, 0.30, 0.70, 0.65),
    "bunker": (0.02, 0.05, 0.10, 14.0),
}
ROLL_BELOW = 1.2  # m/s of vertical speed: under this a landing ball stops bouncing
STOP_SPEED = 0.06  # m/s: a rolling ball this slow is at rest (unless the slope moves it)
SLOPE_GAIN = 5.0 / 7.0 * GRAVITY  # a rolling solid sphere on an incline

# The cup. The real cup is 0.054 m in radius; it is played 0.10 m here (about
# twice the size, like most arcade golf) because a putt's line and speed come
# from a camera-tracked arm with ~100 ms of lag. A ball rolling over it slower
# than CAPTURE_SPEED drops; faster, it skips across.
CUP_RADIUS = 0.10
CAPTURE_SPEED = 1.7
DUNK_SPEED = 9.0  # a ball landing on the cup straight out of the air drops below this

# How a lie changes a full shot's distance. Putts and green shots are not scaled.
LIE_FACTOR = {"tee": 1.0, "fairway": 1.0, "green": 1.0, "rough": 0.8, "bunker": 0.55}

POWER_EXP = 0.58  # launch speed ~ power^0.62, so carry ~ power (see the module docstring)
OVERSWING_MAX = 1.15  # top of the red zone: 15% beyond full power
THIN_DISTANCE = 0.7  # a thin strike flies this fraction of the distance...
THIN_LAUNCH = 0.6  # ...at this fraction of the launch angle


@dataclass(frozen=True)
class Club:
    key: str  # short name for the HUD: "1W", "7I", "PT"
    name: str  # long name: "DRIVER"
    carry_yd: float  # nominal full-power carry, no wind (pinned by a test)
    speed: float  # m/s launch speed at full power
    launch: float  # degrees above the ground
    spin: float  # relative backspin (lift)
    bite: float  # share of the surface's RETAIN kept on the first bounce (backspin stops wedges)
    tee_only: bool = False

    @property
    def carry_m(self) -> float:
        return self.carry_yd / YARDS_PER_M

    @property
    def putter(self) -> bool:
        return self.key == "PT"


# Speeds are fitted so each club carries its nominal yardage (test_golf.py
# checks the simulation agrees within 3%).
CLUBS: tuple[Club, ...] = (
    Club("1W", "DRIVER", 230, 74.3, 12.0, 0.82, 1.00, tee_only=True),
    Club("3W", "3 WOOD", 210, 66.7, 13.5, 0.90, 0.85),
    Club("5I", "5 IRON", 180, 56.5, 17.0, 1.00, 0.60),
    Club("7I", "7 IRON", 155, 48.6, 21.0, 1.10, 0.40),
    Club("9I", "9 IRON", 130, 41.6, 27.0, 1.20, 0.33),
    Club("PW", "WEDGE", 105, 35.5, 33.0, 1.30, 0.28),
    Club("SW", "SAND WEDGE", 75, 29.0, 42.0, 1.35, 0.22),
    Club("PT", "PUTTER", 0, 0.0, 0.0, 0.0, 1.0),
)
PUTTER = CLUBS[-1]
# Full-power putt: a 4.6 m/s roll, 16 m (52 ft) on a flat green. Power maps to
# speed as sqrt(power), so roll distance is proportional to power.
PUTT_SPEED = 4.6

# Hook/slice from an overswing: `curve` 1.0 bends a full driver ~30 m.
CURVE_MAX = 1.0


def club_by_key(key: str) -> Club:
    for c in CLUBS:
        if c.key == key:
            return c
    raise KeyError(key)


def putt_range_m(power: float = 1.0, decel: float = SURFACES["green"][3]) -> float:
    v = PUTT_SPEED * math.sqrt(max(0.0, power))
    return v * v / (2 * decel)


@dataclass
class Shot:
    club: Club
    power: float  # 0..OVERSWING_MAX
    heading: float  # radians, 0 = +y (down the hole), positive turns right (+x)
    lie: str = "fairway"
    quality: float = 1.0  # 1 clean .. 0 thin
    curve: float = 0.0  # -1 hook .. +1 slice (for a right-handed swing); from overswing
    wind: tuple[float, float] = (0.0, 0.0)  # m/s, world x/y


@dataclass
class Flight:
    """One stroke from launch to rest. Call `step(dt)` with any dt; it substeps."""

    hole: Hole
    start: tuple[float, float]
    shot: Shot
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    spin: float = 0.0
    curve: float = 0.0
    t: float = 0.0
    state: str = "air"  # air, roll, rest, holed, water, ob
    bounces: int = 0
    landing: tuple[float, float] | None = None  # first touch of the ground
    apex: float = 0.0
    water_entry: tuple[float, float] | None = None
    path: list = field(default_factory=list)  # (x, y, z) every PATH_EVERY substeps
    hit_trees: set = field(default_factory=set)
    _acc: float = 0.0
    _n: int = 0
    _surface: str = "fairway"

    PATH_EVERY = 4  # path sample every 4 substeps (30 per second)

    def __post_init__(self) -> None:
        s = self.shot
        self.x, self.y = self.start
        sh, ch = math.sin(s.heading), math.cos(s.heading)
        if s.club.putter:
            v = PUTT_SPEED * math.sqrt(max(0.0, s.power))
            self.vx, self.vy, self.vz = v * sh, v * ch, 0.0
            self.state = "roll"
            self._surface = self.hole.lie_at(self.x, self.y)
        else:
            scale = max(0.0, s.power) * LIE_FACTOR.get(s.lie, 1.0)
            q = min(1.0, max(0.0, s.quality))
            scale *= THIN_DISTANCE + (1.0 - THIN_DISTANCE) * q
            v = s.club.speed * scale ** POWER_EXP
            launch = math.radians(s.club.launch * (THIN_LAUNCH + (1.0 - THIN_LAUNCH) * q))
            self.vx, self.vy = v * math.cos(launch) * sh, v * math.cos(launch) * ch
            self.vz = v * math.sin(launch)
            self.spin = s.club.spin
            self.curve = s.curve * CURVE_MAX
        self.path.append((self.x, self.y, self.z))

    # --- public ---

    @property
    def done(self) -> bool:
        return self.state in ("rest", "holed", "water", "ob")

    @property
    def speed(self) -> float:
        return math.sqrt(self.vx * self.vx + self.vy * self.vy + self.vz * self.vz)

    @property
    def carry(self) -> float:
        """Distance from the start to the first touch of the ground (m)."""
        if self.landing is None:
            return 0.0
        return math.hypot(self.landing[0] - self.start[0], self.landing[1] - self.start[1])

    def step(self, dt: float) -> None:
        self._acc += dt
        while self._acc >= SUBSTEP - 1e-12 and not self.done:
            self._acc -= SUBSTEP
            self._substep(SUBSTEP)
        if self.done:
            self._acc = 0.0

    def run(self) -> "Flight":
        """Simulate to the end (bots, previews, tests)."""
        while not self.done:
            self._substep(SUBSTEP)
        return self

    # --- the model ---

    def _substep(self, h: float) -> None:
        self.t += h
        self._n += 1
        if self.state == "air":
            self._fly(h)
        elif self.state == "roll":
            self._roll(h)
        if self.t >= MAX_TIME and not self.done:
            self._settle()
        if self._n % self.PATH_EVERY == 0 or self.done:
            self.path.append((self.x, self.y, self.z))

    def _fly(self, h: float) -> None:
        wx, wy = self.shot.wind
        rx, ry, rz = self.vx - wx, self.vy - wy, self.vz
        sp = math.sqrt(rx * rx + ry * ry + rz * rz)
        hs = math.hypot(rx, ry)
        ax = -DRAG * sp * rx
        ay = -DRAG * sp * ry
        az = -GRAVITY - DRAG * sp * rz
        if hs > 1e-6 and sp > 1e-6:
            # Up, perpendicular to the velocity: (-rx rz, -ry rz, rx^2 + ry^2) / (|v| |v_h|).
            k = LIFT * self.spin * sp / hs
            ax += k * (-rx * rz)
            ay += k * (-ry * rz)
            az += k * (hs * hs)
            # Right of travel, horizontal: (ry, -rx) / |v_h|.
            c = SIDE * self.curve * self.spin * sp * sp / hs
            ax += c * ry
            ay += c * -rx
        self.vx += ax * h
        self.vy += ay * h
        self.vz += az * h
        px, py = self.x, self.y
        self.x += self.vx * h
        self.y += self.vy * h
        self.z += self.vz * h
        self.spin *= math.exp(-h / SPIN_TAU)
        self.apex = max(self.apex, self.z)
        self._check_trees()
        if self.z <= 0.0:
            # Back up to the exact touchdown point along this substep.
            f = (self.z - self.vz * h)
            u = f / (f - self.z) if f - self.z > 1e-9 else 1.0
            self.x = px + (self.x - px) * u
            self.y = py + (self.y - py) * u
            self.z = 0.0
            self._touch()

    def _check_trees(self) -> None:
        for i, tr in enumerate(self.hole.trees):
            if i in self.hit_trees:
                continue
            if self.z > tr.height or self.z < tr.height * 0.35:
                continue  # over it, or under the canopy (trunks are thin)
            if (self.x - tr.x) ** 2 + (self.y - tr.y) ** 2 <= tr.radius * tr.radius:
                # Knocked down: most of the speed gone, a little bounce back.
                self.hit_trees.add(i)
                self.vx *= -0.18
                self.vy *= -0.18
                self.vz = min(self.vz, 0.0) * 0.3
                self.spin = 0.0
                self.curve = 0.0

    def _touch(self) -> None:
        """The ball meets the ground."""
        lie = self.hole.lie_at(self.x, self.y)
        if self.landing is None:
            self.landing = (self.x, self.y)
        if lie == "ob":
            self.state = "ob"
            self.vx = self.vy = self.vz = 0.0
            return
        if lie == "water":
            self.state = "water"
            self.water_entry = (self.x, self.y)
            self.vx = self.vy = self.vz = 0.0
            return
        if self.hole.distance_to_pin(self.x, self.y) <= CUP_RADIUS and self.speed < DUNK_SPEED:
            self._hole_out()
            return
        e, first, keep, _ = SURFACES.get(lie, SURFACES["rough"])
        if self.bounces == 0:
            keep = first * self.shot.club.bite
        self.bounces += 1
        self.vx *= keep
        self.vy *= keep
        self.spin *= 0.4
        self.curve *= 0.3
        if -self.vz * e > ROLL_BELOW:
            self.vz = -self.vz * e
        else:
            self.vz = 0.0
            self.state = "roll"
        self._surface = lie

    def _roll(self, h: float) -> None:
        lie = self.hole.lie_at(self.x, self.y)
        if lie == "ob":
            self.state = "ob"
            self.vx = self.vy = 0.0
            return
        if lie == "water":
            self.state = "water"
            self.water_entry = (self.x, self.y)
            self.vx = self.vy = 0.0
            return
        self._surface = lie
        decel = SURFACES.get(lie, SURFACES["rough"])[3]
        sx = sy = 0.0
        if lie == "green":
            gx, gy = self.hole.green_gradient(self.x, self.y)
            sx, sy = -SLOPE_GAIN * gx, -SLOPE_GAIN * gy
        sp = math.hypot(self.vx, self.vy)
        if sp < STOP_SPEED and math.hypot(sx, sy) < decel:
            self._settle()
            return
        if sp > 1e-9:
            # Friction never reverses the ball: clamp it at a stop.
            dv = min(sp, decel * h)
            self.vx -= self.vx / sp * dv
            self.vy -= self.vy / sp * dv
        self.vx += sx * h
        self.vy += sy * h
        px, py = self.x, self.y
        self.x += self.vx * h
        self.y += self.vy * h
        # Over the cup slowly enough to drop?
        if lie == "green" or self.hole.distance_to_pin(self.x, self.y) < 1.0:
            if (_seg_dist(px, py, self.x, self.y, *self.hole.pin) <= CUP_RADIUS
                    and math.hypot(self.vx, self.vy) <= CAPTURE_SPEED):
                self._hole_out()

    def _hole_out(self) -> None:
        self.x, self.y = self.hole.pin
        self.z = 0.0
        self.vx = self.vy = self.vz = 0.0
        self.state = "holed"

    def _settle(self) -> None:
        self.vx = self.vy = self.vz = 0.0
        self.z = 0.0
        self.state = "rest"

    @property
    def lie(self) -> str:
        """Where the ball is now (meaningful once it is at rest)."""
        if self.state == "holed":
            return "green"
        return self.hole.lie_at(self.x, self.y)


def _seg_dist(ax: float, ay: float, bx: float, by: float, px: float, py: float) -> float:
    dx, dy = bx - ax, by - ay
    seg = dx * dx + dy * dy
    u = 0.0 if seg <= 1e-12 else min(1.0, max(0.0, ((px - ax) * dx + (py - ay) * dy) / seg))
    return math.hypot(ax + dx * u - px, ay + dy * u - py)


def simulate(hole: Hole, start: tuple[float, float], shot: Shot) -> Flight:
    return Flight(hole, start, shot).run()


def preview(hole: Hole, start: tuple[float, float], club: Club, heading: float, lie: str,
            power: float = 1.0) -> Flight:
    """A straight, windless shot through the air only, for the landing target and
    the dotted arc. Stops at the first touch of the ground."""
    f = Flight(hole, start, Shot(club, power, heading, lie))
    if club.putter:
        return f.run()
    while f.state == "air" and f.t < MAX_TIME:
        f._substep(SUBSTEP)
    return f


def drop_point(hole: Hole, flight: Flight) -> tuple[float, float]:
    """Where to play from after a water hazard: on dry land just short of where
    the ball crossed into the water (along its ground track), never nearer the
    hole than that, as in a lateral drop."""
    entry = flight.water_entry or (flight.x, flight.y)
    track = [(p[0], p[1]) for p in flight.path] + [entry]
    # Walk back along the track from the entry to the last dry point.
    for i in range(len(track) - 1, 0, -1):
        x, y = track[i - 1]
        if hole.lie_at(x, y) not in ("water", "ob"):
            # Two more metres back from the edge, along the track.
            bx, by = track[max(0, i - 2)]
            d = math.hypot(x - bx, y - by)
            if d > 1e-6:
                k = min(1.0, 2.0 / d)
                cx, cy = x + (bx - x) * k, y + (by - y) * k
                if hole.lie_at(cx, cy) not in ("water", "ob"):
                    return (cx, cy)
            return (x, y)
    return flight.start
