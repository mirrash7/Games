"""Songs and moves: loading, validation, beat <-> time, the judge timeline and
the coach's animation (docs/DANCE.md sections 4-5).

A song folder holds `song.json` (metadata + chart) and its audio. Moves come
from the shared library (`assets/dance/moves.json`) or the song's own
`custom_moves`. Everything is validated on load, with errors that name the
song, move and beat, so a hand-edited or imported chart fails loudly.

Every song plays on three difficulties (docs/DANCE.md section 7a). NORMAL and
HARD dance the song's chart (`moves`); EASY dances a chart derived from it,
one held pose per bar (`derive_easy`). A song may hand-make either one in
`song.json` under `"charts": {"easy": [...], "hard": [...]}`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

from .pose import (FULL_BODY, REST, forward, keypoints_for, leg_target, lerp_pose, measure, mirror_pose,
                   normalize_pose, similarity, smoothstep)

ASSET_DIR = Path(__file__).resolve().parents[4] / "assets" / "dance"
LIBRARY = ASSET_DIR / "moves.json"
SONGS = ASSET_DIR / "songs"

# The coach reaches each keyframe exactly on its beat, holding it until it
# eases into the next one over this many beats (or the whole gap if shorter).
EASE_BEATS = 0.75

DIFFICULTIES = ("easy", "normal", "hard")
DEFAULT_DIFFICULTY = "normal"
BAR_BEATS = 4  # EASY judges about one pose per bar
EASY_MIN_GAP = 2.0  # beats: EASY never judges two poses closer than this
# The freeze rule: a player frozen in the previous judged pose must score
# below this on the next one: seen whole (feet too), with the feet out of
# frame (arms alone, which is how a close player is judged), and without elbows
# (tests/test_dance_chart.py checks it for every difficulty of every song).
FREEZE_MAX = 0.55
_ELBOWS_HIDDEN = frozenset({"shoulder", "wrist", "hip"})


def difficulty_id(difficulty) -> str:
    """A known difficulty id; anything else (None, typos) is normal."""
    d = str(difficulty).lower() if difficulty is not None else DEFAULT_DIFFICULTY
    return d if d in DIFFICULTIES else DEFAULT_DIFFICULTY


def freeze_score(target: dict, held: dict) -> float:
    """How well a perfect player frozen in `held` matches `target`: the worst
    (highest) of the whole body seen, the feet out of frame (arms only) and the
    elbows hidden. Below FREEZE_MAX means a player can't score on `target` by
    standing still in `held`, however much of them the camera sees."""
    body = similarity(target, measure(keypoints_for(held, **FULL_BODY)))[0]
    # Arms only: unseen legs are left out (None), not given play's half credit.
    arms = similarity(target, measure(keypoints_for(held)), None)[0]
    reach = similarity(target, measure(keypoints_for(held, visible=set(_ELBOWS_HIDDEN))), None)[0]
    return max(body or 0.0, arms or 0.0, reach or 0.0)


class ChartError(ValueError):
    pass


@dataclass(frozen=True)
class Keyframe:
    beat: float  # offset within the move
    pose: dict  # complete (normalized) pose
    label: str | None = None  # this pose's own name on its card ("Y", "M"...); else the move's label


@dataclass(frozen=True)
class Move:
    name: str
    label: str
    beats: float
    keyframes: tuple[Keyframe, ...]
    judge: tuple[int, ...]  # keyframe indexes that are judged
    pictogram: int  # keyframe index for the move's thumbnail (kp dance-preview); the lane shows every judged keyframe


@dataclass(frozen=True)
class Placed:
    """A move placed in a song's chart."""
    move: Move
    beat: float
    gold: bool = False

    @property
    def end(self) -> float:
        return self.beat + self.move.beats


@dataclass(frozen=True)
class JudgePoint:
    index: int
    beat: float
    time: float  # seconds since beat 0
    pose: dict
    gold: bool
    weight: float
    label: str
    placed: int  # index into Song.chart


@dataclass(frozen=True)
class Pictogram:
    beat: float  # when to hit it (absolute beat)
    label: str
    pose: dict
    gold: bool


@dataclass
class Song:
    id: str
    title: str
    artist: str
    bpm: float
    offset: float
    duration: float
    audio: Path
    difficulty: int
    coach: dict
    sections: list[dict]
    chart: list[Placed]
    preview: tuple[float, float] | None = None
    folder: Path | None = None
    level: str = DEFAULT_DIFFICULTY  # the difficulty this chart is for ("easy" | "normal" | "hard")
    charts: dict[str, list[Placed]] = field(default_factory=dict, repr=False)  # "normal" + hand-made ones
    _frames: list[tuple[float, dict]] = field(default_factory=list, repr=False)
    _judges: list[JudgePoint] = field(default_factory=list, repr=False)

    # --- time ---

    @property
    def beat_seconds(self) -> float:
        return 60.0 / self.bpm

    def time_of(self, beat: float) -> float:
        return beat * self.beat_seconds

    def beat_of(self, t: float) -> float:
        return t / self.beat_seconds

    def section_at(self, beat: float) -> str:
        name = ""
        for s in self.sections:
            if s["beat"] <= beat:
                name = s["name"]
        return name

    # --- difficulties ---

    def chart_for(self, difficulty: str) -> list[Placed]:
        """The chart a difficulty dances: a hand-made one from song.json's
        `charts` if there is one, else EASY derives from the normal chart and
        HARD dances the normal chart (judged strictly: judge.DIFFICULTY)."""
        level = difficulty_id(difficulty)
        normal = self.charts.get(DEFAULT_DIFFICULTY, self.chart)
        if level in self.charts:
            return list(self.charts[level])
        if level == "easy":
            return derive_easy(normal)
        return list(normal)

    def for_difficulty(self, difficulty: str) -> "Song":
        """This song on another difficulty: same metadata, that difficulty's chart."""
        level = difficulty_id(difficulty)
        if level == self.level:
            return self
        charts = self.charts or {DEFAULT_DIFFICULTY: list(self.chart)}
        song = replace(self, level=level, chart=[], charts=charts, _frames=[], _judges=[])
        song.chart = song.chart_for(level)
        _index(song)
        return song

    # --- the chart, flattened ---

    def judge_points(self) -> list[JudgePoint]:
        return self._judges

    @property
    def first_move_beat(self) -> float:
        return self.chart[0].beat if self.chart else 0.0

    @property
    def last_judge_time(self) -> float:
        return self._judges[-1].time if self._judges else 0.0

    def pictograms(self) -> list[Pictogram]:
        """The lane's cards: one per judge point, showing the pose judged on
        that beat under its move's label, so card, label and judge always agree.
        (One card per move used to show only the move's `pictogram` keyframe:
        a two-pose move's second judged pose had no card, and "Point Up" was
        labelled over a horizontal point. `Move.pictogram` now only picks the
        move's thumbnail in `kp dance-preview`.)"""
        return [Pictogram(j.beat, j.label, j.pose, j.gold) for j in self._judges]

    def coach_pose(self, beat: float) -> dict:
        """The coach's pose at a (fractional) beat: idle groove before the first
        move, eased keyframes during the chart, the last pose held after it."""
        frames = self._frames
        if not frames or beat < frames[0][0] - EASE_BEATS:
            return idle_groove(beat)
        if beat < frames[0][0]:
            u = smoothstep((beat - (frames[0][0] - EASE_BEATS)) / EASE_BEATS)
            return lerp_pose(idle_groove(beat), frames[0][1], u)
        lo, hi = 0, len(frames) - 1
        if beat >= frames[hi][0]:
            return frames[hi][1]
        while hi - lo > 1:  # last frame at or before `beat`
            mid = (lo + hi) // 2
            if frames[mid][0] <= beat:
                lo = mid
            else:
                hi = mid
        (b0, p0), (b1, p1) = frames[lo], frames[lo + 1]
        ease = min(EASE_BEATS, b1 - b0)
        u = smoothstep((beat - (b1 - ease)) / ease) if ease > 0 else 1.0
        return lerp_pose(p0, p1, u)


def _index(song: Song) -> None:
    """Flatten the chart into the coach's keyframes and the judge points."""
    song._frames, song._judges = [], []
    for p_idx, p in enumerate(song.chart):
        for k_idx, k in enumerate(p.move.keyframes):
            b = p.beat + k.beat
            song._frames.append((b, k.pose))
            if k_idx in p.move.judge:
                song._judges.append(JudgePoint(len(song._judges), b, song.time_of(b), k.pose, p.gold,
                                               2.0 if p.gold else 1.0, k.label or p.move.label, p_idx))


def derive_easy(chart: list[Placed], bar: float = BAR_BEATS) -> list[Placed]:
    """EASY's chart from a normal one: about one judged pose per bar, held.

    - **Short moves** (under a bar) are grouped by the bar they start in. Each
      group becomes one *held* move: the pose kept for the group, from its beat
      to the group's end, with no in-between keyframes. The coach then holds
      it for the bar, so what the player sees is what is asked.
    - **Which pose:** a gold one first, else the earliest. A candidate is
      skipped when it is closer than EASY_MIN_GAP beats to the previous kept
      pose, or when a player frozen in the previous kept pose would score on it
      (`freeze_score` >= FREEZE_MAX): the freeze rule holds on EASY too. A
      group with no candidate left adds nothing (the coach holds on).
    - **Gold:** a group that had a gold point keeps a gold point, so every
      section with gold in normal keeps gold.
    - **Slow moves** (a bar or longer whose judged poses are at least
      EASY_MIN_GAP beats apart: the bridge's 4-beat moves, the final pose)
      and **unjudged moves** (the bow) stay as they are. The coach dances
      them in full and their judged poses stay judged, so every pose the coach
      holds is one the judge asks for. (Judging one pose per bar of them left
      the coach visibly hitting poses that no longer counted.) The same gap and
      freeze checks apply; a pose that fails them is no longer judged. A bar
      long move that judges every beat (a grapevine) is grouped like the short
      moves: one held pose for its bar.
    """
    out: list[Placed] = []
    prev: tuple[float, dict] | None = None  # (beat, pose) of the last kept judge point

    def pick(cands):
        nonlocal prev
        for c in sorted(cands, key=lambda c: (not c[2], c[0])):  # (beat, pose, gold, ...)
            if prev is not None and (c[0] - prev[0] < EASY_MIN_GAP - 1e-9
                                     or freeze_score(c[1], prev[1]) >= FREEZE_MAX):
                continue
            prev = (c[0], c[1])
            return c
        return None

    def bar_of(beat):
        return math.floor(beat / bar + 1e-9)

    def slow(m: Move) -> bool:
        if not m.judge:
            return True
        beats = [m.keyframes[k].beat for k in m.judge]
        return m.beats >= bar - 1e-9 and all(b - a >= EASY_MIN_GAP - 1e-9 for a, b in zip(beats, beats[1:]))

    i, n = 0, len(chart)
    while i < n:
        p = chart[i]
        if slow(p.move):
            keep: list[int] = []
            for k in p.move.judge:
                c = pick([(p.beat + p.move.keyframes[k].beat, p.move.keyframes[k].pose, p.gold, k)])
                if c is not None:
                    keep.append(k)
            move = p.move if tuple(keep) == p.move.judge else replace(p.move, judge=tuple(keep))
            out.append(Placed(move, p.beat, p.gold and bool(keep)))
            i += 1
            continue
        group = [p]
        i += 1
        while (i < n and not slow(chart[i].move) and bar_of(chart[i].beat) == bar_of(p.beat)):
            group.append(chart[i])
            i += 1
        cands = [(g.beat + g.move.keyframes[k].beat, g.move.keyframes[k].pose, g.gold, g, g.move.keyframes[k].label)
                 for g in group for k in g.move.judge]
        c = pick(cands)
        if c is None:
            continue
        beat, pose, gold, src, klabel = c
        end = max(g.end for g in group)
        held = Move(f"{src.move.name}~held", src.move.label, end - beat, (Keyframe(0.0, pose, klabel),), (0,), 0)
        out.append(Placed(held, beat, gold or any(g.gold for g in group)))
    return out


def idle_groove(beat: float) -> dict:
    """Bob to the music while waiting: a sway on the beat, arms loose."""
    sway = math.sin(beat * math.pi)
    pose = normalize_pose(REST)
    pose["lean"] = 4.0 * sway
    pose["head"] = -3.0 * sway
    pose["la"] = [16.0 + 6.0 * sway, 24.0 + 10.0 * sway]
    pose["ra"] = [16.0 - 6.0 * sway, 24.0 - 10.0 * sway]
    pose["squat"] = 0.08 * (0.5 + 0.5 * math.cos(beat * 2 * math.pi))
    return pose


# --- loading ---------------------------------------------------------------

def _parse_move(name: str, spec: dict, where: str, known: dict[str, Move]) -> Move:
    if not isinstance(spec, dict):
        raise ChartError(f"{where}: move '{name}' must be an object")
    if "mirror_of" in spec:
        src = known.get(spec["mirror_of"])
        if src is None:
            raise ChartError(f"{where}: move '{name}' mirrors unknown move '{spec['mirror_of']}'")
        label = spec.get("label", src.label)
        return Move(name, label, src.beats,
                    tuple(Keyframe(k.beat, normalize_pose(mirror_pose(k.pose)), k.label) for k in src.keyframes),
                    src.judge, src.pictogram)
    try:
        beats = float(spec["beats"])
        raw = spec["keyframes"]
    except (KeyError, TypeError, ValueError):
        raise ChartError(f"{where}: move '{name}' needs 'beats' and 'keyframes'") from None
    if beats <= 0 or not isinstance(raw, list) or not raw:
        raise ChartError(f"{where}: move '{name}' needs beats > 0 and at least one keyframe")
    frames, last = [], -1.0
    for i, k in enumerate(raw):
        try:
            b = float(k["beat"])
            pose = normalize_pose(k.get("pose"))
        except (KeyError, TypeError) as exc:
            raise ChartError(f"{where}: move '{name}' keyframe {i}: {exc}") from None
        except ValueError as exc:
            raise ChartError(f"{where}: move '{name}' keyframe {i}: {exc}") from None
        if not (0.0 <= b < beats):
            raise ChartError(f"{where}: move '{name}' keyframe {i} at beat {b} is outside 0..{beats}")
        if b <= last:
            raise ChartError(f"{where}: move '{name}' keyframe beats must ascend (keyframe {i})")
        last = b
        klabel = k.get("label")
        if klabel is not None and not (isinstance(klabel, str) and 0 < len(klabel) <= 15):
            raise ChartError(f"{where}: move '{name}' keyframe {i}: 'label' must be a short name, got {klabel!r}")
        frames.append(Keyframe(b, pose, klabel))
    judge = tuple(spec.get("judge", range(len(frames))))
    if any(not isinstance(i, int) or not 0 <= i < len(frames) for i in judge):
        raise ChartError(f"{where}: move '{name}' judges a keyframe that doesn't exist: {list(judge)}")
    pict = spec.get("pictogram", judge[0] if judge else 0)
    if not isinstance(pict, int) or not 0 <= pict < len(frames):
        raise ChartError(f"{where}: move '{name}' pictogram index {pict} doesn't exist")
    return Move(name, str(spec.get("label", name.replace("_", " ").title())), beats, tuple(frames), judge, pict)


def parse_moves(data: dict, where: str, known: dict[str, Move] | None = None) -> dict[str, Move]:
    """Moves from a {"name": spec} mapping. `mirror_of` may refer to `known`
    or to moves earlier or later in the same mapping."""
    out: dict[str, Move] = {}
    pending = dict(data)
    base = dict(known or {})
    while pending:
        progressed = False
        for name in list(pending):
            spec = pending[name]
            ref = spec.get("mirror_of") if isinstance(spec, dict) else None
            if ref is not None and ref not in out and ref not in base:
                if ref in pending:
                    continue  # its source isn't parsed yet
                raise ChartError(f"{where}: move '{name}' mirrors unknown move '{ref}'")
            out[name] = _parse_move(name, spec, where, {**base, **out})
            del pending[name]
            progressed = True
        if not progressed:
            raise ChartError(f"{where}: moves mirror each other in a loop: {sorted(pending)}")
    return out


def load_library(path: Path = LIBRARY) -> dict[str, Move]:
    try:
        data = json.loads(Path(path).read_text())
    except FileNotFoundError:
        raise ChartError(f"missing move library {path}") from None
    except ValueError as exc:
        raise ChartError(f"{path}: not valid JSON ({exc})") from None
    return parse_moves(data.get("moves", {}), str(path))


def song_ids(songs_dir: Path = SONGS) -> list[str]:
    """Songs in menu order (songs/index.json)."""
    index = Path(songs_dir) / "index.json"
    try:
        ids = json.loads(index.read_text())
    except FileNotFoundError:
        return []
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        raise ChartError(f"{index}: expected a list of song ids")
    return ids


def _parse_chart(entries, moves: dict[str, Move], where: str, last_beat: float) -> list[Placed]:
    """A list of {"move", "beat", "gold"?} entries -> placed moves, validated:
    known moves, no overlaps, nothing past the song's end."""
    if not isinstance(entries, list):
        raise ChartError(f"{where}: expected a list of moves")
    out: list[Placed] = []
    end = -math.inf
    for i, entry in enumerate(entries):
        try:
            name, beat = entry["move"], float(entry["beat"])
        except (KeyError, TypeError, ValueError):
            raise ChartError(f"{where}: chart entry {i} needs 'move' and 'beat'") from None
        move = moves.get(name)
        if move is None:
            raise ChartError(f"{where}: chart entry {i} at beat {beat}: unknown move '{name}'")
        if beat < end - 1e-9:
            raise ChartError(f"{where}: move '{name}' at beat {beat} overlaps the previous move (ends at {end})")
        placed = Placed(move, beat, bool(entry.get("gold", False)))
        if placed.end > last_beat + 1e-9:
            raise ChartError(f"{where}: move '{name}' at beat {beat} runs past the song's end (beat {last_beat:.1f})")
        out.append(placed)
        end = placed.end
    return out


def load_song(folder: Path | str, library: dict[str, Move] | None = None,
              difficulty: str = DEFAULT_DIFFICULTY) -> Song:
    """A song on a difficulty (default normal; unknown ids are normal too).
    Every chart in song.json is validated, whichever difficulty is asked for."""
    folder = Path(folder)
    where = f"song '{folder.name}'"
    try:
        data = json.loads((folder / "song.json").read_text())
    except FileNotFoundError:
        raise ChartError(f"{where}: missing song.json") from None
    except ValueError as exc:
        raise ChartError(f"{where}: song.json is not valid JSON ({exc})") from None
    if library is None:
        library = load_library()
    for key in ("title", "bpm", "duration"):
        if key not in data:
            raise ChartError(f"{where}: song.json needs '{key}'")
    bpm, duration = float(data["bpm"]), float(data["duration"])
    if not 40 <= bpm <= 240 or duration <= 0:
        raise ChartError(f"{where}: bpm must be 40-240 and duration > 0")
    moves = {**library, **parse_moves(data.get("custom_moves", {}), f"{where} custom_moves", library)}

    song = Song(
        id=str(data.get("id", folder.name)), title=str(data["title"]), artist=str(data.get("artist", "")),
        bpm=bpm, offset=float(data.get("offset", 0.0)), duration=duration,
        audio=folder / str(data.get("audio", "audio.m4a")), difficulty=int(data.get("difficulty", 1)),
        coach=dict(data.get("coach", {})), sections=list(data.get("sections", [])), chart=[],
        preview=tuple(data["preview"]) if data.get("preview") else None, folder=folder,
    )
    last_beat = song.beat_of(duration)
    song.chart = _parse_chart(data.get("moves", []), moves, where, last_beat)
    song.charts = {DEFAULT_DIFFICULTY: list(song.chart)}
    extra = data.get("charts", {})
    if not isinstance(extra, dict):
        raise ChartError(f"{where}: 'charts' must be an object like {{\"easy\": [...], \"hard\": [...]}}")
    for level, entries in extra.items():
        if level not in DIFFICULTIES or level == DEFAULT_DIFFICULTY:
            raise ChartError(f"{where}: charts.{level}: only 'easy' and 'hard' charts can be hand-made "
                             f"(normal is 'moves')")
        song.charts[level] = _parse_chart(entries, moves, f"{where} charts.{level}", last_beat)
    _index(song)
    return song.for_difficulty(difficulty)


def load_songs(songs_dir: Path = SONGS, library: dict[str, Move] | None = None) -> list[Song]:
    library = library if library is not None else load_library()
    return [load_song(Path(songs_dir) / sid, library) for sid in song_ids(songs_dir)]


# --- the choreography lint (docs/DANCE.md section 5a) -------------------------------

PHRASE_BEATS = 8  # choreography is built in 8-counts from each section's start
HAND_TRAVEL = 2.35  # SW per beat a hand may travel (relative to its shoulder) between keyframes
WHIP_TRAVEL, WHIP_RUN = 1.9, 3  # this many transitions in a row above WHIP_TRAVEL is whiplash
FOOT_TRAVEL = 1.2  # SW per beat a foot may move (relative to the body)
SHIFT_TRAVEL = 0.9  # SW per beat the hips may move over the feet
SQUAT_TRAVEL = 0.7  # squat per beat
JUMPS_PER_PHRASE = 2  # jumps (both feet leave the floor) per 8-count; the rest go one foot at a time
LEG_SECTION_MIN = 0.25  # of a section's judge points must work the legs
WANDER_MAX = 1.5  # SW the coach may travel from its spot
FOOT_REACH = (1.6, 1.0)  # a foot further out / higher than this (SW) is out of the judge's reliable range
_LEG_EPS = 0.05


@dataclass(frozen=True)
class LintIssue:
    kind: str  # "flow" | "whiplash" | "jumps" | "phrase" | "legs" | "unjudgeable" | "freeze" | "rest" | "wander"
    beat: float
    message: str

    def __str__(self) -> str:
        return f"beat {self.beat:g}: [{self.kind}] {self.message}"


def _twin(name: str) -> str:
    return name[:-2] + "_l" if name.endswith("_r") else name[:-2] + "_r" if name.endswith("_l") else name


def _hand_rel(j: dict, s: str) -> tuple[float, float]:
    return j[f"{s}_hand"][0] - j[f"{s}_shoulder"][0], j[f"{s}_hand"][1] - j[f"{s}_shoulder"][1]


def leg_led(pose: dict) -> bool:
    """Does the judge read this pose's lower body (a step, a lift or a bend)?"""
    return leg_target(normalize_pose(pose)).active


def lint(song: Song, phrases: bool = True) -> list[LintIssue]:
    """Choreography rules (docs/DANCE.md section 5a) as checks on a song's
    chart: flow (hand and foot travel per beat, no whiplash, one foot at a time),
    8-count phrases that repeat, leg work in every section, leg moves the
    camera can judge, the freeze rule and a coach that stays on its spot.
    `phrases=False` skips the phrase checks (a derived EASY chart keeps one pose
    a bar, so its phrases needn't repeat move for move). An empty list is a pass."""
    from .motion import keyframe_travel  # motion imports this module

    out: list[LintIssue] = []
    frames = song._frames
    joints = [forward(p) for _, p in frames]
    run = 0
    for i in range(1, len(frames)):
        (b0, p0), (b1, p1) = frames[i - 1], frames[i]
        gap = max(b1 - b0, 0.5)
        hand = max(math.dist(_hand_rel(joints[i - 1], s), _hand_rel(joints[i], s)) for s in "lr") / gap
        if hand > HAND_TRAVEL:
            out.append(LintIssue("flow", b1, f"a hand travels {hand:.2f} SW a beat into this keyframe "
                                             f"(at most {HAND_TRAVEL:g}): whiplash"))
        run = run + 1 if hand > WHIP_TRAVEL else 0
        if run == WHIP_RUN:
            out.append(LintIssue("whiplash", b1, f"{WHIP_RUN} big arm swings in a row (over {WHIP_TRAVEL:g} SW a beat)"))
        foot = max(math.dist(p0[k], p1[k]) for k in ("lf", "rf")) / gap
        if foot > FOOT_TRAVEL:
            out.append(LintIssue("flow", b1, f"a foot moves {foot:.2f} SW a beat (at most {FOOT_TRAVEL:g})"))
        if abs(p1["shift"] - p0["shift"]) / gap > SHIFT_TRAVEL:
            out.append(LintIssue("flow", b1, f"the hips move {abs(p1['shift'] - p0['shift']) / gap:.2f} SW a beat "
                                             f"(at most {SHIFT_TRAVEL:g})"))
        if abs(p1["squat"] - p0["squat"]) / gap > SQUAT_TRAVEL:
            out.append(LintIssue("flow", b1, f"the squat changes {abs(p1['squat'] - p0['squat']) / gap:.2f} a beat "
                                             f"(at most {SQUAT_TRAVEL:g})"))
    offs, sides, _ = keyframe_travel(frames)
    if offs and max(abs(o) for o in offs) > WANDER_MAX:
        k = max(range(len(offs)), key=lambda i: abs(offs[i]))
        out.append(LintIssue("wander", frames[k][0], f"the coach steps {offs[k]:+.2f} SW off its spot "
                                                     f"(at most {WANDER_MAX:g}): balance the phrase's steps"))

    sections = sorted(song.sections, key=lambda s: s["beat"]) or [{"name": "chart", "beat": 0}]
    end = song.beat_of(song.duration)
    bounds = [(s["name"], s["beat"], sections[i + 1]["beat"] if i + 1 < len(sections) else end)
              for i, s in enumerate(sections)]
    judges = song.judge_points()
    for name, lo, hi in bounds:
        pts = [p for p in judges if lo <= p.beat < hi]
        if pts:
            legs = sum(leg_led(p.pose) for p in pts)
            if legs < LEG_SECTION_MIN * len(pts):
                out.append(LintIssue("legs", lo, f"{name}: only {legs} of {len(pts)} judged poses work the legs "
                                                 f"(at least {LEG_SECTION_MIN:.0%})"))
        # jumps per 8-count
        for start in _phrase_starts(lo, hi):
            n = sum(1 for (b, _), s in zip(frames[1:], sides) if s is None and start <= b < start + PHRASE_BEATS)
            if n > JUMPS_PER_PHRASE:
                out.append(LintIssue("jumps", start, f"{n} jumps in one 8-count (at most {JUMPS_PER_PHRASE}): "
                                                     f"step one foot at a time"))
    if phrases:
        out += _lint_phrases(song, bounds)

    rest = normalize_pose(REST)
    for i, p in enumerate(judges):
        t = normalize_pose(p.pose)
        moved = (any(abs(t[k][0] - rest[k][0]) > _LEG_EPS or t[k][1] > _LEG_EPS for k in ("lf", "rf"))
                 or abs(t["shift"]) > _LEG_EPS or t["squat"] > _LEG_EPS)
        if moved and not leg_led(t):
            out.append(LintIssue("unjudgeable", p.beat, f"'{p.label}': its legs move too little for the camera "
                                                        f"to tell from standing still"))
        if any(t[k][0] > FOOT_REACH[0] or t[k][1] > FOOT_REACH[1] for k in ("lf", "rf")):
            out.append(LintIssue("unjudgeable", p.beat, f"'{p.label}': a foot beyond {FOOT_REACH[0]:g} SW out or "
                                                        f"{FOOT_REACH[1]:g} SW up leaves the judge's reliable range"))
        if freeze_score(t, rest) >= FREEZE_MAX:
            out.append(LintIssue("rest", p.beat, f"'{p.label}': standing still with arms down scores on it"))
        if i and freeze_score(t, judges[i - 1].pose) >= FREEZE_MAX:
            out.append(LintIssue("freeze", p.beat, f"'{p.label}': holding the previous pose "
                                                   f"('{judges[i - 1].label}') scores on it"))
    return sorted(out, key=lambda x: (x.beat, x.kind))


def _phrase_starts(lo: float, hi: float) -> list[float]:
    out, b = [], lo
    while b < hi - 1e-9:
        out.append(b)
        b += PHRASE_BEATS
    return out


def _lint_phrases(song: Song, bounds) -> list[LintIssue]:
    """8-counts: no move straddles one, and every 8-count with more than one
    judged move repeats: it appears elsewhere in the song (as is or mirrored),
    or its two 4-counts repeat or mirror each other."""
    out: list[LintIssue] = []
    phrases: list[tuple[float, tuple]] = []
    for name, lo, hi in bounds:
        for start in _phrase_starts(lo, hi):
            stop = min(start + PHRASE_BEATS, hi)
            inside = [p for p in song.chart if start - 1e-9 <= p.beat < stop - 1e-9]
            for p in inside:
                if p.end > stop + 1e-9 and stop < hi - 1e-9:
                    out.append(LintIssue("phrase", p.beat, f"'{p.move.label}' runs over the 8-count at beat {stop:g}"))
            if sum(1 for p in inside if p.move.judge) > 1:
                phrases.append((start, tuple((round(p.beat - start, 3), p.move.name) for p in inside)))
    seqs = [s for _, s in phrases]
    for i, (start, seq) in enumerate(phrases):
        mirrored = tuple((b, _twin(n)) for b, n in seq)
        elsewhere = any(j != i and other in (seq, mirrored) for j, other in enumerate(seqs))
        first = tuple((b, n) for b, n in seq if b < 4)
        second = tuple((b - 4, n) for b, n in seq if b >= 4)
        halves = bool(first) and (second == first or second == tuple((b, _twin(n)) for b, n in first))
        if not (elsewhere or halves):
            out.append(LintIssue("phrase", start, "this 8-count never repeats (dance it again, mirror it, "
                                                  "or repeat its first 4 counts)"))
    return out
