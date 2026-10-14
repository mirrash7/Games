"""Songs and moves: loading, validation, beat <-> time, the judge timeline and
the coach's animation (docs/DANCE.md sections 4-5).

A song folder holds `song.json` (metadata + chart) and its audio. Moves come
from the shared library (`assets/dance/moves.json`) or the song's own
`custom_moves`. Everything is validated on load, with errors that name the
song, move and beat, so a hand-edited or imported chart fails loudly.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .pose import REST, lerp_pose, mirror_pose, normalize_pose, smoothstep

ASSET_DIR = Path(__file__).resolve().parents[4] / "assets" / "dance"
LIBRARY = ASSET_DIR / "moves.json"
SONGS = ASSET_DIR / "songs"

# The coach reaches each keyframe exactly on its beat, holding it until it
# eases into the next one over this many beats (or the whole gap if shorter).
EASE_BEATS = 0.75


class ChartError(ValueError):
    pass


@dataclass(frozen=True)
class Keyframe:
    beat: float  # offset within the move
    pose: dict  # complete (normalized) pose


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
                    tuple(Keyframe(k.beat, normalize_pose(mirror_pose(k.pose))) for k in src.keyframes),
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
        frames.append(Keyframe(b, pose))
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


def load_song(folder: Path | str, library: dict[str, Move] | None = None) -> Song:
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
    end = -math.inf
    for i, entry in enumerate(data.get("moves", [])):
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
        song.chart.append(placed)
        end = placed.end

    for p_idx, p in enumerate(song.chart):
        for k_idx, k in enumerate(p.move.keyframes):
            b = p.beat + k.beat
            song._frames.append((b, k.pose))
            if k_idx in p.move.judge:
                song._judges.append(JudgePoint(len(song._judges), b, song.time_of(b), k.pose, p.gold,
                                               2.0 if p.gold else 1.0, p.move.label, p_idx))
    return song


def load_songs(songs_dir: Path = SONGS, library: dict[str, Move] | None = None) -> list[Song]:
    library = library if library is not None else load_library()
    return [load_song(Path(songs_dir) / sid, library) for sid in song_ids(songs_dir)]
