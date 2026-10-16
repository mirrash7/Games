"""Recording -> chart: turn someone dancing to a track into a draft chart
(`kp dance-import`, docs/DANCE.md section 11).

The core, `chart_from_poses`, takes timestamped COCO poses from any source (a
video run through the model, the live camera, or synthetic poses in tests).
For every beat step it takes the median of the samples within +-60 ms of the
beat (per keypoint, falling back to the nearest sighting within 120 ms),
measures it into dance angles (`pose.measure`), and emits one judged keyframe
as a custom move `imp_<beat>`. Beats where the dancer couldn't be
measured are skipped. Consecutive near-identical poses can be merged into one
longer held move, judged once.

The result is a draft: review it with `kp dance-preview`, trim it, rename and
mark gold moves by hand.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np

from .pose import MIN_CONF, keypoints_for, measure, similarity

WINDOW = 0.060  # seconds either side of a beat that count as "on" it
FALLBACK = 0.120  # if nothing is that close, the nearest sample within this
MERGE_ABOVE = 0.9  # similarity above which a pose "holds" the previous one


def _wrap_arm(a: float) -> float:
    """Measured angles come in (-180, 180]; the chart convention runs from
    about -100 (across the body) to about 260 (past vertical, over the head)."""
    return a + 360.0 if a < -100.0 else a


def _near(fore: float, upper: float) -> float:
    """Keep the forearm within 180 degrees of the upper arm, so the coach's
    linear blend between keyframes never swings an arm through the body."""
    while fore - upper > 180.0:
        fore -= 360.0
    while fore - upper < -180.0:
        fore += 360.0
    return fore


def beat_pose(near: list):
    """One pose for a beat from (|dt| seconds, Pose) samples around it: per
    keypoint, the median over the samples within WINDOW that see it, else the
    nearest sample within FALLBACK that does. Elbows drop out of single frames
    (AGENTS.md section 6), so a keypoint seen once near the beat is kept."""
    from ...inference import Pose

    if not near:
        return None
    near = sorted(near, key=lambda dp: dp[0])
    xy = np.stack([p.xy for _, p in near]).astype(np.float32)
    conf = np.stack([p.confidence for _, p in near]).astype(np.float32)
    dt = np.array([d for d, _ in near])
    out_xy = np.zeros(xy.shape[1:], np.float32)
    out_conf = np.zeros(xy.shape[1], np.float32)
    for k in range(xy.shape[1]):
        seen = conf[:, k] >= MIN_CONF
        close = seen & (dt <= WINDOW)
        if close.any():
            out_xy[k] = np.median(xy[close, k], axis=0)
            out_conf[k] = float(np.median(conf[close, k]))
        elif seen.any():
            i = int(np.argmax(seen))  # nearest in time: `near` is sorted by |dt|
            out_xy[k], out_conf[k] = xy[i, k], conf[i, k]
    return Pose(xy=out_xy, confidence=out_conf, score=float(np.median([p.score for _, p in near])))


def pose_from_measured(m) -> dict | None:
    """A chart pose dict from a `Measured` player, or None if no arm was seen.

    Arms: upper/forearm when the elbow was seen; otherwise a straight arm along
    the shoulder-to-wrist reach; unseen arms rest. Legs when seen, else rest.
    Lean only when the hips were seen (without them it's the shoulder tilt).
    """
    if m is None or not any(a.reach is not None for a in m.arms.values()):
        return None
    pose: dict = {}
    for s in "lr":
        a = m.arms.get(s)
        if a is None:
            continue
        if a.upper is not None and a.fore is not None:
            up = _wrap_arm(a.upper)
            pose[f"{s}a"] = [round(up), round(_near(_wrap_arm(a.fore), up))]
        elif a.reach is not None:
            r = round(_wrap_arm(a.reach))
            pose[f"{s}a"] = [r, r]
    for s in "lr":
        leg = m.legs.get(s)
        if leg is not None and leg.thigh is not None and leg.shin is not None:
            pose[f"{s}l"] = [round(leg.thigh), round(_near(leg.shin, leg.thigh))]
    if m.has_hips and abs(m.lean) >= 1.0:
        pose["lean"] = round(m.lean)
    return pose


def _sample_at(samples: list, times: np.ndarray, t: float):
    """The pose at song time `t` from the samples within FALLBACK of it (beat_pose)."""
    lo = int(np.searchsorted(times, t - FALLBACK, "left"))
    hi = int(np.searchsorted(times, t + FALLBACK, "right"))
    return beat_pose([(abs(times[i] - t), samples[i][1]) for i in range(lo, hi) if samples[i][1] is not None])


def _num(x: float):
    """Whole beats as ints, so song.json reads "beat": 16, not 16.0."""
    return int(x) if float(x).is_integer() else x


def _beat_name(beat: float) -> str:
    return f"imp_{beat:g}".replace(".", "_")


def chart_from_poses(samples, bpm: float, offset: float, start_beat: float, end_beat: float,
                     every: float = 1.0, mirrored: bool = True, merge: bool = False) -> dict:
    """Draft chart from timestamped poses.

    samples: iterable of (time_seconds, Pose or None), time on the song's clock
    (beat 0 at `offset`). Returns {"moves": [...], "custom_moves": {...}} in the
    song.json format: one `imp_<beat>` move per measurable beat step in
    [start_beat, end_beat), each `every` beats long with one judged keyframe.
    With `merge`, a pose that matches the previous one (similarity > 0.9 for a
    perfect player) extends the previous move into a longer hold instead.
    """
    if every <= 0:
        raise ValueError("every must be > 0")
    samples = sorted(((float(t), p) for t, p in samples), key=lambda s: s[0])
    times = np.array([t for t, _ in samples], np.float64)
    beat_s = 60.0 / bpm
    moves: list[dict] = []
    custom: dict[str, dict] = {}
    prev_pose = None
    prev_entry = None
    n = int(math.floor((end_beat - start_beat) / every + 1e-9))
    for i in range(n):
        beat = round(start_beat + i * every, 6)
        kp = _sample_at(samples, times, offset + beat * beat_s) if len(samples) else None
        pose = pose_from_measured(measure(kp, mirrored=mirrored)) if kp is not None else None
        if pose is None:
            prev_pose = prev_entry = None
            continue
        if merge and prev_pose is not None and prev_entry is not None:
            score, _ = similarity(pose, measure(keypoints_for(prev_pose)), None)  # the arms say it
            if score is not None and score > MERGE_ABOVE and math.isclose(
                    prev_entry["beat"] + custom[prev_entry["move"]]["beats"], beat):
                custom[prev_entry["move"]]["beats"] = _num(custom[prev_entry["move"]]["beats"] + every)
                continue
        name = _beat_name(beat)
        custom[name] = {"label": "Move", "beats": _num(every), "keyframes": [{"beat": 0, "pose": pose}]}
        prev_entry = {"move": name, "beat": _num(beat)}
        moves.append(prev_entry)
        prev_pose = pose
    return {"moves": moves, "custom_moves": custom}


def merge_into_song(song_data: dict, imported: dict, start_beat: float, end_beat: float,
                    library: dict | None = None) -> dict:
    """`song_data` with its chart in [start_beat, end_beat) replaced by
    `imported`. Metadata and moves outside that range are kept (a move that
    starts before the range but runs into the first new move is dropped), and
    imported custom moves that nothing uses any more are removed."""
    from .chart import load_library, parse_moves

    out = dict(song_data)
    custom = {**song_data.get("custom_moves", {}), **imported["custom_moves"]}
    lib = library if library is not None else load_library()
    known = {**lib, **parse_moves(custom, "merge", lib)}
    first_new = min((float(m["beat"]) for m in imported["moves"]), default=end_beat)

    def keep(m) -> bool:
        b = float(m["beat"])
        if start_beat <= b < end_beat:
            return False
        move = known.get(m["move"])
        return not (b < first_new and move is not None and b + move.beats > first_new + 1e-9)

    out["moves"] = sorted([m for m in song_data.get("moves", []) if keep(m)] + imported["moves"],
                          key=lambda m: float(m["beat"]))
    used = {m["move"] for m in out["moves"]}
    out["custom_moves"] = {k: v for k, v in custom.items() if not k.startswith("imp_") or k in used}
    return out


def write_song_json(path: Path, data: dict, backup: bool = True) -> None:
    """Write song.json in the house layout (one chart entry per line), keeping a .bak."""
    path = Path(path)
    if backup and path.exists():
        shutil.copyfile(path, path.with_suffix(".json.bak"))
    path.write_text(format_song_json(data))


def format_song_json(data: dict) -> str:
    lines = ["{"]
    body = []
    for key, value in data.items():
        if key in ("sections", "moves") and isinstance(value, list):
            inner = ",\n".join("    " + json.dumps(v) for v in value)
            body.append(f'  "{key}": [\n{inner}\n  ]' if value else f'  "{key}": []')
        elif key == "custom_moves" and isinstance(value, dict):
            inner = ",\n".join(f"    {json.dumps(k)}: {json.dumps(v)}" for k, v in value.items())
            body.append(f'  "{key}": {{\n{inner}\n  }}' if value else f'  "{key}": {{}}')
        else:
            body.append(f"  {json.dumps(key)}: {json.dumps(value)}")
    lines.append(",\n".join(body))
    lines.append("}")
    return "\n".join(lines) + "\n"


# --- sources ------------------------------------------------------------------

def beat_times(bpm: float, offset: float, start_beat: float, end_beat: float, every: float) -> list[float]:
    n = int(math.floor((end_beat - start_beat) / every + 1e-9))
    return [offset + (start_beat + i * every) * 60.0 / bpm for i in range(n)]


def video_samples(video: Path | str, detect, wanted: list[float], video_start: float, song_offset: float,
                  mirror: bool = True, window: float = WINDOW, progress=None) -> list:
    """Run `detect(frame) -> Pose | None` on the video frames near the wanted
    song times. `video_start` is where beat 0 falls in the video (seconds);
    returns (song_time, Pose) samples on the song's clock (beat 0 at `song_offset`)."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"can't open video {video}")
    shift = video_start - song_offset  # video time = song time + shift
    targets = sorted(t + shift for t in wanted)
    out, j, idx = [], 0, 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    try:
        while j < len(targets):
            ok = cap.grab()
            if not ok:
                break
            pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if pos <= 0.0 and idx > 0:
                pos = idx / fps
            idx += 1
            while j < len(targets) and targets[j] < pos - window:
                j += 1
            if j < len(targets) and abs(pos - targets[j]) <= window:
                ok, frame = cap.retrieve()
                if not ok:
                    continue
                if mirror:
                    frame = cv2.flip(frame, 1)
                out.append((pos - shift, detect(frame)))
                if progress:
                    progress(j, len(targets))
    finally:
        cap.release()
    return out


def song_folder(song_id: str) -> Path:
    from .chart import SONGS

    return SONGS / song_id


def load_song_data(song_id: str) -> tuple[Path, dict]:
    path = song_folder(song_id) / "song.json"
    if not path.exists():
        raise SystemExit(f"[kp] {path} not found: create it with title, bpm, offset and duration first "
                         "(docs/DANCE.md section 11)")
    data = json.loads(path.read_text())
    for key in ("bpm", "duration"):
        if key not in data:
            raise SystemExit(f"[kp] {path} needs '{key}' before importing")
    return path, data
