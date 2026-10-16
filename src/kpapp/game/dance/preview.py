"""Contact sheets for reviewing choreography (`kp dance-preview`).

`chart_sheet(song)` draws every placed move of a song as stick figures, one
per keyframe, grouped under section headers, with beat numbers, labels and
gold moves marked. `library_sheet(moves)` does the same for a move library.
The screen-left arm is drawn cyan and the screen-right arm pink, so a move's
side reads at a glance: the player copies it like a mirror (docs/DANCE.md
section 2).
"""

from __future__ import annotations

import cv2
import numpy as np

from ... import theme
from .chart import Move, Song
from .pose import forward, leg_target, normalize_pose

BG = (34, 26, 30)
CARD = (52, 42, 48)
LEFT_ARM = (235, 205, 80)  # BGR: cyan-blue, the screen-left arm
RIGHT_ARM = (180, 90, 245)  # BGR: pink, the screen-right arm
BODY = (225, 225, 235)
LEGS = (170, 170, 185)
LEGS_JUDGED = (235, 235, 245)  # legs the judge reads (a step, a lift, a bend), on a floor line
FLOOR = (255, 220, 110)

SLOT_W = 118  # one keyframe figure
CARD_H = 214
GAP = 10
SHEET_W = 1880


def draw_figure(img: np.ndarray, pose: dict, hip_xy: tuple[float, float], scale: float,
                judged: bool = True) -> None:
    """A stick figure for `pose`, the rest hip centre at `hip_xy`, `scale` px per shoulder width.
    Legs the judge reads are drawn bright on a floor line, with an arrow to the weight's side."""
    j = forward(pose)
    lt = leg_target(normalize_pose(pose))

    def p(name):
        x, y = j[name]
        return int(round(hip_xy[0] + x * scale)), int(round(hip_xy[1] + y * scale))

    dim = 1.0 if judged else 0.45

    def col(c):
        return tuple(int(v * dim + BG[i] * (1 - dim)) for i, v in enumerate(c))

    t = max(2, int(scale * 0.16))
    legs = LEGS_JUDGED if lt.active else LEGS
    if lt.active:
        fy = max(p("l_ankle")[1], p("r_ankle")[1]) + t
        xs = (p("l_ankle")[0], p("r_ankle")[0])
        cv2.line(img, (min(xs) - t * 3, fy), (max(xs) + t * 3, fy), col(FLOOR), max(1, t // 2), cv2.LINE_AA)
        if lt.step and abs(lt.lower.weight) > 0.12:
            mx, d = (xs[0] + xs[1]) // 2, int(scale * (0.4 if lt.lower.weight > 0 else -0.4))
            cv2.arrowedLine(img, (mx - d, fy + 2 * t), (mx + d, fy + 2 * t), col(FLOOR), max(1, t // 2),
                            cv2.LINE_AA, 0, 0.4)
    for s in "lr":
        cv2.line(img, p(f"{s}_hip"), p(f"{s}_knee"), col(legs), t, cv2.LINE_AA)
        cv2.line(img, p(f"{s}_knee"), p(f"{s}_ankle"), col(legs), t, cv2.LINE_AA)
    torso = np.array([p("l_shoulder"), p("r_shoulder"), p("r_hip"), p("l_hip")], np.int32)
    cv2.fillConvexPoly(img, torso, col((90, 70, 95)), cv2.LINE_AA)
    cv2.polylines(img, [torso], True, col(BODY), max(1, t // 2), cv2.LINE_AA)
    cv2.line(img, p("neck"), p("head"), col(BODY), t, cv2.LINE_AA)
    cv2.circle(img, p("head"), int(scale * 0.3), col(BODY), -1, cv2.LINE_AA)
    for s, c in (("l", LEFT_ARM), ("r", RIGHT_ARM)):
        cv2.line(img, p(f"{s}_shoulder"), p(f"{s}_elbow"), col(c), t + 1, cv2.LINE_AA)
        cv2.line(img, p(f"{s}_elbow"), p(f"{s}_hand"), col(c), t + 1, cv2.LINE_AA)
        cv2.circle(img, p(f"{s}_hand"), t + 1, col(c), -1, cv2.LINE_AA)
        cv2.circle(img, p(f"{s}_elbow"), max(2, t - 1), col(BODY), -1, cv2.LINE_AA)


def _card(move: Move, beat: float | None, gold: bool, title: str | None = None) -> np.ndarray:
    n = len(move.keyframes)
    w = max(150, n * SLOT_W + 16)
    card = np.full((CARD_H, w, 3), BG, np.uint8)
    border = theme.ACCENT if gold else (90, 80, 90)
    theme.panel(card, (0, 0, w, CARD_H), CARD, alpha=1.0, border=border, radius=12, thickness=3 if gold else 1)
    head = title or move.label
    if gold:
        head = f"{head}  GOLD"
    cv2.putText(card, head, (10, 24), theme.TITLE_FONT, 0.55, theme.ACCENT if gold else theme.WHITE, 1, cv2.LINE_AA)
    sub = f"{move.name}  {move.beats:g}b" if beat is None else f"beat {beat:g}  ({move.beats:g}b)"
    cv2.putText(card, sub, (10, 42), theme.BODY_FONT, 0.42, theme.DIM, 1, cv2.LINE_AA)
    x0 = (w - n * SLOT_W) // 2
    for i, k in enumerate(move.keyframes):
        judged = i in move.judge
        cx = x0 + i * SLOT_W + SLOT_W // 2
        draw_figure(card, k.pose, (cx, 136), 25.0, judged)
        b = k.beat if beat is None else beat + k.beat
        tag = f"{b:g}" + ("" if judged else " -")
        (tw, _), _ = cv2.getTextSize(tag, theme.BODY_FONT, 0.45, 1)
        cv2.putText(card, tag, (cx - tw // 2, CARD_H - 10), theme.BODY_FONT, 0.45,
                    theme.CYAN if judged else theme.DIM, 1, cv2.LINE_AA)
        if i == move.pictogram:
            cv2.circle(card, (cx + tw // 2 + 8, CARD_H - 15), 3, theme.ACCENT, -1, cv2.LINE_AA)
    return card


def _flow(cards: list[np.ndarray], width: int) -> list[list[np.ndarray]]:
    rows, row, used = [], [], 0
    for c in cards:
        if row and used + c.shape[1] > width:
            rows.append(row)
            row, used = [], 0
        row.append(c)
        used += c.shape[1] + GAP
    if row:
        rows.append(row)
    return rows


def _assemble(blocks: list[tuple[str, list[np.ndarray]]], title: str, subtitle: str) -> np.ndarray:
    inner = SHEET_W - 2 * GAP
    parts: list[np.ndarray] = []
    top = np.full((70, SHEET_W, 3), BG, np.uint8)
    cv2.putText(top, title, (GAP, 38), theme.TITLE_FONT, 1.1, theme.ACCENT, 2, cv2.LINE_AA)
    cv2.putText(top, subtitle, (GAP, 62), theme.BODY_FONT, 0.5, theme.DIM, 1, cv2.LINE_AA)
    parts.append(top)
    for header, cards in blocks:
        bar = np.full((34, SHEET_W, 3), BG, np.uint8)
        cv2.rectangle(bar, (GAP, 4), (SHEET_W - GAP, 30), (70, 52, 62), -1)
        cv2.putText(bar, header, (GAP + 8, 24), theme.TITLE_FONT, 0.6, theme.CYAN, 1, cv2.LINE_AA)
        parts.append(bar)
        for row in _flow(cards, inner):
            strip = np.full((CARD_H + GAP, SHEET_W, 3), BG, np.uint8)
            x = GAP
            for c in row:
                strip[GAP // 2:GAP // 2 + CARD_H, x:x + c.shape[1]] = c
                x += c.shape[1] + GAP
            parts.append(strip)
        if not cards:
            empty = np.full((30, SHEET_W, 3), BG, np.uint8)
            cv2.putText(empty, "no moves", (GAP + 8, 20), theme.BODY_FONT, 0.5, theme.DIM, 1, cv2.LINE_AA)
            parts.append(empty)
    return np.vstack(parts)


LEGEND = ("cyan = screen-left arm, pink = screen-right arm; bright legs on a floor line = legs judged (arrow: "
          "weight); dim figure / '-' = not judged; dot = pictogram")


def chart_sheet(song: Song) -> np.ndarray:
    """Every placed move of `song`, grouped by section."""
    sections = sorted(song.sections, key=lambda s: s["beat"]) or [{"name": "chart", "beat": 0}]
    end_beat = song.beat_of(song.duration)
    blocks = []
    for i, sec in enumerate(sections):
        lo = sec["beat"]
        hi = sections[i + 1]["beat"] if i + 1 < len(sections) else end_beat
        cards = [_card(p.move, p.beat, p.gold) for p in song.chart if lo <= p.beat < hi]
        if i == 0 and lo > 0:
            cards = [_card(p.move, p.beat, p.gold) for p in song.chart if p.beat < hi]
        blocks.append((f"{sec['name'].upper()}   beats {lo:g}-{hi - 1:g}", cards))
    judges = song.judge_points()
    gold = sum(1 for j in judges if j.gold)
    level = getattr(song, "level", "normal")
    sub = (f"{song.artist}  |  {song.bpm:g} BPM  |  {song.duration:g} s  |  {level.upper()}: {len(song.chart)} moves, "
           f"{len(judges)} judge points, {gold} gold  |  {LEGEND}")
    return _assemble(blocks, f"{song.title}  ({level.upper()})", sub)


def library_sheet(moves: dict[str, Move]) -> np.ndarray:
    """Every move in a library, in definition order."""
    cards = [_card(m, None, False) for m in moves.values()]
    return _assemble([(f"{len(moves)} MOVES", cards)], "Move library", LEGEND)
