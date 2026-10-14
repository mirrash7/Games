// Songs and moves: parsing, validation, beat <-> time, the judge timeline and
// the coach's animation (port of src/kpapp/game/dance/chart.py; docs/DANCE.md
// sections 4-5).
//
// The Python reads files; here everything is a pure function of already-parsed
// JSON, so the browser can fetch it and the Node tests can read it from disk.
// The same validation, with the same messages (naming the song, move and
// beat), so a hand-edited or imported chart fails loudly in both builds.
//
//   parseLibrary(json, where)          moves.json -> Map(name -> Move)
//   parseSongIds(json, where)          songs/index.json -> ["neon_nights", ...]
//   parseSong(json, folder, library, difficulty)   songs/<folder>/song.json -> Song
//
// Moves live in a Map, so a move called "constructor" can't find Object's.
//
// Every song plays on three difficulties (docs/DANCE.md section 7a). NORMAL and
// HARD dance the song's chart (`moves`); EASY dances a chart derived from it,
// one held pose per bar (deriveEasy). A song may hand-make either one in
// song.json under "charts": {"easy": [...], "hard": [...]}.

import {
  REST, keypointsFor, lerpPose, measure, mirrorPose, normalizePose, pyNum, pyRepr, similarity, smoothstep,
} from "./pose.js";

// The coach reaches each keyframe exactly on its beat, holding it until it
// eases into the next one over this many beats (or the whole gap if shorter).
export const EASE_BEATS = 0.75;

export const DIFFICULTIES = Object.freeze(["easy", "normal", "hard"]);
export const DEFAULT_DIFFICULTY = "normal";
export const BAR_BEATS = 4; // EASY judges about one pose per bar
export const EASY_MIN_GAP = 2.0; // beats: EASY never judges two poses closer than this
// The freeze rule: a player frozen in the previous judged pose must score
// below this on the next one, with or without elbows in view
// (dance_chart.test.mjs checks it for every difficulty of every song).
export const FREEZE_MAX = 0.55;
const ELBOWS_HIDDEN = ["shoulder", "wrist", "hip"];

/** A known difficulty id; anything else (null, typos) is normal. */
export function difficultyId(difficulty) {
  const d = difficulty != null ? String(difficulty).toLowerCase() : DEFAULT_DIFFICULTY;
  return DIFFICULTIES.includes(d) ? d : DEFAULT_DIFFICULTY;
}

/**
 * How well a perfect player frozen in `held` matches `target`: the worse
 * (higher) of elbows seen and elbows hidden. Below FREEZE_MAX means a player
 * can't score on `target` by standing still in `held`.
 */
export function freezeScore(target, held) {
  const full = similarity(target, measure(keypointsFor(held)))[0];
  const reach = similarity(target, measure(keypointsFor(held, { visible: ELBOWS_HIDDEN })))[0];
  return Math.max(full ?? 0.0, reach ?? 0.0);
}

export class ChartError extends Error {
  constructor(message) {
    super(message);
    this.name = "ChartError";
  }
}

const isObject = (v) => v !== null && typeof v === "object" && !Array.isArray(v);
const has = (o, k) => isObject(o) && Object.hasOwn(o, k);

/** Python's float(): numbers, booleans and numeric strings; anything else throws. */
function toFloat(v) {
  if (typeof v === "number") return v;
  if (typeof v === "boolean") return v ? 1 : 0;
  if (typeof v === "string" && v.trim() !== "" && Number.isFinite(Number(v))) return Number(v);
  throw new TypeError(`could not convert ${pyRepr(v)} to float`);
}

/** Python's str.title(): "disco_point_r".replace("_", " ") -> "Disco Point R". */
function titleCase(s) {
  return s.replace(/[A-Za-z]+/g, (w) => w[0].toUpperCase() + w.slice(1).toLowerCase());
}

/**
 * Move: {name, label, beats, keyframes: [{beat, pose}], judge: [indexes judged],
 * pictogram: index of the keyframe drawn on the pictogram}.
 */
function makeMove(name, label, beats, keyframes, judge, pictogram) {
  return Object.freeze({ name, label, beats, keyframes, judge, pictogram });
}

/** A move placed in a song's chart. */
export class Placed {
  constructor(move, beat, gold = false) {
    this.move = move;
    this.beat = beat;
    this.gold = gold;
  }

  get end() {
    return this.beat + this.move.beats;
  }
}

/** Bob to the music while waiting: a sway on the beat, arms loose. */
export function idleGroove(beat) {
  const sway = Math.sin(beat * Math.PI);
  const pose = normalizePose(REST);
  pose.lean = 4.0 * sway;
  pose.head = -3.0 * sway;
  pose.la = [16.0 + 6.0 * sway, 24.0 + 10.0 * sway];
  pose.ra = [16.0 - 6.0 * sway, 24.0 - 10.0 * sway];
  pose.squat = 0.08 * (0.5 + 0.5 * Math.cos(beat * 2 * Math.PI));
  return pose;
}

export class Song {
  constructor(fields) {
    this.id = fields.id;
    this.title = fields.title;
    this.artist = fields.artist;
    this.bpm = fields.bpm;
    this.offset = fields.offset;
    this.duration = fields.duration;
    this.audio = fields.audio; // file name inside the song's folder
    this.difficulty = fields.difficulty;
    this.coach = fields.coach;
    this.sections = fields.sections;
    this.chart = fields.chart; // [Placed]
    this.preview = fields.preview ?? null;
    this.folder = fields.folder ?? null;
    this.level = fields.level ?? DEFAULT_DIFFICULTY; // the difficulty this chart is for
    this.charts = fields.charts ?? {}; // "normal" + hand-made ones: {level: [Placed]}
    this._frames = []; // [[beat, pose]]
    this._judges = []; // [JudgePoint]
  }

  // --- time ---

  get beatSeconds() {
    return 60.0 / this.bpm;
  }

  timeOf(beat) {
    return beat * this.beatSeconds;
  }

  beatOf(t) {
    return t / this.beatSeconds;
  }

  sectionAt(beat) {
    let name = "";
    for (const s of this.sections) if (s.beat <= beat) name = s.name;
    return name;
  }

  // --- difficulties ---

  /**
   * The chart a difficulty dances: a hand-made one from song.json's `charts`
   * if there is one, else EASY derives from the normal chart and HARD dances
   * the normal chart (judged strictly: judge.DIFFICULTY).
   */
  chartFor(difficulty) {
    const level = difficultyId(difficulty);
    const normal = this.charts[DEFAULT_DIFFICULTY] ?? this.chart;
    if (Object.hasOwn(this.charts, level)) return [...this.charts[level]];
    if (level === "easy") return deriveEasy(normal);
    return [...normal];
  }

  /** This song on another difficulty: same metadata, that difficulty's chart. */
  forDifficulty(difficulty) {
    const level = difficultyId(difficulty);
    if (level === this.level) return this;
    const charts = Object.keys(this.charts).length ? this.charts : { [DEFAULT_DIFFICULTY]: [...this.chart] };
    const song = new Song({ ...this, level, charts, chart: [] });
    song.chart = song.chartFor(level);
    indexSong(song);
    return song;
  }

  // --- the chart, flattened ---

  /** [{index, beat, time, pose, gold, weight, label, placed}] in time order. */
  judgePoints() {
    return this._judges;
  }

  get firstMoveBeat() {
    return this.chart.length ? this.chart[0].beat : 0.0;
  }

  get lastJudgeTime() {
    return this._judges.length ? this._judges[this._judges.length - 1].time : 0.0;
  }

  /**
   * The lane's cards: one per judge point, showing the pose judged on that
   * beat under its move's label, so card, label and judge always agree.
   * [{beat, label, pose, gold}]. (Move.pictogram now only picks the move's
   * thumbnail in the desktop's `kp dance-preview`.)
   */
  pictograms() {
    return this._judges.map((j) => Object.freeze({ beat: j.beat, label: j.label, pose: j.pose, gold: j.gold }));
  }

  /**
   * The coach's pose at a (fractional) beat: idle groove before the first
   * move, eased keyframes during the chart, the last pose held after it.
   * May return a keyframe's own pose object: read it, don't modify it.
   */
  coachPose(beat) {
    const frames = this._frames;
    if (!frames.length || beat < frames[0][0] - EASE_BEATS) return idleGroove(beat);
    if (beat < frames[0][0]) {
      const u = smoothstep((beat - (frames[0][0] - EASE_BEATS)) / EASE_BEATS);
      return lerpPose(idleGroove(beat), frames[0][1], u);
    }
    let lo = 0, hi = frames.length - 1;
    if (beat >= frames[hi][0]) return frames[hi][1];
    while (hi - lo > 1) { // last frame at or before `beat`
      const mid = (lo + hi) >> 1;
      if (frames[mid][0] <= beat) lo = mid;
      else hi = mid;
    }
    const [b0, p0] = frames[lo], [b1, p1] = frames[lo + 1];
    const ease = Math.min(EASE_BEATS, b1 - b0);
    const u = ease > 0 ? smoothstep((beat - (b1 - ease)) / ease) : 1.0;
    return lerpPose(p0, p1, u);
  }
}

// --- parsing ----------------------------------------------------------------

function parseMove(name, spec, where, known) {
  if (!isObject(spec)) throw new ChartError(`${where}: move '${name}' must be an object`);
  if (has(spec, "mirror_of")) {
    const src = known.get(spec.mirror_of);
    if (src == null) throw new ChartError(`${where}: move '${name}' mirrors unknown move '${spec.mirror_of}'`);
    const label = has(spec, "label") ? spec.label : src.label;
    return makeMove(name, label, src.beats,
      Object.freeze(src.keyframes.map((k) => Object.freeze({ beat: k.beat, pose: normalizePose(mirrorPose(k.pose)) }))),
      src.judge, src.pictogram);
  }
  let beats, raw;
  try {
    if (!has(spec, "beats") || !has(spec, "keyframes")) throw new TypeError("missing");
    beats = toFloat(spec.beats);
    raw = spec.keyframes;
  } catch {
    throw new ChartError(`${where}: move '${name}' needs 'beats' and 'keyframes'`);
  }
  if (beats <= 0 || !Array.isArray(raw) || !raw.length) {
    throw new ChartError(`${where}: move '${name}' needs beats > 0 and at least one keyframe`);
  }
  const frames = [];
  let last = -1.0;
  raw.forEach((k, i) => {
    let b, pose;
    try {
      if (!isObject(k)) throw new TypeError(`keyframe must be an object, got ${pyRepr(k)}`);
      if (!has(k, "beat")) throw new TypeError("'beat'");
      b = toFloat(k.beat);
      pose = normalizePose(has(k, "pose") ? k.pose : null);
    } catch (exc) {
      throw new ChartError(`${where}: move '${name}' keyframe ${i}: ${exc.message}`);
    }
    if (!(b >= 0.0 && b < beats)) {
      throw new ChartError(`${where}: move '${name}' keyframe ${i} at beat ${pyNum(b)} is outside 0..${pyNum(beats)}`);
    }
    if (b <= last) throw new ChartError(`${where}: move '${name}' keyframe beats must ascend (keyframe ${i})`);
    last = b;
    frames.push(Object.freeze({ beat: b, pose }));
  });
  const judge = has(spec, "judge") ? spec.judge : frames.map((_, i) => i);
  if (!Array.isArray(judge) || judge.some((i) => !Number.isInteger(i) || i < 0 || i >= frames.length)) {
    throw new ChartError(`${where}: move '${name}' judges a keyframe that doesn't exist: ${pyRepr(judge)}`);
  }
  const pict = has(spec, "pictogram") ? spec.pictogram : (judge.length ? judge[0] : 0);
  if (!Number.isInteger(pict) || pict < 0 || pict >= frames.length) {
    throw new ChartError(`${where}: move '${name}' pictogram index ${pyRepr(pict)} doesn't exist`);
  }
  const label = String(has(spec, "label") ? spec.label : titleCase(name.replaceAll("_", " ")));
  return makeMove(name, label, beats, Object.freeze(frames), Object.freeze([...judge]), pict);
}

/**
 * Moves from a {"name": spec} mapping -> Map. `mirror_of` may refer to
 * `known` or to moves earlier or later in the same mapping.
 */
export function parseMoves(data, where, known = null) {
  const out = new Map();
  const pending = new Map(Object.entries(data ?? {}));
  const base = new Map(known ?? []);
  while (pending.size) {
    let progressed = false;
    for (const [name, spec] of [...pending]) {
      const ref = isObject(spec) && has(spec, "mirror_of") ? spec.mirror_of : null;
      if (ref != null && !out.has(ref) && !base.has(ref)) {
        if (pending.has(ref)) continue; // its source isn't parsed yet
        throw new ChartError(`${where}: move '${name}' mirrors unknown move '${ref}'`);
      }
      out.set(name, parseMove(name, spec, where, new Map([...base, ...out])));
      pending.delete(name);
      progressed = true;
    }
    if (!progressed) {
      throw new ChartError(`${where}: moves mirror each other in a loop: ${pyRepr([...pending.keys()].sort())}`);
    }
  }
  return out;
}

/** The shared move library from moves.json's parsed contents. */
export function parseLibrary(data, where = "moves.json") {
  if (!isObject(data)) throw new ChartError(`${where}: not valid JSON (expected an object)`);
  return parseMoves(has(data, "moves") ? data.moves : {}, where);
}

/** Songs in menu order, from songs/index.json's parsed contents. */
export function parseSongIds(data, where = "songs/index.json") {
  if (!Array.isArray(data) || !data.every((i) => typeof i === "string")) {
    throw new ChartError(`${where}: expected a list of song ids`);
  }
  return [...data];
}

/**
 * A list of {move, beat, gold?} entries -> placed moves, validated: known
 * moves, no overlaps, nothing past the song's end.
 */
function parseChart(entries, moves, where, lastBeat) {
  if (!Array.isArray(entries)) throw new ChartError(`${where}: expected a list of moves`);
  const out = [];
  let end = -Infinity;
  entries.forEach((entry, i) => {
    let name, beat;
    try {
      if (!has(entry, "move") || !has(entry, "beat")) throw new TypeError("missing");
      name = entry.move;
      beat = toFloat(entry.beat);
    } catch {
      throw new ChartError(`${where}: chart entry ${i} needs 'move' and 'beat'`);
    }
    const move = moves.get(name);
    if (move == null) throw new ChartError(`${where}: chart entry ${i} at beat ${pyNum(beat)}: unknown move '${name}'`);
    if (beat < end - 1e-9) {
      throw new ChartError(`${where}: move '${name}' at beat ${pyNum(beat)} overlaps the previous move (ends at ${pyNum(end)})`);
    }
    const placed = new Placed(move, beat, Boolean(has(entry, "gold") ? entry.gold : false));
    if (placed.end > lastBeat + 1e-9) {
      throw new ChartError(`${where}: move '${name}' at beat ${pyNum(beat)} runs past the song's end (beat ${lastBeat.toFixed(1)})`);
    }
    out.push(placed);
    end = placed.end;
  });
  return out;
}

/** Flatten the chart into the coach's keyframes and the judge points. */
function indexSong(song) {
  song._frames = [];
  song._judges = [];
  song.chart.forEach((p, pIdx) => {
    p.move.keyframes.forEach((k, kIdx) => {
      const b = p.beat + k.beat;
      song._frames.push([b, k.pose]);
      if (p.move.judge.includes(kIdx)) {
        song._judges.push(Object.freeze({
          index: song._judges.length, beat: b, time: song.timeOf(b), pose: k.pose, gold: p.gold,
          weight: p.gold ? 2.0 : 1.0, label: p.move.label, placed: pIdx,
        }));
      }
    });
  });
}

/**
 * EASY's chart from a normal one: about one judged pose per bar, held (port
 * of chart.derive_easy; its docstring has the reasons).
 *
 * - Short moves (under a bar) are grouped by the bar they start in; each group
 *   becomes one held move: the kept pose, from its beat to the group's end.
 * - Which pose: a gold one first, else the earliest, skipping any closer than
 *   EASY_MIN_GAP beats to the previous kept pose or that a player frozen in it
 *   would score on (freezeScore >= FREEZE_MAX). A group with none left adds nothing.
 * - A group that had a gold point keeps a gold point.
 * - Moves a bar or longer and unjudged moves stay whole; the same gap and
 *   freeze checks decide which of their poses stay judged.
 */
export function deriveEasy(chart, bar = BAR_BEATS) {
  const out = [];
  let prev = null; // [beat, pose] of the last kept judge point
  const pick = (cands) => {
    // Gold first, then the earliest (a stable sort, like Python's).
    const sorted = [...cands].sort((a, b) => (a[2] === b[2] ? 0 : a[2] ? -1 : 1) || a[0] - b[0]);
    for (const c of sorted) {
      if (prev != null && (c[0] - prev[0] < EASY_MIN_GAP - 1e-9 || freezeScore(c[1], prev[1]) >= FREEZE_MAX)) continue;
      prev = [c[0], c[1]];
      return c;
    }
    return null;
  };
  const barOf = (beat) => Math.floor(beat / bar + 1e-9);
  const short = (p) => p.move.judge.length && p.move.beats < bar - 1e-9;

  let i = 0;
  const n = chart.length;
  while (i < n) {
    const p = chart[i];
    if (!short(p)) {
      const keep = [];
      for (const k of p.move.judge) {
        if (pick([[p.beat + p.move.keyframes[k].beat, p.move.keyframes[k].pose, p.gold, k]]) != null) keep.push(k);
      }
      const same = keep.length === p.move.judge.length && keep.every((k, x) => k === p.move.judge[x]);
      const move = same ? p.move : makeMove(p.move.name, p.move.label, p.move.beats, p.move.keyframes,
        Object.freeze(keep), p.move.pictogram);
      out.push(new Placed(move, p.beat, p.gold && keep.length > 0));
      i += 1;
      continue;
    }
    const group = [p];
    i += 1;
    while (i < n && short(chart[i]) && barOf(chart[i].beat) === barOf(p.beat)) {
      group.push(chart[i]);
      i += 1;
    }
    const cands = group.flatMap((g) => g.move.judge.map((k) => [g.beat + g.move.keyframes[k].beat,
      g.move.keyframes[k].pose, g.gold, g]));
    const c = pick(cands);
    if (c == null) continue;
    const [beat, pose, gold, src] = c;
    const end = Math.max(...group.map((g) => g.end));
    const held = makeMove(`${src.move.name}~held`, src.move.label, end - beat,
      Object.freeze([Object.freeze({ beat: 0.0, pose })]), Object.freeze([0]), 0);
    out.push(new Placed(held, beat, gold || group.some((g) => g.gold)));
  }
  return out;
}

/**
 * A Song from song.json's parsed contents, on a difficulty (default normal;
 * unknown ids are normal too). `folder` is the song's folder name (its id when
 * song.json doesn't say), `library` the parsed moves.json. Every chart in
 * song.json is validated, whichever difficulty is asked for.
 */
export function parseSong(data, folder, library = new Map(), difficulty = DEFAULT_DIFFICULTY) {
  const where = `song '${folder}'`;
  if (!isObject(data)) throw new ChartError(`${where}: song.json is not valid JSON (expected an object)`);
  for (const key of ["title", "bpm", "duration"]) {
    if (!has(data, key)) throw new ChartError(`${where}: song.json needs '${key}'`);
  }
  let bpm, duration;
  try {
    bpm = toFloat(data.bpm);
    duration = toFloat(data.duration);
  } catch {
    throw new ChartError(`${where}: bpm must be 40-240 and duration > 0`);
  }
  if (!(bpm >= 40 && bpm <= 240) || !(duration > 0)) {
    throw new ChartError(`${where}: bpm must be 40-240 and duration > 0`);
  }
  const custom = parseMoves(has(data, "custom_moves") ? data.custom_moves : {}, `${where} custom_moves`, library);
  const moves = new Map([...(library ?? []), ...custom]);

  const get = (k, dflt) => (has(data, k) ? data[k] : dflt);
  const song = new Song({
    id: String(get("id", folder)), title: String(data.title), artist: String(get("artist", "")),
    bpm, offset: toFloat(get("offset", 0.0)), duration,
    audio: String(get("audio", "audio.m4a")), difficulty: Math.trunc(toFloat(get("difficulty", 1))),
    coach: { ...get("coach", {}) }, sections: [...get("sections", [])], chart: [],
    preview: get("preview", null) ? [...data.preview] : null, folder,
  });
  const lastBeat = song.beatOf(duration);
  song.chart = parseChart(get("moves", []), moves, where, lastBeat);
  song.charts = { [DEFAULT_DIFFICULTY]: [...song.chart] };
  const extra = get("charts", {});
  if (!isObject(extra)) {
    throw new ChartError(`${where}: 'charts' must be an object like {"easy": [...], "hard": [...]}`);
  }
  for (const [level, entries] of Object.entries(extra)) {
    if (!DIFFICULTIES.includes(level) || level === DEFAULT_DIFFICULTY) {
      throw new ChartError(`${where}: charts.${level}: only 'easy' and 'hard' charts can be hand-made (normal is 'moves')`);
    }
    song.charts[level] = parseChart(entries, moves, `${where} charts.${level}`, lastBeat);
  }
  indexSong(song);
  return song.forDifficulty(difficulty);
}
