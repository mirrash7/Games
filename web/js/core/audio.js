// Music playback with a clock games can sync to: the browser twin of src/kpapp/audio.py
// (docs/DANCE.md section 8).
//
//   unlockAudio()                       // from a click: browsers only start audio on a user gesture
//   const buffer = await loadTrack(url) // fetch + decode (AAC .m4a works in every current browser)
//   const player = new WebAudioPlayer(buffer)
//   player.play(start = 0); player.pause(); player.resume(); player.stop(); player.close()
//   player.position()                   // seconds the listener has *heard* (output latency compensated)
//   player.duration, player.playing, player.finished
//
// SilentPlayer keeps the same clock without sound (tests, no audio, a failed load).
// openPlayer(url, duration) gives a WebAudioPlayer, or a SilentPlayer if audio fails.
//
// This module touches no DOM at import time, so tests run it under Node with a fake context.

let shared = null;

/** The shared AudioContext (created on first use), or null where Web Audio is missing. */
export function audioContext() {
  if (!shared) {
    const Ctor = globalThis.AudioContext ?? globalThis.webkitAudioContext;
    if (!Ctor) return null;
    shared = new Ctor({ latencyHint: "interactive" });
  }
  return shared;
}

/** Replace the shared context (tests). */
export function setAudioContext(ctx) {
  shared = ctx;
}

/**
 * Create/resume the AudioContext. Call it synchronously inside a user gesture
 * (a click handler): browsers keep audio suspended until then.
 */
export function unlockAudio() {
  const ctx = audioContext();
  if (!ctx) return null;
  if (ctx.state === "suspended") ctx.resume()?.catch?.(() => {});
  try {
    // Older Safari only unlocks once something has actually played in the gesture.
    const blip = ctx.createBufferSource();
    blip.buffer = ctx.createBuffer(1, 1, ctx.sampleRate);
    blip.connect(ctx.destination);
    blip.start(0);
  } catch {
    // not needed elsewhere
  }
  return ctx;
}

/** Fetch and decode a track into an AudioBuffer on the shared context. */
export async function loadTrack(url, { context = audioContext(), fetchImpl = globalThis.fetch } = {}) {
  if (!context) throw new Error("Web Audio is not available");
  const res = await fetchImpl(url);
  if (!res.ok) throw new Error(`could not load ${url} (${res.status})`);
  const data = await res.arrayBuffer();
  return await context.decodeAudioData(data);
}

const wallClock = () => (globalThis.performance?.now?.() ?? Date.now()) / 1000;

/** The clock of a player, without sound. `clock` (seconds) is injectable for tests. */
export class SilentPlayer {
  constructor(duration = 0, { clock = wallClock } = {}) {
    this.duration = Number(duration) || 0;
    this._clock = clock;
    this._start = null; // clock time at which position 0 played
    this._pausedAt = 0;
    this.playing = false;
  }

  play(start = 0) {
    this._start = this._clock() - start;
    this._pausedAt = null;
    this.playing = true;
  }

  pause() {
    if (this.playing) {
      this._pausedAt = this.position();
      this.playing = false;
    }
  }

  resume() {
    if (!this.playing && this._pausedAt != null) this.play(this._pausedAt);
  }

  stop() {
    this.playing = false;
    this._pausedAt = 0;
  }

  close() {
    this.stop();
  }

  position() {
    if (!this.playing) return this._pausedAt ?? 0;
    const pos = this._clock() - (this._start ?? 0);
    return this.duration ? Math.min(pos, this.duration) : pos;
  }

  get finished() {
    return this.duration > 0 && this.position() >= this.duration;
  }
}

/**
 * Plays an AudioBuffer. A source node can only start once, so every play()
 * and resume() makes a new one.
 *
 * position() = where the playing source is in the buffer at the context time
 * the listener is hearing now: context.currentTime minus outputLatency
 * (baseLatency where that is missing). currentTime moves in steps of an audio
 * callback (3-10 ms); between steps the wall clock fills in, for at most 50 ms,
 * so animations driven by it stay smooth. It never goes backwards, freezes on
 * pause, and after play()/resume() holds until the new audio is audible.
 */
export class WebAudioPlayer {
  constructor(buffer, { context = audioContext(), clock = wallClock, destination = null } = {}) {
    if (!context) throw new Error("Web Audio is not available");
    this.buffer = buffer;
    this.ctx = context;
    this._clock = clock;
    this._out = destination ?? context.destination;
    this._source = null;
    this._startCtx = 0; // context time at which the source started
    this._startPos = 0; // buffer position it started from
    this._floor = 0;
    this._frozen = 0;
    this._lastCt = -1;
    this._lastWall = 0;
    this.playing = false;
  }

  get duration() {
    return this.buffer.duration;
  }

  get latency() {
    return this.ctx.outputLatency || this.ctx.baseLatency || 0;
  }

  _contextNow() {
    const ct = this.ctx.currentTime;
    const wall = this._clock();
    if (ct !== this._lastCt) {
      this._lastCt = ct;
      this._lastWall = wall;
      return ct;
    }
    return ct + Math.min(Math.max(0, wall - this._lastWall), 0.05);
  }

  _stopSource() {
    const src = this._source;
    this._source = null;
    if (!src) return;
    src.onended = null;
    try {
      src.stop();
    } catch {
      // already stopped
    }
    src.disconnect?.();
  }

  play(start = 0) {
    start = Math.min(Math.max(0, Number(start) || 0), this.duration);
    this._stopSource();
    if (this.ctx.state === "suspended") this.ctx.resume?.()?.catch?.(() => {});
    const src = this.ctx.createBufferSource();
    src.buffer = this.buffer;
    src.connect(this._out);
    const when = this.ctx.currentTime;
    src.start(when, start);
    this._source = src;
    this._startCtx = when;
    this._startPos = start;
    this._floor = start;
    this.playing = true;
  }

  pause() {
    if (!this.playing) return;
    const pos = this.position();
    this._stopSource();
    this.playing = false;
    this._frozen = pos;
  }

  resume() {
    if (!this.playing) this.play(this._frozen);
  }

  stop() {
    this._stopSource();
    this.playing = false;
    this._frozen = 0;
  }

  close() {
    this.stop();
  }

  position() {
    if (!this.playing) return this._frozen;
    const heard = this._startPos + (this._contextNow() - this.latency - this._startCtx);
    this._floor = Math.max(this._floor, Math.min(heard, this.duration));
    return this._floor;
  }

  get finished() {
    return this.duration > 0 && this.position() >= this.duration - 1e-3;
  }
}

/** A WebAudioPlayer for `url`, or a SilentPlayer(duration) if audio is unavailable. */
export async function openPlayer(url, duration = 0, { context = audioContext(), fetchImpl } = {}) {
  try {
    const buffer = await loadTrack(url, { context, fetchImpl });
    return new WebAudioPlayer(buffer, { context });
  } catch (err) {
    console.warn(`[kp] audio unavailable (${err?.message ?? err}); playing silently`);
    return new SilentPlayer(duration);
  }
}
