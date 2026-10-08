// High-score tables, one per game, with arcade-style names of up to three
// characters.
//
// Two backends with the same surface (begin, top, submit):
//   LocalBoard   this browser only (localStorage). Always works.
//   ServerBoard  one worldwide table: the Cloudflare Worker in leaderboard/,
//                used when config.js has its URL (docs/LEADERBOARD.md).
// createBoard() picks one, and falls back to the local board if the server
// can't be reached, so a flaky connection never blocks the game.
//
// begin(game) is called when a round starts. The server records the time and
// later refuses scores the game couldn't produce in that time.

export const NAME_MAX = 3;
export const BOARD_SIZE = 10;
export const MAX_SCORE = 100000; // matches the server's limit; anything above is not a real run

// Three letters can still spell abuse on a public board. A short, obvious list
// (the server has the same one); the player is asked for another name.
const BLOCKED = new Set([
  "ASS", "CUM", "FUK", "FUC", "FCK", "FKU", "DIK", "DIC", "COK", "KKK", "NIG", "NGR", "FAG", "TIT",
  "SHT", "CNT", "SEX", "XXX", "NAZ", "SS", "KYS", "PIS", "POO", "WTF",
]);

/** Upper-case A-Z and 0-9 only, at most three. */
export function cleanName(s) {
  return String(s ?? "").toUpperCase().replace(/[^A-Z0-9]/g, "").slice(0, NAME_MAX);
}

export function nameAllowed(name) {
  const n = cleanName(name);
  return n.length > 0 && n === String(name) && !BLOCKED.has(n);
}

/** Highest score first; on a tie, whoever got there first. */
export function sortEntries(entries) {
  return [...entries].sort((a, b) => b.score - a.score || a.at - b.at);
}

/** Would `score` earn a place on a board currently holding `entries`? */
export function qualifies(entries, score, size = BOARD_SIZE) {
  if (!(score > 0) || score > MAX_SCORE) return false;
  if (entries.length < size) return true;
  return score > sortEntries(entries)[size - 1].score; // ties go to the earlier run
}

function safeStorage() {
  try {
    const s = globalThis.localStorage;
    s.setItem("kp.probe", "1");
    s.removeItem("kp.probe");
    return s;
  } catch {
    return null;
  }
}

/** This browser's own table. `storage` is injectable for tests. */
export class LocalBoard {
  constructor(storage = safeStorage(), key = "kp.leaderboard.v1") {
    this.scope = "device";
    this._key = key;
    this._storage = storage;
    this._memory = {}; // private mode: keep scores for this visit at least
  }

  _all() {
    if (!this._storage) return this._memory;
    try {
      return JSON.parse(this._storage.getItem(this._key) ?? "{}") ?? {};
    } catch {
      return {};
    }
  }

  async begin() {
    return null; // nothing to verify locally
  }

  async top(game) {
    return sortEntries(this._all()[game] ?? []).slice(0, BOARD_SIZE);
  }

  async submit(game, name, score) {
    const all = this._all();
    const entry = { id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`, name: cleanName(name), score, at: Date.now() };
    all[game] = sortEntries([...(all[game] ?? []), entry]).slice(0, BOARD_SIZE);
    if (this._storage) {
      try {
        this._storage.setItem(this._key, JSON.stringify(all));
      } catch {
        /* quota: keep the in-memory copy */
      }
    } else this._memory = all;
    return { entries: all[game], id: entry.id };
  }
}

/** The worldwide table: the leaderboard Worker (leaderboard/src/worker.js). */
export class ServerBoard {
  constructor({ url, fetchImpl = globalThis.fetch?.bind(globalThis) }) {
    this.scope = "world";
    this._base = url.replace(/\/$/, "");
    this._fetch = fetchImpl;
  }

  async _call(method, path, body) {
    const res = await this._fetch(this._base + path, {
      method,
      headers: body ? { "Content-Type": "application/json" } : {},
      body: body ? JSON.stringify(body) : undefined,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const err = new Error(data.error ?? `leaderboard ${res.status}`);
      err.status = res.status;
      throw err;
    }
    return data;
  }

  /** Tell the server a round started; returns the session to submit with. */
  async begin(game) {
    return (await this._call("POST", "/sessions", { game })).session;
  }

  async top(game) {
    return (await this._call("GET", `/scores?game=${encodeURIComponent(game)}`)).entries;
  }

  async submit(game, name, score, session) {
    const out = await this._call("POST", "/scores", { game, name: cleanName(name), score: Math.round(score), session });
    return { entries: out.entries, id: out.id };
  }
}

/**
 * The worldwide board when configured, wrapped so any network failure falls
 * back to this device's board (and says so through `scope`).
 */
export function createBoard(config = {}, local = new LocalBoard(), fetchImpl = undefined) {
  if (!config.url) return local;
  const remote = new ServerBoard({ url: config.url, ...(fetchImpl ? { fetchImpl } : {}) });
  return {
    scope: "world",
    async begin(game) {
      try {
        return await remote.begin(game);
      } catch (err) {
        console.warn("[kp] worldwide leaderboard unavailable for this round:", err);
        return null; // the score will be kept on this device
      }
    },
    async top(game) {
      try {
        const entries = await remote.top(game);
        this.scope = "world";
        return entries;
      } catch (err) {
        console.warn("[kp] worldwide leaderboard unavailable, using this device's:", err);
        this.scope = "device";
        return local.top(game);
      }
    },
    async submit(game, name, score, session) {
      if (!session) {
        this.scope = "device";
        return local.submit(game, name, score);
      }
      try {
        const out = await remote.submit(game, name, score, session);
        this.scope = "world";
        return out;
      } catch (err) {
        console.warn("[kp] the worldwide leaderboard refused or missed the score; saved on this device:", err);
        this.scope = "device";
        return local.submit(game, name, score);
      }
    },
  };
}
