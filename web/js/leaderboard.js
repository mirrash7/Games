// High-score tables, one per game, with arcade-style names of up to three
// characters.
//
// Two backends with the same surface:
//   LocalBoard     this browser only (localStorage). Always works.
//   SupabaseBoard  one worldwide table, via Supabase's REST API. Used when
//                  config.js has a project URL and public key (docs/LEADERBOARD.md).
// createBoard() picks one, and falls back to the local board if the network
// table can't be reached, so a flaky connection never blocks the game.

export const NAME_MAX = 3;
export const BOARD_SIZE = 10;
export const MAX_SCORE = 100000; // matches the database check; anything above is not a real run

// Three letters can still spell abuse on a public board. A short, obvious list;
// the player is asked for another name.
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

/** One worldwide table in a Supabase project (see docs/LEADERBOARD.md for the SQL). */
export class SupabaseBoard {
  constructor({ url, key, table = "scores", fetchImpl = globalThis.fetch?.bind(globalThis) }) {
    this.scope = "world";
    this._base = `${url.replace(/\/$/, "")}/rest/v1/${table}`;
    this._headers = { apikey: key, Authorization: `Bearer ${key}`, "Content-Type": "application/json" };
    this._fetch = fetchImpl;
  }

  async top(game) {
    const q = `?select=id,name,score,created_at&game=eq.${encodeURIComponent(game)}`
      + `&order=score.desc,created_at.asc&limit=${BOARD_SIZE}`;
    const res = await this._fetch(this._base + q, { headers: this._headers });
    if (!res.ok) throw new Error(`leaderboard ${res.status}`);
    return (await res.json()).map((r) => ({ id: String(r.id), name: r.name, score: r.score, at: Date.parse(r.created_at) }));
  }

  async submit(game, name, score) {
    const res = await this._fetch(this._base, {
      method: "POST",
      headers: { ...this._headers, Prefer: "return=representation" },
      body: JSON.stringify({ game, name: cleanName(name), score: Math.round(score) }),
    });
    if (!res.ok) throw new Error(`leaderboard ${res.status}`);
    const [row] = await res.json();
    return { entries: await this.top(game), id: String(row?.id ?? "") };
  }
}

/**
 * The worldwide board when configured, wrapped so any network failure falls
 * back to this device's board (and says so through `scope`).
 */
export function createBoard(config = {}) {
  const local = new LocalBoard();
  if (!config.supabaseUrl || !config.supabaseKey) return local;
  const remote = new SupabaseBoard({ url: config.supabaseUrl, key: config.supabaseKey });
  return {
    scope: "world",
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
    async submit(game, name, score) {
      try {
        const out = await remote.submit(game, name, score);
        this.scope = "world";
        return out;
      } catch (err) {
        console.warn("[kp] could not post to the worldwide leaderboard, saved on this device:", err);
        this.scope = "device";
        return local.submit(game, name, score);
      }
    },
  };
}
