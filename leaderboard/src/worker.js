// Leaderboard API for the Motion Arcade (Cloudflare Worker + D1).
//
//   GET  /scores?game=snack          -> {entries: [{id, name, score, at}]}   top 10
//   POST /sessions  {game}           -> {session}        when a round starts
//   POST /scores    {game, name, score, session}  -> {id, entries}
//
// A score needs a session the server issued when the round started. The
// server measures the round's length itself and refuses scores the game's
// rules couldn't produce in that time (rules.js). Players are rate-limited
// by a daily-rotating hash of their address; raw IPs are never stored.

import { BOARD_SIZE, GAMES, SCORES_PER_HOUR, SESSIONS_PER_HOUR, nameError, scoreError } from "./rules.js";

export default {
  async fetch(request, env) {
    const origin = request.headers.get("Origin") ?? "";
    const headers = corsHeaders(origin, env);
    if (request.method === "OPTIONS") return new Response(null, { status: 204, headers });
    try {
      const url = new URL(request.url);
      const route = `${request.method} ${url.pathname.replace(/\/+$/, "")}`;
      if (route === "GET /scores") return reply(await topScores(env, url.searchParams.get("game")), 200, headers);
      if (route === "POST /sessions") return await startSession(request, env, headers);
      if (route === "POST /scores") return await submitScore(request, env, headers);
      if (route === "GET " || route === "GET /health") return reply({ ok: true }, 200, headers);
      return reply({ error: "not found" }, 404, headers);
    } catch (err) {
      if (err instanceof HttpError) return reply({ error: err.message }, err.status, headers);
      console.error(err);
      return reply({ error: "server error" }, 500, headers);
    }
  },
};

class HttpError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

function reply(data, status, headers) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { ...headers, "Content-Type": "application/json", "Cache-Control": "no-store" },
  });
}

/** Only the arcade's own pages may call this from a browser. */
function corsHeaders(origin, env) {
  const allowed = String(env.ALLOWED_ORIGINS ?? "").split(",").map((s) => s.trim()).filter(Boolean);
  const base = { "Cross-Origin-Resource-Policy": "cross-origin" };
  if (!allowed.includes(origin)) return base;
  return {
    ...base,
    "Access-Control-Allow-Origin": origin,
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Max-Age": "86400",
    Vary: "Origin",
  };
}

async function body(request) {
  const text = await request.text();
  if (text.length > 1000) throw new HttpError(413, "request too large");
  try {
    return JSON.parse(text);
  } catch {
    throw new HttpError(400, "expected JSON");
  }
}

function checkGame(game) {
  if (!Object.hasOwn(GAMES, game ?? "")) throw new HttpError(400, "unknown game");
  return game;
}

/** A per-day pseudonym for the caller's address: enough to rate-limit, not to identify. */
async function clientId(request) {
  const ip = request.headers.get("CF-Connecting-IP") ?? "local";
  const day = new Date().toISOString().slice(0, 10);
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(`kp-leaderboard|${day}|${ip}`));
  return [...new Uint8Array(digest).slice(0, 12)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

async function topScores(env, game) {
  checkGame(game);
  const { results } = await env.DB.prepare(
    "SELECT id, name, score, created_at FROM scores WHERE game = ?1 ORDER BY score DESC, created_at ASC LIMIT ?2",
  ).bind(game, BOARD_SIZE).all();
  return { entries: results.map((r) => ({ id: String(r.id), name: r.name, score: r.score, at: r.created_at })) };
}

async function startSession(request, env, headers) {
  const { game } = await body(request);
  checkGame(game);
  const client = await clientId(request);
  const now = Date.now();
  const recent = await env.DB.prepare("SELECT COUNT(*) AS n FROM sessions WHERE client = ?1 AND started_at > ?2")
    .bind(client, now - 3600_000).first();
  if (recent.n >= SESSIONS_PER_HOUR) throw new HttpError(429, "too many rounds - try again later");
  const id = crypto.randomUUID();
  const stmts = [env.DB.prepare("INSERT INTO sessions (id, game, client, started_at) VALUES (?1, ?2, ?3, ?4)")
    .bind(id, game, client, now)];
  // Now and then, forget rounds older than a day.
  if (Math.random() < 0.02) stmts.push(env.DB.prepare("DELETE FROM sessions WHERE started_at < ?1").bind(now - 86400_000));
  await env.DB.batch(stmts);
  return reply({ session: id }, 200, headers);
}

async function submitScore(request, env, headers) {
  const { game, name, score, session } = await body(request);
  checkGame(game);
  const bad = nameError(name);
  if (bad) throw new HttpError(422, bad);
  if (typeof session !== "string" || session.length > 64) throw new HttpError(422, "no round to attach the score to");

  const now = Date.now();
  const round = await env.DB.prepare("SELECT game, started_at, used FROM sessions WHERE id = ?1").bind(session).first();
  if (!round || round.game !== game) throw new HttpError(422, "no round to attach the score to");
  if (round.used) throw new HttpError(409, "this round's score was already saved");
  const why = scoreError(game, score, (now - round.started_at) / 1000);
  if (why) throw new HttpError(422, why);

  const client = await clientId(request);
  const recent = await env.DB.prepare("SELECT COUNT(*) AS n FROM scores WHERE client = ?1 AND created_at > ?2")
    .bind(client, now - 3600_000).first();
  if (recent.n >= SCORES_PER_HOUR) throw new HttpError(429, "too many scores - try again later");

  // Spend the session first; `used = 0` makes a double submit of the same
  // round lose the race instead of saving twice.
  const spent = await env.DB.prepare("UPDATE sessions SET used = 1 WHERE id = ?1 AND used = 0").bind(session).run();
  if (!spent.meta.changes) throw new HttpError(409, "this round's score was already saved");
  const row = await env.DB.prepare(
    "INSERT INTO scores (game, name, score, created_at, client) VALUES (?1, ?2, ?3, ?4, ?5) RETURNING id",
  ).bind(game, name, score, now, client).first();
  const { entries } = await topScores(env, game);
  return reply({ id: String(row.id), entries }, 200, headers);
}
