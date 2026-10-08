// End-to-end check of a running leaderboard Worker (local or deployed):
//   node test/smoke.mjs http://localhost:8787
//   node test/smoke.mjs https://kp-leaderboard.<you>.workers.dev
// Writes one real "TST" score to the snack board, so prefer a local run;
// after a check against production, delete that row in the Cloudflare dashboard.
import assert from "node:assert/strict";

const base = (process.argv[2] ?? "http://localhost:8787").replace(/\/$/, "");
const origin = "https://mirrash7.github.io";
const call = async (method, path, body, headers = {}) => {
  const res = await fetch(base + path, {
    method,
    headers: { Origin: origin, ...(body ? { "Content-Type": "application/json" } : {}), ...headers },
    body: body ? JSON.stringify(body) : undefined,
  });
  return { status: res.status, headers: res.headers, data: res.status === 204 ? null : await res.json() };
};
const sleep = (s) => new Promise((r) => setTimeout(r, s * 1000));
const check = async (label, fn) => {
  await fn();
  console.log(`ok  ${label}`);
};

await check("health", async () => assert.equal((await call("GET", "/health")).data.ok, true));
await check("reads a board", async () => assert.ok(Array.isArray((await call("GET", "/scores?game=snack")).data.entries)));
await check("unknown game refused", async () => assert.equal((await call("GET", "/scores?game=pong")).status, 400));
await check("CORS for the arcade", async () => {
  const r = await call("OPTIONS", "/scores");
  assert.equal(r.headers.get("access-control-allow-origin"), origin);
});
await check("no CORS for other sites", async () => {
  const r = await call("OPTIONS", "/scores", null, { Origin: "https://example.com" });
  assert.equal(r.headers.get("access-control-allow-origin"), null);
});
await check("a score needs a round", async () => {
  const r = await call("POST", "/scores", { game: "snack", name: "TST", score: 5, session: "nope" });
  assert.equal(r.status, 422);
});

const { session } = (await call("POST", "/sessions", { game: "snack" })).data;
assert.ok(session);
await check("too soon after the round started", async () => {
  const r = await call("POST", "/scores", { game: "snack", name: "TST", score: 5, session });
  assert.equal(r.status, 422);
  assert.match(r.data.error, /short/);
});
await sleep(3.3);
await check("impossible score for the time played", async () => {
  const r = await call("POST", "/scores", { game: "snack", name: "TST", score: 5000, session });
  assert.equal(r.status, 422);
  assert.match(r.data.error, /too high/);
});
await check("wrong game for the round", async () => {
  assert.equal((await call("POST", "/scores", { game: "flappy", name: "TST", score: 1, session })).status, 422);
});
await check("bad names", async () => {
  for (const name of ["abc", "ABCD", "", "ASS"]) {
    assert.equal((await call("POST", "/scores", { game: "snack", name, score: 5, session })).status, 422, name);
  }
});
let id;
await check("a real score is saved and ranked", async () => {
  const r = await call("POST", "/scores", { game: "snack", name: "TST", score: 7, session });
  assert.equal(r.status, 200, JSON.stringify(r.data));
  id = r.data.id;
  assert.ok(r.data.entries.some((e) => e.id === id && e.name === "TST" && e.score === 7));
});
await check("one score per round", async () => {
  assert.equal((await call("POST", "/scores", { game: "snack", name: "TST", score: 7, session })).status, 409);
});
console.log(`all good against ${base}`);
