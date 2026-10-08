# Leaderboard server

The worldwide high-score tables for the browser arcade: a Cloudflare Worker
(`src/worker.js`) backed by a D1 (SQLite) database (`schema.sql`). It runs on
Cloudflare's free plan. Setup, limits and anti-cheat rules are described in
[docs/LEADERBOARD.md](../docs/LEADERBOARD.md).

```bash
./deploy.sh                                   # set up + deploy (re-run to redeploy)
node test/smoke.mjs https://<your-worker>.workers.dev   # end-to-end check (adds a TST score)
node --test test/rules.test.mjs               # limits vs the games' own rules
```

To run locally (no Cloudflare account needed), use a copy of `wrangler.jsonc`
that has a `d1_databases` entry with any `database_id`:

```bash
npx wrangler@4.138.0 d1 execute kp-leaderboard --local --file schema.sql
npx wrangler@4.138.0 dev --local --port 8787
node test/smoke.mjs http://localhost:8787
# then open http://localhost:8765/?leaderboard=http://localhost:8787
```
