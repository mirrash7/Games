# Leaderboards

Each game has its own top-10 table with arcade-style names: 1-3 characters,
A-Z and 0-9.

When a round ends, the game's own game-over screen shows for a moment.
- **The score makes the table:** the player enters a name, by hovering their
  palm over the on-screen letters or by typing, and then sees the table with
  their row highlighted.
- **It doesn't:** the table is shown with their score underneath.

Each welcome-screen tile also shows that game's current high score.

## Where scores are kept

`web/js/config.js` decides where scores are kept:

| `LEADERBOARD.url` | table |
|---|---|
| empty | **per browser** (localStorage): each computer keeps its own, like an arcade cabinet |
| the leaderboard Worker's URL | **worldwide**: one table everyone shares |

The game never waits on the network. If the server can't be reached, or it
refuses a score, the score is saved in this browser instead. The table's
subtitle then says "ON THIS DEVICE" instead of "WORLDWIDE".

## The server: Cloudflare Worker + D1 (free)

`leaderboard/` holds a small Cloudflare Worker (`src/worker.js`) with a D1
(SQLite) database (`schema.sql`):

| request | does |
|---|---|
| `GET /scores?game=snack` | top 10 for a game |
| `POST /sessions {game}` | sent when a round starts; the server records the time |
| `POST /scores {game, name, score, session}` | saves a score for that round |

### Why Cloudflare

- **Free:** Workers Free allows 100,000 requests a day, and D1 Free allows
  5 million rows read and 100,000 written a day. Each round costs about 4
  requests and 2 writes.
- **Never pauses:** unlike Supabase's free projects, which pause after a week
  with no activity.
- **Code on the server:** it can check scores before saving them.

### What the server checks (`leaderboard/src/rules.js`)

- **Only from rounds it saw start.** Every score needs a session, which is
  created when the round's countdown ends. Each session can be used once.
- **Round length is measured on the server**, not reported by the browser.
  The score must be one the game's rules could produce in that time:
  - Snack Attack: at most 25 points a second, plus 20.
  - Flappy Raccoon: at most one pipe every 1.5 s, plus 3.

  These caps come from the games' own tuning, and `test/rules.test.mjs` fails
  if a tuning change makes them too tight. Rounds shorter than 3 s, or older
  than 2 hours, are refused.
- **Names:** 1-3 of A-Z/0-9, minus a short list of obvious abuse.
- **Rate limits:** 120 rounds and 30 scores an hour per player address. The
  address is stored only as a hash that changes daily, never as a raw IP.
- **Only the arcade's own pages can call it:** CORS allows
  `https://mirrash7.github.io` and `http://localhost:8765` (`ALLOWED_ORIGINS`
  in `wrangler.jsonc`).

This stops casual cheating and spam, not a determined cheater. Someone could
still start a round, wait, and post a plausible score by hand. Proving a score
was really played would need the server to replay the game. If that ever
matters, the next step is a server-side replay of recorded inputs.

### Deploy (one time, about 5 minutes)

```bash
./leaderboard/deploy.sh
```

The script:

1. Logs in to Cloudflare. It opens your browser; sign up for free if you
   don't have an account. A new account also picks a `workers.dev`
   subdomain, which becomes part of the URL.
2. Creates the `kp-leaderboard` D1 database and adds it to
   `leaderboard/wrangler.jsonc`.
3. Creates the tables.
4. Deploys the Worker and writes its URL into `web/js/config.js`.

Then commit and push `leaderboard/wrangler.jsonc` and `web/js/config.js`; the
Pages workflow puts the site live. Re-run the script any time to redeploy
server changes. To check a deployment:

```bash
node leaderboard/test/smoke.mjs https://kp-leaderboard.<subdomain>.workers.dev
```

The check adds one "TST" score; delete it as described below.

### Moderating

To delete a row, use the Cloudflare dashboard → Storage & Databases → D1 →
kp-leaderboard → Console, or:

```bash
npx wrangler@4.138.0 d1 execute kp-leaderboard --remote --command "DELETE FROM scores WHERE id = 123"
```

### Adding a game

Add its limits to `GAMES` in `leaderboard/src/rules.js`, with the reasoning
from its rules, plus a test in `test/rules.test.mjs`. Then redeploy.

## Develop locally

See `leaderboard/README.md`: `wrangler dev --local` runs the Worker and a
local database. `http://localhost:8765/?leaderboard=http://localhost:8787`
points the site at it.
