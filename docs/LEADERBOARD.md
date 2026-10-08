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

| `LEADERBOARD` in config.js | table |
|---|---|
| empty (the default) | **per browser** (localStorage): each computer keeps its own, like an arcade cabinet |
| a Supabase project URL + public key | **worldwide**: one table everyone shares |

The game never waits on the network. If the worldwide table can't be reached,
scores are read from and saved to this browser, and the table's subtitle says
"ON THIS DEVICE" instead of "WORLDWIDE".

## Turning on the worldwide table (Supabase, free tier)

1. Create a project at https://supabase.com (free).
2. In the project's **SQL Editor**, run:

```sql
create table public.scores (
  id         bigint generated always as identity primary key,
  game       text        not null check (game in ('snack', 'flappy')),
  name       text        not null check (name ~ '^[A-Z0-9]{1,3}$'),
  score      integer     not null check (score between 1 and 100000),
  created_at timestamptz not null default now()
);
create index scores_by_game on public.scores (game, score desc, created_at);

alter table public.scores enable row level security;
create policy "anyone can read scores" on public.scores
  for select using (true);
create policy "anyone can add a score" on public.scores
  for insert with check (true);
-- No update or delete policy: scores can't be changed or removed from the site.
```

3. Copy the **Project URL** and the **anon / publishable key** from Project
   Settings → API into `web/js/config.js`:

```js
export const LEADERBOARD = {
  supabaseUrl: "https://<project>.supabase.co",
  supabaseKey: "<anon or publishable key>",
};
```

4. Push. The Pages workflow deploys it.

Both values are designed to be public. The key can only do what the policies
above allow: read the table and add rows that pass the checks. Never put the
`service_role` / secret key in this file.

To remove a bad entry, delete the row in Supabase's Table Editor.

## Limits worth knowing

- **Anyone can post a fake score.** The game runs in the player's browser, so a
  determined person can send any number to the table. The database checks
  bound it, but they can't prove a score was really played. If that ever
  matters, scores need server-side checks, such as a Supabase Edge Function
  that rate-limits and sanity-checks score against play time.
- **Name filtering is minimal.** Three characters limits abuse, and
  `leaderboard.js` refuses a short list of obvious words. Moderate in the
  Table Editor.
- Adding a game means adding its id to the `check (game in (...))`
  constraint.
