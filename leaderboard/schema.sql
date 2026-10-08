-- Leaderboard tables (D1 / SQLite). Safe to re-run.
CREATE TABLE IF NOT EXISTS scores (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  game       TEXT    NOT NULL,
  name       TEXT    NOT NULL CHECK (length(name) BETWEEN 1 AND 3),
  score      INTEGER NOT NULL CHECK (score > 0),
  created_at INTEGER NOT NULL,  -- ms since epoch
  client     TEXT    NOT NULL   -- daily-rotating hash of the address, for rate limits
);
CREATE INDEX IF NOT EXISTS scores_board  ON scores (game, score DESC, created_at);
CREATE INDEX IF NOT EXISTS scores_client ON scores (client, created_at);

CREATE TABLE IF NOT EXISTS sessions (
  id         TEXT    PRIMARY KEY,  -- one per round played
  game       TEXT    NOT NULL,
  client     TEXT    NOT NULL,
  started_at INTEGER NOT NULL,     -- ms since epoch, measured by the server
  used       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS sessions_client  ON sessions (client, started_at);
CREATE INDEX IF NOT EXISTS sessions_started ON sessions (started_at);
