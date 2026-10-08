// What the leaderboard server accepts. Pure functions, shared by the Worker
// and its tests.
//
// The games run in players' browsers, so the server can't prove a score was
// earned. What it can do is require a round that it saw start, and refuse
// scores faster than the game's own rules allow. A fake score then costs the
// cheater real waiting time and still has to look plausible.

export const BOARD_SIZE = 10;
export const NAME_MAX = 3;

// Fastest possible scoring, from each game's rules, with headroom.
//  snack:  waves come at most every 0.8 s, with at most 5 snacks (2 points
//          each for a cookie) plus a combo bonus of 2 per snack: <= 20 points
//          per 0.8 s = 25/s. (web/js/games/snack: entities.js gap >= 0.8,
//          count <= 5; game.js bonus = count * 2.)
//  flappy: one point per pipe; pipes come at least 1.55 s apart at the
//          hardest, so <= 0.65/s. (web/js/games/flappy/game.js intervalMin.)
export const GAMES = {
  snack: { perSecond: 25, slack: 20, max: 100000 },
  flappy: { perSecond: 1 / 1.5, slack: 3, max: 100000 },
};

export const MIN_ROUND = 3; // seconds: nothing scores before the countdown ends
export const MAX_ROUND = 2 * 60 * 60; // a session older than this is stale
export const SESSIONS_PER_HOUR = 120; // per player address
export const SCORES_PER_HOUR = 30;

// Three letters can still spell abuse on a public board. Same list as the site.
const BLOCKED = new Set([
  "ASS", "CUM", "FUK", "FUC", "FCK", "FKU", "DIK", "DIC", "COK", "KKK", "NIG", "NGR", "FAG", "TIT",
  "SHT", "CNT", "SEX", "XXX", "NAZ", "SS", "KYS", "PIS", "POO", "WTF",
]);

export function nameError(name) {
  if (typeof name !== "string" || !/^[A-Z0-9]{1,3}$/.test(name)) return "names are 1-3 characters, A-Z and 0-9";
  if (BLOCKED.has(name)) return "please pick another name";
  return null;
}

/** Why `score` after `seconds` of play can't be real, or null if it could be. */
export function scoreError(game, score, seconds) {
  const g = GAMES[game];
  if (!g) return "unknown game";
  if (!Number.isInteger(score) || score < 1 || score > g.max) return "score out of range";
  if (seconds < MIN_ROUND) return "round too short";
  if (seconds > MAX_ROUND) return "round expired";
  if (score > g.perSecond * seconds + g.slack) return "score too high for the time played";
  return null;
}
