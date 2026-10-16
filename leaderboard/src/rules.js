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
//  golf:   points arrive only when a hole is finished, at most 1000 (an ace)
//          per hole, and no hole can be finished sooner than
//          FASTEST_HOLE_SECONDS (8.7 s: the intro, an address hold, a swing
//          and an ace's flight), so <= 1000 / 8.7 = 115/s; 125/s with no
//          slack. Three holes: at most 3000. (web/js/games/golf/game.js
//          POINTS_ACE, FASTEST_HOLE_SECONDS.)
export const GAMES = {
  snack: { perSecond: 25, slack: 20, max: 100000 },
  flappy: { perSecond: 1 / 1.5, slack: 3, max: 100000 },
  golf: { perSecond: 125, slack: 0, max: 3000 },
};

// Raccoon Dance has one board per song: "dance-<song id>" (docs/DANCE.md
// section 9), so new songs need no server change. Every song scores 0..10000,
// and 10000 needs every judge point resolved, which happens only after the
// last one (at the song's last judged beat) has been danced. Songs run at
// least ~30 s, so a perfect run earns at most 10000 / 30 = 333 points per
// second of play; 400/s with no slack refuses nothing real as long as a
// song's last judged move is 25 s (10000 / 400) or more into it, which
// test/rules.test.mjs checks for every song in assets/dance/songs/.
export const DANCE_ID = /^dance-[a-z0-9_]{1,40}$/;
export const DANCE = { perSecond: 400, slack: 0, max: 10000 };

/** The rules for a board id, or null if the server doesn't know it. */
export function gameRules(game) {
  if (typeof game !== "string") return null;
  if (Object.hasOwn(GAMES, game)) return GAMES[game];
  if (DANCE_ID.test(game)) return DANCE;
  return null;
}

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
  const g = gameRules(game);
  if (!g) return "unknown game";
  if (!Number.isInteger(score) || score < 1 || score > g.max) return "score out of range";
  if (seconds < MIN_ROUND) return "round too short";
  if (seconds > MAX_ROUND) return "round expired";
  if (score > g.perSecond * seconds + g.slack) return "score too high for the time played";
  return null;
}
