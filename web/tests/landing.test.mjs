// The landing page: catalog.json's schema and links, the menu's pure helpers
// (landing.js) and the hero's dance helpers (hero.js). DOM code in those
// modules only runs in a browser, so they import cleanly here.

import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import {
  countByCategory, filterExperiences, formatHash, groupsFor, menuCategories, parseHash, resolveCategory,
  seededRandom, sortExperiences, validateCatalog, videoSources,
} from "../landing/landing.js";
import {
  BONES, COCO, LOOP, STILL_BEAT, bounds, confidenceAt, heroPose, keypoints17, loopBeat, moveAt,
} from "../landing/hero.js";
import { parseLibrary, parseSong } from "../js/games/dance/chart.js";
import { GAMES } from "../js/games/index.js";

const WEB = new URL("../", import.meta.url);
const REPO = new URL("../../", import.meta.url);
const CATALOG = JSON.parse(readFileSync(new URL("landing/catalog.json", WEB), "utf8"));

// Site paths mounted from the repo (tools/build_web.py MOUNTS).
const MOUNTS = {
  "assets/snack/": "assets/fruitninja/generated/",
  "assets/flappy/": "assets/flappy/generated/",
  "assets/dance/": "assets/dance/",
};
/** A site-relative path -> the file that serves it, locally and on Pages. */
function siteFile(path) {
  for (const [site, repo] of Object.entries(MOUNTS)) {
    if (path.startsWith(site)) return new URL(repo + path.slice(site.length), REPO);
  }
  return new URL(path, WEB);
}

const readJSON = (p) => JSON.parse(readFileSync(new URL(p, REPO), "utf8"));
const NEON = parseSong(readJSON("assets/dance/songs/neon_nights/song.json"), "neon_nights",
  parseLibrary(readJSON("assets/dance/moves.json")));

// --- catalog.json ------------------------------------------------------------

test("catalog.json is valid", () => {
  assert.deepEqual(validateCatalog(CATALOG), []);
});

test("experience and category ids are unique, and every category exists", () => {
  const exp = CATALOG.experiences.map((e) => e.id);
  const cat = CATALOG.categories.map((c) => c.id);
  assert.equal(new Set(exp).size, exp.length);
  assert.equal(new Set(cat).size, cat.length);
  for (const e of CATALOG.experiences) assert.ok(cat.includes(e.category), `${e.id}: ${e.category}`);
});

test("the three arcade games are live, in Body-Controlled Games", () => {
  const live = CATALOG.experiences.filter((e) => e.status === "live");
  assert.deepEqual(live.map((e) => e.href).sort(), ["arcade/?game=dance", "arcade/?game=flappy", "arcade/?game=snack"]);
  for (const e of live) assert.equal(e.category, "body-games");
  const dance = live.find((e) => e.href.endsWith("dance"));
  assert.match(dance.summary, /EASY, NORMAL or HARD/);
  assert.match(dance.summary, /Two .*songs/);
});

test("live hrefs open a real page and a registered game", () => {
  const ids = new Set(GAMES.map((g) => g.id));
  for (const e of CATALOG.experiences.filter((x) => x.status === "live")) {
    const url = new URL(e.href, "https://site.invalid/");
    const page = url.pathname.slice(1);
    const file = siteFile(page.endsWith("/") || page === "" ? `${page}index.html` : page);
    assert.ok(existsSync(file), `${e.id}: ${e.href} has no page (${file.pathname})`);
    const game = url.searchParams.get("game");
    if (game != null) assert.ok(ids.has(game), `${e.id}: no game '${game}' in js/games/index.js`);
  }
});

test("live experiences have preview media; fallback art exists in the repo", () => {
  const mediaShipped = existsSync(new URL("media/", WEB));
  for (const e of CATALOG.experiences.filter((x) => x.status === "live")) {
    const m = e.media;
    assert.ok(m.mp4?.endsWith(".mp4") && m.webm?.endsWith(".webm") && m.poster, `${e.id}: media`);
    assert.ok(m.fallback && existsSync(siteFile(m.fallback)), `${e.id}: fallback ${m.fallback}`);
    // The showcase videos are rendered separately (web/media); once they're there, all must be.
    if (mediaShipped) {
      for (const k of ["mp4", "webm", "poster"]) assert.ok(existsSync(siteFile(m[k])), `${e.id}: ${m[k]}`);
    }
  }
});

test("coming-soon experiences aren't links and draw a placeholder", () => {
  const soon = CATALOG.experiences.filter((e) => e.status === "soon");
  assert.ok(soon.length >= 3);
  for (const e of soon) {
    assert.equal(e.href, undefined);
    assert.ok(["boxes", "mask", "graph"].includes(e.media?.motif), e.id);
  }
});

// --- validation catches mistakes -----------------------------------------------

const mini = () => ({
  categories: [{ id: "a", title: "A", blurb: "b" }, { id: "b", title: "B", blurb: "b" }],
  experiences: [
    { id: "x", title: "X", category: "a", status: "live", summary: "s", tags: ["t"], href: "arcade/?game=snack", media: { poster: "media/x.jpg" } },
    { id: "y", title: "Y", category: "b", status: "soon", summary: "s", tags: [] },
  ],
});

test("validateCatalog reports each kind of mistake", () => {
  assert.deepEqual(validateCatalog(mini()), []);
  const cases = [
    [(c) => { c.experiences[1].id = "x"; }, /duplicate id/],
    [(c) => { c.experiences[0].category = "nope"; }, /unknown category 'nope'/],
    [(c) => { delete c.experiences[0].href; }, /needs a site-relative href/],
    [(c) => { c.experiences[0].href = "https://example.com/"; }, /site-relative href/],
    [(c) => { c.experiences[0].href = "/arcade/"; }, /site-relative href/],
    [(c) => { c.experiences[0].media = {}; }, /media.poster or media.fallback/],
    [(c) => { c.experiences[1].href = "arcade/"; }, /only live experiences have an href/],
    [(c) => { c.experiences[1].status = "beta"; }, /status must be one of/],
    [(c) => { c.experiences[1].media = { motif: "spiral" }; }, /media.motif/],
    [(c) => { c.categories[1].id = "a"; }, /duplicate or reserved id/],
    [(c) => { c.categories[1].id = "all"; }, /duplicate or reserved id 'all'/],
    [(c) => { c.categories[0].id = "Body Games"; }, /lowercase/],
    [(c) => { c.experiences[0].tags = "arcade"; }, /tags must be a list/],
  ];
  for (const [mutate, message] of cases) {
    const c = mini();
    mutate(c);
    const errors = validateCatalog(c);
    assert.ok(errors.some((e) => message.test(e)), `${message}: ${JSON.stringify(errors)}`);
  }
  assert.deepEqual(validateCatalog(null), ["catalog must be an object"]);
});

// --- hash, filtering, menu -------------------------------------------------------

test("parseHash reads #cat=<id>; anything else means all", () => {
  assert.deepEqual(parseHash("#cat=body-games"), { cat: "body-games" });
  assert.deepEqual(parseHash("cat=body-games"), { cat: "body-games" });
  assert.deepEqual(parseHash("#experiences&cat=Segmentation"), { cat: "segmentation" });
  for (const h of ["", "#", "#experiences", "#cat=", "#cat=all", "#cat=%E0%A4%A", undefined, null]) {
    assert.deepEqual(parseHash(h), { cat: null }, String(h));
  }
  assert.equal(formatHash("body-games"), "#cat=body-games");
  assert.equal(formatHash(null), "#cat=all");
  assert.deepEqual(parseHash(formatHash("vision-agents")), { cat: "vision-agents" });
});

test("resolveCategory accepts only categories that have experiences", () => {
  assert.equal(resolveCategory(CATALOG, "body-games"), "body-games");
  assert.equal(resolveCategory(CATALOG, "nope"), null);
  assert.equal(resolveCategory(CATALOG, null), null);
  const c = mini();
  c.categories.push({ id: "empty", title: "E", blurb: "b" });
  assert.equal(resolveCategory(c, "empty"), null);
  assert.deepEqual(menuCategories(c).map((x) => x.id), ["a", "b"], "empty categories stay off the menu");
});

test("filtering: one category, or all with live experiences first", () => {
  assert.deepEqual(filterExperiences(CATALOG, "body-games").map((e) => e.id), ["snack-attack", "flappy-raccoon", "raccoon-dance"]);
  const all = filterExperiences(CATALOG, null);
  assert.equal(all.length, CATALOG.experiences.length);
  const firstSoon = all.findIndex((e) => e.status === "soon");
  assert.ok(all.slice(firstSoon).every((e) => e.status === "soon"));
  // Stable within a status.
  const list = [{ id: 1, status: "soon" }, { id: 2, status: "live" }, { id: 3, status: "soon" }, { id: 4, status: "live" }];
  assert.deepEqual(sortExperiences(list).map((e) => e.id), [2, 4, 1, 3]);
});

test("counts and groups follow the catalog", () => {
  const counts = countByCategory(CATALOG);
  assert.equal(counts.total, CATALOG.experiences.length);
  assert.equal(counts.live, 3);
  assert.equal(counts.byCategory["body-games"], 3);
  const menu = menuCategories(CATALOG);
  assert.deepEqual(menu.map((c) => c.id), CATALOG.categories.map((c) => c.id));
  assert.equal(menu[0].live, 3);
  assert.deepEqual(groupsFor(CATALOG, "segmentation").map((g) => g.category.id), ["segmentation"]);
  assert.equal(groupsFor(CATALOG, null).length, menu.length);
});

test("video sources prefer WebM, then MP4", () => {
  assert.deepEqual(videoSources({ mp4: "a.mp4", webm: "a.webm" }).map((s) => s.type), ["video/webm", "video/mp4"]);
  assert.deepEqual(videoSources({ poster: "a.jpg" }), []);
  assert.deepEqual(videoSources(undefined), []);
});

test("placeholder art is deterministic per id", () => {
  const a = seededRandom("object-finder"), b = seededRandom("object-finder"), c = seededRandom("cutout-studio");
  const xs = [a(), a(), a()];
  assert.deepEqual(xs, [b(), b(), b()]);
  assert.notDeepEqual(xs, [c(), c(), c()]);
  assert.ok(xs.every((x) => x >= 0 && x < 1));
});

// --- hero ------------------------------------------------------------------------

test("the hero loops the chart and crossfades its end into its start", () => {
  const bpm = NEON.bpm;
  assert.equal(loopBeat(0, bpm), LOOP.start);
  const len = LOOP.end - LOOP.start;
  const tLoop = (len * 60) / bpm;
  assert.ok(Math.abs(loopBeat(tLoop + 1, bpm) - loopBeat(1, bpm)) < 1e-9);
  for (let t = 0; t < 2 * tLoop; t += 0.37) {
    const b = loopBeat(t, bpm);
    assert.ok(b >= LOOP.start && b < LOOP.end);
  }
  // No jump at the wrap: the last frame before it is close to the first after it.
  const before = keypoints17(heroPose(NEON, LOOP.end - 1e-6));
  const after = keypoints17(heroPose(NEON, LOOP.start));
  const worst = Math.max(...before.map((p, i) => Math.hypot(p[0] - after[i][0], p[1] - after[i][1])));
  assert.ok(worst < 0.05, `jump of ${worst} shoulder widths at the loop`);
});

test("the hero dances the real choreography", () => {
  const mv = moveAt(NEON, STILL_BEAT);
  assert.ok(mv && mv.beat === STILL_BEAT, "the still frame lands on a move");
  assert.ok(/\w/.test(mv.label));
  assert.equal(moveAt(NEON, 2), null); // before the first move: idle groove
  assert.equal(moveAt(null, 20), null);
  // Different beats, different poses.
  const a = keypoints17(heroPose(NEON, 20)), b = keypoints17(heroPose(NEON, 34));
  assert.ok(a.some((p, i) => Math.hypot(p[0] - b[i][0], p[1] - b[i][1]) > 0.3));
  // Without a song it still moves (idle groove).
  assert.equal(keypoints17(heroPose(null, 3.3)).length, 17);
});

test("keypoints17 is a COCO skeleton, mirrored like the camera feed", () => {
  assert.equal(COCO.length, 17);
  const pts = keypoints17({ ra: [90, 90], la: [10, 5] }); // screen-right arm straight out
  assert.equal(pts.length, 17);
  assert.ok(pts.every((p) => p.length === 3 && p.every(Number.isFinite)));
  const idx = (n) => COCO.indexOf(n);
  // Mirrored: the screen-right limbs are left_* (AGENTS.md section 6).
  assert.ok(pts[idx("left_wrist")][0] > 1.0, "screen-right wrist out to the right");
  assert.ok(pts[idx("left_wrist")][0] > pts[idx("right_wrist")][0]);
  // y up: the head above the hips above the ankles.
  assert.ok(pts[idx("nose")][1] > pts[idx("left_hip")][1]);
  assert.ok(pts[idx("left_hip")][1] > pts[idx("left_ankle")][1]);
  for (const [a, b] of BONES) assert.ok(a >= 5 && b >= 5, "bones join body joints; the face is dots");
  const box = bounds(pts);
  assert.ok(box.max[0] - box.min[0] > 1.5 && box.max[1] - box.min[1] > 3);
});

test("the detection score stays believable", () => {
  for (let t = 0; t < 60; t += 0.1) {
    const v = Number(confidenceAt(t));
    assert.ok(v >= 0.95 && v <= 0.99, String(v));
  }
});
