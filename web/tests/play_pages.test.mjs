// One page per game (games/<slug>/index.html): each names a registered game in
// <body data-game>, keeps every element js/main.js looks up, loads the shared
// runtime and service worker, and every local file it references exists.

import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { GAMES } from "../js/games/index.js";

const WEB = new URL("../", import.meta.url);
const REPO = new URL("../../", import.meta.url);

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

const PAGES = [
  { slug: "snack-attack", game: "snack", title: "Snack Attack" },
  { slug: "flappy-raccoon", game: "flappy", title: "Flappy Raccoon" },
  { slug: "raccoon-dance", game: "dance", title: "Raccoon Dance" },
];

// Element ids js/main.js reads (plus the progress bar's parts).
const REQUIRED_IDS = [
  "stage", "screen", "intro", "device", "go", "progress", "bar", "status", "model-size",
  "failure", "failure-text", "engine", "camera", "fullscreen", "debug",
];

const read = (url) => readFileSync(url, "utf8");
const pageUrl = (slug) => new URL(`games/${slug}/index.html`, WEB);

/** The opening tag of the element with this id, or null. */
function tagWithId(html, id) {
  const m = html.match(new RegExp(`<([a-z0-9]+)\\b[^>]*\\bid="${id}"[^>]*>`, "i"));
  return m ? { name: m[1].toLowerCase(), tag: m[0] } : null;
}

/** Local references in HTML attributes and CSS url()s (no http(s), data:, #, mailto:). */
function localRefs(text) {
  const out = [];
  for (const m of text.matchAll(/\b(?:src|href|poster)="([^"]+)"/g)) out.push(m[1]);
  for (const m of text.matchAll(/url\(\s*["']?([^"')]+)["']?\s*\)/g)) out.push(m[1]);
  for (const m of text.matchAll(/register\("([^"]+)"\)/g)) out.push(m[1]);
  return out.filter((r) => !/^(?:[a-z]+:|#|\/\/)/i.test(r));
}

/** Resolve a reference made from `from` (a URL inside web/) to the file that serves it. */
function resolveRef(ref, from) {
  const url = new URL(ref, from);
  assert.ok(url.href.startsWith(WEB.href), `${ref} (from ${from.pathname}) points outside the site`);
  let path = decodeURIComponent(url.pathname.slice(WEB.pathname.length));
  if (path === "" || path.endsWith("/")) path += "index.html";
  return siteFile(path);
}

test("there is one page per registered game, and it names that game", () => {
  const ids = GAMES.map((g) => g.id).sort();
  assert.deepEqual(PAGES.map((p) => p.game).sort(), ids);
  for (const p of PAGES) {
    assert.ok(existsSync(pageUrl(p.slug)), `missing web/games/${p.slug}/index.html`);
    const html = read(pageUrl(p.slug));
    const body = html.match(/<body\b[^>]*>/i)?.[0] ?? "";
    assert.match(body, new RegExp(`\\bdata-game="${p.game}"`), `${p.slug}: <body data-game="${p.game}">`);
    assert.match(html, new RegExp(`<title>${p.title} · RoboFlow Fun</title>`), `${p.slug}: <title>`);
  }
});

test("each page keeps every element main.js uses, with the right roles", () => {
  for (const p of PAGES) {
    const html = read(pageUrl(p.slug));
    for (const id of REQUIRED_IDS) {
      assert.ok(tagWithId(html, id), `${p.slug}: no element with id="${id}"`);
      assert.equal(html.match(new RegExp(`\\bid="${id}"`, "g")).length, 1, `${p.slug}: id="${id}" is not unique`);
    }
    const canvas = tagWithId(html, "screen");
    assert.equal(canvas.name, "canvas");
    assert.match(canvas.tag, /width="1280"/);
    assert.match(canvas.tag, /height="720"/);
    assert.equal(tagWithId(html, "intro").name, "section");
    assert.equal(tagWithId(html, "failure").name, "section");
    assert.match(tagWithId(html, "failure").tag, /\shidden\b/);
    assert.match(tagWithId(html, "progress").tag, /\shidden\b/);
    const go = tagWithId(html, "go");
    assert.equal(go.name, "button");
    assert.match(go.tag, /\sdisabled\b/, `${p.slug}: #go starts disabled (main.js enables it)`);
    assert.equal(tagWithId(html, "camera").name, "select");
    assert.equal(tagWithId(html, "fullscreen").name, "button");
    assert.equal(tagWithId(html, "debug").name, "button");
    assert.equal(tagWithId(html, "model-size").name, "span");
    assert.match(html, /onclick="location\.reload\(\)"/, `${p.slug}: retry button`);
    // The stage holds the canvas and both overlays (it is the fullscreen target).
    const stage = html.slice(html.indexOf('id="stage"'), html.indexOf("</main>"));
    for (const id of ["screen", "intro", "failure"]) assert.ok(stage.includes(`id="${id}"`), `${p.slug}: #${id} inside #stage`);
  }
});

test("each page loads the shared runtime and the isolation service worker", () => {
  for (const p of PAGES) {
    const html = read(pageUrl(p.slug));
    assert.match(html, /<script type="module" src="\.\.\/\.\.\/js\/main\.js"><\/script>/, `${p.slug}: main.js`);
    assert.match(html, /serviceWorker\.register\("\.\.\/\.\.\/coi-sw\.js"\)/, `${p.slug}: coi-sw.js`);
    assert.match(html, new RegExp(`og:image" content="https://mirrash7\\.github\\.io/Games/media/${p.game}\\.jpg"`));
    // Showcase video: muted autoplay loop inline, with the game's poster and both encodings.
    const video = html.match(/<video\b[^>]*>/)?.[0] ?? "";
    for (const attr of ["muted", "autoplay", "loop", "playsinline"]) assert.match(video, new RegExp(`\\b${attr}\\b`), `${p.slug}: video ${attr}`);
    assert.match(video, new RegExp(`poster="\\.\\./\\.\\./media/${p.game}\\.jpg"`));
    assert.ok(html.includes(`src="../../media/${p.game}.webm"`) && html.includes(`src="../../media/${p.game}.mp4"`), `${p.slug}: video sources`);
    // Cross-origin resources must load under COEP: require-corp (CORS-enabled CDN only).
    for (const m of html.matchAll(/\b(?:src|href)="(https?:\/\/[^"]+)"/g)) {
      const tag = html.slice(html.lastIndexOf("<", m.index), html.indexOf(">", m.index) + 1);
      if (/^<a\b/.test(tag)) continue; // links are navigation, not subresources
      assert.equal(new URL(m[1]).origin, "https://cdn.jsdelivr.net", `${p.slug}: cross-origin subresource ${m[1]}`);
    }
  }
});

test("every local file a page or its stylesheets reference exists", () => {
  for (const p of PAGES) {
    const page = pageUrl(p.slug);
    const html = read(page);
    const styles = [];
    for (const ref of localRefs(html)) {
      const file = resolveRef(ref, page);
      assert.ok(existsSync(file), `${p.slug}: ${ref} -> ${file.pathname} is missing`);
      if (ref.endsWith(".css")) styles.push(new URL(ref, page));
    }
    assert.ok(styles.some((u) => u.pathname.endsWith("/games/play.css")), `${p.slug}: shared play.css`);
    for (const css of styles) {
      for (const ref of localRefs(read(resolveRef(css.href, page)))) {
        const file = resolveRef(ref, css);
        assert.ok(existsSync(file), `${css.pathname}: ${ref} -> ${file.pathname} is missing`);
      }
    }
  }
});
