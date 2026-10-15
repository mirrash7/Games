// The landing page: the experiences catalog, its category menu and the hero.
//
// Everything the page lists comes from landing/catalog.json. Adding an
// experience (or a whole category) is a JSON edit; no HTML changes.
//
//   {
//     "version": 1,
//     "categories": [                       // menu order = this order
//       { "id": "body-games",               // [a-z0-9-], used in #cat=<id>
//         "title": "Body-Controlled Games",
//         "blurb": "One or two sentences." }
//     ],
//     "experiences": [                      // order within a category = this order (live first)
//       { "id": "snack-attack",             // unique, [a-z0-9-]
//         "title": "Snack Attack",
//         "category": "body-games",         // a category id above
//         "status": "live" | "soon",        // soon: shown as "Coming soon", not clickable
//         "summary": "One or two sentences.",
//         "tags": ["hand tracking", ...],   // short; shown as mono chips
//         "href": "arcade/?game=snack",     // live only: where PLAY goes (relative to the site root)
//         "media": {                        // all optional; paths relative to the site root
//           "mp4": "media/snack.mp4",       //   hover / in-view preview, muted, looped
//           "webm": "media/snack.webm",
//           "poster": "media/snack.jpg",    //   still shown until the video plays
//           "fallback": "assets/...png",    //   sprite shown (in a detection box) when the poster is missing
//           "fallbackLabel": "raccoon",     //   that box's label
//           "motif": "boxes" | "mask" | "graph"  // drawn placeholder for "soon" cards
//         },
//         "needs": ["webcam"],              // shown as chips: webcam, microphone, ...
//         "model": "RF-DETR keypoints" }    // optional: which model powers it
//     ]
//   }
//
// The top half of this file is pure (no DOM) so Node can test it
// (web/tests/landing.test.mjs); boot() below it runs only in a browser.

export const STATUSES = Object.freeze(["live", "soon"]);
export const MOTIFS = Object.freeze(["boxes", "mask", "graph"]);
const ID = /^[a-z0-9][a-z0-9-]*$/;

/** Every problem with a catalog, as readable strings ([] = valid). */
export function validateCatalog(catalog) {
  const errors = [];
  if (catalog == null || typeof catalog !== "object") return ["catalog must be an object"];
  const cats = Array.isArray(catalog.categories) ? catalog.categories : null;
  const exps = Array.isArray(catalog.experiences) ? catalog.experiences : null;
  if (!cats) errors.push("'categories' must be a list");
  if (!exps) errors.push("'experiences' must be a list");
  if (!cats || !exps) return errors;

  const catIds = new Set();
  cats.forEach((c, i) => {
    const where = `categories[${i}]`;
    if (!c || typeof c !== "object") return errors.push(`${where} must be an object`);
    if (typeof c.id !== "string" || !ID.test(c.id)) errors.push(`${where}: id must be lowercase letters, digits and dashes`);
    else if (catIds.has(c.id) || c.id === "all") errors.push(`${where}: duplicate or reserved id '${c.id}'`);
    else catIds.add(c.id);
    if (!nonEmpty(c.title)) errors.push(`${where}: needs a title`);
    if (!nonEmpty(c.blurb)) errors.push(`${where}: needs a blurb`);
  });

  const expIds = new Set();
  exps.forEach((e, i) => {
    const where = `experiences[${i}]${e && typeof e.id === "string" ? ` '${e.id}'` : ""}`;
    if (!e || typeof e !== "object") return errors.push(`${where} must be an object`);
    if (typeof e.id !== "string" || !ID.test(e.id)) errors.push(`${where}: id must be lowercase letters, digits and dashes`);
    else if (expIds.has(e.id)) errors.push(`${where}: duplicate id`);
    else expIds.add(e.id);
    if (!nonEmpty(e.title)) errors.push(`${where}: needs a title`);
    if (!nonEmpty(e.summary)) errors.push(`${where}: needs a summary`);
    if (!catIds.has(e.category)) errors.push(`${where}: unknown category '${e.category}'`);
    if (!STATUSES.includes(e.status)) errors.push(`${where}: status must be one of ${STATUSES.join(", ")}`);
    if (!Array.isArray(e.tags) || !e.tags.every(nonEmpty)) errors.push(`${where}: tags must be a list of strings`);
    if (e.needs != null && (!Array.isArray(e.needs) || !e.needs.every(nonEmpty))) errors.push(`${where}: needs must be a list of strings`);
    const m = e.media ?? {};
    if (typeof m !== "object") errors.push(`${where}: media must be an object`);
    for (const k of ["mp4", "webm", "poster", "fallback"]) {
      if (m[k] != null && !isRelativePath(m[k])) errors.push(`${where}: media.${k} must be a site-relative path`);
    }
    if (m.motif != null && !MOTIFS.includes(m.motif)) errors.push(`${where}: media.motif must be one of ${MOTIFS.join(", ")}`);
    if (e.status === "live") {
      if (!nonEmpty(e.href) || !isRelativePath(e.href)) errors.push(`${where}: a live experience needs a site-relative href`);
      if (!m.poster && !m.fallback) errors.push(`${where}: a live experience needs media.poster or media.fallback`);
    } else if (e.href != null) {
      errors.push(`${where}: only live experiences have an href`);
    }
  });
  return errors;
}

const nonEmpty = (s) => typeof s === "string" && s.trim().length > 0;
/** Relative to the site root: no scheme, no leading slash (the site lives under /Games/). */
const isRelativePath = (s) => typeof s === "string" && s.length > 0 && !/^[a-z]+:|^\/|\.\.\//i.test(s);

/**
 * The location hash -> {cat}: "#cat=body-games" -> "body-games"; "#cat=all",
 * "#experiences" or anything else -> null (all categories). Also accepts
 * "#experiences&cat=x".
 */
export function parseHash(hash) {
  const raw = String(hash ?? "").replace(/^#/, "");
  for (const part of raw.split("&")) {
    const [key, value = ""] = part.split("=");
    if (key === "cat") {
      let cat;
      try {
        cat = decodeURIComponent(value).trim().toLowerCase();
      } catch {
        return { cat: null };
      }
      return { cat: cat && cat !== "all" ? cat : null };
    }
  }
  return { cat: null };
}

/** A category id (or null for all) -> its hash. */
export function formatHash(cat) {
  return `#cat=${cat ? encodeURIComponent(cat) : "all"}`;
}

/** A requested category if the catalog has it (with experiences), else null. */
export function resolveCategory(catalog, cat) {
  if (!cat) return null;
  return menuCategories(catalog).some((c) => c.id === cat) ? cat : null;
}

/** Live before soon, otherwise catalog order. */
export function sortExperiences(list) {
  return list
    .map((e, i) => [e, i])
    .sort((a, b) => (a[0].status === "live" ? 0 : 1) - (b[0].status === "live" ? 0 : 1) || a[1] - b[1])
    .map(([e]) => e);
}

/** The experiences shown for a category (null = all), sorted. */
export function filterExperiences(catalog, cat) {
  return sortExperiences(catalog.experiences.filter((e) => !cat || e.category === cat));
}

/** {total, live, byCategory: {id: n}} */
export function countByCategory(catalog) {
  const byCategory = Object.fromEntries(catalog.categories.map((c) => [c.id, 0]));
  let live = 0;
  for (const e of catalog.experiences) {
    if (e.category in byCategory) byCategory[e.category] += 1;
    if (e.status === "live") live += 1;
  }
  return { total: catalog.experiences.length, live, byCategory };
}

/**
 * The menu: categories that have at least one experience, in catalog order,
 * each with {id, title, blurb, count, live} (live = how many are playable).
 */
export function menuCategories(catalog) {
  return catalog.categories
    .map((c) => {
      const items = catalog.experiences.filter((e) => e.category === c.id);
      return { ...c, count: items.length, live: items.filter((e) => e.status === "live").length };
    })
    .filter((c) => c.count > 0);
}

/** What to render: [{category, items}] for one category or all of them. */
export function groupsFor(catalog, cat) {
  return menuCategories(catalog)
    .filter((c) => !cat || c.id === cat)
    .map((category) => ({ category, items: filterExperiences(catalog, category.id) }));
}

/** Video sources in preference order: [{src, type}]. */
export function videoSources(media) {
  const out = [];
  if (media?.webm) out.push({ src: media.webm, type: "video/webm" });
  if (media?.mp4) out.push({ src: media.mp4, type: "video/mp4" });
  return out;
}

/** Deterministic 0..1 values from a string (placeholder art never flickers between visits). */
export function seededRandom(text) {
  let h = 2166136261;
  for (const ch of String(text)) h = Math.imul(h ^ ch.codePointAt(0), 16777619);
  return () => {
    h = Math.imul(h ^ (h >>> 15), 2246822507);
    h = Math.imul(h ^ (h >>> 13), 3266489909);
    h ^= h >>> 16;
    return (h >>> 0) / 4294967296;
  };
}

/** Two-digit counts for mono labels: 3 -> "03". */
export const pad2 = (n) => String(n).padStart(2, "0");

// --- browser ----------------------------------------------------------------

const SVGNS = "http://www.w3.org/2000/svg";

function el(tag, attrs = {}, ...children) {
  const node = tag.startsWith("svg:") ? document.createElementNS(SVGNS, tag.slice(4)) : document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") node.setAttribute("class", v);
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c != null) node.append(c);
  return node;
}

const NEEDS_LABEL = { webcam: "Webcam" };

function renderMenu(nav, catalog, active) {
  const list = nav.querySelector("[data-menu-list]");
  const counts = countByCategory(catalog);
  const items = [{ id: null, title: "All", count: counts.total, live: counts.live }, ...menuCategories(catalog)];
  list.replaceChildren(
    ...items.map((c) =>
      el("li", {},
        el("a", {
          href: formatHash(c.id),
          class: "chip-link",
          "data-cat": c.id ?? "all",
          "aria-current": (c.id ?? null) === active ? "true" : null,
        },
        c.live > 0 ? el("span", { class: "live-dot", "aria-hidden": "true" }) : null,
        el("span", { class: "chip-title", text: c.title }),
        el("span", { class: "chip-count", text: pad2(c.count), "aria-label": `${c.count} experiences` }))),
    ),
  );
  const current = list.querySelector('[aria-current="true"]');
  // Keep the active chip visible when the menu scrolls sideways (phones, many categories).
  if (current && list.scrollWidth > list.clientWidth) {
    const left = current.offsetLeft - (list.clientWidth - current.offsetWidth) / 2;
    list.scrollTo({ left: Math.max(0, left), behavior: "smooth" });
  }
}

function motifSvg(kind, seed) {
  const rnd = seededRandom(seed);
  const svg = el("svg:svg", { viewBox: "0 0 320 180", class: "motif", "aria-hidden": "true", preserveAspectRatio: "xMidYMid slice" });
  if (kind === "mask") {
    // A blob mask with polygon vertices, the way a segmentation overlay looks.
    const cx = 178 + (rnd() - 0.5) * 30, cy = 100, n = 14, pts = [];
    for (let i = 0; i < n; i++) {
      const a = (i / n) * Math.PI * 2;
      const r = 40 + rnd() * 20;
      pts.push([cx + Math.cos(a) * r * 1.25, cy + Math.sin(a) * r * 0.85]);
    }
    const d = pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join("") + "Z";
    svg.append(el("svg:path", { d, class: "m-fill" }), el("svg:path", { d, class: "m-line" }));
    for (const [x, y] of pts) svg.append(el("svg:rect", { x: x - 2.5, y: y - 2.5, width: 5, height: 5, class: "m-vertex" }));
  } else if (kind === "graph") {
    // A small pipeline: camera -> model -> reason -> output.
    const labels = ["camera", "detect", "reason", "act"];
    const ys = [70, 110, 70, 110];
    labels.forEach((t, i) => {
      const x = 22 + i * 74;
      if (i) svg.append(el("svg:path", { d: `M${x - 18},${ys[i - 1] + 14} C${x - 6},${ys[i - 1] + 14} ${x - 10},${ys[i] + 14} ${x},${ys[i] + 14}`, class: "m-line" }));
      svg.append(el("svg:rect", { x, y: ys[i], width: 56, height: 28, rx: 3, class: "m-node" }));
      svg.append(el("svg:text", { x: x + 28, y: ys[i] + 18, class: "m-text", "text-anchor": "middle", text: t }));
    });
  } else {
    // Detection boxes with labels.
    for (let i = 0; i < 3; i++) {
      const w = 58 + rnd() * 60, h = 36 + rnd() * 34;
      const x = 24 + i * 96 + rnd() * 16, y = 72 + rnd() * (78 - h);
      svg.append(el("svg:rect", { x, y, width: Math.min(w, 300 - x), height: h, class: "m-box" }));
      svg.append(el("svg:rect", { x, y: y - 13, width: 46, height: 13, class: "m-tag" }));
      svg.append(el("svg:text", { x: x + 4, y: y - 3.5, class: "m-tagtext", text: `obj 0.${Math.floor(70 + rnd() * 29)}` }));
    }
  }
  return svg;
}

function renderCard(e, catalog, touch, reducedMotion) {
  const live = e.status === "live";
  const media = e.media ?? {};
  const mediaBox = el("div", { class: "card-media" });
  if (live) {
    // The game's sprite in a detection box, shown only when there's no poster.
    const showFallback = () => {
      mediaBox.classList.add("no-media"); // no poster: assume no video either
      if (!media.fallback) return;
      mediaBox.prepend(el("div", { class: "fallback-art" },
        el("div", { class: "det" },
          el("img", { src: media.fallback, alt: `${e.title} artwork`, decoding: "async" }),
          el("span", { class: "det-label", "aria-hidden": "true", text: `${media.fallbackLabel ?? "object"} 0.97` }))));
    };
    if (media.poster) {
      const poster = el("img", { class: "poster", src: media.poster, alt: `${e.title} gameplay`, loading: "lazy", decoding: "async" });
      poster.addEventListener("error", () => { poster.remove(); showFallback(); }, { once: true });
      mediaBox.append(poster);
    } else {
      showFallback();
    }
  } else {
    mediaBox.append(motifSvg(media.motif ?? "boxes", e.id));
  }
  mediaBox.append(
    el("span", { class: "corners", "aria-hidden": "true" }),
    el("span", { class: `status status-${e.status}`, text: live ? "Live" : "Coming soon" }),
  );
  const category = catalog.categories.find((c) => c.id === e.category);
  const titleId = `exp-${e.id}`;
  const body = el("div", { class: "card-body" },
    el("p", { class: "card-eyebrow" }, el("span", { text: e.model ?? (live ? category?.title ?? "" : "Concept") })),
    el("h4", { class: "card-title", id: titleId, text: e.title }),
    el("p", { class: "card-summary", text: e.summary }),
    el("ul", { class: "tags", "aria-label": "Tags" }, (e.tags ?? []).map((t) => el("li", { text: t }))),
    el("div", { class: "card-foot" },
      el("span", { class: "needs" },
        (e.needs ?? []).map((n) => el("span", { class: `need need-${n}`, text: NEEDS_LABEL[n] ?? n }))),
      live
        ? el("a", { class: "btn btn-primary btn-sm card-cta", href: e.href, "aria-label": `Play ${e.title}` },
          "Play", el("span", { class: "arrow", "aria-hidden": "true", text: "→" }))
        : el("span", { class: "soon-note", text: "In the works" })));
  const card = el("article", { class: `card card-${e.status}`, "data-id": e.id, "aria-labelledby": titleId }, mediaBox, body);
  if (live && !reducedMotion) wirePreview(card, mediaBox, media, touch);
  return card;
}

let canWebm = null;
/** Muted looping preview: on hover/focus, or while in view on touch screens. */
function wirePreview(card, box, media, touch) {
  const sources = videoSources(media);
  if (!sources.length) return;
  let video = null, failed = false, wanted = false;
  const ensure = () => {
    if (video || failed || box.classList.contains("no-media")) return video;
    if (canWebm == null) canWebm = !!document.createElement("video").canPlayType('video/webm; codecs="vp9"');
    const order = sources.filter((s) => s.type !== "video/webm" || canWebm);
    if (!order.length) { failed = true; return null; }
    let i = 0;
    video = el("video", { class: "preview", muted: true, loop: true, playsinline: true, preload: "none", "aria-hidden": "true", tabindex: "-1" });
    video.muted = true; // the attribute alone doesn't mute in every browser
    video.addEventListener("error", () => {
      i += 1;
      if (i < order.length) { video.src = order[i].src; if (wanted) play(); return; }
      failed = true;
      video.remove();
      video = null;
    });
    video.addEventListener("playing", () => {
      if (!wanted) video.pause(); // left before the first frame arrived
      else box.classList.add("is-playing");
    });
    video.src = order[0].src;
    box.insertBefore(video, box.querySelector(".corners"));
    return video;
  };
  const play = () => {
    wanted = true;
    const v = ensure();
    if (v) v.play().catch(() => {}); // autoplay refusals: keep the poster
  };
  const stop = () => {
    wanted = false;
    box.classList.remove("is-playing");
    if (video) video.pause();
  };
  if (touch) {
    const io = new IntersectionObserver((entries) => {
      for (const en of entries) (en.isIntersecting ? play : stop)();
    }, { threshold: 0.6 });
    io.observe(card);
  } else {
    card.addEventListener("pointerenter", play);
    card.addEventListener("pointerleave", stop);
    card.addEventListener("focusin", play);
    card.addEventListener("focusout", (ev) => { if (!card.contains(ev.relatedTarget)) stop(); });
  }
}

function renderGroup({ category, items }, catalog, env, inRoadmap = false) {
  const soonOnly = category.live === 0;
  return el("section", { class: `group${soonOnly ? " group-soon-only" : ""}`, "aria-labelledby": `cat-${category.id}` },
    el("header", { class: "group-head" },
      el("h3", { id: `cat-${category.id}` },
        category.title,
        el("span", { class: "group-count", text: pad2(category.count) })),
      el("p", { class: "group-blurb", text: category.blurb }),
      soonOnly && !inRoadmap ? el("span", { class: "group-soon", text: "Coming soon" }) : null),
    el("div", { class: "grid" }, items.map((e) => renderCard(e, catalog, env.touch, env.reducedMotion))));
}

function renderCatalog(root, catalog, active, env) {
  const groups = groupsFor(catalog, active);
  const live = groups.filter((g) => g.category.live > 0);
  const soon = groups.filter((g) => g.category.live === 0);
  const nodes = live.map((g) => renderGroup(g, catalog, env));
  if (soon.length) {
    // Categories with nothing playable yet share one "roadmap" block.
    nodes.push(el("div", { class: "roadmap" },
      el("p", { class: "roadmap-head" },
        el("span", { class: "eyebrow", text: "Roadmap" }),
        el("span", { text: "Experiences in the works. Nothing here is playable yet." })),
      soon.map((g) => renderGroup(g, catalog, env, true))));
  }
  root.replaceChildren(...nodes);
}

async function boot() {
  // Dev overrides: ?motion=reduce (the still hero, no previews), ?hero=video (the no-WebGL fallback).
  const params = new URLSearchParams(location.search);
  const env = {
    touch: matchMedia("(hover: none)").matches,
    reducedMotion: matchMedia("(prefers-reduced-motion: reduce)").matches || params.get("motion") === "reduce",
  };
  document.documentElement.classList.add("js");

  // The hero starts on its own; the catalog must never wait for (or break on) three.js.
  const heroCanvas = document.getElementById("hero-canvas");
  if (heroCanvas) {
    import("./hero.js")
      .then((m) => params.get("hero") === "video"
        ? m.heroFallback(document.getElementById("hero"), env)
        : m.startHero(heroCanvas, { reducedMotion: env.reducedMotion }))
      .catch((err) => {
        console.warn("hero:", err);
        document.getElementById("hero")?.classList.add("hero-static");
      });
  }

  const nav = document.getElementById("category-menu");
  const root = document.getElementById("catalog");
  const status = document.getElementById("catalog-status");
  let catalog;
  try {
    const res = await fetch("landing/catalog.json", { cache: "no-cache" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    catalog = await res.json();
  } catch (err) {
    console.error("catalog:", err);
    root.replaceChildren(el("p", { class: "catalog-error", text: "The experiences list didn't load. Reload the page to try again." }));
    return;
  }
  const errors = validateCatalog(catalog);
  if (errors.length) console.warn("catalog.json:\n" + errors.join("\n"));
  // Drop what can't be shown rather than failing the page.
  const catIds = new Set(catalog.categories?.map((c) => c.id));
  catalog.experiences = (catalog.experiences ?? []).filter((e) => e && catIds.has(e.category) && STATUSES.includes(e.status));

  let active = null;
  const apply = (fromUser) => {
    const wanted = parseHash(location.hash).cat;
    active = resolveCategory(catalog, wanted);
    renderMenu(nav, catalog, active);
    renderCatalog(root, catalog, active, env);
    const shown = filterExperiences(catalog, active);
    const name = active ? catalog.categories.find((c) => c.id === active).title : "all categories";
    status.textContent = `Showing ${shown.length} experience${shown.length === 1 ? "" : "s"} in ${name}.`;
    if (fromUser) {
      // Keep the menu where the reader is: bring the top of the list under the sticky bar.
      const section = document.getElementById("experiences");
      const top = root.getBoundingClientRect().top + scrollY - nav.offsetHeight - 24;
      if (scrollY > top || section.getBoundingClientRect().top > innerHeight * 0.6) {
        scrollTo({ top, behavior: env.reducedMotion ? "auto" : "smooth" });
      }
    }
  };
  apply(false);
  if (parseHash(location.hash).cat) {
    // Deep link (#cat=...): land on the list, menu at the top (not wherever a reload left us).
    if ("scrollRestoration" in history) history.scrollRestoration = "manual";
    const land = () => scrollTo({ top: root.getBoundingClientRect().top + scrollY - nav.offsetHeight - 24, behavior: "auto" });
    land();
    document.fonts?.ready.then(land); // web fonts can shift the layout once they arrive
  }
  addEventListener("hashchange", () => apply(true));

  // Footer category links come from the catalog too.
  const footerCats = document.getElementById("footer-categories");
  if (footerCats) {
    footerCats.replaceChildren(...menuCategories(catalog).map((c) =>
      el("li", {}, el("a", { href: formatHash(c.id), text: c.title }))));
  }
}

if (typeof window !== "undefined" && typeof document !== "undefined") {
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
}
