// Asset loading for Snack Attack (port of game/fruitninja/art.py).
//
// Sprites are generated procedurally (tools/generate_fruitninja_snacks.py,
// tools/generate_fruitninja_scene.py) and described by a manifest, so gameplay
// numbers - collision radius, crumb colour, score - travel with the art.
// Names keep the original fruit theme: a "fruit" is any snack, halfA/halfB
// are the two pieces it breaks into, `juice` is its crumb colour, `bomb` is
// the bag of trash.
//
// Nothing here touches the DOM until loadArt() is called.

import { loadImage, loadJSON } from "../../core/gfx.js";

export const ASSET_URL = new URL("../../../assets/snack/", import.meta.url);

const GENERATE_HINT =
  "Run:\n  uv run python tools/generate_fruitninja_snacks.py\n  uv run python tools/generate_fruitninja_scene.py";

let _cache = null;
let _loading = null;

/** The art loaded by a previous loadArt(), or null. */
export function cachedArt() {
  return _cache;
}

/** Manifest colours are OpenCV BGR; the web build works in RGB. */
const bgrToRgb = (c) => [Number(c[2]), Number(c[1]), Number(c[0])];

function image(name) {
  return loadImage(new URL(name, ASSET_URL).href).catch(() => {
    throw new Error(`missing asset ${name}. ${GENERATE_HINT}`);
  });
}

async function load() {
  let spec;
  try {
    spec = await loadJSON(new URL("manifest.json", ASSET_URL).href);
  } catch {
    throw new Error(`missing assets/snack/manifest.json. ${GENERATE_HINT}`);
  }
  const fruits = await Promise.all(spec.fruits.map(async (f) => {
    const [whole, halfA, halfB] = await Promise.all([
      image(`fruit_${f.name}_whole.png`),
      image(`fruit_${f.name}_half_a.png`),
      image(`fruit_${f.name}_half_b.png`),
    ]);
    return {
      name: f.name,
      whole, halfA, halfB,
      juice: bgrToRgb(f.juice), // RGB, for crumbs and popups
      radius: Number(f.radius), // collision radius, px at native sprite size
      score: Number(f.score ?? 1),
    };
  }));
  const names = ["bomb", "background", "splat", "flash", "life_full", "life_lost",
    "raccoon_idle", "raccoon_chomp"];
  const imgs = await Promise.all(names.map((n) => image(`${n}.png`)));
  const by = Object.fromEntries(names.map((n, i) => [n, imgs[i]]));
  return {
    fruits,
    bomb: by.bomb, // the bag of trash
    background: by.background,
    splat: by.splat, // white, tinted per snack at init
    flash: by.flash,
    lifeFull: by.life_full,
    lifeLost: by.life_lost,
    cursorIdle: by.raccoon_idle, // raccoon head, mouth closed
    cursorChomp: by.raccoon_chomp, // mouth open (eating)
  };
}

/** Load (once) and cache every sprite plus the manifest. */
export async function loadArt({ reload = false } = {}) {
  if (_cache && !reload) return _cache;
  if (!_loading || reload) {
    _loading = load().then((art) => (_cache = art)).finally(() => { _loading = null; });
  }
  return _loading;
}
