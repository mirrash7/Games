// Download the model once, keep it in Cache Storage, and hand it to the worker.
//
// Two variants (tools/export_web_model.py): fp16 (72 MB) for GPUs with
// shader-f16, fp32 (143 MB) for everything else. Each is split into chunks
// under GitHub's file-size limit. Chunks are cached per variant and hash, so a
// return visit starts in about a second, and a new export replaces the old.

const CACHE_PREFIX = "kp-model-";

export async function loadModel(baseUrl, variant = "fp32", onProgress = () => {}) {
  const res = await fetch(new URL(`manifest_${variant}.json`, baseUrl), { cache: "no-cache" });
  if (!res.ok) throw new Error(`could not load the model manifest (${res.status})`);
  const manifest = await res.json();
  const prefix = `${CACHE_PREFIX}${variant}-`;
  const cacheName = prefix + manifest.sha256.slice(0, 16);
  let cache = null;
  try {
    cache = await caches.open(cacheName);
    for (const key of await caches.keys()) {
      if (key.startsWith(prefix) && key !== cacheName) await caches.delete(key);
    }
  } catch {
    cache = null; // private mode or no Cache Storage: just download
  }

  for (let attempt = 0; attempt < 2; attempt++) {
    const buffer = await download(manifest, baseUrl, cache, onProgress);
    if (await matches(buffer, manifest.sha256)) return { buffer, manifest };
    // A truncated or stale chunk: drop the cache and fetch fresh once.
    if (cache) for (const req of await cache.keys()) await cache.delete(req);
  }
  throw new Error("the model download was corrupted - reload to try again");
}

async function download(manifest, baseUrl, cache, onProgress) {
  const out = new Uint8Array(manifest.bytes);
  let offset = 0;
  let cached = true;
  for (const name of manifest.parts) {
    const url = new URL(name, baseUrl).href;
    let res = cache ? await cache.match(url) : null;
    if (!res) {
      cached = false;
      res = await fetch(url);
      if (!res.ok) throw new Error(`model download failed: ${name} (${res.status})`);
      if (cache) {
        try {
          await cache.put(url, res.clone());
        } catch {
          /* quota: fine, it just won't be cached */
        }
      }
    }
    const reader = res.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (offset + value.length > out.length) throw new Error("model chunk larger than expected");
      out.set(value, offset);
      offset += value.length;
      onProgress(offset / manifest.bytes, cached, manifest.bytes);
    }
  }
  if (offset !== manifest.bytes) throw new Error("model download incomplete");
  return out.buffer;
}

async function matches(buffer, sha) {
  if (!crypto?.subtle) return true; // insecure context: cannot verify, trust it
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", buffer));
  return [...digest].map((b) => b.toString(16).padStart(2, "0")).join("") === sha;
}
