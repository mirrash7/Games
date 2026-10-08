// Cross-origin isolation for static hosting (GitHub Pages cannot set headers).
//
// onnxruntime-web needs SharedArrayBuffer to run the model on more than one
// CPU thread, and browsers only allow that on cross-origin-isolated pages
// (COOP + COEP). This service worker adds those headers to every response.
// The GPU path (WebGPU) works without it; it matters for the CPU fallback.

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.cache === "only-if-cached" && req.mode !== "same-origin") return;
  event.respondWith(
    fetch(req)
      .then((res) => {
        if (res.status === 0) return res; // opaque: cannot touch
        const headers = new Headers(res.headers);
        headers.set("Cross-Origin-Embedder-Policy", "require-corp");
        headers.set("Cross-Origin-Opener-Policy", "same-origin");
        headers.set("Cross-Origin-Resource-Policy", "cross-origin");
        return new Response(res.body, { status: res.status, statusText: res.statusText, headers });
      })
      .catch((err) => {
        console.error("[coi-sw]", err);
        return Response.error();
      }),
  );
});
