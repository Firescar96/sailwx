// Minimal service worker for the Sailing Weather Dashboard PWA.
//
// This dashboard is inherently live/real-time (fresh observations,
// forecasts, flag status) -- we deliberately do NOT cache /api/* responses
// at all, only the small set of static assets (HTML/CSS/JS/icons) needed
// to make the page installable and load instantly on repeat visits. This
// is "installability + fast static-asset loads," not "offline-first data
// app" -- there is no meaningful offline experience for live weather data,
// so we don't pretend otherwise.
const CACHE_NAME = 'sailwx-static-v1';
const STATIC_ASSETS = [
  '/',
  '/style.css',
  '/app.js',
  '/manifest.json',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(STATIC_ASSETS))
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((names) =>
      Promise.all(names.filter((n) => n !== CACHE_NAME).map((n) => caches.delete(n)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // Never cache API calls -- always hit the network for live data.
  if (url.pathname.startsWith('/api/')) {
    return; // let the browser handle it normally, no caching
  }

  // Static assets: try the network first (so a code update is picked up
  // on next reload), fall back to cache if offline/unreachable.
  event.respondWith(
    fetch(event.request)
      .then((response) => {
        const copy = response.clone();
        caches.open(CACHE_NAME).then((cache) => cache.put(event.request, copy));
        return response;
      })
      .catch(() => caches.match(event.request))
  );
});
