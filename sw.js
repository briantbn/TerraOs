// TerraOS Service Worker v1
const CACHE = 'terraos-v1';

const ARCHIVOS = [
  './',
  './index.html',
  './manifest.json',
  'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css',
  'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js',
  'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap',
];

// Instalar: cachear archivos esenciales
self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE).then(cache => {
      return cache.addAll(ARCHIVOS).catch(err => {
        console.warn('Cache parcial:', err);
      });
    })
  );
  self.skipWaiting();
});

// Activar: limpiar caches viejos
self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// Fetch: cache-first para assets, network-first para APIs
self.addEventListener('fetch', e => {
  const url = e.request.url;

  // APIs externas siempre van a la red
  if (
    url.includes('script.google.com') ||
    url.includes('api.open-meteo.com') ||
    url.includes('firms.modaps.eosdis.nasa.gov') ||
    url.includes('power.larc.nasa.gov') ||
    url.includes('arcgisonline.com') ||
    url.includes('tile.openstreetmap.org') ||
    url.includes('opentopomap.org')
  ) {
    return; // dejar pasar sin interceptar
  }

  // Para el resto: cache first, fallback a red
  e.respondWith(
    caches.match(e.request).then(cached => {
      if (cached) return cached;
      return fetch(e.request).then(response => {
        if (response && response.status === 200) {
          const clone = response.clone();
          caches.open(CACHE).then(cache => cache.put(e.request, clone));
        }
        return response;
      }).catch(() => {
        // Sin conexión y sin cache: respuesta vacía
        return new Response('Sin conexión', { status: 503 });
      });
    })
  );
});
