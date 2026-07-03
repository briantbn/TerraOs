/**
 * ════════════════════════════════════════════════════════════════════════
 *  FloodRiskEngine.js
 *  Módulo independiente para GeoSentinel — Índice de Probabilidad de
 *  Inundación (IPI) basado en variables hidrológicas reales.
 * ════════════════════════════════════════════════════════════════════════
 *
 *  Este módulo NO modifica ninguna función existente de GeoSentinel.
 *  Se integra leyendo (cuando existen) funciones globales ya presentes en
 *  la app —como hFetchLluvia() para precipitación— y llamando al mismo
 *  backend GEE que ya usa la app (GEE_SERVER) para elevación y pendiente.
 *
 *  TODAS las variables se calculan con datos reales obtenidos en tiempo de
 *  ejecución. Si un dato no está disponible (por ejemplo el WFS de suelos
 *  cae por CORS/timeout), esa variable queda marcada como "no disponible"
 *  y se EXCLUYE del cálculo (redistribuyendo su peso), en vez de inventar
 *  un valor.
 *
 *  Requiere que Leaflet ya esté cargado (igual que el resto de GeoSentinel)
 *  y que exista una variable global `mapa` (instancia de L.Map), tal como
 *  la usa el resto de la aplicación.
 *
 *  Namespace público: window.FloodRiskEngine
 * ════════════════════════════════════════════════════════════════════════
 */

(function (global) {
  'use strict';

  // ────────────────────────────────────────────────────────────────────
  // 0. CONFIGURACIÓN — pesos y umbrales, fácilmente ajustables a futuro
  // ────────────────────────────────────────────────────────────────────

  const CONFIG = {
    // URL del GeoJSON de cuerpos de agua (6.800 polígonos, Corrientes).
    // Debe estar accesible por fetch() relativo a la página, o cambiar
    // a una URL absoluta si se sirve desde otro lado.
    GEOJSON_URL: 'agua_corrientes.geojson',

    // Backend GEE ya usado por GeoSentinel (definido en el archivo principal
    // como GEE_SERVER). Si no se encuentra la variable global, se usa este
    // valor como respaldo.
    GEE_SERVER_FALLBACK: 'https://terraos-1.onrender.com',

    // ── Pesos del modelo multicriterio (deben sumar 1.0) ──────────────
    // Arquitectura pensada para poder tunear esto sin tocar el resto del
    // código: basta con editar estos números.
    PESOS: {
      elevacion: 0.30,
      pendiente: 0.20,
      distancia_agua: 0.20,
      superficie_humedal: 0.10,
      tipo_cuerpo_agua: 0.10,
      precipitacion: 0.10,
    },

    // Modificador de suelo: no forma parte de los pesos fijos de arriba
    // (el pliego original no lo incluye en el 100%), se aplica como
    // ajuste fino de +/- N puntos sobre el índice ya calculado, sólo si
    // hay dato real de textura disponible.
    MODIFICADOR_SUELO_MAX: 5, // puntos de IPI (± sobre el índice final)

    // ── Clasificación de distancia al cuerpo de agua (metros → score 0-100) ──
    ESCALA_DISTANCIA: [
      { max: 100, score: 100, label: 'Muy alto' },
      { max: 500, score: 75, label: 'Alto' },
      { max: 1000, score: 50, label: 'Medio' },
      { max: 3000, score: 25, label: 'Bajo' },
      { max: Infinity, score: 5, label: 'Muy bajo' },
    ],

    // ── Clasificación de superficie del cuerpo de agua (ha → score 0-100) ──
    ESCALA_SUPERFICIE: [
      { max: 10, score: 20 },
      { max: 100, score: 45 },
      { max: 1000, score: 70 },
      { max: Infinity, score: 100 },
    ],

    // ── Peso relativo por tipo de cuerpo de agua (0-100) ──
    // Bañados y esteros mantienen humedad por más tiempo → mayor riesgo.
    PESO_TIPO: {
      'bañado': 100,
      'estero': 95,
      'carrizal': 80,
      'cañada': 70,
      'valle aluvial': 65,
      'laguna': 60,
      'arroyo': 55,
    },
    PESO_TIPO_DEFAULT: 50, // tipo desconocido / no listado

    // ── Umbrales para relativizar elevación respecto del entorno (metros) ──
    // Se compara la elevación del punto contra el promedio de 4 puntos
    // vecinos (a ~200 m N/E/S/O). Diferencia <= -2m → score 100 (muy bajo
    // respecto del entorno = mayor riesgo). Diferencia >= +2m → score 0.
    ELEVACION_REL_RANGO_M: 2,
    ELEVACION_MUESTREO_RADIO_M: 200,

    // ── Umbral de pendiente (%) donde el score de pendiente llega a 0 ──
    PENDIENTE_MAX_PCT: 8,

    // ── Umbral de lluvia acumulada (mm en 7 días) donde el score llega a 100 ──
    LLUVIA_MAX_MM: 150,

    // ── Clasificación final del IPI ──
    CLASIFICACION: [
      { max: 20, nivel: 'Muy Bajo', color: '#16a34a' },   // Verde
      { max: 40, nivel: 'Bajo', color: '#ca8a04' },        // Amarillo
      { max: 60, nivel: 'Medio', color: '#f97316' },       // Naranja
      { max: 80, nivel: 'Alto', color: '#dc2626' },        // Rojo
      { max: 101, nivel: 'Muy Alto', color: '#1e3a8a' },   // Azul oscuro
    ],
  };

  // ────────────────────────────────────────────────────────────────────
  // 1. ESTADO INTERNO DEL MÓDULO
  // ────────────────────────────────────────────────────────────────────

  const state = {
    geojson: null,        // FeatureCollection cargado
    index: null,           // índice espacial en grilla { cellSizeDeg, cells:Map, bbox }
    cargando: null,        // Promise en curso (evita cargas duplicadas)
    inspectorActivo: false,
    marcadorActual: null,
    circuloActual: null,
    _cacheTopo: new Map(), // cache de {lat,lon} -> {elevacion,pendiente} redondeado a 5 decimales
  };

  function geeServer() {
    return (typeof global.GEE_SERVER === 'string' && global.GEE_SERVER) || CONFIG.GEE_SERVER_FALLBACK;
  }

  // ────────────────────────────────────────────────────────────────────
  // 2. GEOMETRÍA — utilidades de distancia sin dependencias externas
  // ────────────────────────────────────────────────────────────────────

  const R_TIERRA_M = 6371000;

  /** Convierte una diferencia de lat/lon (en grados) a metros locales,
   *  usando una proyección equirectangular centrada en latRef.
   *  Válido y preciso para distancias cortas (hasta ~20-30 km), que es
   *  el rango en el que nos interesa medir distancia a cuerpos de agua. */
  function degToLocalXY(lat, lon, latRef, lonRef) {
    const radLat = (latRef * Math.PI) / 180;
    const x = (lon - lonRef) * Math.cos(radLat) * (Math.PI / 180) * R_TIERRA_M;
    const y = (lat - latRef) * (Math.PI / 180) * R_TIERRA_M;
    return [x, y];
  }

  /** Distancia (m) de un punto a un segmento, en coordenadas locales XY. */
  function distPuntoSegmentoXY(px, py, ax, ay, bx, by) {
    const dx = bx - ax, dy = by - ay;
    const lenSq = dx * dx + dy * dy;
    let t = lenSq === 0 ? 0 : ((px - ax) * dx + (py - ay) * dy) / lenSq;
    t = Math.max(0, Math.min(1, t));
    const cx = ax + t * dx, cy = ay + t * dy;
    return Math.hypot(px - cx, py - cy);
  }

  /** Ray casting: ¿el punto [lon,lat] está dentro del anillo (array de [lon,lat])? */
  function puntoEnAnillo(lon, lat, anillo) {
    let dentro = false;
    for (let i = 0, j = anillo.length - 1; i < anillo.length; j = i++) {
      const xi = anillo[i][0], yi = anillo[i][1];
      const xj = anillo[j][0], yj = anillo[j][1];
      const interseca = (yi > lat) !== (yj > lat) &&
        lon < ((xj - xi) * (lat - yi)) / (yj - yi) + xi;
      if (interseca) dentro = !dentro;
    }
    return dentro;
  }

  /**
   * Distancia real (metros) desde un punto [lat,lon] al BORDE de una
   * geometría Polygon/MultiPolygon. Si el punto está dentro del polígono
   * (y fuera de sus huecos), la distancia es 0.
   */
  function distanciaAPoligono(lat, lon, geometry) {
    const poligonos = geometry.type === 'Polygon' ? [geometry.coordinates]
      : geometry.type === 'MultiPolygon' ? geometry.coordinates
      : null;
    if (!poligonos) return Infinity;

    let minDist = Infinity;

    for (const anillos of poligonos) {
      const [exterior, ...huecos] = anillos;

      const dentroExterior = puntoEnAnillo(lon, lat, exterior);
      const dentroHueco = huecos.some(h => puntoEnAnillo(lon, lat, h));
      if (dentroExterior && !dentroHueco) return 0; // está adentro del cuerpo de agua

      // Si no está adentro, medimos distancia al borde más cercano
      // (anillo exterior + huecos, por si el punto está dentro de un hueco).
      for (const anillo of [exterior, ...huecos]) {
        for (let i = 0; i < anillo.length - 1; i++) {
          const [lonA, latA] = anillo[i];
          const [lonB, latB] = anillo[i + 1];
          const [ax, ay] = degToLocalXY(latA, lonA, lat, lon);
          const [bx, by] = degToLocalXY(latB, lonB, lat, lon);
          const d = distPuntoSegmentoXY(0, 0, ax, ay, bx, by);
          if (d < minDist) minDist = d;
        }
      }
    }
    return minDist;
  }

  function bboxDeGeometria(geometry) {
    let minLon = Infinity, minLat = Infinity, maxLon = -Infinity, maxLat = -Infinity;
    const poligonos = geometry.type === 'Polygon' ? [geometry.coordinates]
      : geometry.type === 'MultiPolygon' ? geometry.coordinates : [];
    for (const anillos of poligonos) {
      for (const anillo of anillos) {
        for (const [lon, lat] of anillo) {
          if (lon < minLon) minLon = lon;
          if (lon > maxLon) maxLon = lon;
          if (lat < minLat) minLat = lat;
          if (lat > maxLat) maxLat = lat;
        }
      }
    }
    return [minLon, minLat, maxLon, maxLat];
  }

  // ────────────────────────────────────────────────────────────────────
  // 3. ÍNDICE ESPACIAL EN GRILLA (equivalente funcional a RBush/Flatbush)
  // ────────────────────────────────────────────────────────────────────
  //
  //  Se divide el área total en celdas de tamaño fijo (grados). Cada
  //  feature se registra en todas las celdas que su bounding box toca.
  //  Al consultar, se hace una búsqueda "en espiral" desde la celda del
  //  punto, expandiendo el radio de celdas hasta tener candidatos y
  //  confirmar que ninguna celda más lejana podría contener algo más
  //  cercano. Esto evita recorrer los ~6.800 polígonos en cada click.
  // ────────────────────────────────────────────────────────────────────

  const CELL_SIZE_DEG = 0.03; // ~3.3 km en latitud, tamaño de celda de la grilla

  function construirIndiceEspacial(geojson) {
    const cells = new Map(); // key "cx,cy" -> array de índices de features
    const feats = geojson.features;
    const bboxCache = new Array(feats.length);

    function cellKey(cx, cy) { return cx + ',' + cy; }

    for (let i = 0; i < feats.length; i++) {
      const bbox = bboxDeGeometria(feats[i].geometry);
      bboxCache[i] = bbox;
      const [minLon, minLat, maxLon, maxLat] = bbox;
      const cx0 = Math.floor(minLon / CELL_SIZE_DEG);
      const cx1 = Math.floor(maxLon / CELL_SIZE_DEG);
      const cy0 = Math.floor(minLat / CELL_SIZE_DEG);
      const cy1 = Math.floor(maxLat / CELL_SIZE_DEG);
      for (let cx = cx0; cx <= cx1; cx++) {
        for (let cy = cy0; cy <= cy1; cy++) {
          const key = cellKey(cx, cy);
          if (!cells.has(key)) cells.set(key, []);
          cells.get(key).push(i);
        }
      }
    }

    return { cells, bboxCache, cellSizeDeg: CELL_SIZE_DEG, feats };
  }

  /**
   * Busca el cuerpo de agua más cercano a [lat,lon] usando el índice en
   * grilla, sin recorrer todos los polígonos. Devuelve la geometría real
   * más cercana (no el centroide).
   */
  function buscarCuerpoDeAguaMasCercanoInterno(lat, lon, idx) {
    const cx = Math.floor(lon / idx.cellSizeDeg);
    const cy = Math.floor(lat / idx.cellSizeDeg);
    const cellSizeM = idx.cellSizeDeg * 111000; // aprox metros por celda

    let mejor = null;
    let mejorDist = Infinity;
    const vistos = new Set();
    const MAX_RADIO_CELDAS = 40; // ~130 km de margen máximo de búsqueda

    for (let radio = 0; radio <= MAX_RADIO_CELDAS; radio++) {
      // Cota inferior de distancia que podría aportar el anillo actual:
      // si ya tenemos un candidato más cerca que esto, no hace falta seguir.
      const cotaInferior = (radio - 1) * cellSizeM;
      if (mejor !== null && cotaInferior > mejorDist) break;

      let huboCandidatosEnEsteRadio = false;

      for (let dx = -radio; dx <= radio; dx++) {
        for (let dy = -radio; dy <= radio; dy++) {
          // Sólo el "anillo" exterior del cuadrado (ya visitamos los internos antes)
          if (Math.max(Math.abs(dx), Math.abs(dy)) !== radio) continue;
          const key = (cx + dx) + ',' + (cy + dy);
          const bucket = idx.cells.get(key);
          if (!bucket) continue;
          for (const featIdx of bucket) {
            if (vistos.has(featIdx)) continue;
            vistos.add(featIdx);
            huboCandidatosEnEsteRadio = true;
            const feat = idx.feats[featIdx];
            const d = distanciaAPoligono(lat, lon, feat.geometry);
            if (d < mejorDist) {
              mejorDist = d;
              mejor = feat;
              if (d === 0) return { feature: feat, distancia_m: 0 };
            }
          }
        }
      }
      // Si no hay nada en el radio 0 y tampoco en varios radios siguientes,
      // seguimos ampliando igual (el bucle superior ya lo permite hasta MAX_RADIO_CELDAS).
      void huboCandidatosEnEsteRadio;
    }

    if (!mejor) return null;
    return { feature: mejor, distancia_m: Math.round(mejorDist) };
  }

  // ────────────────────────────────────────────────────────────────────
  // 4. CARGA DEL GEOJSON (perezosa, una sola vez)
  // ────────────────────────────────────────────────────────────────────

  async function asegurarIndiceCargado() {
    if (state.index) return state.index;
    if (state.cargando) return state.cargando;

    state.cargando = (async () => {
      const r = await fetch(CONFIG.GEOJSON_URL);
      if (!r.ok) throw new Error('No se pudo cargar el GeoJSON de cuerpos de agua (' + r.status + ')');
      const geojson = await r.json();
      state.geojson = geojson;
      state.index = construirIndiceEspacial(geojson);
      return state.index;
    })();

    try {
      return await state.cargando;
    } finally {
      state.cargando = null;
    }
  }

  // ────────────────────────────────────────────────────────────────────
  // 5. VARIABLES DEL MODELO — funciones requeridas por la especificación
  // ────────────────────────────────────────────────────────────────────

  /**
   * obtenerElevacion(lat, lon)
   * Consulta el DEM (SRTM 30m) ya expuesto por el backend GEE de
   * GeoSentinel (mismo endpoint que usa el inspector de punto del DEM:
   * /inundacion_punto). Devuelve metros sobre el nivel del mar.
   */
  async function obtenerDatosTopograficos(lat, lon) {
    const key = lat.toFixed(5) + ',' + lon.toFixed(5);
    if (state._cacheTopo.has(key)) return state._cacheTopo.get(key);

    const url = `${geeServer()}/inundacion_punto?lat=${lat}&lon=${lon}`;
    const r = await fetch(url);
    const j = await r.json();
    if (!r.ok || j.ok === false) {
      throw new Error(j.error || 'No se pudo obtener elevación/pendiente desde GEE');
    }
    const datos = {
      elevacion: typeof j.elevacion === 'number' ? j.elevacion : null,
      pendiente_deg: typeof j.pendiente === 'number' ? j.pendiente : null,
    };
    state._cacheTopo.set(key, datos);
    return datos;
  }

  async function obtenerElevacion(lat, lon) {
    const d = await obtenerDatosTopograficos(lat, lon);
    return d.elevacion;
  }

  async function calcularPendiente(lat, lon) {
    const d = await obtenerDatosTopograficos(lat, lon);
    if (d.pendiente_deg == null) return null;
    // Convertimos grados de pendiente a % de pendiente (estándar en hidrología)
    return Math.tan((d.pendiente_deg * Math.PI) / 180) * 100;
  }

  /**
   * Elevación RELATIVA al entorno: compara la elevación del punto contra
   * el promedio de 4 puntos vecinos a ~200 m (N/E/S/O). Un valor negativo
   * significa que el punto está más bajo que su entorno (mayor riesgo).
   */
  async function calcularElevacionRelativa(lat, lon) {
    const dLat = CONFIG.ELEVACION_MUESTREO_RADIO_M / 111000;
    const dLon = CONFIG.ELEVACION_MUESTREO_RADIO_M / (111000 * Math.cos((lat * Math.PI) / 180));

    const vecinos = [
      [lat + dLat, lon],
      [lat - dLat, lon],
      [lat, lon + dLon],
      [lat, lon - dLon],
    ];

    const [centro, ...alrededor] = await Promise.all([
      obtenerDatosTopograficos(lat, lon),
      ...vecinos.map(([la, lo]) => obtenerDatosTopograficos(la, lo).catch(() => null)),
    ]);

    const valoresVecinos = alrededor.filter(v => v && v.elevacion != null).map(v => v.elevacion);
    if (centro.elevacion == null || valoresVecinos.length === 0) {
      return { relativa: null, elevacion: centro.elevacion };
    }
    const promedioVecinos = valoresVecinos.reduce((a, b) => a + b, 0) / valoresVecinos.length;
    return { relativa: centro.elevacion - promedioVecinos, elevacion: centro.elevacion };
  }

  /**
   * buscarCuerpoDeAguaMasCercano(lat, lon)
   * Usa el índice espacial en grilla para encontrar, sin recorrer los
   * 6.800 polígonos, el cuerpo de agua más cercano y la distancia real
   * al borde de su geometría (no al centroide).
   */
  async function buscarCuerpoDeAguaMasCercano(lat, lon) {
    const idx = await asegurarIndiceCargado();
    const resultado = buscarCuerpoDeAguaMasCercanoInterno(lat, lon, idx);
    if (!resultado) return null;
    const p = resultado.feature.properties || {};
    return {
      nombre: p.nombre || 'Sin nombre',
      tipo: (p.tipo || 'Desconocido'),
      superficie_ha: typeof p.superficie_ha === 'number' ? p.superficie_ha : null,
      distancia_m: resultado.distancia_m,
    };
  }

  /**
   * obtenerPrecipitacion(lat, lon)
   * Reutiliza hFetchLluvia() si GeoSentinel ya la tiene cargada en la
   * página (misma fuente que usa el índice hídrico existente: Open-Meteo,
   * lluvia acumulada real de 7 días). Si no está disponible, hace un
   * fallback con la misma API pública.
   */
  async function obtenerPrecipitacion(lat, lon) {
    if (typeof global.hFetchLluvia === 'function') {
      try {
        const d = await global.hFetchLluvia(lat, lon);
        return { lluvia7d_mm: d.lluvia7d ?? null, disponible: d.lluvia7d != null };
      } catch (e) { /* cae al fallback de abajo */ }
    }
    try {
      const url = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}` +
        `&daily=precipitation_sum&past_days=7&forecast_days=1&timezone=America%2FArgentina%2FBuenos_Aires`;
      const r = await fetch(url);
      const d = await r.json();
      const diaria = d.daily?.precipitation_sum || [];
      const lluvia7d = diaria.slice(0, 7).reduce((a, v) => a + (v || 0), 0);
      return { lluvia7d_mm: lluvia7d, disponible: true };
    } catch (e) {
      return { lluvia7d_mm: null, disponible: false };
    }
  }

  /**
   * obtenerTipoSuelo(lat, lon)
   * Best-effort: consulta el WFS de suelos de INTA (misma fuente que ya
   * usa el "Inspector de suelos" de GeoSentinel). Si no hay datos
   * disponibles (CORS, timeout, sin cobertura), devuelve
   * disponible:false — NUNCA se inventa una textura.
   */
  async function obtenerTipoSuelo(lat, lon) {
    const d = 0.005; // ~500 m, mismo bbox que usa el inspector de suelos existente
    const bbox = `${lon - d},${lat - d},${lon + d},${lat + d}`;
    const WFS_BASE =
      `https://geo-backend.inta.gob.ar/geoserver/ows?service=WFS&version=1.0.0` +
      `&request=GetFeature&typeName=geonode:suelos_argentina_1_500` +
      `&outputFormat=json&srs=EPSG:4326&maxFeatures=1&BBOX=${bbox},EPSG:4326`;
    const PROXY_URL = `https://api.allorigins.win/raw?url=${encodeURIComponent(WFS_BASE)}`;

    async function intentar(url) {
      const res = await fetch(url, { signal: AbortSignal.timeout(8000) });
      const text = await res.text();
      if (text.trim().startsWith('<')) throw new Error('CORS_HTML');
      return JSON.parse(text);
    }

    let data;
    try {
      data = await intentar(WFS_BASE);
    } catch (e) {
      try {
        data = await intentar(PROXY_URL);
      } catch (e2) {
        return { disponible: false, textura: null };
      }
    }

    const props = data?.features?.[0]?.properties;
    if (!props) return { disponible: false, textura: null };

    // Los campos de textura varían según la capa; buscamos indicios de
    // clase textural en los campos habituales del shapefile de INTA.
    const camposPosibles = ['textura', 'TEXTURA', 'clase_text', 'drenaje', 'DRENAJE'];
    let textoCombinado = camposPosibles.map(c => props[c]).filter(Boolean).join(' ').toLowerCase();

    let textura = 'desconocida';
    if (/arenos/.test(textoCombinado)) textura = 'arenoso';
    else if (/arcillos/.test(textoCombinado)) textura = 'arcilloso';
    else if (/franc/.test(textoCombinado)) textura = 'franco';

    return { disponible: textura !== 'desconocida', textura, crudo: props };
  }

  // ────────────────────────────────────────────────────────────────────
  // 6. FUNCIONES DE PONDERACIÓN (variable física → score 0-100)
  // ────────────────────────────────────────────────────────────────────

  function calcularPesoPorDistancia(distancia_m) {
    if (distancia_m == null) return null;
    const tramo = CONFIG.ESCALA_DISTANCIA.find(t => distancia_m <= t.max);
    return tramo.score;
  }

  function calcularPesoPorSuperficie(superficie_ha) {
    if (superficie_ha == null) return null;
    const tramo = CONFIG.ESCALA_SUPERFICIE.find(t => superficie_ha <= t.max);
    return tramo.score;
  }

  function calcularPesoPorTipo(tipo) {
    if (!tipo) return CONFIG.PESO_TIPO_DEFAULT;
    const key = tipo.trim().toLowerCase();
    return CONFIG.PESO_TIPO[key] ?? CONFIG.PESO_TIPO_DEFAULT;
  }

  function calcularPesoPorElevacionRelativa(relativa_m) {
    if (relativa_m == null) return null;
    const rango = CONFIG.ELEVACION_REL_RANGO_M;
    // -rango (más bajo que el entorno) => 100 · +rango (más alto) => 0
    const t = Math.max(-rango, Math.min(rango, relativa_m));
    return Math.round(((rango - t) / (2 * rango)) * 100);
  }

  function calcularPesoPorPendiente(pendiente_pct) {
    if (pendiente_pct == null) return null;
    const max = CONFIG.PENDIENTE_MAX_PCT;
    const t = Math.max(0, Math.min(max, pendiente_pct));
    return Math.round(((max - t) / max) * 100);
  }

  function calcularPesoPorLluvia(lluvia_mm) {
    if (lluvia_mm == null) return null;
    const t = Math.max(0, Math.min(CONFIG.LLUVIA_MAX_MM, lluvia_mm));
    return Math.round((t / CONFIG.LLUVIA_MAX_MM) * 100);
  }

  // ────────────────────────────────────────────────────────────────────
  // 7. ÍNDICE FINAL Y CLASIFICACIÓN
  // ────────────────────────────────────────────────────────────────────

  /**
   * calcularIndiceFinal(subscores)
   * Combina los subscores (0-100 cada uno, o null si no hay dato) según
   * los pesos de CONFIG.PESOS. Si alguna variable no tiene dato
   * disponible, su peso se redistribuye proporcionalmente entre las
   * variables restantes (en vez de asumir un valor neutro inventado).
   */
  function calcularIndiceFinal(subscores) {
    const pesos = CONFIG.PESOS;
    let sumaPesosDisponibles = 0;
    let sumaPonderada = 0;

    for (const variable of Object.keys(pesos)) {
      const score = subscores[variable];
      if (score == null) continue;
      sumaPesosDisponibles += pesos[variable];
      sumaPonderada += score * pesos[variable];
    }

    if (sumaPesosDisponibles === 0) return null;

    let indice = sumaPonderada / sumaPesosDisponibles; // re-normalizado 0-100

    // Modificador de suelo (best-effort, acotado)
    if (subscores._modificadorSuelo) {
      indice = Math.max(0, Math.min(100, indice + subscores._modificadorSuelo));
    }

    return Math.round(indice * 10) / 10;
  }

  function clasificarIndice(indice) {
    const tramo = CONFIG.CLASIFICACION.find(t => indice < t.max);
    return tramo || CONFIG.CLASIFICACION[CONFIG.CLASIFICACION.length - 1];
  }

  // ────────────────────────────────────────────────────────────────────
  // 8. ORQUESTADOR — analiza un punto de principio a fin
  // ────────────────────────────────────────────────────────────────────

  async function analizarPunto(lat, lon) {
    const [elevRel, cuerpoAgua, pendientePct, lluvia, suelo] = await Promise.all([
      calcularElevacionRelativa(lat, lon).catch(() => ({ relativa: null, elevacion: null })),
      buscarCuerpoDeAguaMasCercano(lat, lon).catch(() => null),
      calcularPendiente(lat, lon).catch(() => null),
      obtenerPrecipitacion(lat, lon).catch(() => ({ lluvia7d_mm: null })),
      obtenerTipoSuelo(lat, lon).catch(() => ({ disponible: false, textura: null })),
    ]);

    const subscores = {
      elevacion: calcularPesoPorElevacionRelativa(elevRel.relativa),
      pendiente: calcularPesoPorPendiente(pendientePct),
      distancia_agua: cuerpoAgua ? calcularPesoPorDistancia(cuerpoAgua.distancia_m) : null,
      superficie_humedal: cuerpoAgua ? calcularPesoPorSuperficie(cuerpoAgua.superficie_ha) : null,
      tipo_cuerpo_agua: cuerpoAgua ? calcularPesoPorTipo(cuerpoAgua.tipo) : null,
      precipitacion: calcularPesoPorLluvia(lluvia.lluvia7d_mm),
    };

    if (suelo.disponible) {
      subscores._modificadorSuelo = suelo.textura === 'arenoso' ? -CONFIG.MODIFICADOR_SUELO_MAX
        : suelo.textura === 'arcilloso' ? CONFIG.MODIFICADOR_SUELO_MAX : 0;
    }

    const indice = calcularIndiceFinal(subscores);
    const clase = indice == null ? null : clasificarIndice(indice);

    return {
      lat, lon,
      indice,
      nivel: clase?.nivel ?? 'Sin datos suficientes',
      color: clase?.color ?? '#9ca3af',
      variables: {
        elevacion_m: elevRel.elevacion,
        elevacion_relativa_m: elevRel.relativa,
        pendiente_pct: pendientePct,
        cuerpo_agua: cuerpoAgua,
        lluvia_7d_mm: lluvia.lluvia7d_mm,
        suelo: suelo.disponible ? suelo.textura : 'sin datos',
      },
      subscores,
    };
  }

  // ────────────────────────────────────────────────────────────────────
  // 9. VISUALIZACIÓN — panel de resultados + color en el mapa
  // ────────────────────────────────────────────────────────────────────

  function mostrarResultados(resultado) {
    const map = global.mapa;
    if (!map || typeof L === 'undefined') return;

    // Marcador/círculo coloreado en el punto analizado
    if (state.circuloActual) { try { map.removeLayer(state.circuloActual); } catch (e) {} }
    state.circuloActual = L.circle([resultado.lat, resultado.lon], {
      radius: 180,
      color: resultado.color,
      weight: 2,
      fillColor: resultado.color,
      fillOpacity: 0.35,
    }).addTo(map);

    const v = resultado.variables;
    const fmt = (val, suf) => (val == null ? '—' : `${typeof val === 'number' ? val.toFixed(1) : val}${suf || ''}`);
    const cuerpo = v.cuerpo_agua;

    const html = `
      <div style="font-family:Inter,sans-serif;min-width:230px">
        <p style="font-size:11px;font-weight:700;color:#727971;margin:0 0 2px;text-transform:uppercase;letter-spacing:.06em">
          Probabilidad de inundación
        </p>
        <p style="font-size:28px;font-weight:800;color:${resultado.color};margin:0 0 2px">
          ${resultado.indice ?? '—'}%
        </p>
        <p style="font-size:13px;font-weight:700;color:${resultado.color};margin:0 0 10px">
          Nivel: ${resultado.nivel}
        </p>
        <div style="border-top:1.5px solid #e8ebe5;padding-top:8px;font-size:11px;color:#424841;
                    display:grid;grid-template-columns:1fr 1fr;gap:6px 10px">
          <div><span style="color:#727971">Elevación</span><br><b>${fmt(v.elevacion_m, ' m')}</b></div>
          <div><span style="color:#727971">Elev. relativa</span><br><b>${fmt(v.elevacion_relativa_m, ' m')}</b></div>
          <div><span style="color:#727971">Pendiente</span><br><b>${fmt(v.pendiente_pct, ' %')}</b></div>
          <div><span style="color:#727971">Distancia al agua</span><br><b>${cuerpo ? cuerpo.distancia_m + ' m' : '—'}</b></div>
          <div><span style="color:#727971">Tipo</span><br><b>${cuerpo ? cuerpo.tipo : '—'}</b></div>
          <div><span style="color:#727971">Superficie</span><br><b>${cuerpo?.superficie_ha != null ? cuerpo.superficie_ha + ' ha' : '—'}</b></div>
          <div><span style="color:#727971">Lluvia acum. (7d)</span><br><b>${fmt(v.lluvia_7d_mm, ' mm')}</b></div>
          <div><span style="color:#727971">Suelo</span><br><b>${v.suelo}</b></div>
        </div>
        ${cuerpo ? `<p style="font-size:10px;color:#9ca3af;margin:8px 0 0">Cuerpo más cercano: ${cuerpo.nombre}</p>` : ''}
        <p style="font-size:9px;color:#9ca3af;margin:4px 0 0">GEE SRTM 30m · Open-Meteo · INTA · Índice calculado: ${resultado.indice ?? '—'}</p>
      </div>`;

    L.popup({ maxWidth: 280 }).setLatLng([resultado.lat, resultado.lon]).setContent(html).openOn(map);
  }

  // ────────────────────────────────────────────────────────────────────
  // 10. HERRAMIENTA DE MAPA — activar/desactivar y manejar el click
  // ────────────────────────────────────────────────────────────────────

  async function onMapClick(lat, lon) {
    const map = global.mapa;
    if (map) map.getContainer().style.cursor = 'wait';
    try {
      const resultado = await analizarPunto(lat, lon);
      mostrarResultados(resultado);
    } catch (e) {
      console.error('[FloodRiskEngine]', e);
      if (map && typeof L !== 'undefined') {
        L.popup().setLatLng([lat, lon])
          .setContent(`<p style="color:#dc2626;font-size:12px">⚠️ ${e.message || 'Error al calcular el IPI'}</p>`)
          .openOn(map);
      }
    } finally {
      if (map) map.getContainer().style.cursor = state.inspectorActivo ? 'crosshair' : '';
    }
  }

  function activarInspector() {
    state.inspectorActivo = !state.inspectorActivo;
    const map = global.mapa;
    if (map) map.getContainer().style.cursor = state.inspectorActivo ? 'crosshair' : '';
    // Precarga el índice espacial en segundo plano apenas se activa la
    // herramienta, para que el primer click sea rápido.
    if (state.inspectorActivo) asegurarIndiceCargado().catch(e => console.error('[FloodRiskEngine] índice espacial:', e));
    return state.inspectorActivo;
  }

  function limpiar() {
    const map = global.mapa;
    if (state.circuloActual && map) { try { map.removeLayer(state.circuloActual); } catch (e) {} }
    state.circuloActual = null;
  }

  // ────────────────────────────────────────────────────────────────────
  // 11. EXPORTACIÓN PÚBLICA
  // ────────────────────────────────────────────────────────────────────

  global.FloodRiskEngine = {
    CONFIG,
    // Estado de la herramienta (lectura, se usa desde onMapaClick)
    get inspectorActivo() { return state.inspectorActivo; },
    // Ciclo de vida de la herramienta
    activarInspector,
    onMapClick,
    limpiar,
    // Funciones del modelo (expuestas para testing / uso independiente)
    obtenerElevacion,
    calcularPendiente,
    calcularElevacionRelativa,
    buscarCuerpoDeAguaMasCercano,
    obtenerPrecipitacion,
    obtenerTipoSuelo,
    calcularPesoPorDistancia,
    calcularPesoPorSuperficie,
    calcularPesoPorTipo,
    calcularPesoPorElevacionRelativa,
    calcularPesoPorPendiente,
    calcularPesoPorLluvia,
    calcularIndiceFinal,
    clasificarIndice,
    analizarPunto,
    mostrarResultados,
    // Índice espacial (útil para diagnósticos)
    asegurarIndiceCargado,
  };

})(window);
