"""
suelos_inta_regional.py
==========================================
Aptitud productiva del suelo, priorizando siempre la mejor escala
disponible antes de caer a una carta más gruesa. Usado por dos endpoints
de app.py:

  - /suelo_mejor_resolucion  -> consultar_mejor_capa(lat, lon)
        Consulta puntual (Inspector de Suelos, Motor de Aptitud Productiva).
  - /suelo_capa_aptitud      -> capa_visual_bbox((west, south, east, north))
        Overlay visual coloreado sobre el área visible del mapa.

Fuente de datos: cartas de INTA en formato GeoJSON, autoalojadas en
Hugging Face (dataset Briant97/GeoSentinel). Se leen una sola vez por
proceso y quedan cacheadas en memoria con un índice espacial (STRtree de
Shapely) para que las consultas siguientes sean instantáneas.

Orden de prioridad (mejor escala primero) — ver SUELO_CAPAS más abajo:
  1) Santa Fe    — carta de detalle 1:50.000
  2) Córdoba     — carta de detalle 1:50.000
  3) Buenos Aires— carta de detalle 1:50.000
  4) NOA         — aptitud regional (zonas grandes, informativo)
  5) Nacional    — 1:1.000.000 (último recurso: cubre todo el país)

Para agregar más provincias/escalas a futuro (1:100.000, 1:250.000,
1:500.000, etc.): alcanza con sumar una entrada a SUELO_CAPAS en el lugar
que le corresponda según su escala. No hace falta tocar app.py ni el resto
de este archivo.
"""
import threading

import requests
from shapely.geometry import shape, Point, box
from shapely.strtree import STRtree

# ──────────────────────────────────────────────────────────────────────────
# CATÁLOGO DE CAPAS (mejor escala primero)
# ──────────────────────────────────────────────────────────────────────────
_HF_BASE = 'https://huggingface.co/datasets/Briant97/GeoSentinel/resolve/main/'

SUELO_CAPAS = [
    {
        'nombre': 'Santa Fe',
        'url': _HF_BASE + 'Provincia%20de%20Santa%20Fe%20(1_50000)%20%E2%80%94%20'
                           'suelos_santa_fe.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        # bbox aproximado (west, south, east, north), con margen generoso
        # a propósito -- ver nota junto a _capa_puede_cubrir() más abajo.
        'bbox': (-63.5, -34.9, -59.3, -27.5),
    },
    {
        'nombre': 'Córdoba',
        'url': _HF_BASE + 'Suelos_detalle_Cordoba.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-65.9, -35.3, -61.5, -29.4),
    },
    {
        'nombre': 'Buenos Aires',
        'url': _HF_BASE + 'Suelos_detalle_Buenos_Aires.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-64.1, -41.5, -56.5, -32.5),
    },
    {
        'nombre': 'NOA (regional)',
        'url': _HF_BASE + 'Suelos_detalle_NOA.geojson?download=true',
        'escala': 'regional (zonas grandes)',
        'confiabilidad': 'media',
        # Noroeste argentino (Jujuy/Salta/Tucumán/Catamarca/Santiago del
        # Estero) -- OJO: esto NO cubre Corrientes (que es NEA, noreste),
        # aunque el nombre "regional" pueda sugerir que sí.
        'bbox': (-70.0, -29.6, -61.5, -20.8),
    },
    {
        'nombre': 'Nacional',
        'url': _HF_BASE + 'Republica%20Argentina%201%20en%201000.geojson?download=true',
        'escala': '1:1.000.000',
        'confiabilidad': 'baja',
        # Sin bbox: cubre todo el país, es el respaldo final -- siempre
        # elegible, nunca se saltea por el chequeo de _capa_puede_cubrir().
        'bbox': None,
    },
]

# Tope de features que devuelve capa_visual_bbox por pedido, para no mandar
# payloads gigantes si el usuario hace zoom-out sobre una provincia entera.
_MAX_FEATURES_BBOX = 3000


def _capa_puede_cubrir_punto(capa, lon, lat):
    """
    Chequeo BARATO (4 comparaciones de números, sin red ni parseo) para
    saltear por completo la descarga/parseo/indexado de una capa que
    geográficamente no puede contener el punto -- ver _cargar_capa() más
    abajo, que es la parte cara (GET a Hugging Face + shapely.shape() por
    cada polígono + construir el STRtree, todo eso queda cacheado en RAM
    PARA SIEMPRE sin límite).

    Antes de este chequeo, cada consulta desde una zona que ninguna capa
    de detalle cubre (ej. Corrientes, que no está en Santa Fe/Córdoba/
    Buenos Aires/NOA) terminaba cargando las 4 capas de detalle COMPLETAS
    igual, solo para descartarlas una por una, antes de llegar a
    'Nacional'. Con provincias grandes a 1:50.000 (Buenos Aires en
    particular), eso son fácilmente cientos de MB en objetos Shapely +
    STRtree cargados en memoria sin ninguna necesidad real.

    bbox=None (capa 'Nacional', cobertura de todo el país) siempre pasa
    el chequeo -- es el respaldo final.

    Los bbox son aproximados y con margen generoso a propósito: es
    preferible cargar una capa de más en un caso límite cerca del borde
    de una provincia, que descartar por error un polígono real que sí
    cubría el punto.
    """
    bbox = capa.get('bbox')
    if bbox is None:
        return True
    west, south, east, north = bbox
    return west <= lon <= east and south <= lat <= north


def _capa_puede_cubrir_bbox(capa, consulta_bbox):
    """Misma idea que _capa_puede_cubrir_punto(), pero para el overlay
    visual (capa_visual_bbox): compara dos rectángulos en vez de un punto
    contra un rectángulo."""
    bbox = capa.get('bbox')
    if bbox is None:
        return True
    c_west, c_south, c_east, c_north = consulta_bbox
    west, south, east, north = bbox
    return not (c_east < west or c_west > east or c_north < south or c_south > north)

# ──────────────────────────────────────────────────────────────────────────
# CACHE EN MEMORIA + ÍNDICE ESPACIAL
# ──────────────────────────────────────────────────────────────────────────
_cache = {}          # nombre_capa -> {'tree', 'geoms', 'props'} | None (falló)
_cache_lock = threading.Lock()


def _cargar_capa(capa):
    nombre = capa['nombre']
    with _cache_lock:
        if nombre in _cache:
            return _cache[nombre]
    datos = None
    try:
        r = requests.get(capa['url'], timeout=120)
        r.raise_for_status()
        geojson = r.json()
        geoms, props = [], []
        for feat in geojson.get('features', []):
            geom = feat.get('geometry')
            if not geom:
                continue
            try:
                g = shape(geom)
                if not g.is_valid:
                    g = g.buffer(0)  # corrige auto-intersecciones menores
            except Exception:
                continue  # geometría puntualmente corrupta: se ignora
            geoms.append(g)
            props.append(feat.get('properties') or {})
        tree = STRtree(geoms) if geoms else None
        datos = {'tree': tree, 'geoms': geoms, 'props': props}
        print(f'✅ [suelos_inta_regional] Carta "{nombre}" cargada: {len(geoms)} polígonos.')
    except Exception as exc:  # noqa: BLE001
        print(f'⚠️ [suelos_inta_regional] No se pudo cargar la carta "{nombre}": {exc}')
    with _cache_lock:
        _cache[nombre] = datos
    return datos


# ──────────────────────────────────────────────────────────────────────────
# NORMALIZACIÓN DE PROPIEDADES
# ──────────────────────────────────────────────────────────────────────────
# Distintas cartas traen distintos nombres de columna:
#   - Cartas regionales (Santa Fe/Córdoba/Buenos Aires/NOA):
#       clase, subclase, cap_uso, drenaje_estimado
#   - Esquema nacional clásico de INTA (por si el archivo "Nacional" lo usa):
#       ind_prod, drenaje_s1, anegab_s1, limit_ppal
# _normalizar() deja todo en el MISMO esquema de salida para que el
# frontend (_aptitudClasificarProps, _aptitudPintarUI) no tenga que saber
# de dónde vino cada polígono.

_INDICE_POR_CLASE = {
    'I': 95, 'II': 80, 'III': 65, 'IV': 50,
    'V': 35, 'VI': 25, 'VII': 10, 'VIII': 0,
}

_LIMITANTE_POR_LETRA = {
    'e': 'Erosión',
    's': 'Suelo (limitaciones edáficas)',
    'c': 'Clima',
    'w': 'Anegamiento / drenaje',
}


def _limitante_desde_subclase(subclase):
    if not subclase:
        return None
    vistas, orden = set(), []
    for ch in str(subclase).lower():
        etiqueta = _LIMITANTE_POR_LETRA.get(ch)
        if etiqueta and etiqueta not in vistas:
            vistas.add(etiqueta)
            orden.append(etiqueta)
    return ' + '.join(orden) if orden else None


def _drenaje_legible(valor):
    if not valor:
        return None
    return str(valor).replace('_', ' ').strip().capitalize()


def _anegabilidad_desde_drenaje(drenaje_crudo):
    d = str(drenaje_crudo or '').lower()
    if any(p in d for p in ('pobre', 'malo', 'deficient')):
        return 'Probable (drenaje deficiente)'
    return None


def _g(props, *claves):
    """Busca cualquiera de `claves` en `props`, probando también
    MAYUSCULA/minúscula/Capitalizada."""
    for k in claves:
        for variante in (k, k.upper(), k.lower(), k.capitalize()):
            if variante in props and props[variante] not in (None, ''):
                return props[variante]
    return None


def _normalizar(props):
    clase = _g(props, 'clase')
    subclase = _g(props, 'subclase')
    cap_uso = _g(props, 'cap_uso', 'capacidad_uso')
    drenaje_crudo = _g(props, 'drenaje_estimado', 'drenaje', 'drenaje_s1')

    indice_prod = _g(props, 'ind_prod', 'indice_prod')
    if indice_prod in (None, ''):
        indice_prod = _INDICE_POR_CLASE.get(str(clase or '').upper())

    limitante = _g(props, 'limit_ppal', 'limitante') or _limitante_desde_subclase(subclase)
    anegabilidad = _g(props, 'anegab_s1', 'anegabilidad') or _anegabilidad_desde_drenaje(drenaje_crudo)

    return {
        'clase': clase,
        'subclase': subclase,
        'capacidad_uso': cap_uso,
        'indice_prod': indice_prod,
        'limitante': limitante,
        'drenaje': _drenaje_legible(drenaje_crudo),
        'anegabilidad': anegabilidad,
    }


def _advertencia_escala(capa):
    if capa['confiabilidad'] == 'alta':
        return None
    return (
        f'Esta zona todavía no tiene carta de detalle 1:50.000: el dato viene '
        f'de "{capa["nombre"]}" (escala {capa["escala"]}), menos preciso.'
    )


# ──────────────────────────────────────────────────────────────────────────
# API PÚBLICA — llamada desde app.py
# ──────────────────────────────────────────────────────────────────────────
def consultar_mejor_capa(lat, lon):
    """Consulta puntual: prueba cada capa en orden de prioridad (mejor
    escala primero) y devuelve el primer polígono que contiene (lat, lon).
    Formato de salida = el que ya espera _aptitudFetchINTA() en el frontend."""
    punto = Point(lon, lat)

    for capa in SUELO_CAPAS:
        if not _capa_puede_cubrir_punto(capa, lon, lat):
            continue  # fuera del bbox de esta capa: ni se descarga ni se carga en memoria
        datos = _cargar_capa(capa)
        if not datos or not datos['tree']:
            continue
        for i in datos['tree'].query(punto):
            geom = datos['geoms'][i]
            if geom.covers(punto):
                props = datos['props'][i]
                norm = _normalizar(props)
                return {
                    'encontrado': True,
                    'fuente': capa['nombre'],
                    'clase': norm['clase'],
                    'subclase': norm['subclase'],
                    'capacidad_uso': norm['capacidad_uso'],
                    'indice_prod': norm['indice_prod'],
                    'limitante': norm['limitante'],
                    'drenaje': norm['drenaje'],
                    'anegabilidad': norm['anegabilidad'],
                    'advertencia_escala': _advertencia_escala(capa),
                    'confiabilidad': capa['confiabilidad'],
                    'propiedades_crudas': props,
                }

    # Ninguna capa (ni siquiera la Nacional) cubre este punto
    return {'encontrado': False, 'fuente': None}


def capa_visual_bbox(bbox):
    """Overlay visual: devuelve un FeatureCollection GeoJSON con los
    polígonos dentro de `bbox` = (west, south, east, north), con las
    propiedades ya normalizadas para que el frontend los pinte de forma
    uniforme sin importar de qué carta salió cada uno.

    Usa SIEMPRE una sola capa por pedido (la de mejor escala que tenga
    algo dentro del bbox), para no mezclar visualmente polígonos de
    distinta escala/confiabilidad en una misma vista."""
    west, south, east, north = bbox
    caja = box(west, south, east, north)

    for capa in SUELO_CAPAS:
        if not _capa_puede_cubrir_bbox(capa, bbox):
            continue  # fuera del bbox de esta capa: ni se descarga ni se carga en memoria
        datos = _cargar_capa(capa)
        if not datos or not datos['tree']:
            continue

        indices = datos['tree'].query(caja)
        if len(indices) == 0:
            continue  # esta capa no tiene nada en este bbox: probamos la siguiente

        features = []
        for i in indices:
            geom = datos['geoms'][i]
            if not geom.intersects(caja):
                continue
            props = datos['props'][i]
            features.append({
                'type': 'Feature',
                'geometry': geom.__geo_interface__,
                'properties': _normalizar(props),
            })
            if len(features) >= _MAX_FEATURES_BBOX:
                break

        if not features:
            continue

        return {
            'type': 'FeatureCollection',
            'fuente': capa['nombre'],
            'confiabilidad': capa['confiabilidad'],
            'advertencia_escala': _advertencia_escala(capa),
            'features': features,
        }

    return {'type': 'FeatureCollection', 'fuente': None, 'features': []}
