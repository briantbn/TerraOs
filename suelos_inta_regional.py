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
import json
import os
import threading
from collections import OrderedDict

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
    },
    {
        'nombre': 'Córdoba',
        'url': _HF_BASE + 'Suelos_detalle_Cordoba.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
    },
    {
        'nombre': 'Buenos Aires',
        'url': _HF_BASE + 'Suelos_detalle_Buenos_Aires.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
    },
    {
        'nombre': 'NOA (regional)',
        'url': _HF_BASE + 'Suelos_detalle_NOA.geojson?download=true',
        'escala': 'regional (zonas grandes)',
        'confiabilidad': 'media',
    },
    {
        'nombre': 'Nacional',
        'url': _HF_BASE + 'Republica%20Argentina%201%20en%201000.geojson?download=true',
        'escala': '1:1.000.000',
        'confiabilidad': 'baja',
    },
]

# Tope de features que devuelve capa_visual_bbox por pedido, para no mandar
# payloads gigantes si el usuario hace zoom-out sobre una provincia entera.
_MAX_FEATURES_BBOX = 3000

# Caché en DISCO del GeoJSON crudo de cada carta (mismo espíritu que
# CACHE_DIR en local_flood_engine.py). Antes, cada reinicio del proceso
# obligaba a re-descargar TODO de Hugging Face desde cero en el primer
# pedido que llegara — si esa descarga tardaba más que el timeout del
# worker de gunicorn en Render (default 30s), el proceso moría a mitad
# de la descarga y el usuario veía 502 / "Failed to fetch". Con esto,
# después de la primera vez que una carta se descargó bien, los
# reinicios siguientes la leen del disco (rápido) en vez de volver a
# pedirla por HTTP.
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache_suelos_inta')
os.makedirs(CACHE_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────
# CACHE EN MEMORIA + ÍNDICE ESPACIAL
# ──────────────────────────────────────────────────────────────────────────
# CON LÍMITE (LRU): cada carta parseada (geometrías shapely + índice
# STRtree de una provincia entera) puede pesar bastante — guardar las 5
# a la vez en RAM para siempre fue lo que hizo explotar la memoria en
# Render. Ahora solo se mantienen las últimas _CACHE_MEMORIA_MAX cartas
# usadas; el resto se relee del caché en DISCO cuando hace falta de
# nuevo (rápido: no depende de la red, solo re-parsear el GeoJSON local).
_CACHE_MEMORIA_MAX = 2
_cache = OrderedDict()   # nombre_capa -> {'tree', 'geoms', 'props'} | None (falló)
_cache_lock = threading.Lock()


def _cache_set(nombre, valor):
    _cache[nombre] = valor
    _cache.move_to_end(nombre)
    while len(_cache) > _CACHE_MEMORIA_MAX:
        _cache.popitem(last=False)


def _cache_get(nombre):
    if nombre not in _cache:
        return None, False
    _cache.move_to_end(nombre)
    return _cache[nombre], True


def _ruta_cache_disco(nombre):
    seguro = ''.join(c if c.isalnum() else '_' for c in nombre)
    return os.path.join(CACHE_DIR, f'{seguro}.geojson')


def _cargar_capa(capa):
    nombre = capa['nombre']
    with _cache_lock:
        valor, encontrado = _cache_get(nombre)
        if encontrado:
            return valor
    datos = None
    try:
        ruta_disco = _ruta_cache_disco(nombre)
        if os.path.exists(ruta_disco):
            # Ya la habíamos descargado en un proceso anterior: leerla del
            # disco es cuestión de milisegundos, no depende de la red ni
            # de Hugging Face estando arriba en este momento.
            with open(ruta_disco, 'r', encoding='utf-8') as f:
                geojson = json.load(f)
            print(f'📂 [suelos_inta_regional] Carta "{nombre}" leída de caché en disco.')
        else:
            r = requests.get(capa['url'], timeout=120)
            r.raise_for_status()
            geojson = r.json()
            try:
                with open(ruta_disco, 'w', encoding='utf-8') as f:
                    json.dump(geojson, f)
            except Exception as exc_disco:  # noqa: BLE001
                # Si no se pudo escribir (ej. disco de solo lectura en
                # algún entorno), no es grave: seguimos con lo que ya
                # tenemos en memoria, solo que el próximo restart va a
                # tener que volver a descargar.
                print(f'⚠️ [suelos_inta_regional] No se pudo guardar caché en disco de "{nombre}": {exc_disco}')

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
        _cache_set(nombre, datos)
    return datos


def _precargar_en_segundo_plano():
    """Precarga en un hilo aparte SOLO la primera carta del catálogo
    (la de mayor prioridad) para evitar el 502 en el caso más común.
    A propósito NO precarga las 5 — hacerlo tira 5 capas completas a la
    RAM al arrancar el proceso aunque nadie las use, que es lo que
    generaba los OOM en Render. El resto de las cartas se cargan bajo
    demanda, la primera vez que un pedido realmente las necesita (y con
    el caché en disco ya no dependen de la red para eso)."""
    if SUELO_CAPAS:
        _cargar_capa(SUELO_CAPAS[0])


threading.Thread(target=_precargar_en_segundo_plano, daemon=True).start()


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
