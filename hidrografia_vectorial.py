# ============================================================
#  hidrografia_vectorial.py
# ------------------------------------------------------------
#  Módulo independiente y reutilizable (por IIPDI, simulación de
#  inundaciones, riesgo hídrico, expansión de cuerpos de agua, etc.)
#
#  Responsabilidad única: dado un punto (lat, lon), decir qué
#  cuerpo de agua vectorial (río/arroyo/laguna/estero/...) tiene
#  más cerca, de qué tipo es y su coeficiente de influencia.
#
#  Fuente de datos: capas GeoJSON por provincia alojadas en
#  Hugging Face (https://huggingface.co/datasets/Briant97/GeoSentinel).
#  Son públicas: no requieren token para descargarse.
#
#  DISEÑO PARA ESCALA:
#  - NO descarga las 19 capas al arrancar el backend (serían
#    cientos de MB en RAM). Descarga solo la provincia del punto
#    consultado, la reproyecta una vez y la cachea en memoria de
#    proceso para consultas siguientes.
#  - Usa un índice espacial (STRtree de Shapely) por provincia,
#    así buscar "el más cercano" no recorre miles de features
#    secuencialmente.
#
#  Requiere: pip install shapely pyproj requests
# ============================================================

import re
import json
import threading
import requests
from shapely.geometry import shape, Point
from shapely.strtree import STRtree
from pyproj import Transformer

HF_BASE_URL = 'https://huggingface.co/datasets/Briant97/GeoSentinel/resolve/main/'

# ------------------------------------------------------------
# Mapeo provincia -> archivo + bounding box aproximado
# (min_lon, min_lat, max_lon, max_lat). Bboxes aproximados: alcanzan
# para elegir el archivo correcto, no para límites catastrales.
# Se revisa en orden: los específicos ANTES que "NOA" (que agrupa
# varias provincias), para que La Rioja no caiga en el archivo NOA.
# ------------------------------------------------------------
PROVINCIAS = [
    ('La Rioja',      'Cuerpos de agua de La Rioja.geojson',  (-69.0, -31.8, -66.0, -28.0)),
    ('NOA',           'Cuerpos de agua NOA.geojson',          (-69.2, -30.5, -61.5, -21.8)),
    ('Córdoba',       'Cuerpos de agua Córdoba.geojson',      (-65.6, -35.0, -61.8, -29.5)),
    ('Buenos Aires',  'Cuerpos de agua de Bs As.geojson',     (-63.4, -41.0, -56.6, -33.3)),
    ('Chaco',         'Chaco.geojson',                        (-63.0, -27.6, -58.6, -24.0)),
    ('Formosa',       'Cuerpos de agua de Formosa.geojson',   (-62.5, -26.7, -57.5, -22.5)),
    ('Misiones',      'Cuerpos de agua de Msiones.geojson',   (-56.0, -28.2, -53.6, -25.3)),
    ('Corrientes',    'agua_corrientes.geojson',              (-59.7, -30.3, -55.6, -27.0)),
    ('Entre Ríos',    'Entre Rios.geojson',                   (-60.9, -34.0, -57.8, -30.0)),
    ('Santa Fe',      'Santa Fe.geojson',                     (-63.4, -34.5, -59.7, -28.0)),
    ('La Pampa',      'La Pampa.geojson',                     (-68.3, -39.0, -63.3, -35.0)),
    ('Mendoza',       'Mendoza_combined.geojson',             (-70.6, -37.6, -66.5, -32.0)),
    ('San Juan',      'San Juan.geojson',                     (-70.8, -32.6, -66.5, -28.5)),
    ('San Luis',      'San Luis.geojson',                     (-67.2, -36.3, -64.0, -32.0)),
    ('Neuquén',       'Neuquén.geojson',                      (-71.8, -41.0, -68.0, -36.0)),
    ('Río Negro',     'Río Negro.geojson',                    (-71.8, -42.0, -62.0, -38.0)),
    ('Chubut',        'Chubut.geojson',                       (-72.0, -46.3, -63.5, -42.0)),
    ('Santa Cruz',    'Santa Cruz.geojson',                   (-73.5, -52.5, -64.9, -46.0)),
]

# Coeficientes por tipo de cuerpo de agua (configurable: valores
# iniciales del prompt + los tipos reales que aparecieron en los
# datos). Clave en minúsculas y sin acentos para matchear tolerante.
COEFICIENTES_TIPO_AGUA = {
    'rio': 1.00,
    'estero': 0.95,
    'embalse': 0.90,
    'banado': 0.85,
    'laguna': 0.80,
    'canada': 0.75,
    'arroyo': 0.70,
    'valle aluvial': 0.65,
    'carrizal': 0.60,
}
COEFICIENTE_DEFECTO = 0.50  # tipo desconocido / no clasificado

_CACHE_LOCK = threading.Lock()
_CACHE_INDICE = {}   # nombre_archivo -> {'tree': STRtree, 'props': [...], 'geoms': [...]}


def _quitar_acentos(texto):
    reemplazos = str.maketrans('áéíóúñÁÉÍÓÚÑ', 'aeiounAEIOUN')
    return texto.translate(reemplazos)


def _normalizar_tipo(tipo_raw):
    if not tipo_raw:
        return 'desconocido'
    return _quitar_acentos(str(tipo_raw)).strip().lower()


def coeficiente_de_tipo(tipo_raw):
    return COEFICIENTES_TIPO_AGUA.get(_normalizar_tipo(tipo_raw), COEFICIENTE_DEFECTO)


def _archivos_candidatos_para_punto(lat, lon):
    """
    Devuelve TODOS los archivos cuyo bbox contiene el punto (no solo el
    primero). Es necesario porque provincias limítrofes (ej. Chaco y
    Corrientes, separadas por el río Paraná) tienen bboxes aproximados
    que se superponen cerca del límite real. Elegir "el primero que
    matchea" daba resultados incorrectos en esas zonas; en cambio, se
    consultan todos los candidatos y se compara la distancia real al
    cuerpo de agua más cercano en cada uno (ver buscar_cuerpo_mas_cercano).
    """
    return [archivo for _nombre, archivo, bbox in PROVINCIAS
            if bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]]


def _extraer_epsg(geojson_dict):
    """Lee el EPSG del bloque 'crs' del GeoJSON, si existe. Si no
    tiene 'crs', el estándar GeoJSON asume WGS84 (EPSG:4326)."""
    crs = geojson_dict.get('crs')
    if not crs:
        return 4326
    nombre = crs.get('properties', {}).get('name', '')
    m = re.search(r'EPSG::?(\d+)', nombre)
    return int(m.group(1)) if m else 4326


def _construir_indice(archivo):
    """Descarga (si no está cacheado), reproyecta a WGS84 si hace
    falta, y arma el índice espacial de un archivo de provincia."""
    with _CACHE_LOCK:
        if archivo in _CACHE_INDICE:
            return _CACHE_INDICE[archivo]

        url = HF_BASE_URL + requests.utils.quote(archivo)
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        epsg_origen = _extraer_epsg(data)
        transformer = None
        if epsg_origen != 4326:
            transformer = Transformer.from_crs(f'EPSG:{epsg_origen}', 'EPSG:4326', always_xy=True)

        geoms, props = [], []
        for feat in data.get('features', []):
            geom_raw = feat.get('geometry')
            if not geom_raw:
                continue
            try:
                geom = shape(geom_raw)
                if transformer:
                    geom = _reproyectar_geom(geom, transformer)
                geoms.append(geom)
                props.append(feat.get('properties', {}) or {})
            except Exception:
                continue  # feature corrupta: se ignora, no se cae todo el índice

        tree = STRtree(geoms) if geoms else None
        _CACHE_INDICE[archivo] = {'tree': tree, 'props': props, 'geoms': geoms}
        return _CACHE_INDICE[archivo]


def _reproyectar_geom(geom, transformer):
    from shapely.ops import transform as shp_transform
    return shp_transform(lambda x, y, z=None: transformer.transform(x, y), geom)


def buscar_cuerpo_mas_cercano(lat, lon, radio_km=20):
    """
    Devuelve el cuerpo de agua vectorial más cercano al punto, comparando
    entre TODOS los archivos de provincia cuyo bbox contiene el punto
    (relevante en zonas de frontera entre provincias). Devuelve None si
    no hay ningún archivo candidato o no hay features dentro del radio.

    Retorna: { tipo, nombre, distancia_m, coeficiente, provincia_archivo }
    """
    archivos = _archivos_candidatos_para_punto(lat, lon)
    if not archivos:
        return None

    punto = Point(lon, lat)
    radio_grados = radio_km / 111.0

    mejor_global = None
    mejor_dist_grados = None

    for archivo in archivos:
        try:
            indice = _construir_indice(archivo)
        except Exception:
            continue  # un archivo caído/con error no debe tumbar la consulta entera
        if not indice['tree']:
            continue

        candidatos_idx = indice['tree'].query(punto.buffer(radio_grados))
        for idx in candidatos_idx:
            geom = indice['geoms'][idx]
            d = punto.distance(geom)
            if mejor_dist_grados is None or d < mejor_dist_grados:
                mejor_dist_grados = d
                props = indice['props'][idx]
                mejor_global = {'props': props, 'archivo': archivo}

    if mejor_global is None:
        return None

    distancia_m = mejor_dist_grados * 111_000  # aproximación válida a esta escala local
    props = mejor_global['props']
    tipo_raw = props.get('tipo', props.get('TIPO', ''))
    nombre = props.get('nombre', props.get('NOMBRE', props.get('CUENCA', 'Sin nombre')))

    return {
        'tipo': tipo_raw or 'desconocido',
        'nombre': nombre,
        'distancia_m': round(distancia_m, 1),
        'coeficiente': coeficiente_de_tipo(tipo_raw),
        'provincia_archivo': mejor_global['archivo'],
    }
