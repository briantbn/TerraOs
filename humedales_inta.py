# ============================================================
#  humedales_inta.py
# ------------------------------------------------------------
#  Probabilidad de humedal (INTA — mapa nacional por subregión
#  ecológica, Navarro et al.) para un punto (lat, lon).
#
#  Complementa a hidrografia_vectorial.py, no lo reemplaza:
#  - hidrografia_vectorial.py dice "¿cuál es el cuerpo de agua
#    VECTORIAL más cercano?" (ríos/lagunas/esteros ya delineados
#    como polígono en las capas por provincia).
#  - humedales_inta.py dice "¿qué tan probable es que ACÁ MISMO
#    haya humedal/agua superficial?", con una capa continua a
#    nivel de píxel. Sirve para detectar esteros/bañados/cuerpos
#    chicos que la capa vectorial todavía no tiene mapeados —
#    justo el caso que motivó agregar esto: puntos en Corrientes
#    donde "Riesgo Hidrológico" no identificaba un estero real
#    porque no había polígono vectorial ahí.
#
#  Fuente: 18 rásters (GeoTIFF/COG) por subregión de humedales de
#  Argentina, subidos a mano a Hugging Face (mismo dataset que
#  hidrografia_vectorial.py: Briant97/GeoSentinel). Valores 0-1
#  (o 0-100, se normaliza) = probabilidad de humedal en ese pixel.
#
#  DISEÑO:
#  - NO se hardcodea a mano el bbox de cada subregión (fuente de
#    error humano). Los archivos son COG: se abren de forma remota
#    con rasterio/GDAL, que solo pide por HTTP la cabecera (unos
#    pocos KB), no el archivo completo — de ahí se lee la
#    extensión real y se cachea en memoria de proceso. Los
#    archivos rotos/caídos no tumban la consulta: quedan
#    cacheados como "sin bbox" y se ignoran.
#  - Un punto puede caer dentro del bbox RECTANGULAR de más de una
#    subregión sin que haya dato real ahí (las subregiones tienen
#    forma irregular, no rectángulos) — se prueban todos los
#    candidatos por bbox y se usa el primero que devuelva un valor
#    válido (no nodata).
#  - Umbrales de clasificación (a pedido, julio 2026):
#      >= 0.90  -> estero / cuerpo de agua grande
#      >= 0.60  -> cuerpo de agua chico
#      <  0.60  -> sin humedal significativo
#
#  Requiere: pip install rasterio requests
#  (rasterio trae GDAL como dependencia — el build en Render puede
#  tardar un poco más la primera vez, es normal.)
# ============================================================

import threading
import requests
import rasterio
from rasterio.warp import transform as rio_transform

HF_BASE_URL = 'https://huggingface.co/datasets/Briant97/GeoSentinel/resolve/main/'

# Nombres tal cual están subidos a Hugging Face (ver links.txt).
ARCHIVOS_HUMEDALES = [
    'Raster - Región Humedales montanos precordilleranos y subandinos.tif',
    'Raster - Región Humedales del Monte Central.tif',
    'Raster - Región Humedales del Chaco.tif',
    'Raster - Humedales Misioneros.tif',
    'Raster - Humedales - Subregión vegas, lagunas y salares de la Puna.tif',
    'Raster - Humedales - Subregión ríos, estero, bañados y lagunas del río Paraná.tif',
    'Raster - Humedales - Subregión ríos y arroyos de los valles intermontanos.tif',
    'Raster - Humedales - Subregión riachos, esteros y bañados del Chaco Húmedo.tif',
    'Raster - Humedales - Subregión mallines y turberas de la Patagonia sur.tif',
    'Raster - Humedales - Subregión lagunas y vegas de la Patagonia extraandina.tif',
    'Raster - Humedales - Subregión lagos, cursos de agua y mallines patagónicos.tif',
    'Raster - Humedales - Subregión Vegas y Lagunas Altoandina.tif',
    'Raster - Humedales - Subregión Playas y marismas de la costa bonaerense.tif',
    'Raster - Humedales - Subregión Lagunas Salobres Pampa Interior.tif',
    'Raster - Humedales - Subregión Lagunas Pampa Húmeda.tif',
    'Raster - Humedales - Salinas de la depresión central.tif',
    'Raster - Humedales - Malezales, tembladerales y arroyos litoraleños.tif',
    'Raster - Humedales - Arroyos y mallines de las Sierras Centrales.tif',
]

UMBRAL_ESTERO = 0.90       # >= => estero / cuerpo de agua grande
UMBRAL_AGUA_CHICA = 0.60   # >= => cuerpo de agua chico

_CACHE_LOCK = threading.Lock()
_CACHE_BOUNDS = {}   # archivo -> (min_lon, min_lat, max_lon, max_lat) en EPSG:4326, o None si no se pudo abrir


def _url_archivo(archivo):
    return HF_BASE_URL + requests.utils.quote(archivo)


def _abrir(archivo):
    """
    Abre el raster remoto SIN descargarlo entero. Al ser COG, GDAL (vía
    /vsicurl/) solo pide por HTTP range-requests la cabecera y, después,
    únicamente las ventanas de píxeles puntuales que se necesiten — no el
    archivo completo en cada consulta.
    """
    url = _url_archivo(archivo)
    return rasterio.open(f'/vsicurl/{url}')


def _bounds_de(archivo):
    """
    Devuelve (y cachea en memoria de proceso) el bbox del archivo en
    EPSG:4326. None si el archivo no se pudo abrir — no debe tumbar la
    consulta de los otros 17.
    """
    with _CACHE_LOCK:
        if archivo in _CACHE_BOUNDS:
            return _CACHE_BOUNDS[archivo]

    bounds4326 = None
    try:
        with _abrir(archivo) as ds:
            b = ds.bounds
            if ds.crs and ds.crs.to_epsg() != 4326:
                xs, ys = rio_transform(ds.crs, 'EPSG:4326', [b.left, b.right], [b.bottom, b.top])
                bounds4326 = (min(xs), min(ys), max(xs), max(ys))
            else:
                bounds4326 = (b.left, b.bottom, b.right, b.top)
    except Exception:
        bounds4326 = None

    with _CACHE_LOCK:
        _CACHE_BOUNDS[archivo] = bounds4326
    return bounds4326


def _punto_en_bbox(lat, lon, bbox):
    if bbox is None:
        return False
    min_lon, min_lat, max_lon, max_lat = bbox
    return min_lon <= lon <= max_lon and min_lat <= lat <= max_lat


def _muestrear(archivo, lat, lon):
    """
    Lee el valor de UN pixel (probabilidad, se normaliza a 0-1) en (lat,
    lon). None si el punto cae en 'nodata' de este raster (fuera del área
    real de la subregión, aunque esté dentro de su bbox rectangular) o si
    algo falla (archivo caído, timeout, etc.).
    """
    try:
        with _abrir(archivo) as ds:
            x, y = lon, lat
            if ds.crs and ds.crs.to_epsg() != 4326:
                xs, ys = rio_transform('EPSG:4326', ds.crs, [lon], [lat])
                x, y = xs[0], ys[0]
            muestras = list(ds.sample([(x, y)]))
            if not muestras:
                return None
            valor = muestras[0][0]
            if valor is None:
                return None
            if ds.nodata is not None and valor == ds.nodata:
                return None
            valor = float(valor)
            if valor != valor:  # NaN
                return None
            # Algunos productos vienen en escala 0-1, otros en 0-100.
            if valor > 1.0:
                valor = valor / 100.0
            return valor
    except Exception:
        return None


def consultar_probabilidad_humedal(lat, lon):
    """
    Punto de entrada principal: prueba todas las subregiones cuyo bbox
    (real, leído del archivo) contiene el punto, y devuelve el resultado
    de la primera que dé un valor válido (no nodata) en ese punto exacto.

    Retorna:
      { encontrado: True, probabilidad: 0.0-1.0,
        nivel: 'estero_o_cuerpo_grande' | 'cuerpo_chico' | 'sin_humedal',
        subregion: <nombre del archivo/subregión> }
    o { encontrado: False } si ningún raster cubre el punto, o todos
    dieron nodata ahí (punto realmente fuera de zona de humedal mapeada).
    """
    candidatos = [a for a in ARCHIVOS_HUMEDALES if _punto_en_bbox(lat, lon, _bounds_de(a))]
    for archivo in candidatos:
        valor = _muestrear(archivo, lat, lon)
        if valor is None:
            continue
        if valor >= UMBRAL_ESTERO:
            nivel = 'estero_o_cuerpo_grande'
        elif valor >= UMBRAL_AGUA_CHICA:
            nivel = 'cuerpo_chico'
        else:
            nivel = 'sin_humedal'
        return {
            'encontrado': True,
            'probabilidad': round(valor, 4),
            'nivel': nivel,
            'subregion': archivo,
        }
    return {'encontrado': False}
