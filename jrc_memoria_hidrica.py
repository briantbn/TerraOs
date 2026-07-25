# ============================================================
#  jrc_memoria_hidrica.py
# ------------------------------------------------------------
#  Fase 2 del "Hydrologic Decision Engine": Factor Histórico
#  (FHIS) — los 6 indicadores del JRC Global Surface Water
#  (Pekel et al. 2016, Comisión Europea), SIN depender de GEE en
#  producción. Mismo patrón que humedales_inta.py y
#  terreno_merit_hydro.py: COG remoto + rasterio, sin descargar
#  el archivo entero por consulta.
#
#  A diferencia de MERIT Hydro, esta fuente NO pide registro —
#  descarga directa desde global-surface-water.appspot.com/download,
#  tiles de 10°×10°, licencia solo con atribución ("Source: EC
#  JRC/Google"), sin restricción de uso comercial.
#
#  Las 6 capas (una por indicador, incluidas en el nombre de
#  archivo esperado {capa}_{tile}.tif — AJUSTAR TILES_JRC/CAPAS
#  si el nombre real en la página de descarga difiere):
#    - occurrence:  frecuencia histórica de agua (%, 0-100)
#    - change:      intensidad de cambio entre épocas (%, -100 a 100;
#                    positivo = más agua ahora, negativo = menos)
#    - recurrence:  qué tan seguido vuelve el agua año a año (%, 0-100)
#    - seasonality: cuántos meses del año suele haber agua (0-12)
#    - transitions: clase de transición (código 1-10, ver TRANSICIONES)
#    - extent:      máscara binaria de extensión máxima histórica (0/1)
#
#  DISEÑO: igual que los módulos anteriores — no se hardcodea el
#  bbox de cada tile a mano; se abren los COG remotos (rasterio/
#  GDAL vía /vsicurl/, solo pide la cabecera) y se cachea en
#  memoria de proceso. Tiles no subidos/rotos se ignoran sin
#  tumbar el resto.
#
#  Requiere: pip install rasterio requests
# ============================================================

# ============================================================
#  jrc_memoria_hidrica.py
# ------------------------------------------------------------
#  Fase 2 del "Hydrologic Decision Engine": Factor Histórico
#  (FHIS) — los 6 indicadores del JRC Global Surface Water
#  (Pekel et al. 2016, Comisión Europea), SIN depender de GEE.
#
#  A diferencia de humedales_inta.py y terreno_merit_hydro.py,
#  ACÁ NO HACE FALTA autoalojar nada en Hugging Face: los archivos
#  ya son URLs públicas y estables de Google Cloud Storage
#  (storage.googleapis.com/water-world/...), sin registro ni
#  contraseña. El módulo los lee directo con rasterio/vsicurl,
#  igual de "sin descargar el archivo entero" que los otros COG.
#
#  Patrón de archivo CONFIRMADO (julio 2026, viendo la página real
#  de descarga tile por tile):
#    https://storage.googleapis.com/water-world/download2024/VER1-5/
#      {capa}/{capa}_{lonW}W_{latS}S_v1_5_2024.tif
#  Tile nombrado por su esquina NOROESTE (borde oeste + borde
#  norte); se extiende 10° hacia el este y 10° hacia el sur.
#  Ej.: occurrence_70W_20S_v1_5_2024.tif cubre lon -70 a -60,
#  lat -20 a -30.
#
#  Las 6 capas:
#    - occurrence:  frecuencia histórica de agua (%, 0-100)
#    - change:      intensidad de cambio entre épocas (%, -100 a 100;
#                    positivo = más agua ahora, negativo = menos)
#    - recurrence:  qué tan seguido vuelve el agua año a año (%, 0-100)
#    - seasonality: cuántos meses del año suele haber agua (0-12)
#    - transitions: clase de transición (código 1-10, ver TRANSICIONES)
#    - extent:      máscara binaria de extensión máxima histórica (0/1)
#
#  Requiere: pip install rasterio requests
# ============================================================

import threading
import rasterio
from rasterio.warp import transform as rio_transform

GCS_BASE_URL = 'https://storage.googleapis.com/water-world/download2024/VER1-5/'

# ------------------------------------------------------------
# Grilla de tiles 10°x10° que cubre Argentina (calculada del bbox
# continental + Tierra del Fuego), con la convención real
# confirmada: (lon_oeste, lat_sur) = esquina NOROESTE del tile.
# ------------------------------------------------------------
TILES_ARGENTINA = [
    (60, 20), (60, 30), (60, 40), (60, 50),
    (70, 20), (70, 30), (70, 40), (70, 50),
    (80, 20), (80, 30), (80, 40), (80, 50),
]

CAPAS = ('occurrence', 'change', 'recurrence', 'seasonality', 'transitions', 'extent')

# Códigos de la capa 'transitions' (ver Data Users Guide del JRC).
TRANSICIONES = {
    1: 'Agua permanente, sin cambio',
    2: 'Nueva agua permanente',
    3: 'Agua permanente perdida',
    4: 'Agua estacional, sin cambio',
    5: 'Nueva agua estacional',
    6: 'Agua estacional perdida',
    7: 'Agua estacional a permanente',
    8: 'Agua permanente a estacional',
    9: 'Agua estacional, ausente algunos años',
    10: 'Sin datos / océano',
}

_CACHE_LOCK = threading.Lock()
_CACHE_BOUNDS = {}   # (tile, capa) -> (min_lon, min_lat, max_lon, max_lat) | None


def _tile_str(tile):
    lon_w, lat_s = tile
    return f'{lon_w}W_{lat_s}S'


def _url_archivo(tile, capa):
    return f'{GCS_BASE_URL}{capa}/{capa}_{_tile_str(tile)}_v1_5_2024.tif'


def _abrir(tile, capa):
    """Abre el raster remoto SIN descargarlo entero (rasterio/GDAL vía
    /vsicurl/ — pide por HTTP range-requests solo lo que necesita)."""
    url = _url_archivo(tile, capa)
    return rasterio.open(f'/vsicurl/{url}')


def _bounds_de(tile, capa='occurrence'):
    clave = (tile, capa)
    with _CACHE_LOCK:
        if clave in _CACHE_BOUNDS:
            return _CACHE_BOUNDS[clave]

    bounds4326 = None
    try:
        with _abrir(tile, capa) as ds:
            b = ds.bounds
            if ds.crs and ds.crs.to_epsg() != 4326:
                xs, ys = rio_transform(ds.crs, 'EPSG:4326', [b.left, b.right], [b.bottom, b.top])
                bounds4326 = (min(xs), min(ys), max(xs), max(ys))
            else:
                bounds4326 = (b.left, b.bottom, b.right, b.top)
    except Exception:
        bounds4326 = None

    with _CACHE_LOCK:
        _CACHE_BOUNDS[clave] = bounds4326
    return bounds4326


def _punto_en_bbox(lat, lon, bbox):
    if bbox is None:
        return False
    min_lon, min_lat, max_lon, max_lat = bbox
    return min_lon <= lon <= max_lon and min_lat <= lat <= max_lat


def _tile_para_punto(lat, lon):
    # Cálculo directo por la convención de nombre (más rápido), con
    # fallback a bbox real para puntos justo en el borde de un tile.
    import math
    lon_w_calc = abs(math.floor(lon / 10) * 10)
    lat_s_calc = abs(math.ceil(lat / 10) * 10) if lat < 0 else 0
    candidato = (lon_w_calc, lat_s_calc)
    if candidato in TILES_ARGENTINA and _bounds_de(candidato) is not None:
        return candidato
    for t in TILES_ARGENTINA:
        if _punto_en_bbox(lat, lon, _bounds_de(t)):
            return t
    return None


def _muestrear(tile, capa, lat, lon):
    """Valor de un pixel en (lat, lon). None si nodata o si falla."""
    try:
        with _abrir(tile, capa) as ds:
            x, y = lon, lat
            if ds.crs and ds.crs.to_epsg() != 4326:
                xs, ys = rio_transform('EPSG:4326', ds.crs, [lon], [lat])
                x, y = xs[0], ys[0]
            muestras = list(ds.sample([(x, y)]))
            if not muestras:
                return None
            valor = muestras[0][0]
            if valor is None or (ds.nodata is not None and valor == ds.nodata):
                return None
            return float(valor)
    except Exception:
        return None


def consultar_memoria_hidrica(lat, lon):
    """
    Punto de entrada principal: los 6 indicadores JRC para un punto.

    Retorna:
      { encontrado: True, tile: <'70W_20S'>,
        occurrence_pct, change_pct, recurrence_pct, seasonality_meses,
        transition_codigo, transition_desc, extent_bool,
        memoria_hidrologica, persistencia_agua, tendencia }
    o { encontrado: False } si el punto no cae en ningún tile de la lista.
    """
    tile = _tile_para_punto(lat, lon)
    if tile is None:
        return {'encontrado': False}

    valores = {capa: _muestrear(tile, capa, lat, lon) for capa in CAPAS}

    occurrence = valores['occurrence']
    change = valores['change']
    recurrence = valores['recurrence']
    seasonality = valores['seasonality']
    transition_codigo = int(valores['transitions']) if valores['transitions'] is not None else None
    extent = bool(valores['extent']) if valores['extent'] is not None else None

    resultado = {
        'encontrado': True,
        'tile': _tile_str(tile),
        'occurrence_pct': round(occurrence, 1) if occurrence is not None else None,
        'change_pct': round(change, 1) if change is not None else None,
        'recurrence_pct': round(recurrence, 1) if recurrence is not None else None,
        'seasonality_meses': round(seasonality, 1) if seasonality is not None else None,
        'transition_codigo': transition_codigo,
        'transition_desc': TRANSICIONES.get(transition_codigo),
        'extent_bool': extent,
    }

    # ── Indicadores derivados ──
    if occurrence is not None and recurrence is not None:
        base = (occurrence * 0.5 + recurrence * 0.5)
        if extent:
            base = min(100, base + 10)
        resultado['memoria_hidrologica'] = round(base, 1)
    else:
        resultado['memoria_hidrologica'] = None

    if occurrence is not None and seasonality is not None:
        resultado['persistencia_agua'] = round(occurrence * (seasonality / 12), 1)
    else:
        resultado['persistencia_agua'] = None

    if change is not None:
        if change >= 10:
            resultado['tendencia'] = 'creciendo'
        elif change <= -10:
            resultado['tendencia'] = 'disminuyendo'
        else:
            resultado['tendencia'] = 'estable'
    else:
        resultado['tendencia'] = None

    return resultado

