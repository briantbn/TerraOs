# ============================================================
#  terreno_merit_hydro.py
# ------------------------------------------------------------
#  Fase 1 del "Hydrologic Decision Engine": Factor Topográfico
#  (HAND, pendiente, TWI) SIN depender de Google Earth Engine en
#  producción — MERIT Hydro autoalojado, mismo patrón que ya
#  funciona en humedales_inta.py (COG remoto + rasterio, sin
#  descargar el archivo entero por consulta).
#
#  MERIT Hydro (Yamazaki et al. 2019) se descarga directo del
#  desarrollador (registro + licencia, ver
#  hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/), NO exclusivo de
#  GEE. Grilla de 5°x5° (6000x6000 px, ~90m). Este módulo espera
#  3 bandas por tile, ya provistas por MERIT Hydro (no las
#  calculamos nosotros):
#    - hnd: Height Above Nearest Drainage (HAND), en metros.
#    - elv: elevación hidrológicamente corregida, en metros.
#    - upa: área de drenaje acumulada, en km².
#
#  Lo que SÍ calculamos acá (MERIT Hydro no lo da directo):
#    - Pendiente: por diferencias finitas sobre 'elv' (misma idea
#      que ee.Terrain.slope, pero con numpy/rasterio).
#    - TWI = ln( upa_m2 / tan(pendiente_rad) ), fórmula estándar
#      (Beven & Kirkby 1979). Se recorta tan(pendiente) a un
#      mínimo pequeño para evitar división por cero en terreno
#      perfectamente plano.
#
#  DISEÑO: igual que humedales_inta.py — no se hardcodea el bbox
#  de cada tile a mano; se abren los COG remotos (rasterio/GDAL,
#  vía /vsicurl/, solo pide la cabecera) y se cachea en memoria de
#  proceso. Tiles no subidos/rotos se ignoran sin tumbar el resto.
#
#  Requiere: pip install rasterio numpy requests
# ============================================================

import math
import threading
import requests
import numpy as np
import rasterio
from rasterio.warp import transform as rio_transform

HF_BASE_URL = 'https://huggingface.co/datasets/Briant97/GeoSentinel/resolve/main/'
# Subcarpeta donde se suban los tiles MERIT Hydro dentro del mismo dataset
# de Hugging Face (ajustar acá si terminan en otra ruta/carpeta).
MERIT_SUBCARPETA = 'merit_hydro/'

# ------------------------------------------------------------
# Grilla de tiles 5°x5° que cubre Argentina (calculada del bbox
# continental + Tierra del Fuego). Se completa según se vayan
# subiendo los archivos reales — un tile sin archivo simplemente
# no aporta datos ahí (fallback natural, no rompe nada).
# ------------------------------------------------------------
TILES_ARGENTINA = [
    's25w055', 's25w060', 's25w065', 's25w070', 's25w075',
    's30w055', 's30w060', 's30w065', 's30w070', 's30w075',
    's35w055', 's35w060', 's35w065', 's35w070', 's35w075',
    's40w055', 's40w060', 's40w065', 's40w070', 's40w075',
    's45w055', 's45w060', 's45w065', 's45w070', 's45w075',
    's50w055', 's50w060', 's50w065', 's50w070', 's50w075',
    's55w055', 's55w060', 's55w065', 's55w070', 's55w075',
    's60w055', 's60w060', 's60w065', 's60w070', 's60w075',
]

CAPAS = ('hnd', 'elv', 'upa')

_CACHE_LOCK = threading.Lock()
_CACHE_BOUNDS = {}   # (tile, capa) -> (min_lon, min_lat, max_lon, max_lat) | None


def _nombre_archivo(tile, capa):
    return f'{MERIT_SUBCARPETA}{tile}_{capa}.tif'


def _url_archivo(tile, capa):
    return HF_BASE_URL + requests.utils.quote(_nombre_archivo(tile, capa))


def _abrir(tile, capa):
    """Abre el tile remoto SIN descargarlo entero (COG + /vsicurl/)."""
    url = _url_archivo(tile, capa)
    return rasterio.open(f'/vsicurl/{url}')


def _bounds_de(tile, capa='hnd'):
    """
    bbox del tile en EPSG:4326, leído (y cacheado) de la cabecera real del
    archivo. Se usa la capa 'hnd' como referencia — las 3 capas de un mismo
    tile deberían compartir extensión, así no hace falta abrir las 3 para
    decidir si el tile es candidato.
    """
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
        bounds4326 = None  # tile todavía no subido, o roto: se ignora, no tumba nada

    with _CACHE_LOCK:
        _CACHE_BOUNDS[clave] = bounds4326
    return bounds4326


def _punto_en_bbox(lat, lon, bbox):
    if bbox is None:
        return False
    min_lon, min_lat, max_lon, max_lat = bbox
    return min_lon <= lon <= max_lon and min_lat <= lat <= max_lat


def _tile_para_punto(lat, lon):
    """Tile candidato por coincidencia de nombre (más rápido que revisar
    bbox real de los 40), con fallback a búsqueda por bbox real si el
    nombre calculado no tiene archivo subido (ej. tile en el borde)."""
    lat_lbl = f's{abs(math.floor(lat / 5) * 5):02d}' if lat < 0 else f'n{math.floor(lat/5)*5:02d}'
    lon_lbl = f'w{abs(math.floor(lon / 5) * 5):03d}' if lon < 0 else f'e{math.floor(lon/5)*5:03d}'
    candidato = f'{lat_lbl}{lon_lbl}'
    if candidato in TILES_ARGENTINA and _bounds_de(candidato) is not None:
        return candidato
    # Fallback: el punto puede caer justo en el borde de un tile con
    # redondeo distinto al esperado — se prueban todos por bbox real.
    for t in TILES_ARGENTINA:
        if _punto_en_bbox(lat, lon, _bounds_de(t)):
            return t
    return None


def _muestrear_ventana(tile, capa, lat, lon, radio_px=1):
    """
    Lee una ventanita de radio_px alrededor del píxel del punto (no un
    único píxel) — necesario para calcular pendiente por diferencias
    finitas. Devuelve (array numpy, resolución en grados) o (None, None)
    si falla.
    """
    try:
        with _abrir(tile, capa) as ds:
            x, y = lon, lat
            if ds.crs and ds.crs.to_epsg() != 4326:
                xs, ys = rio_transform('EPSG:4326', ds.crs, [lon], [lat])
                x, y = xs[0], ys[0]
            fila, col = ds.index(x, y)
            ventana = rasterio.windows.Window(
                col_off=max(0, col - radio_px), row_off=max(0, fila - radio_px),
                width=radio_px * 2 + 1, height=radio_px * 2 + 1,
            )
            datos = ds.read(1, window=ventana, boundless=True, fill_value=ds.nodata)
            if ds.nodata is not None:
                datos = np.where(datos == ds.nodata, np.nan, datos).astype(float)
            resolucion = abs(ds.transform.a)  # grados/píxel (tiles en EPSG:4326)
            return datos, resolucion
    except Exception:
        return None, None


def consultar_terreno(lat, lon):
    """
    Punto de entrada principal: HAND, pendiente y TWI para un punto.

    Retorna:
      { encontrado: True, tile: <nombre>,
        hand_m: float | None,
        pendiente_grados: float | None,
        pendiente_pct: float | None,
        area_drenaje_km2: float | None,
        twi: float | None }
    o { encontrado: False } si el punto no cae en ningún tile subido.
    """
    tile = _tile_para_punto(lat, lon)
    if tile is None:
        return {'encontrado': False}

    resultado = {'encontrado': True, 'tile': tile,
                 'hand_m': None, 'pendiente_grados': None, 'pendiente_pct': None,
                 'area_drenaje_km2': None, 'twi': None}

    # HAND: valor puntual, no hace falta ventana.
    datos_hnd, _ = _muestrear_ventana(tile, 'hnd', lat, lon, radio_px=0)
    if datos_hnd is not None and datos_hnd.size and not np.isnan(datos_hnd[0, 0]):
        resultado['hand_m'] = round(float(datos_hnd[0, 0]), 2)

    # Pendiente: por diferencias finitas sobre una ventana 3x3 de 'elv'.
    datos_elv, res_grados = _muestrear_ventana(tile, 'elv', lat, lon, radio_px=1)
    pendiente_rad = None
    if datos_elv is not None and datos_elv.shape == (3, 3) and not np.isnan(datos_elv).any():
        metros_por_grado = 111_320.0  # aproximación válida a esta latitud/escala
        paso_m = res_grados * metros_por_grado
        dz_dx = (datos_elv[1, 2] - datos_elv[1, 0]) / (2 * paso_m)
        dz_dy = (datos_elv[2, 1] - datos_elv[0, 1]) / (2 * paso_m)
        pendiente_rad = math.atan(math.hypot(dz_dx, dz_dy))
        resultado['pendiente_grados'] = round(math.degrees(pendiente_rad), 2)
        resultado['pendiente_pct'] = round(math.tan(pendiente_rad) * 100, 2)

    # Área de drenaje acumulada (upa, en km² según MERIT Hydro).
    datos_upa, _ = _muestrear_ventana(tile, 'upa', lat, lon, radio_px=0)
    area_km2 = None
    if datos_upa is not None and datos_upa.size and not np.isnan(datos_upa[0, 0]):
        area_km2 = float(datos_upa[0, 0])
        resultado['area_drenaje_km2'] = round(area_km2, 4)

    # TWI = ln(área_acumulada_m2 / tan(pendiente)) — fórmula estándar
    # (Beven & Kirkby, 1979). tan(pendiente) con un piso mínimo para no
    # dividir por cero en terreno perfectamente plano.
    if area_km2 is not None and pendiente_rad is not None:
        area_m2 = area_km2 * 1_000_000
        tan_pendiente = max(math.tan(pendiente_rad), 0.001)
        resultado['twi'] = round(math.log(area_m2 / tan_pendiente), 3)

    return resultado
