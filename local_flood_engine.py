# ============================================================
#  local_flood_engine.py
# ------------------------------------------------------------
#  Motor de HAND (Height Above Nearest Drainage) LOCAL, sin Earth
#  Engine, para reemplazar el cuello de botella de zona_baja /
#  anegamiento / inundacion_animacion (30 min por simulación en
#  el enfoque actual basado en GEE + cumulativeCost).
#
#  POR QUÉ ES LENTO HOY (GEE): cada click pide a un servicio
#  remoto que reparte el cómputo en teselas con rate-limiting
#  (ya se vio 429 con el motor multicriterio) — no está pensado
#  para "dame el HAND de esta zona YA". Cada tesela nueva vuelve
#  a pagar ese costo de red + cola, aunque sea la MISMA zona que
#  ya se consultó antes.
#
#  ENFOQUE ACÁ: separar cómputo pesado (una vez por zona, unos
#  segundos con pysheds corriendo en este mismo proceso) de
#  consulta (instantánea, léctura de un raster ya cacheado en
#  disco). Sin red externa de por medio en el camino caliente.
#
#  FUENTE DEL DEM: Copernicus GLO-30 (30 m), bucket público de
#  AWS Open Data (Sinergise/ESA), sin autenticación, servido como
#  COG — se lee por HTTP range-request solo la ventana de píxeles
#  que hace falta (rasterio /vsicurl/ + merge con bounds), no el
#  tile de 1°x1° completo. https://registry.opendata.aws/copernicus-dem/
#
#  POR QUÉ NO HACE FALTA UNA MÁSCARA DE AGUA APARTE PARA LA
#  CONECTIVIDAD: HAND ya resuelve esto por construcción. HAND(celda)
#  se calcula caminando por la ruta de descenso de esa celda HASTA
#  la celda de cauce (definida acá por acumulación de flujo alta)
#  y restando la elevación de esa celda de cauce. Si una celda
#  tiene HAND bajo, es PORQUE existe un camino descendente real
#  hasta un cauce — una depresión aislada sin salida hacia el
#  cauce nunca tiene HAND bajo, aunque esté topográficamente baja
#  en términos absolutos. Por eso acá "candidatas" (hand <= umbral)
#  y "zona_conectada" (lo que en el motor GEE requería un
#  cumulativeCost aparte) son LO MISMO — no hace falta repetir esa
#  segunda pasada costosa que sí necesitaba el enfoque de elevación
#  absoluta / GEE.
#
#  CACHÉ: por celda de grilla de ~0.08° (~9 km en esta latitud),
#  para que clicks cercanos reutilicen el mismo raster ya
#  calculado. Se guarda como GeoTIFF de 2 bandas (hand, acumulación
#  de flujo normalizada) en CACHE_DIR. Nunca se borra sola — si
#  hace falta refrescar una zona (DEM actualizado, bug en el
#  cálculo), se borra el archivo a mano.
#
#  LIMITACIÓN CONOCIDA (asumida a propósito, igual que el resto
#  del motor de inundación): el umbral de HAND efectivo a partir
#  de mm de lluvia (ver mm_a_umbral_hand) es una aproximación
#  Curve Number + factor de escala, NO una simulación
#  hidrodinámica real (no resuelve Saint-Venant/onda de crecida).
#  Sirve para el mismo propósito que ya cumple SIMULADOR_HAND_UMBRAL
#  hoy: dar una superficie de referencia rápida y consistente, no
#  un pronóstico de precisión de ingeniería.
#
#  Requiere: pip install pysheds rasterio numpy
# ============================================================

import os
import math
import time
import threading

import numpy as np

try:
    import rasterio
    from rasterio.merge import merge as _rio_merge
    from rasterio.io import MemoryFile
    from pysheds.grid import Grid
    _MOTOR_LOCAL_DISPONIBLE = True
    _error_import = None
except Exception as _exc:
    _MOTOR_LOCAL_DISPONIBLE = False
    _error_import = str(_exc)


# ------------------------------------------------------------
# Configuración
# ------------------------------------------------------------

# Bucket público AWS Open Data — GLO-30, sin autenticación.
COP30_BASE_URL = 'https://copernicus-dem-30m.s3.amazonaws.com'

# Radio de análisis FIJO para el cálculo del HAND (independiente del
# radius_km que pide el frontend para mostrar el mapa, que puede llegar
# a 800 km). No hace falta más: la propagación de inundación real ya
# estaba topeada a 50 km en el motor GEE actual (MAX_DIST_CONECTIVIDAD_M);
# 60 km da margen sin disparar el tamaño del raster innecesariamente.
RADIO_ANALISIS_KM = 60.0

# Tamaño de celda de la grilla de caché (grados). Dos clicks dentro de la
# misma celda reutilizan el mismo raster ya calculado sin recalcular nada.
CACHE_GRID_DEG = 0.08

# Umbral de acumulación de flujo (en celdas contribuyentes aguas arriba)
# para considerar una celda "cauce" y usarla como red de drenaje del
# cálculo de HAND. A 30 m, 500 celdas ≈ 0.45 km² de cuenca aportante —
# suficiente para captar arroyos permanentes chicos sin marcar como
# "cauce" cualquier surco de escorrentía menor.
UMBRAL_ACUMULACION_CAUCE = 500

# Curve Number por defecto (suelo franco/mixto, condición de humedad
# media II) cuando no se tiene dato de suelo real del punto. El
# endpoint puede pasar uno mejor si ya consultó suelos_inta_regional.
CURVE_NUMBER_DEFECTO = 75

# Factor que traduce "escorrentía acumulada (mm)" del método SCS-CN a
# "cuánto sube el HAND (m)" en el punto. ESTO ES UNA APROXIMACIÓN: el
# método CN da profundidad de escorrentía sobre la cuenca, no nivel de
# agua en el cauce, que depende de la geometría del canal (ancho,
# pendiente) — no resoluble sin datos batimétricos que no tenemos.
# Con CN=75 (suelo mixto), este valor da ~0.5 m de HAND para 60 mm de
# lluvia y ~1.4 m para 100 mm — un evento fuerte del NEA. Es un punto
# de partida razonable, no una calibración contra eventos reales; el
# mismo criterio de honestidad que ya usa /calibrar_escenario_inundacion
# para los escenarios históricos aplica acá: si más adelante se quiere
# ajustar contra una crecida real conocida, este es el número a tocar.
FACTOR_ESCORRENTIA_A_HAND = 0.035  # metros de HAND por mm de escorrentía

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache_hand')

_CACHE_LOCK = threading.Lock()
_CACHE_MEMORIA = {}  # clave -> {'hand': ndarray, 'transform': Affine, 'crs': CRS, 'nodata': float}


# ------------------------------------------------------------
# Descarga del DEM (Copernicus GLO-30, COG, lectura por ventana)
# ------------------------------------------------------------

def _nombre_tile_cop30(lat_floor, lon_floor):
    ns = 'S' if lat_floor < 0 else 'N'
    ew = 'W' if lon_floor < 0 else 'E'
    norte = abs(lat_floor)
    este = abs(lon_floor)
    return f'Copernicus_DSM_COG_10_{ns}{norte:02d}_00_{ew}{este:03d}_00_DEM'


def _tiles_para_bbox(min_lon, min_lat, max_lon, max_lat):
    """Lista de nombres de tile GLO-30 (1°x1°) que cubren el bbox dado."""
    tiles = []
    lat0 = math.floor(min_lat)
    lat1 = math.floor(max_lat)
    lon0 = math.floor(min_lon)
    lon1 = math.floor(max_lon)
    for lat_floor in range(lat0, lat1 + 1):
        for lon_floor in range(lon0, lon1 + 1):
            tiles.append(_nombre_tile_cop30(lat_floor, lon_floor))
    return tiles


def _url_tile(nombre_tile):
    return f'/vsicurl/{COP30_BASE_URL}/{nombre_tile}/{nombre_tile}.tif'


def _descargar_dem(bbox):
    """
    Mosaiquea (y recorta directo al bbox) los tiles GLO-30 necesarios.
    rasterio.merge con bounds= lee, vía GDAL /vsicurl/, solo las ventanas
    de píxeles del bbox pedido en cada tile fuente (range-requests HTTP
    sobre el COG) — NUNCA descarga un tile de 1°x1° completo para esto.
    Devuelve (array 2D float32, transform, crs, nodata). Algunos tiles
    del mosaico pueden no existir (zonas oceánicas sin dato, o alguno de
    los pocos países todavía no liberados en GLO-30 Public) — se
    ignoran individualmente, no tumban el mosaico completo si al menos
    un tile responde.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    nombres = _tiles_para_bbox(min_lon, min_lat, max_lon, max_lat)

    fuentes_abiertas = []
    for nombre in nombres:
        try:
            ds = rasterio.open(_url_tile(nombre))
            fuentes_abiertas.append(ds)
        except Exception:
            continue  # tile inexistente (océano / no liberado): se sigue sin él

    if not fuentes_abiertas:
        raise RuntimeError(
            f'Ningún tile Copernicus GLO-30 disponible para el bbox {bbox} '
            f'(tiles intentados: {nombres})'
        )

    try:
        mosaico, transform = _rio_merge(
            fuentes_abiertas, bounds=(min_lon, min_lat, max_lon, max_lat),
            resampling=rasterio.enums.Resampling.bilinear,
        )
        crs = fuentes_abiertas[0].crs
        nodata = fuentes_abiertas[0].nodata
    finally:
        for ds in fuentes_abiertas:
            ds.close()

    dem = mosaico[0].astype('float32')
    return dem, transform, crs, nodata


# ------------------------------------------------------------
# HAND local (pysheds): fill pits/depressions, resolve flats,
# flow direction D8, acumulación, HAND.
# ------------------------------------------------------------

def _calcular_hand(dem, transform, crs, nodata):
    """
    Corre el pipeline hidrológico estándar de pysheds sobre el DEM ya
    recortado a la zona de análisis, y devuelve (hand, acumulacion,
    fdir) como arrays numpy, mismo shape que `dem`.

    pysheds necesita un archivo/raster real para armar su Grid (no un
    array suelto en memoria) — se escribe el mosaico a un GeoTIFF
    temporal EN MEMORIA (rasterio MemoryFile, sin tocar disco) y se
    arma el Grid desde ahí.
    """
    perfil = {
        'driver': 'GTiff', 'height': dem.shape[0], 'width': dem.shape[1],
        'count': 1, 'dtype': 'float32', 'crs': crs, 'transform': transform,
        'nodata': nodata if nodata is not None else -32768.0,
    }
    with MemoryFile() as memfile:
        with memfile.open(**perfil) as dataset:
            dataset.write(dem, 1)
        with memfile.open() as dataset:
            grid = Grid.from_raster(dataset)
            dem_grid = grid.read_raster(dataset)

    dem_sin_pits = grid.fill_pits(dem_grid)
    dem_sin_depresiones = grid.fill_depressions(dem_sin_pits)
    dem_final = grid.resolve_flats(dem_sin_depresiones)

    fdir = grid.flowdir(dem_final)
    acumulacion = grid.accumulation(fdir)

    mascara_cauce = acumulacion > UMBRAL_ACUMULACION_CAUCE
    hand = grid.compute_hand(fdir, dem_final, mascara_cauce)

    return np.asarray(hand), np.asarray(acumulacion), np.asarray(fdir)


# ------------------------------------------------------------
# Caché en disco (por celda de grilla ~9 km)
# ------------------------------------------------------------

def _clave_cache(lat, lon):
    lat_snap = round(lat / CACHE_GRID_DEG) * CACHE_GRID_DEG
    lon_snap = round(lon / CACHE_GRID_DEG) * CACHE_GRID_DEG
    return f'hand_{lat_snap:.2f}_{lon_snap:.2f}'.replace('-', 'm')


def _ruta_cache(clave):
    return os.path.join(CACHE_DIR, f'{clave}.tif')


def _guardar_cache(clave, hand, acumulacion, transform, crs, nodata_out):
    os.makedirs(CACHE_DIR, exist_ok=True)
    perfil = {
        'driver': 'GTiff', 'height': hand.shape[0], 'width': hand.shape[1],
        'count': 2, 'dtype': 'float32', 'crs': crs, 'transform': transform,
        'nodata': nodata_out,
    }
    with rasterio.open(_ruta_cache(clave), 'w', **perfil) as dst:
        dst.write(hand.astype('float32'), 1)
        dst.write(acumulacion.astype('float32'), 2)


def _cargar_cache(clave):
    ruta = _ruta_cache(clave)
    if not os.path.exists(ruta):
        return None
    with rasterio.open(ruta) as ds:
        hand = ds.read(1)
        acumulacion = ds.read(2)
        return {
            'hand': hand, 'acumulacion': acumulacion,
            'transform': ds.transform, 'crs': ds.crs, 'nodata': ds.nodata,
        }


# ------------------------------------------------------------
# Punto de entrada principal
# ------------------------------------------------------------

def obtener_hand_local(lat, lon, forzar_recalculo=False):
    """
    Devuelve el HAND local (calculado con Copernicus GLO-30 + pysheds,
    SIN pasar por Earth Engine) para la zona alrededor de (lat, lon).

    Primera consulta de una zona nueva: descarga (COG, por ventana) +
    calcula el pipeline hidrológico completo. Con RADIO_ANALISIS_KM=60
    (raster de ~4000x4000 px a 30 m) esto corre en el orden de
    segundos/pocos minutos en un servidor sin GPU — no los 30 minutos
    del enfoque GEE, porque no hay red/cola de por medio en el cómputo
    en sí, solo la descarga inicial del DEM.
    Consultas siguientes a la MISMA zona (o una celda de grilla cercana,
    ver CACHE_GRID_DEG): lectura de disco, milisegundos.

    Retorna dict:
      { hand: ndarray, acumulacion: ndarray, transform: Affine,
        crs: CRS, nodata: float, de_cache: bool, tiempo_seg: float }
    Lanza RuntimeError si el motor no está disponible (pysheds/rasterio
    no instalados) o si Copernicus no tiene cobertura en esa zona.
    """
    if not _MOTOR_LOCAL_DISPONIBLE:
        raise RuntimeError(
            f'Motor de HAND local no disponible (falta pysheds/rasterio: {_error_import}). '
            f'Agregalos a requirements.txt y volvé a desplegar.'
        )

    t0 = time.time()
    clave = _clave_cache(lat, lon)

    with _CACHE_LOCK:
        if not forzar_recalculo and clave in _CACHE_MEMORIA:
            resultado = dict(_CACHE_MEMORIA[clave])
            resultado['de_cache'] = True
            resultado['tiempo_seg'] = round(time.time() - t0, 3)
            return resultado

    if not forzar_recalculo:
        desde_disco = _cargar_cache(clave)
        if desde_disco is not None:
            with _CACHE_LOCK:
                _CACHE_MEMORIA[clave] = desde_disco
            resultado = dict(desde_disco)
            resultado['de_cache'] = True
            resultado['tiempo_seg'] = round(time.time() - t0, 3)
            return resultado

    # Cache miss real: descargar DEM + correr el pipeline hidrológico.
    radio_grados = RADIO_ANALISIS_KM / 111.0
    bbox = (lon - radio_grados, lat - radio_grados, lon + radio_grados, lat + radio_grados)

    dem, transform, crs, nodata = _descargar_dem(bbox)
    hand, acumulacion, _fdir = _calcular_hand(dem, transform, crs, nodata)

    nodata_out = -9999.0
    hand = np.where(np.isnan(hand), nodata_out, hand)

    _guardar_cache(clave, hand, acumulacion, transform, crs, nodata_out)

    resultado = {
        'hand': hand, 'acumulacion': acumulacion,
        'transform': transform, 'crs': crs, 'nodata': nodata_out,
    }
    with _CACHE_LOCK:
        _CACHE_MEMORIA[clave] = resultado

    resultado = dict(resultado)
    resultado['de_cache'] = False
    resultado['tiempo_seg'] = round(time.time() - t0, 3)
    return resultado


# ------------------------------------------------------------
# mm de lluvia -> umbral de HAND (Curve Number, aproximación —
# ver nota extensa arriba en FACTOR_ESCORRENTIA_A_HAND)
# ------------------------------------------------------------

def mm_a_umbral_hand(mm_lluvia, curve_number=CURVE_NUMBER_DEFECTO):
    """
    Convierte mm de lluvia en un umbral de HAND (metros de "subida"
    equivalente), vía el método SCS Curve Number (escorrentía directa)
    + un factor de escala fijo. A más mm, más escorrentía efectiva, más
    umbral de HAND -> se pintan más celdas -> el efecto de "ir tiñendo
    más zonas a medida que subís los mm" que pediste, igual al patrón
    del sitio de referencia del INTA.

    Devuelve 0.0 si la lluvia no supera la abstracción inicial del
    suelo (0.2*S) — es decir, lluvias chicas no generan ninguna mancha,
    como es físicamente correcto.
    """
    s = (25400.0 / curve_number) - 254.0  # retención potencial máxima, mm
    ia = 0.2 * s  # abstracción inicial (intercepción, infiltración inicial)
    if mm_lluvia <= ia:
        return 0.0
    escorrentia_mm = ((mm_lluvia - ia) ** 2) / (mm_lluvia - ia + s)
    return round(escorrentia_mm * FACTOR_ESCORRENTIA_A_HAND, 3)


def generar_mascara_inundacion(hand, umbral_m, nodata):
    """
    Máscara booleana de celdas "inundadas" para un umbral de HAND dado.
    Por construcción de HAND (ver nota grande arriba del módulo), esto
    YA incluye la conectividad real al cauce — no hace falta una pasada
    de cumulativeCost aparte como en el motor GEE.
    """
    valido = hand != nodata
    return valido & (hand <= umbral_m)


# ------------------------------------------------------------
# Render a PNG (para L.imageOverlay en el frontend — no es un
# tile server con pirámide de zoom, es una sola imagen recortada
# a la zona de análisis, que alcanza para el caso de uso del
# simulador: "quiero ver YA la mancha de esta zona").
# ------------------------------------------------------------

# Paleta por profundidad relativa (hand vs umbral): más "por debajo" del
# umbral => color más oscuro/saturado, igual idea visual que el mapa de
# referencia del INTA (zonas de anegamiento más profundo se distinguen
# de las recién alcanzadas por el agua).
_COLOR_AGUA_CLARO = (125, 211, 252)   # celda recién alcanzada (hand ~= umbral)
_COLOR_AGUA_OSCURO = (3, 105, 161)    # celda muy por debajo del umbral (hand ~= 0)
_ALPHA_MASCARA = 190


def bounds_geograficos(transform, shape):
    """
    [[sur, oeste], [norte, este]] a partir del transform afín de rasterio
    y el shape (alto, ancho) del raster — formato que espera directo
    L.imageOverlay(url, bounds) en Leaflet.
    """
    alto, ancho = shape
    oeste, norte = transform * (0, 0)
    este, sur = transform * (ancho, alto)
    return [[sur, oeste], [norte, este]]


def generar_imagen_png_base64(hand, umbral_m, nodata):
    """
    Renderiza la máscara de inundación (hand <= umbral_m) como PNG RGBA
    en base64 (para insertar directo en un data URL del frontend, sin
    guardar archivo). Transparente donde no hay agua candidata.
    """
    from PIL import Image
    import io
    import base64

    mascara = generar_mascara_inundacion(hand, umbral_m, nodata)
    alto, ancho = hand.shape
    rgba = np.zeros((alto, ancho, 4), dtype=np.uint8)

    if mascara.any():
        # Profundidad relativa 0 (recién cubierta) .. 1 (la más profunda
        # dentro de la zona) SOLO entre las celdas inundadas, para que la
        # gradación de color siempre use el rango completo de la paleta
        # sin importar qué tan grande sea el umbral pedido.
        hand_mascara = np.where(mascara, hand, np.nan)
        hand_min = np.nanmin(hand_mascara)
        rango = max(umbral_m - hand_min, 1e-6)
        profundidad = np.clip((umbral_m - hand) / rango, 0, 1)

        for canal in range(3):
            claro = _COLOR_AGUA_CLARO[canal]
            oscuro = _COLOR_AGUA_OSCURO[canal]
            rgba[..., canal] = np.where(
                mascara,
                (claro + (oscuro - claro) * profundidad).astype(np.uint8),
                0,
            )
        rgba[..., 3] = np.where(mascara, _ALPHA_MASCARA, 0)

    imagen = Image.fromarray(rgba, mode='RGBA')
    buffer = io.BytesIO()
    imagen.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('ascii')
