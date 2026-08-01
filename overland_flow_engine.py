"""
overland_flow_engine.py
─────────────────────────────────────────────────────────────────────────
Motor de simulación DINÁMICA de inundación — sin Earth Engine.

A diferencia de local_flood_engine.py (HAND + umbral estático: "esta
celda queda bajo agua si está a menos de X metros del cauce", instantáneo
pero sin noción de tiempo), este módulo resuelve de verdad las ecuaciones
de aguas someras con landlab.OverlandFlow: el agua entra como lluvia,
escurre celda a celda según la pendiente real, y se puede ver cómo avanza
en el tiempo — la diferencia entre "está bajo un umbral" y "el agua
efectivamente llegó ahí en N horas".

El costo de esto es CPU: no es viable a los 60 km de radio de
local_flood_engine. Se usa un radio chico (RADIO_ANALISIS_KM, default
3 km → grilla ~200x200 a 30 m) para que una tormenta de unas horas corra
en el orden de 10-20 segundos en un servidor sin GPU — medido en pruebas
reales antes de fijar este radio.

Reusa la descarga de DEM de local_flood_engine (Copernicus GLO-30, mismos
tiles, mismo caché de red) — no duplica esa lógica.

Requiere: landlab, rasterio, numpy, Pillow (ya en requirements.txt para
local_flood_engine; landlab es la única dependencia nueva).
"""

import math
import os
import time
import threading

import numpy as np

import local_flood_engine  # reusa _descargar_dem() — mismo DEM, mismo caché de tiles

try:
    from landlab import RasterModelGrid
    from landlab.components import OverlandFlow
    _MOTOR_DINAMICO_DISPONIBLE = True
    _error_import = None
except ImportError as _exc:
    RasterModelGrid = None
    OverlandFlow = None
    _MOTOR_DINAMICO_DISPONIBLE = False
    _error_import = str(_exc)


# Radio chico a propósito — ver docstring del módulo. 3 km / 30 m ≈ grilla
# 200x200 (40.000 nodos), probado en ~15s de tiempo real para 6h simuladas.
RADIO_ANALISIS_KM = 3.0
RESOLUCION_M = 30.0

# Límite de pasos de integración por si el CFL da un dt patológicamente
# chico (terreno muy plano/ruidoso) — corta la simulación en vez de
# colgar el request para siempre.
MAX_PASOS_INTEGRACION = 30000
# Tope de tiempo REAL (de reloj), independiente del tope de pasos -- ver
# comentario en el loop de simular_crecida_dinamica. 25s deja margen bajo
# el timeout típico de 30s de proxies/balanceadores delante de Render.
MAX_TIEMPO_REAL_SEG = 25.0

# Umbral de profundidad (m) para considerar una celda "inundada" en el
# render final — por debajo es humedad superficial, no agua de pie.
UMBRAL_PROFUNDIDAD_INUNDADA_M = 0.05

_CACHE_LOCK = threading.Lock()


def _bbox_desde_centro(lat, lon, radio_km):
    radio_grados = radio_km / 111.0
    return (lon - radio_grados, lat - radio_grados, lon + radio_grados, lat + radio_grados)


def simular_crecida_dinamica(lat, lon, mm_totales, horas_lluvia, horas_total,
                              radio_km=None, resolucion_m=None):
    """
    Corre una simulación dinámica de inundación (OverlandFlow) alrededor
    de (lat, lon), con una tormenta de `mm_totales` mm repartidos
    uniformemente en las primeras `horas_lluvia` horas, y `horas_total`
    horas de simulación en total (lluvia + escurrimiento posterior).

    Devuelve dict:
      { profundidad_m: ndarray (m, snapshot final), es_inundada: ndarray bool,
        transform: Affine, crs: CRS, pasos: int, tiempo_seg_simulado: float,
        tiempo_seg_real: float, area_inundada_ha: float, radio_km: float,
        resolucion_m: float }

    Lanza RuntimeError si landlab no está instalado, o si Copernicus no
    tiene cobertura en esa zona (mismo error que local_flood_engine).
    """
    if not _MOTOR_DINAMICO_DISPONIBLE:
        raise RuntimeError(
            f'Motor dinámico no disponible (falta "landlab": {_error_import}). '
            f'Agregalo a requirements.txt y volvé a desplegar.'
        )

    radio_km = radio_km or RADIO_ANALISIS_KM
    resolucion_m = resolucion_m or RESOLUCION_M
    t0 = time.time()

    bbox = _bbox_desde_centro(lat, lon, radio_km)
    # Reusa la descarga/mosaico de tiles Copernicus GLO-30 de local_flood_engine
    # (misma función que usa el HAND estático — no hay lógica nueva acá).
    dem, transform, crs, _nodata = local_flood_engine._descargar_dem(bbox)

    # OverlandFlow necesita una grilla regular en metros, y el DEM viene
    # en grados (WGS84) a ~30m de resolución nativa de Copernicus GLO-30
    # (prácticamente la misma que RESOLUCION_M por defecto, así que este
    # remuestreo es casi 1:1 salvo que se pida una resolución distinta).
    # Se remuestrea a una grilla cuadrada de `resolucion_m` metros de lado
    # usando el tamaño de píxel real (convertido a metros con la misma
    # aproximación esférica que ya usa local_flood_engine para pendiente).
    alto_px, ancho_px = dem.shape
    lat_ref = lat
    px_m = abs(transform.a) * 111320.0 * math.cos(math.radians(lat_ref))
    py_m = abs(transform.e) * 111320.0
    ancho_total_m = ancho_px * px_m
    alto_total_m = alto_px * py_m

    n_col = max(20, int(round(ancho_total_m / resolucion_m)))
    n_fil = max(20, int(round(alto_total_m / resolucion_m)))

    # Remuestreo simple por indexado (nearest) — alcanza para esta escala;
    # evita traer scipy solo para esto.
    idx_fil = np.linspace(0, alto_px - 1, n_fil).round().astype(int)
    idx_col = np.linspace(0, ancho_px - 1, n_col).round().astype(int)
    dem_grilla = dem[np.ix_(idx_fil, idx_col)].astype(float)

    # Relleno de NaN/nodata (bordes de tile, agua ya conocida, etc.) por
    # el valor válido más cercano — OverlandFlow no tolera NaN en la
    # elevación.
    if np.isnan(dem_grilla).any():
        valido = ~np.isnan(dem_grilla)
        if valido.any():
            dem_grilla = np.where(valido, dem_grilla, np.nanmedian(dem_grilla))
        else:
            raise RuntimeError('DEM sin datos válidos en la zona pedida.')

    grid = RasterModelGrid((n_fil, n_col), xy_spacing=resolucion_m)
    grid.add_field('topographic__elevation', dem_grilla.flatten(), at='node')
    grid.add_zeros('surface_water__depth', at='node')
    # Bordes cerrados: el agua no "se escapa" por los 4 lados del recorte,
    # se acumula en las depresiones locales — consistente con que este es
    # un recorte chico dentro de una cuenca mucho más grande, no la cuenca
    # completa con su propia salida natural.
    grid.set_closed_boundaries_at_grid_edges(True, True, True, True)

    of = OverlandFlow(grid, steep_slopes=True)

    lluvia_m_s = (mm_totales / horas_lluvia / 1000.0) / 3600.0 if horas_lluvia > 0 else 0.0
    duracion_lluvia_s = horas_lluvia * 3600.0
    duracion_total_s = horas_total * 3600.0

    t = 0.0
    pasos = 0
    # Freno de TIEMPO REAL, no solo de pasos: en terreno muy plano (llanura,
    # deltas de río) el paso de tiempo estable (CFL) que calcula OverlandFlow
    # puede volverse minúsculo, y un tope de 30.000 pasos puede seguir
    # tardando varios minutos de reloj real -- suficiente para que el
    # request HTTP nunca vuelva (visto en la práctica). Cortamos antes de
    # llegar a eso, devolviendo un resultado parcial con el flag explícito
    # en vez de colgar el request.
    while (t < duracion_total_s
           and pasos < MAX_PASOS_INTEGRACION
           and (time.time() - t0) < MAX_TIEMPO_REAL_SEG):
        of.dt = of.calc_time_step()
        of.rainfall_intensity = lluvia_m_s if t < duracion_lluvia_s else 0.0
        of.overland_flow()
        t += of.dt
        pasos += 1

    profundidad = grid.at_node['surface_water__depth'].reshape(n_fil, n_col)
    es_inundada = profundidad > UMBRAL_PROFUNDIDAD_INUNDADA_M

    area_inundada_ha = float(es_inundada.sum()) * (resolucion_m ** 2) / 10000.0

    # Bounds geográficos del recorte remuestreado (para L.imageOverlay) —
    # mismo bbox que se descargó, no el DEM original a resolución nativa.
    min_lon, min_lat, max_lon, max_lat = bbox

    return {
        'profundidad_m': profundidad.astype('float32'),
        'es_inundada': es_inundada,
        'bounds': [[min_lat, min_lon], [max_lat, max_lon]],
        'pasos': pasos,
        'tiempo_seg_simulado': round(t, 1),
        'tiempo_seg_real': round(time.time() - t0, 2),
        'area_inundada_ha': round(area_inundada_ha, 2),
        'radio_km': radio_km,
        'resolucion_m': resolucion_m,
        'grilla': [n_fil, n_col],
        'truncado_por_max_pasos': pasos >= MAX_PASOS_INTEGRACION,
        'truncado_por_tiempo': (time.time() - t0) >= MAX_TIEMPO_REAL_SEG,
    }


PALETA_PROFUNDIDAD = ['#a8d5ff', '#5b9bd5', '#2e5f9e', '#1a3a6e', '#0d1f42']


def generar_imagen_profundidad_base64(profundidad, es_inundada, vmax=1.5):
    """PNG RGBA en base64 con degradado de profundidad de agua — mismo
    tipo de función que generar_imagen_gradiente_base64() en
    local_flood_engine.py, pero acá el corte es por profundidad > umbral,
    no por HAND."""
    return local_flood_engine.generar_imagen_gradiente_base64(
        profundidad, 0.0, vmax, PALETA_PROFUNDIDAD, mascara_extra=es_inundada,
    )
