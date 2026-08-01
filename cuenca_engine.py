"""
cuenca_engine.py
─────────────────────────────────────────────────────────────────────────
Delimitación de cuenca hidrológica (watershed) aguas arriba de un punto
de salida (outlet) marcado por el usuario en el mapa — típicamente donde
un arroyo cruza el lote, una tranquera, o cualquier punto de interés.

Mismo motor que local_flood_engine.py (pysheds sobre Copernicus GLO-30):
reusa _descargar_dem() para no duplicar la lógica de mosaico de tiles,
y el mismo parche de np.in1d / construcción de Grid vía Raster+ViewFinder
(pysheds 0.5 nunca aceptó un dataset de rasterio directo — ver la nota
grande en local_flood_engine.py).

Diferencia clave con el HAND estático: acá no hay un radio fijo chico
como en lisflood_engine.py — una cuenca puede ser mucho más grande que
el lote del productor (por diseño: el usuario pidió la cuenca COMPLETA
aguas arriba, no solo el área cercana al lote). Por eso el radio de
descarga es bastante más generoso (RADIO_ANALISIS_KM_CUENCA), y aun así
se avisa explícitamente si la cuenca resultante toca el borde del área
descargada (truncada = probablemente sea más grande en la realidad).
"""

import math
import time

import numpy as np

import local_flood_engine  # reusa _descargar_dem(), Grid/Raster/ViewFinder, el parche de np.in1d

try:
    from pysheds.grid import Grid
    from pysheds.view import Raster, ViewFinder
    from shapely.geometry import shape as _shapely_shape, mapping as _shapely_mapping
    from shapely.ops import unary_union
    _MOTOR_CUENCA_DISPONIBLE = local_flood_engine._MOTOR_LOCAL_DISPONIBLE
    _error_import = None if _MOTOR_CUENCA_DISPONIBLE else local_flood_engine._error_import
except Exception as _exc:
    Grid = Raster = ViewFinder = None
    _shapely_shape = _shapely_mapping = unary_union = None
    _MOTOR_CUENCA_DISPONIBLE = False
    _error_import = str(_exc)

# Radio generoso a propósito — una cuenca hidrológica puede abarcar
# mucho más que el lote del productor. 40 km de radio (80x80 km de área)
# cubre la enorme mayoría de cuencas de arroyos/cañadas de la región
# NEA/Pampeana. Si la cuenca real es mayor, se detecta (toca el borde
# del recorte) y se avisa — no se inventa un límite falso.
RADIO_ANALISIS_KM_CUENCA = 40.0

# Mismo umbral que usa local_flood_engine.py para distinguir "esto es un
# cauce" de "esto es solo ladera" — reusado tal cual para consistencia
# (así el outlet se "engancha" al mismo cauce que ya detecta el HAND).
UMBRAL_ACUMULACION_CAUCE = local_flood_engine.UMBRAL_ACUMULACION_CAUCE

# Radio de búsqueda (en celdas) para enganchar el click del usuario al
# cauce real más cercano — el productor difícilmente clickea el píxel
# exacto del cauce.
DISTANCIA_SNAP_MAX_CELDAS = 15


def _bbox_desde_centro(lat, lon, radio_km):
    radio_grados = radio_km / 111.0
    return (lon - radio_grados, lat - radio_grados, lon + radio_grados, lat + radio_grados)


def delimitar_cuenca(lat, lon, radio_km=None):
    """
    Delimita la cuenca hidrológica completa aguas arriba del punto
    (lat, lon) marcado por el usuario.

    Devuelve dict:
      { geojson: {type: 'Polygon'|'MultiPolygon', coordinates: [...]},
        area_km2: float,
        outlet_click: {lat, lon},       # el punto que clickeó el usuario
        outlet_enganchado: {lat, lon},  # el punto real usado (enganchado al cauce)
        distancia_enganche_m: float,
        truncada: bool,   # True si la cuenca toca el borde del área
                          # descargada -> probablemente sea más grande
        radio_km: float,
        tiempo_seg: float }

    Lanza RuntimeError si el motor no está disponible, o ValueError si
    no se pudo enganchar el punto a ningún cauce cercano (ej. el click
    quedó en medio de una loma sin drenaje definido a esta resolución).
    """
    if not _MOTOR_CUENCA_DISPONIBLE:
        raise RuntimeError(
            f'Motor de delimitación de cuenca no disponible (falta pysheds/shapely: '
            f'{_error_import}). Agregalos a requirements.txt y volvé a desplegar.'
        )

    radio_km = radio_km or RADIO_ANALISIS_KM_CUENCA
    t0 = time.time()

    bbox = _bbox_desde_centro(lat, lon, radio_km)
    dem, transform, crs, nodata = local_flood_engine._descargar_dem(bbox)

    hand, acumulacion, fdir, dem_sin_dep = local_flood_engine._calcular_hand(
        dem, transform, crs, nodata,
    )

    # _calcular_hand ya corrió fill_pits/fill_depressions/resolve_flats/
    # flowdir/accumulation — pero necesitamos el Grid y el Raster de fdir
    # de nuevo acá para catchment()/snap_to_mask()/polygonize(), que
    # _calcular_hand no expone (solo devuelve arrays numpy sueltos).
    # Se reconstruye el mismo Grid, barato (no vuelve a correr pysheds).
    nodata_val = nodata if nodata is not None else -32768.0
    viewfinder = ViewFinder(affine=transform, shape=dem.shape, nodata=nodata_val, crs=crs)
    grid = Grid.from_raster(Raster(dem.astype('float32'), viewfinder=viewfinder))

    fdir_raster = Raster(fdir, viewfinder=viewfinder)
    acc_raster = Raster(acumulacion, viewfinder=viewfinder)
    mascara_cauce = acc_raster > UMBRAL_ACUMULACION_CAUCE

    if not mascara_cauce.any():
        raise ValueError(
            'No se detectó ningún cauce dentro del área descargada — '
            'probá con un punto más cerca de un arroyo/cañada visible en el mapa.'
        )

    try:
        x_snap, y_snap = grid.snap_to_mask(mascara_cauce, (lon, lat))
    except Exception as exc:
        raise ValueError(f'No se pudo enganchar el punto a un cauce cercano: {exc}')

    # Distancia real del enganche, para avisar si el click quedó lejos
    # del cauce (probable error de click, no de datos).
    px_m = abs(transform.a) * 111320.0 * math.cos(math.radians(lat))
    py_m = abs(transform.e) * 111320.0
    distancia_m = math.hypot((x_snap - lon) * 111320.0 * math.cos(math.radians(lat)),
                              (y_snap - lat) * 111320.0)

    catch = grid.catchment(x=x_snap, y=y_snap, fdir=fdir_raster, xytype='coordinate')
    catch_arr = np.asarray(catch).astype(bool)

    if not catch_arr.any():
        raise ValueError(
            'La delimitación no encontró ninguna celda de cuenca aguas arriba de ese punto.'
        )

    # Truncamiento: ¿la cuenca toca el borde del área descargada? Si sí,
    # la cuenca real probablemente sigue más allá de lo que se bajó.
    truncada = bool(
        catch_arr[0, :].any() or catch_arr[-1, :].any()
        or catch_arr[:, 0].any() or catch_arr[:, -1].any()
    )

    area_km2 = float(catch_arr.sum()) * px_m * py_m / 1_000_000.0

    grid.clip_to(catch)
    catch_view = grid.view(catch)
    catch_view_int = Raster(np.asarray(catch_view).astype('int32'), viewfinder=catch_view.viewfinder)

    polys = [
        _shapely_shape(geom) for geom, valor in grid.polygonize(catch_view_int)
        if valor == 1
    ]
    if not polys:
        raise ValueError('No se pudo vectorizar el polígono de la cuenca.')

    geom_final = unary_union(polys)
    # Simplifica un poco (tolerancia ~1 píxel) — sin esto el polígono
    # tiene un vértice por cada borde de píxel, extremadamente pesado
    # para mandar al navegador sin ninguna ganancia visual real.
    geom_final = geom_final.simplify(abs(transform.a) * 1.5, preserve_topology=True)

    return {
        'geojson': _shapely_mapping(geom_final),
        'area_km2': round(area_km2, 2),
        'outlet_click': {'lat': lat, 'lon': lon},
        'outlet_enganchado': {'lat': y_snap, 'lon': x_snap},
        'distancia_enganche_m': round(distancia_m, 1),
        'truncada': truncada,
        'radio_km': radio_km,
        'tiempo_seg': round(time.time() - t0, 2),
    }
