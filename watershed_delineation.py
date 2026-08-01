"""
watershed_delineation.py
=========================

Delimitación de la cuenca hidrológica aportante a un punto de salida (el
punto más bajo del lote de un productor, por donde el agua sale del
predio), usando WhiteboxTools sobre un recorte LOCAL del DEM de MERIT
Hydro (banda 'elv', obtenido vía Earth Engine).

Por qué este enfoque y no `delineator.py` / MERIT-Basins:
- delineator.py (Upstream-Tech/delineator, mheberger/delineator) requiere
  autoalojar datasets GLOBALES de MERIT-Basins/MERIT-Hydro (decenas de GB
  de vectores + rásters) y un entorno Python 3.11 con uv/mise -- pensado
  para delimitar cuencas a escala continental/de investigación, corriendo
  como script por lotes, no como respuesta en vivo a un click en el mapa.
- Este módulo, en cambio, descarga SOLO el recorte de DEM necesario
  alrededor del punto (unos pocos MB) y corre el pipeline hidrológico
  estándar D8 con WhiteboxTools -- resuelve en segundos, sin datasets
  globales, con la misma arquitectura que ya usa el resto de la app
  (Earth Engine para el dato crudo + procesamiento local en Python).

Pipeline (igual al de referencia, adaptado de un script en R con
WhiteboxTools, aquí en Python puro vía los bindings oficiales `whitebox`):

  DEM (MERIT Hydro, EE) -> suavizado -> relleno de depresiones
      -> dirección de flujo D8 -> acumulación de flujo D8
      -> enganche del punto de salida al cauce más cercano
      -> delimitación de cuenca -> vectorización a polígono (rasterio)

No inventa reglas de clasificación: es un único resultado geométrico (el
polígono de la cuenca aportante), no una clasificación por umbrales.

Requiere: ee (ya inicializado en app.py), whitebox, rasterio, shapely,
requests, pyshp (para escribir el punto de salida como shapefile, formato
que exige WhiteboxTools para pour points).
(agregar a requirements.txt: whitebox, pyshp)
"""

import os
import shutil
import tempfile

import numpy as np
import rasterio
import requests
import ee
from rasterio.features import shapes
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

# WKT de WGS84, para el .prj del shapefile del punto de salida -- mismo
# CRS que el DEM descargado de Earth Engine (EPSG:4326), así WhiteboxTools
# no necesita reproyectar nada entre el punto y el ráster.
_WGS84_WKT = (
    'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",'
    '6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],'
    'UNIT["Degree",0.0174532925199433]]'
)

# Distancia (en grados, ~500 m) para "enganchar" el punto de salida al
# píxel de mayor acumulación de flujo más cercano -- absorbe el error de
# unos metros al marcar manualmente el punto más bajo del lote.
SNAP_DIST_GRADOS = 0.005


def _descargar_dem_recorte(lat, lon, buffer_km, workdir):
    """
    Descarga (vía Earth Engine, sin exportar a Google Drive) el recorte
    de MERIT Hydro (banda 'elv', ~90 m) alrededor del punto, como GeoTIFF
    local en `workdir`. Para un buffer de 15 km esto son unos pocos MB.
    """
    region = ee.Geometry.Point([lon, lat]).buffer(buffer_km * 1000).bounds()
    dem = ee.Image('MERIT/Hydro/v1_0_1').select('elv').clip(region)

    url = dem.getDownloadURL({
        'region': region,
        'scale': 90,
        'format': 'GEO_TIFF',
    })

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    dem_path = os.path.join(workdir, 'dem.tif')
    with open(dem_path, 'wb') as f:
        f.write(resp.content)
    return dem_path


def _escribir_punto_salida(lat, lon, workdir):
    """
    Escribe el punto de salida (outlet/pour point) como shapefile --
    formato que exigen `snap_pour_points`/`watershed` de WhiteboxTools.
    """
    import shapefile  # pyshp

    path = os.path.join(workdir, 'outlet.shp')
    w = shapefile.Writer(path, shapeType=shapefile.POINT)
    w.field('id', 'N')
    w.point(lon, lat)
    w.record(1)
    w.close()

    with open(os.path.join(workdir, 'outlet.prj'), 'w') as f:
        f.write(_WGS84_WKT)

    return path


def _correr_paso(nombre, funcion, *args, **kwargs):
    """
    WhiteboxTools no lanza excepciones en Python: devuelve un código de
    salida (0 = éxito). Este wrapper lo chequea y convierte una falla
    silenciosa en una excepción explícita, con el nombre del paso que
    falló -- para no terminar vectorizando un raster vacío sin saber por
    qué.
    """
    codigo = funcion(*args, **kwargs)
    if codigo != 0:
        raise RuntimeError(f'WhiteboxTools falló en el paso "{nombre}" (código {codigo}).')


def _raster_a_poligono(path, banda=1):
    """
    Vectoriza el raster binario de la cuenca (>0 = dentro de la cuenca) a
    un único polígono (unión de todas las celdas con valor > 0), y estima
    su área en km².
    """
    with rasterio.open(path) as src:
        data = src.read(banda)
        transform = src.transform

    mascara = data > 0
    if not mascara.any():
        return None, 0.0

    geoms = [
        shape(geom)
        for geom, val in shapes(mascara.astype(np.uint8), mask=mascara, transform=transform)
        if val == 1
    ]
    if not geoms:
        return None, 0.0

    poligono = unary_union(geoms).simplify(0.0003, preserve_topology=True)

    # Área aproximada en km² (factor por latitud, sin reproyectar a un CRS
    # de igual área -- suficiente para cuencas de escala de campo/predio).
    lat_media = transform.f + (transform.e * data.shape[0] / 2)
    factor_km2_por_grado2 = 111.0 * 111.0 * max(0.15, np.cos(np.radians(lat_media)))
    area_km2 = poligono.area * factor_km2_por_grado2

    return poligono, area_km2


def delimitar_cuenca_aportante(lat, lon, buffer_km=15.0):
    """
    Delimita la cuenca hidrológica que aporta agua al punto (lat, lon) --
    pensado para ser el punto más bajo del lote del productor, el lugar
    por donde el agua sale del predio hacia aguas abajo.

    `buffer_km` acota el área de búsqueda: debe ser mayor que el radio
    esperado de la cuenca aportante. Si la cuenca real es más grande que
    el buffer, el resultado queda recortado en el borde del buffer (no es
    una cuenca "cerrada" completa) -- en ese caso conviene reintentar con
    un buffer_km mayor.

    Devuelve: { ok, fuente, buffer_km, area_km2, geojson: {Feature con el
    polígono de la cuenca} } o { ok:false, error }.
    """
    workdir = tempfile.mkdtemp(prefix='cuenca_')
    try:
        _descargar_dem_recorte(lat, lon, buffer_km, workdir)
        _escribir_punto_salida(lat, lon, workdir)

        import whitebox
        wbt = whitebox.WhiteboxTools()
        wbt.set_working_dir(workdir)
        wbt.verbose = False

        # Mismo pipeline D8 estándar que el script de referencia:
        _correr_paso('suavizado', wbt.fast_almost_gaussian_filter,
                     'dem.tif', 'dem_smooth.tif', sigma=1.8)
        _correr_paso('relleno de depresiones', wbt.breach_depressions,
                     'dem_smooth.tif', 'dem_breach.tif')
        _correr_paso('dirección de flujo (D8)', wbt.d8_pointer,
                     'dem_breach.tif', 'fdr.tif')
        _correr_paso('acumulación de flujo (D8)', wbt.d8_flow_accumulation,
                     'dem_breach.tif', 'fac.tif', 'cells')
        _correr_paso('enganche del punto de salida', wbt.snap_pour_points,
                     'outlet.shp', 'fac.tif', 'snap.shp', SNAP_DIST_GRADOS)
        _correr_paso('delimitación de cuenca', wbt.watershed,
                     'fdr.tif', 'snap.shp', 'watershed.tif')

        poligono, area_km2 = _raster_a_poligono(os.path.join(workdir, 'watershed.tif'))

        if poligono is None:
            return {
                'ok': False,
                'error': ('No se pudo delimitar una cuenca en este punto. Probá marcando '
                          'un punto más cercano a un cauce definido, o aumentá buffer_km '
                          'si la cuenca real es más grande que el área analizada.'),
            }

        return {
            'ok': True,
            'fuente': 'MERIT Hydro (Earth Engine, recorte local) + WhiteboxTools (D8)',
            'buffer_km': buffer_km,
            'area_km2': round(area_km2, 3),
            'geojson': {
                'type': 'Feature',
                'geometry': mapping(poligono),
                'properties': {'area_km2': round(area_km2, 3)},
            },
        }
    except Exception as e:  # noqa: BLE001
        return {'ok': False, 'error': f'No se pudo delimitar la cuenca: {e}'}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
