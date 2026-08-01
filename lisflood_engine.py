"""
lisflood_engine.py
─────────────────────────────────────────────────────────────────────────
Motor de simulación DINÁMICA de inundación usando LISFLOOD-FP — sin GEE,
sin landlab.

Reemplaza a overland_flow_engine.py (landlab.OverlandFlow). El problema
que resuelve: importar landlab cuesta ~150-180 MB de RAM (medido) SOLO
por cargarlo, aun sin correr nada — en un servidor de 512 MB compartido
con rasterio/pysheds/scipy/scikit-image ya cargados, eso alcanza para
tirar el proceso por memoria (confirmado: crash real en producción).

LISFLOOD-FP es un modelo hidrodinámico C++ (Universidad de Bristol,
GPL-3.0) que resuelve la ecuación de inercia local para flujo superficial
sobre una grilla — el mismo tipo de física que OverlandFlow, pero:
  - Corre como PROCESO EXTERNO (subprocess), no una librería Python
    cargada en el proceso de gunicorn. Su memoria se libera apenas
    termina — no se acumula en el worker de por vida.
  - Medido en pruebas: ~5 MB de RAM para una grilla 50x50, unos
    segundos de ejecución. Muchísimo más liviano que cargar landlab.

Requiere el binario `lisflood` compilado (ver lisflood_src/ y el Build
Command de Render — el código fuente está vendorizado en el repo, se
compila con `make` durante el deploy). Este módulo NO compila nada en
runtime, solo invoca el binario ya compilado.
"""

import math
import os
import shutil
import subprocess
import tempfile
import time

import numpy as np

import local_flood_engine  # reusa _descargar_dem() — mismo DEM, mismo caché de tiles

# Ruta del binario compilado. LISFLOOD_BIN permite overridearla por env var
# si el binario termina en otro lado (ver notas de deploy).
RUTA_BINARIO = os.environ.get('LISFLOOD_BIN') or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'bin', 'lisflood',
)

_MOTOR_DINAMICO_DISPONIBLE = os.path.isfile(RUTA_BINARIO) and os.access(RUTA_BINARIO, os.X_OK)
_error_import = None if _MOTOR_DINAMICO_DISPONIBLE else (
    f'Binario lisflood no encontrado o sin permiso de ejecución en {RUTA_BINARIO}. '
    f'Ver instrucciones de compilación (Build Command de Render) en lisflood_src/README_DEPLOY.md.'
)

# Radio/resolución medidos en pruebas reales: 3km/30m tardaba ~99s (muy
# cerca del límite de un request HTTP síncrono); 2km/40m corre en ~27s,
# con margen cómodo. Si hace falta más área, hay que pasar a un modelo
# asíncrono (job en background + polling) en vez de agrandar esto.
RADIO_ANALISIS_KM = 2.0
RESOLUCION_M = 40.0

# Manning's n (fricción de superficie) para pastizal/campo típico de la
# región — valor estándar de referencia hidrológica (0.03-0.05 pasto
# corto, 0.06-0.10 pastizal/cultivo), no calibrado a una parcela real.
FPFRIC_DEFECTO = 0.06

UMBRAL_PROFUNDIDAD_INUNDADA_M = 0.05
# Antes en 100s -- en terreno muy plano (llanura/humedal, ej. Mesopotamia),
# el esquema explícito necesita pasos de tiempo internos más chicos para
# la misma duración simulada, y 100s no siempre alcanza aunque el radio/
# resolución ya sean los "seguros" (2km/40m). Subido a 160s, dejando ~20s
# de margen bajo el --timeout 180 de gunicorn (ver Start Command de
# Render) -- no elimina el problema de fondo (es físico/numérico, no un
# bug), pero le da más chance real a los casos de terreno plano.
TIMEOUT_SUBPROCESO_SEG = 160


def _bbox_desde_centro(lat, lon, radio_km):
    radio_grados = radio_km / 111.0
    return (lon - radio_grados, lat - radio_grados, lon + radio_grados, lat + radio_grados)


def _escribir_dem_ascii(ruta, dem, cellsize, nodata=-9999.0):
    """Escribe un DEM en formato Arc/Info ASCII grid (el que espera
    LISFLOOD-FP vía DEMfile) — header de 6 líneas + filas de valores."""
    alto, ancho = dem.shape
    with open(ruta, 'w') as f:
        f.write(f'ncols        {ancho}\n')
        f.write(f'nrows        {alto}\n')
        f.write('xllcorner    0.0\n')
        f.write('yllcorner    0.0\n')
        f.write(f'cellsize     {cellsize}\n')
        f.write(f'NODATA_value {nodata}\n')
        dem_out = np.where(np.isnan(dem), nodata, dem)
        for fila in dem_out:
            f.write(' '.join(f'{v:.3f}' for v in fila) + '\n')


def _escribir_rain(ruta, mm_totales, horas_lluvia, horas_total):
    """Archivo .rain de LISFLOOD-FP: tasa constante (mm_totales/horas_lluvia
    mm/h) durante horas_lluvia, después corta a 0 hasta el final."""
    mm_hora = mm_totales / horas_lluvia if horas_lluvia > 0 else 0.0
    with open(ruta, 'w') as f:
        f.write('rain\n')
        f.write('2 hours\n')
        f.write(f'{mm_hora:.4f} 0.0\n')
        f.write(f'0.0 {horas_lluvia:.4f}\n')


def _escribir_par(ruta, dem_filename, rain_filename, resroot, sim_time_seg,
                   fpfric=FPFRIC_DEFECTO):
    with open(ruta, 'w') as f:
        f.write(f'DEMfile          {dem_filename}\n')
        f.write(f'resroot          {resroot}\n')
        f.write(f'sim_time         {sim_time_seg}\n')
        f.write('initial_tstep    1\n')
        f.write(f'massint          {max(sim_time_seg, 1)}\n')
        f.write(f'saveint          {max(sim_time_seg, 1)}\n')
        f.write(f'fpfric           {fpfric}\n')
        f.write('acceleration\n')  # esquema explícito de inercia local — rápido, estable
        f.write(f'rainfall         {rain_filename}\n')


def _leer_ascii_grid(ruta):
    with open(ruta) as f:
        cabecera = {}
        for _ in range(6):
            clave, valor = f.readline().split()
            cabecera[clave] = float(valor)
        datos = np.loadtxt(f)
    nodata = cabecera.get('NODATA_value', -9999.0)
    datos = np.where(datos == nodata, np.nan, datos)
    return datos, cabecera


def simular_crecida_dinamica(lat, lon, mm_totales, horas_lluvia, horas_total,
                              radio_km=None, resolucion_m=None):
    """
    Misma firma que overland_flow_engine.simular_crecida_dinamica() — así
    /inundacion_dinamica en app.py no necesita cambios, solo el import.

    Corre LISFLOOD-FP como subproceso en un directorio temporal (se borra
    al terminar, incluida cualquier memoria que haya usado el binario).

    Devuelve dict: { profundidad_m, es_inundada, bounds, pasos (None —
    LISFLOOD-FP no expone el conteo de pasos vía CLI), tiempo_seg_simulado,
    tiempo_seg_real, area_inundada_ha, radio_km, resolucion_m, grilla,
    truncado_por_max_pasos (siempre False acá) }.
    """
    if not _MOTOR_DINAMICO_DISPONIBLE:
        raise RuntimeError(
            f'Motor dinámico (LISFLOOD-FP) no disponible: {_error_import}'
        )

    radio_km = radio_km or RADIO_ANALISIS_KM
    resolucion_m = resolucion_m or RESOLUCION_M
    t0 = time.time()

    bbox = _bbox_desde_centro(lat, lon, radio_km)
    dem, transform, _crs, _nodata = local_flood_engine._descargar_dem(bbox)

    alto_px, ancho_px = dem.shape
    px_m = abs(transform.a) * 111320.0 * math.cos(math.radians(lat))
    py_m = abs(transform.e) * 111320.0
    n_col = max(20, int(round((ancho_px * px_m) / resolucion_m)))
    n_fil = max(20, int(round((alto_px * py_m) / resolucion_m)))

    idx_fil = np.linspace(0, alto_px - 1, n_fil).round().astype(int)
    idx_col = np.linspace(0, ancho_px - 1, n_col).round().astype(int)
    dem_grilla = dem[np.ix_(idx_fil, idx_col)].astype(float)
    del dem  # el DEM a resolución nativa ya no hace falta — liberar memoria

    if np.isnan(dem_grilla).any():
        valido = ~np.isnan(dem_grilla)
        if not valido.any():
            raise RuntimeError('DEM sin datos válidos en la zona pedida.')
        dem_grilla = np.where(valido, dem_grilla, np.nanmedian(dem_grilla))

    tmpdir = tempfile.mkdtemp(prefix='lisflood_')
    try:
        ruta_dem = os.path.join(tmpdir, 'dem.asc')
        ruta_rain = os.path.join(tmpdir, 'rain.rain')
        ruta_par = os.path.join(tmpdir, 'modelo.par')

        _escribir_dem_ascii(ruta_dem, dem_grilla, cellsize=resolucion_m)
        _escribir_rain(ruta_rain, mm_totales, horas_lluvia, horas_total)
        sim_time_seg = int(horas_total * 3600)
        _escribir_par(ruta_par, 'dem.asc', 'rain.rain', 'out', sim_time_seg)

        proceso = subprocess.run(
            [RUTA_BINARIO, 'modelo.par'],
            cwd=tmpdir, capture_output=True, text=True,
            timeout=TIMEOUT_SUBPROCESO_SEG,
        )
        if proceso.returncode != 0:
            raise RuntimeError(
                f'LISFLOOD-FP terminó con error (código {proceso.returncode}): '
                f'{proceso.stderr[-500:] or proceso.stdout[-500:]}'
            )

        ruta_max = os.path.join(tmpdir, 'out.max')
        if not os.path.isfile(ruta_max):
            raise RuntimeError(
                f'LISFLOOD-FP no generó out.max (salida: {proceso.stdout[-500:]})'
            )
        profundidad, _cabecera = _leer_ascii_grid(ruta_max)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    profundidad = np.nan_to_num(profundidad, nan=0.0).astype('float32')
    es_inundada = profundidad > UMBRAL_PROFUNDIDAD_INUNDADA_M
    area_inundada_ha = float(es_inundada.sum()) * (resolucion_m ** 2) / 10000.0

    min_lon, min_lat, max_lon, max_lat = bbox
    return {
        'profundidad_m': profundidad,
        'es_inundada': es_inundada,
        'bounds': [[min_lat, min_lon], [max_lat, max_lon]],
        'pasos': None,
        'tiempo_seg_simulado': sim_time_seg,
        'tiempo_seg_real': round(time.time() - t0, 2),
        'area_inundada_ha': round(area_inundada_ha, 2),
        'radio_km': radio_km,
        'resolucion_m': resolucion_m,
        'grilla': [n_fil, n_col],
        'truncado_por_max_pasos': False,
    }


PALETA_PROFUNDIDAD = ['#a8d5ff', '#5b9bd5', '#2e5f9e', '#1a3a6e', '#0d1f42']


def generar_imagen_profundidad_base64(profundidad, es_inundada, vmax=1.5):
    return local_flood_engine.generar_imagen_gradiente_base64(
        profundidad, 0.0, vmax, PALETA_PROFUNDIDAD, mascara_extra=es_inundada,
    )
