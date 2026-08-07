"""
Motor de Aptitud Productiva del Suelo — cascada de escalas
════════════════════════════════════════════════════════════════
Reemplaza/alimenta el endpoint /suelo_mejor_resolucion que ya usa el
frontend (ver _aptitudFetchINTA en GeoSentinel_v4.html). En vez de ir
directo al WFS nacional 1:500.000, prueba tus propios COGs (Cloud
Optimized GeoTIFF) alojados en Hugging Face, de más detallado a más
general, y devuelve el primer resultado que efectivamente cubra el
punto consultado:

    1:50.000 → 1:100.000 → 1:200.000 → 1:250.000 → 1:500.000 → 1:1.000.000

Sigue el mismo patrón que ya usás en jrc_raster_regional.py: lectura
por ventana/punto de un COG remoto vía GDAL /vsicurl/, sin descargar
el archivo completo.

ANTES DE USAR — completá 3 cosas:
  1) ESCALAS_APTITUD: la URL real de cada COG en Hugging Face.
  2) LEYENDA_CLASES: qué significa cada valor de píxel de tus COGs
     (clase agrológica, índice de productividad, limitante, etc.).
     Si distintas escalas usan distinta codificación, armá una
     leyenda por escala en vez de una sola compartida (ver nota abajo).
  3) NODATA_VALUES: qué valor(es) representan "sin dato" en tus COGs,
     además del nodata que ya declara cada archivo internamente.

Dependencias: rasterio, pyproj (ambas ya suelen venir si usás GDAL
para los otros COGs del proyecto).
"""

import os
import threading

import rasterio
from rasterio.errors import RasterioIOError
from pyproj import Transformer

# ── Ajustes de GDAL para lectura eficiente de COGs remotos (Hugging Face) ──
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
os.environ.setdefault("VSI_CACHE", "TRUE")
os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "YES")

# ────────────────────────────────────────────────────────────────
# Escalas ordenadas de más detallada a más general. Se prueban en
# este orden; se usa la primera que cubra el punto con un valor
# válido (no nodata).
# ────────────────────────────────────────────────────────────────
ESCALAS_APTITUD = [
    {
        "id": "1:50.000",
        "cog_url": "https://huggingface.co/datasets/Briant97/GeoSentinel/resolve/main/TU_ARCHIVO_1_50000.tif",
        "confiabilidad": "alta",
    },
    {
        "id": "1:100.000",
        "cog_url": "https://huggingface.co/datasets/Briant97/GeoSentinel/resolve/main/TU_ARCHIVO_1_100000.tif",
        "confiabilidad": "alta",
    },
    {
        "id": "1:200.000",
        "cog_url": "https://huggingface.co/datasets/Briant97/GeoSentinel/resolve/main/TU_ARCHIVO_1_200000.tif",
        "confiabilidad": "media",
    },
    {
        "id": "1:250.000",
        "cog_url": "https://huggingface.co/datasets/Briant97/GeoSentinel/resolve/main/TU_ARCHIVO_1_250000.tif",
        "confiabilidad": "media",
    },
    {
        "id": "1:500.000",
        "cog_url": "https://huggingface.co/datasets/Briant97/GeoSentinel/resolve/main/TU_ARCHIVO_1_500000.tif",
        "confiabilidad": "media",
    },
    {
        "id": "1:1.000.000",
        "cog_url": "https://huggingface.co/datasets/Briant97/GeoSentinel/resolve/main/TU_ARCHIVO_1_1000000.tif",
        "confiabilidad": "baja",
    },
]

# Valor de píxel -> atributos de aptitud. AJUSTAR según la codificación
# real de tus COGs. Si cada escala tiene su propia leyenda, cambiá esto
# por un dict de dicts: LEYENDAS_POR_ESCALA["1:50.000"] = {...}, y en
# _resolver_atributos() usá LEYENDAS_POR_ESCALA[escala["id"]].
LEYENDA_CLASES = {
    1: {"clase": "I",    "subclase": None, "indice_prod": 90, "limitante": None,          "drenaje": "Bueno",   "anegabilidad": "Nula"},
    2: {"clase": "II",   "subclase": "e",  "indice_prod": 75, "limitante": "Erosion",     "drenaje": "Bueno",   "anegabilidad": "Nula"},
    3: {"clase": "III",  "subclase": "s",  "indice_prod": 55, "limitante": "Suelo",       "drenaje": "Moderado","anegabilidad": "Baja"},
    4: {"clase": "IV",   "subclase": "w",  "indice_prod": 40, "limitante": "Anegamiento", "drenaje": "Pobre",   "anegabilidad": "Media"},
    5: {"clase": "V",    "subclase": "w",  "indice_prod": 25, "limitante": "Anegamiento", "drenaje": "Pobre",   "anegabilidad": "Alta"},
    6: {"clase": "VI",   "subclase": "e",  "indice_prod": 15, "limitante": "Pendiente",   "drenaje": "Moderado","anegabilidad": "Baja"},
    7: {"clase": "VII",  "subclase": "e",  "indice_prod": 8,  "limitante": "Pendiente",   "drenaje": "Moderado","anegabilidad": "Nula"},
    8: {"clase": "VIII", "subclase": None, "indice_prod": 0,  "limitante": "Multiple",    "drenaje": "Variable","anegabilidad": "Variable"},
}

# Además del nodata que declara cada COG internamente (ds.nodata), acá
# podés listar otros valores que en tus rásters signifiquen "sin dato".
NODATA_VALUES = {0, 255}

# ── Cachés a nivel de proceso (se llenan en la primera consulta) ──
_datasets_cache = {}
_datasets_lock = threading.Lock()
_read_lock = threading.Lock()  # rasterio no es thread-safe para lecturas concurrentes
_transformer_cache = {}


def _abrir_dataset(escala):
    """Abre (o reutiliza) el dataset rasterio para una escala dada.
    La primera consulta paga el costo de abrir el COG remoto; las
    siguientes reutilizan el handle ya abierto."""
    escala_id = escala["id"]
    if escala_id in _datasets_cache:
        return _datasets_cache[escala_id]
    with _datasets_lock:
        if escala_id in _datasets_cache:
            return _datasets_cache[escala_id]
        vsicurl_url = f"/vsicurl/{escala['cog_url']}"
        try:
            ds = rasterio.open(vsicurl_url)
        except RasterioIOError as exc:
            print(f"[aptitud-cascada] No se pudo abrir COG {escala_id}: {exc}")
            ds = None
        _datasets_cache[escala_id] = ds
        return ds


def _leer_valor(ds, lat, lon):
    """Devuelve el valor de píxel en (lat, lon), o None si el punto
    cae fuera de la cobertura de este raster, en nodata, o falla la
    lectura (raster caído, timeout de red, etc.)."""
    if ds is None:
        return None

    crs_key = ds.crs.to_string()
    if crs_key not in _transformer_cache:
        _transformer_cache[crs_key] = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
    x, y = _transformer_cache[crs_key].transform(lon, lat)

    if not (ds.bounds.left <= x <= ds.bounds.right and ds.bounds.bottom <= y <= ds.bounds.top):
        return None  # fuera de la cobertura espacial de este COG

    try:
        with _read_lock:
            valor = next(ds.sample([(x, y)]))[0]
    except Exception as exc:
        print(f"[aptitud-cascada] Error leyendo píxel: {exc}")
        return None

    if ds.nodata is not None and valor == ds.nodata:
        return None
    if valor in NODATA_VALUES:
        return None
    return int(valor)


def consultar_aptitud_cascada(lat, lon):
    """Prueba las escalas en orden (más detallada -> más general) y
    devuelve el primer resultado con dato válido. Esta es la función
    que debe llamar el endpoint /suelo_mejor_resolucion, en el mismo
    formato que ya espera _aptitudFetchINTA() en el frontend."""
    for escala in ESCALAS_APTITUD:
        ds = _abrir_dataset(escala)
        valor = _leer_valor(ds, lat, lon)
        if valor is None:
            continue  # esta escala no cubre el punto (o falló) -> probar la siguiente

        atributos = LEYENDA_CLASES.get(valor)
        if atributos is None:
            print(f"[aptitud-cascada] Valor {valor} sin leyenda en {escala['id']}, se ignora")
            continue

        subclase_txt = atributos["subclase"] or ""
        return {
            "encontrado": True,
            "fuente": f"GeoSentinel — Aptitud Productiva {escala['id']} (COG propio)",
            "clase": atributos["clase"],
            "subclase": atributos["subclase"],
            "capacidad_uso": f"Clase {atributos['clase']}{subclase_txt}",
            "indice_prod": atributos["indice_prod"],
            "limitante": atributos["limitante"],
            "drenaje": atributos["drenaje"],
            "anegabilidad": atributos["anegabilidad"],
            "advertencia_escala": (
                None if escala["confiabilidad"] == "alta" else
                f"Dato a escala {escala['id']}: la precisión posicional y temática es menor "
                f"que en zonas cubiertas por cartografía más detallada."
            ),
            "confiabilidad": escala["confiabilidad"],
            "propiedades_crudas": {"valor_raster": valor, "escala": escala["id"]},
        }

    # Ninguna escala, ni siquiera 1:1.000.000, cubre el punto consultado
    return {"encontrado": False, "fuente": None}


# ────────────────────────────────────────────────────────────────
# Integración con Flask (app.py / GEE_SERVER)
# ────────────────────────────────────────────────────────────────
#
#   from flask import request, jsonify
#   from suelo_aptitud_cascada import consultar_aptitud_cascada
#
#   @app.route("/suelo_mejor_resolucion")
#   def suelo_mejor_resolucion():
#       lat = request.args.get("lat", type=float)
#       lon = request.args.get("lon", type=float)
#       if lat is None or lon is None:
#           return jsonify({"error": "Faltan parámetros lat/lon"}), 400
#       try:
#           return jsonify(consultar_aptitud_cascada(lat, lon))
#       except Exception as exc:
#           return jsonify({"error": str(exc)}), 500
#
# No hace falta tocar el frontend: GeoSentinel_v4.html ya llama a
# ${GEE_SERVER}/suelo_mejor_resolucion?lat=...&lon=... y ya sabe leer
# encontrado / fuente / clase / indice_prod / limitante / drenaje /
# anegabilidad / advertencia_escala / confiabilidad / propiedades_crudas.
