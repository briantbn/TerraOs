"""
jrc_raster_regional.py
=======================

Envoltura histórica del comportamiento del agua, a partir de los rasters
oficiales JRC Global Surface Water v1.5 (2024) — occurrence, recurrence y
seasonality — alojados como Cloud-Optimized GeoTIFF (COG) en Hugging Face.

Lee SOLO la ventana necesaria alrededor de un punto (no descarga el archivo
completo en cada consulta, gracias al formato COG + lectura por HTTP range).

No inventa índices: usa directamente occurrence (%), recurrence (%) y
seasonality (meses/año), las variables oficiales del JRC, combinadas con
reglas lógicas explícitas (no una fórmula ponderada ni un puntaje compuesto).

Requiere: rasterio, shapely, numpy
(agregar a requirements.txt: rasterio, shapely  — numpy ya debería estar)
"""

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.features import shapes
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

# ── Fuentes de datos (COG en Hugging Face) ──────────────────────────────
OCCURRENCE_URL = "/vsicurl/https://huggingface.co/datasets/Briant97/GeoSentinel/resolve/main/occurrence_ar_cog.tif"
RECURRENCE_URL = "/vsicurl/https://huggingface.co/datasets/Briant97/GeoSentinel/resolve/main/recurrence_ar_cog.tif"
SEASONALITY_URL = "/vsicurl/https://huggingface.co/datasets/Briant97/GeoSentinel/resolve/main/seasonality_ar_cog.tif"

# JRC codifica "sin dato" (tierra firme, nunca agua) típicamente como 0 o 255
# según el producto; se filtran ambos casos por seguridad.
NODATA_VALORES = (0, 255)

NIVEL_INFO = {
    3: {"nombre": "Permanente",           "color": "#0b4f8a", "desc": "Prácticamente siempre hubo agua en este sector (occurrence ≥ 75%)."},
    2: {"nombre": "Estacional confiable", "color": "#3b82c4", "desc": "El agua aparece y desaparece, pero vuelve casi todos los años (occurrence 25-75% y recurrence ≥ 50%)."},
    1: {"nombre": "Esporádico",           "color": "#a8c8e6", "desc": "Hubo agua alguna vez en este sector, pero no es un comportamiento habitual."},
}

# Niveles de la clasificación por SEASONALITY (JRC GSW v1.5 2024).
# IMPORTANTE: seasonality es un CONTEO de meses/año con agua (0-12), no un
# calendario — no indica EN QUÉ meses hubo agua, solo cuántos. Por eso esta
# clasificación describe "confiabilidad estacional", no una ventana de fechas.
NIVEL_INFO_SEASONALIDAD = {
    4: {"nombre": "Permanente",       "color": "#0b4f8a", "desc": "Agua presente los 12 meses del año en el registro histórico."},
    3: {"nombre": "Casi permanente",  "color": "#2f76b0", "desc": "Agua presente entre 9 y 11 meses al año en promedio."},
    2: {"nombre": "Estacional",       "color": "#5fa3d6", "desc": "Agua presente entre 4 y 8 meses al año: comportamiento estacional marcado."},
    1: {"nombre": "Esporádico",       "color": "#a8c8e6", "desc": "Agua presente solo entre 1 y 3 meses al año: muy poco confiable como fuente estable."},
}


def _bbox_desde_punto(lat, lon, buffer_km):
    """Bounding box aproximado (grados) alrededor de un punto, dado un buffer en km."""
    dlat = buffer_km / 111.0
    dlon = buffer_km / (111.0 * max(0.15, np.cos(np.radians(lat))))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)  # west, south, east, north


def _leer_ventana(url, bbox):
    """Abre el COG remoto y lee solo la ventana que cubre el bbox pedido."""
    with rasterio.open(url) as src:
        window = from_bounds(*bbox, transform=src.transform)
        window = window.round_offsets().round_lengths()
        limite = Window(0, 0, src.width, src.height)
        window = window.intersection(limite)
        if window.width <= 0 or window.height <= 0:
            return None, None
        data = src.read(1, window=window)
        transform = src.window_transform(window)
        return data, transform


def _clasificar(occ, rec):
    """
    Reglas explícitas (no fórmula ponderada):
      3 = Permanente          -> occurrence >= 75
      2 = Estacional confiable -> 25 <= occurrence < 75  Y  recurrence >= 50
      1 = Esporádico           -> hay algo de señal de agua pero no cumple lo anterior
      0 = Sin señal relevante
    """
    valido = ~np.isin(occ, NODATA_VALORES) & ~np.isin(rec, NODATA_VALORES)

    clase = np.zeros(occ.shape, dtype=np.uint8)
    permanente = valido & (occ >= 75)
    estacional = valido & (occ >= 25) & (occ < 75) & (rec >= 50)
    esporadico = valido & (occ >= 1) & ~permanente & ~estacional

    clase[permanente] = 3
    clase[estacional] = 2
    clase[esporadico] = 1
    return clase


def _clasificar_estacionalidad(seasonality):
    """
    Reglas explícitas sobre el conteo de meses/año con agua (1-12):
      4 = Permanente      -> 12 meses
      3 = Casi permanente -> 9-11 meses
      2 = Estacional      -> 4-8 meses
      1 = Esporádico      -> 1-3 meses
      0 = Sin señal (nodata: 0 o 255, "tierra firme"/"sin observación")
    """
    valido = ~np.isin(seasonality, NODATA_VALORES)

    clase = np.zeros(seasonality.shape, dtype=np.uint8)
    permanente = valido & (seasonality >= 12)
    casi_permanente = valido & (seasonality >= 9) & (seasonality < 12)
    estacional = valido & (seasonality >= 4) & (seasonality < 9)
    esporadico = valido & (seasonality >= 1) & (seasonality < 4)

    clase[permanente] = 4
    clase[casi_permanente] = 3
    clase[estacional] = 2
    clase[esporadico] = 1
    return clase


def estacionalidad_agua(lat, lon, buffer_km=5.0):
    """
    Devuelve un GeoJSON FeatureCollection con los contornos de confiabilidad
    estacional del agua (SEASONALITY, JRC GSW v1.5 2024) alrededor de un
    punto. Cada Feature: properties = {nivel, nombre, color, desc}.

    ADVERTENCIA: seasonality es un conteo de meses/año con agua (0-12), NO
    un calendario — no indica en qué meses del año hubo agua. Este endpoint
    responde "¿qué tan confiable es esta fuente en general?", no "¿va a
    haber agua en marzo?". Para eso último haría falta el producto Monthly
    History del JRC (no disponible en este servidor).
    """
    bbox = _bbox_desde_punto(lat, lon, buffer_km)

    try:
        seasonality, transform = _leer_ventana(SEASONALITY_URL, bbox)
    except Exception as e:
        return {"ok": False, "error": f"no se pudo leer el raster JRC (seasonality): {e}"}

    if seasonality is None or seasonality.size == 0:
        return {"ok": False, "error": "sin datos JRC para este punto (fuera de cobertura del raster nacional)"}

    clase = _clasificar_estacionalidad(seasonality)

    features = []
    for nivel in (4, 3, 2, 1):
        mascara = (clase == nivel)
        if not mascara.any():
            continue
        geoms = [
            shape(geom)
            for geom, val in shapes(mascara.astype(np.uint8), mask=mascara, transform=transform)
            if val == 1
        ]
        if not geoms:
            continue
        geom_unida = unary_union(geoms).simplify(0.0003, preserve_topology=True)
        info = NIVEL_INFO_SEASONALIDAD[nivel]
        features.append({
            "type": "Feature",
            "geometry": mapping(geom_unida),
            "properties": {"nivel": nivel, **info},
        })

    return {
        "ok": True,
        "fuente": "JRC Global Surface Water v1.5 (2024) — seasonality",
        "nota": ("seasonality es un conteo de meses/año con agua (0-12), no un calendario: "
                 "no indica en qué meses hubo agua, solo cuán confiable es la fuente en general."),
        "buffer_km": buffer_km,
        "geojson": {"type": "FeatureCollection", "features": features},
    }


def envoltura_historica(lat, lon, buffer_km=5.0):
    """
    Devuelve un GeoJSON FeatureCollection con los contornos de comportamiento
    histórico del agua (occurrence + recurrence, JRC GSW v1.5 2024) alrededor
    de un punto. Cada Feature: properties = {nivel, nombre, color, desc}.
    """
    bbox = _bbox_desde_punto(lat, lon, buffer_km)

    try:
        occ, transform = _leer_ventana(OCCURRENCE_URL, bbox)
        rec, _ = _leer_ventana(RECURRENCE_URL, bbox)
    except Exception as e:
        return {"ok": False, "error": f"no se pudo leer el raster JRC: {e}"}

    if occ is None or rec is None or occ.size == 0:
        return {"ok": False, "error": "sin datos JRC para este punto (fuera de cobertura del raster nacional)"}

    if occ.shape != rec.shape:
        h = min(occ.shape[0], rec.shape[0])
        w = min(occ.shape[1], rec.shape[1])
        occ, rec = occ[:h, :w], rec[:h, :w]

    clase = _clasificar(occ, rec)

    features = []
    for nivel in (3, 2, 1):
        mascara = (clase == nivel)
        if not mascara.any():
            continue
        geoms = [
            shape(geom)
            for geom, val in shapes(mascara.astype(np.uint8), mask=mascara, transform=transform)
            if val == 1
        ]
        if not geoms:
            continue
        geom_unida = unary_union(geoms).simplify(0.0003, preserve_topology=True)
        info = NIVEL_INFO[nivel]
        features.append({
            "type": "Feature",
            "geometry": mapping(geom_unida),
            "properties": {"nivel": nivel, **info},
        })

    return {
        "ok": True,
        "fuente": "JRC Global Surface Water v1.5 (2024) — occurrence + recurrence",
        "buffer_km": buffer_km,
        "geojson": {"type": "FeatureCollection", "features": features},
    }
