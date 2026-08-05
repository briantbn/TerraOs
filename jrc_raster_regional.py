"""
jrc_raster_regional.py
=======================

Envoltura histórica del comportamiento del agua, a partir de los rasters
oficiales JRC Global Surface Water v1.5 (2024) — occurrence, recurrence,
seasonality, transitions, change_abs y extent — alojados como
Cloud-Optimized GeoTIFF (COG) en Hugging Face.

Lee SOLO la ventana necesaria alrededor de un punto (no descarga el archivo
completo en cada consulta, gracias al formato COG + lectura por HTTP range).

No inventa índices: usa directamente occurrence (%), recurrence (%),
seasonality (meses/año), transitions (categoría de cambio 1984-2024),
change_abs (diferencia de occurrence entre épocas) y extent (máxima
superficie histórica, binario), las variables oficiales del JRC, combinadas
con reglas lógicas explícitas (no una fórmula ponderada ni un puntaje
compuesto).

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
TRANSITIONS_URL = "/vsicurl/https://huggingface.co/datasets/Briant97/GeoSentinel/resolve/main/transitions_ar_cog.tif"
# El archivo "change" del JRC trae 2 bandas: banda 1 = change_norm, banda 2 =
# change_abs. Usamos change_abs (diferencia directa en puntos de occurrence
# entre épocas 1984-1999 y 2000-2024) por ser más fácil de interpretar.
CHANGE_URL = "/vsicurl/https://huggingface.co/datasets/Briant97/GeoSentinel/resolve/main/change_ar_cog.tif"
CHANGE_BANDA_ABS = 2

# "extent" (maximum water extent) es binario: 0 = nunca hubo agua, 1 =
# alguna vez ocupó ese píxel en 40 años. Confirmado con gdallocationinfo
# sobre el centro de la Laguna Mar Chiquita (Córdoba): Value=1. (No es 255,
# a diferencia de otras capas del JRC — cada producto trae su propia
# codificación y no hay que asumirla sin comprobarla.)
EXTENT_URL = "/vsicurl/https://huggingface.co/datasets/Briant97/GeoSentinel/resolve/main/extent_ar_cog.tif"
EXTENT_VALOR_SI = 1

# El archivo "change" es Int8 (rango -128 a 127), a diferencia de las otras
# capas (Byte, 0-255). No trae NoData grabado en el archivo original; -128
# es el valor centinela típico de Int8 y quedó confirmado empíricamente al
# fusionar los tiles (ver notas del proceso). 0 SÍ es un valor válido aquí
# ("sin cambio"), a diferencia de NODATA_VALORES que se usa en las otras capas.
NODATA_CHANGE = (-128,)

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

# Códigos oficiales JRC GSW v1.5 (2024) para TRANSITIONS (cambio 1984→2024):
#   1 Permanente estable   2 Nuevo permanente      3 Permanente perdido
#   4 Estacional estable   5 Nuevo estacional      6 Estacional perdido
#   7 Estacional→Permanente 8 Permanente→Estacional
#   9 Efímero (antes permanente)  10 Efímero (antes estacional)
TRANSITIONS_GANO_AGUA = (2, 5, 7)      # pasó a tener agua donde antes no (o menos)
TRANSITIONS_PERDIO_AGUA = (3, 6, 8)     # tenía agua y la perdió/redujo
TRANSITIONS_EFIMERO = (9, 10)           # osciló sin consolidarse
TRANSITIONS_ESTABLE = (1, 4)            # sin cambio de fondo

# Niveles de la clasificación cruzada TRANSITIONS + OCCURRENCE (combo 5).
# No es una capa nueva "sobre" transitions: es transitions interpretado con
# el contexto histórico de occurrence, para distinguir expansión esperable
# de humedal natural vs. inundación anómala sobre terreno que casi nunca
# tuvo agua en 40 años.
NIVEL_INFO_HUMEDAL = {
    1: {"nombre": "Posible anómalo",       "color": "#dc2626", "desc": "Ganó agua donde el registro histórico dice que casi nunca la tuvo (occurrence < 25%): posible falla de infraestructura u obra mal drenada, no un humedal natural."},
    2: {"nombre": "Expansión esperable",   "color": "#f97316", "desc": "Ganó agua, pero ya tenía historial previo de agua en la zona (occurrence ≥ 25%): comportamiento consistente con un humedal natural que se expande."},
    3: {"nombre": "Inestable / efímero",   "color": "#9333ea", "desc": "Osciló entre agua y tierra firme sin consolidarse en ningún sentido durante el período analizado."},
    4: {"nombre": "Retracción / pérdida",  "color": "#1f2937", "desc": "Perdió agua o redujo su condición (de permanente a estacional, o directamente desapareció)."},
    5: {"nombre": "Estable",               "color": "#16a34a", "desc": "Sin cambios de fondo: el comportamiento del agua se mantuvo igual entre el inicio y el fin del período analizado."},
}

# Niveles de la clasificación por CHANGE_ABS (JRC GSW, diferencia directa de
# occurrence en puntos porcentuales entre la época 1984-1999 y 2000-2024).
# Combo "alerta temprana de degradación de aguada": a diferencia de
# occurrence/recurrence/seasonality (que describen un ESTADO), esta es la
# única capa que mide TENDENCIA — una aguada puede seguir clasificando como
# confiable hoy y aun así venir en caída sostenida.
NIVEL_INFO_CAMBIO = {
    1: {"nombre": "Retracción severa",     "color": "#7f1d1d", "desc": "Perdió más de 50 puntos de occurrence entre épocas: caída fuerte, alerta temprana de degradación."},
    2: {"nombre": "Retracción moderada",   "color": "#dc2626", "desc": "Perdió entre 20 y 50 puntos de occurrence: tendencia negativa a vigilar."},
    3: {"nombre": "Estable",               "color": "#6b7280", "desc": "Cambio menor a 20 puntos de occurrence en cualquier sentido: sin tendencia clara."},
    4: {"nombre": "Expansión moderada",    "color": "#22c55e", "desc": "Ganó entre 20 y 50 puntos de occurrence: tendencia positiva."},
    5: {"nombre": "Expansión fuerte",      "color": "#15803d", "desc": "Ganó más de 50 puntos de occurrence entre épocas: crecimiento marcado."},
}

# Niveles del combo EXTENT + OCCURRENCE ("chequeo de sitio antes de
# construir"): distingue un lugar que hoy se ve seco pero llegó a mojarse
# alguna vez en 40 años (el caso peligroso, porque parece seguro a simple
# vista) de uno que ni siquiera entra en el extent máximo histórico.
NIVEL_INFO_EXTENT = {
    1: {"nombre": "Riesgo oculto",  "color": "#b91c1c", "desc": "Entra en el área máxima que el agua llegó a ocupar alguna vez (extent), pero occurrence es bajo (<25%): parece seco casi siempre, pero ya se mojó en un evento extremo — riesgo no evidente a simple vista."},
    2: {"nombre": "Zona activa",    "color": "#2563eb", "desc": "Entra en el extent máximo histórico Y occurrence es alto (≥25%): zona de comportamiento de agua conocido y frecuente, no una sorpresa."},
    3: {"nombre": "Fuera de extent", "color": "#9ca3af", "desc": "Nunca, en 40 años de registro satelital, el agua llegó a ocupar este punto."},
}


# Códigos oficiales JRC GSW v1.5 (2024) para TRANSITIONS, expuestos SIN
# agrupar (a diferencia de NIVEL_INFO_HUMEDAL, que los cruza con occurrence
# y los agrupa en 5 buckets tipo gano/perdió/efímero/estable). Acá se
# muestra la categoría real tal cual la clasifica JRC — más fino, permite
# ver por ejemplo que una laguna pasó de estacional a permanente (se está
# consolidando) sin que eso se note en change_abs (que mide magnitud de
# occurrence, no tipo de comportamiento — un cambio 7 u 8 puede tener
# change_abs chico y aun así ser una transición de fondo importante).
NIVEL_INFO_TRANSICIONES = {
    1:  {"nombre": "Permanente estable",        "color": "#0b4f8a", "desc": "Agua permanente en 1984 y sigue siendo permanente hoy — sin cambio de fondo."},
    2:  {"nombre": "Nuevo permanente",          "color": "#0891b2", "desc": "No había agua permanente en 1984 y hoy sí la hay — se consolidó como fuente estable nueva."},
    3:  {"nombre": "Permanente perdido",        "color": "#7f1d1d", "desc": "Era agua permanente en 1984 y dejó de serlo — señal fuerte de degradación de la fuente."},
    4:  {"nombre": "Estacional estable",        "color": "#3b82c4", "desc": "Agua estacional en 1984 y sigue siendo estacional hoy — sin cambio de fondo."},
    5:  {"nombre": "Nuevo estacional",          "color": "#10b981", "desc": "No había señal de agua estacional en 1984 y hoy sí la hay — comportamiento nuevo, a monitorear."},
    6:  {"nombre": "Estacional perdido",        "color": "#dc2626", "desc": "Era agua estacional en 1984 y dejó de comportarse así — perdió el patrón que tenía."},
    7:  {"nombre": "Estacional → Permanente",   "color": "#16a34a", "desc": "Se está CONSOLIDANDO: pasó de estacional a permanente. Es un cambio de TIPO de comportamiento, puede no notarse en change_abs si la magnitud de occurrence no varió tanto."},
    8:  {"nombre": "Permanente → Estacional",   "color": "#f97316", "desc": "Se está DEGRADANDO: pasó de permanente a estacional, aunque no lo veas reflejado en change_abs (que mide magnitud, no tipo de cambio) — vale la pena cruzar con /degradacion_aguada."},
    9:  {"nombre": "Efímero (era permanente)",  "color": "#9333ea", "desc": "Era agua permanente y ahora aparece de forma efímera/inconsistente — degradación fuerte, perdió toda regularidad."},
    10: {"nombre": "Efímero (era estacional)",  "color": "#c084fc", "desc": "Era agua estacional y ahora aparece de forma efímera — perdió la regularidad que tenía, aunque nunca fue una fuente permanente."},
}


def _bbox_desde_punto(lat, lon, buffer_km):
    """Bounding box aproximado (grados) alrededor de un punto, dado un buffer en km."""
    dlat = buffer_km / 111.0
    dlon = buffer_km / (111.0 * max(0.15, np.cos(np.radians(lat))))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)  # west, south, east, north


def _leer_ventana(url, bbox, banda=1):
    """Abre el COG remoto y lee solo la ventana que cubre el bbox pedido."""
    with rasterio.open(url) as src:
        window = from_bounds(*bbox, transform=src.transform)
        window = window.round_offsets().round_lengths()
        limite = Window(0, 0, src.width, src.height)
        window = window.intersection(limite)
        if window.width <= 0 or window.height <= 0:
            return None, None
        data = src.read(banda, window=window)
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


def _clasificar_humedal(transitions, occ):
    """
    Reglas explícitas, cruzando transitions (categoría de cambio 1984-2024)
    con occurrence (contexto histórico, umbral 25% igual que en _clasificar):
      1 = Posible anómalo      -> ganó agua Y occurrence < 25
      2 = Expansión esperable  -> ganó agua Y occurrence >= 25
      3 = Inestable / efímero  -> transición efímera (9 o 10)
      4 = Retracción / pérdida -> perdió agua
      5 = Estable               -> sin cambio de fondo (1 o 4)
      0 = Sin señal (nodata en transitions o en occurrence)
    """
    valido = ~np.isin(transitions, NODATA_VALORES) & ~np.isin(occ, NODATA_VALORES) \
             & np.isin(transitions, range(1, 11))

    clase = np.zeros(transitions.shape, dtype=np.uint8)
    gano = valido & np.isin(transitions, TRANSITIONS_GANO_AGUA)
    anomalo = gano & (occ < 25)
    expansion = gano & (occ >= 25)
    efimero = valido & np.isin(transitions, TRANSITIONS_EFIMERO)
    perdio = valido & np.isin(transitions, TRANSITIONS_PERDIO_AGUA)
    estable = valido & np.isin(transitions, TRANSITIONS_ESTABLE)

    clase[anomalo] = 1
    clase[expansion] = 2
    clase[efimero] = 3
    clase[perdio] = 4
    clase[estable] = 5
    return clase


def humedal_vs_anomalia(lat, lon, buffer_km=5.0):
    """
    Devuelve un GeoJSON FeatureCollection que distingue, alrededor de un
    punto, "humedal natural (expansión esperable)" de "posible inundación
    anómala" — cruzando TRANSITIONS (cambio 1984→2024) con OCCURRENCE
    (contexto histórico), ambos JRC GSW v1.5 (2024). Cada Feature:
    properties = {nivel, nombre, color, desc}.
    """
    bbox = _bbox_desde_punto(lat, lon, buffer_km)

    try:
        transitions, transform = _leer_ventana(TRANSITIONS_URL, bbox)
        occ, _ = _leer_ventana(OCCURRENCE_URL, bbox)
    except Exception as e:
        return {"ok": False, "error": f"no se pudo leer el raster JRC (transitions/occurrence): {e}"}

    if transitions is None or occ is None or transitions.size == 0:
        return {"ok": False, "error": "sin datos JRC para este punto (fuera de cobertura del raster nacional)"}

    if transitions.shape != occ.shape:
        h = min(transitions.shape[0], occ.shape[0])
        w = min(transitions.shape[1], occ.shape[1])
        transitions, occ = transitions[:h, :w], occ[:h, :w]

    clase = _clasificar_humedal(transitions, occ)

    features = []
    for nivel in (1, 2, 3, 4, 5):
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
        info = NIVEL_INFO_HUMEDAL[nivel]
        features.append({
            "type": "Feature",
            "geometry": mapping(geom_unida),
            "properties": {"nivel": nivel, **info},
        })

    return {
        "ok": True,
        "fuente": "JRC Global Surface Water v1.5 (2024) — transitions + occurrence",
        "buffer_km": buffer_km,
        "geojson": {"type": "FeatureCollection", "features": features},
    }


def _clasificar_cambio(change_abs):
    """
    Reglas explícitas sobre change_abs (puntos de occurrence, -100 a 100):
      1 = Retracción severa   -> <= -50
      2 = Retracción moderada -> -50 < x <= -20
      3 = Estable             -> -20 < x < 20
      4 = Expansión moderada  -> 20 <= x < 50
      5 = Expansión fuerte    -> >= 50
      0 = Sin señal (nodata: -128)
    """
    valido = ~np.isin(change_abs, NODATA_CHANGE)

    clase = np.zeros(change_abs.shape, dtype=np.uint8)
    retraccion_severa = valido & (change_abs <= -50)
    retraccion_moderada = valido & (change_abs > -50) & (change_abs <= -20)
    estable = valido & (change_abs > -20) & (change_abs < 20)
    expansion_moderada = valido & (change_abs >= 20) & (change_abs < 50)
    expansion_fuerte = valido & (change_abs >= 50)

    clase[retraccion_severa] = 1
    clase[retraccion_moderada] = 2
    clase[estable] = 3
    clase[expansion_moderada] = 4
    clase[expansion_fuerte] = 5
    return clase


def degradacion_aguada(lat, lon, buffer_km=5.0):
    """
    Devuelve un GeoJSON FeatureCollection con la tendencia de cambio del
    agua (CHANGE_ABS, JRC GSW) alrededor de un punto — a diferencia de las
    otras funciones de este módulo, que describen un ESTADO histórico, esta
    mide TENDENCIA: alerta temprana de degradación (o mejora) de una fuente
    de agua, aunque su clasificación de estado actual (occurrence/recurrence)
    todavía luzca confiable. Cada Feature: properties = {nivel, nombre,
    color, desc}.
    """
    bbox = _bbox_desde_punto(lat, lon, buffer_km)

    try:
        change_abs, transform = _leer_ventana(CHANGE_URL, bbox, banda=CHANGE_BANDA_ABS)
    except Exception as e:
        return {"ok": False, "error": f"no se pudo leer el raster JRC (change_abs): {e}"}

    if change_abs is None or change_abs.size == 0:
        return {"ok": False, "error": "sin datos JRC para este punto (fuera de cobertura del raster nacional)"}

    clase = _clasificar_cambio(change_abs)

    features = []
    for nivel in (1, 2, 3, 4, 5):
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
        info = NIVEL_INFO_CAMBIO[nivel]
        features.append({
            "type": "Feature",
            "geometry": mapping(geom_unida),
            "properties": {"nivel": nivel, **info},
        })

    return {
        "ok": True,
        "fuente": "JRC Global Surface Water (2024) — change_abs (occurrence 1984-1999 vs. 2000-2024)",
        "buffer_km": buffer_km,
        "geojson": {"type": "FeatureCollection", "features": features},
    }


def _clasificar_extent(extent, occ):
    """
    Reglas explícitas, cruzando extent (binario) con occurrence:
      1 = Riesgo oculto    -> dentro del extent Y occurrence < 25
      2 = Zona activa      -> dentro del extent Y occurrence >= 25
      3 = Fuera de extent  -> nunca ocupado por agua en el registro
    (occ con nodata se trata como "bajo", ya que si nunca hubo observación
    de occurrence pero SÍ hay extent positivo, sigue siendo relevante avisar).
    """
    dentro_extent = (extent == EXTENT_VALOR_SI)
    occ_bajo = np.isin(occ, NODATA_VALORES) | (occ < 25)

    clase = np.zeros(extent.shape, dtype=np.uint8)
    clase[dentro_extent & occ_bajo] = 1
    clase[dentro_extent & ~occ_bajo] = 2
    clase[~dentro_extent] = 3
    return clase


def riesgo_oculto_extent(lat, lon, buffer_km=5.0):
    """
    Devuelve un GeoJSON FeatureCollection cruzando EXTENT (máxima superficie
    de agua alguna vez ocupada) con OCCURRENCE, para el combo "chequeo de
    sitio antes de construir": identifica lugares que hoy se ven secos casi
    siempre pero que SÍ llegaron a mojarse en algún evento extremo de los
    últimos 40 años — el caso más peligroso, porque parece seguro a simple
    vista. Cada Feature: properties = {nivel, nombre, color, desc}.
    """
    bbox = _bbox_desde_punto(lat, lon, buffer_km)

    try:
        extent, transform = _leer_ventana(EXTENT_URL, bbox)
        occ, _ = _leer_ventana(OCCURRENCE_URL, bbox)
    except Exception as e:
        return {"ok": False, "error": f"no se pudo leer el raster JRC (extent/occurrence): {e}"}

    if extent is None or occ is None or extent.size == 0:
        return {"ok": False, "error": "sin datos JRC para este punto (fuera de cobertura del raster nacional)"}

    if extent.shape != occ.shape:
        h = min(extent.shape[0], occ.shape[0])
        w = min(extent.shape[1], occ.shape[1])
        extent, occ = extent[:h, :w], occ[:h, :w]

    clase = _clasificar_extent(extent, occ)

    features = []
    for nivel in (1, 2):  # nivel 3 ("fuera de extent") no se dibuja: sería casi todo el buffer, sin valor visual
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
        info = NIVEL_INFO_EXTENT[nivel]
        features.append({
            "type": "Feature",
            "geometry": mapping(geom_unida),
            "properties": {"nivel": nivel, **info},
        })

    return {
        "ok": True,
        "fuente": "JRC Global Surface Water (2024) — extent + occurrence",
        "buffer_km": buffer_km,
        "geojson": {"type": "FeatureCollection", "features": features},
    }


def _clasificar_transiciones_detalle(transitions):
    """
    A diferencia de _clasificar_humedal() (que agrupa en 5 buckets), acá
    se usa DIRECTAMENTE el código oficial JRC (1-10) — no hay reglas que
    aplicar, solo filtrar nodata/valores fuera de rango.
    """
    valido = ~np.isin(transitions, NODATA_VALORES) & np.isin(transitions, list(range(1, 11)))
    return np.where(valido, transitions, 0).astype(np.uint8)


def transiciones_detalladas(lat, lon, buffer_km=5.0):
    """
    Devuelve un GeoJSON FeatureCollection con las 10 categorías REALES de
    TRANSITIONS (JRC GSW v1.5 2024) alrededor de un punto, sin agrupar —
    a diferencia de humedal_vs_anomalia() (combo 5), que cruza transitions
    con occurrence y las agrupa en 5 buckets tipo gano/perdió/efímero/
    estable. Acá se ve el diagnóstico fino: por ejemplo, distinguir
    "estacional → permanente" (se consolida) de "permanente → estacional"
    (se degrada) — dos cosas que humedal_vs_anomalia() mete juntas en el
    mismo bucket "gano/perdió agua" según corresponda, perdiendo el detalle
    de que un cambio de TIPO no siempre se nota en change_abs (que mide
    magnitud, no tipo de comportamiento).

    Cada Feature: properties = {nivel, nombre, color, desc}.
    """
    bbox = _bbox_desde_punto(lat, lon, buffer_km)

    try:
        transitions, transform = _leer_ventana(TRANSITIONS_URL, bbox)
    except Exception as e:
        return {"ok": False, "error": f"no se pudo leer el raster JRC (transitions): {e}"}

    if transitions is None or transitions.size == 0:
        return {"ok": False, "error": "sin datos JRC para este punto (fuera de cobertura del raster nacional)"}

    clase = _clasificar_transiciones_detalle(transitions)

    features = []
    for nivel in range(1, 11):
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
        info = NIVEL_INFO_TRANSICIONES[nivel]
        features.append({
            "type": "Feature",
            "geometry": mapping(geom_unida),
            "properties": {"nivel": nivel, **info},
        })

    return {
        "ok": True,
        "fuente": "JRC Global Surface Water v1.5 (2024) — transitions (categorías detalladas 1984→2024)",
        "nota": ("Estas son las 10 categorías oficiales de JRC sin agrupar. Para el diagnóstico "
                 "combinado humedal-natural-vs-anomalía (que sí las agrupa cruzando con occurrence), "
                 "ver /humedal_vs_anomalia."),
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
