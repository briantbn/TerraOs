"""
Capa de inundación histórica REAL (no simulada) para los 3 escenarios
calibrados: 1998, 2013, Abril 2026.
════════════════════════════════════════════════════════════════════
Reemplaza el render vía FloodSimulationEngine/HAND SOLO para estos 3
botones. El motor HAND sigue siendo correcto para "Nivel actual" o
cualquier subida ingresada a mano (ahí SÍ tiene sentido simular "qué
pasaría si..."), pero para un escenario histórico ya ocurrido no hace
falta simular nada: hay que MOSTRAR lo que el satélite vio de verdad.

Fuentes (mismas que ya usás para calibrar el valor en metros, ahora
también usadas para el polígono):
  - 1998, 2013 -> JRC Global Surface Water, YearlyHistory (30m).
    Clases de water_class: 0=sin dato, 1=no-agua, 2=agua estacional,
    3=agua permanente. Para una crecida histórica conviene incluir
    2 y 3 (agua estacional captura bien la extensión de la creciente,
    que en años normales no está inundada).
  - Abril 2026 -> Sentinel-1 SAR (GRD), banda VH, umbral de
    backscatter. El radar atraviesa nubes, por eso sirve para eventos
    recientes puntuales (a diferencia de JRC, que es agregado anual).

Requiere: earthengine-api ya inicializado en el proceso (mismo patrón
que el resto de app.py).
"""

import ee

# ────────────────────────────────────────────────────────────────
# Ajustar estos 2 valores si tu calibración usó un año/rango de
# fechas distinto para cada evento (ver de dónde salió el "8.39 m"
# / "7.20 m" / "2.83 m" en tu calibración actual y usar la MISMA
# fuente/fecha acá, para que el número y el dibujo sean consistentes).
# ────────────────────────────────────────────────────────────────
_JRC_ANIO_POR_ESCENARIO = {
    "1998": 1998,
    "2013": 2013,
}

# Rango de fechas real de la crecida de abril 2026 (ajustar a la
# fecha exacta del pico si la sabés con precisión — cuanto más
# angosto el rango, más fiel es la imagen SAR al momento del pico).
_S1_RANGO_ABRIL_2026 = ("2026-04-01", "2026-04-30")


def mascara_jrc_yearly(anio, geometria):
    """Máscara binaria (1=agua) de JRC GSW YearlyHistory para un año
    específico, recortada a la geometría (círculo de radio o polígono
    delimitado). Incluye agua estacional + permanente."""
    img = ee.Image(f"JRC/GSW1_4/YearlyHistory/{anio}").select("waterClass")
    agua = img.gte(2).And(img.lte(3))  # 2=estacional, 3=permanente
    return agua.selfMask().clip(geometria)


def mascara_sentinel1(fecha_inicio, fecha_fin, geometria, umbral_db=-17):
    """Máscara binaria (1=agua) a partir de backscatter Sentinel-1 VH.
    Agua abierta da backscatter muy bajo (superficie lisa = reflexión
    especular, poca energía vuelve al sensor) -> umbral típico entre
    -15 y -20 dB según rugosidad del agua (viento/oleaje). Ajustar
    `umbral_db` si el resultado queda con ruido o corta de más."""
    coleccion = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(geometria)
        .filterDate(fecha_inicio, fecha_fin)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select("VH")
    )
    mediana = coleccion.median()  # reduce ruido speckle y huecos de una sola pasada
    agua = mediana.lt(umbral_db)
    return agua.selfMask().clip(geometria)


def capa_inundacion_historica_real(escenario, geometria):
    """`escenario` en {'1998', '2013', 'abril_2026'}. Devuelve un
    ee.Image binario recortado a la geometría, listo para generar
    tile_url igual que ya hacés con las capas HAND actuales."""
    if escenario in _JRC_ANIO_POR_ESCENARIO:
        return mascara_jrc_yearly(_JRC_ANIO_POR_ESCENARIO[escenario], geometria)
    if escenario == "abril_2026":
        inicio, fin = _S1_RANGO_ABRIL_2026
        return mascara_sentinel1(inicio, fin, geometria)
    raise ValueError(f"Escenario histórico desconocido: {escenario}")


# ────────────────────────────────────────────────────────────────
# Integración sugerida en app.py (endpoint separado, no reemplaza
# /inundacion_tiles genérico que sigue usando HAND para "nivel
# actual" o valores manuales):
#
#   from flood_historico_real import capa_inundacion_historica_real
#
#   @app.route('/inundacion_historica_real')
#   def inundacion_historica_real():
#       escenario = request.args.get('escenario')  # '1998'|'2013'|'abril_2026'
#       geom = _region_poligono_desde_query(request) or ee.Geometry.Point(
#           [lon, lat]).buffer(radio_km * 1000)
#       try:
#           img = capa_inundacion_historica_real(escenario, geom)
#           tile = img.getMapId({'palette': ['#4a90d9'], 'opacity': 0.55})
#           return jsonify({'tile_url': tile['tile_fetcher'].url_format,
#                            'fuente': 'satelital_real', 'escenario': escenario})
#       except Exception as exc:
#           return jsonify({'error': str(exc)}), 500
#
# Frontend: cuando el usuario aprieta uno de los 3 botones de
# escenario histórico, llamar a este endpoint en vez de al genérico
# de simulación HAND. Los demás controles (nivel actual, subida
# manual en metros) siguen yendo por el flujo HAND existente sin
# cambios.
# ════════════════════════════════════════════════════════════════
