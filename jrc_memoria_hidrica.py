"""
jrc_memoria_hidrica.py
─────────────────────────────────────────────────────────────────────────
Módulo opcional para GeoSentinel / app.py.

Calcula los 6 indicadores de JRC Global Surface Water (v1.4) que la app
todavía no usaba (occurrence ya se usa en otro endpoint) — aparición,
intensidad del cambio, reaparición, estacionalidad, transición y
extensión máxima — más 3 indicadores derivados: memoria hidrológica,
persistencia del agua y tendencia hidrológica.

NOTA TÉCNICA: se evaluó leer directamente los tiles públicos de
https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GSWE/Aggregated/LATEST/
con rasterio, pero esos archivos usan una grilla de píxeles globales
(nombre tipo change-0000000000-0000040000.tif) cuyo origen exacto no
está documentado con la precisión necesaria para georreferenciar sin
riesgo de error silencioso. En cambio, este módulo usa el MISMO dataset
(JRC/GSW1_4/GlobalSurfaceWater) vía Google Earth Engine, que ya está en
uso en otras partes de app.py y tiene la georreferenciación resuelta y
confiable. Requiere que 'earthengine-api' ya esté inicializado en el
proceso (mismo patrón que el resto del backend — no se hace ee.Initialize()
acá para no duplicar la autenticación).
"""

import ee

TRANSICION_LABELS = {
    0: 'Sin datos',
    1: 'Agua permanente (sin cambios)',
    2: 'Nueva agua permanente',
    3: 'Agua permanente perdida',
    4: 'Agua estacional (sin cambios)',
    5: 'Nueva agua estacional',
    6: 'Agua estacional perdida',
    7: 'Estacional pasó a permanente',
    8: 'Permanente pasó a estacional',
    9: 'Permanente efímera',
    10: 'Estacional efímera',
}

_GSW_ASSET = 'JRC/GSW1_4/GlobalSurfaceWater'


def consultar_memoria_hidrica(lat, lon, radius_m=200):
    """Punto de entrada usado por app.py (/memoria_hidrica_punto).

    Devuelve un dict con los 6 indicadores JRC GSW + 3 derivados, o
    {'ok': False, 'error': ...} si no hay cobertura del dataset en ese
    punto (puede pasar tierra adentro, lejos de cualquier cuerpo de agua
    detectado alguna vez por Landsat).
    """
    try:
        punto = ee.Geometry.Point([lon, lat]).buffer(radius_m)
        gsw = ee.Image(_GSW_ASSET)

        bandas_continuas = ['occurrence', 'change_abs', 'change_norm', 'seasonality', 'recurrence', 'max_extent']
        stats = (gsw.select(bandas_continuas)
                 .reduceRegion(reducer=ee.Reducer.mean(), geometry=punto, scale=30,
                               maxPixels=1e9, bestEffort=True)
                 .getInfo())

        if not stats or stats.get('occurrence') is None:
            return {'ok': False, 'error': 'Sin cobertura de JRC Global Surface Water en este punto.'}

        # 'transition' es categórica (códigos 1-10): la media no tiene sentido,
        # se usa la moda (valor más frecuente en el buffer).
        transicion_stats = (gsw.select('transition')
                             .reduceRegion(reducer=ee.Reducer.mode(), geometry=punto, scale=30,
                                           maxPixels=1e9, bestEffort=True)
                             .getInfo())
        transition_code = int(round((transicion_stats or {}).get('transition', 0) or 0))

        occurrence = stats.get('occurrence', 0) or 0
        recurrence = stats.get('recurrence', 0) or 0
        seasonality = stats.get('seasonality', 0) or 0
        change_norm = stats.get('change_norm', 0) or 0
        max_extent = stats.get('max_extent', 0) or 0

        # ── Indicadores derivados ──
        # Memoria hidrológica: qué tan acostumbrado está el terreno a tener agua
        # (combina frecuencia histórica, reaparición año a año y si alguna vez
        # llegó a estar cubierto de agua).
        memoria_hidrologica = round((occurrence + recurrence + max_extent * 100) / 3, 1)

        # Persistencia del agua: cuánto tiempo tiende a quedarse el agua una vez
        # que aparece (frecuencia + meses de estacionalidad sobre 12).
        persistencia_agua = round((occurrence + (seasonality / 12 * 100)) / 2, 1)

        # Tendencia hidrológica: si el cuerpo de agua está creciendo, estable o
        # achicándose con los años (según el signo de la intensidad del cambio).
        if change_norm > 5:
            tendencia = 'creciendo'
        elif change_norm < -5:
            tendencia = 'disminuyendo'
        else:
            tendencia = 'estable'

        return {
            'ok': True,
            'occurrence': round(occurrence, 1),
            'change_abs': round(stats.get('change_abs', 0) or 0, 2),
            'change_norm': round(change_norm, 1),
            'seasonality_meses': round(seasonality, 1),
            'recurrence': round(recurrence, 1),
            'max_extent': bool(max_extent >= 0.5),
            'transition_code': transition_code,
            'transition_label': TRANSICION_LABELS.get(transition_code, 'Sin datos'),
            'memoria_hidrologica': memoria_hidrologica,
            'persistencia_agua': persistencia_agua,
            'tendencia_hidrologica': tendencia,
            'fuente': 'JRC Global Surface Water v1.4 (Google Earth Engine)',
        }
    except Exception as e:
        return {'ok': False, 'error': f'Error consultando JRC GSW: {e}'}
