# ════════════════════════════════════════════════════════════════
#  NUEVO ENDPOINT: /historial_hidrologico
#  Suma los 6 indicadores de JRC Global Surface Water que hoy no se
#  usaban (occurrence ya se usa en otro lado; el resto son nuevos):
#    - occurrence     (aparición / frecuencia histórica de agua, %)
#    - change_norm    (intensidad del cambio de ocurrencia, %)
#    - recurrence     (reaparición año a año, %)
#    - seasonality    (estacionalidad: meses de agua por año, 0-12)
#    - transition     (categoría de transición 1-10, ver TRANSICION_LABELS)
#    - max_extent     (extensión máxima de agua alguna vez detectada, 0/1)
#
#  Pegar junto a los otros @app.route(...) del archivo. No modifica
#  ningún endpoint existente.
# ════════════════════════════════════════════════════════════════

TRANSICION_LABELS = {
    0: 'Sin datos', 1: 'Agua permanente (sin cambios)', 2: 'Nueva agua permanente',
    3: 'Agua permanente perdida', 4: 'Agua estacional (sin cambios)', 5: 'Nueva agua estacional',
    6: 'Agua estacional perdida', 7: 'Estacional pasó a permanente', 8: 'Permanente pasó a estacional',
    9: 'Permanente efímera', 10: 'Estacional efímera',
}


def _indicadores_gsw_punto(lat, lon, radius_m=200):
    """Consulta los 6 indicadores de JRC GSW1_4 promediados en un buffer
    chico alrededor del punto (o de un cuerpo de agua). Devuelve None si
    Earth Engine no tiene datos ahí (ej. fuera de cobertura del dataset)."""
    try:
        punto = ee.Geometry.Point([lon, lat]).buffer(radius_m)
        gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')

        stats_continuas = (gsw.select(['occurrence', 'change_abs', 'change_norm', 'seasonality', 'recurrence', 'max_extent'])
                            .reduceRegion(reducer=ee.Reducer.mean(), geometry=punto, scale=30,
                                          maxPixels=1e9, bestEffort=True)
                            .getInfo())
        # 'transition' es categórico (1-10): la media no sirve, se usa la moda.
        transicion_val = (gsw.select('transition')
                           .reduceRegion(reducer=ee.Reducer.mode(), geometry=punto, scale=30,
                                         maxPixels=1e9, bestEffort=True)
                           .getInfo())

        if not stats_continuas or stats_continuas.get('occurrence') is None:
            return None  # sin cobertura de GSW en este punto (puede pasar tierra adentro sin agua nunca)

        transition_code = int(round(transicion_val.get('transition', 0) or 0))

        # ── Indicadores derivados (pedidos explícitamente) ──
        occurrence = stats_continuas.get('occurrence', 0) or 0
        recurrence = stats_continuas.get('recurrence', 0) or 0
        seasonality = stats_continuas.get('seasonality', 0) or 0
        change_norm = stats_continuas.get('change_norm', 0) or 0
        max_extent = stats_continuas.get('max_extent', 0) or 0

        memoria_hidrologica = round((occurrence + recurrence + max_extent * 100) / 3, 1)
        persistencia_agua = round((occurrence + (seasonality / 12 * 100)) / 2, 1)
        tendencia = 'creciendo' if change_norm > 5 else ('disminuyendo' if change_norm < -5 else 'estable')

        return {
            'occurrence': round(occurrence, 1),
            'change_abs': round(stats_continuas.get('change_abs', 0) or 0, 2),
            'change_norm': round(change_norm, 1),
            'seasonality_meses': round(seasonality, 1),
            'recurrence': round(recurrence, 1),
            'max_extent': bool(max_extent >= 0.5),
            'transition_code': transition_code,
            'transition_label': TRANSICION_LABELS.get(transition_code, 'Sin datos'),
            'memoria_hidrologica': memoria_hidrologica,
            'persistencia_agua': persistencia_agua,
            'tendencia_hidrologica': tendencia,
        }
    except Exception as e:
        print(f"[historial_hidrologico] Error: {e}")
        return None


@app.route('/historial_hidrologico')
def historial_hidrologico():
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    if lat is None or lon is None:
        return jsonify({'error': 'faltan lat/lon'}), 400

    datos = _indicadores_gsw_punto(lat, lon)
    if datos is None:
        return jsonify({'error': 'sin cobertura de JRC Global Surface Water en este punto', 'ok': False}), 200

    datos['ok'] = True
    return jsonify(datos)
