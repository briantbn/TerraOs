# ════════════════════════════════════════════════════════════════
# PEGAR ESTO EN app.py, reemplazando el endpoint /suelo_mejor_resolucion
# que ya existe (el que llama a consultar_mejor_capa de
# suelos_inta_regional.py). Requiere que suelo_aptitud_cascada.py esté
# también en el repo y desplegado junto a app.py.
# ════════════════════════════════════════════════════════════════

from suelos_inta_regional import consultar_mejor_capa
# Import defensivo: si el módulo de respaldo no está desplegado (o
# rasterio/pyproj no están instalados en el entorno), la app sigue
# funcionando SOLO con polígonos, sin caerse — mismo patrón defensivo
# que ya usás para los demás módulos opcionales (suelos_inta_regional,
# hidrografia_vectorial, etc.).
try:
    from suelo_aptitud_cascada import consultar_aptitud_cascada
    _COG_RESPALDO_DISPONIBLE = True
except Exception as _exc:
    print(f'⚠️ [app] Respaldo COG de aptitud (suelo_aptitud_cascada) no disponible: {_exc}')
    _COG_RESPALDO_DISPONIBLE = False


@app.route('/suelo_mejor_resolucion')
def suelo_mejor_resolucion():
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    if lat is None or lon is None:
        return jsonify({'error': 'Faltan parámetros lat/lon'}), 400

    # 1) Fuente principal: polígonos INTA reales (Santa Fe/Córdoba/
    #    Buenos Aires/NOA/Nacional por provincia).
    resultado = consultar_mejor_capa(lat, lon)

    # 2) Respaldo: SOLO si el punto no cayó en ningún polígono de
    #    ninguna de las capas anteriores (ni siquiera la Nacional
    #    1:1.000.000, que ya es el último recurso de ese sistema).
    #    No se usa para "mejorar" un resultado que ya vino con dato.
    if not resultado.get('encontrado') and _COG_RESPALDO_DISPONIBLE:
        try:
            resultado_cog = consultar_aptitud_cascada(lat, lon)
        except Exception as exc:
            print(f'⚠️ [app] Error consultando respaldo COG: {exc}')
            resultado_cog = {'encontrado': False}
        if resultado_cog.get('encontrado'):
            resultado = resultado_cog

    return jsonify(resultado)
