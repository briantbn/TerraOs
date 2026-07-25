# ════════════════════════════════════════════════════════════════
#  NUEVO MÓDULO OPCIONAL: enso_oni
#  Pegar este bloque junto a los otros "MÓDULO OPCIONAL" (cerca de
#  jrc_memoria_hidrica), y el endpoint junto a /memoria_hidrica_punto.
# ════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────────
# MÓDULO OPCIONAL: enso_oni
# ──────────────────────────────────────────────────────────────────────────
try:
    import enso_oni
    _enso_disponible = True
except ImportError as _exc_enso:
    enso_oni = None
    _enso_disponible = False
    print(f'⚠️ Módulo enso_oni no disponible ({_exc_enso}). '
          f'Revisá que enso_oni.py esté desplegado junto a app.py '
          f'y que "requests" esté en requirements.txt. '
          f'/enso_actual devolverá error hasta que se resuelva.')


@app.route('/enso_actual')
def enso_actual():
    if not _enso_disponible:
        return jsonify({
            'error': ('Módulo enso_oni no disponible en el servidor. '
                      'Subí enso_oni.py junto a app.py y volvé a desplegar.'),
        }), 503
    return jsonify(enso_oni.consultar_oni())
