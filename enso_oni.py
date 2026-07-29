"""
enso_oni.py
─────────────────────────────────────────────────────────────────────────
Módulo opcional para GeoSentinel / app.py.

Consulta el ONI (Oceanic Niño Index) directo de la fuente oficial de
NOAA CPC — el mismo índice que usan los meteorólogos para declarar si
hay El Niño, La Niña o condición Neutral:
    https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt

Es un archivo de texto simple (SEAS YR TOTAL ANOM), actualizado una vez
por mes por NOAA. Este módulo lo descarga, toma la temporada más
reciente, y clasifica la fase e intensidad según los umbrales oficiales
de NOAA:
    ONI >= +0.5  → El Niño   (0.5-0.9 débil, 1.0-1.4 moderado,
                               1.5-1.9 fuerte, >=2.0 muy fuerte)
    ONI <= -0.5  → La Niña   (mismos umbrales, en negativo)
    -0.5 < ONI < 0.5 → Neutral

Cachea el resultado en memoria (NOAA solo actualiza 1 vez al mes, no
tiene sentido pedirlo en cada request).
"""

import time
import requests

_ONI_URL = 'https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt'
_CACHE_HORAS = 12  # NOAA actualiza ~1 vez al mes; refrescar cada 12h alcanza de sobra
_cache = {'datos': None, 'timestamp': 0}

# ─────────────────────────────────────────────────────────────────────────
# Tormenta de diseño por fase/intensidad ONI — Corrientes / NEA argentino
# ─────────────────────────────────────────────────────────────────────────
# Esto NO es una regresión estadística formal (no hay series de mm por
# evento cargadas en el sistema todavía) — es una tabla de referencia
# construida a partir de reportes públicos recientes, para dar una
# tormenta de entrada razonable al simulador dinámico (OverlandFlow)
# según la fase e intensidad del ONI. Fuentes consultadas:
#
#   • SMN (smn.gob.ar/las-precipitaciones): en el trimestre SON de 1997
#     (El Niño muy fuerte, ONI pico ~2.3) Misiones y Corrientes recibieron
#     ~500 mm POR ENCIMA de lo normal — el evento de referencia histórica
#     junto con 1982-83.
#   • Meteored (jul-2026): para el ciclo 2026-27, catalogado por NOAA con
#     81% de probabilidad de ser "muy fuerte", los modelos ubican
#     anomalías de +50 a +60 mm/mes sobre Corrientes en el pico
#     (nov-dic), escalando fuerte en el verano.
#   • Reportes de Defensa Civil de Corrientes (jul-2026): tormentas
#     puntuales de 170-200 mm en 24-48h ya provocando desborde de arroyos
#     durante la fase de instalación del fenómeno.
#
# Los valores de esta tabla son "mm en una tormenta de diseño de pocas
# horas" (no el acumulado del trimestre completo) — pensados como el
# evento de lluvia intensa puntual que dispara una crecida repentina
# dentro de un trimestre con esa anomalía de fondo. Son una
# APROXIMACIÓN — ajustalos si conseguís series históricas reales.
TORMENTA_DISENO = {
    # (fase, intensidad): (mm_totales, horas_lluvia, nota)
    ('El Niño', 'muy fuerte'): (180.0, 3.0, 'Referencia: SON 1997 (~500mm/trimestre por encima de lo normal, SMN)'),
    ('El Niño', 'fuerte'):     (120.0, 3.0, 'Referencia: tormentas puntuales de 170-200mm/24-48h reportadas en Corrientes, jul-2026'),
    ('El Niño', 'moderado'):   (70.0, 3.0,  'Referencia: anomalías de +50 a +60mm/mes proyectadas por Meteored para ciclo 2026-27'),
    ('El Niño', 'débil'):      (40.0, 3.0,  'Señal húmeda más débil/menos consistente en eventos débiles'),
    ('La Niña', None):         (25.0, 3.0,  'La Niña tiende a un NEA más seco — tormenta de diseño base, no de anomalía'),
    ('Neutral', None):         (25.0, 3.0,  'Sin señal ENSO — tormenta de diseño base (lluvia intensa típica de la región)'),
}
DURACION_TOTAL_HORAS_DEFECTO = 6.0  # lluvia + escurrimiento posterior


def tormenta_diseno_desde_oni(fase, intensidad):
    """
    Traduce fase/intensidad ONI (las que devuelve consultar_oni) en una
    tormenta de diseño { mm_totales, horas_lluvia, horas_total, nota }
    lista para alimentar overland_flow_engine.simular_crecida_dinamica().
    """
    clave = (fase, intensidad if fase == 'El Niño' else None)
    mm_totales, horas_lluvia, nota = TORMENTA_DISENO.get(clave, TORMENTA_DISENO[('Neutral', None)])
    return {
        'mm_totales': mm_totales,
        'horas_lluvia': horas_lluvia,
        'horas_total': DURACION_TOTAL_HORAS_DEFECTO,
        'mm_hora': round(mm_totales / horas_lluvia, 1),
        'nota': nota,
        'aproximado': True,
    }


def _clasificar_oni(valor):
    if valor >= 0.5:
        fase = 'El Niño'
    elif valor <= -0.5:
        fase = 'La Niña'
    else:
        return fase_neutral()

    intensidad_abs = abs(valor)
    if intensidad_abs >= 2.0:
        intensidad = 'muy fuerte'
    elif intensidad_abs >= 1.5:
        intensidad = 'fuerte'
    elif intensidad_abs >= 1.0:
        intensidad = 'moderado'
    else:
        intensidad = 'débil'
    return fase, intensidad


def fase_neutral():
    return 'Neutral', None


def consultar_oni():
    """Devuelve el ONI más reciente disponible, con fase e intensidad.
    Cachea 12h. Si NOAA no responde, devuelve el último valor cacheado
    (aunque esté vencido) en vez de fallar, con un aviso de que es dato
    viejo — mejor un ONI de hace unos días que ningún dato."""
    ahora = time.time()
    if _cache['datos'] and (ahora - _cache['timestamp']) < _CACHE_HORAS * 3600:
        return _cache['datos']

    try:
        r = requests.get(_ONI_URL, timeout=15)
        r.raise_for_status()
        lineas = [l.strip() for l in r.text.strip().split('\n') if l.strip()]
        # Primera línea es el encabezado "SEAS YR TOTAL ANOM"
        ultima = lineas[-1].split()
        temporada, anio, total, anom = ultima[0], int(ultima[1]), float(ultima[2]), float(ultima[3])

        fase, intensidad = _clasificar_oni(anom)

        # Tendencia: comparar contra la temporada anterior (dato de 3 meses atrás)
        tendencia = None
        if len(lineas) >= 2:
            anterior = lineas[-2].split()
            anom_prev = float(anterior[3])
            diff = round(anom - anom_prev, 2)
            if diff > 0.15:
                tendencia = 'intensificándose'
            elif diff < -0.15:
                tendencia = 'debilitándose'
            else:
                tendencia = 'estable'

        resultado = {
            'ok': True,
            'oni': round(anom, 2),
            'temporada': temporada,   # ej. "MJJ" (May-Jun-Jul)
            'anio': anio,
            'fase': fase,             # 'El Niño' | 'La Niña' | 'Neutral'
            'intensidad': intensidad, # 'débil'|'moderado'|'fuerte'|'muy fuerte'|None (si Neutral)
            'tendencia': tendencia,
            'fuente': 'NOAA CPC - Oceanic Niño Index (ONI)',
        }
        _cache['datos'] = resultado
        _cache['timestamp'] = ahora
        return resultado

    except Exception as e:
        if _cache['datos']:
            vencido = dict(_cache['datos'])
            vencido['aviso'] = f'Dato cacheado (posiblemente desactualizado) — NOAA no respondió: {e}'
            return vencido
        return {'ok': False, 'error': f'No se pudo consultar el ONI de NOAA: {e}'}
