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
