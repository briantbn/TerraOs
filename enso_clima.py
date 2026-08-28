"""
enso_clima.py
==========================================
Estado y pronóstico ENSO/El Niño (ONI) para el módulo premium "Climate
Intelligence" del panel. A diferencia de suelos_inta_regional.py, esto NO
es un dato por lote/polígono: el ENSO es un índice regional/nacional,
igual para toda la Argentina en un momento dado.

Usado por UN endpoint de app.py:

  - /enso_estado  -> obtener_estado_enso()
        Devuelve el estado actual del ONI + el pronóstico probabilístico
        oficial de NOAA/CPC para los próximos ~9 trimestres.

Fuentes (ambas oficiales, gratuitas, sin necesidad de scraping de
imágenes -- confirmado a mano antes de escribir este archivo):

  1) ONI actual/histórico (texto plano, actualizado mensualmente):
     http://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt
     Formato: columnas SEAS YR TOTAL ANOM (ej. "JAS 2026 27.94 1.10").

  2) Pronóstico probabilístico por temporada (tabla HTML real, NO un
     gráfico/imagen -- se confirmó bajando la página):
     https://cpc.ncep.noaa.gov/products/analysis_monitoring/enso/roni/probabilities/
     Se actualiza el 2do jueves de cada mes (misma fecha que la
     Discusión Diagnóstica ENSO de CPC).

Cacheado en memoria con expiración de 24hs -- ninguna de las dos fuentes
cambia más seguido que una vez al mes, así que no tiene sentido pegarle
a NOAA en cada request del usuario.

⚠️ El parser de la tabla de probabilidades (import_probabilidades_html)
depende de la estructura HTML actual de esa página de NOAA. Si NOAA
cambia el diseño de la página, esa función puede empezar a fallar --
está armada para devolver una lista vacía en vez de romper si no
encuentra el patrón esperado, así el endpoint sigue funcionando con
el estado actual del ONI aunque el pronóstico no esté disponible.
"""
import html as _html_module
import re
import time

import requests

_ONI_URL = 'http://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt'
_PROB_URL = 'https://cpc.ncep.noaa.gov/products/analysis_monitoring/enso/roni/probabilities/'

_TIMEOUT = 15
_CACHE_TTL_SEG = 24 * 60 * 60  # 24hs -- la fuente no cambia más seguido

_cache = {'valor': None, 'ts': 0}


def _clasificar_fase(anomalia):
    """Umbral estándar de NOAA/CPC: +/-0.5°C sobre el promedio 1991-2020."""
    if anomalia >= 0.5:
        return 'El Niño'
    if anomalia <= -0.5:
        return 'La Niña'
    return 'Neutral'


def _obtener_oni_actual():
    """Descarga oni.ascii.txt y devuelve la última fila (temporada más
    reciente disponible). El archivo viene ordenado cronológicamente, así
    que la última línea con datos es siempre la más nueva."""
    r = requests.get(_ONI_URL, timeout=_TIMEOUT)
    r.raise_for_status()
    lineas = [l.strip() for l in r.text.splitlines() if l.strip()]
    # Primera línea es el encabezado "SEAS YR TOTAL ANOM"
    ultima = lineas[-1].split()
    if len(ultima) != 4:
        raise ValueError(f'Formato inesperado en oni.ascii.txt, última línea: {lineas[-1]!r}')
    temporada, anio, total, anom = ultima
    anomalia = round(float(anom), 2)
    return {
        'temporada': temporada,
        'anio': int(anio),
        'sst': round(float(total), 2),
        'anomalia': anomalia,
        'fase': _clasificar_fase(anomalia),
    }


def _html_a_texto_plano(html_crudo):
    """Convierte HTML crudo a texto plano simple para poder aplicarle
    regex de forma confiable: saca <script>/<style>, inserta un espacio
    en los límites de celda/fila/salto de línea (<br>, <td>, <tr>, etc.,
    para no pegar el contenido de dos celdas de tabla distintas) y
    decodifica entidades HTML (&ntilde; -> ñ, &nbsp; -> espacio, etc.).

    FIX: la versión anterior de este archivo aplicaba el regex
    directamente sobre requests.get(url).text (HTML crudo, con
    <table>/<tr>/<td> de por medio) -- nunca iba a poder matchear, porque
    solo se había probado a mano contra el texto YA extraído/aplanado
    (como lo devuelve un lector de página, sin tags). Esto es lo que
    causaba que /enso_estado devolviera 'pronostico': [] en producción
    aunque el ONI actual sí funcionara (ese viene de un .txt plano, no de
    HTML). No se suma BeautifulSoup como dependencia nueva solo para esta
    tabla puntual -- alcanza con esta limpieza simple."""
    texto = re.sub(r'<(script|style)[^>]*>.*?</\1>', ' ', html_crudo, flags=re.S | re.I)
    texto = re.sub(r'</?(br|td|th|tr|p|div|li)\b[^>]*>', ' ', texto, flags=re.I)
    texto = re.sub(r'<[^>]+>', '', texto)
    texto = _html_module.unescape(texto)
    texto = re.sub(r'[ \t\r\n]+', ' ', texto).strip()
    return texto


def _partir_tres_porcentajes(digitos):
    """Los 3 porcentajes de una fila a veces vienen pegados sin separador
    (ej. '08416' = 0 / 84 / 16), con 1 a 3 dígitos cada uno -- se usa
    SOLO como fallback si _obtener_probabilidades_html() no logra
    separarlos por espacios/celdas reales (ver ahí). Se prueban todas las
    formas de partir la cadena en 3 partes (1-3 dígitos cada una) y se
    elige la que suma más cerca de 100 (los 3 porcentajes de NOAA siempre
    suman ~100). Devuelve None si ninguna combinación es razonable."""
    n = len(digitos)
    mejor, mejor_diff = None, None
    for l1 in range(1, min(3, n - 2) + 1):
        for l2 in range(1, min(3, n - l1 - 1) + 1):
            l3 = n - l1 - l2
            if not (1 <= l3 <= 3):
                continue
            a, b, c = digitos[:l1], digitos[l1:l1 + l2], digitos[l1 + l2:]
            ai, bi, ci = int(a), int(b), int(c)
            if ai > 100 or bi > 100 or ci > 100:
                continue
            diff = abs(ai + bi + ci - 100)
            if mejor is None or diff < mejor_diff:
                mejor, mejor_diff = (ai, bi, ci), diff
    return mejor


def _obtener_probabilidades_html():
    """Parsea la tabla 'ENSO Probabilities' (Season/La Niña/Neutral/El
    Niño) de la página de NOAA/CPC. Primero limpia el HTML crudo a texto
    plano (ver _html_a_texto_plano) y después prueba dos formas de leer
    cada fila, porque no se puede confirmar de antemano si las 3 celdas
    de porcentaje van a llegar separadas por espacio (celdas de tabla
    reales, caso A -- el esperable) o pegadas sin separador (si la
    limpieza de HTML las junta igual, caso B, mismo formato que ya se
    había visto antes al leer la página con un extractor de texto)."""
    r = requests.get(_PROB_URL, timeout=_TIMEOUT)
    r.raise_for_status()
    texto = _html_a_texto_plano(r.text)

    m = re.search(r'Season\s*La Ni.a\s*Neutral\s*El Ni.o(.*?)Back to top', texto, re.S | re.I)
    if not m:
        return []
    bloque = m.group(1)

    filas = []
    # Caso A: celdas separadas por espacio, ej. "AMJ Apr May Jun 0 84 16"
    for fila in re.finditer(
        r'\b([A-Z]{3})\s+[A-Za-z]{3}\s+[A-Za-z]{3}\s+[A-Za-z]{3}\s+(\d{1,3})\s+(\d{1,3})\s+(\d{1,3})\b',
        bloque,
    ):
        temporada, la_nina, neutral, el_nino = fila.groups()
        filas.append({
            'temporada': temporada,
            'la_nina': int(la_nina),
            'neutral': int(neutral),
            'el_nino': int(el_nino),
        })
    if filas:
        return filas

    # Caso B (fallback): celdas pegadas sin separador, ej. "AMJ Apr May Jun08416"
    for fila in re.finditer(
        r'\b([A-Z]{3})\s+[A-Za-z]{3}\s+[A-Za-z]{3}\s+[A-Za-z]{3}(\d{3,9})\b',
        bloque,
    ):
        temporada, digitos = fila.groups()
        partido = _partir_tres_porcentajes(digitos)
        if partido is None:
            continue
        la_nina, neutral, el_nino = partido
        filas.append({
            'temporada': temporada,
            'la_nina': la_nina,
            'neutral': neutral,
            'el_nino': el_nino,
        })
    return filas


def obtener_estado_enso(forzar_actualizacion=False):
    """Punto de entrada único, cacheado. Devuelve:
    {
      'actual': {...},       # ver _obtener_oni_actual()
      'pronostico': [...],   # ver _obtener_probabilidades_html(), puede
                              # venir vacío si NOAA cambió el formato
      'actualizado': 'YYYY-MM-DD HH:MM',
    }
    """
    ahora = time.time()
    if not forzar_actualizacion and _cache['valor'] and (ahora - _cache['ts'] < _CACHE_TTL_SEG):
        return _cache['valor']

    actual = _obtener_oni_actual()
    try:
        pronostico = _obtener_probabilidades_html()
    except Exception:
        # Si falla el parser de la tabla (ej. NOAA cambió el HTML), no se
        # cae el endpoint entero -- se devuelve al menos el estado actual.
        pronostico = []

    resultado = {
        'actual': actual,
        'pronostico': pronostico,
        'actualizado': time.strftime('%Y-%m-%d %H:%M', time.gmtime(ahora)) + ' UTC',
    }
    _cache['valor'] = resultado
    _cache['ts'] = ahora
    return resultado
