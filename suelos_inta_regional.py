# ============================================================
#  suelos_inta_regional.py
# ------------------------------------------------------------
#  Resuelve, para un punto (lat, lon), cuál es la MEJOR capa de
#  suelos INTA disponible EN VIVO por WFS, priorizando resolución:
#  1:50.000 > 1:100.000 > 1:500.000 (nacional, siempre disponible
#  como último recurso).
#
#  INVESTIGACIÓN (julio 2026): no existe un servicio único
#  nacional con mejor resolución que 1:500.000. Lo que sí existe
#  son ~8 nodos geoservidores REGIONALES (geo-nodoXX.inta.gob.ar),
#  cada uno publicando sus propias capas, con su propio esquema de
#  campos. Confirmado a mano (GetCapabilities de cada nodo):
#    - Corrientes (geo-nodo09): 10 departamentos a 1:50.000 +
#      Esquina/Lomas Arenosas a 1:100.000.
#    - Córdoba (geo-nodo08): capa continua provincial a 1:100.000
#      (más nueva, 2025) — con 1:250.000/1:500.000 de respaldo en
#      el mismo nodo, no usadas acá porque el nacional ya cubre
#      esa franja de resolución.
#    - Entre Ríos (geo-nodo03): capa provincial a 1:100.000 (serie
#      histórica 1986-2011).
#    - Buenos Aires (geo-nodo07): NO tiene clasificación de suelo
#      publicada en vivo (solo "coberturas de suelo"/uso de la
#      tierra de 2 departamentos, que es otra cosa). Pendiente.
#
#  DISEÑO:
#  - Cada capa candidata tiene su bbox real (de su propio
#    <ows:WGS84BoundingBox>). Se arma la lista de candidatas cuyo
#    bbox contiene el punto, ordenada de mejor a peor resolución.
#  - Se prueban en orden por WFS GetFeature (bbox chico alrededor
#    del punto). La primera que devuelva un feature gana. Si una
#    capa no tiene datos ahí (puede pasar aunque el punto esté
#    dentro de su bbox declarado — el bbox es rectangular, la
#    cobertura real no), se sigue probando la siguiente.
#  - Siempre termina en la capa nacional 1:500.000 como último
#    recurso, así el comportamiento actual nunca se rompe.
#  - Los esquemas de campo NO son todos iguales (ya confirmamos
#    que Córdoba usa 'Cap Uso'/'Clase'/'IP' con espacios, distinto
#    del nacional 'simbc'/'ind_prod'). Se normalizan con matching
#    tolerante por palabras clave, igual criterio que ya usa
#    hidrografia_vectorial.py para cuerpos de agua.
#
#  Requiere: pip install requests
# ============================================================

import requests

try:
    from shapely.geometry import shape, Point, box
    from shapely.strtree import STRtree
    _SHAPELY_DISPONIBLE = True
except ImportError:
    shape = Point = box = STRtree = None
    _SHAPELY_DISPONIBLE = False

# ------------------------------------------------------------
# Capas GEOJSON estáticas (alojadas en Hugging Face) — a diferencia de
# CAPAS_REGIONALES (WFS en vivo), estas se descargan UNA vez, se cachean
# en memoria, y el point-in-polygon se resuelve local con shapely (más
# rápido que un WFS y no depende de que el geoservidor de origen esté
# arriba). Mismo bbox real, mismo criterio de prioridad por escala que
# las WFS — se mezclan juntas en capas_candidatas().
# ------------------------------------------------------------
CAPAS_GEOJSON = [
    {'provincia': 'Córdoba', 'depto': None, 'escala': 50_000,
     'url': 'https://huggingface.co/datasets/Briant97/GeoSentinel/resolve/main/Suelos%20de%20C%C3%B3rdoba%20escala%201%20en%2050.geojson',
     # bbox provincial aproximado (mismo criterio que las demás entradas
     # sin bbox exacto de capa) — alcanza para el enrutamiento, el punto
     # exacto lo resuelve el point-in-polygon real más abajo.
     'bbox': (-65.8, -35.0, -61.8, -29.4)},
]

_CACHE_GEOJSON = {}  # url -> {'features': [...], 'arbol': STRtree, 'geoms': [...]}


def _cargar_geojson_cacheado(url, timeout=60):
    """Descarga (una sola vez, cachea en memoria del proceso) y arma un
    índice espacial (STRtree) para resolver point-in-polygon rápido, sin
    recorrer todas las features en cada consulta. Devuelve None si falla
    la descarga o si shapely no está disponible."""
    if not _SHAPELY_DISPONIBLE:
        return None
    if url in _CACHE_GEOJSON:
        return _CACHE_GEOJSON[url]
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        features = data.get('features', [])
        geoms = []
        features_validas = []
        for f in features:
            geom_raw = f.get('geometry')
            if not geom_raw:
                continue
            try:
                geoms.append(shape(geom_raw))
                features_validas.append(f)
            except Exception:
                continue  # geometría corrupta puntual — se ignora esa feature, no rompe todo el archivo
        arbol = STRtree(geoms) if geoms else None
        cache = {'features': features_validas, 'geoms': geoms, 'arbol': arbol}
        _CACHE_GEOJSON[url] = cache
        return cache
    except Exception as exc:
        print(f'⚠️ No se pudo descargar/parsear el geojson de suelos {url}: {exc}')
        return None


def _geojson_get_feature(candidata, lat, lon):
    """Equivalente a _wfs_get_feature() pero para una capa geojson
    estática ya cacheada — devuelve una lista de 0 o 1 feature (la que
    contiene el punto), o None si falló la carga."""
    cache = _cargar_geojson_cacheado(candidata['url'])
    if cache is None or cache['arbol'] is None:
        return None
    punto = Point(lon, lat)
    indices = cache['arbol'].query(punto)
    for idx in indices:
        geom = cache['geoms'][idx]
        if geom.contains(punto) or geom.intersects(punto):
            return [cache['features'][idx]]
    return []

# ------------------------------------------------------------
# Catálogo de capas regionales confirmadas. bbox = (min_lon,
# min_lat, max_lon, max_lat), tomado del WGS84BoundingBox real de
# cada FeatureType en el GetCapabilities de su nodo.
# ------------------------------------------------------------
CAPAS_REGIONALES = [
    # ---- Corrientes (geo-nodo09) — departamentos a 1:50.000 ----
    {'provincia': 'Corrientes', 'depto': 'Curuzú Cuatiá', 'escala': 50_000,
     'servidor': 'https://geo-nodo09.inta.gob.ar/geoserver/wfs',
     'capa': 'geonode:suelos_curuzu',
     'bbox': (-58.910672, -30.422429, -57.609350, -29.044594)},
    {'provincia': 'Corrientes', 'depto': 'Empedrado', 'escala': 50_000,
     'servidor': 'https://geo-nodo09.inta.gob.ar/geoserver/wfs',
     'capa': 'geonode:suelos_empedrado_50000_1d36f0eac40a59619d2452fa1dc7f3d8',
     'bbox': (-59.093039, -28.223361, -58.303918, -27.602679)},
    {'provincia': 'Corrientes', 'depto': 'General Alvear', 'escala': 50_000,
     'servidor': 'https://geo-nodo09.inta.gob.ar/geoserver/wfs',
     'capa': 'geonode:suelos_alvear',
     'bbox': (-56.708420, -29.116661, -56.287373, -28.427718)},
    {'provincia': 'Corrientes', 'depto': 'Goya', 'escala': 50_000,
     'servidor': 'https://geo-nodo09.inta.gob.ar/geoserver/wfs',
     'capa': 'geonode:Suelos_Goya',
     'bbox': (-59.665247, -30.047477, -58.787984, -29.050332)},
    {'provincia': 'Corrientes', 'depto': 'Lavalle', 'escala': 50_000,
     'servidor': 'https://geo-nodo09.inta.gob.ar/geoserver/wfs',
     'capa': 'geonode:Suelos_Lavalle',
     'bbox': (-59.238623, -29.257638, -58.600947, -28.768828)},
    {'provincia': 'Corrientes', 'depto': 'Mercedes', 'escala': 50_000,
     'servidor': 'https://geo-nodo09.inta.gob.ar/geoserver/wfs',
     'capa': 'geonode:suelos_mercedes_50000',
     'bbox': (-58.601389, -29.744732, -57.164395, -28.499366)},
    {'provincia': 'Corrientes', 'depto': 'Monte Caseros', 'escala': 50_000,
     'servidor': 'https://geo-nodo09.inta.gob.ar/geoserver/wfs',
     'capa': 'geonode:suelos_monte_caseros_50000',
     'bbox': (-58.083121, -30.721002, -57.559066, -29.818855)},
    {'provincia': 'Corrientes', 'depto': 'Santo Tomé', 'escala': 50_000,
     'servidor': 'https://geo-nodo09.inta.gob.ar/geoserver/wfs',
     'capa': 'geonode:suelos_santotome',
     'bbox': (-56.881437, -28.814645, -55.611488, -27.764881)},
    {'provincia': 'Corrientes', 'depto': 'Sauce', 'escala': 50_000,
     'servidor': 'https://geo-nodo09.inta.gob.ar/geoserver/wfs',
     'capa': 'geonode:suelos_sauce',
     'bbox': (-59.126224, -30.273430, -58.445075, -29.630278)},
    {'provincia': 'Corrientes', 'depto': 'Yacyretá (área de influencia)', 'escala': 50_000,
     'servidor': 'https://geo-nodo09.inta.gob.ar/geoserver/wfs',
     'capa': 'geonode:suelos_yacireta',
     'bbox': (-58.870010, -28.005835, -55.830430, -27.264251)},

    # ---- Corrientes (geo-nodo09) — mosaico provincial "multiescala" (respaldo) ----
    # Cubre TODA la provincia, incluidos los departamentos sin carta puntual propia
    # (ej. Concepción). Cada feature trae su propia 'escala' real (100 o 500,
    # mezcladas según lo que había disponible al armar el mosaico) — por eso
    # 'escala' acá es un valor representativo para el ORDEN de prioridad
    # (se prueba después de las cartas departamentales de 1:50.000, antes del
    # nacional), y el valor real por punto se toma del campo 'escala' de la
    # propia feature en consultar_mejor_capa(). Esquema de campos propio:
    # hasta 4 componentes de suelo por polígono (suelo_1..4, taxonom_1..4,
    # cu_1..4 = capacidad de uso, ip_1..4 = índice de productividad,
    # percent_1..4 = % de cada componente), más paisaje y riesgo_ero.
    {'provincia': 'Corrientes', 'depto': None, 'escala': 100_000,
     'servidor': 'https://geo-nodo09.inta.gob.ar/geoserver/wfs',
     'capa': 'geonode:suelos_multiescala',
     'bbox': (-59.712776, -30.723748, -55.620044, -27.253312)},

    # ---- Corrientes (geo-nodo09) — 1:100.000 (respaldo dentro de la provincia) ----
    {'provincia': 'Corrientes', 'depto': 'Esquina', 'escala': 100_000,
     'servidor': 'https://geo-nodo09.inta.gob.ar/geoserver/wfs',
     'capa': 'geonode:Suelos_Esquina2',
     'bbox': (-59.711684, -30.433981, -58.717913, -29.468219)},
    {'provincia': 'Corrientes', 'depto': 'Lomas Arenosas', 'escala': 100_000,
     'servidor': 'https://geo-nodo09.inta.gob.ar/geoserver/wfs',
     'capa': 'geonode:Lomas_arenosas',
     'bbox': (-59.121089, -29.024789, -57.151074, -27.483856)},

    # ---- Formosa (geo-nodo05) — Piraré (bbox APROXIMADO, sacado del punto
    # de click GetFeatureInfo, no del WGS84BoundingBox real de la capa —
    # verificar con GetCapabilities de geo-nodo05 y ajustar si hace falta) ----
    {'provincia': 'Formosa', 'depto': 'Pirané', 'escala': 100_000,  # escala sin confirmar, asumida típica de carta departamental
     'servidor': 'https://geo-nodo05.inta.gob.ar/geoserver/wfs',
     'capa': 'geonode:mapa_de_suelos_pirane_final_latlong',
     'bbox': (-60.3, -26.4, -58.9, -25.2)},

    # ---- Salta (geo-backend, no un geo-nodo regional) — Valle de Lerma
    # (bbox APROXIMADO, mismo criterio que Piraré arriba) ----
    {'provincia': 'Salta', 'depto': 'Valle de Lerma', 'escala': 100_000,  # escala sin confirmar
     'servidor': 'https://geo-backend.inta.gob.ar/geoserver/wfs',
     'capa': 'geonode:carta_suelos_valle_lerma_ll',
     'bbox': (-66.0, -25.5, -65.0, -24.5)},

    # ---- NOA (Salta + Jujuy) 1:250.000 (geo-backend) — el escalón
    # intermedio que faltaba en la cascada. Escala CONFIRMADA (el propio
    # título del layer en el catálogo INTA dice "suelos NOA Escala
    # 1:250000"). bbox aproximado provincial Salta+Jujuy combinado —
    # verificar con GetCapabilities si hace falta más precisión. ----
    {'provincia': 'Salta/Jujuy', 'depto': None, 'escala': 250_000,
     'servidor': 'https://geo-backend.inta.gob.ar/geoserver/wfs',
     'capa': 'geonode:asociaciones_suelos_salta_jujuy',
     'bbox': (-68.5, -26.5, -62.5, -21.5)},

    # ---- Santa Fe (geo-nodo10) — capa provincial (confirmada 1:50.000 por el usuario, viendo el WFS directo) ----
    {'provincia': 'Santa Fe', 'depto': None, 'escala': 50_000,
     'servidor': 'https://geo-nodo10.inta.gob.ar/geoserver/wfs',
     'capa': 'geonode:suelos_santa_fe_25',
     # bbox aproximado de la provincia (mismo criterio que hidrografia_vectorial.py,
     # no el bbox exacto de la capa — alcanza para el enrutamiento).
     'bbox': (-63.4, -34.5, -59.7, -28.0)},

    # ---- Córdoba (geo-nodo08) — capa continua provincial (2025) ----
    {'provincia': 'Córdoba', 'depto': None, 'escala': 100_000,
     'servidor': 'https://geo-nodo08.inta.gob.ar/geoserver/wfs',
     'capa': 'geonode:2025_cad_suelos_cordoba_100mil',
     'bbox': (-65.752094, -33.001873, -62.499999, -29.498756)},

    # ---- Entre Ríos (geo-nodo03) — capa provincial (serie 1986-2011) ----
    {'provincia': 'Entre Ríos', 'depto': None, 'escala': 100_000,
     'servidor': 'https://geo-nodo03.inta.gob.ar/geoserver/wfs',
     'capa': 'geonode:carta_de_suelos_unociemil_df05b067570a0dea80e67f5aa41aa5d6',
     'bbox': (-60.774918, -34.038593, -57.801026, -30.158963)},
]

# Capa nacional de respaldo — siempre disponible como último recurso,
# cubre todo el país (misma que ya usa el resto de la app).
CAPA_NACIONAL = {
    'provincia': None, 'depto': None, 'escala': 500_000,
    'servidor': 'https://geo-backend.inta.gob.ar/geoserver/wfs',
    'capa': 'geonode:suelos_argentina_1_500',
    'bbox': None,
}


# Mensaje de advertencia cuando la única fuente disponible fue el nacional
# 1:500.000. Ese dataset es el "Atlas de Suelos de la República Argentina"
# de los años 90 — catalogado oficialmente por el propio INTA como "Nivel
# de levantamiento: Reconocimiento" (la escala más gruesa que existe).
# Confirmado en la práctica (julio 2026, revisión de un ingeniero
# agrónomo de INTA): puede mostrar clases de capacidad de uso equivocadas
# a nivel de lote (ej. Clase 8 donde en realidad es Clase 4). Sirve como
# panorama regional, NO para decisiones puntuales de manejo.
ADVERTENCIA_ESCALA_NACIONAL = (
    'Este dato viene del mapa nacional 1:500.000 (Atlas de Suelos de la '
    'República Argentina, INTA — nivel de levantamiento "Reconocimiento", '
    'la escala más gruesa disponible) porque todavía no hay una capa '
    'regional de mejor resolución para este punto. A esta escala la clase '
    'de capacidad de uso puede estar equivocada a nivel de lote — usalo '
    'solo como panorama general, no como base para decisiones de manejo. '
    'Si tenés una carta de suelos más detallada de esta zona, valdría la '
    'pena contrastar.'
)


def _punto_en_bbox(lat, lon, bbox):
    if bbox is None:
        return True
    min_lon, min_lat, max_lon, max_lat = bbox
    return min_lon <= lon <= max_lon and min_lat <= lat <= max_lat


def capas_candidatas(lat, lon):
    """
    Lista de capas a probar para un punto, de MEJOR a PEOR resolución
    (menor escala numérica primero). Mezcla WFS en vivo y geojson
    estático (cada una marcada con 'tipo': 'wfs'|'geojson'). Siempre
    termina con la capa nacional 1:500.000 como último recurso.
    """
    candidatas = [dict(c, tipo='wfs') for c in CAPAS_REGIONALES if _punto_en_bbox(lat, lon, c['bbox'])]
    candidatas += [dict(c, tipo='geojson') for c in CAPAS_GEOJSON if _punto_en_bbox(lat, lon, c['bbox'])]
    candidatas.sort(key=lambda c: c['escala'])
    candidatas.append(dict(CAPA_NACIONAL, tipo='wfs'))
    return candidatas


def _wfs_get_feature(servidor, capa, lat, lon, delta_grados=0.005, timeout=15):
    """
    WFS GetFeature con un bbox chico centrado en el punto (~500m,
    mismo criterio que ya usa el frontend para el mapa nacional).
    Devuelve la lista de features (puede ser vacía) o None si la
    consulta falló (servidor caído, capa inexistente, timeout, etc.).
    """
    bbox = f'{lon - delta_grados},{lat - delta_grados},{lon + delta_grados},{lat + delta_grados},EPSG:4326'
    params = {
        'service': 'WFS', 'version': '2.0.0', 'request': 'GetFeature',
        'typeName': capa, 'outputFormat': 'application/json',
        'bbox': bbox, 'srsName': 'EPSG:4326',
    }
    try:
        resp = requests.get(servidor, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get('features', [])
    except Exception:
        return None


# ------------------------------------------------------------
# Matching tolerante de campos: cada fuente regional puede tener su
# propio esquema (Córdoba: 'Cap Uso'/'Clase'/'IP' con espacios;
# nacional: 'simbc'/'tipo_uc'/'ind_prod' en snake_case; las de
# Corrientes/Entre Ríos: esquema todavía sin confirmar). Se buscan
# por palabras clave, sin importar mayúsculas/acentos/espacios/
# guiones bajos.
# ------------------------------------------------------------
def _quitar_acentos(texto):
    reemplazos = str.maketrans('áéíóúñÁÉÍÓÚÑ', 'aeiounAEIOUN')
    return texto.translate(reemplazos)


def _norm(s):
    return _quitar_acentos(str(s or '')).lower().replace(' ', '').replace('_', '').replace('-', '')


_CAMPOS_BUSCADOS = {
    'clase':          ['clase'],
    'subclase':       ['subclase'],
    'capacidad_uso':  ['capuso', 'capacidaduso', 'tipouc', 'aptitud', 'cu'],
    'indice_prod':    ['ip', 'indiceproductividad', 'indprod', 'iat'],
    'simbolo':        ['simbolo', 'simbc', 'ucar'],
    'nombre_unidad':  ['nombre', 'tipounidad'],
    'limitante':      ['limitante', 'limitppal', 'limit'],
    'drenaje':        ['drenaje'],
    'anegabilidad':   ['anegab', 'anegamiento'],
}


def normalizar_propiedades(props):
    """
    Mapea las properties crudas de CUALQUIER capa de suelo (nacional o
    regional) a un esquema común, buscando por palabras clave tolerantes.
    Los campos que no se puedan mapear quedan en None.
    'propiedades_crudas' siempre incluye el original completo, por si el
    frontend necesita mostrar algo puntual de una fuente específica que
    este mapeo no contempló.

    Dos pasadas, no una sola:
      1) Coincidencia EXACTA de la clave normalizada. Necesario porque
         algunos nombres de campo son substring uno del otro (ej.
         'Subclase' contiene 'clase' — con una sola pasada por substring,
         'Subclase' podía pisar el valor de 'Clase' si se procesaba antes
         en el diccionario, dando un resultado incorrecto).
      2) Coincidencia por substring, solo para los campos que la pasada 1
         no pudo resolver — cubre esquemas con nombres de columna más
         largos/compuestos (ej. 'tipo_uc' → capacidad_uso).
    """
    resultado = {campo: None for campo in _CAMPOS_BUSCADOS}
    props = props or {}

    for clave_original, valor in props.items():
        if valor in (None, ''):
            continue
        clave_norm = _norm(clave_original)
        for campo, palabras in _CAMPOS_BUSCADOS.items():
            if resultado[campo] is None and clave_norm in palabras:
                resultado[campo] = valor

    for clave_original, valor in props.items():
        if valor in (None, ''):
            continue
        clave_norm = _norm(clave_original)
        for campo, palabras in _CAMPOS_BUSCADOS.items():
            if resultado[campo] is None and any(p in clave_norm for p in palabras):
                resultado[campo] = valor

    resultado['propiedades_crudas'] = props
    return resultado


def consultar_mejor_capa(lat, lon):
    """
    Punto de entrada principal: prueba las capas candidatas para
    (lat, lon) de mejor a peor resolución y devuelve el resultado de
    la PRIMERA que responda con al menos un feature en ese punto
    exacto.

    Retorna dict:
      { encontrado: True, fuente: {provincia, depto, escala, capa},
        clase, subclase, capacidad_uso, indice_prod, simbolo,
        nombre_unidad, limitante, drenaje, anegabilidad,
        propiedades_crudas }
    o { encontrado: False } si ni la capa nacional respondió (caso
    raro: todos los servidores caídos).
    """
    for candidata in capas_candidatas(lat, lon):
        if candidata.get('tipo') == 'geojson':
            features = _geojson_get_feature(candidata, lat, lon)
        else:
            features = _wfs_get_feature(candidata['servidor'], candidata['capa'], lat, lon)
        if not features:
            continue  # sin datos en esta capa para este punto: se prueba la siguiente
        props = features[0].get('properties', {}) or {}
        normalizado = normalizar_propiedades(props)

        # Algunas capas (ej. suelos_multiescala de Corrientes) traen su propia
        # escala REAL por feature, mezclada dentro de un mismo mosaico — si
        # está presente, pisa el valor fijo del catálogo para que la
        # confiabilidad reportada sea la del punto exacto, no un promedio.
        escala_real = candidata['escala']
        escala_cruda = props.get('escala')
        if escala_cruda not in (None, ''):
            try:
                escala_reportada = float(escala_cruda)
                # Algunas fuentes guardan la escala ya multiplicada por 1000
                # (ej. 100 = 1:100.000); normalizamos a la misma unidad que
                # usa el resto del catálogo (denominador completo).
                escala_real = escala_reportada * 1000 if escala_reportada < 10_000 else escala_reportada
            except (TypeError, ValueError):
                pass

        resultado = {
            'encontrado': True,
            'fuente': {
                'provincia': candidata['provincia'],
                'depto': candidata['depto'],
                'escala': escala_real,
                'capa': candidata.get('capa') or candidata.get('url'),
                'tipo': candidata.get('tipo', 'wfs'),
            },
            **normalizado,
        }
        if escala_real >= 500_000:
            resultado['advertencia_escala'] = ADVERTENCIA_ESCALA_NACIONAL
            resultado['confiabilidad'] = 'baja'
        elif escala_real >= 250_000:
            resultado['confiabilidad'] = 'media'
        else:
            resultado['confiabilidad'] = 'alta'
        return resultado

    return {'encontrado': False}


# ============================================================
#  CAPA VISUAL DE APTITUD (overlay coloreado en el área visible del mapa)
# ------------------------------------------------------------
#  A diferencia de consultar_mejor_capa() (un punto exacto), esto trae
#  TODAS las features dentro de un bbox, para pintar el mapa. Usa el mismo
#  catálogo y el mismo normalizador de campos, así el frontend puede
#  clasificar el color leyendo siempre los mismos nombres normalizados
#  (indice_prod, capacidad_uso, drenaje, etc.) sin importar de qué
#  provincia/esquema haya salido cada polígono.
#
#  Simplificación consciente (v1): si hay capas puntuales (departamentales)
#  que cubren el bbox, se usan TODAS ellas (no se solapan entre sí). Si
#  ninguna cubre nada, se prueba UN respaldo provincial (multiescala,
#  geojson, etc.) y si tampoco, el nacional. No se mezclan puntuales +
#  respaldo en la misma vista todavía (ej. si el mapa muestra Concepción
#  junto a Curuzú Cuatiá a la vez, hoy se resuelve por el primero que
#  responda) — mejora pendiente si hace falta más precisión ahí.
# ============================================================

def _bbox_se_solapan(a, b):
    if a is None or b is None:
        return True
    aminx, aminy, amaxx, amaxy = a
    bminx, bminy, bmaxx, bmaxy = b
    return not (amaxx < bminx or aminx > bmaxx or amaxy < bminy or aminy > bmaxy)


def _wfs_get_features_bbox(servidor, capa, bbox, max_features=400, timeout=15):
    """Como _wfs_get_feature() pero para un bbox real (no un puntito
    alrededor de una coordenada) — trae varias features para pintar el mapa."""
    minx, miny, maxx, maxy = bbox
    params = {
        'service': 'WFS', 'version': '2.0.0', 'request': 'GetFeature',
        'typeName': capa, 'outputFormat': 'application/json',
        'bbox': f'{minx},{miny},{maxx},{maxy},EPSG:4326', 'srsName': 'EPSG:4326',
        'count': max_features, 'maxFeatures': max_features,  # count=2.0.0, maxFeatures=1.0.0 — mandamos ambos por las dudas
    }
    try:
        resp = requests.get(servidor, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get('features', [])
    except Exception:
        return None


def _geojson_get_features_bbox(candidata, bbox, max_features=400):
    """Equivalente a _geojson_get_feature() pero para un bbox — usa el
    mismo índice espacial STRtree ya cacheado, filtrando por intersección
    real de geometría (no solo el bbox aproximado del catálogo)."""
    cache = _cargar_geojson_cacheado(candidata['url'])
    if cache is None or cache['arbol'] is None:
        return None
    minx, miny, maxx, maxy = bbox
    caja = box(minx, miny, maxx, maxy)
    indices = cache['arbol'].query(caja)
    resultado = []
    for idx in indices:
        if cache['geoms'][idx].intersects(caja):
            resultado.append(cache['features'][idx])
            if len(resultado) >= max_features:
                break
    return resultado


def _normalizar_feature_geojson(f, candidata):
    """Arma una feature GeoJSON de salida con properties ya normalizadas
    (mismos nombres de campo sin importar la fuente). Se descarta
    'propiedades_crudas' acá para no inflar el payload del mapa con cientos
    de polígonos — sí se conserva en consultar_mejor_capa() (un solo punto)."""
    props = f.get('properties', {}) or {}
    norm = normalizar_propiedades(props)
    norm.pop('propiedades_crudas', None)
    norm['_fuente_provincia'] = candidata.get('provincia')
    norm['_fuente_depto'] = candidata.get('depto')
    norm['_fuente_escala'] = candidata.get('escala')
    return {'type': 'Feature', 'geometry': f.get('geometry'), 'properties': norm}


def capa_visual_bbox(bbox, max_features_por_capa=400):
    """
    Punto de entrada para el overlay visual de aptitud: dado un bbox
    (min_lon, min_lat, max_lon, max_lat), devuelve un FeatureCollection
    GeoJSON con properties ya normalizadas, usando la mejor cobertura
    regional disponible para esa área y cayendo al nacional 1:500.000
    solo si no hay nada regional en esa zona.
    """
    candidatas_regionales = [c for c in CAPAS_REGIONALES if _bbox_se_solapan(bbox, c['bbox'])]
    candidatas_geojson = [c for c in CAPAS_GEOJSON if _bbox_se_solapan(bbox, c['bbox'])]

    puntuales = sorted(
        [c for c in candidatas_regionales if c['depto'] is not None],
        key=lambda c: c['escala'],
    )
    respaldo = sorted(
        [c for c in candidatas_regionales if c['depto'] is None] + candidatas_geojson,
        key=lambda c: c['escala'],
    )

    features_salida = []
    fuentes_usadas = []

    for c in puntuales:
        feats = _wfs_get_features_bbox(c['servidor'], c['capa'], bbox, max_features_por_capa)
        if feats:
            features_salida.extend(_normalizar_feature_geojson(f, c) for f in feats)
            fuentes_usadas.append(c)

    if not features_salida:
        for c in respaldo:
            if 'url' in c:
                feats = _geojson_get_features_bbox(c, bbox, max_features_por_capa)
            else:
                feats = _wfs_get_features_bbox(c['servidor'], c['capa'], bbox, max_features_por_capa)
            if feats:
                features_salida.extend(_normalizar_feature_geojson(f, c) for f in feats)
                fuentes_usadas.append(c)
                break  # un respaldo alcanza — no mezclar varios respaldos genéricos entre sí

    if not features_salida:
        feats = _wfs_get_features_bbox(CAPA_NACIONAL['servidor'], CAPA_NACIONAL['capa'], bbox, max_features_por_capa)
        if feats:
            features_salida.extend(_normalizar_feature_geojson(f, CAPA_NACIONAL) for f in feats)
            fuentes_usadas.append(CAPA_NACIONAL)

    return {
        'type': 'FeatureCollection',
        'features': features_salida,
        'fuentes': [
            {'provincia': c['provincia'], 'depto': c.get('depto'), 'escala': c['escala'],
             'capa': c.get('capa') or c.get('url')}
            for c in fuentes_usadas
        ],
    }
