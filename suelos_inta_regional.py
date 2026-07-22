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

    # ---- Corrientes (geo-nodo09) — 1:100.000 (respaldo dentro de la provincia) ----
    {'provincia': 'Corrientes', 'depto': 'Esquina', 'escala': 100_000,
     'servidor': 'https://geo-nodo09.inta.gob.ar/geoserver/wfs',
     'capa': 'geonode:Suelos_Esquina2',
     'bbox': (-59.711684, -30.433981, -58.717913, -29.468219)},
    {'provincia': 'Corrientes', 'depto': 'Lomas Arenosas', 'escala': 100_000,
     'servidor': 'https://geo-nodo09.inta.gob.ar/geoserver/wfs',
     'capa': 'geonode:Lomas_arenosas',
     'bbox': (-59.121089, -29.024789, -57.151074, -27.483856)},

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


def _punto_en_bbox(lat, lon, bbox):
    if bbox is None:
        return True
    min_lon, min_lat, max_lon, max_lat = bbox
    return min_lon <= lon <= max_lon and min_lat <= lat <= max_lat


def capas_candidatas(lat, lon):
    """
    Lista de capas a probar para un punto, de MEJOR a PEOR resolución
    (menor escala numérica primero). Siempre termina con la capa
    nacional 1:500.000 como último recurso.
    """
    candidatas = [c for c in CAPAS_REGIONALES if _punto_en_bbox(lat, lon, c['bbox'])]
    candidatas.sort(key=lambda c: c['escala'])
    candidatas.append(CAPA_NACIONAL)
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
    'capacidad_uso':  ['capuso', 'capacidaduso', 'tipouc', 'aptitud'],
    'indice_prod':    ['ip', 'indiceproductividad', 'indprod'],
    'simbolo':        ['simbolo', 'simbc'],
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
        features = _wfs_get_feature(candidata['servidor'], candidata['capa'], lat, lon)
        if not features:
            continue  # sin datos en esta capa para este punto: se prueba la siguiente
        props = features[0].get('properties', {}) or {}
        normalizado = normalizar_propiedades(props)
        return {
            'encontrado': True,
            'fuente': {
                'provincia': candidata['provincia'],
                'depto': candidata['depto'],
                'escala': candidata['escala'],
                'capa': candidata['capa'],
            },
            **normalizado,
        }

    return {'encontrado': False}
