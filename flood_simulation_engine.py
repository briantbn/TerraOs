"""
FloodSimulationEngine
======================
Motor de simulación de inundación multicriterio, separado de Flask/app.py.

Este módulo NO reemplaza la lógica existente en app.py (HAND, corrección de
dosel/edificios, conectividad hidráulica vía cumulativeCost) — la REUTILIZA
y la envuelve en una superficie de costo más rica, en vez de tratar cada
celda "candidata" como costo uniforme (1) para todas.

Qué cambia respecto del enfoque anterior:
    ANTES: costo = 1 para toda celda candidata (HAND <= umbral), sin importar
           pendiente, cobertura, ni si esa celda está "en el sentido" del
           drenaje real. cumulativeCost() encontraba el camino más CORTO en
           distancia pura.
    AHORA: costo = 1 (base) + resistencia continua (0-4), donde la
           resistencia combina pendiente, cobertura vegetal/edificios,
           un sesgo direccional aproximado (hacia dónde escurre el agua)
           y el peso de acumulación de flujo (MERIT 'upa'). El mismo
           cumulativeCost() ahora encuentra el camino de MENOR
           RESISTENCIA, no solo el más corto — el agua "prefiere" avanzar
           por cauces y a favor de la pendiente/drenaje.
           (La versión inicial incluía además rugosidad del terreno vía
           reduceNeighborhood(), sacada luego por costo de cómputo — ver
           nota de rendimiento en calculate_resistance()).

Limitación conocida y asumida a propósito (ver charla de diseño):
    Earth Engine no tiene cost-distance direccional/anisotrópico nativo.
    El "sesgo de dirección de flujo" de este módulo es una APROXIMACIÓN:
    compara el aspect de cada celda (dirección de máxima pendiente
    descendente) contra la dirección hacia el agua conocida más cercana
    (gradiente del distance-transform), y penaliza el avance cuando no
    coinciden. No es un flow-routing D8 real, pero orienta la expansión
    en el sentido del escurrimiento natural en vez de un círculo parejo.

Perfiles regionales:
    En vez de un diccionario fijo por región (Iberá, Chaco, Delta del
    Paraná...), infer_regional_profile() muestrea estadísticas reales de
    la región consultada (fracción de humedal, de bosque, pendiente media,
    densidad de drenaje) y devuelve factores de ajuste CONTINUOS. Esto
    evita mantener polígonos de región a mano y no tiene saltos raros en
    el límite entre dos zonas — generaliza a cualquier punto del país.

Calibración automática contra imágenes satelitales históricas: DESCARTADA
a pedido explícito (demasiado pesada para la app — requiere jobs de fondo
y persistencia de parámetros que no están resueltos en la infraestructura
actual). Ver charla de diseño en la sesión donde se armó este módulo.

Todas las funciones reciben imágenes/geometrías de Earth Engine ya
preparadas (DEM, HAND, agua_fuente) — este módulo no llama a
ee.Initialize() ni depende de Flask en ningún punto.
"""

import math

import ee

# ──────────────────────────────────────────────────────────────────────────
# PARÁMETROS DEL MOTOR (ajustables sin tocar la lógica)
# ──────────────────────────────────────────────────────────────────────────

# Pendiente (en grados) a partir de la cual se considera resistencia máxima
# por pendiente. 15° es una pendiente ya notoria para agua superficial
# lenta (crecida de río, no torrente de montaña).
RESISTENCIA_PENDIENTE_MAX_DEG = 15.0

# Códigos de ESA WorldCover v200 relevantes para el índice de resistencia.
WORLDCOVER_ARBOLADO = 10
WORLDCOVER_URBANO = 50
WORLDCOVER_HUMEDAL_HERBACEO = 90

# Pesos del índice de resistencia hidráulica combinado (deben sumar 1.0).
# NOTA DE RENDIMIENTO: originalmente incluía un término de "rugosidad" vía
# reduceNeighborhood() con kernel circular -- se sacó (ver
# calculate_resistance) porque las operaciones de vecindad son mucho más
# caras que operaciones pixel-a-pixel en Earth Engine, y en la práctica
# estaban provocando error 429 (rate limit) del tile server con carga
# normal de uso (varias teselas pedidas en simultáneo al mover/hacer zoom
# en el simulador). Su peso se redistribuyó entre los criterios restantes.
PESOS_RESISTENCIA = {
    'pendiente': 0.35,
    'cobertura': 0.25,
    'direccion': 0.25,
    'acumulacion': 0.15,
}

# Escala de trabajo (metros) para las operaciones de costo/reproject —
# misma escala que ya usaba _conectividad_hidraulica en app.py, para no
# cambiar el balance rendimiento/precisión que ya estaba calibrado.
ESCALA_COSTO_M = 90

# Tope de distancia de propagación (mismo valor que el original en app.py).
MAX_DIST_CONECTIVIDAD_M = 50_000


# ──────────────────────────────────────────────────────────────────────────
# CRITERIOS INDIVIDUALES
# ──────────────────────────────────────────────────────────────────────────

def calculate_slope(elevacion):
    """Pendiente en grados (wrapper directo de ee.Terrain.slope, para que
    todo el módulo pase por esta función y quede documentado en un solo
    lugar de dónde sale)."""
    return ee.Terrain.slope(elevacion)


def calculate_flow_direction_bias(elevacion, agua_fuente, region, escala=ESCALA_COSTO_M):
    """
    Sesgo direccional aproximado, 0-1 (1 = el aspect de la celda apunta
    hacia el agua conocida más cercana -> favorece avance; 0 = apunta en
    contra -> penaliza avance).

    Método: se compara el vector unitario del aspect (dirección de máxima
    pendiente descendente) contra el vector que apunta hacia el agua más
    cercana, obtenido del gradiente de un distance-transform a la fuente de
    agua (el gradiente de "distancia a X" apunta en el sentido de ALEJARSE
    de X; el sentido hacia el agua es el opuesto, por eso se invierte).
    La similitud entre ambos vectores (coseno) se normaliza a 0-1.

    RENDIMIENTO: elevación y agua_fuente se reproyectan a `escala` (90 m
    por defecto, igual que el resto del motor) ANTES de aspect/distance
    transform/gradient. Esto no es cosmético: hacer estas operaciones a
    la resolución nativa de SRTM (30 m) multiplica por 9 la cantidad de
    píxeles que Earth Engine tiene que procesar por tesela, comparado con
    90 m -- eso, sumado al radio de búsqueda del distance transform, fue
    lo que estaba disparando error 429 (rate limit) del tile server bajo
    uso normal. El radio de búsqueda también se redujo de 256 a 96
    píxeles (a 90 m de escala, ~8.6 km -- de sobra para esta aproximación,
    que no necesita más alcance que el de la propia propagación).

    Ver limitación conocida en el docstring del módulo: esto no reemplaza
    un flow-routing D8 real, es una aproximación intencional.
    """
    elevacion_90 = elevacion.reproject(crs='EPSG:4326', scale=escala)
    agua_90 = agua_fuente.reproject(crs='EPSG:4326', scale=escala)

    aspect_rad = ee.Terrain.aspect(elevacion_90).multiply(math.pi / 180.0)
    dx_terreno = aspect_rad.sin()
    dy_terreno = aspect_rad.cos()

    distancia_agua = agua_90.fastDistanceTransform(96).sqrt()
    gradiente = distancia_agua.gradient()
    norma = gradiente.select('x').hypot(gradiente.select('y')).max(1e-6)
    dx_agua = gradiente.select('x').divide(norma).multiply(-1)
    dy_agua = gradiente.select('y').divide(norma).multiply(-1)

    alineacion = (dx_terreno.multiply(dx_agua)
                  .add(dy_terreno.multiply(dy_agua)))
    bias = alineacion.add(1).divide(2)
    return bias.rename('flow_direction_bias')


def calculate_flow_accumulation_weight(region):
    """
    Peso de acumulación de flujo, 0-1, a partir de MERIT Hydro banda 'upa'
    (área de drenaje acumulada aguas arriba, km²). Valores altos = celda
    está sobre o cerca de un cauce real -> debería inundarse antes.
    Se usa log10 porque 'upa' tiene un rango muy asimétrico (de <1 km² a
    miles de km² en los cauces principales).
    """
    upa = ee.Image('MERIT/Hydro/v1_0_1').select('upa').clip(region)
    log_upa = upa.add(1).log10()
    return log_upa.unitScale(0, 4).clamp(0, 1).rename('flow_accum_weight')


def calculate_resistance(elevacion, region, flow_dir_bias, flow_accum_weight,
                          escala=ESCALA_COSTO_M):
    """
    Índice de resistencia hidráulica combinado, 0-1 (0 = el agua avanza
    libre, 1 = resistencia máxima). Combina pendiente, cobertura del suelo
    (bosque/urbano como proxy de rugosidad de Manning), y el INVERSO del
    sesgo direccional y del peso de acumulación (una celda bien alineada
    con el drenaje real, o sobre un cauce, tiene BAJA resistencia).

    RENDIMIENTO: la elevación se reproyecta a `escala` (90 m) antes de
    calcular la pendiente, por la misma razón que en
    calculate_flow_direction_bias -- 9x menos píxeles que a la resolución
    nativa de SRTM. Se sacó el término de rugosidad por vecindad
    (reduceNeighborhood con kernel circular) que tenía la primera versión:
    era, con diferencia, la operación más cara del motor y estaba
    provocando error 429 (rate limit) del tile server de Earth Engine bajo
    uso normal. Ver nota en PESOS_RESISTENCIA sobre cómo se redistribuyó
    su peso.
    """
    elevacion_90 = elevacion.reproject(crs='EPSG:4326', scale=escala)
    pendiente = calculate_slope(elevacion_90)
    lc = (ee.ImageCollection('ESA/WorldCover/v200')
            .mosaic().select('Map').clip(region))
    es_urbano = lc.eq(WORLDCOVER_URBANO)
    es_bosque = lc.eq(WORLDCOVER_ARBOLADO)

    r_pendiente = pendiente.divide(RESISTENCIA_PENDIENTE_MAX_DEG).clamp(0, 1)
    r_cobertura = (ee.Image(0.1)
                   .where(es_bosque, 0.6)
                   .where(es_urbano, 0.9))
    r_direccion = ee.Image(1).subtract(flow_dir_bias)
    r_acumulacion = ee.Image(1).subtract(flow_accum_weight)

    resistencia = (r_pendiente.multiply(PESOS_RESISTENCIA['pendiente'])
                   .add(r_cobertura.multiply(PESOS_RESISTENCIA['cobertura']))
                   .add(r_direccion.multiply(PESOS_RESISTENCIA['direccion']))
                   .add(r_acumulacion.multiply(PESOS_RESISTENCIA['acumulacion'])))
    return resistencia.clamp(0, 1).rename('resistencia')


def calculate_hydraulic_distance(costo_superficie, agua_fuente, radius_km,
                                  perfil_factor_alcance=1.0,
                                  escala=ESCALA_COSTO_M):
    """
    Distancia hidráulica ponderada: envuelve ee.Image.cumulativeCost(), pero
    ahora la superficie de costo recibida ya no es binaria (1 por celda
    candidata) sino continua (1 + resistencia), así que el camino de menor
    costo ya no es necesariamente el más corto en línea recta/pixeles, sino
    el de menor resistencia acumulada — más parecido a una distancia
    hidráulica real que a una distancia euclidiana o de grilla pura.
    """
    max_dist_m = min(radius_km * 1000, MAX_DIST_CONECTIVIDAD_M * perfil_factor_alcance)

    costo_90 = costo_superficie.reproject(crs='EPSG:4326', scale=escala)
    fuente_90 = agua_fuente.reproject(crs='EPSG:4326', scale=escala)

    return costo_90.cumulativeCost(source=fuente_90, maxDistance=max_dist_m)


# ──────────────────────────────────────────────────────────────────────────
# PERFIL REGIONAL (inferido, no una lista fija de regiones)
# ──────────────────────────────────────────────────────────────────────────

def infer_regional_profile(region, escala=200):
    """
    Infiere factores de ajuste continuos a partir de estadísticas reales de
    la región consultada — NO de una lista fija de regiones con nombre
    (Iberá, Chaco, Delta del Paraná...). Ventajas sobre una lista fija:
    generaliza a cualquier punto sin mantener polígonos a mano, y no
    produce saltos artificiales en el límite entre dos "zonas".

    Variables muestreadas:
      frac_humedal      -> fracción de humedal herbáceo (WorldCover 90) o
                            agua de ocurrencia media (JRC > 30%)
      frac_bosque       -> fracción de bosque/arbolado (WorldCover 10)
      pendiente_media    -> pendiente media de la región, en grados
      frac_drenaje_denso -> fracción con acumulación de flujo alta (MERIT
                            'upa' >= 10 km²), proxy de densidad de red

    Factores devueltos (todos como multiplicadores sobre el valor base):
      factor_umbral_hand  -> en zonas de humedal, una subida chica de HAND
                              ya representa una inundación real (terreno
                              casi plano, saturado) -> reduce el umbral
                              efectivo.
      factor_resistencia  -> más bosque o más pendiente media -> más
                              resistencia general al avance.
      factor_alcance      -> red de drenaje más densa -> el agua se
                              distribuye más lejos del punto de origen
                              (hay más "caminos" de baja resistencia).
    """
    lc = (ee.ImageCollection('ESA/WorldCover/v200')
            .mosaic().select('Map').clip(region))
    jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence').clip(region)
    dem_crudo = ee.Image('USGS/SRTMGL1_003').select('elevation').clip(region)
    pendiente = ee.Terrain.slope(dem_crudo)
    upa = ee.Image('MERIT/Hydro/v1_0_1').select('upa').clip(region)

    frac_humedal_img = (lc.eq(WORLDCOVER_HUMEDAL_HERBACEO)
                         .Or(jrc.gt(30))).rename('humedal')
    frac_bosque_img = lc.eq(WORLDCOVER_ARBOLADO).rename('bosque')
    frac_drenaje_img = upa.gte(10).rename('drenaje_denso')

    muestra = ee.Image.cat([
        frac_humedal_img, frac_bosque_img,
        pendiente.rename('pendiente'), frac_drenaje_img,
    ]).reduceRegion(
        reducer=ee.Reducer.mean(), geometry=region, scale=escala,
        bestEffort=True, maxPixels=1e9, tileScale=4,
    ).getInfo()

    frac_humedal = muestra.get('humedal') or 0.0
    frac_bosque = muestra.get('bosque') or 0.0
    pendiente_media = muestra.get('pendiente') or 0.0
    frac_drenaje = muestra.get('drenaje_denso') or 0.0

    factor_umbral_hand = max(1.0 - 0.3 * frac_humedal, 0.5)
    factor_resistencia = 1.0 + 0.5 * frac_bosque + 0.2 * min(pendiente_media / 5.0, 1.0)
    factor_alcance = 1.0 + 0.4 * frac_drenaje

    return {
        'frac_humedal': round(frac_humedal, 3),
        'frac_bosque': round(frac_bosque, 3),
        'pendiente_media_deg': round(pendiente_media, 2),
        'frac_drenaje_denso': round(frac_drenaje, 3),
        'factor_umbral_hand': round(factor_umbral_hand, 3),
        'factor_resistencia': round(factor_resistencia, 3),
        'factor_alcance': round(factor_alcance, 3),
    }


# ──────────────────────────────────────────────────────────────────────────
# SIMULACIÓN COMPLETA
# ──────────────────────────────────────────────────────────────────────────

def simulate_flood(region, hand, agua_fuente, radius_km, umbral_m, perfil=None):
    """
    Corre la simulación multicriterio completa y devuelve un diccionario
    con la MISMA forma de información que la _conectividad_hidraulica()
    original de app.py (candidatas, zona_conectada, costo_acumulado,
    conectividad_ok), más los datos nuevos (perfil, umbral efectivo) para
    que el endpoint los agregue a la respuesta JSON de forma opcional, sin
    romper a ningún consumidor del frontend que no los espere.
    """
    if perfil is None:
        perfil = infer_regional_profile(region)

    umbral_efectivo = umbral_m * perfil['factor_umbral_hand']

    dem_crudo = ee.Image('USGS/SRTMGL1_003').select('elevation').clip(region)

    flow_dir_bias = calculate_flow_direction_bias(dem_crudo, agua_fuente, region)
    flow_accum_weight = calculate_flow_accumulation_weight(region)
    resistencia = calculate_resistance(dem_crudo, region, flow_dir_bias, flow_accum_weight)
    resistencia = resistencia.multiply(perfil['factor_resistencia']).clamp(0, 1)

    candidatas = hand.lte(umbral_efectivo)

    costo_superficie = candidatas.selfMask().add(resistencia.multiply(4))

    costo_acumulado = calculate_hydraulic_distance(
        costo_superficie, agua_fuente, radius_km,
        perfil_factor_alcance=perfil['factor_alcance'],
    )

    zona_conectada = candidatas.updateMask(costo_acumulado.gte(0)).selfMask()

    hay_agua_fuente = agua_fuente.reduceRegion(
        reducer=ee.Reducer.anyNonZero(), geometry=region, scale=ESCALA_COSTO_M,
        bestEffort=True, maxPixels=1e9, tileScale=4,
    ).get('occurrence')
    conectividad_ok = ee.Algorithms.If(hay_agua_fuente, True, False).getInfo()

    return {
        'candidatas': candidatas,
        'zona_conectada': zona_conectada,
        'costo_acumulado': costo_acumulado,
        'conectividad_ok': conectividad_ok,
        'umbral_efectivo_m': round(umbral_efectivo, 2),
        'perfil': perfil,
    }


def generate_animation_frames(costo_acumulado, candidatas, region, frames,
                               escala_max_costo=150):
    """
    Reparte 'costo_acumulado' en `frames` bandas proporcionales, igual que
    hacía /inundacion_animacion original con distancia en metros.
    """
    max_costo_dict = costo_acumulado.reduceRegion(
        reducer=ee.Reducer.percentile([95]),
        geometry=region, scale=escala_max_costo,
        bestEffort=True, maxPixels=1e9, tileScale=4,
    ).getInfo()
    max_costo = 0
    for v in max_costo_dict.values():
        if v:
            max_costo = v
            break
    if not max_costo or max_costo <= 0:
        return None

    fotogramas = []
    for i in range(1, frames + 1):
        umbral_costo = max_costo * (i / frames)
        frame_img = candidatas.updateMask(costo_acumulado.lte(umbral_costo)).selfMask()
        fotogramas.append({
            'orden': i,
            'valor_costo_acumulado': round(umbral_costo, 2),
            'porcentaje': round(100 * i / frames, 1),
            'imagen': frame_img,
        })
    return fotogramas, max_costo
