"""
TerraOS - Backend de Google Earth Engine
==========================================
Expone endpoints REST que devuelven URLs de teselas (tiles) de Earth Engine
para mostrar capas satelitales (NDVI, EVI, NBR, Falso Color, Térmico) y un
índice combinado de "riesgo de incendio" sobre el mapa de TerraOS.

La app NO requiere que los usuarios finales tengan cuenta de Google ni de
Earth Engine: todo se autentica con UNA cuenta de servicio (service account)
configurada en el servidor.

Variables de entorno necesarias:
    GEE_SERVICE_ACCOUNT_JSON  -> contenido completo del archivo JSON de la
                                  cuenta de servicio (como string)
    GEE_PROJECT_ID            -> ID del proyecto de Google Cloud
                                  (ej: terraos-12345)

Ver README.md para instrucciones de creación de la cuenta de servicio
y despliegue.
"""

import os
import json
import math
import time
import threading
import datetime
import ee
import requests
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

# ── FloodSimulationEngine ──────────────────────────────────────────────
# Motor de simulación de inundación multicriterio (pendiente, sesgo de
# dirección de flujo, acumulación de flujo, distancia hidráulica, índice
# de resistencia, perfil regional inferido). Importado con fallback: si
# el archivo no está presente en el deploy por algún motivo, la app sigue
# funcionando con la conectividad hidráulica original (_conectividad_hidraulica
# más abajo), solo que sin las mejoras multicriterio.
try:
    import flood_simulation_engine as _motor_inundacion
    _MOTOR_DISPONIBLE = True
except ImportError:
    _motor_inundacion = None
    _MOTOR_DISPONIBLE = False

app = Flask(__name__)
CORS(app)  # permite que TerraOS_v4b.html (servido desde otro dominio) consuma la API

# ──────────────────────────────────────────────────────────────────────────
# MÓDULO OPCIONAL: hidrografia_vectorial
# ──────────────────────────────────────────────────────────────────────────
# Este módulo local (hidrografia_vectorial.py, junto a este app.py) solo lo
# usa /iipdi_punto para identificar el cuerpo de agua más cercano. Si el
# archivo no está desplegado junto con este app.py, un `import` normal a
# nivel de módulo tira ImportError ANTES de que Flask llegue a registrar
# ninguna ruta — es decir, revienta la app ENTERA, incluidas /zonas_inundables,
# /inundacion_tiles, /inundacion_animacion, etc., aunque no tengan nada que
# ver con este módulo. Por eso el import queda protegido: si falta, solo se
# desactiva /iipdi_punto (con un mensaje claro) y todo lo demás sigue andando.
try:
    import hidrografia_vectorial as hidro_vectorial
    _hidro_vectorial_disponible = True
except ImportError as _exc_hidro:
    hidro_vectorial = None
    _hidro_vectorial_disponible = False
    print(f'⚠️ Módulo hidrografia_vectorial no disponible ({_exc_hidro}). '
          f'Asegurate de que hidrografia_vectorial.py esté en el mismo repo/'
          f'directorio que app.py y se haya desplegado junto con él. '
          f'El resto de los endpoints (incluida la simulación de inundación) '
          f'funciona igual; solo /iipdi_punto queda deshabilitado hasta que '
          f'el módulo esté presente.')

# ──────────────────────────────────────────────────────────────────────────
# INICIALIZACIÓN DE EARTH ENGINE
# ──────────────────────────────────────────────────────────────────────────
_ee_ready = False
_ee_error = None


def init_earth_engine():
    global _ee_ready, _ee_error
    try:
        raw = os.environ.get('GEE_SERVICE_ACCOUNT_JSON')
        project_id = os.environ.get('GEE_PROJECT_ID')

        if not raw:
            raise RuntimeError(
                'Falta la variable de entorno GEE_SERVICE_ACCOUNT_JSON '
                '(contenido del JSON de la cuenta de servicio).'
            )

        service_account_info = json.loads(raw)
        credentials = ee.ServiceAccountCredentials(
            service_account_info['client_email'],
            key_data=raw
        )
        ee.Initialize(
            credentials,
            project=project_id or service_account_info.get('project_id')
        )
        _ee_ready = True
        _ee_error = None
        print('✅ Earth Engine inicializado correctamente.')
    except Exception as exc:  # noqa: BLE001
        _ee_ready = False
        _ee_error = str(exc)
        print(f'⚠️ Earth Engine NO se pudo inicializar: {exc}')


init_earth_engine()

# ──────────────────────────────────────────────────────────────────────────
# PALETAS DE VISUALIZACIÓN
# ──────────────────────────────────────────────────────────────────────────
PALETA_VEGETACION = ['#a52a2a', '#d2b48c', '#ffff66', '#9acd32', '#006400']
PALETA_NBR = ['#7f0000', '#ff0000', '#ffff66', '#66cc66', '#003300']
PALETA_TERMICO = ['#313695', '#74add1', '#fee090', '#f46d43', '#a50026']
PALETA_RIESGO = ['#16a34a', '#facc15', '#ea580c', '#dc2626']

# Slider "poca red" (1) -> "red densa" (5) de la UI, mapeado al umbral de
# área de drenaje acumulada (km²) de MERIT Hydro (banda 'upa'). Valores más
# bajos de umbral dejan pasar más píxeles -> red más densa/ramificada.
# NOTA: en zonas de llanura/esteros (ej. Corrientes) el drenaje es difuso,
# así que umbrales bajos (los que se usaban antes: 1-5 km²) fusionan casi
# toda la región en un solo polígono ancho. Se subieron los umbrales para
# que incluso el nivel 5 muestre solo cauces con acumulación real.
MERIT_HYDRO_UMBRALES_KM2 = {1: 500, 2: 200, 3: 80, 4: 30, 5: 10}

# Tope de seguridad para radius_km en /inundacion_tiles y /inundacion_animacion.
# NOTA: se elevó a 800 km a pedido explícito — con esto puede reaparecer el
# error 400 de Earth Engine en tipo='zona_baja'/'anegamiento'
# para radios grandes, porque _conectividad_hidraulica() hace .reproject()
# + cumulativeCost() sobre TODA la región recortada (cálculo "eager", no por
# tile), y esa combinación no escala bien a regiones de cientos de km.
RADIUS_KM_MAX_INUNDACION = 800

# ──────────────────────────────────────────────────────────────────────────
# HELPERS POR DATASET
# Cada función recibe (coleccion_filtrada, geometria) y devuelve
# (ee.Image, vis_params) listos para getMapId()
# ──────────────────────────────────────────────────────────────────────────

def _mask_s2_clouds(image):
    """
    Aplica máscara de nubes/sombras usando la banda SCL de Sentinel-2 SR.
    SCL valores a enmascarar:
      3  = nubes bajas
      8  = nubes de media probabilidad
      9  = nubes de alta probabilidad
      10 = cirrus
      11 = nieve/hielo  (opcional, útil en zonas sin nieve)
    Los píxeles enmascarados quedan transparentes y son ignorados por .mosaic(),
    permitiendo que la siguiente imagen en el stack los complete.
    """
    scl = image.select('SCL')
    mask = (scl.neq(3)
              .And(scl.neq(8))
              .And(scl.neq(9))
              .And(scl.neq(10)))
    return image.updateMask(mask)


def _band_sentinel(col, band):
    # 1) Enmascarar píxeles nublados en CADA imagen antes de compositar.
    #    Sin este paso, .mosaic() usa la imagen menos nublada COMPLETA —
    #    incluyendo sus píxeles nublados — y tapa escenas más limpias debajo.
    # 2) Ordenar por nubosidad ascendente: la imagen con menos nubes queda
    #    encima en el stack, maximizando píxeles limpios en zonas de overlap.
    # 3) .mosaic() completa cada píxel con la imagen más arriba del stack
    #    que tenga dato válido (no enmascarado) → mosaico sin huecos.
    col_masked = col.map(_mask_s2_clouds)
    img = col_masked.sort('CLOUDY_PIXEL_PERCENTAGE').mosaic()

    if band == 'NDVI':
        return img.normalizedDifference(['B8', 'B4']).rename('idx'), \
            {'min': -0.2, 'max': 0.9, 'palette': PALETA_VEGETACION}
    if band == 'EVI':
        refl = img.select(['B2', 'B4', 'B8']).multiply(0.0001)  # DN -> reflectancia 0-1
        idx = refl.expression(
            '2.5 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)',
            {'NIR': refl.select('B8'), 'RED': refl.select('B4'), 'BLUE': refl.select('B2')}
        ).rename('idx')
        return idx, {'min': -0.2, 'max': 0.9, 'palette': PALETA_VEGETACION}
    if band == 'NBR':
        return img.normalizedDifference(['B8', 'B12']).rename('idx'), \
            {'min': -0.5, 'max': 0.8, 'palette': PALETA_NBR}
    if band == 'FalseColor':
        return img.select(['B8', 'B4', 'B3']), {'min': 0, 'max': 3500, 'gamma': 1.2}
    if band == 'Thermal':
        raise ValueError(
            'Sentinel-2 no tiene banda térmica. Probá con MODIS, VIIRS o Landsat para "Térmico".'
        )
    raise ValueError(f'Banda no soportada para Sentinel: {band}')


def _band_landsat(col, band):
    # Bandas ópticas SR: reflectancia = DN * 0.0000275 - 0.2
    # Banda térmica ST_B10: temperatura(K) = DN * 0.00341802 + 149.0
    def optico(im):
        opt = im.select('SR_B.*').multiply(0.0000275).add(-0.2)
        term = im.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15)
        return im.addBands(opt, overwrite=True).addBands(term.rename('ST_C'), overwrite=True)

    # FIX: mismo criterio que en Sentinel — mosaico ordenado por nubosidad
    # en vez de .median(), para no dejar huecos sin datos en el mapa.
    img = col.map(optico).sort('CLOUD_COVER').mosaic()

    if band == 'NDVI':
        return img.normalizedDifference(['SR_B5', 'SR_B4']).rename('idx'), \
            {'min': -0.2, 'max': 0.9, 'palette': PALETA_VEGETACION}
    if band == 'EVI':
        idx = img.expression(
            '2.5 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)',
            {'NIR': img.select('SR_B5'), 'RED': img.select('SR_B4'), 'BLUE': img.select('SR_B2')}
        ).rename('idx')
        return idx, {'min': -0.2, 'max': 0.9, 'palette': PALETA_VEGETACION}
    if band == 'NBR':
        return img.normalizedDifference(['SR_B5', 'SR_B7']).rename('idx'), \
            {'min': -0.5, 'max': 0.8, 'palette': PALETA_NBR}
    if band == 'FalseColor':
        return img.select(['SR_B5', 'SR_B4', 'SR_B3']), {'min': 0, 'max': 0.4, 'gamma': 1.2}
    if band == 'Thermal':
        return img.select('ST_C'), {'min': 10, 'max': 55, 'palette': PALETA_TERMICO}
    raise ValueError(f'Banda no soportada para Landsat: {band}')


def _band_modis(col_ndvi, col_refl, col_lst, band):
    if band == 'NDVI':
        img = col_ndvi.median().select('NDVI').multiply(0.0001)
        return img.rename('idx'), {'min': -0.2, 'max': 0.9, 'palette': PALETA_VEGETACION}
    if band == 'EVI':
        img = col_ndvi.median().select('EVI').multiply(0.0001)
        return img.rename('idx'), {'min': -0.2, 'max': 0.9, 'palette': PALETA_VEGETACION}
    if band in ('NBR', 'FalseColor'):
        refl = col_refl.median().multiply(0.0001)
        nir, red, green, swir2 = (refl.select('sur_refl_b02'), refl.select('sur_refl_b01'),
                                   refl.select('sur_refl_b04'), refl.select('sur_refl_b07'))
        if band == 'NBR':
            idx = nir.subtract(swir2).divide(nir.add(swir2)).rename('idx')
            return idx, {'min': -0.5, 'max': 0.8, 'palette': PALETA_NBR}
        return ee.Image.cat([nir, red, green]), {'min': 0, 'max': 0.4, 'gamma': 1.2}
    if band == 'Thermal':
        img = col_lst.median().select('LST_Day_1km').multiply(0.02).subtract(273.15)
        return img.rename('idx'), {'min': 10, 'max': 55, 'palette': PALETA_TERMICO}
    raise ValueError(f'Banda no soportada para MODIS: {band}')


def _band_viirs(col_ndvi, col_refl, col_lst, band):
    if band == 'NDVI':
        img = col_ndvi.median().select('NDVI').multiply(0.0001)
        return img.rename('idx'), {'min': -0.2, 'max': 0.9, 'palette': PALETA_VEGETACION}
    if band == 'EVI':
        img = col_ndvi.median().select('EVI').multiply(0.0001)
        return img.rename('idx'), {'min': -0.2, 'max': 0.9, 'palette': PALETA_VEGETACION}
    if band in ('NBR', 'FalseColor'):
        refl = col_refl.median().multiply(0.0001)
        nir, swir, red, green = (refl.select('I2'), refl.select('I3'),
                                  refl.select('I1'), refl.select('M4'))
        if band == 'NBR':
            idx = nir.subtract(swir).divide(nir.add(swir)).rename('idx')
            return idx, {'min': -0.5, 'max': 0.8, 'palette': PALETA_NBR}
        return ee.Image.cat([nir, red, green]), {'min': 0, 'max': 0.4, 'gamma': 1.2}
    if band == 'Thermal':
        img = col_lst.median().select('LST_1KM').multiply(0.02).subtract(273.15)
        return img.rename('idx'), {'min': 10, 'max': 55, 'palette': PALETA_TERMICO}
    raise ValueError(f'Banda no soportada para VIIRS: {band}')


# ──────────────────────────────────────────────────────────────────────────
# HELPERS HIDROLÓGICOS COMPARTIDOS (usados por /inundacion_tiles y
# /inundacion_animacion — factorizados acá para no duplicar lógica)
# ──────────────────────────────────────────────────────────────────────────

def _dem_composite_correccion_dosel(elevacion, region):
    """
    Motor de corrección de dosel v3.0 (adaptado de 'Elevación corregida.js').

    Corrige el sesgo de altura de copa de árboles en el DEM SRTM. SRTM es un
    DSM (Digital Surface Model): el radar mide la altura de lo primero que
    encuentra, que en zonas de bosque cerrado es la copa de los árboles, no
    el suelo real. En parques con monte denso (ej. Mburucuyá) esto genera
    "lomas" falsas que no existen en el terreno, distorsionando la
    simulación de inundación (el agua "rebota" contra un bosque que en el
    DEM parece una loma, cuando en realidad puede ser una zona baja).

    A diferencia de la versión interina anterior (que solo usaba NDVI como
    proxy de "esto es bosque"), esta versión usa:

      1. CHM real (ETH Global Canopy Height 2020, 10 m — Lang et al. 2023)
         en vez de un proxy.
      2. Factor de corrección DINÁMICO por píxel: f(CHM, pendiente,
         rugosidad) — no un porcentaje fijo arbitrario. En laderas y
         terreno rugoso se corrige menos (para no "aplanar" relieve real).
      3. Filtro de bosque perdido después de 2005 (Hansen Global Forest
         Change): si el bosque ya no está, el CHM de 2005 no aplica y no
         se resta nada.
      4. Corrección específica para EDIFICIOS (ESA WorldCover, clase
         urbana): el CHM de vegetación no sirve acá (mide dosel, no
         techo — en ciudad normalmente da ~0 aunque haya un edificio de
         20 pisos). Se usa el dataset GHSL Building Height (altura media
         real de edificios, 100 m) en su lugar, con un factor de
         corrección propio (0.70) y de forma MUTUAMENTE EXCLUYENTE con
         la corrección de vegetación: cada píxel recibe una sola.
      5. Validación física: si la pendiente resultante post-corrección es
         > 75° o cambió más de 40° respecto de la original, se descarta la
         corrección en ese píxel (es un artefacto, no relieve real).
      6. Suavizado condicional SOLO en bordes artificiales (donde la
         corrección crea un salto > 3 m en 30 m), para no introducir
         "acantilados" que rompan el ruteo hidrológico (Manning).
      7. Índice de confianza por píxel (1=baja, 2=media, 3=alta), útil para
         saber qué tan confiable es la elevación corregida en cada zona.

    Devuelve un ee.Image con las bandas:
      correctedTerrain, confidence, canopyHeight, canopyHeightLimitado,
      buildingHeight, correctionFactor, correctionApplied, landCover
    """
    # 1. Datasets auxiliares (recortados a la región para no traer info de más)
    lc = (ee.ImageCollection('ESA/WorldCover/v200')
            .mosaic().select('Map').clip(region))
    # CHM: ETH Global Canopy Height 2020 (Lang et al. 2023, Nature Ecology &
    # Evolution) — 10 m de resolución real, Sentinel-2 + GEDI. Reemplaza al
    # dataset anterior (NASA/JPL/global_forest_canopy_height_2005, ~1 km,
    # año 2005), que a escala de parque/predio quedaba directamente
    # desalineado con la vegetación real (ej. marcaba 0 m de dosel arriba
    # de un árbol y 16 m en el pastizal de al lado, en Mburucuyá).
    chm = (ee.Image('users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1')
             .rename('chmRaw').clip(region))
    hansen = ee.Image('UMD/hansen/global_forest_change_2023_v1_11').clip(region)
    loss_year = hansen.select('lossyear')  # 1-23 = pérdida en 2001-2023

    # Altura de edificios GHSL (JRC Global Human Settlement Layer, 2018,
    # resolución nativa 100m; GEE la remuestrea automáticamente a 30m al
    # operar con el DEM). Se usa EXCLUSIVAMENTE en zonas urbanas.
    building_height = (ee.ImageCollection('JRC/GHSL/P2023A/GHS_BUILT_H')
                          .first()
                          .select('built_height')
                          .clip(region)
                          .rename('buildingHeight'))
    # Ruido: edificios <2m se consideran error de medición, no edificio real.
    building_height = building_height.where(building_height.lt(2), 0)
    # Tope: evita valores extremos anómalos del dataset.
    building_height = building_height.min(60)
    building_factor = 0.70  # fracción de la altura del edificio que se resta del DEM

    # 2. Variables topográficas de apoyo
    pendiente = ee.Terrain.slope(elevacion)
    kernel_90 = ee.Kernel.circle(radius=90, units='meters')

    rugosidad = elevacion.reduceNeighborhood(
        reducer=ee.Reducer.stdDev(), kernel=kernel_90,
    ).rename('roughness')

    std_local = elevacion.reduceNeighborhood(
        reducer=ee.Reducer.stdDev(), kernel=ee.Kernel.circle(radius=30, units='meters'),
    ).rename('stdLocal')

    # Desnivel local (rango en 90 m) — para limitar CHM contra anomalías
    local_relief = (elevacion.focal_max(radius=90, units='meters', kernelType='circle')
                    .subtract(elevacion.focal_min(radius=90, units='meters', kernelType='circle'))
                    .rename('localRelief'))

    # 3. Detección de cambios y edificios
    bosque_perdido_reciente = loss_year.gt(15)  # pérdida posterior a 2015
    es_urbano = lc.eq(50)                        # WorldCover: 50 = construido
    # No vegetado según WorldCover: construido(50), desnudo/disperso(60),
    # nieve/hielo(70), agua permanente(80). Todo lo demás (árboles, arbustos,
    # pastizal, cultivo, humedal herbáceo, manglar, musgo/liquen) se
    # considera vegetación real y NO se le aplica el cap por relieve local.
    es_no_vegetado = lc.eq(50).Or(lc.eq(60)).Or(lc.eq(70)).Or(lc.eq(80))

    # 4. Factor dinámico de corrección
    # Limitar CHM si supera 1.5x el desnivel local, PERO solo en superficies
    # confirmadas NO vegetadas (ver es_no_vegetado arriba). Este límite
    # busca descartar lecturas de CHM que sean ruido del dataset sobre
    # terreno sin vegetación, pero en cualquier cobertura vegetal real
    # (monte, pastizal, humedal, palmar) el relieve local del DSM crudo
    # puede salir bajo aunque el dosel sea alto y parejo -- eso no es un
    # error del CHM, es la firma normal de una cobertura uniforme. Con el
    # CHM de 10 m (ETH 2020) ya no hace falta desconfiar de esos casos.
    cap_anomalia_chm = local_relief.multiply(1.5).max(2)
    chm_limitado = chm.where(es_no_vegetado, chm.min(cap_anomalia_chm))

    base_factor = chm_limitado.divide(chm_limitado.add(15)).multiply(0.75)  # saturación
    slope_penalty = ee.Image(1).subtract(pendiente.divide(60).clamp(0, 1))   # menos corrección en laderas
    rough_penalty = ee.Image(1).subtract(rugosidad.divide(10).clamp(0, 1))   # menos corrección en terreno rugoso
    factor_dinamico = base_factor.multiply(slope_penalty).multiply(rough_penalty)

    # En bosque recién perdido no restamos nada (el árbol ya no está).
    # El factor urbano ya NO se aplica acá — los edificios se corrigen
    # aparte con GHSL más abajo (ver "Corrección por edificios").
    factor = (factor_dinamico
              .where(bosque_perdido_reciente, 0)
              .clamp(0, 0.85)
              .rename('correctionFactor'))

    # 5. Corrección parcial del DEM
    # ------------------------------------------------------------------
    # Corrección por vegetación (lógica dinámica existente, basada en CHM)
    correccion_vegetacion = chm_limitado.multiply(factor)

    # Corrección por edificios (GHSL) — solo en píxeles urbanos
    es_urbano_mask = es_urbano  # alias por claridad, ya calculado arriba
    correccion_edificios = (building_height
                             .multiply(building_factor)
                             .updateMask(es_urbano_mask)
                             .rename('buildingCorrection'))

    # Fusión MUTUAMENTE EXCLUYENTE: urbano -> edificios (GHSL), resto -> vegetación (CHM)
    correccion_aplicada = (correccion_vegetacion
                            .where(es_urbano_mask, correccion_edificios)
                            .unmask(0)
                            .rename('correctionApplied'))
    corregido = elevacion.subtract(correccion_aplicada)

    # Banda de diagnóstico: qué fracción se aplicó en cada píxel (vegetación
    # o, en zonas urbanas, el factor fijo de edificios).
    factor = factor.where(es_urbano_mask, building_factor).rename('correctionFactor')

    # 6. Validaciones físicas
    corregido = corregido.max(0).min(elevacion)
    mascara_agua = lc.eq(80)  # WorldCover: 80 = cuerpo de agua permanente
    corregido = corregido.where(mascara_agua, elevacion)

    slope_after = ee.Terrain.slope(corregido)
    slope_diff = slope_after.subtract(pendiente).abs()
    # Pendiente resultante imposible (>75°) o que cambió demasiado (>40°)
    # respecto de la original -> es un artefacto de la corrección, se descarta.
    pendiente_invalida = slope_after.gt(75).Or(slope_diff.gt(40))
    corregido = corregido.where(pendiente_invalida, elevacion)

    # 7. Suavizado condicional (solo donde la corrección dejó un borde
    # artificial de más de 3 m en 30 m; en el resto se deja tal cual, para
    # no perder microtopografía real como cauces y cárcavas).
    rango_local = (corregido.focal_max(radius=30, units='meters', kernelType='circle')
                   .subtract(corregido.focal_min(radius=30, units='meters', kernelType='circle')))
    bordes_artificiales = rango_local.gt(3)
    suavizado = corregido.focal_mean(radius=30, units='meters', kernelType='circle', iterations=1)
    corregido = suavizado.where(bordes_artificiales.Not(), corregido)

    # Re-validar tras el suavizado
    corregido = corregido.max(0).min(elevacion).where(mascara_agua, elevacion)
    corregido = corregido.rename('correctedTerrain')

    # 9. Índice de confianza (multi-factor)
    norm_chm  = chm_limitado.divide(30).clamp(0, 1)
    norm_pend = pendiente.divide(35).clamp(0, 1)
    norm_rug  = rugosidad.divide(15).clamp(0, 1)
    norm_std  = std_local.divide(5).clamp(0, 1)
    penalty_invalida = pendiente_invalida.multiply(0.5)

    score_confianza = (ee.Image(1)
                        .subtract(norm_chm.multiply(0.30))
                        .subtract(norm_pend.multiply(0.25))
                        .subtract(norm_rug.multiply(0.20))
                        .subtract(norm_std.multiply(0.15))
                        .subtract(penalty_invalida)
                        .clamp(0, 1))
    confianza = (score_confianza.multiply(2.99).floor().add(1).toInt()
                 .clamp(1, 3).rename('confidence'))

    return ee.Image.cat([
        corregido,
        confianza,
        chm.rename('canopyHeight'),
        chm_limitado.rename('canopyHeightLimitado'),
        building_height,
        factor,
        correccion_aplicada,
        lc.rename('landCover'),
    ])


def _dem_corregido_por_dosel(elevacion, region):
    """
    Wrapper de compatibilidad: mantiene la misma firma/salida que usan todos
    los endpoints existentes (_preparar_dem_y_agua, /inundacion_punto,
    /direccion_flujo), pero ahora corriendo el motor v3.0 por debajo.
    Devuelve solo la banda de elevación corregida (banda 'elevation'), igual
    que antes.
    """
    return (_dem_composite_correccion_dosel(elevacion, region)
            .select('correctedTerrain')
            .rename('elevation'))


def _correccion_dosel_offset_m(region):
    """
    Cuántos metros se le restaron al SRTM por dosel/edificios en cada
    píxel de la región (raw - corregido). Se reutiliza para aplicar el
    MISMO sesgo a HAND (ver _hand_corregido_por_dosel), en vez de
    recalcular todo el motor de corrección dos veces.
    """
    elevacion_cruda = ee.Image('USGS/SRTMGL1_003').select('elevation').clip(region)
    elevacion_corregida = _dem_corregido_por_dosel(elevacion_cruda, region)
    return elevacion_cruda.subtract(elevacion_corregida).rename('correccionDoselM')


def _hand_corregido_por_dosel(hand, region):
    """
    Aplica la corrección de dosel arbóreo + edificios (pensada
    originalmente para el SRTM) también al HAND (MERIT Hydro) que usa el
    simulador de crecidas del río (zona_baja / anegamiento /
    animación).

    Por qué hace falta aparte: HAND es un producto GLOBAL ya calculado por
    MERIT Hydro (altura sobre el drenaje más cercano) — no es un simple
    filtro sobre el SRTM de este backend, así que corregir el SRTM propio
    NO corrige el HAND automáticamente. Son dos rasters distintos.

    Fundamento de la aproximación: HAND(celda) ≈ elevación(celda) -
    elevación(drenaje más cercano). En la celda de drenaje (agua/río) la
    corrección ya vale ~0 (queda enmascarada por _dem_corregido_por_dosel),
    así que restarle a cada celda la MISMA corrección que le aplicamos a su
    elevación SRTM aproxima bien cuánto de su HAND está inflado por dosel o
    edificios, sin necesitar reconstruir HAND desde cero con FillSinks/D8
    sobre el DEM corregido (Earth Engine no expone ese pipeline para MERIT
    Hydro).

    Limitación conocida: es una aproximación de primer orden, no un HAND
    recalculado con un pipeline hidrológico completo. Corrige el sesgo
    dominante (bosque cerrado / edificios que "tapan" zonas bajas) pero no
    reordena la red de drenaje en sí.
    """
    offset = _correccion_dosel_offset_m(region)
    return hand.subtract(offset).max(0).rename('hnd')


def _preparar_dem_y_agua(lat, lon, radius_km):
    """DEM SRTM (corregido por dosel arbóreo) + agua permanente conocida (JRC)."""
    region        = ee.Geometry.Point([lon, lat]).buffer(radius_km * 1000)
    elevacion_cruda = ee.Image('USGS/SRTMGL1_003').select('elevation').clip(region)
    elevacion       = _dem_corregido_por_dosel(elevacion_cruda, region)
    agua_fuente = (ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
                     .select('occurrence')
                     .gt(50)
                     .clip(region))
    return region, elevacion, agua_fuente


# ──────────────────────────────────────────────────────────────────────────
# HAND (Height Above Nearest Drainage) — reemplaza elevación absoluta como
# criterio de "candidata a inundarse" en zona_baja/anegamiento.
#
# POR QUÉ: con elevación absoluta, "candidata" = toda celda por debajo de
# una cota fija. En una provincia como Corrientes, donde gran parte del
# territorio está entre 50-90 m s.n.m., eso marca como inundable casi toda
# la provincia — no distingue "bajo y en la planicie de inundación real"
# de "bajo pero en una loma lejana sin relación con el río". Sumado a que
# hay agua (esteros/lagunas/arroyos) cada pocos km, la conectividad de 50 km
# terminaba alcanzando prácticamente todo. Resultado: manchas gigantes que
# no se parecen a las inundaciones reales que releva INTA.
#
# HAND (MERIT/Hydro/v1_0_1, banda 'hnd') resuelve esto de raíz: es la altura
# de cada celda POR ENCIMA DEL DRENAJE MÁS CERCANO siguiendo el relieve real
# (no en línea recta ni en cota absoluta). Un punto topográficamente "bajo"
# pero separado del río por una loma tiene HAND alto — cumulativeCost no
# puede atravesar esa loma. Es el método estándar en delineación operacional
# de planicies de inundación (USGS, NOAA), y ya lo usa este mismo backend en
# el módulo de Conectividad Hidráulica / IIPDI (_calcular_conectividad_hidraulica_punto).
#
# NUEVO SIGNIFICADO DE `umbral_m`: deja de ser una cota absoluta (m s.n.m.)
# calibrada contra una estación distante, y pasa a ser DIRECTAMENTE
# "cuántos metros por encima de su cauce normal sube el río" — que es
# exactamente la definición de HAND. Esto también elimina la necesidad de
# calibrar un offset DEM↔estación en el frontend: la "subida" ya viene
# expresada en el mismo sistema de referencia que usa HAND.
SIMULADOR_HAND_UMBRAL_DEFECTO_M = 5.0  # subida moderada cuando no hay dato de estación/GloFAS


def _preparar_hand_y_agua(lat, lon, radius_km, region=None):
    """HAND (MERIT Hydro, corregido por dosel/edificios) + agua permanente (JRC), recortados a la región."""
    if region is None:
        region = ee.Geometry.Point([lon, lat]).buffer(radius_km * 1000)
    hand_crudo = ee.Image('MERIT/Hydro/v1_0_1').select('hnd').clip(region)
    hand = _hand_corregido_por_dosel(hand_crudo, region)
    agua_fuente = (ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
                     .select('occurrence')
                     .gt(50)
                     .clip(region))
    return region, hand, agua_fuente


def _conectividad_hidraulica(elevacion, agua_fuente, region, radius_km, umbral_elev):
    """
    Dado un umbral de elevación, devuelve:
      candidatas       -> todas las celdas <= umbral (bathtub-fill puro)
      zona_conectada   -> subconjunto alcanzable desde agua real sin cruzar
                           terreno por encima del umbral
      costo_acumulado  -> distancia acumulada (m) desde el agua real hasta
                           cada celda, atravesando solo celdas candidatas.
                           Es un proxy de "orden de llegada": las celdas más
                           cerca del río (costo bajo) se inundarían primero.
                           Se usa para animar la expansión (ver
                           /inundacion_animacion).
      conectividad_ok  -> True si se pudo propagar (había agua_fuente visible
                           en la región consultada)
    RENDIMIENTO: se reproyecta a 90 m y se topea la propagación a 50 km
    (cumulativeCost es caro — ver nota extendida en el commit del punto 1).
    """
    candidatas = elevacion.lte(umbral_elev)
    costo = candidatas.selfMask()

    escala_conectividad = 90
    max_dist_conectividad = min(radius_km * 1000, 50_000)

    costo_90  = costo.reproject(crs='EPSG:4326', scale=escala_conectividad)
    fuente_90 = agua_fuente.reproject(crs='EPSG:4326', scale=escala_conectividad)

    costo_acumulado = costo_90.cumulativeCost(
        source=fuente_90,
        maxDistance=max_dist_conectividad,
    )

    zona_conectada = candidatas.updateMask(costo_acumulado.gte(0)).selfMask()

    hay_agua_fuente = agua_fuente.reduceRegion(
        reducer=ee.Reducer.anyNonZero(),
        geometry=region,
        scale=90,
        bestEffort=True,
        maxPixels=1e9,
        tileScale=4,
    ).get('occurrence')
    conectividad_ok = ee.Algorithms.If(hay_agua_fuente, True, False).getInfo()

    return candidatas, zona_conectada, costo_acumulado, conectividad_ok


def _simular_inundacion(hand, agua_fuente, region, radius_km, umbral_m, perfil=None):
    """
    Punto único desde donde los endpoints de inundación piden la
    simulación. Usa FloodSimulationEngine (multicriterio: pendiente, sesgo
    de dirección de flujo, acumulación de flujo, resistencia, perfil
    regional inferido) cuando está disponible; si el módulo no cargó por
    algún motivo, cae automáticamente a _conectividad_hidraulica() (el
    comportamiento original, sin las mejoras) para que la app nunca se
    caiga por esto.

    Devuelve siempre la misma forma:
        candidatas, zona_conectada, costo_acumulado, conectividad_ok, info_motor

    info_motor es None si se usó el fallback legacy, o el diccionario con
    'perfil' y 'umbral_efectivo_m' si se usó el motor nuevo — los
    endpoints lo agregan a la respuesta JSON de forma opcional, sin romper
    a ningún consumidor del frontend que no lo espere.
    """
    if _MOTOR_DISPONIBLE:
        try:
            resultado = _motor_inundacion.simulate_flood(
                region, hand, agua_fuente, radius_km, umbral_m, perfil=perfil,
            )
            return (
                resultado['candidatas'],
                resultado['zona_conectada'],
                resultado['costo_acumulado'],
                resultado['conectividad_ok'],
                {
                    'perfil': resultado['perfil'],
                    'umbral_efectivo_m': resultado['umbral_efectivo_m'],
                },
            )
        except ee.EEException:
            # Si el motor nuevo falla en tiempo de ejecución (ej. algún
            # dataset no disponible para esa región), no tumbamos el
            # endpoint -- caemos al comportamiento original.
            pass

    candidatas, zona_conectada, costo_acumulado, conectividad_ok = (
        _conectividad_hidraulica(hand, agua_fuente, region, radius_km, umbral_m)
    )
    return candidatas, zona_conectada, costo_acumulado, conectividad_ok, None


# ──────────────────────────────────────────────────────────────────────────
# JURISDICCIÓN ADMINISTRATIVA (para "Direcciones de Foco")
# ──────────────────────────────────────────────────────────────────────────
# Objetivo: cuando la herramienta "Direcciones de Foco" analiza un área, los
# focos de calor NASA FIRMS usados para calcular la dirección de propagación
# deben pertenecer SOLO a la misma jurisdicción administrativa (provincia,
# estado, departamento, región, etc.) donde está esa área. Antes, el frontend
# traía todos los focos dentro de un bbox de ~2.5° alrededor del punto, sin
# importar límites administrativos, por lo que cerca de una frontera aparecían
# focos de la provincia/estado/país vecino.
#
# Estrategia: identificar automáticamente el límite administrativo de nivel 1
# (primer nivel debajo del país) que contiene el punto analizado, usando el
# dataset global FAO GAUL 2015 nivel 1 (cobertura mundial, sin reglas
# hardcodeadas por país) y devolver su geometría al frontend, que la usa para
# filtrar los focos por intersección espacial real (punto-en-polígono), no
# por distancia ni por bounding box.

# Nombre "amigable" del primer nivel administrativo según el país, SOLO para
# mostrarlo en pantalla (ej. "Provincia" en vez de "ADM1"). No afecta en nada
# al filtro geográfico en sí, que es genérico para cualquier país vía GAUL.
_NIVEL1_LABELS_POR_PAIS = {
    'Argentina': 'Provincia',
    'Brazil': 'Estado', 'Brasil': 'Estado',
    'United States of America': 'Estado',
    'Mexico': 'Estado', 'México': 'Estado',
    'Paraguay': 'Departamento',
    'Uruguay': 'Departamento',
    'Chile': 'Región',
    'Spain': 'Comunidad Autónoma', 'España': 'Comunidad Autónoma',
    'Germany': 'Bundesland', 'Alemania': 'Bundesland',
    'France': 'Región', 'Francia': 'Región',
    'Canada': 'Provincia',
    'Australia': 'Estado',
    'Bolivia': 'Departamento',
    'Peru': 'Región', 'Perú': 'Región',
    'Colombia': 'Departamento',
}

_gaul_level1_fc = None       # cache del FeatureCollection (se pide una sola vez por proceso)
_jurisdiccion_cache = {}     # cache en memoria: (lat_redondeado, lon_redondeado) -> resultado
_JURISDICCION_CACHE_MAX = 500


def _gaul_level1():
    """FeatureCollection global de límites administrativos de nivel 1 (FAO GAUL 2015).
    Se pide una sola vez y se reutiliza en llamadas subsiguientes."""
    global _gaul_level1_fc
    if _gaul_level1_fc is None:
        _gaul_level1_fc = ee.FeatureCollection('FAO/GAUL/2015/level1')
    return _gaul_level1_fc


def getCountry(feature_info):
    """Extrae el nombre del país (ADM0_NAME) de las propiedades de un feature GAUL."""
    return (feature_info.get('properties') or {}).get('ADM0_NAME', 'Desconocido')


def getJurisdiction(feature_info):
    """Extrae el nombre de la jurisdicción de nivel 1 (ADM1_NAME, ej. provincia/
    estado/departamento) de las propiedades de un feature GAUL."""
    return (feature_info.get('properties') or {}).get('ADM1_NAME', 'Desconocida')


def getAdministrativeBoundary(lat, lon):
    """
    Función principal de detección de jurisdicción.

    Dado un punto (lat, lon), detecta automáticamente el país y la jurisdicción
    administrativa de primer nivel (provincia/estado/departamento/región/etc.)
    que lo contiene, usando el dataset mundial FAO GAUL 2015 nivel 1 (una sola
    consulta espacial a Earth Engine, sin reglas particulares por país).

    Devuelve un dict:
      - ok=True  -> {ok, pais, jurisdiccion, nivel, geometry}  (geometry = GeoJSON
        simplificado del polígono de la jurisdicción, para que el frontend haga
        el filtrado punto-en-polígono de los focos)
      - ok=False -> {ok, motivo}  (ej. punto en el mar o fuera de cobertura GAUL)

    Usa una caché en memoria (redondeando lat/lon a 2 decimales, ~1 km) para no
    volver a golpear Earth Engine si se consulta repetidamente la misma zona
    (por ejemplo, al refrescar el panel de focos varias veces sobre el mismo campo).
    """
    clave = (round(lat, 2), round(lon, 2))
    if clave in _jurisdiccion_cache:
        return _jurisdiccion_cache[clave]

    punto = ee.Geometry.Point([lon, lat])
    feature = _gaul_level1().filterBounds(punto).first()
    info = feature.getInfo()

    if info is None:
        resultado = {
            'ok': False,
            'motivo': 'No se encontró una jurisdicción administrativa para ese punto '
                      '(puede estar en el mar o fuera de la cobertura del dataset GAUL).',
        }
    else:
        pais = getCountry(info)
        jurisdiccion = getJurisdiction(info)
        # Simplificar la geometría antes de traerla al cliente: algunas provincias/
        # estados tienen miles de vértices y no hace falta esa precisión para un
        # filtro punto-en-polígono a escala de focos de incendio.
        # Tolerancia baja (100 m): prioriza que el límite quede lo más fiel posible
        # al real, para no descartar por error focos que están justo sobre el borde
        # de la jurisdicción. Si en el futuro el tamaño del GeoJSON es un problema
        # de rendimiento, subir este valor (a costa de precisión en el borde).
        geometria = feature.geometry().simplify(100).getInfo()  # tolerancia ~100 m
        resultado = {
            'ok': True,
            'pais': pais,
            'jurisdiccion': jurisdiccion,
            'nivel': _NIVEL1_LABELS_POR_PAIS.get(pais, 'División administrativa'),
            'geometry': geometria,
        }

    if len(_jurisdiccion_cache) >= _JURISDICCION_CACHE_MAX:
        _jurisdiccion_cache.clear()
    _jurisdiccion_cache[clave] = resultado
    return resultado


# ──────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────────────────────────────────

@app.route('/health')
def health():
    return jsonify({
        'ok': True,
        'earth_engine': _ee_ready,
        'error': _ee_error,
        'hidrografia_vectorial_disponible': _hidro_vectorial_disponible,
    })


@app.route('/jurisdiccion_area')
def jurisdiccion_area():
    """
    Identifica automáticamente la jurisdicción administrativa (país + nivel 1:
    provincia/estado/departamento/región/comunidad autónoma/etc.) que contiene
    el punto (lat, lon), usando el dataset mundial FAO GAUL 2015 nivel 1.

    Usado por la herramienta "Direcciones de Foco" del frontend para filtrar
    los focos NASA FIRMS y quedarse solo con los que pertenecen a la misma
    jurisdicción que el área analizada, descartando focos de una provincia,
    estado o país vecino aunque estén geográficamente cerca.

    Parámetros: lat, lon
    Devuelve: { ok, pais, jurisdiccion, nivel, geometry } o { ok:false, motivo }
    """
    if not _ee_ready:
        return jsonify({'error': f'Earth Engine no inicializado: {_ee_error}'}), 503
    try:
        lat = float(request.args.get('lat'))
        lon = float(request.args.get('lon'))
        resultado = getAdministrativeBoundary(lat, lon)
        return jsonify(resultado)
    except (TypeError, ValueError):
        return jsonify({'error': 'Parámetros lat/lon inválidos o faltantes.'}), 400
    except Exception as exc:  # noqa: BLE001
        return jsonify({'error': str(exc)}), 500


@app.route('/vegetacion')
def vegetacion():
    """
    Promedio de NDVI en una zona (Sentinel-2) para los últimos `dias` días.
    Parámetros: lat, lon, dias (default 30), radius_km (default 15)
    Devuelve: { ndvi_promedio }
    """
    if not _ee_ready:
        return jsonify({'error': f'Earth Engine no inicializado: {_ee_error}'}), 503
    try:
        lat = float(request.args.get('lat'))
        lon = float(request.args.get('lon'))
        dias = int(request.args.get('dias', 30))
        radius_km = float(request.args.get('radius_km', 15))

        region = ee.Geometry.Point([lon, lat]).buffer(radius_km * 1000)
        hoy = ee.Date(datetime.date.today().isoformat())
        desde = hoy.advance(-dias, 'day')

        col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
               .filterDate(desde, hoy)
               .filterBounds(region)
               .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 40)))

        if col.size().getInfo() == 0:
            return jsonify({'error': 'Sin imágenes disponibles para esa zona/período'}), 404

        img = col.median()
        ndvi = img.normalizedDifference(['B8', 'B4']).rename('ndvi')
        refl = img.select(['B2', 'B4', 'B8']).multiply(0.0001)  # DN -> reflectancia 0-1
        evi = refl.expression(
            '2.5 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)',
            {'NIR': refl.select('B8'), 'RED': refl.select('B4'), 'BLUE': refl.select('B2')}
        ).rename('evi')

        stats = ee.Image.cat([ndvi, evi]).reduceRegion(
            reducer=ee.Reducer.mean(), geometry=region, scale=100, bestEffort=True
        )
        ndvi_val = stats.get('ndvi').getInfo()
        evi_val = stats.get('evi').getInfo()

        return jsonify({
            'ndvi_promedio': round(ndvi_val, 3) if ndvi_val is not None else None,
            'evi_promedio': round(evi_val, 3) if evi_val is not None else None,
        })
    except ee.EEException as exc:
        return jsonify({'error': f'Error de Earth Engine: {exc}'}), 502
    except Exception as exc:  # noqa: BLE001
        return jsonify({'error': str(exc)}), 500


@app.route('/lluvia')
def lluvia():
    """
    Precipitación acumulada en una zona (CHIRPS) para los últimos `dias` días.
    Parámetros: lat, lon, dias (default 30), radius_km (default 15)
    Devuelve: { lluvia_acumulada_mm }
    """
    if not _ee_ready:
        return jsonify({'error': f'Earth Engine no inicializado: {_ee_error}'}), 503
    try:
        lat = float(request.args.get('lat'))
        lon = float(request.args.get('lon'))
        dias = int(request.args.get('dias', 30))
        radius_km = float(request.args.get('radius_km', 15))

        region = ee.Geometry.Point([lon, lat]).buffer(radius_km * 1000)
        hoy = ee.Date(datetime.date.today().isoformat())
        desde = hoy.advance(-dias, 'day')

        chirps = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
                  .filterDate(desde, hoy)
                  .filterBounds(region)
                  .sum())

        valor = chirps.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=region, scale=5000, bestEffort=True
        ).get('precipitation').getInfo()

        return jsonify({'lluvia_acumulada_mm': round(valor, 1) if valor is not None else None})
    except ee.EEException as exc:
        return jsonify({'error': f'Error de Earth Engine: {exc}'}), 502
    except Exception as exc:  # noqa: BLE001
        return jsonify({'error': str(exc)}), 500


@app.route('/gee_tiles')
def gee_tiles():
    """
    Parámetros:
        dataset     MODIS | VIIRS | Sentinel | Landsat
        band        NDVI | EVI | NBR | FalseColor | Thermal
        start       YYYY-MM-DD
        end         YYYY-MM-DD
        cloud       0-100 (cobertura de nubes máxima, Sentinel/Landsat)
        lat, lon    centro del área de interés
        radius_km   radio del área (default 150 km)
    """
    if not _ee_ready:
        return jsonify({'error': f'Earth Engine no inicializado: {_ee_error}'}), 503
    try:
        dataset = request.args.get('dataset', 'Sentinel')
        band = request.args.get('band', 'NDVI')
        start = request.args.get('start')
        end = request.args.get('end')
        cloud = float(request.args.get('cloud', 30))
        lat = float(request.args.get('lat'))
        lon = float(request.args.get('lon'))
        # FIX: se amplía el radio por defecto de 75 a 150 km para que el
        # área con datos siempre exceda lo que se ve en pantalla a niveles
        # de zoom habituales, y así la capa no se "corte" a mitad de mapa.
        radius_km = float(request.args.get('radius_km', 150))

        if not start or not end:
            return jsonify({'error': 'Faltan parámetros start y end (YYYY-MM-DD)'}), 400

        # FIX: ventana mínima de búsqueda. Sentinel-2/Landsat cubren la
        # superficie en "cuadrantes" (granules) que se fotografían en días
        # distintos. Con rangos cortos (ej. preset "7 días") es común que
        # un cuadrante vecino no tenga NINGUNA pasada en ese lapso, dejando
        # un borde recto sin datos en el mapa. Para evitarlo, si el rango
        # pedido es menor a MIN_DIAS_COBERTURA, lo extendemos hacia atrás
        # (manteniendo la fecha "end" tal cual la pidió el usuario) — así
        # siempre hay suficientes pasadas para cubrir toda la zona, y el
        # mosaico ordenado por nubosidad sigue priorizando lo más reciente
        # y menos nublado en cada pixel.
        MIN_DIAS_COBERTURA = 90
        fecha_fin_dt = datetime.date.fromisoformat(end)
        fecha_inicio_dt = datetime.date.fromisoformat(start)
        if (fecha_fin_dt - fecha_inicio_dt).days < MIN_DIAS_COBERTURA:
            fecha_inicio_dt = fecha_fin_dt - datetime.timedelta(days=MIN_DIAS_COBERTURA)
            start = fecha_inicio_dt.isoformat()

        region = ee.Geometry.Point([lon, lat]).buffer(radius_km * 1000)

        if dataset == 'Sentinel':
            # BBOX amplio en vez de círculo: el círculo de filterBounds limita
            # los granules que entran al mosaico. Si hay pocos días de datos,
            # solo entra 1-2 franjas orbitales y quedan bordes rectos vacíos.
            # Usando un rectángulo de ~6° × 6° (~660 km) cubrimos varias
            # franjas orbitales adyacentes → mosaico continuo sin cortes.
            bbox = ee.Geometry.Rectangle([
                lon - 3, lat - 3,
                lon + 3, lat + 3
            ])
            col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                   .filterDate(start, end)
                   .filterBounds(bbox)
                   .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', cloud)))
            if col.filterBounds(region).size().getInfo() == 0:
                return jsonify({'error': 'Sin imágenes Sentinel-2 para ese rango/zona/nubosidad.'}), 404
            img, vis = _band_sentinel(col, band)

        elif dataset == 'Landsat':
            # Mismo criterio que Sentinel: bbox amplio para cubrir varias
            # franjas orbitales y evitar bordes rectos sin datos.
            bbox = ee.Geometry.Rectangle([
                lon - 3, lat - 3,
                lon + 3, lat + 3
            ])
            col = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                   .filterDate(start, end)
                   .filterBounds(bbox)
                   .filter(ee.Filter.lte('CLOUD_COVER', cloud)))
            if col.filterBounds(region).size().getInfo() == 0:
                return jsonify({'error': 'Sin imágenes Landsat 8 para ese rango/zona/nubosidad.'}), 404
            img, vis = _band_landsat(col, band)

        elif dataset == 'MODIS':
            col_ndvi = ee.ImageCollection('MODIS/061/MOD13Q1').filterDate(start, end).filterBounds(region)
            col_refl = ee.ImageCollection('MODIS/061/MOD09GA').filterDate(start, end).filterBounds(region)
            col_lst = ee.ImageCollection('MODIS/061/MOD11A1').filterDate(start, end).filterBounds(region)
            img, vis = _band_modis(col_ndvi, col_refl, col_lst, band)

        elif dataset == 'VIIRS':
            col_ndvi = ee.ImageCollection('NOAA/VIIRS/001/VNP13A1').filterDate(start, end).filterBounds(region)
            col_refl = ee.ImageCollection('NOAA/VIIRS/001/VNP09GA').filterDate(start, end).filterBounds(region)
            col_lst = ee.ImageCollection('NOAA/VIIRS/001/VNP21A1D').filterDate(start, end).filterBounds(region)
            img, vis = _band_viirs(col_ndvi, col_refl, col_lst, band)

        else:
            return jsonify({'error': f'Dataset no soportado: {dataset}'}), 400

        # BUG PRINCIPAL CORREGIDO: se elimina .clip(region).
        # El clip recortaba la imagen a un círculo → tiles fuera del círculo
        # quedaban transparentes → parches en el mapa.
        # filterBounds() ya garantiza imágenes de la zona; el mosaico
        # cubre las escenas completas, llenando toda la pantalla sin huecos.
        map_id = img.getMapId(vis)

        return jsonify({
            'tile_url': map_id['tile_fetcher'].url_format,
            'dataset': dataset,
            'band': band,
            'vis': vis,
        })
    except ee.EEException as exc:
        return jsonify({'error': f'Error de Earth Engine: {exc}'}), 502
    except Exception as exc:  # noqa: BLE001
        return jsonify({'error': str(exc)}), 500


@app.route('/riesgo_incendio_tiles')
def riesgo_incendio_tiles():
    """
    Índice combinado de riesgo de incendio (0-1):
      - Vegetación seca: NDVI bajo (Sentinel-2, últimos 30 días) -> peso 0.6
      - Déficit de lluvia: precipitación acumulada baja (CHIRPS,
        últimos 30 días, normalizada contra el promedio histórico
        del mismo período de los últimos 5 años) -> peso 0.4

    Parámetros: lat, lon, radius_km (default 150)
    """
    if not _ee_ready:
        return jsonify({'error': f'Earth Engine no inicializado: {_ee_error}'}), 503
    try:
        lat = float(request.args.get('lat'))
        lon = float(request.args.get('lon'))
        # FIX: mismo ajuste de radio que en /gee_tiles (75 -> 150 km).
        radius_km = float(request.args.get('radius_km', 150))

        region = ee.Geometry.Point([lon, lat]).buffer(radius_km * 1000)
        hoy = ee.Date(datetime.date.today().isoformat())  # fecha actual del servidor de EE
        hace_30 = hoy.advance(-30, 'day')

        # ── Vegetación seca (NDVI invertido) ──────────────────────────
        s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filterDate(hace_30, hoy)
              .filterBounds(region)
              .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 40)))
        # FIX: mosaico ordenado por nubosidad en vez de .median(), mismo
        # motivo que en /gee_tiles — evita huecos sin datos en el mapa.
        # Aplicar máscara SCL antes de compositar (mismo criterio que /gee_tiles)
        ndvi = s2.map(_mask_s2_clouds).sort('CLOUDY_PIXEL_PERCENTAGE').mosaic().normalizedDifference(['B8', 'B4'])

        # NDVI ~ -0.1 (suelo desnudo / muy seco) a ~0.9 (vegetación sana)
        # Invertimos y normalizamos a 0-1: 1 = muy seco, 0 = muy verde
        sequedad = ndvi.multiply(-1).add(0.9).divide(1.0).clamp(0, 1)

        # ── Déficit de lluvia (CHIRPS) ────────────────────────────────
        chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
        lluvia_actual = chirps.filterDate(hace_30, hoy).filterBounds(region).sum()

        # Promedio histórico del mismo período (últimos 5 años)
        anios_hist = ee.List.sequence(1, 5)

        def lluvia_anio(n):
            ini = hace_30.advance(ee.Number(n).multiply(-1), 'year')
            fin = hoy.advance(ee.Number(n).multiply(-1), 'year')
            return chirps.filterDate(ini, fin).filterBounds(region).sum()

        lluvia_hist = ee.ImageCollection(anios_hist.map(lluvia_anio)).mean()

        # Déficit normalizado: 1 = sin lluvia respecto al promedio, 0 = lluvia normal o mayor
        deficit = (lluvia_hist.subtract(lluvia_actual)
                   .divide(lluvia_hist.add(1))  # +1 evita división por cero
                   .clamp(0, 1))

        # ── Índice combinado ──────────────────────────────────────────
        # Sin .clip(region): mismo motivo que en /gee_tiles (evita parches circulares)
        riesgo = sequedad.multiply(0.6).add(deficit.multiply(0.4)).rename('riesgo')
        vis = {'min': 0, 'max': 1, 'palette': PALETA_RIESGO}
        map_id = riesgo.getMapId(vis)

        return jsonify({
            'tile_url': map_id['tile_fetcher'].url_format,
            'vis': vis,
            'descripcion': 'Riesgo combinado: sequedad de vegetación (NDVI, 60%) + déficit de lluvia vs. promedio histórico (CHIRPS, 40%)',
        })
    except ee.EEException as exc:
        return jsonify({'error': f'Error de Earth Engine: {exc}'}), 502
    except Exception as exc:  # noqa: BLE001
        return jsonify({'error': str(exc)}), 500


@app.route('/inundacion_tiles')
@app.route('/zonas_inundables')   # alias usado por TerraOS_v10 HTML
def inundacion_tiles():
    """
    Teselas del índice de riesgo de inundación basado en DEM SRTM + pendiente + humedad SMAP.
    Replica exacta del script GEE "Elevaciones - riesgo de inundación".

    Parámetros:
        tipo       riesgo (default) | critico | elevacion | pendiente |
                   zona_baja | zona_media | zona_alta
        lat, lon   centro del área (default: Corrientes, Argentina)
        radius_km  radio del área (default: 200 km, TOPE: 800 km — ver
                   RADIUS_KM_MAX_INUNDACION; valores mayores se recortan
                   automáticamente. Para tipo='zona_baja'/'anegamiento',
                   un radio grande puede provocar 400 de
                   Earth Engine al cargar los tiles — ver nota junto a la
                   constante)
        umbral_m   SOLO para tipo=zona_baja/anegamiento.
                   Metros de SUBIDA por encima del cauce normal (no una cota
                   absoluta — ver nota HAND más abajo). Si no se manda, usa
                   SIMULADOR_HAND_UMBRAL_DEFECTO_M (5 m).

    Nota sobre 'zona_baja'/'anegamiento' (usados por el
    Simulador de Inundación del frontend):
    Usan HAND (Height Above Nearest Drainage, MERIT Hydro) en vez de
    elevación absoluta como criterio de "candidata a inundarse" — ver nota
    extensa junto a SIMULADOR_HAND_UMBRAL_DEFECTO_M. Además, se aplica
    corrección de conectividad hidráulica: solo se pinta una celda si,
    además de tener HAND por debajo del umbral, está conectada "caminando"
    por celdas también bajas hasta un cuerpo de agua real (JRC Global
    Surface Water). Esto evita marcar como inundadas depresiones aisladas
    que en la realidad el agua nunca alcanza.
    """
    if not _ee_ready:
        return jsonify({'error': f'Earth Engine no inicializado: {_ee_error}'}), 503
    try:
        tipo      = request.args.get('tipo', 'riesgo')
        lat       = float(request.args.get('lat',  -27.48))   # centro Corrientes
        lon       = float(request.args.get('lon',  -58.83))
        radius_km = min(float(request.args.get('radius_km', 200)), RADIUS_KM_MAX_INUNDACION)

        umbral_m = None
        if 'umbral_m' in request.args:
            try:
                umbral_m = float(request.args['umbral_m'])
                if umbral_m <= 0:
                    umbral_m = None
            except ValueError:
                umbral_m = None

        region, elevacion, agua_fuente = _preparar_dem_y_agua(lat, lon, radius_km)
        pendiente = ee.Terrain.slope(elevacion)

        # ── Humedad de suelo SMAP (período El Niño 2023-2024) ─────────
        smap = (ee.ImageCollection('NASA_USDA/HSL/SMAP10KM_soil_moisture')
                  .filterBounds(region)
                  .filterDate('2023-10-01', '2024-03-31')
                  .select('ssm')
                  .mean()
                  .clip(region))

        # ── Normalización (igual que el script GEE) ───────────────────
        def normalizar(img, min_val, max_val):
            img_safe = ee.Image(ee.Algorithms.If(
                img.bandNames().size().gt(0), img, ee.Image.constant(0)
            ))
            return img_safe.subtract(min_val).divide(max_val - min_val).clamp(0, 1)

        n_elev = normalizar(elevacion, 30, 140).subtract(1).multiply(-1)
        n_pend = normalizar(pendiente, 0,  3 ).subtract(1).multiply(-1)
        n_smap = normalizar(smap,      10, 40)

        riesgo_compuesto = (n_elev.multiply(0.40)
                             .add(n_pend.multiply(0.30))
                             .add(n_smap.multiply(0.30))
                             .rename('riesgo'))

        # ── Selección de capa y visualización ────────────────────────
        if tipo == 'riesgo':
            imagen = riesgo_compuesto
            vis    = {'min': 0.2, 'max': 0.8,
                      'palette': ['#2c7bb6', '#abd9e9', '#ffffbf', '#fdae61', '#d7191c']}

        elif tipo == 'critico':
            imagen = riesgo_compuesto.gt(0.65).selfMask()
            vis    = {'palette': ['#ff0000']}

        elif tipo == 'elevacion':
            imagen = elevacion
            vis    = {'min': 30, 'max': 140,
                      'palette': ['#1a3a5c', '#2e6da4', '#67a9cf', '#d1e5f0',
                                  '#f7f7f7', '#fddbc7', '#d6604d']}

        elif tipo == 'pendiente':
            imagen = pendiente
            vis    = {'min': 0, 'max': 3,
                      'palette': ['#ffffff', '#fee090', '#fc8d59', '#d73027']}

        elif tipo == 'zona_baja':
            hand_crudo = ee.Image('MERIT/Hydro/v1_0_1').select('hnd').clip(region)
            hand = _hand_corregido_por_dosel(hand_crudo, region)
            if umbral_m is not None:
                umbral = umbral_m
                umbral_metodo = 'calibrado_usuario'
            else:
                umbral = SIMULADOR_HAND_UMBRAL_DEFECTO_M
                umbral_metodo = 'default_hand'

            candidatas, zona_conectada, _costo, conectividad_aplicada, info_motor = (
                _simular_inundacion(hand, agua_fuente, region, radius_km, umbral)
            )

            # Fallback: si no hay agua conocida en el radio, mejor mostrar de
            # más (candidatas sin filtrar) que no mostrar nada.
            imagen = zona_conectada if conectividad_aplicada else candidatas.selfMask()
            vis    = {'palette': ['#bcd2ee']}

        elif tipo == 'anegamiento':
            # ── Anegamiento por lluvia local (SCS Curve Number) ────────────
            # OJO — esto es un fenómeno distinto de 'zona_baja':
            #   zona_baja    = desborde del RÍO (depende de la cuenca entera,
            #                  no de la lluvia local — no se puede simular
            #                  con una fórmula simple).
            #   anegamiento  = agua de lluvia LOCAL que no se infiltra y
            #                  queda empozada porque no tiene por dónde
            #                  escurrir hacia el río.
            #
            # Reutilizamos la MISMA conectividad del punto anterior: las
            # celdas "candidatas" (bajas) que NO están conectadas al río real
            # son exactamente las depresiones topográficas aisladas — el
            # lugar físico donde el agua de lluvia se estanca en vez de
            # drenar. Se usa el mismo umbral que 'zona_baja' (default 55 m,
            # o el que mande el simulador) para identificarlas.
            umbral = umbral_m if umbral_m is not None else SIMULADOR_HAND_UMBRAL_DEFECTO_M
            precipitacion_mm = float(request.args.get('precipitacion_mm', 0))
            tipo_suelo = request.args.get('tipo_suelo', 'franco')

            hand_crudo = ee.Image('MERIT/Hydro/v1_0_1').select('hnd').clip(region)
            hand = _hand_corregido_por_dosel(hand_crudo, region)
            candidatas, zona_conectada, _costo, conectividad_aplicada, info_motor = (
                _simular_inundacion(hand, agua_fuente, region, radius_km, umbral)
            )
            conectada_bool = zona_conectada.mask().unmask(0)
            zona_aislada = candidatas.And(conectada_bool.Not()).selfMask()

            # ── Curve Number (SCS/NRCS) por tipo de suelo ──────────────────
            # Valores típicos para cobertura mixta rural (pastizal/cultivo en
            # condición media), agrupados por grupo hidrológico de suelo:
            #   A (arena, alta infiltración)      CN ≈ 45
            #   B (franco)                         CN ≈ 65
            #   C/D (arcilloso, baja infiltración) CN ≈ 80
            #   orgánico (alta retención inicial, satura rápido) CN ≈ 70
            #   humedal (prácticamente saturado)   CN ≈ 92
            # Es una tabla estándar de ingeniería hidrológica (no inventada),
            # simplificada acá a un único valor por tipo — un estudio real
            # usaría el mapa de grupos hidrológicos + uso de suelo real.
            CN_POR_SUELO = {
                'arena'    : 45,
                'franco'   : 65,
                'arcilloso': 80,
                'organico' : 70,
                'humedal'  : 92,
            }
            cn_custom = request.args.get('cn')
            if cn_custom is not None:
                try:
                    cn = max(30, min(98, float(cn_custom)))
                except ValueError:
                    cn = CN_POR_SUELO.get(tipo_suelo, 65)
            else:
                cn = CN_POR_SUELO.get(tipo_suelo, 65)

            # Ecuación SCS-CN estándar (P y S en mm):
            #   S  = retención potencial máxima
            #   Ia = abstracción inicial (intercepción + almacenamiento superficial)
            #   Q  = escorrentía efectiva (acá: agua que no se infiltra y,
            #        al no tener salida en una depresión aislada, se acumula)
            s_ret = (25400.0 / cn) - 254.0
            ia = 0.2 * s_ret
            if precipitacion_mm > ia:
                q_mm = ((precipitacion_mm - ia) ** 2) / (precipitacion_mm - ia + s_ret)
            else:
                q_mm = 0.0

            # Severidad visual en función de la escorrentía estimada.
            if q_mm < 5:
                vis = {'palette': ['#a5b4fc']}      # apenas visible, riesgo mínimo
            elif q_mm < 15:
                vis = {'palette': ['#818cf8']}      # leve
            elif q_mm < 40:
                vis = {'palette': ['#4f46e5']}      # moderado
            else:
                vis = {'palette': ['#3730a3']}      # severo

            imagen = zona_aislada

        elif tipo == 'zona_media':
            imagen = elevacion.gte(55).And(elevacion.lt(90)).selfMask()
            vis    = {'palette': ['#e3ffc2']}

        elif tipo == 'zona_alta':
            imagen = elevacion.gte(90).selfMask()
            vis    = {'palette': ['#e6c280']}

        else:
            return jsonify({'error': f"tipo '{tipo}' no reconocido"}), 400

        map_id = imagen.getMapId(vis)
        respuesta = {
            'tile_url': map_id['tile_fetcher'].url_format,
            'tipo'    : tipo,
            'vis'     : vis,
        }
        if tipo == 'zona_baja':
            respuesta['umbral_m'] = umbral
            respuesta['umbral_metodo'] = umbral_metodo
            respuesta['umbral_tipo'] = 'subida_hand_m'
            respuesta['conectividad_hidraulica'] = conectividad_aplicada
            if not conectividad_aplicada:
                respuesta['aviso'] = (
                    'No se detectó agua superficial conocida (JRC) en el radio '
                    'consultado; se muestran todas las zonas bajas sin filtrar '
                    'por conectividad.'
                )
            if info_motor is not None:
                respuesta['simulacion_multicriterio'] = True
                respuesta['umbral_efectivo_m'] = info_motor['umbral_efectivo_m']
                respuesta['perfil_regional'] = info_motor['perfil']
        if tipo == 'anegamiento':
            respuesta['umbral_m'] = umbral
            respuesta['umbral_tipo'] = 'subida_hand_m'
            respuesta['precipitacion_mm'] = precipitacion_mm
            respuesta['tipo_suelo'] = tipo_suelo
            respuesta['curve_number'] = cn
            respuesta['retencion_potencial_mm'] = round(s_ret, 1)
            respuesta['abstraccion_inicial_mm'] = round(ia, 1)
            respuesta['escorrentia_mm'] = round(q_mm, 1)
            respuesta['conectividad_hidraulica'] = conectividad_aplicada
            if info_motor is not None:
                respuesta['simulacion_multicriterio'] = True
                respuesta['umbral_efectivo_m'] = info_motor['umbral_efectivo_m']
                respuesta['perfil_regional'] = info_motor['perfil']
        return jsonify(respuesta)

    except ee.EEException as exc:
        return jsonify({'error': f'Error de Earth Engine: {exc}'}), 502
    except Exception as exc:  # noqa: BLE001
        return jsonify({'error': str(exc)}), 500


# ════════════════════════════════════════════════
#  CUERPOS DE AGUA — JRC Global Surface Water v1.4
#  Endpoint: /aguas_superficiales_tiles
#
#  A diferencia de /gee_tiles o /inundacion_tiles, esta capa es GLOBAL:
#  no recibe lat/lon ni radio, porque el dataset cubre todo el planeta.
#  El frontend pide esta URL una sola vez y la reusa sin importar a
#  dónde se mueva el mapa (mismo patrón de getMapId(), solo que sin
#  región/clip porque no hace falta).
# ════════════════════════════════════════════════

@app.route('/aguas_superficiales_tiles')
def aguas_superficiales_tiles():
    """
    Devuelve la URL de tiles de JRC Global Surface Water (banda 'occurrence').
    Sin parámetros: es un mosaico global, mismo request sirve para todo el mapa.
    """
    if not _ee_ready:
        return jsonify({'error': f'Earth Engine no inicializado: {_ee_error}'}), 503
    try:
        img = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence')
        vis = {'min': 0, 'max': 100, 'palette': ['ffffff', 'ffbbbb', '0000ff']}
        map_id = img.getMapId(vis)

        return jsonify({
            'tile_url': map_id['tile_fetcher'].url_format,
            'vis': vis,
        })
    except ee.EEException as exc:
        return jsonify({'error': f'Error de Earth Engine: {exc}'}), 502
    except Exception as exc:  # noqa: BLE001
        return jsonify({'error': str(exc)}), 500


# ════════════════════════════════════════════════
#  CUERPO DE AGUA MÁS CERCANO — endpoint liviano
#  Endpoint: /cuerpo_agua_cercano
#
#  A diferencia de /iipdi_punto (que hace todo el cálculo del índice de
#  inundación, con imágenes de GEE incluidas), este endpoint SOLO busca el
#  cuerpo de agua vectorial más cercano. Lo usa el panel de "Incendio
#  activo" para la feature de cercanía a agua (recarga de autobombas +
#  barrera natural), y no necesita nada de Earth Engine, así que responde
#  mucho más rápido.
# ════════════════════════════════════════════════

@app.route('/cuerpo_agua_cercano')
def cuerpo_agua_cercano():
    """
    Devuelve el cuerpo de agua vectorial más cercano a (lat, lon), con la
    distancia y las coordenadas del punto más próximo sobre su geometría
    (no un centroide). El resto —clasificación de distancia, alineación
    con la dirección del viento/avance del fuego y el veredicto combinado—
    se calcula en el frontend, que ya tiene esa lógica (cono de propagación).

    Parámetros: lat, lon, radius_km (default 15, tope 30)
    """
    if not _hidro_vectorial_disponible:
        return jsonify({
            'error': (
                'Módulo hidrografia_vectorial no disponible en el servidor. '
                'Subí hidrografia_vectorial.py junto a app.py y volvé a desplegar.'
            ),
        }), 503
    try:
        lat = float(request.args.get('lat'))
        lon = float(request.args.get('lon'))
        radius_km = min(float(request.args.get('radius_km', 15)), 30)

        cuerpo_agua = hidro_vectorial.buscar_cuerpo_mas_cercano(lat, lon, radio_km=radius_km)
        if cuerpo_agua is None:
            return jsonify({'encontrado': False})

        return jsonify({'encontrado': True, **cuerpo_agua})

    except Exception as exc:  # noqa: BLE001
        return jsonify({'error': str(exc)}), 500


@app.route('/inundacion_animacion')
def inundacion_animacion():
    """
    Fotogramas para animar la EXPANSIÓN de la crecida simulada (punto 3 del
    roadmap del simulador).

    Qué representa cada fotograma: se reutiliza `costo_acumulado` (distancia
    real, en metros, que hay que recorrer desde agua conocida atravesando
    solo terreno por debajo del umbral) como proxy de "orden de llegada":
    las celdas más cerca del río tienen costo bajo y se muestran en los
    primeros fotogramas; las más lejanas, en los últimos.

    IMPORTANTE — esto NO es una animación con tiempo real (horas/minutos).
    Convertir distancia a tiempo real requeriría velocidad de flujo real
    (caudal, pendiente, rugosidad de Manning), que es exactamente el modelo
    hidráulico completo que decidimos no construir. El frontend puede, si
    quiere, dejar que el usuario ingrese una "velocidad de avance estimada"
    y convertir la distancia a horas él mismo — eso queda explícito como
    una estimación del usuario, no un cálculo del modelo.

    Parámetros:
        lat, lon, radius_km, umbral_m   -> igual que /inundacion_tiles (tipo=zona_baja).
                                            umbral_m = metros de SUBIDA sobre el
                                            cauce normal (HAND), no cota absoluta.
        frames  cantidad de fotogramas (default 10, límite 4-15 por costo
                de cómputo — cada fotograma es una consulta a Earth Engine)
    """
    if not _ee_ready:
        return jsonify({'error': f'Earth Engine no inicializado: {_ee_error}'}), 503
    try:
        lat       = float(request.args.get('lat',  -27.48))
        lon       = float(request.args.get('lon',  -58.83))
        radius_km = min(float(request.args.get('radius_km', 200)), RADIUS_KM_MAX_INUNDACION)
        umbral_m_raw = request.args.get('umbral_m')
        frames    = int(request.args.get('frames', 10))
        frames    = max(4, min(15, frames))

        region, hand, agua_fuente = _preparar_hand_y_agua(lat, lon, radius_km)

        if umbral_m_raw is not None:
            umbral_m = float(umbral_m_raw)
            umbral_metodo = 'calibrado_usuario'
        else:
            umbral_m = SIMULADOR_HAND_UMBRAL_DEFECTO_M
            umbral_metodo = 'default_hand'

        candidatas, zona_conectada, costo_acumulado, conectividad_ok, info_motor = (
            _simular_inundacion(hand, agua_fuente, region, radius_km, umbral_m)
        )

        if not conectividad_ok:
            return jsonify({
                'error': (
                    'No se detectó agua superficial conocida (JRC) en el radio '
                    'consultado — sin eso no hay desde dónde animar la expansión.'
                ),
            }), 422

        # Distancia máxima real alcanzada (para repartir los fotogramas en
        # bandas proporcionales).
        # RENDIMIENTO — 3 ajustes para evitar timeouts en regiones grandes
        # (ej. radius_km=200 abarcando varios países/provincias):
        #   1) percentile(95) en vez de max(): un máximo exacto puede quedar
        #      dominado por un solo píxel outlier en el borde de la región,
        #      forzando a Earth Engine a escanear hasta encontrarlo. El
        #      percentil 95 da un resultado casi idéntico en la práctica
        #      (recordar: esto es un proxy de orden de avance, no un cálculo
        #      que necesite precisión de píxel) con muchísimo menos costo.
        #   2) scale=150 (en vez de 90) para esta reducción puntual — no
        #      afecta la resolución de los fotogramas en sí, solo de esta
        #      estimación de rango.
        #   3) tileScale=4: le pide a Earth Engine que parta el cómputo en
        #      tiles más chicos, reduciendo uso de memoria por tile a costa
        #      de algo más de overhead — mitiga el error típico
        #      "Too many pixels in the region" / timeouts en áreas grandes.
        max_costo_dict = costo_acumulado.reduceRegion(
            reducer=ee.Reducer.percentile([95]),
            geometry=region,
            scale=150,
            bestEffort=True,
            maxPixels=1e9,
            tileScale=4,
        ).getInfo()
        max_costo_m = 0
        for v in max_costo_dict.values():
            if v:
                max_costo_m = v
                break
        if not max_costo_m or max_costo_m <= 0:
            return jsonify({'error': 'No se pudo determinar el alcance de la crecida en esta zona.'}), 422

        fotogramas = []
        for i in range(1, frames + 1):
            umbral_costo = max_costo_m * (i / frames)
            frame_img = candidatas.updateMask(costo_acumulado.lte(umbral_costo)).selfMask()
            map_id = frame_img.getMapId({'palette': ['#1e40af']})
            fotogramas.append({
                'orden'        : i,
                'distancia_m'  : round(umbral_costo, 1),
                'porcentaje'   : round(100 * i / frames, 1),
                'tile_url'     : map_id['tile_fetcher'].url_format,
            })

        respuesta = {
            'umbral_m'        : umbral_m,
            'umbral_metodo'   : umbral_metodo,
            'umbral_tipo'     : 'subida_hand_m',
            'frames'          : frames,
            'distancia_max_m' : round(max_costo_m, 1),
            'fotogramas'      : fotogramas,
            'nota'            : (
                'La distancia es un proxy de orden de llegada (celdas más '
                'cerca del río primero), NO tiempo real. Convertir a horas '
                'requiere una velocidad de avance estimada por el usuario.'
            ),
        }
        if info_motor is not None:
            respuesta['simulacion_multicriterio'] = True
            respuesta['umbral_efectivo_m'] = info_motor['umbral_efectivo_m']
            respuesta['perfil_regional'] = info_motor['perfil']
            respuesta['nota'] += (
                ' Con la simulación multicriterio activa, "distancia_m" ya '
                'no es distancia euclidiana pura: incorpora resistencia '
                'por pendiente, cobertura y sesgo de dirección de flujo '
                '-- sigue sirviendo como orden relativo de llegada, pero '
                'el valor en metros es una unidad de costo, no una '
                'distancia física exacta.'
            )
        return jsonify(respuesta)

    except ee.EEException as exc:
        return jsonify({'error': f'Error de Earth Engine: {exc}'}), 502
    except Exception as exc:  # noqa: BLE001
        return jsonify({'error': str(exc)}), 500


@app.route('/inundacion_punto')
def inundacion_punto():
    """
    Inspección de un punto en el mapa (equivale al click interactivo del script GEE).
    Devuelve elevación, pendiente y score de riesgo hidrológico para lat/lon dados.

    Parámetros: lat, lon
    """
    if not _ee_ready:
        return jsonify({'error': f'Earth Engine no inicializado: {_ee_error}'}), 503
    try:
        lat = float(request.args['lat'])
        lon = float(request.args['lon'])
    except (KeyError, ValueError):
        return jsonify({'error': 'Parámetros lat y lon requeridos'}), 400

    try:
        region       = ee.Geometry.Point([lon, lat]).buffer(200_000)  # 200 km como AOI
        dem          = ee.Image('USGS/SRTMGL1_003').clip(region)
        elevacion_cr = dem.select('elevation')
        dosel        = _dem_composite_correccion_dosel(elevacion_cr, region)
        elevacion    = dosel.select('correctedTerrain').rename('elevation')
        pendiente    = ee.Terrain.slope(elevacion)

        smap = (ee.ImageCollection('NASA_USDA/HSL/SMAP10KM_soil_moisture')
                  .filterBounds(region)
                  .filterDate('2023-10-01', '2024-03-31')
                  .select('ssm')
                  .mean()
                  .clip(region))

        def normalizar(img, min_val, max_val):
            img_safe = ee.Image(ee.Algorithms.If(
                img.bandNames().size().gt(0), img, ee.Image.constant(0)
            ))
            return img_safe.subtract(min_val).divide(max_val - min_val).clamp(0, 1)

        n_elev = normalizar(elevacion, 30, 140).subtract(1).multiply(-1)
        n_pend = normalizar(pendiente, 0,  3 ).subtract(1).multiply(-1)
        n_smap = normalizar(smap,      10, 40)

        riesgo_compuesto = (n_elev.multiply(0.40)
                             .add(n_pend.multiply(0.30))
                             .add(n_smap.multiply(0.30))
                             .rename('riesgo'))

        punto = ee.Geometry.Point([lon, lat])
        vals  = (riesgo_compuesto
                   .addBands(elevacion)
                   .addBands(pendiente)
                   .addBands(dosel.select(['confidence', 'canopyHeight', 'correctionApplied', 'buildingHeight']))
                   .reduceRegion(
                       reducer=ee.Reducer.first(),
                       geometry=punto,
                       scale=30,
                       bestEffort=True
                   ))

        result   = vals.getInfo()
        elev     = result.get('elevation')
        pend     = result.get('slope')
        ries     = result.get('riesgo')
        conf_num = result.get('confidence')
        dosel_m  = result.get('correctionApplied')
        chm_m    = result.get('canopyHeight')
        edif_m   = result.get('buildingHeight')

        conf_map = {1: '🔴 Baja', 2: '🟡 Media', 3: '🟢 Alta'}
        confianza_dem = conf_map.get(int(conf_num)) if conf_num is not None else None

        # Clasificación topográfica (igual que el script GEE)
        if elev is not None:
            if   elev < 55:  topo = 'Zona Baja (<55 m)'
            elif elev < 90:  topo = 'Zona Media (55-90 m)'
            else:            topo = 'Zona Alta (>90 m)'
        else:
            topo = 'Sin datos'

        # Nivel de riesgo
        if ries is not None:
            pct = round(ries * 100, 1)
            if   pct > 65: nivel = '🔴 ALTO'
            elif pct > 40: nivel = '🟡 MEDIO'
            else:          nivel = '🟢 BAJO'
        else:
            pct   = None
            nivel = 'Sin datos'

        return jsonify({
            'ok'             : True,
            'lat'            : lat,
            'lon'            : lon,
            'elevacion'      : round(elev, 1) if elev is not None else None,
            'pendiente'      : round(pend, 2) if pend is not None else None,
            'riesgo'         : pct,
            'nivel'          : nivel,
            'topo'           : topo,
            'confianza_dem'  : confianza_dem,
            'dosel_altura_m' : round(chm_m, 1) if chm_m is not None else None,
            'dosel_corregido_m': round(dosel_m, 2) if dosel_m is not None else None,
            'edificio_altura_m': round(edif_m, 1) if edif_m is not None else None,
        })

    except ee.EEException as exc:
        return jsonify({'error': f'Error de Earth Engine: {exc}'}), 502
    except Exception as exc:  # noqa: BLE001
        return jsonify({'error': str(exc)}), 500


@app.route('/direccion_flujo')
def direccion_flujo():
    """
    Grilla de dirección de flujo superficial real, calculada sobre el DEM SRTM.

    Usa ee.Terrain.products(), que además de 'slope' calcula 'aspect':
    el rumbo (0-360°, 0°=Norte, sentido horario) de la línea de máxima
    pendiente DESCENDENTE en cada píxel. Esa es, por definición, la
    dirección hacia donde escurre el agua en ese punto del terreno.

    A diferencia de /zonas_inundables?tipo=pendiente (que solo da la
    MAGNITUD de la pendiente como tile de imagen), este endpoint devuelve
    valores numéricos por punto, necesarios para dibujar vectores de
    dirección reales en el frontend.

    La grilla de puntos se arma en Python (simple grilla lat/lon) y se
    muestrea toda junta con reduceRegions() -> una sola llamada a Earth
    Engine en vez de una por punto.

    Parámetros:
        lat, lon    centro del área (default: Corrientes, Argentina)
        radius_km   radio del área a muestrear (default 50, max 150)
        n           puntos por lado de la grilla (default 12, max 25 ->
                    hasta 625 puntos, para no saturar la respuesta ni EE)

    Devuelve: { puntos: [{lat, lon, aspecto_deg, pendiente_deg}, ...] }
    """
    if not _ee_ready:
        return jsonify({'error': f'Earth Engine no inicializado: {_ee_error}'}), 503
    try:
        lat       = float(request.args.get('lat',  -27.48))
        lon       = float(request.args.get('lon',  -58.83))
        radius_km = min(float(request.args.get('radius_km', 50)), 150)
        n         = max(5, min(int(request.args.get('n', 12)), 25))

        region     = ee.Geometry.Point([lon, lat]).buffer(radius_km * 1000)
        dem_crudo  = ee.Image('USGS/SRTMGL1_003').clip(region)
        dem        = _dem_corregido_por_dosel(dem_crudo.select('elevation'), region).rename('elevation')
        terreno    = ee.Terrain.products(dem)                   # elevation, slope, aspect, hillshade
        superficie = terreno.select(['slope', 'aspect'])

        # Grilla regular de puntos en grados (aprox. — suficiente para
        # separar vectores visualmente, no requiere precisión geodésica).
        deg_radius = radius_km / 111.0
        paso = (2 * deg_radius) / max(n - 1, 1)
        features = []
        for i in range(n):
            la = lat - deg_radius + i * paso
            for j in range(n):
                lo = lon - deg_radius + j * paso
                features.append(
                    ee.Feature(ee.Geometry.Point([lo, la]), {'lat': la, 'lon': lo})
                )
        puntos = ee.FeatureCollection(features)

        # scale=90 (~3 celdas SRTM) suaviza ruido píxel a píxel manteniendo
        # el patrón general de drenaje — evita vectores "nerviosos".
        muestreo = superficie.reduceRegions(
            collection=puntos,
            reducer=ee.Reducer.first(),
            scale=90,
        )

        info = muestreo.getInfo()
        resultado = []
        for feat in info.get('features', []):
            p   = feat.get('properties', {})
            asp = p.get('aspect')
            pen = p.get('slope')
            if asp is None or pen is None:
                continue
            resultado.append({
                'lat'          : round(p['lat'], 5),
                'lon'          : round(p['lon'], 5),
                'aspecto_deg'  : round(asp, 1),
                'pendiente_deg': round(pen, 2),
            })

        return jsonify({
            'ok'       : True,
            'puntos'   : resultado,
            'n_lado'   : n,
            'radius_km': radius_km,
            'nota'     : ('aspecto_deg: 0-360°, 0=Norte, sentido horario. '
                          'Dirección de máxima pendiente descendente '
                          '(ee.Terrain.aspect) — apunta hacia donde escurre el agua.'),
        })

    except ee.EEException as exc:
        return jsonify({'error': f'Error de Earth Engine: {exc}'}), 502
    except Exception as exc:  # noqa: BLE001
        return jsonify({'error': str(exc)}), 500


@app.route('/escurrimiento_red')
def escurrimiento_red():
    """
    Red de drenaje vectorizada a partir de MERIT Hydro (Camino A: dataset
    global precomputado, en vez de correr D8/D∞ a mano sobre el SRTM local).

    MERIT Hydro ('MERIT/Hydro/v1_0_1') ya trae, por píxel (~90m):
      - 'upa': área de drenaje acumulada aguas arriba, en km².
      - 'dir': código de dirección de flujo D8 (1,2,4,8,16,32,64,128).

    Se arma una máscara binaria 'upa >= umbral_km2' (umbral definido por el
    slider "poca red / red densa" del frontend) y se vectoriza con
    reduceToVectors(). El resultado son POLÍGONOS delgados de píxeles
    conectados -- no líneas centrales topológicas reales (GEE no tiene
    esqueletización nativa) -- pero visualmente, estilados como cinta azul
    fina, se leen como cauces. Cada polígono lleva la acumulación y
    pendiente PROMEDIO de sus píxeles, para graduar grosor/color en el
    frontend.

    Parámetros:
        lat, lon    centro del área (default: Corrientes, Argentina)
        radius_km   radio del área a analizar (default 40, max 100 --
                    reduceToVectors es pesado, no conviene abusar del radio)
        densidad    1 (poca red) a 5 (red densa) -> mapea a umbral de km²

    Devuelve: GeoJSON FeatureCollection de polígonos con propiedades
    'acumulacion_km2' y 'pendiente_grados' por tramo.
    """
    if not _ee_ready:
        return jsonify({'error': f'Earth Engine no inicializado: {_ee_error}'}), 503
    try:
        lat       = float(request.args.get('lat', -27.48))
        lon       = float(request.args.get('lon', -58.83))
        radius_km = min(float(request.args.get('radius_km', 40)), 100)
        densidad  = max(1, min(int(request.args.get('densidad', 3)), 5))
        umbral_km2 = MERIT_HYDRO_UMBRALES_KM2[densidad]

        region = ee.Geometry.Point([lon, lat]).buffer(radius_km * 1000)
        merit  = ee.Image('MERIT/Hydro/v1_0_1').clip(region)
        acumulacion = merit.select('upa')

        dem_crudo = ee.Image('USGS/SRTMGL1_003').select('elevation').clip(region)
        pendiente = ee.Terrain.slope(dem_crudo)

        mascara = acumulacion.gte(umbral_km2).selfMask()

        # Imagen multibanda: 1ra banda = zonas a agrupar (mascara), el resto
        # se promedia por zona con el reducer -> quedan como propiedades.
        insumo = mascara.rename('zona').addBands(acumulacion.rename('acumulacion_km2')) \
                         .addBands(pendiente.rename('pendiente_grados'))

        vectores = insumo.reduceToVectors(
            geometry=region,
            scale=90,
            geometryType='polygon',
            eightConnected=True,
            reducer=ee.Reducer.mean(),
            labelProperty='zona',
            maxPixels=1e9,
            bestEffort=True,
            tileScale=4,
        )

        geojson = vectores.getInfo()

        return jsonify({
            'ok'            : True,
            'geojson'       : geojson,
            'densidad'      : densidad,
            'umbral_km2'    : umbral_km2,
            'radius_km'     : radius_km,
            'nota'          : ('Polígonos de MERIT Hydro (upa >= umbral). Aproximación '
                               'visual de la red de drenaje, no líneas centrales exactas.'),
        })

    except ee.EEException as exc:
        return jsonify({'error': f'Error de Earth Engine: {exc}'}), 502
    except Exception as exc:  # noqa: BLE001
        return jsonify({'error': str(exc)}), 500


@app.route('/escurrimiento_punto')
def escurrimiento_punto():
    """
    Info hidrológica puntual para el click sobre la capa de escurrimiento
    profesional: dirección de flujo, pendiente, acumulación y elevación en
    un punto exacto (en vez de atributos pre-calculados por tramo, que GEE
    no puede dar de forma topológicamente exacta -- ver /escurrimiento_red).

    Parámetros: lat, lon (obligatorios)
    Devuelve: aspecto_deg, pendiente_deg, acumulacion_km2, elevacion_m
    """
    if not _ee_ready:
        return jsonify({'error': f'Earth Engine no inicializado: {_ee_error}'}), 503
    try:
        lat = float(request.args['lat'])
        lon = float(request.args['lon'])

        punto  = ee.Geometry.Point([lon, lat])
        region = punto.buffer(200)  # margen chico para el reduceRegion

        merit     = ee.Image('MERIT/Hydro/v1_0_1')
        dem_crudo = ee.Image('USGS/SRTMGL1_003').select('elevation')
        terreno   = ee.Terrain.products(dem_crudo)

        insumo = (merit.select('upa').rename('acumulacion_km2')
                  .addBands(terreno.select('aspect').rename('aspecto_deg'))
                  .addBands(terreno.select('slope').rename('pendiente_deg'))
                  .addBands(dem_crudo.rename('elevacion_m')))

        valores = insumo.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=region,
            scale=90,
            bestEffort=True,
        ).getInfo()

        if not valores or valores.get('acumulacion_km2') is None:
            return jsonify({'error': 'Sin datos hidrológicos en ese punto.'}), 404

        return jsonify({
            'ok'              : True,
            'lat'             : round(lat, 5),
            'lon'             : round(lon, 5),
            'acumulacion_km2' : round(valores['acumulacion_km2'], 2),
            'aspecto_deg'     : round(valores['aspecto_deg'], 1) if valores.get('aspecto_deg') is not None else None,
            'pendiente_deg'   : round(valores['pendiente_deg'], 2) if valores.get('pendiente_deg') is not None else None,
            'elevacion_m'     : round(valores['elevacion_m'], 1) if valores.get('elevacion_m') is not None else None,
        })

    except ee.EEException as exc:
        return jsonify({'error': f'Error de Earth Engine: {exc}'}), 502
    except Exception as exc:  # noqa: BLE001
        return jsonify({'error': str(exc)}), 500


# ──────────────────────────────────────────────────────────────────────────
# MÓDULO COPERNICUS (Sentinel-2 vía Sentinel Hub OGC WMS / CDSE)
# ──────────────────────────────────────────────────────────────────────────
# Módulo independiente: no modifica ni reutiliza ninguna función de Earth
# Engine de arriba. Solo agrega dos rutas nuevas.
#
# Variables de entorno requeridas (configurar en Render):
#   COPERNICUS_CLIENT_ID
#   COPERNICUS_CLIENT_SECRET
#   COPERNICUS_INSTANCE_ID
#
# Requiere agregar 'requests' a requirements.txt (ya agregado si seguiste
# las instrucciones previas).
# ──────────────────────────────────────────────────────────────────────────

CDSE_TOKEN_URL = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
SH_WMS_BASE_TEMPLATE = 'https://sh.dataspace.copernicus.eu/ogc/wms/{instance_id}'

COPERNICUS_CLIENT_ID = os.environ.get('COPERNICUS_CLIENT_ID')
COPERNICUS_CLIENT_SECRET = os.environ.get('COPERNICUS_CLIENT_SECRET')
COPERNICUS_INSTANCE_ID = os.environ.get('COPERNICUS_INSTANCE_ID')

# Cache simple de token OAuth2 (client_credentials) en memoria de proceso.
_cop_token_lock = threading.Lock()
_cop_token_cache = {'access_token': None, 'expires_at': 0}


def _copernicus_obtener_token():
    """Devuelve un access_token válido, renovándolo si falta poco para expirar."""
    with _cop_token_lock:
        ahora = time.time()
        if _cop_token_cache['access_token'] and _cop_token_cache['expires_at'] - 60 > ahora:
            return _cop_token_cache['access_token']

        resp = requests.post(CDSE_TOKEN_URL, data={
            'grant_type': 'client_credentials',
            'client_id': COPERNICUS_CLIENT_ID,
            'client_secret': COPERNICUS_CLIENT_SECRET,
        }, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        _cop_token_cache['access_token'] = data['access_token']
        _cop_token_cache['expires_at'] = ahora + data.get('expires_in', 300)
        return _cop_token_cache['access_token']


# Evalscripts por modo. Los de índices (NDVI, NDWI, MNDWI, NBR) están
# simplificados en escala de grises; antes de producción, reemplazar por
# los evalscripts oficiales con paleta de color desde
# https://custom-scripts.sentinel-hub.com/
COPERNICUS_EVALSCRIPTS = {
    'RGB': """
        //VERSION=3
        function setup() { return { input: ["B04","B03","B02"], output: { bands: 3 } }; }
        function evaluatePixel(s) { return [2.5*s.B04, 2.5*s.B03, 2.5*s.B02]; }
    """,
    'FALSO_COLOR': """
        //VERSION=3
        function setup() { return { input: ["B08","B04","B03"], output: { bands: 3 } }; }
        function evaluatePixel(s) { return [2.5*s.B08, 2.5*s.B04, 2.5*s.B03]; }
    """,
    'NDVI': """
        //VERSION=3
        function setup() { return { input: ["B08","B04"], output: { bands: 3 } }; }
        function evaluatePixel(s) {
            let ndvi = (s.B08 - s.B04) / (s.B08 + s.B04);
            return [ndvi, ndvi, ndvi];
        }
    """,
    'NDWI': """
        //VERSION=3
        function setup() { return { input: ["B03","B08"], output: { bands: 3 } }; }
        function evaluatePixel(s) {
            let ndwi = (s.B03 - s.B08) / (s.B03 + s.B08);
            return [ndwi, ndwi, ndwi];
        }
    """,
    'MNDWI': """
        //VERSION=3
        function setup() { return { input: ["B03","B11"], output: { bands: 3 } }; }
        function evaluatePixel(s) {
            let mndwi = (s.B03 - s.B11) / (s.B03 + s.B11);
            return [mndwi, mndwi, mndwi];
        }
    """,
    'NBR': """
        //VERSION=3
        function setup() { return { input: ["B08","B12"], output: { bands: 3 } }; }
        function evaluatePixel(s) {
            let nbr = (s.B08 - s.B12) / (s.B08 + s.B12);
            return [nbr, nbr, nbr];
        }
    """,
}


@app.route('/copernicus_wms')
def copernicus_wms():
    """
    Proxy hacia el WMS de Sentinel Hub / Copernicus Data Space Ecosystem.
    El frontend (Leaflet) pide tiles a ESTA ruta; acá se agrega el token
    OAuth2 y se reenvía. El navegador nunca ve las credenciales.
    """
    if not (COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET and COPERNICUS_INSTANCE_ID):
        return jsonify({'error': 'Copernicus no configurado en el servidor (faltan variables de entorno).'}), 503

    try:
        token = _copernicus_obtener_token()
        modo = request.args.get('evalscript', 'RGB')
        evalscript = COPERNICUS_EVALSCRIPTS.get(modo, COPERNICUS_EVALSCRIPTS['RGB'])

        wms_url = SH_WMS_BASE_TEMPLATE.format(instance_id=COPERNICUS_INSTANCE_ID)
        wms_params = {
            'SERVICE': 'WMS',
            'REQUEST': 'GetMap',
            'LAYERS': request.args.get('layers', 'SENTINEL-2-L2A'),
            'FORMAT': request.args.get('format', 'image/png'),
            'TRANSPARENT': request.args.get('transparent', 'true'),
            'BBOX': request.args.get('bbox'),
            'WIDTH': request.args.get('width', '256'),
            'HEIGHT': request.args.get('height', '256'),
            'SRS': request.args.get('srs', 'EPSG:3857'),
            'TIME': request.args.get('time'),
            'MAXCC': request.args.get('maxcc', '30'),
            'EVALSCRIPT': evalscript,
        }

        r = requests.get(wms_url, params=wms_params,
                          headers={'Authorization': f'Bearer {token}'},
                          timeout=20)

        return Response(r.content, status=r.status_code,
                         content_type=r.headers.get('Content-Type', 'image/png'),
                         headers={'Cache-Control': 'public, max-age=3600'})
    except Exception as exc:  # noqa: BLE001
        return jsonify({'error': str(exc)}), 502


@app.route('/copernicus_info')
def copernicus_info():
    """
    Busca (vía Catalog/STAC API de CDSE) la escena Sentinel-2 real más
    adecuada para lat/lon/fecha/nubosidad, en vez de asumir que existe.

    Parámetros: lat, lon, fecha (YYYY-MM-DD), nubosidad (0-100)
    """
    if not (COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET):
        return jsonify({'disponible': False, 'mensaje': 'Copernicus no configurado en el servidor.'}), 503

    try:
        lat = float(request.args.get('lat'))
        lon = float(request.args.get('lon'))
        fecha = request.args.get('fecha')
        nubosidad_max = float(request.args.get('nubosidad', 30))

        if not fecha:
            return jsonify({'disponible': False, 'mensaje': 'Falta el parámetro fecha.'}), 400

        delta = 0.02  # ~2 km alrededor del punto
        bbox = [lon - delta, lat - delta, lon + delta, lat + delta]

        # Sentinel-2 revisita el mismo punto cada ~5 días: buscamos en una
        # ventana de +-4 días alrededor de la fecha pedida y nos quedamos
        # con la escena más cercana a esa fecha (entre las que cumplen
        # nubosidad). Buscar un único día exacto suele no encontrar nada.
        fecha_dt = datetime.datetime.strptime(fecha, '%Y-%m-%d')
        desde = (fecha_dt - datetime.timedelta(days=4)).strftime('%Y-%m-%dT00:00:00Z')
        hasta = (fecha_dt + datetime.timedelta(days=4)).strftime('%Y-%m-%dT23:59:59Z')

        # NOTA: el catálogo STAC "legacy" (catalogue.dataspace.copernicus.eu/stac)
        # fue discontinuado por Copernicus (nov. 2025). Usamos el catálogo STAC
        # nuevo, que además es de lectura pública (no requiere token).
        catalog_url = 'https://stac.dataspace.copernicus.eu/v1/collections/sentinel-2-l2a/items'
        r = requests.get(catalog_url, params={
            'bbox': ','.join(map(str, bbox)),
            'datetime': f'{desde}/{hasta}',
            'limit': 50,
        }, timeout=20)

        if r.status_code != 200:
            return jsonify({
                'disponible': False,
                'mensaje': f'Error consultando el catálogo de Copernicus (HTTP {r.status_code}).'
            }), 502

        items = r.json().get('features', [])
        candidatas = [it for it in items
                      if it.get('properties', {}).get('eo:cloud_cover', 100) <= nubosidad_max]

        if not candidatas:
            return jsonify({
                'disponible': False,
                'mensaje': ('No hay imágenes Sentinel-2 con esa nubosidad '
                            'en +-4 días de la fecha elegida.')
            })

        # De las candidatas, la más cercana a la fecha pedida
        def _dist_fecha(it):
            dt = it['properties'].get('datetime', '')
            try:
                d = datetime.datetime.strptime(dt[:10], '%Y-%m-%d')
                return abs((d - fecha_dt).days)
            except ValueError:
                return 999

        mejor = min(candidatas, key=_dist_fecha)
        props = mejor['properties']

        return jsonify({
            'disponible': True,
            'fecha_adquisicion': props.get('datetime', fecha)[:10],
            'satelite': props.get('platform', 'Sentinel-2'),
            'nubosidad_pct': round(props.get('eo:cloud_cover', 0), 1),
            'resolucion_m': 10,
        })
    except Exception as exc:  # noqa: BLE001
        return jsonify({'disponible': False, 'mensaje': str(exc)}), 500


# ──────────────────────────────────────────────────────────────────────────
# MÓDULO CONECTIVIDAD HIDRÁULICA
# ──────────────────────────────────────────────────────────────────────────
# Reutiliza _preparar_dem_y_agua y _conectividad_hidraulica TAL CUAL existen
# (no se modifican). La novedad es:
#   - MERIT Hydro (MERIT/Hydro/v1_0_1): banda 'hnd' = altura sobre el
#     drenaje más cercano SIGUIENDO EL RELIEVE (ya resuelve "barreras
#     topográficas" y "diferencia de altura" sin tener que derivarlo a
#     mano), y 'upa' = área de drenaje acumulada (proxy de acumulación
#     de flujo).
#   - _conectividad_hidraulica() se reusa pasándole `hand` en vez de
#     `elevacion`: la función solo hace candidatas = imagen.lte(umbral),
#     así que sirve igual para umbral de HAND que para umbral de
#     elevación absoluta. Es una barrera más realista: si hay una loma
#     entre el río y el punto, el HAND ahí es alto y cumulativeCost no
#     puede atravesarla.
#   - hidrografia_vectorial: identifica el cuerpo de agua real más
#     cercano (río/arroyo/laguna/estero/...) y su coeficiente.
#
# Pesos y umbrales configurables acá abajo, sin tocar el algoritmo.
# ──────────────────────────────────────────────────────────────────────────

CONECTIVIDAD_UMBRAL_HAND_M = 15       # HAND por debajo del cual una celda es "candidata"
CONECTIVIDAD_ESCALA_DIST_M = 3000     # escala de decaimiento de la distancia hidráulica
CONECTIVIDAD_PESOS = {                # deben sumar 1.0
    'distancia': 0.5,
    'pendiente': 0.3,
    'flujo':     0.2,
}


def _score_exponencial_decreciente(valor, escala):
    """0-1: 1 cuando valor=0, decae exponencialmente al crecer valor."""
    try:
        return math.exp(-valor / escala) if valor is not None else 0.0
    except (TypeError, ZeroDivisionError):
        return 0.0


def _calcular_conectividad_hidraulica_punto(lat, lon, radius_km):
    """
    Función reutilizable (usada por /conectividad_hidraulica y por el IIPDI):
    calcula el índice de conectividad hidráulica para un punto. Devuelve un
    dict con exactamente los mismos campos que ya devolvía el endpoint.
    """
    region, elevacion, agua_fuente = _preparar_dem_y_agua(lat, lon, radius_km)

    merit = ee.Image('MERIT/Hydro/v1_0_1').clip(region)
    hand = merit.select('hnd')          # altura sobre drenaje más cercano
    upa = merit.select('upa')           # área de drenaje acumulada (flujo)
    pendiente = ee.Terrain.slope(elevacion)

    _candidatas, _zona_conectada, costo_acumulado, conectividad_ok = (
        _conectividad_hidraulica(hand, agua_fuente, region, radius_km,
                                  CONECTIVIDAD_UMBRAL_HAND_M)
    )

    punto = ee.Geometry.Point([lon, lat])

    valores_punto = ee.Image.cat([
        hand.rename('hand'),
        pendiente.rename('pendiente'),
        upa.rename('upa'),
        costo_acumulado.rename('distancia_hidraulica'),
    ]).reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=punto,
        scale=90,
        bestEffort=True,
        tileScale=4,
    ).getInfo()

    hand_punto = valores_punto.get('hand')
    pendiente_punto = valores_punto.get('pendiente')
    upa_punto = valores_punto.get('upa')
    distancia_hidraulica_m = valores_punto.get('distancia_hidraulica')

    punto_conectado = conectividad_ok and distancia_hidraulica_m is not None

    # Si el módulo hidrografia_vectorial no está desplegado, seguimos sin el
    # coeficiente por tipo de cuerpo de agua (queda en su valor neutro 1.0)
    # en vez de que esta función completa reviente con AttributeError.
    cuerpo_agua = (
        hidro_vectorial.buscar_cuerpo_mas_cercano(lat, lon, radio_km=radius_km)
        if _hidro_vectorial_disponible else None
    )
    coeficiente_tipo = cuerpo_agua['coeficiente'] if cuerpo_agua else 1.0

    if not punto_conectado:
        indice = 0.0
    else:
        score_distancia = _score_exponencial_decreciente(
            distancia_hidraulica_m, CONECTIVIDAD_ESCALA_DIST_M)
        score_pendiente = _score_exponencial_decreciente(pendiente_punto or 0, 8.0)
        score_flujo = min((math.log10(upa_punto + 1) / 6.0), 1.0) if upa_punto else 0.0

        conectividad_terreno = 100 * (
            CONECTIVIDAD_PESOS['distancia'] * score_distancia +
            CONECTIVIDAD_PESOS['pendiente'] * score_pendiente +
            CONECTIVIDAD_PESOS['flujo'] * score_flujo
        )
        indice = round(min(max(conectividad_terreno * coeficiente_tipo, 0), 100), 1)

    return {
        'indice_conectividad': indice,
        'conectado': bool(punto_conectado),
        'distancia_hidraulica_m': round(distancia_hidraulica_m, 1) if distancia_hidraulica_m is not None else None,
        'diferencia_altura_m': round(hand_punto, 2) if hand_punto is not None else None,
        'pendiente_grados': round(pendiente_punto, 2) if pendiente_punto is not None else None,
        'acumulacion_flujo': upa_punto,
        'cuerpo_agua_dominante': cuerpo_agua['nombre'] if cuerpo_agua else None,
        'tipo_cuerpo_agua': cuerpo_agua['tipo'] if cuerpo_agua else None,
        'coeficiente_tipo_agua': coeficiente_tipo,
        'distancia_al_cuerpo_vectorial_m': cuerpo_agua['distancia_m'] if cuerpo_agua else None,
        'superficie_ha_cuerpo_agua': cuerpo_agua.get('superficie_ha') if cuerpo_agua else None,
    }


@app.route('/conectividad_hidraulica')
def conectividad_hidraulica():
    """
    Índice de Conectividad Hidráulica (0-100) para un punto: qué tan
    probable es que un cuerpo de agua cercano, si sube de nivel, llegue
    a ese punto siguiendo el relieve real (no en línea recta).

    Parámetros: lat, lon, radius_km (default 15, tope 30)
    """
    if not _ee_ready:
        return jsonify({'error': f'Earth Engine no inicializado: {_ee_error}'}), 503

    try:
        lat = float(request.args.get('lat'))
        lon = float(request.args.get('lon'))
        radius_km = min(float(request.args.get('radius_km', 15)), 30)
        return jsonify(_calcular_conectividad_hidraulica_punto(lat, lon, radius_km))
    except Exception as exc:  # noqa: BLE001
        return jsonify({'error': str(exc)}), 500


# ──────────────────────────────────────────────────────────────────────────
# MÓDULO IIPDI (Índice Inteligente de Probabilidad de Desborde e Inundación)
# ──────────────────────────────────────────────────────────────────────────
# FASE ACTUAL: Modo Simple, consulta puntual. Combina:
#   Módulo 1 - Susceptibilidad Geomorfológica (elevación relativa, pendiente,
#              acumulación de flujo, TWI)
#   Módulo 2 - Conectividad Hidráulica (reusa _calcular_conectividad_hidraulica_punto,
#              SIN recalcularla desde cero)
#   Módulo 3 - Condición Hidrometeorológica (lluvia 24h/72h/7d/30d vía CHIRPS,
#              NDWI/MNDWI/AWEI vía Sentinel-2)
#   Módulo 5 - Influencia del cuerpo de agua (tipo, superficie, vía capa vectorial)
#
# PENDIENTE PARA PRÓXIMAS FASES (documentado, no implementado todavía):
#   - Módulo 4 (expansión observada: cambio NDWI/MNDWI/Sentinel-1 entre 2 fechas)
#   - Modo AHP con calibración automática por registros históricos
#   - Modo raster (área visible completa, tileado)
#   - Humedad de suelo (SMAP) y evapotranspiración (MODIS ET) en Módulo 3
#
# Pesos en archivo de configuración (no hardcodeados en el algoritmo).
# ──────────────────────────────────────────────────────────────────────────

IIPDI_PESOS = {
    'geomorfologia': 0.30,
    'conectividad_hidraulica': 0.25,
    'condicion_hidrometeorologica': 0.20,
    'expansion_observada': 0.15,   # Módulo 4: no implementado aún, peso reservado
    'cuerpo_agua': 0.10,
}

IIPDI_NIVELES = [
    (20, 'Muy Bajo'), (40, 'Bajo'), (60, 'Moderado'), (80, 'Alto'), (101, 'Muy Alto'),
]


def _nivel_iipdi(valor):
    for umbral, nombre in IIPDI_NIVELES:
        if valor < umbral:
            return nombre
    return 'Muy Alto'


def _susceptibilidad_geomorfologica_punto(lat, lon, region, elevacion, radius_km):
    """
    Módulo 1: 0-100. Combina elevación relativa (respecto al entorno),
    pendiente, acumulación de flujo y TWI (Topographic Wetness Index).
    TWI = ln(area_acumulada / tan(pendiente)): a mayor TWI, terreno más
    propenso a saturarse de agua.
    """
    merit = ee.Image('MERIT/Hydro/v1_0_1').clip(region)
    upa = merit.select('upa')  # km2 aprox, proxy de área acumulada de drenaje
    pendiente_grados = ee.Terrain.slope(elevacion)
    pendiente_rad = pendiente_grados.multiply(math.pi / 180).max(0.001)  # evita tan(0)

    twi = upa.multiply(1e6).divide(pendiente_rad.tan()).log().rename('twi')

    punto = ee.Geometry.Point([lon, lat])
    elev_media_region = elevacion.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=region, scale=90, bestEffort=True
    ).get('elevation')

    valores = ee.Image.cat([
        elevacion.rename('elev'),
        pendiente_grados.rename('pendiente'),
        twi,
    ]).reduceRegion(
        reducer=ee.Reducer.first(), geometry=punto, scale=90, bestEffort=True, tileScale=4
    ).getInfo()

    elev_punto = valores.get('elev')
    pendiente_punto = valores.get('pendiente')
    twi_punto = valores.get('twi')
    elev_media = elev_media_region.getInfo() if elev_media_region else None

    elev_relativa = (elev_punto - elev_media) if (elev_punto is not None and elev_media is not None) else 0

    # Scores 0-1: elevación relativa baja/negativa = más susceptible;
    # pendiente baja = más susceptible; TWI alto = más susceptible.
    score_elev = _score_exponencial_decreciente(max(elev_relativa, 0), 5.0)
    score_pendiente = _score_exponencial_decreciente(pendiente_punto or 0, 6.0)
    score_twi = min(max((twi_punto or 0) / 12.0, 0), 1) if twi_punto is not None else 0

    geomorfologia_0_100 = 100 * (0.4 * score_elev + 0.3 * score_pendiente + 0.3 * score_twi)

    return {
        'valor': round(min(max(geomorfologia_0_100, 0), 100), 1),
        'elevacion_relativa_m': round(elev_relativa, 2) if elev_relativa is not None else None,
        'pendiente_grados': round(pendiente_punto, 2) if pendiente_punto is not None else None,
        'twi': round(twi_punto, 2) if twi_punto is not None else None,
    }


def _condicion_hidrometeorologica_punto(lat, lon, radius_km):
    """
    Módulo 3: 0-100. Lluvia acumulada en 4 ventanas (CHIRPS) + índices de
    agua superficial NDWI/MNDWI/AWEI (Sentinel-2, últimos 30 días).
    NOTA: humedad de suelo (SMAP) y evapotranspiración (MODIS ET) quedan
    documentadas como pendientes para una próxima iteración.
    """
    region = ee.Geometry.Point([lon, lat]).buffer(radius_km * 1000)
    hoy = ee.Date(datetime.date.today().isoformat())

    chirps_col = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY').filterBounds(region)

    def _lluvia_ventana(dias):
        desde = hoy.advance(-dias, 'day')
        coleccion_ventana = chirps_col.filterDate(desde, hoy)
        if coleccion_ventana.size().getInfo() == 0:
            return 0.0  # CHIRPS suele tener unos días de retraso en publicarse
        suma = coleccion_ventana.sum()
        val = suma.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=region, scale=5000, bestEffort=True
        ).get('precipitation').getInfo()
        return round(val, 1) if val is not None else 0.0

    lluvia_24h = _lluvia_ventana(1)
    lluvia_72h = _lluvia_ventana(3)
    lluvia_7d = _lluvia_ventana(7)
    lluvia_30d = _lluvia_ventana(30)

    # Índices de agua superficial: Sentinel-2, últimos 30 días, mediana.
    col_s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filterDate(hoy.advance(-30, 'day'), hoy)
              .filterBounds(region)
              .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 40)))

    indices = {'ndwi': None, 'mndwi': None, 'awei': None}
    if col_s2.size().getInfo() > 0:
        img = col_s2.median()
        ndwi = img.normalizedDifference(['B3', 'B8']).rename('ndwi')
        mndwi = img.normalizedDifference(['B3', 'B11']).rename('mndwi')
        refl = img.select(['B2', 'B3', 'B8', 'B11', 'B12']).multiply(0.0001)
        awei = refl.expression(
            '4 * (GREEN - SWIR1) - (0.25 * NIR + 2.75 * SWIR2)',
            {'GREEN': refl.select('B3'), 'SWIR1': refl.select('B11'),
             'NIR': refl.select('B8'), 'SWIR2': refl.select('B12')}
        ).rename('awei')

        stats = ee.Image.cat([ndwi, mndwi, awei]).reduceRegion(
            reducer=ee.Reducer.mean(), geometry=region, scale=100, bestEffort=True
        ).getInfo()
        indices = {
            'ndwi': round(stats.get('ndwi'), 3) if stats.get('ndwi') is not None else None,
            'mndwi': round(stats.get('mndwi'), 3) if stats.get('mndwi') is not None else None,
            'awei': round(stats.get('awei'), 3) if stats.get('awei') is not None else None,
        }

    # Scores 0-1: más lluvia reciente = más peso (24h/72h pesan más que 30d,
    # porque son más indicativas de saturación actual). Índices de agua
    # positivos y altos = terreno ya saturado/con agua superficial.
    score_lluvia = min(
        (0.4 * min(lluvia_24h / 80, 1) +
         0.3 * min(lluvia_72h / 150, 1) +
         0.2 * min(lluvia_7d / 250, 1) +
         0.1 * min(lluvia_30d / 400, 1)), 1
    )
    score_indices_agua = 0.0
    valores_indices = [v for v in indices.values() if v is not None]
    if valores_indices:
        promedio_indices = sum(valores_indices) / len(valores_indices)
        score_indices_agua = min(max((promedio_indices + 0.3) / 0.6, 0), 1)

    hidrometeorologica_0_100 = 100 * (0.7 * score_lluvia + 0.3 * score_indices_agua)

    return {
        'valor': round(min(max(hidrometeorologica_0_100, 0), 100), 1),
        'lluvia_24h_mm': lluvia_24h,
        'lluvia_72h_mm': lluvia_72h,
        'lluvia_7d_mm': lluvia_7d,
        'lluvia_30d_mm': lluvia_30d,
        'ndwi': indices['ndwi'],
        'mndwi': indices['mndwi'],
        'awei': indices['awei'],
    }


def _influencia_cuerpo_agua_punto(cuerpo_agua):
    """
    Módulo 5: 0-100. Basado en el coeficiente por tipo (ya calculado por
    hidrografia_vectorial) y, si está disponible, la superficie del cuerpo
    (cuerpos más grandes = más influencia potencial).
    NOTA: perímetro/forma/jerarquía con otros cuerpos quedan pendientes
    (requieren procesar geometría completa, no solo el punto más cercano).
    """
    if not cuerpo_agua:
        return {'valor': 0.0, 'superficie_ha': None}

    coef = cuerpo_agua.get('coeficiente', 0.5)
    superficie_ha = cuerpo_agua.get('superficie_ha')
    score_superficie = min(math.log10((superficie_ha or 1) + 1) / 4.0, 1.0) if superficie_ha else 0.5

    valor = 100 * (0.7 * coef + 0.3 * score_superficie)
    return {'valor': round(min(max(valor, 0), 100), 1), 'superficie_ha': superficie_ha}


@app.route('/iipdi_punto')
def iipdi_punto():
    """
    Índice Inteligente de Probabilidad de Desborde e Inundación (0-100),
    Modo Simple, consulta puntual. Combina Módulos 1, 2, 3 y 5 con pesos
    configurables (IIPDI_PESOS). El Módulo 4 (expansión observada) todavía
    no está implementado; su peso se redistribuye proporcionalmente entre
    los demás módulos para que el total siga sumando 100%.

    Parámetros: lat, lon, radius_km (default 15, tope 30)
    """
    if not _ee_ready:
        return jsonify({'error': f'Earth Engine no inicializado: {_ee_error}'}), 503
    if not _hidro_vectorial_disponible:
        return jsonify({
            'error': (
                'Módulo hidrografia_vectorial no disponible en el servidor. '
                'Subí hidrografia_vectorial.py junto a app.py y volvé a desplegar.'
            ),
        }), 503

    try:
        lat = float(request.args.get('lat'))
        lon = float(request.args.get('lon'))
        radius_km = min(float(request.args.get('radius_km', 15)), 30)

        region, elevacion, _agua_fuente = _preparar_dem_y_agua(lat, lon, radius_km)

        geomorfologia = _susceptibilidad_geomorfologica_punto(lat, lon, region, elevacion, radius_km)
        conectividad = _calcular_conectividad_hidraulica_punto(lat, lon, radius_km)
        hidrometeorologica = _condicion_hidrometeorologica_punto(lat, lon, radius_km)
        cuerpo_agua = hidro_vectorial.buscar_cuerpo_mas_cercano(lat, lon, radio_km=radius_km)
        influencia_agua = _influencia_cuerpo_agua_punto(cuerpo_agua)

        # Módulo 4 aún no implementado: redistribuir su peso proporcionalmente.
        peso_disponible = (IIPDI_PESOS['geomorfologia'] + IIPDI_PESOS['conectividad_hidraulica'] +
                            IIPDI_PESOS['condicion_hidrometeorologica'] + IIPDI_PESOS['cuerpo_agua'])
        factor_redistribucion = 1.0 / peso_disponible

        contribuciones = {
            'Susceptibilidad geomorfológica': IIPDI_PESOS['geomorfologia'] * factor_redistribucion * geomorfologia['valor'],
            'Conectividad hidráulica': IIPDI_PESOS['conectividad_hidraulica'] * factor_redistribucion * conectividad['indice_conectividad'],
            'Condición hidrometeorológica': IIPDI_PESOS['condicion_hidrometeorologica'] * factor_redistribucion * hidrometeorologica['valor'],
            'Características del cuerpo de agua': IIPDI_PESOS['cuerpo_agua'] * factor_redistribucion * influencia_agua['valor'],
        }
        iipdi = round(sum(contribuciones.values()), 1)

        # Explicabilidad: motor de reglas simple (no IA generativa), ordena
        # los módulos por su aporte real al índice final.
        factores_ordenados = sorted(contribuciones.items(), key=lambda kv: kv[1], reverse=True)
        variable_dominante = factores_ordenados[0][0] if factores_ordenados else None
        segundo_factor = factores_ordenados[1][0] if len(factores_ordenados) > 1 else None

        return jsonify({
            'iipdi': iipdi,
            'nivel_riesgo': _nivel_iipdi(iipdi),
            'variable_dominante': variable_dominante,
            'segundo_factor': segundo_factor,
            'cuerpo_agua_responsable': cuerpo_agua['nombre'] if cuerpo_agua else None,
            'modulos': {
                'geomorfologia': geomorfologia,
                'conectividad_hidraulica': conectividad,
                'condicion_hidrometeorologica': hidrometeorologica,
                'cuerpo_agua': influencia_agua,
            },
            'nota': ('Módulo 4 (expansión observada) y modo AHP todavía no '
                     'implementados; su peso fue redistribuido proporcionalmente.'),
        })

    except Exception as exc:  # noqa: BLE001
        return jsonify({'error': str(exc)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)