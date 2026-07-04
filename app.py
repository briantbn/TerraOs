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
import datetime
import ee
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # permite que TerraOS_v4b.html (servido desde otro dominio) consuma la API

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
# ENDPOINTS
# ──────────────────────────────────────────────────────────────────────────

@app.route('/health')
def health():
    return jsonify({'ok': True, 'earth_engine': _ee_ready, 'error': _ee_error})


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
        tipo   riesgo (default) | critico | elevacion | pendiente |
               zona_baja | zona_media | zona_alta
        lat, lon   centro del área (default: Corrientes, Argentina)
        radius_km  radio del área (default: 200 km)
    """
    if not _ee_ready:
        return jsonify({'error': f'Earth Engine no inicializado: {_ee_error}'}), 503
    try:
        tipo      = request.args.get('tipo', 'riesgo')
        lat       = float(request.args.get('lat',  -27.48))   # centro Corrientes
        lon       = float(request.args.get('lon',  -58.83))
        radius_km = float(request.args.get('radius_km', 200))

        region    = ee.Geometry.Point([lon, lat]).buffer(radius_km * 1000)

        # ── DEM y capas topográficas ──────────────────────────────────
        dem       = ee.Image('USGS/SRTMGL1_003').clip(region)
        elevacion = dem.select('elevation')
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
            imagen = elevacion.lt(55).selfMask()
            vis    = {'palette': ['#bcd2ee']}

        elif tipo == 'zona_media':
            imagen = elevacion.gte(55).And(elevacion.lt(90)).selfMask()
            vis    = {'palette': ['#e3ffc2']}

        elif tipo == 'zona_alta':
            imagen = elevacion.gte(90).selfMask()
            vis    = {'palette': ['#e6c280']}

        else:
            return jsonify({'error': f"tipo '{tipo}' no reconocido"}), 400

        map_id = imagen.getMapId(vis)
        return jsonify({
            'tile_url': map_id['tile_fetcher'].url_format,
            'tipo'    : tipo,
            'vis'     : vis,
        })

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
        region    = ee.Geometry.Point([lon, lat]).buffer(200_000)  # 200 km como AOI
        dem       = ee.Image('USGS/SRTMGL1_003').clip(region)
        elevacion = dem.select('elevation')
        pendiente = ee.Terrain.slope(elevacion)

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
                   .reduceRegion(
                       reducer=ee.Reducer.first(),
                       geometry=punto,
                       scale=30,
                       bestEffort=True
                   ))

        result = vals.getInfo()
        elev   = result.get('elevation')
        pend   = result.get('slope')
        ries   = result.get('riesgo')

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
            'ok'       : True,
            'lat'      : lat,
            'lon'      : lon,
            'elevacion': round(elev, 1) if elev is not None else None,
            'pendiente': round(pend, 2) if pend is not None else None,
            'riesgo'   : pct,
            'nivel'    : nivel,
            'topo'     : topo,
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

        region  = ee.Geometry.Point([lon, lat]).buffer(radius_km * 1000)
        dem     = ee.Image('USGS/SRTMGL1_003').clip(region)
        terreno = ee.Terrain.products(dem)                      # elevation, slope, aspect, hillshade
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


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
