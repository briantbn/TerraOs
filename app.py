"""
TerraOS - Backend híbrido: Copernicus Sentinel Hub + Google Earth Engine
=========================================================================

Capas Sentinel (NDVI, EVI, NBR, Falso Color)
  → Copernicus Data Space / Sentinel Hub Process API
  → Devuelve una URL de tile XYZ armada con evalscript.
  → Sin cómputo en el servidor: el cliente llama a Sentinel Hub
    directamente con un token OAuth2 de corta vida (~1 hora).
  → Ventaja clave: tiles globales y continuos, sin parches ni bordes.

Capas GEE (Térmico VIIRS, Riesgo de incendio CHIRPS, Zonas inundables DEM)
  → Google Earth Engine (sin cambios respecto a la versión anterior).

Variables de entorno necesarias:
    GEE_SERVICE_ACCOUNT_JSON   → JSON de la cuenta de servicio GEE
    GEE_PROJECT_ID             → ID del proyecto Google Cloud
    SH_CLIENT_ID               → OAuth2 Client ID de Sentinel Hub
    SH_CLIENT_SECRET           → OAuth2 Client Secret de Sentinel Hub

Cómo obtener SH_CLIENT_ID y SH_CLIENT_SECRET:
    1. Ir a https://shapps.dataspace.copernicus.eu/
    2. User Settings → OAuth Clients → Create new
    3. Copiar Client ID y Client Secret
"""

import os
import json
import datetime
import time
import requests
import ee
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ══════════════════════════════════════════════════════════════════════════════
# SENTINEL HUB — autenticación OAuth2 con caché de token
# ══════════════════════════════════════════════════════════════════════════════
_sh_token        = None
_sh_token_expiry = 0   # timestamp UNIX

SH_TOKEN_URL = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
SH_PROCESS_URL = 'https://sh.dataspace.copernicus.eu/api/v1/process'
SH_WMTS_BASE   = 'https://sh.dataspace.copernicus.eu/ogc/wmts/{instance_id}'


def _sh_get_token() -> str:
    """Obtiene (o reutiliza) el token OAuth2 de Sentinel Hub."""
    global _sh_token, _sh_token_expiry
    if _sh_token and time.time() < _sh_token_expiry - 60:
        return _sh_token

    client_id     = os.environ.get('SH_CLIENT_ID', '')
    client_secret = os.environ.get('SH_CLIENT_SECRET', '')

    if not client_id or not client_secret:
        raise RuntimeError(
            'Faltan variables de entorno SH_CLIENT_ID y/o SH_CLIENT_SECRET. '
            'Creá un OAuth Client en https://shapps.dataspace.copernicus.eu/'
        )

    resp = requests.post(SH_TOKEN_URL, data={
        'grant_type'   : 'client_credentials',
        'client_id'    : client_id,
        'client_secret': client_secret,
    }, timeout=15)

    if not resp.ok:
        raise RuntimeError(f'Error autenticando con Sentinel Hub: {resp.status_code} {resp.text[:200]}')

    data = resp.json()
    _sh_token        = data['access_token']
    _sh_token_expiry = time.time() + data.get('expires_in', 3600)
    print(f'✅ Token Sentinel Hub renovado (expira en {data.get("expires_in", 3600)}s)')
    return _sh_token


# ── Evalscripts por banda ────────────────────────────────────────────────────
# Cada evalscript devuelve una imagen RGBA de 1 banda visual ya coloreada.
# Paleta NDVI/EVI: marrón → amarillo → verde oscuro (igual que GEE anterior)
# Paleta NBR:      rojo oscuro → amarillo → verde oscuro

_EVALSCRIPT_NDVI = """
//VERSION=3
function setup() {
  return { input: ["B04","B08","SCL","dataMask"], output: { bands: 4 } };
}
const ramp = [
  [-0.2, 0xa52a2a], [-0.05, 0xd2b48c], [0.15, 0xffff66],
  [0.35, 0x9acd32], [0.6, 0x228b22],   [0.9,  0x006400]
];
function lerp(a, b, t) { return a + (b - a) * t; }
function colorRamp(val) {
  for (let i = 1; i < ramp.length; i++) {
    if (val <= ramp[i][0]) {
      const t = (val - ramp[i-1][0]) / (ramp[i][0] - ramp[i-1][0]);
      const c1 = ramp[i-1][1], c2 = ramp[i][1];
      return [
        lerp((c1>>16)&0xff, (c2>>16)&0xff, t)/255,
        lerp((c1>>8 )&0xff, (c2>>8 )&0xff, t)/255,
        lerp( c1     &0xff,  c2     &0xff, t)/255
      ];
    }
  }
  return [0, 0.39, 0];
}
function evaluatePixel(s) {
  if (s.dataMask === 0) return [0,0,0,0];
  const scl = s.SCL;
  if (scl===3||scl===8||scl===9||scl===10) return [0,0,0,0]; // nubes
  const ndvi = (s.B08 - s.B04) / (s.B08 + s.B04 + 1e-9);
  const [r,g,b] = colorRamp(ndvi);
  return [r, g, b, 0.85];
}
"""

_EVALSCRIPT_EVI = """
//VERSION=3
function setup() {
  return { input: ["B02","B04","B08","SCL","dataMask"], output: { bands: 4 } };
}
const ramp = [
  [-0.2, 0xa52a2a], [-0.05, 0xd2b48c], [0.15, 0xffff66],
  [0.35, 0x9acd32], [0.6, 0x228b22],   [0.9,  0x006400]
];
function lerp(a, b, t) { return a + (b - a) * t; }
function colorRamp(val) {
  for (let i = 1; i < ramp.length; i++) {
    if (val <= ramp[i][0]) {
      const t = (val - ramp[i-1][0]) / (ramp[i][0] - ramp[i-1][0]);
      const c1 = ramp[i-1][1], c2 = ramp[i][1];
      return [
        lerp((c1>>16)&0xff, (c2>>16)&0xff, t)/255,
        lerp((c1>>8 )&0xff, (c2>>8 )&0xff, t)/255,
        lerp( c1     &0xff,  c2     &0xff, t)/255
      ];
    }
  }
  return [0, 0.39, 0];
}
function evaluatePixel(s) {
  if (s.dataMask === 0) return [0,0,0,0];
  const scl = s.SCL;
  if (scl===3||scl===8||scl===9||scl===10) return [0,0,0,0];
  const evi = 2.5 * (s.B08 - s.B04) / (s.B08 + 6*s.B04 - 7.5*s.B02 + 1 + 1e-9);
  const [r,g,b] = colorRamp(evi);
  return [r, g, b, 0.85];
}
"""

_EVALSCRIPT_NBR = """
//VERSION=3
function setup() {
  return { input: ["B08","B12","SCL","dataMask"], output: { bands: 4 } };
}
const ramp = [
  [-0.5, 0x7f0000], [-0.2, 0xff0000], [0.0, 0xffff66],
  [0.3,  0x66cc66], [0.8,  0x003300]
];
function lerp(a, b, t) { return a + (b - a) * t; }
function colorRamp(val) {
  for (let i = 1; i < ramp.length; i++) {
    if (val <= ramp[i][0]) {
      const t = (val - ramp[i-1][0]) / (ramp[i][0] - ramp[i-1][0]);
      const c1 = ramp[i-1][1], c2 = ramp[i][1];
      return [
        lerp((c1>>16)&0xff, (c2>>16)&0xff, t)/255,
        lerp((c1>>8 )&0xff, (c2>>8 )&0xff, t)/255,
        lerp( c1     &0xff,  c2     &0xff, t)/255
      ];
    }
  }
  return [0, 0.4, 0];
}
function evaluatePixel(s) {
  if (s.dataMask === 0) return [0,0,0,0];
  const scl = s.SCL;
  if (scl===3||scl===8||scl===9||scl===10) return [0,0,0,0];
  const nbr = (s.B08 - s.B12) / (s.B08 + s.B12 + 1e-9);
  const [r,g,b] = colorRamp(nbr);
  return [r, g, b, 0.85];
}
"""

_EVALSCRIPT_FALSECOLOR = """
//VERSION=3
function setup() {
  return { input: ["B08","B04","B03","SCL","dataMask"], output: { bands: 4 } };
}
function evaluatePixel(s) {
  if (s.dataMask === 0) return [0,0,0,0];
  const scl = s.SCL;
  if (scl===3||scl===8||scl===9||scl===10) return [0,0,0,0];
  return [s.B08/3500, s.B04/3500, s.B03/3500, 0.9];
}
"""

_EVALSCRIPTS = {
    'NDVI'      : _EVALSCRIPT_NDVI,
    'EVI'       : _EVALSCRIPT_EVI,
    'NBR'       : _EVALSCRIPT_NBR,
    'FalseColor': _EVALSCRIPT_FALSECOLOR,
}


def _sh_tile_url(band: str, date_from: str, date_to: str) -> str:
    """
    Arma la URL de tiles XYZ de Sentinel Hub usando la Process API en modo
    tile (OGC XYZ). El token OAuth2 va en el query-string para que el
    cliente (Leaflet) pueda pedirlo directamente sin proxy.

    URL resultante: https://sh.dataspace.copernicus.eu/ogc/wms/{token}/...
    No — usamos el endpoint de tiles autenticado via header.

    En cambio, devolvemos los parámetros necesarios para que el CLIENTE
    construya la petición autenticada. El cliente usa un proxy liviano
    en nuestro propio servidor (/sh_tile/{z}/{x}/{y}) que agrega el header.
    """
    evalscript = _EVALSCRIPTS.get(band)
    if not evalscript:
        raise ValueError(f'Banda no soportada para Sentinel Hub: {band}')

    token = _sh_get_token()

    # Devolvemos la URL del proxy de tiles que corre en este mismo servidor.
    # El proxy agrega el Authorization header antes de llamar a Sentinel Hub.
    # Así los tiles fluyen: Leaflet → /sh_tile/{z}/{x}/{y}?band=X → SH API
    return f'/sh_tile/{{z}}/{{x}}/{{y}}?band={band}&from={date_from}&to={date_to}'


# ── Proxy de tiles para Sentinel Hub ────────────────────────────────────────

@app.route('/sh_tile/<int:z>/<int:x>/<int:y>')
def sh_tile_proxy(z, x, y):
    """
    Proxy liviano: recibe peticiones de tiles XYZ de Leaflet y las reenvía
    a la Process API de Sentinel Hub con el token OAuth2 en el header.
    Así Leaflet puede cargar tiles de SH sin exponer credenciales al cliente.

    Parámetros query: band, from (YYYY-MM-DD), to (YYYY-MM-DD)
    """
    band      = request.args.get('band', 'NDVI')
    date_from = request.args.get('from', '')
    date_to   = request.args.get('to',   '')

    evalscript = _EVALSCRIPTS.get(band)
    if not evalscript:
        return jsonify({'error': f'Banda desconocida: {band}'}), 400

    # Convertir XYZ → bbox (EPSG:3857 Web Mercator)
    import math
    def tile_to_latlon(xt, yt, zt):
        n = 2 ** zt
        lon_deg = xt / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * yt / n)))
        lat_deg = math.degrees(lat_rad)
        return lat_deg, lon_deg

    lat_n, lon_w = tile_to_latlon(x,   y,   z)
    lat_s, lon_e = tile_to_latlon(x+1, y+1, z)

    payload = {
        'input': {
            'bounds': {
                'bbox': [lon_w, lat_s, lon_e, lat_n],
                'properties': {'crs': 'http://www.opengis.net/def/crs/EPSG/0/4326'}
            },
            'data': [{
                'type': 'sentinel-2-l2a',
                'dataFilter': {
                    'timeRange': {
                        'from': date_from + 'T00:00:00Z' if date_from else '',
                        'to'  : date_to   + 'T23:59:59Z' if date_to   else '',
                    },
                    'mosaickingOrder': 'leastCC',   # priorizar menor nubosidad
                    'maxCloudCoverage': 80,
                },
            }]
        },
        'output': {
            'width' : 256,
            'height': 256,
            'responses': [{'identifier': 'default', 'format': {'type': 'image/png'}}]
        },
        'evalscript': evalscript,
    }

    try:
        token = _sh_get_token()
        resp = requests.post(
            SH_PROCESS_URL,
            json=payload,
            headers={
                'Authorization': f'Bearer {token}',
                'Accept'       : 'image/png',
            },
            timeout=30,
        )
    except requests.exceptions.Timeout:
        # Tile tardó demasiado → transparente (Leaflet lo ignora)
        return _transparent_tile()
    except Exception as exc:
        print(f'[sh_tile] Error: {exc}')
        return _transparent_tile()

    if resp.status_code == 200:
        from flask import Response
        return Response(resp.content, mimetype='image/png')

    if resp.status_code in (204, 404):
        # Sin datos para este tile → transparente
        return _transparent_tile()

    # Token expirado → forzar renovación y reintentar una vez
    if resp.status_code == 401:
        global _sh_token, _sh_token_expiry
        _sh_token = None
        _sh_token_expiry = 0
        try:
            token = _sh_get_token()
            resp2 = requests.post(
                SH_PROCESS_URL, json=payload,
                headers={'Authorization': f'Bearer {token}', 'Accept': 'image/png'},
                timeout=30,
            )
            if resp2.status_code == 200:
                from flask import Response
                return Response(resp2.content, mimetype='image/png')
        except Exception:
            pass

    return _transparent_tile()


def _transparent_tile():
    """PNG 1×1 transparente — Leaflet lo renderiza como nada."""
    import base64
    from flask import Response
    data = base64.b64decode(
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=='
    )
    return Response(data, mimetype='image/png')


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT PRINCIPAL: /gee_tiles (ahora híbrido)
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/sentinel_tiles')
def sentinel_tiles():
    """
    Endpoint para capas Sentinel-2 via Copernicus Sentinel Hub.
    Devuelve la URL del proxy de tiles /sh_tile/{z}/{x}/{y}?...
    que Leaflet puede usar directamente como tileLayer.

    Parámetros:
        band    NDVI | EVI | NBR | FalseColor
        start   YYYY-MM-DD
        end     YYYY-MM-DD
        cloud   0-100 (máx. nubosidad, default 80 — SH lo maneja internamente)
    """
    band      = request.args.get('band',  'NDVI')
    start     = request.args.get('start', '')
    end       = request.args.get('end',   '')

    if band not in _EVALSCRIPTS:
        return jsonify({'error': f'Banda no soportada: {band}. Usar: NDVI, EVI, NBR, FalseColor'}), 400

    if not start or not end:
        # Default: últimos 90 días
        hoy    = datetime.date.today()
        end    = hoy.isoformat()
        start  = (hoy - datetime.timedelta(days=90)).isoformat()

    # Verificar que las credenciales de SH están configuradas
    try:
        _sh_get_token()
    except RuntimeError as exc:
        return jsonify({'error': str(exc)}), 503

    # La URL usa la ruta del proxy en este mismo servidor.
    # {z}/{x}/{y} son los placeholders que Leaflet reemplaza.
    base = request.host_url.rstrip('/')
    tile_url = f'{base}/sh_tile/{{z}}/{{x}}/{{y}}?band={band}&from={start}&to={end}'

    return jsonify({
        'tile_url': tile_url,
        'source'  : 'Copernicus Sentinel Hub',
        'dataset' : 'Sentinel-2 L2A',
        'band'    : band,
        'from'    : start,
        'to'      : end,
        'nota'    : 'mosaickingOrder=leastCC: píxeles con menor nubosidad tienen prioridad',
    })


# ══════════════════════════════════════════════════════════════════════════════
# GOOGLE EARTH ENGINE — inicialización y endpoints sin cambios
# ══════════════════════════════════════════════════════════════════════════════
_ee_ready = False
_ee_error = None


def init_earth_engine():
    global _ee_ready, _ee_error
    try:
        raw        = os.environ.get('GEE_SERVICE_ACCOUNT_JSON')
        project_id = os.environ.get('GEE_PROJECT_ID')
        if not raw:
            raise RuntimeError('Falta GEE_SERVICE_ACCOUNT_JSON')
        info = json.loads(raw)
        creds = ee.ServiceAccountCredentials(info['client_email'], key_data=raw)
        ee.Initialize(creds, project=project_id or info.get('project_id'))
        _ee_ready = True
        print('✅ Earth Engine inicializado.')
    except Exception as exc:
        _ee_ready = False
        _ee_error = str(exc)
        print(f'⚠️ Earth Engine no disponible: {exc}')


init_earth_engine()

PALETA_TERMICO = ['#313695', '#74add1', '#fee090', '#f46d43', '#a50026']
PALETA_RIESGO  = ['#16a34a', '#facc15', '#ea580c', '#dc2626']


# ── Health ───────────────────────────────────────────────────────────────────

@app.route('/health')
def health():
    sh_ok = False
    try:
        _sh_get_token()
        sh_ok = True
    except Exception:
        pass
    return jsonify({
        'ok'           : True,
        'earth_engine' : _ee_ready,
        'sentinel_hub' : sh_ok,
        'gee_error'    : _ee_error,
    })


# ── /gee_tiles (solo Térmico ahora; Sentinel→ /sentinel_tiles) ──────────────

@app.route('/gee_tiles')
def gee_tiles():
    """
    Ahora solo maneja dataset=VIIRS|MODIS para la banda Thermal.
    Para Sentinel-2 (NDVI/EVI/NBR/FalseColor) usar /sentinel_tiles.
    Se mantiene compatibilidad: si llega dataset=Sentinel, redirige internamente.
    """
    if not _ee_ready:
        return jsonify({'error': f'Earth Engine no inicializado: {_ee_error}'}), 503

    dataset = request.args.get('dataset', 'Sentinel')
    band    = request.args.get('band',    'NDVI')
    start   = request.args.get('start',   '')
    end     = request.args.get('end',     '')

    # Redirigir Sentinel → Copernicus automáticamente (retrocompatibilidad)
    if dataset in ('Sentinel', 'Sentinel-2') and band in _EVALSCRIPTS:
        return sentinel_tiles()

    if not _ee_ready:
        return jsonify({'error': f'Earth Engine no inicializado: {_ee_error}'}), 503

    try:
        lat       = float(request.args.get('lat', -27.48))
        lon       = float(request.args.get('lon', -58.83))
        radius_km = float(request.args.get('radius_km', 150))
        region    = ee.Geometry.Point([lon, lat]).buffer(radius_km * 1000)

        if dataset == 'VIIRS':
            col_lst = (ee.ImageCollection('NOAA/VIIRS/001/VNP21A1D')
                       .filterDate(start, end)
                       .filterBounds(region))
            img = col_lst.median().select('LST_1KM').multiply(0.02).subtract(273.15)
            vis = {'min': 10, 'max': 55, 'palette': PALETA_TERMICO}

        elif dataset == 'MODIS':
            col_lst = (ee.ImageCollection('MODIS/061/MOD11A1')
                       .filterDate(start, end)
                       .filterBounds(region))
            img = col_lst.median().select('LST_Day_1km').multiply(0.02).subtract(273.15)
            vis = {'min': 10, 'max': 55, 'palette': PALETA_TERMICO}

        else:
            return jsonify({'error': f'Dataset no soportado: {dataset}'}), 400

        map_id = img.getMapId(vis)
        return jsonify({
            'tile_url': map_id['tile_fetcher'].url_format,
            'dataset' : dataset,
            'band'    : band,
        })

    except ee.EEException as exc:
        return jsonify({'error': f'Error GEE: {exc}'}), 502
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


# ── /vegetacion ──────────────────────────────────────────────────────────────

@app.route('/vegetacion')
def vegetacion():
    """NDVI/EVI promedio en una zona vía Sentinel Hub Process API (punto)."""
    try:
        lat       = float(request.args.get('lat'))
        lon       = float(request.args.get('lon'))
        dias      = int(request.args.get('dias', 30))
        radius_km = float(request.args.get('radius_km', 15))
    except (TypeError, ValueError):
        return jsonify({'error': 'Parámetros lat, lon requeridos'}), 400

    hoy   = datetime.date.today()
    desde = hoy - datetime.timedelta(days=max(dias, 60))

    # Usar Sentinel Hub Statistics API para obtener el promedio de NDVI
    # en el polígono circular de radius_km
    import math
    def circle_coords(lat0, lon0, r_km, n=32):
        pts = []
        for i in range(n):
            ang = 2 * math.pi * i / n
            dlat = (r_km / 111.0) * math.cos(ang)
            dlon = (r_km / (111.0 * math.cos(math.radians(lat0)))) * math.sin(ang)
            pts.append([lon0 + dlon, lat0 + dlat])
        pts.append(pts[0])
        return pts

    payload = {
        'input': {
            'bounds': {
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [circle_coords(lat, lon, radius_km)]
                },
                'properties': {'crs': 'http://www.opengis.net/def/crs/EPSG/0/4326'}
            },
            'data': [{
                'type': 'sentinel-2-l2a',
                'dataFilter': {
                    'timeRange': {
                        'from': desde.isoformat() + 'T00:00:00Z',
                        'to'  : hoy.isoformat()   + 'T23:59:59Z',
                    },
                    'maxCloudCoverage': 40,
                }
            }]
        },
        'aggregation': {
            'timeRange': {
                'from': desde.isoformat() + 'T00:00:00Z',
                'to'  : hoy.isoformat()   + 'T23:59:59Z',
            },
            'aggregationInterval': {'of': 'P1D'},
            'width': 512, 'height': 512,
            'evalscript': """
//VERSION=3
function setup() {
  return {
    input: [{bands:["B04","B08","B02","SCL","dataMask"]}],
    output: [
      {id:"ndvi", bands:1, sampleType:"FLOAT32"},
      {id:"evi",  bands:1, sampleType:"FLOAT32"},
      {id:"dataMask", bands:1}
    ]
  };
}
function evaluatePixel(s) {
  const scl = s.SCL;
  const ok = s.dataMask===1 && scl!==3 && scl!==8 && scl!==9 && scl!==10;
  if (!ok) return {ndvi:[NaN], evi:[NaN], dataMask:[0]};
  const ndvi = (s.B08-s.B04)/(s.B08+s.B04+1e-9);
  const evi  = 2.5*(s.B08-s.B04)/(s.B08+6*s.B04-7.5*s.B02+1+1e-9);
  return {ndvi:[ndvi], evi:[evi], dataMask:[1]};
}
"""
        },
        'calculations': {
            'default': {
                'statistics': {
                    'ndvi': {'path': 'ndvi', 'statistics': ['mean']},
                    'evi' : {'path': 'evi',  'statistics': ['mean']},
                }
            }
        }
    }

    try:
        token = _sh_get_token()
        resp  = requests.post(
            'https://sh.dataspace.copernicus.eu/api/v1/statistics',
            json=payload,
            headers={'Authorization': f'Bearer {token}'},
            timeout=30,
        )
        if not resp.ok:
            raise RuntimeError(f'SH Statistics API error {resp.status_code}: {resp.text[:200]}')

        data = resp.json()
        intervals = data.get('data', [])
        ndvi_vals = [i['outputs']['default']['bands']['ndvi']['stats']['mean']
                     for i in intervals
                     if i.get('outputs', {}).get('default', {}).get('bands', {}).get('ndvi', {}).get('stats', {}).get('mean') is not None]
        evi_vals  = [i['outputs']['default']['bands']['evi']['stats']['mean']
                     for i in intervals
                     if i.get('outputs', {}).get('default', {}).get('bands', {}).get('evi',  {}).get('stats', {}).get('mean') is not None]

        ndvi_mean = round(sum(ndvi_vals)/len(ndvi_vals), 3) if ndvi_vals else None
        evi_mean  = round(sum(evi_vals) /len(evi_vals),  3) if evi_vals  else None

        return jsonify({'ndvi_promedio': ndvi_mean, 'evi_promedio': evi_mean})

    except Exception as exc:
        # Fallback a GEE si SH falla
        if _ee_ready:
            return _vegetacion_gee(lat, lon, dias, radius_km)
        return jsonify({'error': str(exc)}), 500


def _vegetacion_gee(lat, lon, dias, radius_km):
    """Fallback: calcula NDVI/EVI promedio con GEE."""
    try:
        region = ee.Geometry.Point([lon, lat]).buffer(radius_km * 1000)
        hoy    = ee.Date(datetime.date.today().isoformat())
        desde  = hoy.advance(-max(dias, 60), 'day')
        col    = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterDate(desde, hoy)
                  .filterBounds(region)
                  .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 40)))
        if col.size().getInfo() == 0:
            return jsonify({'ndvi_promedio': None, 'evi_promedio': None})
        img  = col.median()
        ndvi = img.normalizedDifference(['B8', 'B4']).rename('ndvi')
        refl = img.select(['B2', 'B4', 'B8']).multiply(0.0001)
        evi  = refl.expression(
            '2.5*(NIR-RED)/(NIR+6*RED-7.5*BLUE+1)',
            {'NIR': refl.select('B8'), 'RED': refl.select('B4'), 'BLUE': refl.select('B2')}
        ).rename('evi')
        stats = (ee.Image.cat([ndvi, evi])
                 .reduceRegion(reducer=ee.Reducer.mean(), geometry=region, scale=100, bestEffort=True))
        nv = stats.get('ndvi').getInfo()
        ev = stats.get('evi').getInfo()
        return jsonify({
            'ndvi_promedio': round(nv, 3) if nv is not None else None,
            'evi_promedio' : round(ev, 3) if ev is not None else None,
        })
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


# ── /lluvia (sin cambios — CHIRPS solo está en GEE) ─────────────────────────

@app.route('/lluvia')
def lluvia():
    if not _ee_ready:
        return jsonify({'error': f'Earth Engine no inicializado: {_ee_error}'}), 503
    try:
        lat       = float(request.args.get('lat'))
        lon       = float(request.args.get('lon'))
        dias      = int(request.args.get('dias', 30))
        radius_km = float(request.args.get('radius_km', 15))
        region    = ee.Geometry.Point([lon, lat]).buffer(radius_km * 1000)
        hoy       = ee.Date(datetime.date.today().isoformat())
        desde     = hoy.advance(-dias, 'day')
        chirps    = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
                     .filterDate(desde, hoy).filterBounds(region).sum())
        valor     = chirps.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=region, scale=5000, bestEffort=True
        ).get('precipitation').getInfo()
        return jsonify({'lluvia_acumulada_mm': round(valor, 1) if valor is not None else None})
    except ee.EEException as exc:
        return jsonify({'error': f'Error GEE: {exc}'}), 502
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


# ── /riesgo_incendio_tiles (sin cambios — depende de CHIRPS) ─────────────────

@app.route('/riesgo_incendio_tiles')
def riesgo_incendio_tiles():
    if not _ee_ready:
        return jsonify({'error': f'Earth Engine no inicializado: {_ee_error}'}), 503
    try:
        lat       = float(request.args.get('lat'))
        lon       = float(request.args.get('lon'))
        radius_km = float(request.args.get('radius_km', 150))
        region    = ee.Geometry.Point([lon, lat]).buffer(radius_km * 1000)
        hoy       = ee.Date(datetime.date.today().isoformat())
        hace_90   = hoy.advance(-90, 'day')

        # Vegetación seca: NDVI promedio últimos 90 días
        s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filterDate(hace_90, hoy)
              .filterBounds(region)
              .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 40)))
        ndvi     = s2.median().normalizedDifference(['B8', 'B4'])
        sequedad = ndvi.multiply(-1).add(0.9).clamp(0, 1)

        # Déficit de lluvia CHIRPS
        chirps       = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
        lluvia_actual = chirps.filterDate(hace_90, hoy).filterBounds(region).sum()
        anios_hist   = ee.List.sequence(1, 5)
        def lluvia_anio(n):
            ini = hace_90.advance(ee.Number(n).multiply(-1), 'year')
            fin = hoy.advance(ee.Number(n).multiply(-1), 'year')
            return chirps.filterDate(ini, fin).filterBounds(region).sum()
        lluvia_hist = ee.ImageCollection(anios_hist.map(lluvia_anio)).mean()
        deficit     = (lluvia_hist.subtract(lluvia_actual)
                       .divide(lluvia_hist.add(1)).clamp(0, 1))

        riesgo = sequedad.multiply(0.6).add(deficit.multiply(0.4)).rename('riesgo')
        vis    = {'min': 0, 'max': 1, 'palette': PALETA_RIESGO}
        map_id = riesgo.getMapId(vis)
        return jsonify({'tile_url': map_id['tile_fetcher'].url_format, 'vis': vis})

    except ee.EEException as exc:
        return jsonify({'error': f'Error GEE: {exc}'}), 502
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


# ── /zonas_inundables y /inundacion_punto (sin cambios — dependen de SRTM) ──

@app.route('/inundacion_tiles')
@app.route('/zonas_inundables')
def inundacion_tiles():
    if not _ee_ready:
        return jsonify({'error': f'Earth Engine no inicializado: {_ee_error}'}), 503
    try:
        tipo      = request.args.get('tipo', 'riesgo')
        lat       = float(request.args.get('lat',  -27.48))
        lon       = float(request.args.get('lon',  -58.83))
        radius_km = float(request.args.get('radius_km', 200))
        region    = ee.Geometry.Point([lon, lat]).buffer(radius_km * 1000)

        dem       = ee.Image('USGS/SRTMGL1_003').clip(region)
        elevacion = dem.select('elevation')
        pendiente = ee.Terrain.slope(elevacion)
        smap      = (ee.ImageCollection('NASA_USDA/HSL/SMAP10KM_soil_moisture')
                     .filterBounds(region).filterDate('2023-10-01', '2024-03-31')
                     .select('ssm').mean().clip(region))

        def normalizar(img, mn, mx):
            safe = ee.Image(ee.Algorithms.If(img.bandNames().size().gt(0), img, ee.Image.constant(0)))
            return safe.subtract(mn).divide(mx - mn).clamp(0, 1)

        riesgo_compuesto = (normalizar(elevacion, 30, 140).subtract(1).multiply(-1).multiply(0.40)
                            .add(normalizar(pendiente, 0, 3).subtract(1).multiply(-1).multiply(0.30))
                            .add(normalizar(smap, 10, 40).multiply(0.30))
                            .rename('riesgo'))

        mapa = {
            'riesgo'    : (riesgo_compuesto,
                           {'min':0.2,'max':0.8,'palette':['#2c7bb6','#abd9e9','#ffffbf','#fdae61','#d7191c']}),
            'critico'   : (riesgo_compuesto.gt(0.65).selfMask(), {'palette':['#ff0000']}),
            'elevacion' : (elevacion, {'min':30,'max':140,
                           'palette':['#1a3a5c','#2e6da4','#67a9cf','#d1e5f0','#f7f7f7','#fddbc7','#d6604d']}),
            'pendiente' : (pendiente, {'min':0,'max':3,'palette':['#ffffff','#fee090','#fc8d59','#d73027']}),
            'zona_baja' : (elevacion.lt(55).selfMask(), {'palette':['#bcd2ee']}),
            'zona_media': (elevacion.gte(55).And(elevacion.lt(90)).selfMask(), {'palette':['#e3ffc2']}),
            'zona_alta' : (elevacion.gte(90).selfMask(), {'palette':['#e6c280']}),
        }
        if tipo not in mapa:
            return jsonify({'error': f"tipo '{tipo}' no reconocido"}), 400
        imagen, vis = mapa[tipo]
        map_id = imagen.getMapId(vis)
        return jsonify({'tile_url': map_id['tile_fetcher'].url_format, 'tipo': tipo, 'vis': vis})

    except ee.EEException as exc:
        return jsonify({'error': f'Error GEE: {exc}'}), 502
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/inundacion_punto')
def inundacion_punto():
    if not _ee_ready:
        return jsonify({'error': f'Earth Engine no inicializado: {_ee_error}'}), 503
    try:
        lat = float(request.args['lat'])
        lon = float(request.args['lon'])
    except (KeyError, ValueError):
        return jsonify({'error': 'lat y lon requeridos'}), 400
    try:
        region    = ee.Geometry.Point([lon, lat]).buffer(200_000)
        dem       = ee.Image('USGS/SRTMGL1_003').clip(region)
        elevacion = dem.select('elevation')
        pendiente = ee.Terrain.slope(elevacion)
        smap      = (ee.ImageCollection('NASA_USDA/HSL/SMAP10KM_soil_moisture')
                     .filterBounds(region).filterDate('2023-10-01','2024-03-31')
                     .select('ssm').mean().clip(region))
        def normalizar(img, mn, mx):
            safe = ee.Image(ee.Algorithms.If(img.bandNames().size().gt(0), img, ee.Image.constant(0)))
            return safe.subtract(mn).divide(mx - mn).clamp(0, 1)
        riesgo = (normalizar(elevacion,30,140).subtract(1).multiply(-1).multiply(0.40)
                  .add(normalizar(pendiente,0,3).subtract(1).multiply(-1).multiply(0.30))
                  .add(normalizar(smap,10,40).multiply(0.30)).rename('riesgo'))
        vals   = (riesgo.addBands(elevacion).addBands(pendiente)
                  .reduceRegion(reducer=ee.Reducer.first(),
                                geometry=ee.Geometry.Point([lon,lat]),
                                scale=30, bestEffort=True))
        r      = vals.getInfo()
        elev   = r.get('elevation')
        pend   = r.get('slope')
        ries   = r.get('riesgo')
        topo   = ('Zona Baja (<55 m)' if elev and elev < 55
                  else 'Zona Media (55-90 m)' if elev and elev < 90
                  else 'Zona Alta (>90 m)' if elev else 'Sin datos')
        pct    = round(ries * 100, 1) if ries is not None else None
        nivel  = ('🔴 ALTO' if pct and pct > 65
                  else '🟡 MEDIO' if pct and pct > 40
                  else '🟢 BAJO' if pct else 'Sin datos')
        return jsonify({'ok':True,'lat':lat,'lon':lon,
                        'elevacion':round(elev,1) if elev else None,
                        'pendiente':round(pend,2) if pend else None,
                        'riesgo':pct,'nivel':nivel,'topo':topo})
    except ee.EEException as exc:
        return jsonify({'error': f'Error GEE: {exc}'}), 502
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
