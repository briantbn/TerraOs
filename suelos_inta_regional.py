"""
suelos_inta_regional.py
==========================================
Aptitud productiva del suelo, priorizando siempre la mejor escala
disponible antes de caer a una carta más gruesa. Usado por dos endpoints
de app.py:

  - /suelo_mejor_resolucion  -> consultar_mejor_capa(lat, lon)
        Consulta puntual (Inspector de Suelos, Motor de Aptitud Productiva).
  - /suelo_capa_aptitud      -> capa_visual_bbox((west, south, east, north))
        Overlay visual coloreado sobre el área visible del mapa.

Fuente de datos: cartas de INTA en formato GeoJSON, autoalojadas en
Hugging Face (dataset Briant97/GeoSentinel). Se leen una sola vez por
proceso y quedan cacheadas en memoria con un índice espacial (STRtree de
Shapely) para que las consultas siguientes sean instantáneas.

Orden de prioridad (mejor escala primero) — ver SUELO_CAPAS más abajo:
  1) Santa Fe    — carta de detalle 1:50.000
  2) Córdoba     — carta de detalle 1:50.000
  3) Buenos Aires— carta de detalle 1:50.000
  4) NOA         — aptitud regional (zonas grandes, informativo)
  5) Nacional, partido por provincia — 1:1.000.000 (último recurso, 23
     archivos -- ver SUELO_CAPAS_NACIONAL). Antes era UN solo archivo de
     ~80MB que cubría todo el país; partirlo por provincia hace que cada
     consulta solo baje y cargue el tile de la provincia correspondiente
     (unos cientos de KB) en vez del país entero.

Para agregar más provincias/escalas de detalle a futuro (1:100.000,
1:250.000, 1:500.000, etc.): alcanza con sumar una entrada a SUELO_CAPAS
en el lugar que le corresponda según su escala, antes de los tiles
nacionales. No hace falta tocar app.py ni el resto de este archivo.
"""
import gc
import math
import threading
import time
from collections import OrderedDict

import requests
from shapely.geometry import shape, Point, box
from shapely.strtree import STRtree
from shapely.ops import unary_union

# ──────────────────────────────────────────────────────────────────────────
# CATÁLOGO DE CAPAS (mejor escala primero)
# ──────────────────────────────────────────────────────────────────────────
_HF_BASE = 'https://huggingface.co/datasets/Briant97/GeoSentinel/resolve/main/'

SUELO_CAPAS = [
    # EEA INTA Concordia (Entre Ríos) 1:5.000 -- la MEJOR escala de todo
    # el catálogo, por eso va primera. Área muy chica (el predio
    # experimental), pero donde cubre, gana a cualquier otra capa.
    # 'alternate' del dataset no traía attribute_set en GeoNode (capa
    # remota) -- los nombres de campo reales quedan sin confirmar hasta
    # una consulta real; _normalizar() ya intenta varios alias genéricos
    # (clase/cap_uso/ind_prod) por las dudas, pero puede no encontrar
    # nada si el esquema es distinto. Confirmar con una consulta de
    # prueba tras el deploy.
    {
        'nombre': 'EEA INTA Concordia (predio)',
        'url': 'https://geo-nodo05.inta.gob.ar/geoserver/wfs?service=WFS&version=2.0.0&request=GetFeature&typeName=geonode:suelos_eea_concordia_ll&outputFormat=application/json',
        'escala': '1:5.000',
        'confiabilidad': 'alta',
        'bbox': (-58.1248, -31.3824, -58.0972, -31.3507),
        'simplificar_grados': None,
    },
    # Santa Fe 1:50.000 -- ANTES era un solo archivo de 366MB (53.046
    # polígonos), que directamente reventaba la memoria del dyno de Render
    # con solo intentar descargarlo/parsearlo una vez. Partido en 20 tiles
    # balanceados por cantidad de features (grilla adaptativa 4x5, no por
    # grados fijos, para que ningún tile quede desproporcionadamente más
    # pesado que otro), y pre-simplificado a la misma tolerancia que ya
    # se usaba (~30m). Los tiles de borde se superponen levemente a
    # propósito -- normal, consultar_mejor_capa() ya prueba todos los que
    # puedan cubrir el punto en orden y devuelve el primero que matchea.
    {
        'nombre': 'Santa Fe (c0f0)',
        'url': _HF_BASE + 'santa_fe_1_50000_c0f0.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-62.9039, -34.4075, -61.4169, -33.415),
        'simplificar_grados': None,  # ya viene pre-simplificado en el archivo
    },
    {
        'nombre': 'Santa Fe (c0f1)',
        'url': _HF_BASE + 'santa_fe_1_50000_c0f1.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-62.2367, -33.6835, -61.5057, -32.0525),
        'simplificar_grados': None,  # ya viene pre-simplificado en el archivo
    },
    {
        'nombre': 'Santa Fe (c0f2)',
        'url': _HF_BASE + 'santa_fe_1_50000_c0f2.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-62.2608, -32.1925, -61.4152, -30.6462),
        'simplificar_grados': None,  # ya viene pre-simplificado en el archivo
    },
    {
        'nombre': 'Santa Fe (c0f3)',
        'url': _HF_BASE + 'santa_fe_1_50000_c0f3.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-62.185, -30.7884, -61.4702, -29.7864),
        'simplificar_grados': None,  # ya viene pre-simplificado en el archivo
    },
    {
        'nombre': 'Santa Fe (c0f4)',
        'url': _HF_BASE + 'santa_fe_1_50000_c0f4.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-62.0513, -29.9133, -61.4035, -28.0936),
        'simplificar_grados': None,  # ya viene pre-simplificado en el archivo
    },
    {
        'nombre': 'Santa Fe (c1f0)',
        'url': _HF_BASE + 'santa_fe_1_50000_c1f0.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-61.7836, -34.4062, -61.1265, -32.7614),
        'simplificar_grados': None,  # ya viene pre-simplificado en el archivo
    },
    {
        'nombre': 'Santa Fe (c1f1)',
        'url': _HF_BASE + 'santa_fe_1_50000_c1f1.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-61.8083, -32.9337, -61.0946, -31.3875),
        'simplificar_grados': None,  # ya viene pre-simplificado en el archivo
    },
    {
        'nombre': 'Santa Fe (c1f2)',
        'url': _HF_BASE + 'santa_fe_1_50000_c1f2.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-61.7058, -31.533, -60.9853, -30.5909),
        'simplificar_grados': None,  # ya viene pre-simplificado en el archivo
    },
    {
        'nombre': 'Santa Fe (c1f3)',
        'url': _HF_BASE + 'santa_fe_1_50000_c1f3.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-61.8651, -30.7126, -60.8531, -29.8313),
        'simplificar_grados': None,  # ya viene pre-simplificado en el archivo
    },
    {
        'nombre': 'Santa Fe (c1f4)',
        'url': _HF_BASE + 'santa_fe_1_50000_c1f4.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-61.7191, -29.9812, -61.0397, -28.1198),
        'simplificar_grados': None,  # ya viene pre-simplificado en el archivo
    },
    {
        'nombre': 'Santa Fe (c2f0)',
        'url': _HF_BASE + 'santa_fe_1_50000_c2f0.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-61.3004, -33.9022, -60.4432, -32.6758),
        'simplificar_grados': None,  # ya viene pre-simplificado en el archivo
    },
    {
        'nombre': 'Santa Fe (c2f1)',
        'url': _HF_BASE + 'santa_fe_1_50000_c2f1.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-61.8148, -33.1141, -60.7042, -31.7244),
        'simplificar_grados': None,  # ya viene pre-simplificado en el archivo
    },
    {
        'nombre': 'Santa Fe (c2f2)',
        'url': _HF_BASE + 'santa_fe_1_50000_c2f2.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-61.3995, -31.9464, -60.7117, -31.04),
        'simplificar_grados': None,  # ya viene pre-simplificado en el archivo
    },
    {
        'nombre': 'Santa Fe (c2f3)',
        'url': _HF_BASE + 'santa_fe_1_50000_c2f3.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-61.436, -31.3117, -60.6858, -30.5324),
        'simplificar_grados': None,  # ya viene pre-simplificado en el archivo
    },
    {
        'nombre': 'Santa Fe (c2f4)',
        'url': _HF_BASE + 'santa_fe_1_50000_c2f4.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-61.3723, -30.7565, -60.6646, -28.557),
        'simplificar_grados': None,  # ya viene pre-simplificado en el archivo
    },
    {
        'nombre': 'Santa Fe (c3f0)',
        'url': _HF_BASE + 'santa_fe_1_50000_c3f0.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-61.5782, -33.6688, -60.0819, -31.0679),
        'simplificar_grados': None,  # ya viene pre-simplificado en el archivo
    },
    {
        'nombre': 'Santa Fe (c3f1)',
        'url': _HF_BASE + 'santa_fe_1_50000_c3f1.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-60.8135, -31.5066, -59.9307, -30.5735),
        'simplificar_grados': None,  # ya viene pre-simplificado en el archivo
    },
    {
        'nombre': 'Santa Fe (c3f2)',
        'url': _HF_BASE + 'santa_fe_1_50000_c3f2.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-60.9954, -31.7101, -59.8823, -30.1275),
        'simplificar_grados': None,  # ya viene pre-simplificado en el archivo
    },
    {
        'nombre': 'Santa Fe (c3f3)',
        'url': _HF_BASE + 'santa_fe_1_50000_c3f3.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-60.8079, -31.3129, -59.7156, -29.4747),
        'simplificar_grados': None,  # ya viene pre-simplificado en el archivo
    },
    {
        'nombre': 'Santa Fe (c3f4)',
        'url': _HF_BASE + 'santa_fe_1_50000_c3f4.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-61.7521, -32.6554, -58.8121, -27.9771),
        'simplificar_grados': None,  # ya viene pre-simplificado en el archivo
    },
    # Córdoba 1:50.000 -- ANTES un solo archivo de 242MB (1.505 polígonos
    # pero MUY complejos: hasta 26 columnas por feature y geometrías con
    # miles de vértices). Igual que Santa Fe: partido en grilla + columnas
    # recortadas a solo las que usa _normalizar_cordoba() (Clase, Subclase,
    # 'Cap Uso', IP) + pre-simplificado.
    {
        'nombre': 'Córdoba (-66_-35)',
        'url': _HF_BASE + 'cordoba_-66_-35.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-65.1732, -35.0657, -64.7648, -33.6897),
        'simplificar_grados': None,  # ya viene pre-simplificado en el archivo
    },
    {
        'nombre': 'Córdoba (-66_-34)',
        'url': _HF_BASE + 'cordoba_-66_-34.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-65.1929, -33.8106, -64.7516, -32.95),
        'simplificar_grados': None,  # ya viene pre-simplificado en el archivo
    },
    {
        'nombre': 'Córdoba (-65_-35)',
        'url': _HF_BASE + 'cordoba_-65_-35.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-65.1689, -35.0647, -63.4685, -33.7224),
        'simplificar_grados': None,  # ya viene pre-simplificado en el archivo
    },
    {
        'nombre': 'Córdoba (-65_-34)',
        'url': _HF_BASE + 'cordoba_-65_-34.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-65.1925, -35.0425, -62.7725, -30.6075),
        'simplificar_grados': None,  # ya viene pre-simplificado en el archivo
    },
    {
        'nombre': 'Córdoba (-65_-33)',
        'url': _HF_BASE + 'cordoba_-65_-33.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-65.1858, -33.7199, -62.021, -30.9795),
        'simplificar_grados': None,  # ya viene pre-simplificado en el archivo
    },
    {
        'nombre': 'Córdoba (-65_-32)',
        'url': _HF_BASE + 'cordoba_-65_-32.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-64.5542, -32.3815, -63.1953, -30.6167),
        'simplificar_grados': None,  # ya viene pre-simplificado en el archivo
    },
    {
        'nombre': 'Córdoba (-65_-31)',
        'url': _HF_BASE + 'cordoba_-65_-31.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-64.554, -31.3833, -63.5318, -30.4351),
        'simplificar_grados': None,  # ya viene pre-simplificado en el archivo
    },
    {
        'nombre': 'Córdoba (-64_-35)',
        'url': _HF_BASE + 'cordoba_-64_-35.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-65.1782, -35.0615, -62.7093, -33.2535),
        'simplificar_grados': None,  # ya viene pre-simplificado en el archivo
    },
    {
        'nombre': 'Córdoba (-64_-34)',
        'url': _HF_BASE + 'cordoba_-64_-34.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-65.1814, -35.0616, -61.8668, -30.2894),
        'simplificar_grados': None,  # ya viene pre-simplificado en el archivo
    },
    {
        'nombre': 'Córdoba (-64_-33)',
        'url': _HF_BASE + 'cordoba_-64_-33.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-64.4719, -33.3145, -62.5058, -31.3524),
        'simplificar_grados': None,  # ya viene pre-simplificado en el archivo
    },
    {
        'nombre': 'Córdoba (-64_-32)',
        'url': _HF_BASE + 'cordoba_-64_-32.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-64.4355, -32.853, -62.5806, -30.4783),
        'simplificar_grados': None,  # ya viene pre-simplificado en el archivo
    },
    {
        'nombre': 'Córdoba (-64_-31)',
        'url': _HF_BASE + 'cordoba_-64_-31.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-64.2991, -31.3814, -62.95, -30.2833),
        'simplificar_grados': None,  # ya viene pre-simplificado en el archivo
    },
    {
        'nombre': 'Córdoba (-63_-35)',
        'url': _HF_BASE + 'cordoba_-63_-35.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-63.0867, -34.436, -62.5453, -33.9184),
        'simplificar_grados': None,  # ya viene pre-simplificado en el archivo
    },
    {
        'nombre': 'Córdoba (-63_-34)',
        'url': _HF_BASE + 'cordoba_-63_-34.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-63.5179, -34.2891, -61.7376, -32.6913),
        'simplificar_grados': None,  # ya viene pre-simplificado en el archivo
    },
    {
        'nombre': 'Córdoba (-63_-33)',
        'url': _HF_BASE + 'cordoba_-63_-33.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-63.5135, -33.4786, -61.7274, -31.6031),
        'simplificar_grados': None,  # ya viene pre-simplificado en el archivo
    },
    {
        'nombre': 'Córdoba (-63_-32)',
        'url': _HF_BASE + 'cordoba_-63_-32.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-63.3488, -32.3826, -61.9833, -31.2833),
        'simplificar_grados': None,  # ya viene pre-simplificado en el archivo
    },
    {
        'nombre': 'Córdoba (-62_-34)',
        'url': _HF_BASE + 'cordoba_-62_-34.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-62.1541, -33.2335, -61.7382, -32.874),
        'simplificar_grados': None,  # ya viene pre-simplificado en el archivo
    },
    {
        'nombre': 'Córdoba (-62_-33)',
        'url': _HF_BASE + 'cordoba_-62_-33.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-62.2985, -33.1572, -61.7209, -32.2203),
        'simplificar_grados': None,  # ya viene pre-simplificado en el archivo
    },
    {
        'nombre': 'Buenos Aires',
        # ANTES 33MB con ~35 columnas por feature de las cuales solo se usa
        # CAP_USO -- recortado a esa única columna + pre-simplificado.
        'url': _HF_BASE + 'buenos_aires_1_50000.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-64.1, -41.5, -56.5, -32.5),
        'simplificar_grados': None,  # ya viene pre-simplificado en el archivo
    },
    # ── Corrientes 1:50.000 -- capas remotas de GeoINTA (geo-nodo09),
    # descargadas en vivo vía WFS (no pre-alojadas en Hugging Face como
    # las de arriba). Ninguna traía attribute_set en la metadata de
    # GeoNode (son capas "remote" -- GeoNode no las introspecciona) --
    # los nombres de campo reales quedan sin confirmar hasta una consulta
    # de prueba tras el deploy; si _normalizar() no encuentra clase/
    # cap_uso/indice_prod con sus alias genéricos actuales, avisar para
    # sumar los alias correctos.
    {
        'nombre': 'Corrientes — Monte Caseros / Empedrado',
        'url': 'https://geo-nodo09.inta.gob.ar/geoserver/wfs?service=WFS&version=2.0.0&request=GetFeature&typeName=geonode:suelos_monte_caseros1_50000&outputFormat=application/json',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-58.0831, -30.7210, -57.5591, -29.8189),
        'simplificar_grados': 0.0005,
    },
    {
        'nombre': 'Corrientes — Yaciretá',
        'url': 'https://geo-nodo09.inta.gob.ar/geoserver/wfs?service=WFS&version=2.0.0&request=GetFeature&typeName=geonode:suelos_yacireta&outputFormat=application/json',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-58.8570, -28.0058, -55.8304, -27.2643),
        'simplificar_grados': 0.0005,
    },
    {
        'nombre': 'Corrientes — Sauce',
        'url': 'https://geo-nodo09.inta.gob.ar/geoserver/wfs?service=WFS&version=2.0.0&request=GetFeature&typeName=geonode:suelos_sauce&outputFormat=application/json',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-59.1262, -30.2734, -58.4451, -29.6303),
        'simplificar_grados': 0.0005,
    },
    {
        'nombre': 'Corrientes — Curuzú Cuatiá',
        'url': 'https://geo-nodo09.inta.gob.ar/geoserver/wfs?service=WFS&version=2.0.0&request=GetFeature&typeName=geonode:suelos_curuzu&outputFormat=application/json',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-58.9107, -30.4224, -57.6099, -29.0446),
        'simplificar_grados': 0.0005,
    },
    {
        'nombre': 'Corrientes — Goya',
        'url': 'https://geo-nodo09.inta.gob.ar/geoserver/wfs?service=WFS&version=2.0.0&request=GetFeature&typeName=geonode:Suelos_Goya&outputFormat=application/json',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-59.6652, -30.0475, -58.7880, -29.0503),
        'simplificar_grados': 0.0005,
    },
    # ── Santiago del Estero 1:50.000 (geo-nodo02 / geo-nodo05) ──
    {
        'nombre': 'Santiago del Estero — Norte Belgrano',
        'url': 'https://geo-nodo02.inta.gob.ar/geoserver/wfs?service=WFS&version=2.0.0&request=GetFeature&typeName=geonode:geoformas_uc_belgrano&outputFormat=application/json',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-62.3227, -29.1320, -61.8565, -28.8629),
        'simplificar_grados': 0.0005,
    },
    {
        'nombre': 'Santiago del Estero — Subcuenca La Esperanza',
        'url': 'https://geo-nodo05.inta.gob.ar/geoserver/wfs?service=WFS&version=2.0.0&request=GetFeature&typeName=geonode:mapa_de_suelos_la_esperanza_latlong&outputFormat=application/json',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-61.8193, -27.1056, -61.7129, -26.9745),
        'simplificar_grados': 0.0005,
    },
    # ── Misiones 1:50.000 (geo-nodo13 / geo-nodo05) ──
    {
        'nombre': 'Misiones — Leandro N. Alem',
        'url': 'https://geo-nodo13.inta.gob.ar/geoserver/wfs?service=WFS&version=2.0.0&request=GetFeature&typeName=geonode:suelos_alemp7&outputFormat=application/json',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-55.6623, -27.7846, -55.1617, -27.4529),
        'simplificar_grados': 0.0005,
    },
    {
        'nombre': 'Misiones — Guaraní',
        'url': 'https://geo-nodo05.inta.gob.ar/geoserver/wfs?service=WFS&version=2.0.0&request=GetFeature&typeName=geonode:mapa_guarani_final&outputFormat=application/json',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-54.6136, -27.4480, -53.8761, -26.6417),
        'simplificar_grados': 0.0005,
    },
    # ── Salta 1:50.000 (aprox.) -- subidas por el usuario directo a
    # Los 3 archivos de Valle de Lerma cubren el mismo valle (zona
    # Salta capital / Cerrillos / San Agustín hasta el sur), pero la
    # división Norte/Centro/Sur original por franjas de latitud estaba
    # mal -- el archivo "Centro" en realidad cae en la parte NORTE real
    # del valle (confirmado bajando el contenido real). Bbox unificado
    # y generoso para los 3 hasta poder confirmar la división real.
    # Campos reales confirmados (bajando el archivo Centro): cap_uso
    # (coincide con el alias que ya usa _normalizar), Ipc (índice de
    # productividad numérico -- NO coincide con los alias actuales de
    # indice_prod, hay que sumar 'Ipc' a la lista), Ind_Prod (texto
    # descriptivo "Baja/Regular/Buena Productividad", no numérico).
    {
        'nombre': 'Salta — Valle de Lerma (Norte)',
        'url': _HF_BASE + 'Suelos_detalle_Valle_de_Lerma_Norte_Salta.geojson?download=true',
        'escala': 'detalle (sin escala numérica declarada)',
        'confiabilidad': 'alta',
        'bbox': (-65.60, -25.50, -65.10, -24.50),
        'simplificar_grados': 0.0005,
    },
    {
        'nombre': 'Salta — Valle de Lerma (Centro)',
        'url': _HF_BASE + 'Suelos_detalle_Valle_de_Lerma_Centro_Salta.geojson?download=true',
        'escala': 'detalle (sin escala numérica declarada)',
        'confiabilidad': 'alta',
        'bbox': (-65.60, -25.50, -65.10, -24.50),
        'simplificar_grados': 0.0005,
    },
    {
        'nombre': 'Salta — Valle de Lerma (Sur)',
        'url': _HF_BASE + 'Suelos_detalle_Valle_de_Lerma_Sur_Salta.geojson?download=true',
        'escala': 'detalle (sin escala numérica declarada)',
        'confiabilidad': 'alta',
        'bbox': (-65.60, -25.50, -65.10, -24.50),
        'simplificar_grados': 0.0005,
    },
    {
        'nombre': 'Salta — Valles Calchaquíes',
        'url': _HF_BASE + 'Suelos_detalle_Valles_Calchaquies_Salta.geojson?download=true',
        'escala': 'detalle (sin escala numérica declarada)',
        'confiabilidad': 'alta',
        'bbox': (-66.50, -26.50, -65.30, -24.60),
        'simplificar_grados': 0.0005,
    },
    {
        # Bbox recalculado del contenido REAL del archivo (antes era una
        # estimación por geografía general, que quedaba mal ubicada).
        # Campos reales del geojson: Suelo, aso, Ap_Agricol (A/B/C/D,
        # aptitud agrícola -- mapea a 'clase'), Ap_Riego, Suelo_1..5,
        # CM_Suelo1..5, ajusteCN. No trae cap_uso ni ind_prod numérico
        # -- _normalizar() necesita un alias nuevo para 'Ap_Agricol'.
        'nombre': 'Salta — Miraflores / El Galpón',
        'url': _HF_BASE + 'Suelos_detalle_Miraflores_El_Galpon_Salta.geojson?download=true',
        'escala': 'detalle (sin escala numérica declarada)',
        'confiabilidad': 'alta',
        'bbox': (-64.95, -25.50, -64.55, -25.25),
        'simplificar_grados': 0.0005,
    },
    # ── Formosa 1:50.000 (geo-nodo05) ──
    {
        'nombre': 'Formosa — Pirané Sur',
        'url': 'https://geo-nodo05.inta.gob.ar/geoserver/wfs?service=WFS&version=2.0.0&request=GetFeature&typeName=geonode:mapa_de_suelos_pirane_final_latlong&outputFormat=application/json',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-59.6517, -26.3887, -58.9259, -25.4734),
        'simplificar_grados': 0.0005,
    },
    {
        'nombre': 'NOA (regional)',
        'url': _HF_BASE + 'Suelos_detalle_NOA.geojson?download=true',
        'escala': 'regional (zonas grandes)',
        'confiabilidad': 'media',
        # Noroeste argentino (Jujuy/Salta/Tucumán/Catamarca/Santiago del
        # Estero) -- OJO: esto NO cubre Corrientes (que es NEA, noreste),
        # aunque el nombre "regional" pueda sugerir que sí.
        'bbox': (-70.0, -29.6, -61.5, -20.8),
        # Solo 9 polígonos en total -- liviano de por sí, no hace falta
        # tilearlo. Esquema propio (grupo_tier A-E + descri), ver
        # _normalizar_noa().
        'simplificar_grados': 0.005,
    },
    # ── Corrientes 1:100.000 (geo-nodo09) ──
    {
        'nombre': 'Corrientes — Esquina',
        'url': 'https://geo-nodo09.inta.gob.ar/geoserver/wfs?service=WFS&version=2.0.0&request=GetFeature&typeName=geonode:Suelos_Esquina2&outputFormat=application/json',
        'escala': '1:100.000',
        'confiabilidad': 'alta',
        'bbox': (-59.7117, -30.4340, -58.7179, -29.4683),
        'simplificar_grados': 0.001,
    },
    {
        # Mezcla Esquina 1:100k + Goya/Lavalle 1:50k en una sola capa
        # remota (no se puede separar desde la metadata) -- se ubica en
        # el escalón 1:100.000 por precaución (el peor de los dos).
        'nombre': 'Corrientes — Esquina/Goya/Lavalle',
        'url': 'https://geo-nodo09.inta.gob.ar/geoserver/wfs?service=WFS&version=2.0.0&request=GetFeature&typeName=geonode:Suelos_Lavalle&outputFormat=application/json',
        'escala': '1:50.000-1:100.000 (mixta)',
        'confiabilidad': 'alta',
        'bbox': (-59.2386, -29.2576, -58.6001, -28.7688),
        'simplificar_grados': 0.0007,
    },
    {
        'nombre': 'Corrientes — Llanura Arenosa',
        'url': 'https://geo-nodo09.inta.gob.ar/geoserver/wfs?service=WFS&version=2.0.0&request=GetFeature&typeName=geonode:Lomas_arenosas&outputFormat=application/json',
        'escala': '1:100.000',
        'confiabilidad': 'alta',
        'bbox': (-59.1211, -29.0248, -57.1511, -27.4839),
        'simplificar_grados': 0.001,
    },
    # ── Entre Ríos 1:100.000 (geo-nodo03) -- OJO: esquema de atributos
    # complejo por serie (fase_2/3/5, serie_6, pos_1/2/7, ipc, ipcp,
    # eros_act, etc., hasta 7 series por unidad cartográfica), muy
    # distinto al esquema simple clase/cap_uso que espera _normalizar()
    # hoy. Va a cargar y devolver 'propiedades_crudas' igual, pero los
    # campos normalizados (clase/indice_prod/etc.) van a salir en blanco
    # hasta que se sume un parser dedicado tipo _normalizar_entre_rios().
    {
        'nombre': 'Entre Ríos (provincial)',
        'url': 'https://geo-nodo03.inta.gob.ar/geoserver/wfs?service=WFS&version=2.0.0&request=GetFeature&typeName=geonode:carta_de_suelos_unocienmil_df05b067570a0dea80e67f5aa41aa5d6&outputFormat=application/json',
        'escala': '1:100.000',
        'confiabilidad': 'media',  # normalización de campos aún pendiente, ver nota arriba
        'bbox': (-60.7749, -34.0386, -57.8013, -30.1590),
        'simplificar_grados': 0.001,
    },
    # ── San Luis 1:100.000 (geo-nodo15) -- estas SÍ confirmadas con
    # campos uc/cap_uso/ip (probado con Villa Mercedes vía WFS real). ──
    {
        'nombre': 'San Luis — Villa Mercedes',
        'url': 'https://geo-nodo15.inta.gob.ar/geoserver/wfs?service=WFS&version=2.0.0&request=GetFeature&typeName=geonode:villamercedes_v0&outputFormat=application/json',
        'escala': '1:100.000',
        'confiabilidad': 'alta',
        'bbox': (-66.0010, -33.9997, -65.0016, -32.9994),
        'simplificar_grados': 0.001,
    },
    {
        'nombre': 'San Luis — Buena Esperanza',
        'url': 'https://geo-nodo15.inta.gob.ar/geoserver/wfs?service=WFS&version=2.0.0&request=GetFeature&typeName=geonode:buenaesperanza_v0&outputFormat=application/json',
        'escala': '1:100.000',
        'confiabilidad': 'alta',
        'bbox': (-66.0009, -34.9998, -65.1046, -33.9994),
        'simplificar_grados': 0.001,
    },
    {
        'nombre': 'San Luis — San Luis (general)',
        'url': 'https://geo-nodo15.inta.gob.ar/geoserver/wfs?service=WFS&version=2.0.0&request=GetFeature&typeName=geonode:sanluis_v1&outputFormat=application/json',
        'escala': '1:100.000',
        'confiabilidad': 'alta',
        'bbox': (-67.1977, -33.9998, -66.0007, -32.9995),
        'simplificar_grados': 0.001,
    },
    {
        'nombre': 'San Luis — Arizona',
        'url': 'https://geo-nodo15.inta.gob.ar/geoserver/wfs?service=WFS&version=2.0.0&request=GetFeature&typeName=geonode:arizona_v0_02d647e0a3661237e9442c9ce388df2d&outputFormat=application/json',
        'escala': '1:100.000',
        'confiabilidad': 'alta',
        'bbox': (-66.0009, -36.0004, -65.1046, -34.9996),
        'simplificar_grados': 0.001,
    },
    {
        'nombre': 'San Luis — Martín de Loyola y Varela',
        'url': 'https://geo-nodo15.inta.gob.ar/geoserver/wfs?service=WFS&version=2.0.0&request=GetFeature&typeName=geonode:loyola_varela_v0_017028c132f4111b991dff14de997fd8&outputFormat=application/json',
        'escala': '1:100.000',
        'confiabilidad': 'alta',
        'bbox': (-66.8129, -35.9999, -66.0008, -33.9990),
        'simplificar_grados': 0.001,
    },
    {
        'nombre': 'San Luis — Concarán',
        'url': 'https://geo-nodo15.inta.gob.ar/geoserver/wfs?service=WFS&version=2.0.0&request=GetFeature&typeName=geonode:concaran_v0_676e7dd96118736d2ea36e767bd7c7fd&outputFormat=application/json',
        'escala': '1:100.000',
        'confiabilidad': 'alta',
        'bbox': (-66.0009, -32.9996, -64.8714, -31.8883),
        'simplificar_grados': 0.001,
    },
    # ── San Luis 1:200.000 (geo-nodo15) ──
    {
        'nombre': 'San Luis — Villa General Roca',
        'url': 'https://geo-nodo15.inta.gob.ar/geoserver/wfs?service=WFS&version=2.0.0&request=GetFeature&typeName=geonode:villagralroca_v0&outputFormat=application/json',
        'escala': '1:200.000',
        'confiabilidad': 'alta',
        'bbox': (-67.3471, -32.9997, -66.0006, -31.8339),
        'simplificar_grados': 0.002,
    },
    # ── Buenos Aires 1:250.000 (geo-nodo05) -- solo estos 2 partidos, no
    # sustituyen a la capa 'Buenos Aires' 1:50.000 general (más precisa
    # donde existe: consultar_mejor_capa()/capa_visual_bbox() ya prueban
    # las capas en este mismo orden de SUELO_CAPAS, así que estas dos de
    # 1:250.000 NUNCA se llegan a usar en un punto/bbox donde la capa
    # 'Buenos Aires' 1:50.000 (que va antes en la lista) ya tenga un
    # polígono real -- solo entran si esa carta general tiene un hueco
    # ahí, cosa poco probable ya que es la misma provincia.
    {
        'nombre': 'Buenos Aires — Villarino',
        'url': 'https://geo-nodo05.inta.gob.ar/geoserver/wfs?service=WFS&version=2.0.0&request=GetFeature&typeName=geonode:suelos_villarino&outputFormat=application/json',
        'escala': '1:250.000',
        'confiabilidad': 'media',
        'bbox': (-63.3836, -39.8476, -61.8508, -38.4558),
        'simplificar_grados': 0.0025,
    },
    {
        'nombre': 'Buenos Aires — Patagones',
        'url': 'https://geo-nodo05.inta.gob.ar/geoserver/wfs?service=WFS&version=2.0.0&request=GetFeature&typeName=geonode:suelos_patagones_final_pg2007f4&outputFormat=application/json',
        'escala': '1:250.000',
        'confiabilidad': 'media',
        'bbox': (-63.3930, -41.0383, -62.0612, -39.3241),
        'simplificar_grados': 0.0025,
    },
    # Santiago del Estero suroeste (Lavalle-Tapso-Frías) -- el título no
    # confirma escala clara, se ubica acá por precaución (peor de lo
    # esperado en vez de mejor). Atributos también sin confirmar.
    {
        'nombre': 'Santiago del Estero — Suroeste (Lavalle-Tapso-Frías)',
        'url': 'https://geo-nodo02.inta.gob.ar/geoserver/wfs?service=WFS&version=2.0.0&request=GetFeature&typeName=geonode:suelos_ltf&outputFormat=application/json',
        'escala': '1:250.000 (sin confirmar)',
        'confiabilidad': 'media',
        'bbox': (-65.3064, -28.7595, -64.9868, -28.0012),
        'simplificar_grados': 0.0025,
    },
    # ── Santiago del Estero 1:500.000, provincial completa (geo-nodo02)
    # -- sirve de relleno para el resto de la provincia donde ninguna
    # carta de detalle (Belgrano/La Esperanza/suroeste) llega.
    {
        'nombre': 'Santiago del Estero (provincial)',
        'url': 'https://geo-nodo02.inta.gob.ar/geoserver/wfs?service=WFS&version=2.0.0&request=GetFeature&typeName=geonode:suelos&outputFormat=application/json',
        'escala': '1:500.000',
        'confiabilidad': 'media',
        'bbox': (-65.3, -30.0, -61.5, -25.0),  # aproximado, provincia completa
        'simplificar_grados': 0.01,
    },
]

# ──────────────────────────────────────────────────────────────────────────
# NACIONAL, PARTIDO POR PROVINCIA (1:1.000.000, último recurso)
# ──────────────────────────────────────────────────────────────────────────
# Antes esto era UNA sola entrada 'Nacional' con bbox=None que cubría todo
# el país en un solo archivo de ~80MB (7783 polígonos). Cargar eso una sola
# vez ya alcanzaba para tirar abajo un dyno de Render con 512MB de RAM: se
# bajaba y parseaba el país ENTERO en memoria para responder una consulta
# de UN punto.
#
# Ahora está partido en 23 archivos, uno por provincia (mismo esquema
# nacional clásico de INTA: ind_prod, drenaje_s1, anegab_s1, limit_ppal),
# cada uno con su propio bbox real (calculado a partir de sus geometrías,
# + margen de ~5km) para que _capa_puede_cubrir_punto() descarte las que
# no corresponden sin descargar nada. Una consulta en Corrientes ahora solo
# baja y parsea el tile de Corrientes (~450KB, 200 polígonos) en vez del
# país entero -- una reducción de ~180x en ese caso.
#
# Además, los polígonos ya vienen simplificados de antemano (tolerancia
# 0.01°, la misma que se usaba en runtime antes) al generar estos archivos,
# así que 'simplificar_grados' queda en None acá: ya no hace falta volver a
# simplificar en cada carga.
SUELO_CAPAS_NACIONAL = [
    {
        'nombre': 'Nacional — Buenos Aires',
        'url': _HF_BASE + 'nacional_1_1000000_buenos_aires.geojson?download=true',
        'escala': '1:1.000.000',
        'confiabilidad': 'baja',
        'bbox': (-63.4424, -41.0937, -56.6184, -33.2322),
        'simplificar_grados': None,
    },
    {
        'nombre': 'Nacional — Catamarca',
        'url': _HF_BASE + 'nacional_1_1000000_catamarca.geojson?download=true',
        'escala': '1:1.000.000',
        'confiabilidad': 'baja',
        'bbox': (-69.1462, -30.1588, -64.733, -25.1144),
        'simplificar_grados': None,
    },
    {
        'nombre': 'Nacional — Chaco',
        'url': _HF_BASE + 'nacional_1_1000000_chaco.geojson?download=true',
        'escala': '1:1.000.000',
        'confiabilidad': 'baja',
        'bbox': (-63.4755, -28.0729, -57.5169, -22.4012),
        'simplificar_grados': None,
    },
    {
        'nombre': 'Nacional — Chubut',
        'url': _HF_BASE + 'nacional_1_1000000_chubut.geojson?download=true',
        'escala': '1:1.000.000',
        'confiabilidad': 'baja',
        'bbox': (-72.2372, -46.0573, -63.5313, -41.9462),
        'simplificar_grados': None,
    },
    {
        'nombre': 'Nacional — Cordoba',
        'url': _HF_BASE + 'nacional_1_1000000_cordoba.geojson?download=true',
        'escala': '1:1.000.000',
        'confiabilidad': 'baja',
        'bbox': (-65.8325, -35.0535, -61.7264, -29.457),
        'simplificar_grados': None,
    },
    {
        'nombre': 'Nacional — Corrientes',
        'url': _HF_BASE + 'nacional_1_1000000_corrientes.geojson?download=true',
        'escala': '1:1.000.000',
        'confiabilidad': 'baja',
        'bbox': (-59.7218, -30.7706, -55.574, -27.2147),
        'simplificar_grados': None,
    },
    {
        'nombre': 'Nacional — Entre Rios',
        'url': _HF_BASE + 'nacional_1_1000000_entre_rios.geojson?download=true',
        'escala': '1:1.000.000',
        'confiabilidad': 'baja',
        'bbox': (-60.8267, -34.0747, -57.7514, -30.1091),
        'simplificar_grados': None,
    },
    {
        'nombre': 'Nacional — Formosa',
        'url': _HF_BASE + 'nacional_1_1000000_formosa.geojson?download=true',
        'escala': '1:1.000.000',
        'confiabilidad': 'baja',
        'bbox': (-62.4015, -26.9175, -57.5238, -22.4069),
        'simplificar_grados': None,
    },
    {
        'nombre': 'Nacional — Jujuy',
        'url': _HF_BASE + 'nacional_1_1000000_jujuy.geojson?download=true',
        'escala': '1:1.000.000',
        'confiabilidad': 'baja',
        'bbox': (-67.2681, -24.6803, -64.1165, -21.7344),
        'simplificar_grados': None,
    },
    {
        'nombre': 'Nacional — La Pampa',
        'url': _HF_BASE + 'nacional_1_1000000_la_pampa.geojson?download=true',
        'escala': '1:1.000.000',
        'confiabilidad': 'baja',
        'bbox': (-68.3507, -39.3748, -63.3285, -34.948),
        'simplificar_grados': None,
    },
    {
        'nombre': 'Nacional — La Rioja',
        'url': _HF_BASE + 'nacional_1_1000000_la_rioja.geojson?download=true',
        'escala': '1:1.000.000',
        'confiabilidad': 'baja',
        'bbox': (-69.6988, -32.0282, -65.3664, -27.6309),
        'simplificar_grados': None,
    },
    {
        'nombre': 'Nacional — Mendoza',
        'url': _HF_BASE + 'nacional_1_1000000_mendoza.geojson?download=true',
        'escala': '1:1.000.000',
        'confiabilidad': 'baja',
        'bbox': (-70.6315, -37.627, -66.4291, -31.9568),
        'simplificar_grados': None,
    },
    {
        'nombre': 'Nacional — Misiones',
        'url': _HF_BASE + 'nacional_1_1000000_misiones.geojson?download=true',
        'escala': '1:1.000.000',
        'confiabilidad': 'baja',
        'bbox': (-56.1049, -28.2108, -53.5903, -25.4503),
        'simplificar_grados': None,
    },
    {
        'nombre': 'Nacional — Neuquen',
        'url': _HF_BASE + 'nacional_1_1000000_neuquen.geojson?download=true',
        'escala': '1:1.000.000',
        'confiabilidad': 'baja',
        'bbox': (-71.9919, -41.1677, -67.9793, -36.0592),
        'simplificar_grados': None,
    },
    {
        'nombre': 'Nacional — Rio Negro',
        'url': _HF_BASE + 'nacional_1_1000000_rio_negro.geojson?download=true',
        'escala': '1:1.000.000',
        'confiabilidad': 'baja',
        'bbox': (-71.9815, -42.0569, -62.7346, -37.5172),
        'simplificar_grados': None,
    },
    {
        'nombre': 'Nacional — Salta',
        'url': _HF_BASE + 'nacional_1_1000000_salta.geojson?download=true',
        'escala': '1:1.000.000',
        'confiabilidad': 'baja',
        'bbox': (-68.6095, -26.4365, -62.2877, -21.947),
        'simplificar_grados': None,
    },
    {
        'nombre': 'Nacional — San Juan',
        'url': _HF_BASE + 'nacional_1_1000000_san_juan.geojson?download=true',
        'escala': '1:1.000.000',
        'confiabilidad': 'baja',
        'bbox': (-70.6492, -32.6707, -66.614, -28.3575),
        'simplificar_grados': None,
    },
    {
        'nombre': 'Nacional — San Luis',
        'url': _HF_BASE + 'nacional_1_1000000_san_luis.geojson?download=true',
        'escala': '1:1.000.000',
        'confiabilidad': 'baja',
        'bbox': (-67.5338, -36.0544, -64.8261, -31.797),
        'simplificar_grados': None,
    },
    {
        'nombre': 'Nacional — Santa Cruz',
        'url': _HF_BASE + 'nacional_1_1000000_santa_cruz.geojson?download=true',
        'escala': '1:1.000.000',
        'confiabilidad': 'baja',
        'bbox': (-73.6102, -52.4334, -65.6799, -45.9461),
        'simplificar_grados': None,
    },
    {
        'nombre': 'Nacional — Santa Fe',
        'url': _HF_BASE + 'nacional_1_1000000_santa_fe.geojson?download=true',
        'escala': '1:1.000.000',
        'confiabilidad': 'baja',
        'bbox': (-62.9309, -34.4359, -57.9898, -27.9472),
        'simplificar_grados': None,
    },
    {
        'nombre': 'Nacional — Santiago Del Estero',
        'url': _HF_BASE + 'nacional_1_1000000_santiago_del_estero.geojson?download=true',
        'escala': '1:1.000.000',
        'confiabilidad': 'baja',
        'bbox': (-65.2263, -30.5286, -61.6584, -25.6008),
        'simplificar_grados': None,
    },
    {
        'nombre': 'Nacional — Tierra Del Fuego E Islas Malvinas',
        'url': _HF_BASE + 'nacional_1_1000000_tierra_del_fuego_e_islas_malvinas.geojson?download=true',
        'escala': '1:1.000.000',
        'confiabilidad': 'baja',
        'bbox': (-68.6726, -59.1479, -27.2037, -50.9754),
        'simplificar_grados': None,
    },
    {
        'nombre': 'Nacional — Tucuman',
        'url': _HF_BASE + 'nacional_1_1000000_tucuman.geojson?download=true',
        'escala': '1:1.000.000',
        'confiabilidad': 'baja',
        'bbox': (-66.2116, -28.0547, -64.421, -26.0155),
        'simplificar_grados': None,
    },
]

# El respaldo final SIEMPRE tiene que poder intentarse aunque el punto no
# caiga en ningún bbox conocido (ej. algún borde/gap entre provincias): se
# agrega una última entrada sin bbox con el archivo de Buenos Aires como
# resguardo NO es lo ideal -- mejor: si ninguna capa (ni las de detalle, ni
# NOA, ni ningún tile provincial) cubre el punto, consultar_mejor_capa()
# ya devuelve {'encontrado': False}, que es la respuesta correcta cuando
# el punto realmente no tiene ningún polígono de origen (ej. en el mar).

SUELO_CAPAS = SUELO_CAPAS + SUELO_CAPAS_NACIONAL

# Tope de features que devuelve capa_visual_bbox por pedido, para no mandar
# payloads gigantes si el usuario hace zoom-out sobre una provincia entera.
_MAX_FEATURES_BBOX = 3000


def _capa_puede_cubrir_punto(capa, lon, lat):
    """
    Chequeo BARATO (4 comparaciones de números, sin red ni parseo) para
    saltear por completo la descarga/parseo/indexado de una capa que
    geográficamente no puede contener el punto -- ver _cargar_capa() más
    abajo, que es la parte cara (GET a Hugging Face + shapely.shape() por
    cada polígono + construir el STRtree, todo eso queda cacheado en RAM
    PARA SIEMPRE sin límite).

    Antes de este chequeo, cada consulta desde una zona que ninguna capa
    de detalle cubre (ej. Corrientes, que no está en Santa Fe/Córdoba/
    Buenos Aires/NOA) terminaba cargando las 4 capas de detalle COMPLETAS
    igual, solo para descartarlas una por una, antes de llegar a
    'Nacional'. Con provincias grandes a 1:50.000 (Buenos Aires en
    particular), eso son fácilmente cientos de MB en objetos Shapely +
    STRtree cargados en memoria sin ninguna necesidad real.

    bbox=None (capa 'Nacional', cobertura de todo el país) siempre pasa
    el chequeo -- es el respaldo final.

    Los bbox son aproximados y con margen generoso a propósito: es
    preferible cargar una capa de más en un caso límite cerca del borde
    de una provincia, que descartar por error un polígono real que sí
    cubría el punto.
    """
    bbox = capa.get('bbox')
    if bbox is None:
        return True
    west, south, east, north = bbox
    return west <= lon <= east and south <= lat <= north


def _capa_puede_cubrir_bbox(capa, consulta_bbox):
    """Misma idea que _capa_puede_cubrir_punto(), pero para el overlay
    visual (capa_visual_bbox): compara dos rectángulos en vez de un punto
    contra un rectángulo."""
    bbox = capa.get('bbox')
    if bbox is None:
        return True
    c_west, c_south, c_east, c_north = consulta_bbox
    west, south, east, north = bbox
    return not (c_east < west or c_west > east or c_north < south or c_south > north)

# ──────────────────────────────────────────────────────────────────────────
# CACHE EN MEMORIA + ÍNDICE ESPACIAL (con tope LRU)
# ──────────────────────────────────────────────────────────────────────────
# Antes: _cache = {} común, sin límite -- cada capa que se cargaba una vez
# quedaba en RAM para siempre mientras viviera el proceso. En un dyno de
# Render con 512MB, esto se acumulaba: una consulta en Corrientes cargaba
# "Nacional" (todo el país) entero, una consulta posterior en Buenos Aires
# sumaba ENCIMA esa otra capa (potencialmente cientos de MB, ver nota en
# _capa_puede_cubrir_punto), y ninguna de las dos se liberaba nunca.
#
# Ahora: OrderedDict como caché LRU con tope _MAX_CAPAS_EN_CACHE. Cada
# acceso mueve la capa al final (más "reciente"); al agregar una nueva
# capa que supera el tope, se descarta la menos usada recientemente
# (la del principio) y se le pide a Python que libere esa memoria. Esto
# no evita que una capa pesada (ej. "Nacional") se cargue cuando hace
# falta, pero evita que dos o tres capas pesadas queden acumuladas para
# siempre en el mismo proceso.
# Ahora que "Nacional" está partido en 23 tiles chicos (cientos de KB cada
# uno, en vez de un solo archivo de ~80MB), un tope de 3 capas simultáneas
# sigue siendo seguro en memoria y evita recargar de red tan seguido si el
# tráfico va saltando entre 2-3 zonas distintas.
_MAX_CAPAS_EN_CACHE = 3
_cache = OrderedDict()  # nombre_capa -> {'tree', 'geoms', 'props'} | None (falló)
_cache_lock = threading.Lock()


# ──────────────────────────────────────────────────────────────────────────
# FILTRADO POR BBOX EN EL PROPIO PEDIDO WFS (evita OOM en Render)
# ──────────────────────────────────────────────────────────────────────────
# Las capas nuevas de GeoINTA (Corrientes, Formosa, San Luis, etc.) son
# remotas -- antes _cargar_capa() bajaba y parseaba la carta COMPLETA del
# departamento entero cada vez, aunque la vista actual del mapa solo
# necesitara un rincón chiquito. Eso hacía que Render se quedara sin
# memoria (plan free, 512MB) con cartas grandes como Pirané (5.104 km²).
# Fix: cuando la capa es un WFS en vivo, se le agrega un filtro BBOX al
# pedido GetFeature para que el propio servidor de INTA mande solo el
# recorte que hace falta -- mismo principio que ya se usa para no cargar
# capas fuera de rango (_capa_puede_cubrir_bbox), pero a nivel de la
# descarga en sí, no solo de "cuál capa probar".
_WFS_BBOX_TILE_DEG = 0.25   # tamaño de "tile" para cuantizar el bbox pedido
_WFS_BBOX_MARGEN_DEG = 0.05  # margen extra alrededor de lo pedido


def _es_capa_wfs_en_vivo(capa):
    """True si 'url' apunta a un WFS GetFeature en vivo (geo-nodoXX.inta.gob.ar),
    no a un archivo ya pre-alojado y liviano (ej. Hugging Face)."""
    return 'geoserver/wfs' in capa['url'] and 'GetFeature' in capa['url']


def _bbox_tile_para_wfs(bbox_consulta):
    """Cuantiza bbox_consulta=(west,south,east,north) a una grilla de
    _WFS_BBOX_TILE_DEG grados con margen, para (a) pedirle al WFS de INTA
    solo el pedazo que hace falta y (b) poder cachear ese pedazo en vez de
    la capa entera. Vistas cercanas caen en el mismo tile y reusan caché."""
    west, south, east, north = bbox_consulta

    def _piso(v):
        return math.floor((v - _WFS_BBOX_MARGEN_DEG) / _WFS_BBOX_TILE_DEG) * _WFS_BBOX_TILE_DEG

    def _techo(v):
        return math.ceil((v + _WFS_BBOX_MARGEN_DEG) / _WFS_BBOX_TILE_DEG) * _WFS_BBOX_TILE_DEG

    return (round(_piso(west), 2), round(_piso(south), 2), round(_techo(east), 2), round(_techo(north), 2))


def _cargar_capa(capa, bbox_consulta=None):
    nombre = capa['nombre']
    usar_bbox_wfs = bbox_consulta is not None and _es_capa_wfs_en_vivo(capa)
    tile = _bbox_tile_para_wfs(bbox_consulta) if usar_bbox_wfs else None
    clave_cache = f"{nombre}::{tile}" if tile else nombre

    with _cache_lock:
        if clave_cache in _cache:
            _cache.move_to_end(clave_cache)  # se acaba de usar -> pasa a "más reciente"
            return _cache[clave_cache]

    def _descargar(url_pedido, con_filtro):
        _t0 = time.monotonic()
        r = requests.get(url_pedido, timeout=120)
        r.raise_for_status()
        geojson = r.json()
        etiqueta = f'{nombre} [tile {tile}]' if con_filtro else nombre
        print(f'⏱️ [suelos_inta_regional] "{etiqueta}" descargada+parseada en {time.monotonic() - _t0:.1f}s')
        geoms, props = [], []
        for feat in geojson.get('features', []):
            geom = feat.get('geometry')
            if not geom:
                continue
            try:
                g = shape(geom)
                if not g.is_valid:
                    g = g.buffer(0)  # corrige auto-intersecciones menores
                tolerancia = capa.get('simplificar_grados')
                if tolerancia:
                    # Reduce la cantidad de vértices por polígono acorde a
                    # la escala real de la carta -- ver el valor de
                    # 'simplificar_grados' de cada capa en SUELO_CAPAS.
                    # preserve_topology=True evita que la simplificación
                    # rompa polígonos (auto-intersecciones nuevas).
                    g_simple = g.simplify(tolerancia, preserve_topology=True)
                    if not g_simple.is_empty:
                        g = g_simple
            except Exception:
                continue  # geometría puntualmente corrupta: se ignora
            geoms.append(g)
            props.append(feat.get('properties') or {})
        tree = STRtree(geoms) if geoms else None
        resultado = {'tree': tree, 'geoms': geoms, 'props': props}
        print(f'✅ [suelos_inta_regional] Carta "{etiqueta}" cargada: {len(geoms)} polígonos.')
        # El dict crudo del GeoJSON (r.json()) puede pesar bastante más que
        # el resultado final ya convertido a Shapely -- liberarlo explícito
        # en vez de esperar a que salga de scope ayuda a que el proceso le
        # devuelva esa memoria al SO antes del próximo pedido (mismo
        # patrón que ya usa cuenca_engine.py con sus arrays intermedios).
        del geojson, r
        gc.collect()
        return resultado

    datos = None
    try:
        if tile:
            w, s, e, n = tile
            # BBOX en orden lon/lat explícito (CRS84) -- evita el lío de
            # orden de ejes que trae EPSG:4326 en WFS 2.0.0 según el
            # servidor. Reduce lo que INTA manda a solo este recorte, en
            # vez de la carta del departamento entero -- clave para no
            # quedarse sin memoria en Render con cartas grandes.
            url_filtrada = f"{capa['url']}&bbox={w},{s},{e},{n},urn:ogc:def:crs:OGC::CRS84"
            try:
                datos = _descargar(url_filtrada, con_filtro=True)
            except Exception as exc_filtro:
                # Si el servidor no soporta este filtro (o cualquier otro
                # fallo puntual), reintentar sin filtro -- capa completa,
                # como antes -- en vez de dejar la zona sin datos. Es más
                # lento/pesado, pero sigue funcionando.
                print(f'⚠️ [suelos_inta_regional] BBOX filtrado falló en "{nombre}" '
                      f'({exc_filtro}); reintentando sin filtro (carta completa).')
                datos = _descargar(capa['url'], con_filtro=False)
        else:
            datos = _descargar(capa['url'], con_filtro=False)
    except Exception as exc:  # noqa: BLE001
        print(f'⚠️ [suelos_inta_regional] No se pudo cargar la carta "{nombre}": {exc}')
        # OJO: si falló (datos sigue en None acá), NO se guarda en _cache --
        # antes se guardaba igual, así que un fallo puntual (ej. sin memoria
        # a mitad de la descarga, timeout de red) dejaba esta carta marcada
        # como "sin datos" PARA SIEMPRE en ese proceso, sin volver a
        # intentarlo nunca. Ahora el próximo pedido a esta misma carta la
        # vuelve a intentar desde cero.
        return None
    with _cache_lock:
        _cache[clave_cache] = datos
        _cache.move_to_end(clave_cache)
        while len(_cache) > _MAX_CAPAS_EN_CACHE:
            nombre_descartado, _ = _cache.popitem(last=False)  # la menos usada recientemente
            print(f'🗑️ [suelos_inta_regional] Descartando capa "{nombre_descartado}" del caché '
                  f'(tope de {_MAX_CAPAS_EN_CACHE} capas simultáneas) para liberar memoria.')
        gc.collect()
    return datos


# ──────────────────────────────────────────────────────────────────────────
# NORMALIZACIÓN DE PROPIEDADES
# ──────────────────────────────────────────────────────────────────────────
# Distintas cartas traen distintos nombres de columna:
#   - Cartas regionales (Santa Fe/Córdoba/Buenos Aires/NOA):
#       clase, subclase, cap_uso, drenaje_estimado
#   - Esquema nacional clásico de INTA (por si el archivo "Nacional" lo usa):
#       ind_prod, drenaje_s1, anegab_s1, limit_ppal
# _normalizar() deja todo en el MISMO esquema de salida para que el
# frontend (_aptitudClasificarProps, _aptitudPintarUI) no tenga que saber
# de dónde vino cada polígono.

_INDICE_POR_CLASE = {
    'I': 95, 'II': 80, 'III': 65, 'IV': 50,
    'V': 35, 'VI': 25, 'VII': 10, 'VIII': 0,
}

_LIMITANTE_POR_LETRA = {
    'e': 'Erosión',
    's': 'Suelo (limitaciones edáficas)',
    'c': 'Clima',
    'w': 'Anegamiento / drenaje',
    'p': 'Pedregosidad / profundidad efectiva',
}

_ROMANO_POR_DIGITO = {
    '1': 'I', '2': 'II', '3': 'III', '4': 'IV',
    '5': 'V', '6': 'VI', '7': 'VII', '8': 'VIII',
}

# Carta de Santa Fe (y potencialmente otras cartas de detalle de INTA):
# en vez de columnas 'clase'/'indice_prod' ya legibles, usa el esquema
# clásico GAT/IAT:
#   gat = "Grupo de Aptitud de Tierras", ej. "6ws(e)", "4/5wp(s)", "2w"
#         -> dígito(s) inicial = clase (1-8, equivalente a I-VIII);
#            letras después = subclase/limitantes (e,s,c,w,p);
#            grupo entre paréntesis = limitante secundaria.
#   iat = "Índice de Aptitud de Tierras", 0-100 -- EQUIVALENTE directo a
#         indice_prod de las otras cartas, es el dato que el frontend usa
#         para colorear el mapa (_aptitudClasificarProps en el HTML).
import re as _re

_GAT_RE = _re.compile(r'^(\d+)(?:/(\d+))?([a-z()]*)$')


def _parse_gat(gat_crudo):
    """'6ws(e)' -> ('VI', 'ws(e)'); '4/5wp(s)' -> ('IV/V', 'wp(s)').
    Devuelve (clase, subclase) o (None, None) si no matchea el patrón
    esperado (para no inventar datos de un formato inesperado)."""
    if not gat_crudo:
        return None, None
    s = _re.sub(r'\s+', '', str(gat_crudo)).lower()
    m = _GAT_RE.match(s)
    if not m:
        return None, None
    n1, n2, resto = m.groups()
    clase = _ROMANO_POR_DIGITO.get(n1)
    if not clase:
        return None, None
    if n2:
        clase = f"{clase}/{_ROMANO_POR_DIGITO.get(n2, n2)}"
    return clase, (resto or None)


_ROMANO_TXT_RE = _re.compile(r'^(VIII|VII|VI|V|IV|III|II|I)', _re.IGNORECASE)


def _parse_cap_uso_romano(texto):
    """'IIIes' -> ('III','es'); 'IIw o I-2' -> ('II','w'); 'VI/VIIes' ->
    ('VI', None). Para cartas que traen la clase de capacidad de uso ya
    en números romanos pero combinada con la subclase en un solo campo
    (ej. Buenos Aires: CAP_USO), en vez de columnas separadas."""
    if not texto:
        return None, None
    t = str(texto).strip()
    t = _re.split(r'[/]| o ', t)[0].strip()  # rangos/alternativas: toma la primera opción
    m = _ROMANO_TXT_RE.match(t.upper())
    if not m:
        return None, None
    clase = m.group(1)
    resto = t[len(clase):]
    resto = _re.sub(r'[^a-zA-Z]', '', resto).lower()
    return clase, (resto or None)


# NOA usa un esquema propio de 5 grupos de tierras (A mejor -> E peor), no
# el I-VIII del resto de las cartas. Ver Suelos_NOA.geojson: 'grupo_tier'
# ('A'..'E', a veces transicional 'B-C') + 'descri' (texto). Equivalencia
# aproximada a clase romana + índice, solo para que se pueda comparar/
# colorear junto con el resto de las capas -- es una carta "regional, zonas
# grandes" (confiabilidad 'media'), no se pretende precisión fina acá.
_NOA_GRUPO_A_CLASE = {
    'A': 'I', 'B': 'III', 'C': 'V', 'D': 'VI', 'E': 'VIII',
}


def _parse_grupo_tier_noa(grupo_tier):
    if not grupo_tier:
        return None
    letra = str(grupo_tier).strip().upper().split('-')[0]  # "B-C" -> "B" (el mejor extremo)
    return _NOA_GRUPO_A_CLASE.get(letra)


def _limitante_desde_subclase(subclase):
    if not subclase:
        return None
    vistas, orden = set(), []
    for ch in str(subclase).lower():
        etiqueta = _LIMITANTE_POR_LETRA.get(ch)
        if etiqueta and etiqueta not in vistas:
            vistas.add(etiqueta)
            orden.append(etiqueta)
    return ' + '.join(orden) if orden else None


def _drenaje_legible(valor):
    if not valor:
        return None
    return str(valor).replace('_', ' ').strip().capitalize()


def _anegabilidad_desde_drenaje(drenaje_crudo):
    d = str(drenaje_crudo or '').lower()
    if any(p in d for p in ('pobre', 'malo', 'deficient')):
        return 'Probable (drenaje deficiente)'
    return None


def _g(props, *claves):
    """Busca cualquiera de `claves` en `props`, probando también
    MAYUSCULA/minúscula/Capitalizada."""
    for k in claves:
        for variante in (k, k.upper(), k.lower(), k.capitalize()):
            if variante in props and props[variante] not in (None, ''):
                return props[variante]
    return None


def _normalizar(props):
    clase = _g(props, 'clase')
    subclase = _g(props, 'subclase')
    # 'Cap Uso' (con espacio, Córdoba) e 'IP' (Córdoba) se agregan acá como
    # alias explícitos porque _g() solo prueba variantes de MAYÚSCULA/
    # minúscula/Capitalizada de la MISMA palabra -- 'cap_uso' con guion
    # bajo nunca matchea 'Cap Uso' con espacio, son strings distintos.
    cap_uso = _g(props, 'cap_uso', 'capacidad_uso', 'Cap Uso')
    drenaje_crudo = _g(props, 'drenaje_estimado', 'drenaje', 'drenaje_s1')
    # 'Ipc' (Valle de Lerma / Valles Calchaquíes, Salta) es un índice de
    # productividad numérico igual que 'IP' de Córdoba, solo con otro
    # nombre de columna -- se agrega como alias explícito por el mismo
    # motivo que 'Cap Uso' arriba: _g() no matchea nombres distintos.
    indice_prod = _g(props, 'ind_prod', 'indice_prod', 'IP', 'Ipc')

    # Esquema GAT/IAT (ej. Santa Fe 1:50.000) -- ver _parse_gat() arriba.
    # Solo se usa como respaldo si el esquema estándar no trajo nada, así
    # no pisa datos válidos de otras cartas que sí usan 'clase'/'ind_prod'.
    if clase is None:
        gat_crudo = _g(props, 'gat')
        clase_gat, subclase_gat = _parse_gat(gat_crudo)
        if clase_gat:
            clase = clase_gat
            if subclase is None:
                subclase = subclase_gat

    # Esquema "capacidad de uso combinada en un solo campo" (ej. Buenos
    # Aires: CAP_USO="IIIes", sin columnas Clase/Subclase separadas).
    # También respaldo, mismo criterio que GAT arriba.
    if clase is None and cap_uso:
        clase_cu, subclase_cu = _parse_cap_uso_romano(cap_uso)
        if clase_cu:
            clase = clase_cu
            if subclase is None:
                subclase = subclase_cu

    # Esquema NOA (grupo_tier A-E). Último respaldo: es la carta "regional,
    # zonas grandes" -- solo se usa si ninguna de las anteriores dio clase.
    if clase is None:
        clase_noa = _parse_grupo_tier_noa(_g(props, 'grupo_tier'))
        if clase_noa:
            clase = clase_noa

    # Esquema "Ap_Agricol" (ej. capas de Salta: Miraflores/El Galpón, Valle
    # de Lerma, Valles Calchaquíes -- subidas directo a Hugging Face, no
    # forman parte del catálogo INTA revisado hoy). Viene como letra A-D
    # (a veces combinada tipo "C/D") en vez de número romano de clase.
    # Se traduce a la clase romana equivalente (A=I la mejor, D=IV la peor)
    # solo como respaldo -- igual criterio que los esquemas de arriba.
    if clase is None:
        ap_agricol = _g(props, 'Ap_Agricol', 'ap_agricol')
        if ap_agricol:
            letra = str(ap_agricol).split('/')[0].strip().upper()
            clase = {'A': 'I', 'B': 'II', 'C': 'III', 'D': 'IV'}.get(letra)

    if indice_prod in (None, ''):
        indice_prod = _g(props, 'iat')  # Índice de Aptitud de Tierras, ya en escala 0-100
    if indice_prod in (None, ''):
        indice_prod = _INDICE_POR_CLASE.get(str(clase or '').split('/')[0].upper())

    limitante = _g(props, 'limit_ppal', 'limitante') or _limitante_desde_subclase(subclase)
    anegabilidad = _g(props, 'anegab_s1', 'anegabilidad') or _anegabilidad_desde_drenaje(drenaje_crudo)
    if anegabilidad is None and subclase and 'w' in str(subclase).lower():
        anegabilidad = 'Probable (limitante por anegamiento/drenaje según carta GAT)'
    # Santa Fe: 'bajo'='1' marca explícitamente zona baja/anegable en la
    # cartografía original, más allá de lo que diga la subclase.
    if anegabilidad is None and str(_g(props, 'bajo') or '') == '1':
        anegabilidad = 'Probable (zona baja según carta de Santa Fe)'

    if cap_uso is None and clase:
        cap_uso = f"Clase {clase}{subclase or ''}"

    # NOA es descriptivo, no numérico -- si no hay nada más, al menos
    # mostrar el texto de la carta original en vez de dejarlo vacío.
    if cap_uso is None:
        descri_noa = _g(props, 'descri')
        if descri_noa:
            cap_uso = str(descri_noa).strip()[:120]

    return {
        'clase': clase,
        'subclase': subclase,
        'capacidad_uso': cap_uso,
        'indice_prod': indice_prod,
        'limitante': limitante,
        'drenaje': _drenaje_legible(drenaje_crudo),
        'anegabilidad': anegabilidad,
    }


def _advertencia_escala(capa):
    if capa['confiabilidad'] == 'alta':
        return None
    return (
        f'Esta zona todavía no tiene carta de detalle 1:50.000: el dato viene '
        f'de "{capa["nombre"]}" (escala {capa["escala"]}), menos preciso.'
    )


# ──────────────────────────────────────────────────────────────────────────
# API PÚBLICA — llamada desde app.py
# ──────────────────────────────────────────────────────────────────────────
def consultar_mejor_capa(lat, lon):
    """Consulta puntual: prueba cada capa en orden de prioridad (mejor
    escala primero) y devuelve el primer polígono que contiene (lat, lon).
    Formato de salida = el que ya espera _aptitudFetchINTA() en el frontend."""
    punto = Point(lon, lat)
    # Bbox chico alrededor del punto -- para que las capas WFS en vivo
    # bajen solo ese recorte en vez de la carta entera (ver _cargar_capa).
    margen_punto = 0.02
    bbox_punto = (lon - margen_punto, lat - margen_punto, lon + margen_punto, lat + margen_punto)

    for capa in SUELO_CAPAS:
        if not _capa_puede_cubrir_punto(capa, lon, lat):
            continue  # fuera del bbox de esta capa: ni se descarga ni se carga en memoria
        datos = _cargar_capa(capa, bbox_consulta=bbox_punto)
        if not datos or not datos['tree']:
            continue
        for i in datos['tree'].query(punto):
            geom = datos['geoms'][i]
            if geom.covers(punto):
                props = datos['props'][i]
                norm = _normalizar(props)
                return {
                    'encontrado': True,
                    'fuente': capa['nombre'],
                    'clase': norm['clase'],
                    'subclase': norm['subclase'],
                    'capacidad_uso': norm['capacidad_uso'],
                    'indice_prod': norm['indice_prod'],
                    'limitante': norm['limitante'],
                    'drenaje': norm['drenaje'],
                    'anegabilidad': norm['anegabilidad'],
                    'advertencia_escala': _advertencia_escala(capa),
                    'confiabilidad': capa['confiabilidad'],
                    'propiedades_crudas': props,
                }

    # Ninguna capa (ni siquiera la Nacional) cubre este punto
    return {'encontrado': False, 'fuente': None}


# ──────────────────────────────────────────────────────────────────────────
# RELLENO CON NACIONAL PARA HUECOS DE LA CARTA DE DETALLE
# ──────────────────────────────────────────────────────────────────────────
# Las cartas de detalle (Santa Fe/Córdoba/Buenos Aires) no necesariamente
# tienen un polígono para CADA rincón de su provincia -- Córdoba en
# particular viene con unidades cartográficas grandes (pocas, extensas),
# a diferencia de Santa Fe (mosaico denso, un polígono por campo). Cuando
# la vista del mapa cae en una zona que la carta de detalle apenas
# cubre, en vez de mostrar un área enorme vacía sin colorear, se
# complementa con el tile Nacional (1:1.000.000) de esa misma provincia,
# dibujado DETRÁS de los polígonos de detalle (para no toparlos ni
# perder precisión donde sí hay dato fino).
#
# NOA queda afuera de este mapeo a propósito: cubre varias provincias a
# la vez, no hay un único tile Nacional al que corresponda 1 a 1.
_PROVINCIA_A_NACIONAL = {
    'Córdoba': 'Nacional — Cordoba',
    'Santa Fe': 'Nacional — Santa Fe',
    'Buenos Aires': 'Nacional — Buenos Aires',
    'Corrientes': 'Nacional — Corrientes',
    'Santiago del Estero': 'Nacional — Santiago Del Estero',
    'Misiones': 'Nacional — Misiones',
    'Formosa': 'Nacional — Formosa',
    'San Luis': 'Nacional — San Luis',
    'Salta': 'Nacional — Salta',
    'Entre Ríos': 'Nacional — Entre Rios',
}

# Si la carta de detalle ya cubre esta fracción (o más) del área visible
# con polígonos reales, no vale la pena sumar el tile Nacional encima --
# el hueco es chico y el relleno agregaría una consulta/descarga extra
# sin beneficio visual apreciable.
_COBERTURA_MINIMA_SIN_RELLENO = 0.55


def _provincia_de_capa_detalle(capa):
    """'Córdoba (-65_-34)' -> 'Córdoba'; 'Santa Fe (c0f0)' -> 'Santa Fe';
    'Buenos Aires' -> 'Buenos Aires'; 'San Luis — Villa Mercedes' ->
    'San Luis' (patrón con raya usado por las capas remotas de GeoINTA
    agregadas después de Córdoba/Santa Fe/Buenos Aires). None si no es
    una carta de detalle provincial mapeada (ej. NOA, o ya es Nacional)."""
    nombre = capa['nombre']
    for provincia in _PROVINCIA_A_NACIONAL:
        if (nombre == provincia
                or nombre.startswith(provincia + ' (')
                or nombre.startswith(provincia + ' — ')):
            return provincia
    return None


def _capa_nacional_de(provincia):
    nombre_nacional = _PROVINCIA_A_NACIONAL.get(provincia)
    if not nombre_nacional:
        return None
    for capa in SUELO_CAPAS_NACIONAL:
        if capa['nombre'] == nombre_nacional:
            return capa
    return None


def _features_de_capa_en_bbox(capa, caja):
    """Devuelve (features_geojson, geoms_shapely) de los polígonos de
    `capa` que intersectan `caja`, con propiedades ya normalizadas.
    Reutilizado tanto por la carta de detalle como por el relleno
    Nacional en capa_visual_bbox()."""
    datos = _cargar_capa(capa, bbox_consulta=caja.bounds)
    if not datos or not datos['tree']:
        return [], []
    indices = datos['tree'].query(caja)
    features, geoms = [], []
    for i in indices:
        geom = datos['geoms'][i]
        if not geom.intersects(caja):
            continue
        geoms.append(geom)
        features.append({
            'type': 'Feature',
            'geometry': geom.__geo_interface__,
            'properties': _normalizar(datos['props'][i]),
        })
        if len(features) >= _MAX_FEATURES_BBOX:
            break
    return features, geoms


def capa_visual_bbox(bbox):
    """Overlay visual: devuelve un FeatureCollection GeoJSON con los
    polígonos dentro de `bbox` = (west, south, east, north), con las
    propiedades ya normalizadas para que el frontend los pinte de forma
    uniforme sin importar de qué carta salió cada uno.

    Usa SIEMPRE una sola capa por pedido (la de mejor escala que tenga
    algo dentro del bbox), para no mezclar visualmente polígonos de
    distinta escala/confiabilidad en una misma vista."""
    west, south, east, north = bbox
    caja = box(west, south, east, north)

    # DIAGNÓSTICO: registra qué pasó con cada capa candidata para este bbox
    # (bbox-match, carga OK/fallida, cuántas features intersectan de verdad).
    # Ver estos logs en Render (dashboard > servicio TerraOs-1 > Logs) tras
    # reproducir un caso de "capa no aparece" es la forma más rápida de saber
    # si el problema es: (a) ninguna capa cubre ese bbox geográficamente,
    # (b) alguna capa cubre pero no tiene features ahí (hueco real de datos),
    # o (c) una capa falló al descargar/parsear (ver _cargar_capa).
    intentos = []

    for capa in SUELO_CAPAS:
        if not _capa_puede_cubrir_bbox(capa, bbox):
            continue  # fuera del bbox de esta capa: ni se descarga ni se carga en memoria
        datos = _cargar_capa(capa, bbox_consulta=bbox)
        if not datos or not datos['tree']:
            intentos.append(f"{capa['nombre']}: FALLÓ AL CARGAR")
            continue

        indices = datos['tree'].query(caja)
        if len(indices) == 0:
            intentos.append(f"{capa['nombre']}: 0 candidatos en STRtree para este bbox")
            continue  # esta capa no tiene nada en este bbox: probamos la siguiente

        features = []
        geoms = []
        for i in indices:
            geom = datos['geoms'][i]
            if not geom.intersects(caja):
                continue
            props = datos['props'][i]
            geoms.append(geom)
            features.append({
                'type': 'Feature',
                'geometry': geom.__geo_interface__,
                'properties': _normalizar(props),
            })
            if len(features) >= _MAX_FEATURES_BBOX:
                break

        if not features:
            intentos.append(
                f"{capa['nombre']}: {len(indices)} candidatos en STRtree pero 0 "
                f"intersectan realmente el bbox (hueco de datos o simplificación "
                f"movió el polígono fuera)"
            )
            continue

        print(
            f'🗺️ [suelos_inta_regional] capa_visual_bbox {bbox} -> "{capa["nombre"]}" '
            f'({len(features)} features). Intentos previos: {intentos or "ninguno"}'
        )

        features_finales = features
        fuente_txt = capa['nombre']
        advertencia = _advertencia_escala(capa)

        # ── Relleno con Nacional si la carta de detalle deja huecos ──
        provincia = _provincia_de_capa_detalle(capa)
        if provincia:
            try:
                geoms_validas = [g for g in geoms if g.is_valid]
                cobertura = unary_union(geoms_validas).intersection(caja).area / caja.area if geoms_validas else 0
            except Exception as exc:
                print(f'⚠️ [suelos_inta_regional] no se pudo calcular cobertura de "{capa["nombre"]}": {exc}')
                cobertura = 1  # ante la duda, no molestar con un relleno de más

            if cobertura < _COBERTURA_MINIMA_SIN_RELLENO:
                capa_nacional = _capa_nacional_de(provincia)
                if capa_nacional and capa_nacional['nombre'] != capa['nombre'] and _capa_puede_cubrir_bbox(capa_nacional, bbox):
                    features_relleno, _ = _features_de_capa_en_bbox(capa_nacional, caja)
                    if features_relleno:
                        # El relleno va PRIMERO en la lista -> el frontend lo dibuja
                        # detrás de los polígonos de detalle (bringToBack global ya
                        # pone toda la capa al fondo del mapa; el orden acá solo
                        # define qué queda arriba DENTRO de esta misma capa).
                        features_finales = features_relleno + features
                        fuente_txt = f"{capa['nombre']} + relleno {capa_nacional['nombre']} ({cobertura:.0%} cobertura de detalle)"
                        advertencia = (
                            f'Esta vista combina la carta de detalle de {provincia} (más precisa, '
                            f'cubre ~{cobertura:.0%} del área) con la carta Nacional 1:1.000.000 '
                            f'para el resto — el resto tiene menor precisión.'
                        )
                        print(
                            f'🧩 [suelos_inta_regional] relleno aplicado: "{capa["nombre"]}" '
                            f'({cobertura:.0%} cobertura) + "{capa_nacional["nombre"]}" '
                            f'({len(features_relleno)} features de relleno)'
                        )

        return {
            'type': 'FeatureCollection',
            'fuente': fuente_txt,
            'confiabilidad': capa['confiabilidad'],
            'advertencia_escala': advertencia,
            'features': features_finales,
        }

    print(f'⚠️ [suelos_inta_regional] capa_visual_bbox {bbox} -> SIN RESULTADO. Intentos: {intentos}')
    return {'type': 'FeatureCollection', 'fuente': None, 'features': []}
