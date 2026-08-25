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
  5) Nacional, partido por provincia — 1:500.000 (último recurso, 23
     archivos -- ver SUELO_CAPAS_NACIONAL). Antes era UN solo archivo de
     ~80MB que cubría todo el país; partirlo por provincia hace que cada
     consulta solo baje y cargue el tile de la provincia correspondiente
     (unos cientos de KB a ~15MB para Buenos Aires) en vez del país entero.
     Reemplaza la carta nacional 1:1.000.000 anterior (mismo particionado
     por provincia, mejor detalle de origen).

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
    # con solo intentar descargarlo/parsearlo una vez. Partido en 15 tiles
    # balanceados por PESO de coordenadas (proxy de tamaño real de archivo,
    # no por cantidad de polígonos), en una grilla espacial de ~5 columnas
    # (longitud) x 3 filas (latitud) -- ver index_tiles.json subido junto
    # a los archivos en el dataset de Hugging Face para el detalle de cada
    # bbox. A diferencia de Córdoba/Buenos Aires, estos tiles NO vienen
    # pre-simplificados -- por eso 'simplificar_grados' va en 0.0005 (igual
    # que las demás cartas de detalle que se simplifican al cargar).
    # Los tiles vecinos pueden solaparse levemente en el borde -- normal,
    # consultar_mejor_capa() prueba todos los que puedan cubrir el punto
    # en orden y devuelve el primero que matchea.
    {
        'nombre': 'Santa Fe (tile01)',
        'url': _HF_BASE + 'Santa_Fe_1_en_50_tile01.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-62.8663, -34.3855, -61.7443, -33.7235),
        'simplificar_grados': 0.0005,  # NO viene pre-simplificado -- se simplifica al cargar (igual que Córdoba/Buenos Aires)
    },
    {
        'nombre': 'Santa Fe (tile02)',
        'url': _HF_BASE + 'Santa_Fe_1_en_50_tile02.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-62.3415, -33.7234, -61.7448, -32.2454),
        'simplificar_grados': 0.0005,  # NO viene pre-simplificado -- se simplifica al cargar (igual que Córdoba/Buenos Aires)
    },
    {
        'nombre': 'Santa Fe (tile03)',
        'url': _HF_BASE + 'Santa_Fe_1_en_50_tile03.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-62.2279, -32.2448, -61.7442, -28.4396),
        'simplificar_grados': 0.0005,  # NO viene pre-simplificado -- se simplifica al cargar (igual que Córdoba/Buenos Aires)
    },
    {
        'nombre': 'Santa Fe (tile04)',
        'url': _HF_BASE + 'Santa_Fe_1_en_50_tile04.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-61.7442, -34.3789, -61.4143, -32.8158),
        'simplificar_grados': 0.0005,  # NO viene pre-simplificado -- se simplifica al cargar (igual que Córdoba/Buenos Aires)
    },
    {
        'nombre': 'Santa Fe (tile05)',
        'url': _HF_BASE + 'Santa_Fe_1_en_50_tile05.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-61.7442, -32.8146, -61.4143, -30.8146),
        'simplificar_grados': 0.0005,  # NO viene pre-simplificado -- se simplifica al cargar (igual que Córdoba/Buenos Aires)
    },
    {
        'nombre': 'Santa Fe (tile06)',
        'url': _HF_BASE + 'Santa_Fe_1_en_50_tile06.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-61.7441, -30.8140, -61.4143, -28.1344),
        'simplificar_grados': 0.0005,  # NO viene pre-simplificado -- se simplifica al cargar (igual que Córdoba/Buenos Aires)
    },
    {
        'nombre': 'Santa Fe (tile07)',
        'url': _HF_BASE + 'Santa_Fe_1_en_50_tile07.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-61.4139, -34.0827, -61.0387, -32.7129),
        'simplificar_grados': 0.0005,  # NO viene pre-simplificado -- se simplifica al cargar (igual que Córdoba/Buenos Aires)
    },
    {
        'nombre': 'Santa Fe (tile08)',
        'url': _HF_BASE + 'Santa_Fe_1_en_50_tile08.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-61.4143, -32.7125, -61.0387, -30.9831),
        'simplificar_grados': 0.0005,  # NO viene pre-simplificado -- se simplifica al cargar (igual que Córdoba/Buenos Aires)
    },
    {
        'nombre': 'Santa Fe (tile09)',
        'url': _HF_BASE + 'Santa_Fe_1_en_50_tile09.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-61.4143, -30.9828, -61.0387, -28.1747),
        'simplificar_grados': 0.0005,  # NO viene pre-simplificado -- se simplifica al cargar (igual que Córdoba/Buenos Aires)
    },
    {
        'nombre': 'Santa Fe (tile10)',
        'url': _HF_BASE + 'Santa_Fe_1_en_50_tile10.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-61.0386, -33.7339, -60.5001, -31.5015),
        'simplificar_grados': 0.0005,  # NO viene pre-simplificado -- se simplifica al cargar (igual que Córdoba/Buenos Aires)
    },
    {
        'nombre': 'Santa Fe (tile11)',
        'url': _HF_BASE + 'Santa_Fe_1_en_50_tile11.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-61.0386, -31.5015, -60.5000, -30.7748),
        'simplificar_grados': 0.0005,  # NO viene pre-simplificado -- se simplifica al cargar (igual que Córdoba/Buenos Aires)
    },
    {
        'nombre': 'Santa Fe (tile12)',
        'url': _HF_BASE + 'Santa_Fe_1_en_50_tile12.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-61.0386, -30.7744, -60.4998, -28.6514),
        'simplificar_grados': 0.0005,  # NO viene pre-simplificado -- se simplifica al cargar (igual que Córdoba/Buenos Aires)
    },
    {
        'nombre': 'Santa Fe (tile13)',
        'url': _HF_BASE + 'Santa_Fe_1_en_50_tile13.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-60.4997, -33.6321, -59.8852, -30.0145),
        'simplificar_grados': 0.0005,  # NO viene pre-simplificado -- se simplifica al cargar (igual que Córdoba/Buenos Aires)
    },
    {
        'nombre': 'Santa Fe (tile14)',
        'url': _HF_BASE + 'Santa_Fe_1_en_50_tile14.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-60.4979, -30.0145, -59.6390, -29.2113),
        'simplificar_grados': 0.0005,  # NO viene pre-simplificado -- se simplifica al cargar (igual que Córdoba/Buenos Aires)
    },
    {
        'nombre': 'Santa Fe (tile15)',
        'url': _HF_BASE + 'Santa_Fe_1_en_50_tile15.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-60.0489, -29.2112, -59.1798, -28.0066),
        'simplificar_grados': 0.0005,  # NO viene pre-simplificado -- se simplifica al cargar (igual que Córdoba/Buenos Aires)
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
    # ── Corrientes 1:50.000/1:100.000/1:500.000 -- Instituto de Suelos INTA,
    # cartas provinciales completas (17.847 polígonos), autoalojadas en HF
    # (antes: 5 capas remotas WFS de INTA con cobertura parcial, reemplazadas
    # del todo por esta carta completa). Igual criterio que Santa Fe/Córdoba:
    # partida en grilla adaptativa (0.5° base, subdividida solo donde hacía
    # falta) para que ningún tile pase ~8MB. bbox = extensión REAL de cada
    # archivo (no el borde nominal de la celda).
    {
        'nombre': 'Corrientes (m56_5_m27_5)',
        'url': _HF_BASE + 'corrientes_m56_5_m27_5.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-56.2839, -27.5608, -55.9747, -27.3126),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m56_5_m28)',
        'url': _HF_BASE + 'corrientes_m56_5_m28.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-56.54, -28.0258, -55.8543, -27.4704),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m56_5_m28_5)',
        'url': _HF_BASE + 'corrientes_m56_5_m28_5.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-56.5527, -28.7834, -55.8841, -27.8401),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m56_5_m29)',
        'url': _HF_BASE + 'corrientes_m56_5_m29.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-56.5856, -29.1162, -56.0104, -28.4519),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m56_5_m29_5)',
        'url': _HF_BASE + 'corrientes_m56_5_m29_5.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-56.5021, -29.0692, -56.4321, -29.0042),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m56_m28)',
        'url': _HF_BASE + 'corrientes_m56_m28.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-56.0421, -28.0235, -55.7703, -27.5473),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m56_m28_5)',
        'url': _HF_BASE + 'corrientes_m56_m28_5.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-56.3063, -28.8129, -55.62, -27.9658),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m57_5_m27_5)',
        'url': _HF_BASE + 'corrientes_m57_5_m27_5.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-57.6539, -27.5849, -57.2274, -27.4025),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m57_5_m28)',
        'url': _HF_BASE + 'corrientes_m57_5_m28.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-57.6554, -27.9497, -56.9127, -27.4336),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m57_5_m29)',
        'url': _HF_BASE + 'corrientes_m57_5_m29.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-57.5346, -28.9938, -57.1712, -28.4994),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m57_5_m29_5)',
        'url': _HF_BASE + 'corrientes_m57_5_m29_5.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-57.5612, -29.283, -57.1644, -28.5674),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m57_75_m30_25)',
        'url': _HF_BASE + 'corrientes_m57_75_m30_25.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-57.7799, -30.2634, -57.5591, -29.9918),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m57_75_m30_5)',
        'url': _HF_BASE + 'corrientes_m57_75_m30_5.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-57.7685, -30.3967, -57.6212, -30.2082),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m57_m27_5)',
        'url': _HF_BASE + 'corrientes_m57_m27_5.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-56.717, -27.5147, -56.5467, -27.4554),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m57_m28)',
        'url': _HF_BASE + 'corrientes_m57_m28.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-57.0332, -28.0194, -56.4852, -27.4713),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m57_m28_5)',
        'url': _HF_BASE + 'corrientes_m57_m28_5.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-56.8809, -28.5049, -56.3642, -27.8493),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m57_m29)',
        'url': _HF_BASE + 'corrientes_m57_m29.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-56.7076, -29.1139, -56.3655, -28.4288),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m57_m29_5)',
        'url': _HF_BASE + 'corrientes_m57_m29_5.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-56.6068, -29.106, -56.4978, -28.9886),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m58_25_m29_25)',
        'url': _HF_BASE + 'corrientes_m58_25_m29_25.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-58.3325, -29.2726, -57.9513, -28.9635),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m58_25_m29_5)',
        'url': _HF_BASE + 'corrientes_m58_25_m29_5.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-58.2868, -29.5159, -57.9339, -29.2115),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m58_25_m29_75)',
        'url': _HF_BASE + 'corrientes_m58_25_m29_75.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-58.2923, -29.7733, -57.9715, -29.4788),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m58_25_m30)',
        'url': _HF_BASE + 'corrientes_m58_25_m30.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-58.2732, -30.018, -57.9861, -29.6939),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m58_375_m29_625)',
        'url': _HF_BASE + 'corrientes_m58_375_m29_625.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-58.3889, -29.6309, -58.237, -29.4703),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m58_375_m29_75)',
        'url': _HF_BASE + 'corrientes_m58_375_m29_75.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-58.9107, -30.2115, -57.6396, -29.0446),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m58_5_m27_5)',
        'url': _HF_BASE + 'corrientes_m58_5_m27_5.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-58.63, -27.468, -57.9119, -27.2643),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m58_5_m28)',
        'url': _HF_BASE + 'corrientes_m58_5_m28.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-58.5282, -27.9944, -58.3039, -27.7565),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m58_5_m29)',
        'url': _HF_BASE + 'corrientes_m58_5_m29.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-58.5186, -29.0254, -57.9594, -28.7004),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m58_5_m29_25)',
        'url': _HF_BASE + 'corrientes_m58_5_m29_25.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-58.6018, -29.3184, -58.0222, -28.9031),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m58_5_m29_5)',
        'url': _HF_BASE + 'corrientes_m58_5_m29_5.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-58.5213, -29.5202, -58.2296, -29.1882),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m58_5_m29_625)',
        'url': _HF_BASE + 'corrientes_m58_5_m29_625.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-58.5213, -29.6327, -58.3474, -29.4726),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m58_5_m29_75)',
        'url': _HF_BASE + 'corrientes_m58_5_m29_75.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-58.5883, -29.7538, -58.3336, -29.6137),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m58_5_m30)',
        'url': _HF_BASE + 'corrientes_m58_5_m30.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-58.5828, -30.0358, -58.2287, -29.7241),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m58_5_m30_5)',
        'url': _HF_BASE + 'corrientes_m58_5_m30_5.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-58.5775, -30.6406, -57.9623, -29.978),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m58_5_m31)',
        'url': _HF_BASE + 'corrientes_m58_5_m31.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-58.056, -30.598, -57.9897, -30.4854),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m58_m27_5)',
        'url': _HF_BASE + 'corrientes_m58_m27_5.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-58.0552, -27.5393, -57.4863, -27.2706),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m58_m28)',
        'url': _HF_BASE + 'corrientes_m58_m28.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-57.6919, -27.8077, -57.4583, -27.4773),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m58_m29)',
        'url': _HF_BASE + 'corrientes_m58_m29.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-58.1199, -29.0384, -57.2142, -28.5107),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m58_m29_5)',
        'url': _HF_BASE + 'corrientes_m58_m29_5.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-58.1529, -29.7447, -57.4909, -28.9213),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m58_m30)',
        'url': _HF_BASE + 'corrientes_m58_m30.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-58.034, -30.0313, -57.5102, -29.4589),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m58_m30_25)',
        'url': _HF_BASE + 'corrientes_m58_m30_25.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-58.0336, -30.2888, -57.7015, -29.9558),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m58_m30_5)',
        'url': _HF_BASE + 'corrientes_m58_m30_5.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-58.0635, -30.5145, -57.7256, -30.2147),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m58_m31)',
        'url': _HF_BASE + 'corrientes_m58_m31.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-58.0251, -30.721, -57.7292, -30.3617),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m59_25_m29_25)',
        'url': _HF_BASE + 'corrientes_m59_25_m29_25.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-59.2593, -29.2542, -58.9805, -28.9582),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m59_25_m29_5)',
        'url': _HF_BASE + 'corrientes_m59_25_m29_5.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-59.4579, -29.8431, -58.788, -29.0847),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m59_5_m29)',
        'url': _HF_BASE + 'corrientes_m59_5_m29.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-59.2386, -29.0951, -58.9551, -28.7688),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m59_5_m29_25)',
        'url': _HF_BASE + 'corrientes_m59_5_m29_25.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-59.4927, -29.2778, -59.232, -29.088),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m59_5_m29_5)',
        'url': _HF_BASE + 'corrientes_m59_5_m29_5.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-59.6652, -30.0475, -59.1878, -29.0693),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m59_5_m30)',
        'url': _HF_BASE + 'corrientes_m59_5_m30.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-59.6548, -30.0173, -58.983, -29.1613),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m59_5_m30_5)',
        'url': _HF_BASE + 'corrientes_m59_5_m30_5.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-59.1262, -30.2734, -58.9341, -30.0067),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m59_m27_5)',
        'url': _HF_BASE + 'corrientes_m59_m27_5.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-58.857, -27.5175, -58.395, -27.2807),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m59_m28)',
        'url': _HF_BASE + 'corrientes_m59_m28.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-59.093, -28.2026, -58.3469, -27.4626),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m59_m28_5)',
        'url': _HF_BASE + 'corrientes_m59_m28_5.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-59.0592, -28.2234, -58.5757, -27.9851),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m59_m29)',
        'url': _HF_BASE + 'corrientes_m59_m29.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-59.0183, -29.0187, -58.6631, -28.825),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m59_m29_5)',
        'url': _HF_BASE + 'corrientes_m59_m29_5.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-59.2144, -29.5317, -58.4587, -28.9414),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m59_m30)',
        'url': _HF_BASE + 'corrientes_m59_m30.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-59.047, -30.0199, -58.481, -29.4581),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m59_m30_5)',
        'url': _HF_BASE + 'corrientes_m59_m30_5.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-59.113, -30.2692, -58.4451, -29.6303),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m60_m29_5)',
        'url': _HF_BASE + 'corrientes_m60_m29_5.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-59.5941, -29.49, -59.491, -29.265),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m60_m30)',
        'url': _HF_BASE + 'corrientes_m60_m30.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-59.6277, -29.7425, -59.5017, -29.4886),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    {
        'nombre': 'Corrientes (m60_m30_5)',
        'url': _HF_BASE + 'corrientes_m60_m30_5.geojson?download=true',
        'escala': '1:50.000',
        'confiabilidad': 'alta',
        'bbox': (-59.5883, -30.0411, -59.5575, -30.0),
        'simplificar_grados': None,  # sin pre-simplificar -- si pesa mucho en el mapa, ajustar acá
    },
    # Corrientes 1:100.000 -- respaldo de escala intermedia (menos detalle que
    # la grilla de arriba, pero más que la carta nacional).
    {
        'nombre': 'Corrientes (1:100.000)',
        'url': _HF_BASE + 'corrientes_1_100000.geojson?download=true',
        'escala': '1:100.000',
        'confiabilidad': 'alta',
        'bbox': (-59.7117, -30.434, -57.2734, -27.6376),
        'simplificar_grados': None,
    },
    # Corrientes 1:500.000 -- último respaldo antes de caer a la carta
    # nacional 1:1.000.000 (menos detalle, pero mejor que nada).
    {
        'nombre': 'Corrientes (1:500.000)',
        'url': _HF_BASE + 'corrientes_1_500000.geojson?download=true',
        'escala': '1:500.000',
        'confiabilidad': 'media',
        'bbox': (-59.0837, -30.7221, -56.4607, -27.4039),
        'simplificar_grados': None,
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
    # (Las 3 capas remotas WFS de Corrientes 1:100.000 que estaban acá --
    # Esquina, Esquina/Goya/Lavalle, Llanura Arenosa -- se sacaron: la
    # carta provincial completa autoalojada ya cubre esa misma zona con
    # más detalle, ver el bloque 'Corrientes (...)' más arriba.)
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
# NACIONAL, PARTIDO POR PROVINCIA (1:500.000, último recurso)
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
        'url': _HF_BASE + 'nacional_1_500000_buenos_aires.geojson?download=true',
        'escala': '1:500.000',
        'confiabilidad': 'baja',
        'bbox': (-63.4425, -41.0971, -56.6181, -33.2293),
        'simplificar_grados': None,
    },
    {
        'nombre': 'Nacional — Catamarca',
        'url': _HF_BASE + 'nacional_1_500000_catamarca.geojson?download=true',
        'escala': '1:500.000',
        'confiabilidad': 'baja',
        'bbox': (-69.1494, -30.1588, -64.733, -25.1144),
        'simplificar_grados': None,
    },
    {
        'nombre': 'Nacional — Chaco',
        'url': _HF_BASE + 'nacional_1_500000_chaco.geojson?download=true',
        'escala': '1:500.000',
        'confiabilidad': 'baja',
        'bbox': (-63.4755, -28.0748, -57.5169, -22.4012),
        'simplificar_grados': None,
    },
    {
        'nombre': 'Nacional — Chubut',
        'url': _HF_BASE + 'nacional_1_500000_chubut.geojson?download=true',
        'escala': '1:500.000',
        'confiabilidad': 'baja',
        'bbox': (-72.2372, -46.0573, -63.5306, -41.9462),
        'simplificar_grados': None,
    },
    {
        'nombre': 'Nacional — Cordoba',
        'url': _HF_BASE + 'nacional_1_500000_cordoba.geojson?download=true',
        'escala': '1:500.000',
        'confiabilidad': 'baja',
        'bbox': (-65.8325, -35.0535, -61.7258, -29.457),
        'simplificar_grados': None,
    },
    {
        'nombre': 'Nacional — Corrientes',
        'url': _HF_BASE + 'nacional_1_500000_corrientes.geojson?download=true',
        'escala': '1:500.000',
        'confiabilidad': 'baja',
        'bbox': (-59.7218, -30.7721, -55.574, -27.2147),
        'simplificar_grados': None,
    },
    {
        'nombre': 'Nacional — Entre Rios',
        'url': _HF_BASE + 'nacional_1_500000_entre_rios.geojson?download=true',
        'escala': '1:500.000',
        'confiabilidad': 'baja',
        'bbox': (-60.8267, -34.0747, -57.7514, -30.1091),
        'simplificar_grados': None,
    },
    {
        'nombre': 'Nacional — Formosa',
        'url': _HF_BASE + 'nacional_1_500000_formosa.geojson?download=true',
        'escala': '1:500.000',
        'confiabilidad': 'baja',
        'bbox': (-62.4015, -26.9175, -57.5238, -22.4069),
        'simplificar_grados': None,
    },
    {
        'nombre': 'Nacional — Jujuy',
        'url': _HF_BASE + 'nacional_1_500000_jujuy.geojson?download=true',
        'escala': '1:500.000',
        'confiabilidad': 'baja',
        'bbox': (-67.2739, -24.6803, -64.1165, -21.7344),
        'simplificar_grados': None,
    },
    {
        'nombre': 'Nacional — La Pampa',
        'url': _HF_BASE + 'nacional_1_500000_la_pampa.geojson?download=true',
        'escala': '1:500.000',
        'confiabilidad': 'baja',
        'bbox': (-68.3507, -39.3748, -63.3284, -34.948),
        'simplificar_grados': None,
    },
    {
        'nombre': 'Nacional — La Rioja',
        'url': _HF_BASE + 'nacional_1_500000_la_rioja.geojson?download=true',
        'escala': '1:500.000',
        'confiabilidad': 'baja',
        'bbox': (-69.6988, -32.0282, -65.3664, -27.6309),
        'simplificar_grados': None,
    },
    {
        'nombre': 'Nacional — Mendoza',
        'url': _HF_BASE + 'nacional_1_500000_mendoza.geojson?download=true',
        'escala': '1:500.000',
        'confiabilidad': 'baja',
        'bbox': (-70.6315, -37.627, -66.4291, -31.9498),
        'simplificar_grados': None,
    },
    {
        'nombre': 'Nacional — Misiones',
        'url': _HF_BASE + 'nacional_1_500000_misiones.geojson?download=true',
        'escala': '1:500.000',
        'confiabilidad': 'baja',
        'bbox': (-56.1082, -28.2116, -53.59, -25.4473),
        'simplificar_grados': None,
    },
    {
        'nombre': 'Nacional — Neuquen',
        'url': _HF_BASE + 'nacional_1_500000_neuquen.geojson?download=true',
        'escala': '1:500.000',
        'confiabilidad': 'baja',
        'bbox': (-71.9919, -41.1677, -67.9781, -36.0592),
        'simplificar_grados': None,
    },
    {
        'nombre': 'Nacional — Rio Negro',
        'url': _HF_BASE + 'nacional_1_500000_rio_negro.geojson?download=true',
        'escala': '1:500.000',
        'confiabilidad': 'baja',
        'bbox': (-71.9815, -42.0569, -62.734, -37.5172),
        'simplificar_grados': None,
    },
    {
        'nombre': 'Nacional — Salta',
        'url': _HF_BASE + 'nacional_1_500000_salta.geojson?download=true',
        'escala': '1:500.000',
        'confiabilidad': 'baja',
        'bbox': (-68.6129, -26.4368, -62.2877, -21.947),
        'simplificar_grados': None,
    },
    {
        'nombre': 'Nacional — San Juan',
        'url': _HF_BASE + 'nacional_1_500000_san_juan.geojson?download=true',
        'escala': '1:500.000',
        'confiabilidad': 'baja',
        'bbox': (-70.6492, -32.6773, -66.614, -28.3554),
        'simplificar_grados': None,
    },
    {
        'nombre': 'Nacional — San Luis',
        'url': _HF_BASE + 'nacional_1_500000_san_luis.geojson?download=true',
        'escala': '1:500.000',
        'confiabilidad': 'baja',
        'bbox': (-67.5338, -36.0544, -64.8261, -31.7898),
        'simplificar_grados': None,
    },
    {
        'nombre': 'Nacional — Santa Cruz',
        'url': _HF_BASE + 'nacional_1_500000_santa_cruz.geojson?download=true',
        'escala': '1:500.000',
        'confiabilidad': 'baja',
        'bbox': (-73.6102, -52.4334, -65.6799, -45.9461),
        'simplificar_grados': None,
    },
    {
        'nombre': 'Nacional — Santa Fe',
        'url': _HF_BASE + 'nacional_1_500000_santa_fe.geojson?download=true',
        'escala': '1:500.000',
        'confiabilidad': 'baja',
        'bbox': (-62.9309, -34.4359, -57.9898, -27.9472),
        'simplificar_grados': None,
    },
    {
        'nombre': 'Nacional — Santiago Del Estero',
        'url': _HF_BASE + 'nacional_1_500000_santiago_del_estero.geojson?download=true',
        'escala': '1:500.000',
        'confiabilidad': 'baja',
        'bbox': (-65.2268, -30.5286, -61.6584, -25.6008),
        'simplificar_grados': None,
    },
    {
        'nombre': 'Nacional — Tierra Del Fuego E Islas Malvinas',
        'url': _HF_BASE + 'nacional_1_500000_tierra_del_fuego_e_islas_malvinas.geojson?download=true',
        'escala': '1:500.000',
        'confiabilidad': 'baja',
        'bbox': (-68.6726, -59.148, -27.2006, -50.9729),
        'simplificar_grados': None,
    },
    {
        'nombre': 'Nacional — Tucuman',
        'url': _HF_BASE + 'nacional_1_500000_tucuman.geojson?download=true',
        'escala': '1:500.000',
        'confiabilidad': 'baja',
        'bbox': (-66.2117, -28.0547, -64.421, -26.0149),
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
    # 'cu_s1' (Corrientes, componente de suelo DOMINANTE del polígono --
    # la propia carta lo lista primero por tener el mayor % de superficie,
    # ver porcent_s1) viene en el mismo formato "romano+letras" que ya usa
    # Buenos Aires (ej. 'VIIws', 'IIIes'), así que cae directo en el mismo
    # parser de abajo (_parse_cap_uso_romano) sin necesitar uno nuevo.
    cap_uso = _g(props, 'cap_uso', 'capacidad_uso', 'Cap Uso', 'cu_s1')
    drenaje_crudo = _g(props, 'drenaje_estimado', 'drenaje', 'drenaje_s1')
    # 'Ipc' (Valle de Lerma / Valles Calchaquíes, Salta) es un índice de
    # productividad numérico igual que 'IP' de Córdoba, solo con otro
    # nombre de columna -- se agrega como alias explícito por el mismo
    # motivo que 'Cap Uso' arriba: _g() no matchea nombres distintos.
    # 'ip_s1' (Corrientes) es el mismo concepto, del componente dominante.
    indice_prod = _g(props, 'ind_prod', 'indice_prod', 'IP', 'Ipc', 'ip_s1')

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

    limitante = _g(props, 'limit_ppal', 'limitante', 'limit_s1') or _limitante_desde_subclase(subclase)
    anegabilidad = _g(props, 'anegab_s1', 'anegabilidad') or _anegabilidad_desde_drenaje(drenaje_crudo)
    if anegabilidad is None and subclase and 'w' in str(subclase).lower():
        anegabilidad = 'Probable (limitante por anegamiento/drenaje según carta GAT)'
    # Santa Fe: 'bajo'='1' marca explícitamente zona baja/anegable en la
    # cartografía original, más allá de lo que diga la subclase.
    if anegabilidad is None and str(_g(props, 'bajo') or '') == '1':
        anegabilidad = 'Probable (zona baja según carta de Santa Fe)'
    # Corrientes trae 3 flags binarios (0/1) explícitos en vez de una
    # descripción de texto -- mismo criterio que 'bajo' de Santa Fe arriba:
    # si la carta ya lo marca a nivel de polígono, se refleja tal cual, sin
    # inferir nada adicional.
    if anegabilidad is None and str(_g(props, 'inundables') or '') == '1':
        anegabilidad = 'Probable (zona inundable según carta de Corrientes)'
    if anegabilidad is None and str(_g(props, 'anegables') or '') == '1':
        anegabilidad = 'Probable (zona anegable según carta de Corrientes)'
    if anegabilidad is None and str(_g(props, 'encharcabl') or '') == '1':
        anegabilidad = 'Probable (riesgo de encharcamiento según carta de Corrientes)'

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
def _escala_a_numero(escala_txt):
    """'1:50.000' -> 50000; '1:500.000' -> 500000. Devuelve None si no puede
    parsearlo (nunca inventa un número -- el frontend ya maneja fuente=None
    con su propio texto genérico de respaldo)."""
    if not escala_txt:
        return None
    try:
        parte = str(escala_txt).split(':')[-1]  # '1:50.000' -> '50.000'
        return int(parte.replace('.', '').replace(',', ''))
    except (ValueError, TypeError):
        return None


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
                # FIX: acá estaba el bug de "siempre dice 1:500.000 en la
                # app aunque el dato haya salido de una carta 1:50.000" --
                # antes 'fuente' era directamente capa['nombre'] (un texto
                # plano, ej. "Corrientes (m58_m29_5)"). El frontend (tanto
                # en Aptitud Productiva como en el Inspector de Suelos)
                # SIEMPRE esperó un OBJETO con .provincia/.depto/.escala
                # (ver 'fuenteInfo.escala?.toLocaleString(...)' del lado
                # JS) -- como un string no tiene esas propiedades, siempre
                # caía en el texto genérico de respaldo "1:500.000",
                # sin importar qué carta hubiera respondido en realidad.
                return {
                    'encontrado': True,
                    'fuente': {
                        'provincia': _provincia_de_capa_detalle(capa) or capa['nombre'],
                        'depto': None,  # nuestras cartas no tienen división por departamento -- si en el futuro se agrega, va acá
                        'escala': _escala_a_numero(capa.get('escala')),
                    },
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
# complementa con el tile Nacional (1:500.000) de esa misma provincia,
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
                            f'cubre ~{cobertura:.0%} del área) con la carta Nacional 1:500.000 '
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
