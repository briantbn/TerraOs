"""
soc_incendios_analisis.py
─────────────────────────────────────────────────────────────────────────
Análisis exploratorio: ¿los incendios en Argentina muestran criticidad
auto-organizada (SOC)?

La firma clásica de SOC en incendios (Malamud, Morein & Turcotte, 1998,
"Forest fires: an example of self-organized critical behavior", Science)
es que la distribución de TAMAÑOS de incendio (frecuencia vs. superficie
quemada) sigue una LEY DE POTENCIA en varios órdenes de magnitud, sin una
escala característica — como las avalanchas del modelo de pila de arena
de Bak-Tang-Wiesenfeld.

Para eso hace falta el tamaño de cada incendio INDIVIDUAL, no totales
agregados por año/provincia (que es todo lo que ofrece datos.gob.ar).
Este script usa detecciones satelitales de foco de calor de NASA FIRMS
(MODIS, cobertura 2000-presente) — cada detección es un punto individual
con fecha — y las agrupa en "eventos" de incendio discretos mediante
clustering espacio-temporal (DBSCAN sobre distancia + tiempo combinados).

Pipeline:
    1. descargar_firms_argentina()   — baja y cachea los CSV anuales de FIRMS
    2. cargar_focos()                — junta todos los años en un DataFrame
    3. clusterizar_eventos()         — agrupa focos cercanos en el tiempo/
                                        espacio en un mismo "incendio"
    4. calcular_tamano_eventos()     — área aproximada de cada evento
    5. ajustar_ley_potencia()        — ajuste MLE (Clauset-Shalizi-Newman,
                                        vía el paquete `powerlaw`) + test
                                        de bondad de ajuste contra
                                        alternativas (lognormal, exponencial)
    6. graficar_ccdf()               — gráfico log-log con el ajuste

Requiere: pandas, numpy, scikit-learn, powerlaw, matplotlib, requests
    pip install pandas numpy scikit-learn powerlaw matplotlib requests

Uso:
    python soc_incendios_analisis.py --anio-inicio 2005 --anio-fin 2024
    python soc_incendios_analisis.py --self-test   # valida el pipeline
                                                     # con datos sintéticos,
                                                     # sin descargar nada
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests
from sklearn.cluster import DBSCAN

# Radio de 111km/grado aprox a la latitud media de Argentina (~-35°) —
# suficiente para esta primera pasada exploratoria; no corrige por
# proyección, es una aproximación (igual que el resto de bbox_desde_centro
# en el resto del proyecto).
KM_POR_GRADO = 111.0

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache_firms')

# ─────────────────────────────────────────────────────────────────────
#  Descarga vía API de FIRMS (NASA)
# ─────────────────────────────────────────────────────────────────────
# NASA discontinuó la descarga estática por país en .zip que usaba la
# primera versión de este script (esos archivos daban 404 — la descarga
# "yearly summary by country" ahora exige login/email y llega por correo,
# ya no es un archivo predecible por URL). La alternativa que SÍ sigue
# siendo un endpoint HTTP directo, sin login ni espera por email, es la
# API pública de FIRMS (requiere una MAP_KEY gratuita, ver
# https://firms.modaps.eosdis.nasa.gov/api/map_key/ — el registro tarda
# dos minutos y no hace falta esperar ningún email de vuelta).
#
# Endpoint usado: /api/country/csv/[MAP_KEY]/[SENSOR]/[ISO3]/[DIAS]/[FECHA]
#   - ISO3 de Argentina: ARG
#   - SENSOR: 'MODIS_SP' (Standard Processing, calibrado, con ~1-4 meses
#     de rezago — es el que corresponde para un análisis histórico serio,
#     cubre desde 2000-11-01). 'MODIS_NRT' es para los meses más recientes
#     que SP todavía no cubre.
#   - DIAS: cuántos días de datos devuelve esa consulta (partimos en
#     bloques de a 10 para no pasarnos de una sola vez — no hay un tope
#     "oficial" documentado con un número exacto, pero 10 es el que usan
#     los ejemplos de NASA y es un tamaño de bloque conservador y seguro).
#   - FECHA: fecha de INICIO del bloque de `dias` días.
#
# Límite de cuota: 5000 "transacciones" cada 10 minutos por MAP_KEY (no
# es 1 transacción = 1 request; una consulta con más días/más datos puede
# contar como varias). El script chequea la cuota restante cada tantas
# consultas (mapkey_status) y hace una pausa si se está por quedar sin
# margen, en vez de romper a mitad de camino.
BASE_API_FIRMS = 'https://firms.modaps.eosdis.nasa.gov/api'
ISO3_ARGENTINA = 'ARG'


def verificar_map_key(map_key):
    """Chequea que la MAP_KEY sea válida y cuánta cuota de transacciones
    queda en la ventana actual de 10 minutos."""
    url = f'{BASE_API_FIRMS.replace("/api", "")}/mapserver/mapkey_status/?MAP_KEY={map_key}'
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    datos = r.json()
    if 'transaction_limit' not in datos:
        raise RuntimeError(f'MAP_KEY inválida o respuesta inesperada: {datos}')
    return datos


def _pausar_si_hace_falta(map_key, umbral=0.85, segundos_ventana=600):
    """Si ya se usó más del `umbral` de la cuota de 10 minutos, espera a
    que la ventana se reinicie en vez de arriesgarse a que NASA empiece a
    devolver error 429 (límite excedido) a mitad de la descarga."""
    try:
        estado = verificar_map_key(map_key)
    except Exception as exc:  # noqa: BLE001
        print(f'[FIRMS] No se pudo chequear la cuota ({exc}) — se sigue igual, con cautela.')
        return
    usado = estado['current_transactions'] / estado['transaction_limit']
    if usado >= umbral:
        print(f'[FIRMS] Cuota casi agotada ({estado["current_transactions"]}/'
              f'{estado["transaction_limit"]}) — pausando {segundos_ventana}s '
              f'a que se reinicie la ventana de 10 minutos…')
        time.sleep(segundos_ventana)


def descargar_firms_argentina(fecha_inicio, fecha_fin, map_key, sensor='MODIS_SP',
                               dias_por_bloque=10, pausa_seg=1.0, cache_dir=CACHE_DIR):
    """Descarga (con caché local en disco, por bloque de fechas) los
    focos de calor de Argentina entre `fecha_inicio` y `fecha_fin`
    (objetos date/Timestamp), usando la API de FIRMS."""
    os.makedirs(cache_dir, exist_ok=True)
    rutas = []

    fecha_actual = pd.Timestamp(fecha_inicio)
    fecha_fin = pd.Timestamp(fecha_fin)
    n_bloques_desde_ultimo_chequeo = 0

    while fecha_actual <= fecha_fin:
        etiqueta = fecha_actual.strftime('%Y-%m-%d')
        ruta_local = os.path.join(
            cache_dir, f'{sensor}_{etiqueta}_{dias_por_bloque}d.csv',
        )
        if os.path.isfile(ruta_local):
            rutas.append(ruta_local)
            fecha_actual += pd.Timedelta(days=dias_por_bloque)
            continue

        if n_bloques_desde_ultimo_chequeo >= 20:
            _pausar_si_hace_falta(map_key)
            n_bloques_desde_ultimo_chequeo = 0

        url = (f'{BASE_API_FIRMS}/country/csv/{map_key}/{sensor}/{ISO3_ARGENTINA}/'
               f'{dias_por_bloque}/{etiqueta}')
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            # Si la MAP_KEY es inválida o se excedió la cuota, FIRMS
            # devuelve texto plano de error (no un 4xx/5xx siempre) — lo
            # detectamos porque no tiene la primera columna esperada.
            primera_linea = r.text.split('\n', 1)[0]
            if 'latitude' not in primera_linea.lower():
                print(f'[FIRMS] {etiqueta}: respuesta inesperada ({primera_linea[:120]!r}) — se omite.')
                fecha_actual += pd.Timedelta(days=dias_por_bloque)
                continue
        except requests.RequestException as exc:
            print(f'[FIRMS] {etiqueta}: falló la descarga ({exc}) — se omite ese bloque.')
            fecha_actual += pd.Timedelta(days=dias_por_bloque)
            continue

        with open(ruta_local, 'w', encoding='utf-8') as f:
            f.write(r.text)
        rutas.append(ruta_local)
        print(f'[FIRMS] {etiqueta} (+{dias_por_bloque}d): OK ({len(r.text.splitlines())} líneas)')

        n_bloques_desde_ultimo_chequeo += 1
        time.sleep(pausa_seg)
        fecha_actual += pd.Timedelta(days=dias_por_bloque)

    return rutas


def cargar_focos(rutas_csv):
    """Junta todos los CSV anuales en un único DataFrame con lat/lon/fecha/frp."""
    marcos = []
    for ruta in rutas_csv:
        df = pd.read_csv(ruta)
        # Nombres de columna estándar de FIRMS: latitude, longitude,
        # acq_date, acq_time, confidence, frp (Fire Radiative Power, MW).
        columnas = {c.lower(): c for c in df.columns}
        df = df.rename(columns={
            columnas.get('latitude', 'latitude'): 'lat',
            columnas.get('longitude', 'longitude'): 'lon',
            columnas.get('acq_date', 'acq_date'): 'fecha',
            columnas.get('frp', 'frp'): 'frp',
        })
        marcos.append(df[['lat', 'lon', 'fecha', 'frp']] if 'frp' in df.columns
                       else df[['lat', 'lon', 'fecha']].assign(frp=np.nan))

    focos = pd.concat(marcos, ignore_index=True)
    focos['fecha'] = pd.to_datetime(focos['fecha'])
    return focos.dropna(subset=['lat', 'lon', 'fecha'])


def clusterizar_eventos(focos, radio_km=3.0, ventana_dias=4, min_focos=1):
    """Agrupa detecciones individuales en eventos de incendio discretos.

    DBSCAN sobre un espacio de features (lat, lon, tiempo) donde el
    tiempo se reescala a "equivalente en km" para que una sola métrica
    euclídea combine ambas dimensiones: dos focos quedan en el mismo
    evento si están a menos de `radio_km` Y ocurrieron dentro de
    `ventana_dias`. Esto reconstruye, a partir de detecciones puntuales
    diarias, la extensión espacio-temporal real de cada incendio.

    min_focos: focos mínimos para considerar el cluster un "evento" real
    y no ruido de una sola detección aislada (DBSCAN usa min_samples).
    """
    focos = focos.reset_index(drop=True).copy()
    lat_rad = np.radians(focos['lat'].to_numpy())
    x_km = focos['lon'].to_numpy() * KM_POR_GRADO * np.cos(lat_rad)
    y_km = focos['lat'].to_numpy() * KM_POR_GRADO

    t0 = focos['fecha'].min()
    dias = (focos['fecha'] - t0).dt.total_seconds().to_numpy() / 86400.0
    # Reescalamos los días a "km equivalentes" usando la misma escala
    # (radio_km / ventana_dias) para que DBSCAN con eps=radio_km trate
    # ambas dimensiones de forma consistente.
    escala_tiempo = radio_km / max(ventana_dias, 1e-6)
    t_km = dias * escala_tiempo

    X = np.column_stack([x_km, y_km, t_km])
    labels = DBSCAN(eps=radio_km, min_samples=min_focos).fit_predict(X)

    focos['evento_id'] = labels
    return focos[focos['evento_id'] != -1]  # -1 = ruido (focos aislados sin evento)


def calcular_tamano_eventos(focos_clusterizados, resolucion_km=1.0):
    """Área aproximada de cada evento: conteo de celdas ~1km únicas
    ocupadas por detecciones, no el hull convexo (más robusto cuando las
    detecciones son dispersas o el incendio tiene forma irregular — un
    hull convexo sobrestima mucho en esos casos)."""
    df = focos_clusterizados.copy()
    df['celda_x'] = (df['lon'] * KM_POR_GRADO / resolucion_km).round().astype(int)
    df['celda_y'] = (df['lat'] * KM_POR_GRADO / resolucion_km).round().astype(int)

    por_evento = df.groupby('evento_id').agg(
        n_focos=('lat', 'size'),
        n_dias=('fecha', lambda s: (s.max() - s.min()).days + 1),
        frp_total=('frp', 'sum'),
        celdas_unicas=('celda_x', lambda s: len(set(zip(s, df.loc[s.index, 'celda_y'])))),
    )
    por_evento['area_km2'] = por_evento['celdas_unicas'] * (resolucion_km ** 2)
    return por_evento.reset_index()


@dataclass
class ResultadoAjuste:
    exponente: float
    xmin: float
    n_eventos_usados: int
    n_eventos_totales: int
    favorece_lognormal: bool
    p_valor_comparacion: float


def ajustar_ley_potencia(tamanos, discreto=False):
    """Ajuste MLE de ley de potencia (Clauset-Shalizi-Newman) vía el
    paquete `powerlaw`, con test de razón de verosimilitud contra una
    lognormal (la alternativa más común que puede confundirse con una
    ley de potencia en muestras chicas)."""
    import powerlaw

    tamanos = np.asarray(tamanos, dtype=float)
    tamanos = tamanos[tamanos > 0]

    fit = powerlaw.Fit(tamanos, discrete=discreto, verbose=False)
    R, p = fit.distribution_compare('power_law', 'lognormal')
    # R > 0 favorece power_law sobre lognormal; R < 0, al revés.
    # p bajo (<0.1) = la comparación es estadísticamente significativa.
    favorece_lognormal = R < 0

    return ResultadoAjuste(
        exponente=fit.power_law.alpha,
        xmin=fit.power_law.xmin,
        n_eventos_usados=int((tamanos >= fit.power_law.xmin).sum()),
        n_eventos_totales=len(tamanos),
        favorece_lognormal=favorece_lognormal,
        p_valor_comparacion=p,
    ), fit


def graficar_ccdf(fit, ruta_salida):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5.5))
    fit.plot_ccdf(ax=ax, color='#1a3b22', linewidth=0, marker='o', markersize=4,
                  label='Datos (CCDF empírica)')
    fit.power_law.plot_ccdf(ax=ax, color='#b91c1c', linestyle='--',
                             label=f'Ley de potencia (α={fit.power_law.alpha:.2f}, '
                                   f'xmin={fit.power_law.xmin:.2f})')
    ax.set_xlabel('Área del evento (km²)')
    ax.set_ylabel('P(Área ≥ x)  —  CCDF')
    ax.set_title('Distribución de tamaños de incendios — Argentina (FIRMS/MODIS)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(ruta_salida, dpi=150)
    print(f'[Gráfico] Guardado en {ruta_salida}')


# ─────────────────────────────────────────────────────────────────────
#  Self-test: valida el pipeline completo con datos SINTÉTICOS (sin
#  descargar nada) — genera "eventos" con tamaños que siguen una ley de
#  potencia conocida, los desparrama en focos individuales simulados, y
#  confirma que clusterizar_eventos() + ajustar_ley_potencia() recuperan
#  un exponente cercano al usado para generarlos. Sirve para confiar en
#  el código antes de gastar tiempo bajando ~25 años de FIRMS de verdad.
# ─────────────────────────────────────────────────────────────────────
def _self_test(alpha_verdadero=2.2, n_eventos=2000, semilla=42):
    rng = np.random.default_rng(semilla)

    # Ley de potencia truncada: tamaños de evento (en "número de focos")
    # vía muestreo por transformada inversa, xmin=1.
    xmin = 1.0
    u = rng.uniform(0, 1, n_eventos)
    tamanos_evento = xmin * (1 - u) ** (-1.0 / (alpha_verdadero - 1.0))
    tamanos_evento = np.clip(tamanos_evento.round().astype(int), 1, 500)

    filas = []
    t0 = pd.Timestamp('2010-01-01')
    for i, n_focos in enumerate(tamanos_evento):
        # Centro del evento en algún punto de Argentina (bbox burdo)
        centro_lat = rng.uniform(-52, -22)
        centro_lon = rng.uniform(-72, -58)
        centro_dia = rng.integers(0, 365 * 15)
        # Los focos de un mismo evento: dispersos ~sqrt(n_focos) celdas de
        # 1km (así el área crece con el tamaño, como un incendio real) y
        # en un rango de días proporcional a su tamaño (incendios grandes
        # duran más).
        radio_evento_km = max(0.5, np.sqrt(n_focos) * 0.6)
        duracion_dias = max(1, int(np.sqrt(n_focos)))
        for _ in range(n_focos):
            dlat = rng.normal(0, radio_evento_km / KM_POR_GRADO)
            dlon = rng.normal(0, radio_evento_km / KM_POR_GRADO)
            ddia = rng.integers(0, duracion_dias + 1)
            filas.append({
                'lat': centro_lat + dlat, 'lon': centro_lon + dlon,
                'fecha': t0 + pd.Timedelta(days=int(centro_dia + ddia)),
                'frp': rng.exponential(10),
            })

    focos_sinteticos = pd.DataFrame(filas)
    print(f'[Self-test] {len(focos_sinteticos)} focos sintéticos generados '
          f'a partir de {n_eventos} eventos (α verdadero = {alpha_verdadero}).')

    clusterizados = clusterizar_eventos(focos_sinteticos, radio_km=3.0, ventana_dias=4)
    eventos = calcular_tamano_eventos(clusterizados, resolucion_km=1.0)
    print(f'[Self-test] Clustering reconstruyó {len(eventos)} eventos '
          f'(de {n_eventos} originales — algo de fusión/pérdida es esperable).')

    resultado, fit = ajustar_ley_potencia(eventos['area_km2'].to_numpy(), discreto=True)
    print(f'[Self-test] Exponente recuperado: α={resultado.exponente:.2f} '
          f'(verdadero: {alpha_verdadero}) · xmin={resultado.xmin:.2f} · '
          f'n usados={resultado.n_eventos_usados}/{resultado.n_eventos_totales}')

    ok = abs(resultado.exponente - alpha_verdadero) < 0.6
    print('[Self-test] ' + ('✅ OK — el pipeline recupera un exponente razonable.'
                             if ok else '⚠️  El exponente recuperado se aleja bastante '
                                        'del verdadero — revisar antes de confiar en '
                                        'resultados con datos reales.'))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--map-key', help='MAP_KEY gratuita de FIRMS — sacala en '
                                       'https://firms.modaps.eosdis.nasa.gov/api/map_key/ '
                                       '(obligatoria salvo con --self-test)')
    ap.add_argument('--sensor', default='MODIS_SP',
                     choices=['MODIS_SP', 'MODIS_NRT', 'VIIRS_SNPP_SP', 'VIIRS_SNPP_NRT',
                              'VIIRS_NOAA20_SP', 'VIIRS_NOAA20_NRT'],
                     help='MODIS_SP = histórico calibrado (2000-presente, ~1-4 meses de '
                          'rezago) — el que corresponde para este análisis. Usar *_NRT '
                          'solo para cubrir los meses más recientes que SP no llegó a cubrir.')
    ap.add_argument('--fecha-inicio', default='2005-01-01', help='YYYY-MM-DD')
    ap.add_argument('--fecha-fin', default=None,
                     help='YYYY-MM-DD (default: hoy)')
    ap.add_argument('--dias-por-bloque', type=int, default=10,
                     help='Tamaño de cada consulta a la API (días). No hay un tope oficial '
                          'documentado con un número exacto; 10 es conservador y seguro.')
    ap.add_argument('--pausa-seg', type=float, default=1.0,
                     help='Pausa entre consultas sucesivas a la API (para no ráfaguear el servidor)')
    ap.add_argument('--radio-km', type=float, default=3.0, help='Radio de clustering espacial (km)')
    ap.add_argument('--ventana-dias', type=int, default=4, help='Ventana de clustering temporal (días)')
    ap.add_argument('--min-focos', type=int, default=1, help='Focos mínimos para contar como evento')
    ap.add_argument('--salida', default='eventos_incendio_argentina.csv')
    ap.add_argument('--grafico', default='ccdf_ley_potencia.png')
    ap.add_argument('--self-test', action='store_true',
                     help='Valida el pipeline con datos sintéticos, sin descargar nada')
    args = ap.parse_args()

    if args.self_test:
        ok = _self_test()
        sys.exit(0 if ok else 1)

    if not args.map_key:
        print('Falta --map-key. Sacá una gratis (2 minutos, sin espera por email) en:\n'
              '  https://firms.modaps.eosdis.nasa.gov/api/map_key/\n'
              'y volvé a correr: python soc_incendios_analisis.py --map-key TU_CLAVE_ACA')
        sys.exit(1)

    try:
        estado = verificar_map_key(args.map_key)
        print(f'[FIRMS] MAP_KEY OK — cuota: {estado["current_transactions"]}/'
              f'{estado["transaction_limit"]} usada en esta ventana de '
              f'{estado["transaction_interval"]}.')
    except Exception as exc:  # noqa: BLE001
        print(f'La MAP_KEY no parece válida ({exc}). Revisá que la copiaste completa, sin comillas.')
        sys.exit(1)

    fecha_fin = args.fecha_fin or pd.Timestamp.today().strftime('%Y-%m-%d')
    rutas = descargar_firms_argentina(
        args.fecha_inicio, fecha_fin, map_key=args.map_key, sensor=args.sensor,
        dias_por_bloque=args.dias_por_bloque, pausa_seg=args.pausa_seg,
    )
    if not rutas:
        print('No se pudo descargar ningún bloque — revisá la conexión o las fechas.')
        sys.exit(1)

    focos = cargar_focos(rutas)
    print(f'Total de focos cargados: {len(focos)}')

    clusterizados = clusterizar_eventos(focos, radio_km=args.radio_km,
                                         ventana_dias=args.ventana_dias, min_focos=args.min_focos)
    eventos = calcular_tamano_eventos(clusterizados)
    eventos.to_csv(args.salida, index=False)
    print(f'{len(eventos)} eventos reconstruidos — guardado en {args.salida}')

    resultado, fit = ajustar_ley_potencia(eventos['area_km2'].to_numpy(), discreto=True)
    print(f'\n── Resultado del ajuste ──')
    print(f'Exponente (α)         : {resultado.exponente:.3f}')
    print(f'xmin (a partir de qué tamaño vale la ley de potencia): {resultado.xmin:.2f} km²')
    print(f'Eventos usados en el ajuste: {resultado.n_eventos_usados} de {resultado.n_eventos_totales}')
    print(f'¿Se ajusta mejor una lognormal?: {"sí" if resultado.favorece_lognormal else "no"} '
          f'(p={resultado.p_valor_comparacion:.3f})')

    graficar_ccdf(fit, args.grafico)


if __name__ == '__main__':
    main()
