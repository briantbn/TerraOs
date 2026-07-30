# Compilar LISFLOOD-FP en Render

Este directorio (`lisflood_src/`) trae el código fuente de LISFLOOD-FP
(versión BMI de la Universidad de Bristol / openearth, licencia GPL-3.0)
vendorizado dentro del repo — no se descarga nada externo en el deploy,
solo se compila.

## Build Command en Render

En el dashboard de Render → tu servicio → Settings → Build Command,
reemplazá el que tengas por:

```
pip install -r requirements.txt && cd lisflood_src && make && cd .. && mkdir -p bin && cp lisflood_src/lisflood bin/lisflood && chmod +x bin/lisflood
```

Esto:
1. Instala las dependencias Python de siempre.
2. Compila LISFLOOD-FP con `make` (genera `lisflood_src/lisflood`).
3. Copia el binario a `bin/lisflood` en la raíz del repo — ahí es donde
   `lisflood_engine.py` lo busca por defecto (`RUTA_BINARIO`).

## Si el build falla por falta de g++/make

Probado y confirmado que compila limpio con `g++` 13 / GNU Make en un
entorno Ubuntu 24.04 estándar (warnings, pero compila y linkea sin
errores). Si el entorno de build de Render no tiene `g++`/`make`
preinstalados (algunos planes "Native Environment" restringen esto),
las opciones son:

- **Opción A** — cambiar el servicio a deploy con **Docker** en Render
  (permite un Dockerfile con `apt-get install -y g++ make` sin
  restricciones), o
- **Opción B** — pedirle a Render soporte que confirme si `build-essential`
  está disponible en el plan actual.

## Variable de entorno opcional

Si por algún motivo el binario termina en otra ruta, se puede forzar con:

```
LISFLOOD_BIN=/ruta/al/binario/lisflood
```

## Verificar que compiló bien

Después del deploy, en el Shell de Render (`Shell` en el menú lateral):

```
ls -la bin/lisflood
./bin/lisflood
```

Si imprime el banner de LISFLOOD-FP (`LISFLOOD-FP version 5.99.0`), está
todo OK. `/inundacion_dinamica` debería dejar de devolver 503.
