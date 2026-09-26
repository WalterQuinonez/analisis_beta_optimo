#!/bin/bash
#SBATCH --job-name=beta_grid_INL_1.6e-1_p100
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=2800
#SBATCH --partition=regular
#
# IMPORTANTE: el rango de --array tiene que cubrir todos los indices de
# BETAS (0 .. len(BETAS)-1). Si cambias la lista, actualiza tambien el
# --array de la linea de abajo.
#SBATCH --array=0-39
#
# Los logs quedan DENTRO de OUT_DIR (definido mas abajo). OJO: SLURM lee
# TODAS las directivas #SBATCH ANTES de ejecutar nada, y deja de buscarlas
# en cuanto encuentra la PRIMERA linea de codigo bash real (ej: OUT_DIR=...
# mas abajo) -las que vengan despues de esa linea quedan IGNORADAS EN
# SILENCIO, sin error ni warning (bash las trata como comentario normal y
# no rompe nada, pero SLURM nunca las aplica). Por eso --output/--error
# TIENEN que ir aca arriba, junto con el resto de #SBATCH, y no pueden leer
# la variable de bash $OUT_DIR -son rutas LITERALES que hay que mantener
# sincronizadas a mano con el OUT_DIR de mas abajo (y esa carpeta tiene que
# existir de antemano: `mkdir -p beta_grid_search_SP_INL_1.6e-1_p_100`
# antes de este sbatch).
#SBATCH --output=beta_grid_search_SP_INL_1.6e-1_p_100/slurm_beta_%A_%a.out
#SBATCH --error=beta_grid_search_SP_INL_1.6e-1_p_100/slurm_beta_%A_%a.err
#
# Job array: UN SOLO `sbatch train_job.sh` lanza tantas tareas como betas
# haya en BETAS, cada una un entrenamiento INDEPENDIENTE (su propio proceso,
# su propia asignacion de recursos, su propio log). Si una tarea se corta
# (tiempo/memoria/nodo caido) no afecta a las demas, y cada una hace
# checkpoint por serie como siempre (ver Crossbar_train_experimento_beta_por_capa.py).
#
# ---------------------------------------------------------------------------
# CONFIG DE ESTA CORRIDA: carpeta de salida + curva P/D (cantidad de puntos
# y su 'a' correspondiente, elegido para que el INL quede igual entre
# curvas con distinta cantidad de puntos).
#
# Para correr la MISMA grilla de beta con OTRA curva P/D:
#   1) copia este archivo entero: `cp train_job.sh train_job_<algo>.sh`
#   2) en la copia, cambia este bloque (OUT_DIR/PULSOS_.../A_...)
#   3) en la copia, cambia TAMBIEN --output/--error de MAS ARRIBA (en el
#      bloque de #SBATCH) para que apunten a la MISMA carpeta que pusiste
#      en OUT_DIR -no te olvides: SLURM no lo detecta solo (ver el aviso
#      de mas arriba), y si no coinciden los logs terminan silenciosamente
#      en el nombre por defecto de SLURM (slurm-%A_%a.out) sin ningun aviso.
#   4) crea esa carpeta (`mkdir -p <OUT_DIR>`) y someté la copia aparte
#      (`sbatch train_job_<algo>.sh`)
# Cada copia escribe en su propia carpeta -no se pisan entre si- y podes
# tener las dos corriendo (o encoladas) al mismo tiempo.
# ---------------------------------------------------------------------------
OUT_DIR="beta_grid_search_SP_INL_1.6e-1_p_100"    # <- carpeta de resultados de ESTA corrida
PULSOS_POT=100                   # <- puntos de la curva de potenciacion
PULSOS_DEP=100                   # <- puntos de la curva de depresion
A_POT=15.94                     # <- 'a' de potenciacion (mismo INL que arriba)
A_DEP=15.94                     # <- 'a' de depresion   (mismo INL que arriba)
#
# %A = job id de la grilla (igual para todas las tareas), %a = indice de
# la tarea (SLURM_ARRAY_TASK_ID) -> un log por beta, nada se pisa.

# (sin -u/nounset: el venv de $HOME/venvs/crossbars tiene un `deactivate`
# que referencia $1 sin default seguro -comun en virtualenv-; con -u eso
# aborta el script como error FATAL al final, DESPUES de que el
# entrenamiento ya termino y se guardo bien, y SLURM marca la tarea como
# FAILED aunque los resultados esten completos y validos.)
set -eo pipefail

# ---------------------------------------------------------------------------
# Lista de valores de beta a explorar (uno por tarea del array). El indice
# de esta tarea dentro de la lista es $SLURM_ARRAY_TASK_ID (0-indexado).
# Se usa la MISMA lista para todas las curvas P/D que corras, asi los betas
# son directamente comparables entre curvas.
# ---------------------------------------------------------------------------
BETAS=( 1 50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000 1050 1100 1150 1200 1250 1300 1350 1400 1450 1500 1550 1600 1650 1700 1750 1800 1850 1900 1950)
N_BETAS=${#BETAS[@]}
BETA=${BETAS[$SLURM_ARRAY_TASK_ID]}

# Prefijo compartido por TODAS las tareas de esta grilla, para que sus
# archivos de salida (resultados_*.npz, config_*.json, ...) queden
# agrupados con el mismo nombre base aunque cada tarea corra en un
# proceso/nodo distinto y en un momento distinto. SLURM_ARRAY_JOB_ID es el
# mismo para las N tareas de un mismo `sbatch` y unico por sometimiento, asi
# que no hace falta coordinar timestamps entre procesos concurrentes. Al
# incluir OUT_DIR tambien queda distinto entre curvas P/D distintas, aunque
# por las dudas coincidiera el job id (no deberia).
EXPERIMENT_ID_BASE="${OUT_DIR}_job${SLURM_ARRAY_JOB_ID}"

echo "Tarea ${SLURM_ARRAY_TASK_ID}/$((N_BETAS - 1)) del array ${SLURM_ARRAY_JOB_ID}"
echo "  -> beta = ${BETA}"
echo "  -> out_dir = ${OUT_DIR}"
echo "  -> pulsos_pot=${PULSOS_POT} pulsos_dep=${PULSOS_DEP} a_pot=${A_POT} a_dep=${A_DEP}"
echo "  -> experiment_id_base = ${EXPERIMENT_ID_BASE}"

module purge
module load Python/3.9.6-GCCcore-11.2.0

source "$HOME/venvs/crossbars/bin/activate"

# Chequeo defensivo: $HOME/venvs/crossbars/bin/python3 quedo enlazado al
# `/usr/bin/python3` DEL SISTEMA OPERATIVO en vez de al Python del modulo
# de arriba (bug conocido, ver diagnostico_python_cluster_habrok.txt). Ese
# `/usr/bin/python3` es LOCAL a cada nodo (no es una ruta compartida como
# $HOME o los modulos de /cvmfs/...), y con la migracion de Habrok de
# AlmaLinux 8 a 9 en curso, distintos nodos de computo pueden tener un
# `/usr/bin/python3` distinto (o directamente sin el Python 3.6 del que
# depende este venv) -por eso numpy/torch a veces "desaparecen" segun en
# que nodo caiga la tarea, aunque el mismo chequeo ande bien en el login
# node. Este chequeo corta la tarea ACA, con un mensaje claro, en vez de
# dejar que reviente mas abajo con un traceback de Python confuso.
# SOLUCION DEFINITIVA (hacer una sola vez, con el modulo cargado):
#   mv $HOME/venvs/crossbars $HOME/venvs/crossbars_old
#   python3 -m venv $HOME/venvs/crossbars
#   source $HOME/venvs/crossbars/bin/activate
#   pip install --upgrade pip && pip install numpy torch
if ! python3 -c "import numpy, torch" 2>/dev/null; then
    echo "ERROR: el venv de \$HOME/venvs/crossbars no tiene numpy/torch" \
         "disponibles en este nodo (posible symlink de python3 roto/atado" \
         "al Python del sistema operativo, ver comentario arriba en" \
         "train_job.sh). Abortando esta tarea SIN correr el entrenamiento." >&2
    exit 1
fi

python3 Crossbar_train_experimento_beta_por_capa.py \
    --beta "$BETA" \
    --beta-idx "$SLURM_ARRAY_TASK_ID" \
    --experiment-id-base "$EXPERIMENT_ID_BASE" \
    --n-betas-totales "$N_BETAS" \
    --out-dir "$OUT_DIR" \
    --pulsos-pot "$PULSOS_POT" \
    --pulsos-dep "$PULSOS_DEP" \
    --a-pot "$A_POT" \
    --a-dep "$A_DEP"

# || true: por las dudas (venv distinto, etc.) esto nunca debe poder tirar
# abajo el codigo de salida del job DESPUES de que el entrenamiento ya
# termino y guardo sus resultados.
deactivate || true
