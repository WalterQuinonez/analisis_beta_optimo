#!/bin/bash
#SBATCH --job-name=beta_grid
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=2800
#SBATCH --partition=regular
#
# Job array: UN SOLO `sbatch train_job.sh` lanza tantas tareas como betas
# haya en BETAS, cada una un entrenamiento INDEPENDIENTE (su propio proceso,
# su propia asignacion de recursos, su propio log). Si una tarea se corta
# (tiempo/memoria/nodo caido) no afecta a las demas, y cada una hace
# checkpoint por serie como siempre (ver Crossbar_train_experimento_beta_por_capa.py).
#
# IMPORTANTE: el rango de --array tiene que cubrir todos los indices de
# BETAS (0 .. len(BETAS)-1). Si cambias la lista, actualiza tambien el
# --array de la linea de abajo.
#SBATCH --array=0-39
#
# %A = job id de la grilla (igual para todas las tareas), %a = indice de
# la tarea (SLURM_ARRAY_TASK_ID) -> un log por beta, nada se pisa.
#SBATCH --output=slurm_beta_%A_%a.out
#SBATCH --error=slurm_beta_%A_%a.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Lista de valores de beta a explorar (uno por tarea del array). El indice
# de esta tarea dentro de la lista es $SLURM_ARRAY_TASK_ID (0-indexado).
# ---------------------------------------------------------------------------
BETAS=( 1 50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000 1050 1100 1150 1200 1250 1300 1350 1400 1450 1500 1550 1600 1650 1700 1750 1800 1850 1900 1950)
N_BETAS=${#BETAS[@]}
BETA=${BETAS[$SLURM_ARRAY_TASK_ID]}

# Prefijo compartido por TODAS las tareas de esta grilla, para que sus
# archivos de salida (resultados_*.npz, config_*.json, ...) queden
# agrupados con el mismo nombre base aunque cada tarea corra en un
# proceso/nodo distinto y en un momento distinto. SLURM_ARRAY_JOB_ID es el
# mismo para las N tareas de un mismo `sbatch` y unico por sometimiento, asi
# que no hace falta coordinar timestamps entre procesos concurrentes.
EXPERIMENT_ID_BASE="SP_1_09_2026_job${SLURM_ARRAY_JOB_ID}"

echo "Tarea ${SLURM_ARRAY_TASK_ID}/$((N_BETAS - 1)) del array ${SLURM_ARRAY_JOB_ID}"
echo "  -> beta = ${BETA}"
echo "  -> experiment_id_base = ${EXPERIMENT_ID_BASE}"

module purge
module load Python/3.9.6-GCCcore-11.2.0

source "$HOME/venvs/crossbars/bin/activate"

python3 Crossbar_train_experimento_beta_por_capa.py \
    --beta "$BETA" \
    --beta-idx "$SLURM_ARRAY_TASK_ID" \
    --experiment-id-base "$EXPERIMENT_ID_BASE" \
    --n-betas-totales "$N_BETAS"

deactivate
