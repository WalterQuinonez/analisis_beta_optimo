#!/bin/bash
set -e

# Zona de transicion en INL (entre 5e-2 y 2e-1), ver
# valores_a_pot_a_dep_INL_7e-2_a_1.6e-1.txt
for OUT_DIR in \
    beta_grid_search_SP_INL_7e-2_p_50 \
    beta_grid_search_SP_INL_7e-2_p_100 \
    beta_grid_search_SP_INL_7e-2_p_200 \
    beta_grid_search_SP_INL_1e-1_p_50 \
    beta_grid_search_SP_INL_1e-1_p_100 \
    beta_grid_search_SP_INL_1e-1_p_200 \
    beta_grid_search_SP_INL_1.3e-1_p_50 \
    beta_grid_search_SP_INL_1.3e-1_p_100 \
    beta_grid_search_SP_INL_1.3e-1_p_200 \
    beta_grid_search_SP_INL_1.6e-1_p_50 \
    beta_grid_search_SP_INL_1.6e-1_p_100 \
    beta_grid_search_SP_INL_1.6e-1_p_200 ; do
    mkdir -p "$OUT_DIR"
done

sbatch train_job_INL_7e-2_p_50.sh
sbatch train_job_INL_7e-2_p_100.sh
sbatch train_job_INL_7e-2_p_200.sh
sbatch train_job_INL_1e-1_p_50.sh
sbatch train_job_INL_1e-1_p_100.sh
sbatch train_job_INL_1e-1_p_200.sh
sbatch train_job_INL_1.3e-1_p_50.sh
sbatch train_job_INL_1.3e-1_p_100.sh
sbatch train_job_INL_1.3e-1_p_200.sh
sbatch train_job_INL_1.6e-1_p_50.sh
sbatch train_job_INL_1.6e-1_p_100.sh
sbatch train_job_INL_1.6e-1_p_200.sh
