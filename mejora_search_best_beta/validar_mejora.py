#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compara, con datos nuevos (semillas frescas) y contra el ground truth ya
guardado en `data_beta_grid_search_SP/`, tres formas de elegir beta:

  (a) ORIGINAL tal como se usa en la practica: `Crossbar_train_experimento_
      beta_por_capa.py` llama a `search_best_beta` con `beta_max=300,
      points_beta=2` (linea ~396) -> np.linspace(1,300,2) = [1, 300]. Con
      k_folds=5 (el de la config real) y 1 epoca (hardcodeado en el
      original).
  (b) ORIGINAL con la firma "de fabrica" (sin el mal override de la
      practica): `beta_max=400, points=20` (los defaults de la funcion),
      igual 1 epoca, k_folds=5.
  (c) MEJORADO (`search_best_beta_mejorado.MemDNNMejorado`): grilla log de
      16 puntos en [1,2000], k_folds_busqueda=3, epochs_busqueda=8, semillas
      comunes entre betas (common random numbers).

Para cada metodo, se repite N_REPETICIONES veces (con distinta semilla
GLOBAL de arranque cada vez, no relacionada con las 20 series del grid
search original) y se mide el REGRET real: cuanto accuracy final (epoca 100,
protocolo COMPLETO) se pierde respecto del beta verdaderamente optimo, ya
conocido en `data_beta_grid_search_SP/`.

NO se modifica `Crossbar_train_experimento_beta_por_capa.py` ni
`MemCrossbarClass_beta_por_capa.py`; el metodo (a)/(b) se corre llamando
DIRECTAMENTE al `search_best_beta` original (importado sin copiar).
"""

import os
import sys
import time
import json

import numpy as np
import torch
import torch.nn as nn

ANALISIS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "analisis_epochs_series_minimos")
)
FUERZA_BRUTA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Fuerza_bruta")
)
sys.path.insert(0, ANALISIS_DIR)
sys.path.insert(0, FUERZA_BRUTA_DIR)
sys.path.insert(0, os.path.dirname(__file__))

from comun_carga_datos import cargar_combo  # noqa: E402
from MemCrossbarClass_beta_por_capa import MemDNN, generar_curvas_pot_dep, DataSetLoader  # noqa: E402
from search_best_beta_mejorado import MemDNNMejorado  # noqa: E402

DIR_GRID = os.path.join(os.path.dirname(FUERZA_BRUTA_DIR), "data_beta_grid_search_SP")
DIR_SALIDA = os.path.dirname(os.path.abspath(__file__))
N_REPETICIONES = 5

COMBOS = [
    {
        "nombre": "INL_2e-1_p_50", "carpeta_ground_truth": "beta_grid_search_SP_INL_2e-1_p_50",
        "pulsos_pot": 50, "pulsos_dep": 50, "a_pot": 5.85, "a_dep": 5.85,
        "concavidad_pot": "pos", "concavidad_dep": "neg", "Rhigh": 10000, "Rlow": 1000,
        "G0_distribution": "random",
    },
    {
        "nombre": "INL_1e-2_p_100", "carpeta_ground_truth": "beta_grid_search_SP_INL_1e-2_p_100",
        "pulsos_pot": 100, "pulsos_dep": 100, "a_pot": 98.55, "a_dep": 98.55,
        "concavidad_pot": "pos", "concavidad_dep": "neg", "Rhigh": 10000, "Rlow": 1000,
        "G0_distribution": "random",
    },
]

sizes = [784, 10]
device = torch.device("cpu")
criterion = nn.CrossEntropyLoss()
Vr, Vs = -1, 1
delta_t_forward = delta_t_pulse = 1e-8
lr = 1
batch_number = 32

DATASET_X = os.path.join(FUERZA_BRUTA_DIR, "X_train_mnist.npy")
DATASET_Y = os.path.join(FUERZA_BRUTA_DIR, "y_train_mnist.npy")


def ground_truth_regret(carpeta, beta_elegido):
    betas, acc = cargar_combo(os.path.join(DIR_GRID, carpeta))
    acc_final = acc[:, :, :, -1].mean(axis=(1, 2))
    idx_opt = int(np.argmax(acc_final))
    idx_elegido = int(np.argmin(np.abs(betas - beta_elegido)))
    return float(acc_final[idx_opt] - acc_final[idx_elegido]), float(betas[idx_opt]), float(acc_final[idx_opt])


def preparar_combo(combo):
    Gmin, Gmax = 1 / combo["Rhigh"], 1 / combo["Rlow"]
    pot, dep, ratio, inl_pot, inl_dep = generar_curvas_pot_dep(
        combo["pulsos_pot"], combo["pulsos_dep"], combo["a_pot"], combo["a_dep"],
        Gmin, Gmax, combo["concavidad_dep"], combo["concavidad_pot"],
    )
    pot = torch.tensor(pot, dtype=torch.float32)
    dep = torch.tensor(dep, dtype=torch.float32)
    X = torch.from_numpy(np.load(DATASET_X)).float()
    y = torch.from_numpy(np.load(DATASET_Y)).long()
    return pot, dep, X, y


def correr_original(pot, dep, X, y, combo, seed, beta_max, points, k_folds):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loaders, val_loaders = DataSetLoader(X, y, k_folds, batch_number, random_state=seed)
    model = MemDNN(sizes=sizes, beta=250.0, pot=pot, dep=dep,
                   G0_distribution=combo["G0_distribution"], fixed=False, device=device)
    t0 = time.time()
    resultado = model.search_best_beta(
        train_loaders, val_loaders, criterion, combo["a_pot"], lr,
        delta_t_forward, delta_t_pulse, Vr, Vs,
        beta_max=beta_max, points=points,
    )
    dt = time.time() - t0
    if resultado is None:
        return None, dt
    best_beta = resultado[0]
    return float(best_beta), dt


def correr_mejorado(pot, dep, X, y, combo, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loaders, val_loaders = DataSetLoader(X, y, 3, batch_number, random_state=seed)
    model = MemDNNMejorado(sizes=sizes, beta=250.0, pot=pot, dep=dep,
                            G0_distribution=combo["G0_distribution"], fixed=False, device=device)
    t0 = time.time()
    best_beta, best_acc, info = model.search_best_beta(
        train_loaders, val_loaders, criterion, combo["a_pot"], lr,
        delta_t_forward, delta_t_pulse, Vr, Vs,
        seed_base=seed, verbose=False,
    )
    dt = time.time() - t0
    return float(best_beta), dt


def main():
    resultados = []
    for combo in COMBOS:
        print(f"\n===== combo {combo['nombre']} =====")
        pot, dep, X, y = preparar_combo(combo)

        for rep in range(N_REPETICIONES):
            seed = 20000 + rep  # semillas frescas, no relacionadas con las 20 series del grid original
            fila = {"combo": combo["nombre"], "rep": rep, "seed": seed}

            beta_a, dt_a = correr_original(pot, dep, X, y, combo, seed, beta_max=300, points=2, k_folds=5)
            beta_b, dt_b = correr_original(pot, dep, X, y, combo, seed, beta_max=400, points=20, k_folds=5)
            beta_c, dt_c = correr_mejorado(pot, dep, X, y, combo, seed)

            for tag, beta_elegido, dt in [
                ("a_original_misconfigurado", beta_a, dt_a),
                ("b_original_defaults_de_fabrica", beta_b, dt_b),
                ("c_mejorado", beta_c, dt_c),
            ]:
                if beta_elegido is None:
                    regret, beta_opt, acc_opt = float("nan"), float("nan"), float("nan")
                else:
                    regret, beta_opt, acc_opt = ground_truth_regret(combo["carpeta_ground_truth"], beta_elegido)
                print(f"  rep={rep} [{tag}] beta_elegido={beta_elegido} "
                      f"regret={None if np.isnan(regret) else round(regret*100,3)}pp "
                      f"tiempo={dt:.1f}s")
                resultados.append(dict(fila, metodo=tag, beta_elegido=beta_elegido,
                                        beta_optimo_ground_truth=beta_opt,
                                        regret_pp=None if np.isnan(regret) else regret * 100,
                                        tiempo_s=dt))

            # checkpoint incremental
            with open(os.path.join(DIR_SALIDA, "resultados_validacion.json"), "w") as f:
                json.dump(resultados, f, indent=2)

    print("\nGuardado resultados_validacion.json")


if __name__ == "__main__":
    main()
