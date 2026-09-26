#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fase 3 (corroboracion): corridas NUEVAS, minimas, con el protocolo reducido
recomendado por las Fases 1-2 (series/epochs/k_folds mucho menores que el
grid search original de `Fuerza_bruta/`), para confirmar que igual se
recupera un beta_optimo cercano al de la corrida completa (ground truth ya
guardado en data_beta_grid_search_SP/).

ESTE ARCHIVO ES UNA COPIA/ADAPTACION, NO UNA MODIFICACION del original:
- No se toca `Crossbar_train_experimento_beta_por_capa.py` ni
  `MemCrossbarClass_beta_por_capa.py` (viven intactos en Fuerza_bruta/; este
  script los importa via sys.path, sin copiarlos).
- Lo unico que cambia respecto del protocolo original es: series, epochs,
  k_folds, una grilla de beta mas angosta (subconjunto de la original de 40
  valores), semillas de fold NUEVAS (no reutiliza las 20 semillas ya usadas
  en el grid search original, para que esto sea una corroboracion genuina y
  no una reevaluacion de los mismos datos), y un guardado de G_history minimo
  (solo la ultima epoca de cada serie, no todas) para no gastar disco de mas.

Antes de cada serie se chequea espacio libre en disco y se aborta la corrida
(dejando el checkpoint ya escrito) si quedarian menos de 50 GB libres.
"""

import json
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn as nn

FUERZA_BRUTA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "Fuerza_bruta")
)
sys.path.insert(0, FUERZA_BRUTA_DIR)  # para importar el codigo ORIGINAL, sin copiarlo
from MemCrossbarClass_beta_por_capa import MemDNN, generar_curvas_pot_dep, DataSetLoader  # noqa: E402

# =============================================================================
# 0) PROTOCOLO REDUCIDO (recomendado por Fase 1 / Fase 2 de este estudio)
# =============================================================================
SEED_CONFIRMATORIO = 9035  # distinto del seed=35 del grid search original
series = 8       # vs. 20 en el grid original
epochs = 8        # vs. 100 en el grid original
k_folds = 3       # vs. 5 en el grid original
batch_number = 32
lr = 1
sizes = [784, 10]
device = torch.device("cpu")
save_G_every = epochs  # solo la ultima epoca de cada serie (ahorra disco)

DISCO_MIN_LIBRE_GB = 50
UNIDAD_DISCO = os.path.splitdrive(os.path.abspath(__file__))[0] + os.sep

# Grilla de beta ANGOSTA: subconjunto de la grilla original completa
# (1,50,100,...,1950), no se asume de antemano donde esta el optimo.
BETA_GRID = [1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 900, 1200, 1600]

criterion = nn.CrossEntropyLoss()
Vr, Vs = -1, 1
delta_t_forward = 1e-8
delta_t_pulse = 1e-8
amplitud_imagen = 1

DATASET_X = os.path.join(FUERZA_BRUTA_DIR, "X_train_mnist.npy")
DATASET_Y = os.path.join(FUERZA_BRUTA_DIR, "y_train_mnist.npy")

# Combos a confirmar: parametros de curva P/D copiados TAL CUAL de los
# config_*.json ya guardados por el grid search original (misma Gmin/Gmax,
# pulsos, a_pot/a_dep, concavidad, G0_distribution), para que la unica
# diferencia real sea el protocolo (series/epochs/k_folds/grilla de beta) y
# las semillas de fold.
COMBOS_A_CONFIRMAR = [
    {
        "nombre": "INL_2e-1_p_50",
        "pulsos_pot": 50, "pulsos_dep": 50, "a_pot": 5.85, "a_dep": 5.85,
        "concavidad_pot": "pos", "concavidad_dep": "neg",
        "Rhigh": 10000, "Rlow": 1000, "G0_distribution": "random",
        "beta_optimo_ground_truth": 100.0, "acc_ground_truth": 0.847898,
    },
    {
        "nombre": "INL_1e-2_p_100",
        "pulsos_pot": 100, "pulsos_dep": 100, "a_pot": 98.55, "a_dep": 98.55,
        "concavidad_pot": "pos", "concavidad_dep": "neg",
        "Rhigh": 10000, "Rlow": 1000, "G0_distribution": "random",
        "beta_optimo_ground_truth": 350.0, "acc_ground_truth": 0.909067,
    },
]

DIR_SALIDA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "confirmatorio_resultados")
os.makedirs(DIR_SALIDA, exist_ok=True)


def gb_libres():
    return shutil.disk_usage(UNIDAD_DISCO).free / (1024 ** 3)


def chequear_disco_o_abortar(contexto):
    libres = gb_libres()
    print(f"[disco] libres en {UNIDAD_DISCO}: {libres:.1f} GB ({contexto})")
    if libres < DISCO_MIN_LIBRE_GB:
        raise RuntimeError(
            f"Espacio libre ({libres:.1f} GB) por debajo del piso de seguridad "
            f"({DISCO_MIN_LIBRE_GB} GB) en {contexto}. Corrida abortada; el "
            f"checkpoint ya escrito hasta ahora queda intacto."
        )


def correr_combo(combo):
    print(f"\n================ Combo {combo['nombre']} ================")
    chequear_disco_o_abortar(f"inicio de combo {combo['nombre']}")

    Gmin = 1 / combo["Rhigh"]
    Gmax = 1 / combo["Rlow"]
    pot, dep, ratio, inl_pot, inl_dep = generar_curvas_pot_dep(
        combo["pulsos_pot"], combo["pulsos_dep"], combo["a_pot"], combo["a_dep"],
        Gmin, Gmax, combo["concavidad_dep"], combo["concavidad_pot"],
    )
    pot = torch.tensor(pot, dtype=torch.float32)
    dep = torch.tensor(dep, dtype=torch.float32)
    print(f"  INL_pot={inl_pot:.6g} INL_dep={inl_dep:.6g} (referencia ground truth en el nombre del combo)")

    X_train = np.load(DATASET_X) * amplitud_imagen
    y_train = np.load(DATASET_Y)
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()

    fold_seeds_por_serie = [SEED_CONFIRMATORIO + s for s in range(series)]
    folds_por_serie = [
        DataSetLoader(X_train, y_train, k_folds, batch_number, random_state=fs)
        for fs in fold_seeds_por_serie
    ]

    n_betas = len(BETA_GRID)
    acc_all = np.zeros((n_betas, series, k_folds, epochs), dtype=np.float32)

    t0_combo = time.time()
    for b_idx, beta_value in enumerate(BETA_GRID):
        print(f"\n-- beta {b_idx + 1}/{n_betas} = {beta_value} --")
        for serie in range(series):
            chequear_disco_o_abortar(f"{combo['nombre']} beta={beta_value} serie={serie}")
            train_loaders, val_loaders = folds_por_serie[serie]
            for fold in range(k_folds):
                model = MemDNN(
                    sizes=sizes, beta=float(beta_value), pot=pot, dep=dep,
                    G0_distribution=combo["G0_distribution"], fixed=False, device=device,
                )
                train_loader = train_loaders[fold]
                val_loader = val_loaders[fold]
                for e in range(epochs):
                    model.train_epoch(train_loader, lr, criterion,
                                       delta_t_forward, delta_t_pulse, Vr, Vs)
                    acc_val, _ = model.evaluate_model(val_loader, criterion)
                    acc_all[b_idx, serie, fold, e] = acc_val
            print(f"   serie {serie}: acc final por fold = "
                  f"{acc_all[b_idx, serie, :, -1].round(4).tolist()}")

        # checkpoint atomico tras cada beta completado (por si se corta la corrida)
        tmp = os.path.join(DIR_SALIDA, f"confirmatorio_{combo['nombre']}.npz.tmp")
        final = os.path.join(DIR_SALIDA, f"confirmatorio_{combo['nombre']}.npz")
        with open(tmp, "wb") as f:
            np.savez_compressed(
                f, acc_result=acc_all[: b_idx + 1], betas=np.array(BETA_GRID[: b_idx + 1]),
                betas_completados=b_idx + 1, series=series, epochs=epochs, k_folds=k_folds,
                seed_confirmatorio=SEED_CONFIRMATORIO,
                beta_optimo_ground_truth=combo["beta_optimo_ground_truth"],
                acc_ground_truth=combo["acc_ground_truth"],
            )
        os.replace(tmp, final)
        print(f"   [checkpoint] {final}")

    acc_mean = acc_all.mean(axis=(1, 2))  # (n_betas, epochs)
    idx_opt = int(np.argmax(acc_mean[:, -1]))
    beta_opt_confirmatorio = BETA_GRID[idx_opt]
    acc_opt_confirmatorio = float(acc_mean[idx_opt, -1])

    resumen = {
        "combo": combo["nombre"],
        "protocolo": {"series": series, "epochs": epochs, "k_folds": k_folds,
                      "n_betas_grilla": n_betas, "seed": SEED_CONFIRMATORIO},
        "beta_optimo_confirmatorio": beta_opt_confirmatorio,
        "acc_optimo_confirmatorio": acc_opt_confirmatorio,
        "beta_optimo_ground_truth": combo["beta_optimo_ground_truth"],
        "acc_ground_truth": combo["acc_ground_truth"],
        "tiempo_total_s": round(time.time() - t0_combo, 1),
    }
    with open(os.path.join(DIR_SALIDA, f"resumen_{combo['nombre']}.json"), "w") as f:
        json.dump(resumen, f, indent=2)
    print(f"\n>>> {combo['nombre']}: beta_optimo confirmatorio={beta_opt_confirmatorio} "
          f"(ground truth={combo['beta_optimo_ground_truth']}) "
          f"acc={acc_opt_confirmatorio:.4f} (ground truth={combo['acc_ground_truth']:.4f}) "
          f"tiempo={resumen['tiempo_total_s']}s")
    return resumen


def main():
    chequear_disco_o_abortar("antes de empezar")
    resumenes = [correr_combo(c) for c in COMBOS_A_CONFIRMAR]
    with open(os.path.join(DIR_SALIDA, "resumen_final.json"), "w") as f:
        json.dump(resumenes, f, indent=2)
    print("\n=== Resumen final ===")
    for r in resumenes:
        print(r)


if __name__ == "__main__":
    main()
