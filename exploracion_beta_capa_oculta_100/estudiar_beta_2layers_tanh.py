#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verificación numérica de la Sección 5 ("Extensión analítica") de
`reporte_beta_capa_oculta.pdf`: repite EXACTAMENTE el mismo experimento de
`estudiar_beta_2layers.py` (mismas curvas P/D, misma grilla 10x10, mismo
protocolo mínimo, mismas semillas) pero con `MemDNNTanh` (tanh en la capa
oculta) en vez de `MemDNN` (ReLU), para poder comparar directamente si la
ganancia de usar beta-por-capa es mayor con tanh que con ReLU, como predice
el análisis matemático.

NO SE MODIFICA ningún código existente (`MemCrossbarClass_beta_por_capa.py`
ni `estudiar_beta_2layers.py`): usa `MemDNNTanh` de `mem_dnn_tanh.py`
(subclase nueva en este mismo folder).

Resultados: `resultados_beta_2layers_tanh.csv` (mismo formato que
`resultados_beta_2layers.csv`, con checkpointing).
"""

import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(THIS_DIR, "resultados_beta_2layers_tanh.csv")
LOG_PATH = os.path.join(THIS_DIR, "log_estudiar_beta_2layers_tanh.txt")

# Mismas curvas P/D que estudiar_beta_2layers.py (prompt.txt)
CURVAS = [
    dict(nombre="lineal", pulsos_pot=100, pulsos_dep=100, a_pot=4567.9, a_dep=4567.9,
         Gmin=0.0001, Gmax=0.001, concavidad_pot="pos", concavidad_dep="neg"),
    dict(nombre="no_lineal", pulsos_pot=100, pulsos_dep=100, a_pot=98.55, a_dep=98.55,
         Gmin=0.0001, Gmax=0.001, concavidad_pot="pos", concavidad_dep="neg"),
]

MARGEN = 8.0
N_PUNTOS = 10

sizes = [784, 100, 10]
k_folds = 3
epochs = 8
batch_number = 32
lr = 1
delta_t_forward = 1e-8
delta_t_pulse = 1e-8
Vr = -1
Vs = 1
amplitud_imagen = 1
seed_dataset = 35
seed_base = 12345

dataset_X = os.path.join(REPO_ROOT, "X_train_mnist.npy")
dataset_y = os.path.join(REPO_ROOT, "y_train_mnist.npy")

N_WORKERS = max(1, (os.cpu_count() or 4) - 2)

CSV_FIELDS = ["curva", "beta_hidden", "beta_output", "acc_mean", "acc_sem", "tiempo_seg"]


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    linea = f"[{ts}] {msg}"
    print(linea, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(linea + "\n")


def ya_calculado_set():
    hechos = set()
    if not os.path.exists(CSV_PATH):
        return hechos
    with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            hechos.add((row["curva"], round(float(row["beta_hidden"]), 6), round(float(row["beta_output"]), 6)))
    return hechos


def guardar_fila(fila):
    existe = os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not existe:
            w.writeheader()
        w.writerow(fila)


_WORKER_X = None
_WORKER_Y = None


def _worker_init():
    global _WORKER_X, _WORKER_Y
    sys.path.insert(0, REPO_ROOT)
    sys.path.insert(0, THIS_DIR)
    torch.set_num_threads(1)
    X_raw = np.load(dataset_X)
    y_raw = np.load(dataset_y)
    _WORKER_X = torch.from_numpy(X_raw * amplitud_imagen).float()
    _WORKER_Y = torch.from_numpy(y_raw).long()


def _worker_run(task):
    from MemCrossbarClass_beta_por_capa import generar_curvas_pot_dep, DataSetLoader  # noqa: E402
    from mem_dnn_tanh import MemDNNTanh  # noqa: E402

    curva, beta_hidden, beta_output = task["curva"], task["beta_hidden"], task["beta_output"]
    t0 = time.time()

    pot, dep, ratio, inl_pot, inl_dep = generar_curvas_pot_dep(
        curva["pulsos_pot"], curva["pulsos_dep"],
        curva["a_pot"], curva["a_dep"],
        curva["Gmin"], curva["Gmax"],
        curva["concavidad_dep"], curva["concavidad_pot"],
    )
    pot_t = torch.tensor(pot, dtype=torch.float32)
    dep_t = torch.tensor(dep, dtype=torch.float32)

    train_loaders, val_loaders = DataSetLoader(
        _WORKER_X, _WORKER_Y, k_folds, batch_number, random_state=seed_dataset
    )

    criterion = nn.CrossEntropyLoss()
    fold_accs = []
    for k in range(k_folds):
        torch.manual_seed(seed_base + k)
        np.random.seed(seed_base + k)

        model = MemDNNTanh(
            sizes=sizes,
            beta=[float(beta_hidden), float(beta_output)],
            pot=pot_t, dep=dep_t,
            G0_distribution="random", fixed=False, device=torch.device("cpu"),
        )
        model.train()

        for _ in range(epochs):
            model.train_epoch(
                train_loaders[k], lr, criterion,
                delta_t_forward, delta_t_pulse, Vr, Vs,
            )

        acc, _ = model.evaluate_model(val_loaders[k], criterion)
        fold_accs.append(acc)

    acc_mean = float(np.mean(fold_accs))
    acc_sem = float(np.std(fold_accs) / np.sqrt(len(fold_accs)))
    tiempo = time.time() - t0
    return dict(
        curva=curva["nombre"], beta_hidden=float(beta_hidden), beta_output=float(beta_output),
        acc_mean=acc_mean, acc_sem=acc_sem, tiempo_seg=round(tiempo, 1),
    )


def main():
    sys.path.insert(0, REPO_ROOT)
    from MemCrossbarClass_beta_por_capa import (  # noqa: E402
        generar_curvas_pot_dep, beta_pred_formula, estimar_sigma_W_y_paso_medio,
    )

    hechos = ya_calculado_set()
    tareas = []

    for curva in CURVAS:
        pot, dep, ratio, inl_pot, inl_dep = generar_curvas_pot_dep(
            curva["pulsos_pot"], curva["pulsos_dep"],
            curva["a_pot"], curva["a_dep"],
            curva["Gmin"], curva["Gmax"],
            curva["concavidad_dep"], curva["concavidad_pot"],
        )
        pot_t = torch.tensor(pot, dtype=torch.float32)
        dep_t = torch.tensor(dep, dtype=torch.float32)
        sigma_W, paso_medio = estimar_sigma_W_y_paso_medio(pot_t, dep_t, n_muestras=sizes[0])
        beta_pred = beta_pred_formula(sigma_W, paso_medio)
        beta_grid = np.geomspace(beta_pred / MARGEN, beta_pred * MARGEN, N_PUNTOS)

        log(f"=== curva={curva['nombre']} (tanh) beta_pred_formula={beta_pred:.2f} "
            f"-> grilla en [{beta_grid[0]:.1f}, {beta_grid[-1]:.1f}] ===")

        for beta_hidden in beta_grid:
            for beta_output in beta_grid:
                key = (curva["nombre"], round(float(beta_hidden), 6), round(float(beta_output), 6))
                if key in hechos:
                    continue
                tareas.append(dict(curva=curva, beta_hidden=float(beta_hidden), beta_output=float(beta_output)))

    total = len(tareas) + len(hechos)
    log(f"Tareas pendientes: {len(tareas)} (ya calculadas: {len(hechos)}, total: {total}). Workers: {N_WORKERS}")

    if not tareas:
        log("Nada que hacer.")
        return

    completadas = len(hechos)
    t_inicio = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_worker_init) as ex:
        futuros = {ex.submit(_worker_run, t): t for t in tareas}
        for fut in as_completed(futuros):
            t = futuros[fut]
            try:
                fila = fut.result()
                guardar_fila(fila)
                completadas += 1
                transcurrido = time.time() - t_inicio
                ritmo = (completadas - len(hechos)) / transcurrido if transcurrido > 0 else 0
                restante = (total - completadas) / ritmo if ritmo > 0 else float("nan")
                log(f"[{completadas}/{total}] {fila['curva']} "
                    f"bh={fila['beta_hidden']:9.2f} bo={fila['beta_output']:9.2f} -> "
                    f"acc={fila['acc_mean']:.4f}+-{fila['acc_sem']:.4f} "
                    f"[{fila['tiempo_seg']:.1f}s] ETA restante ~{restante/60:.1f} min")
            except Exception as e:
                log(f"ERROR curva={t['curva']['nombre']} bh={t['beta_hidden']:.2f} "
                    f"bo={t['beta_output']:.2f}: {type(e).__name__}: {e}")

    log("Estudio completo.")


if __name__ == "__main__":
    main()
