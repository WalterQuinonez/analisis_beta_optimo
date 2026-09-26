#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pregunta: la busqueda mejorada de beta (`MemDNNMejorado.search_best_beta`,
en `mejora_search_best_beta/`) se calibro/valido con el rango de
conductancia FIJO que usa todo `data_beta_grid_search_SP/`
(Rlow=1000, Rhigh=10000 -> ratio Gmax/Gmin=10). La formula analitica que
centra su grilla (`beta_pred_formula`, basada en sigma_W y paso_medio) fue
ajustada con datos de ESE mismo ratio=10. ¿Sigue funcionando si cambia el
rango de conductancia?

Metodologia (sin tocar NINGUN archivo original ni el de la mejora):
  Para cada ratio Gmax/Gmin en {2, 20, 100} (Rlow=1000 fijo, Rhigh=Rlow*ratio;
  ratio=10 ya esta validado en `mejora_search_best_beta/`, no se repite aca):
    1) REFERENCIA (independiente de la formula): grilla ancha y fija de 10
       betas log-espaciados en [20, 8000] (no centrada en ninguna
       prediccion), evaluada con el protocolo MINIMO ya encontrado y
       validado en `analisis_epochs_series_minimos/` (Fase 3, el mas
       agresivo: series=8, k_folds=3, epochs=8), semillas nuevas. El
       beta con mayor accuracy media (8 series x 3 folds) en esa grilla es
       la referencia de "beta optimo real" para ese ratio (no es un ground
       truth perfecto tipo el grid search de 40 betas x 20 series x 100
       epocas, pero usa el mismo protocolo que ya demostramos suficiente).
    2) BUSQUEDA MEJORADA: se corre `MemDNNMejorado.search_best_beta` (que
       internamente ya usa el protocolo minimo: k_folds_busqueda=3,
       epochs_busqueda=8) 3 veces con distintas semillas, y se compara el
       beta elegido contra la referencia del paso 1 (regret: cuanto
       accuracy pierde el beta elegido, evaluado en la MISMA grilla de
       referencia, respecto del mejor de esa grilla).

INL_pot se mantiene practicamente constante entre ratios (a_pot=a_dep fijo,
INL solo depende de la forma normalizada de la curva, no de Gmin/Gmax) -ver
chequeo previo, INL entre 0.0065 y 0.0101 para ratio 2..100- para aislar el
efecto del rango de conductancia del efecto de no linealidad.

Guardrail de disco: aborta si quedarian <50GB libres (igual que en los
estudios anteriores de este proyecto).
"""

import sys
import os
import json
import shutil
import time

import numpy as np
import torch
import torch.nn as nn

FUERZA_BRUTA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Fuerza_bruta"))
MEJORA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mejora_search_best_beta"))
sys.path.insert(0, FUERZA_BRUTA_DIR)
sys.path.insert(0, MEJORA_DIR)

from MemCrossbarClass_beta_por_capa import MemDNN, generar_curvas_pot_dep, DataSetLoader  # noqa: E402
from search_best_beta_mejorado import MemDNNMejorado, beta_pred_formula, estimar_sigma_W_y_paso_medio  # noqa: E402

DIR_SALIDA = os.path.dirname(os.path.abspath(__file__))
DISCO_MIN_LIBRE_GB = 50
UNIDAD_DISCO = os.path.splitdrive(os.path.abspath(__file__))[0] + os.sep

# --- protocolo minimo ya encontrado y validado (Fase 3 de analisis_epochs_series_minimos) ---
SERIES_REFERENCIA = 8
K_FOLDS = 3
EPOCHS = 8
BETAS_REFERENCIA = np.geomspace(20, 8000, 10)  # ancha y fija, independiente de la formula
N_REPS_BUSQUEDA = 3

RATIOS_A_PROBAR = [2, 20, 100]  # ratio=10 ya validado en mejora_search_best_beta/
Rlow_FIJO = 1000
a_pot = a_dep = 98.55
pulsos_pot = pulsos_dep = 100

sizes = [784, 10]
device = torch.device("cpu")
criterion = nn.CrossEntropyLoss()
Vr, Vs = -1, 1
delta_t_forward = delta_t_pulse = 1e-8
lr = 1
batch_number = 32
DATASET_X = os.path.join(FUERZA_BRUTA_DIR, "X_train_mnist.npy")
DATASET_Y = os.path.join(FUERZA_BRUTA_DIR, "y_train_mnist.npy")


def gb_libres():
    return shutil.disk_usage(UNIDAD_DISCO).free / (1024 ** 3)


def chequear_disco(contexto):
    libres = gb_libres()
    print(f"[disco] libres: {libres:.1f} GB ({contexto})")
    if libres < DISCO_MIN_LIBRE_GB:
        raise RuntimeError(f"Espacio libre ({libres:.1f} GB) por debajo de {DISCO_MIN_LIBRE_GB} GB. Corrida abortada.")


def preparar_ratio(ratio):
    Rhigh = Rlow_FIJO * ratio
    Gmin, Gmax = 1 / Rhigh, 1 / Rlow_FIJO
    pot, dep, r, inl_pot, inl_dep = generar_curvas_pot_dep(
        pulsos_pot, pulsos_dep, a_pot, a_dep, Gmin, Gmax, "neg", "pos")
    pot_t = torch.tensor(pot, dtype=torch.float32)
    dep_t = torch.tensor(dep, dtype=torch.float32)
    return dict(Rhigh=Rhigh, Gmin=Gmin, Gmax=Gmax, pot=pot_t, dep=dep_t, inl_pot=inl_pot)


def evaluar_beta_en_grilla(beta, pot, dep, X, y, seed_base, series=SERIES_REFERENCIA):
    """Accuracy media (sobre `series` series x K_FOLDS folds), protocolo minimo,
    para UN beta. Usado para construir la grilla de referencia."""
    accs = []
    for s in range(series):
        seed = seed_base + s
        train_loaders, val_loaders = DataSetLoader(X, y, K_FOLDS, batch_number, random_state=seed)
        for k in range(K_FOLDS):
            torch.manual_seed(seed * 1000 + k)
            np.random.seed(seed * 1000 + k)
            model = MemDNN(sizes=sizes, beta=float(beta), pot=pot, dep=dep,
                            G0_distribution="random", fixed=False, device=device)
            model.train()
            for _ in range(EPOCHS):
                model.train_epoch(train_loaders[k], lr, criterion, delta_t_forward, delta_t_pulse, Vr, Vs)
            acc, _ = model.evaluate_model(val_loaders[k], criterion)
            accs.append(acc)
    return float(np.mean(accs))


def cargar_estado_previo():
    """Retoma resultados_rangos_conductancia.json (ratios ya completos: referencia
    + N_REPS_BUSQUEDA busquedas) si existe, para no recalcular lo ya hecho."""
    path = os.path.join(DIR_SALIDA, "resultados_rangos_conductancia.json")
    if not os.path.exists(path):
        return {}, []
    with open(path) as f:
        data = json.load(f)
    referencias = {int(k): v for k, v in data.get("referencias", {}).items()}
    resultados = data.get("busquedas", [])
    return referencias, resultados


def cargar_referencia_parcial(ratio):
    """Retoma un checkpoint incremental de referencia_ratio{ratio}.json (corrida
    de construir_referencia cortada a mitad de camino) si existe."""
    path = os.path.join(DIR_SALIDA, f"referencia_ratio{ratio}.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        d = json.load(f)
    return d["acc"]


def construir_referencia(ratio, ctx, X, y, seed_base, accs_previas=None):
    print(f"\n--- Referencia (grilla ancha fija) para ratio={ratio} ---")
    accs = list(accs_previas) if accs_previas else []
    inicio = len(accs)
    if inicio:
        print(f"  (retomando checkpoint: {inicio}/{len(BETAS_REFERENCIA)} betas ya calculados)")
    for beta in BETAS_REFERENCIA[inicio:]:
        t0 = time.time()
        acc = evaluar_beta_en_grilla(beta, ctx["pot"], ctx["dep"], X, y, seed_base)
        accs.append(acc)
        print(f"  beta={beta:8.1f}  acc_media={acc:.4f}  ({time.time()-t0:.1f}s)")
        chequear_disco(f"ratio={ratio} beta={beta:.1f}")
        # checkpoint incremental
        with open(os.path.join(DIR_SALIDA, f"referencia_ratio{ratio}.json"), "w") as f:
            json.dump({"ratio": ratio, "betas": BETAS_REFERENCIA[:len(accs)].tolist(), "acc": accs}, f, indent=2)
    idx_opt = int(np.argmax(accs))
    return dict(betas=BETAS_REFERENCIA.tolist(), acc=accs,
                beta_optimo_referencia=float(BETAS_REFERENCIA[idx_opt]),
                acc_optima_referencia=float(accs[idx_opt]))


def regret_contra_referencia(referencia, beta_elegido):
    betas = np.array(referencia["betas"])
    acc = np.array(referencia["acc"])
    idx_cercano = int(np.argmin(np.abs(betas - beta_elegido)))
    return referencia["acc_optima_referencia"] - acc[idx_cercano]


def correr_busqueda(ctx, X, y, seed_base):
    torch.manual_seed(seed_base)
    np.random.seed(seed_base)
    train_loaders, val_loaders = DataSetLoader(X, y, K_FOLDS, batch_number, random_state=seed_base)
    model = MemDNNMejorado(sizes=sizes, beta=250.0, pot=ctx["pot"], dep=ctx["dep"],
                            G0_distribution="random", fixed=False, device=device)
    t0 = time.time()
    best_beta, best_acc, info = model.search_best_beta(
        train_loaders, val_loaders, criterion, a_pot, lr,
        delta_t_forward, delta_t_pulse, Vr, Vs,
        seed_base=seed_base, verbose=False,
        max_expansiones=5,  # mas margen que el default (2): esta prueba busca
                             # justamente ver si la formula falla lejos de su
                             # rango de calibracion (ratio=10), y no queremos
                             # que la red de seguridad se quede corta por eso
    )
    dt = time.time() - t0
    return best_beta, info, dt


def main():
    chequear_disco("inicio")

    referencias_acumuladas, resultados = cargar_estado_previo()
    resultados_path = os.path.join(DIR_SALIDA, "resultados_rangos_conductancia.json")
    X = y = None  # se cargan solo si falta trabajo por hacer, para no pagar el costo si todo esta completo

    for idx_r, ratio in enumerate(RATIOS_A_PROBAR):
        reps_previos = [r for r in resultados if r["ratio"] == ratio]
        reps_hechos = {r["rep"] for r in reps_previos}
        referencia_previa = referencias_acumuladas.get(ratio)
        if referencia_previa is not None and len(reps_hechos) >= N_REPS_BUSQUEDA:
            print(f"\n===== ratio={ratio}: ya completo (checkpoint previo), se omite =====")
            continue

        if X is None:
            X = torch.from_numpy(np.load(DATASET_X)).float()
            y = torch.from_numpy(np.load(DATASET_Y)).long()

        ctx = preparar_ratio(ratio)
        print(f"\n===== ratio={ratio} (Rhigh={ctx['Rhigh']}) INL_pot={ctx['inl_pot']:.5f} =====")

        seed_ref = 40000 + idx_r * 1000
        if referencia_previa is not None:
            accs_prev = referencia_previa["acc"]
        else:
            accs_prev = cargar_referencia_parcial(ratio)
        referencia = construir_referencia(ratio, ctx, X, y, seed_ref, accs_previas=accs_prev)
        referencias_acumuladas[ratio] = referencia
        print(f"  beta_optimo (referencia) = {referencia['beta_optimo_referencia']} "
              f"acc = {referencia['acc_optima_referencia']:.4f}")

        for rep in range(N_REPS_BUSQUEDA):
            if rep in reps_hechos:
                continue
            seed_busqueda = 50000 + idx_r * 1000 + rep
            best_beta, info, dt = correr_busqueda(ctx, X, y, seed_busqueda)
            regret = regret_contra_referencia(referencia, best_beta)
            fila = dict(
                ratio=ratio, rep=rep, seed=seed_busqueda,
                beta_pred_formula=info["beta_pred_formula"],
                sigma_W=info["sigma_W"], paso_medio=info["paso_medio"],
                beta_elegido=best_beta, expansiones=info["expansiones"],
                beta_optimo_referencia=referencia["beta_optimo_referencia"],
                regret_pp=regret * 100, tiempo_s=dt,
            )
            resultados.append(fila)
            print(f"  [busqueda rep={rep}] beta_pred_formula={info['beta_pred_formula']:.1f} "
                  f"beta_elegido={best_beta:.1f} (expansiones={info['expansiones']}) "
                  f"regret={regret*100:.3f}pp tiempo={dt:.1f}s")

            with open(resultados_path, "w") as f:
                json.dump(dict(referencias=referencias_acumuladas, busquedas=resultados), f, indent=2)

    print("\n=== Resumen ===")
    for f in resultados:
        print(f)


if __name__ == "__main__":
    main()
