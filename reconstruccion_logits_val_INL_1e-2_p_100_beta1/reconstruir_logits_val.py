#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reconstruir_logits_val.py

Reconstruye EXACTAMENTE los logits de validacion de una corrida real de
grid-search (INL_1e-2_p_100, beta=1), a partir de los pesos guardados
(G_history_*.npz) y la particion de folds guardada (fold_seeds_por_serie en
el config), y compara la distribucion de corriente real resultante contra
el Monte Carlo de `analisis_beta_optimo.py` (que arma corrientes sinteticas
a partir de pesos combinatorios pot[i]-dep[j] y voltajes equipesados de
np.unique(X_train), no de la red ya entrenada).

Por que se puede reconstruir EXACTO (sin reentrenar nada):
  - forward() de MemDNN es puramente determinista: I = x @ (G_pos - G_neg),
    sin dropout ni ruido (el config de esta corrida tiene ruido=0%).
  - El val_loader usa shuffle=False, y la particion train/val en si sale de
    KFold(shuffle=True, random_state=fold_seed) -- fold_seed esta guardado
    en el config (`folds.fold_seeds_por_serie`), y sklearn.KFold no toca el
    estado global de numpy/torch, asi que la particion es 100% reproducible
    sin depender de nada mas.
  - save_G_every=1 en esta corrida -> se guardo el peso de TODAS las 100
    epocas (G_layer0_history), no solo la ultima.

Que NO se puede reconstruir asi (ver conversacion): los logits calculados
EN VIVO sobre los minibatches de TRAIN durante el propio paso de SGD,
porque ese shuffle (train_loader, shuffle=True) consume el estado global de
torch, que solo se fijo una vez al arrancar el proceso original y no quedo
guardado en ningun archivo. Este script solo reconstruye VALIDACION.

Uso:
    python reconstruir_logits_val.py
Salida (en esta misma carpeta):
    resultados_reconstruccion.csv, comparacion_corrientes_MC_vs_real.png,
    evolucion_corriente_por_epoca.png, sanity_check_accuracy.csv
"""
import os
import csv
import json

import numpy as np
import torch
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CARPETA = os.path.join(REPO_ROOT, "data_beta_grid_search_SP", "beta_grid_search_SP_INL_1e-2_p_100")
EXP_ID = "beta_grid_search_SP_INL_1e-2_p_100_job31136791_beta00_b1"
SERIE = 0  # serie usada para la reconstruccion (fold_seed = fold_seeds_por_serie[0])

# ---------------------------------------------------------------
# 1) cargar config + resultados + G_history REALES de esta corrida
# ---------------------------------------------------------------
with open(os.path.join(CARPETA, f"config_{EXP_ID}.json"), "r", encoding="utf-8") as fh:
    cfg = json.load(fh)

k_folds = cfg["entrenamiento"]["k_folds"]
fold_seeds_por_serie = cfg["folds"]["fold_seeds_por_serie"]
beta_value = cfg["beta"]["valor"]
Gmin, Gmax = cfg["curva_pd"]["Gmin"], cfg["curva_pd"]["Gmax"]
a_pot, a_dep = cfg["curva_pd"]["a_pot"], cfg["curva_pd"]["a_dep"]
pulsos_pot = cfg["curva_pd"]["pulsos_pot"]
print(f"beta={beta_value}  k_folds={k_folds}  fold_seed(serie {SERIE})={fold_seeds_por_serie[SERIE]}  "
      f"a_pot={a_pot}  Gmin={Gmin}  Gmax={Gmax}")

resultados = np.load(os.path.join(CARPETA, f"resultados_{EXP_ID}.npz"), allow_pickle=True)
acc_result = resultados["acc_result"]          # (series, k_folds, epochs)
saved_epochs = resultados["saved_epochs"]      # (epochs,) -- aca: 1..100 (save_G_every=1)

g_hist = np.load(os.path.join(CARPETA, f"G_history_{EXP_ID}_serie{SERIE:03d}.npz"))
G_all = g_hist["G_layer0_history"]             # (k_folds, n_saved, 784, 20)
print(f"G_layer0_history shape = {G_all.shape}  (k_folds, n_saved_epochs, D_in, 2*D_out)")

# ---------------------------------------------------------------
# 2) cargar dataset REAL (igual que Crossbar_train_experimento_beta_por_capa.py)
# ---------------------------------------------------------------
X = torch.from_numpy(np.load(os.path.join(REPO_ROOT, "X_train_mnist.npy"))).float()
y = torch.from_numpy(np.load(os.path.join(REPO_ROOT, "y_train_mnist.npy"))).long()
print(f"X_train_mnist.npy: {tuple(X.shape)}")

# ---------------------------------------------------------------
# 3) reconstruir la particion KFold EXACTA de esta serie (igual que
#    DataSetLoader() en MemCrossbarClass_beta_por_capa.py: mismo random_state,
#    mismo orden de folds al iterar kf.split())
# ---------------------------------------------------------------
fold_seed = fold_seeds_por_serie[SERIE]
kf = KFold(n_splits=k_folds, shuffle=True, random_state=fold_seed)
splits = list(kf.split(np.arange(len(X))))  # [(train_idx, val_idx), ...] uno por fold, mismo orden 0..k_folds-1


def forward_logits(G_arr, X_sub, beta):
    """Replica EXACTA de MemDNN.forward() para una red de 1 capa (sizes=[784,10]):
    I = x @ (G_pos - G_neg); logits = beta * I (para la ultima/unica capa)."""
    G = torch.from_numpy(G_arr).float()
    G_pos = G[:, 0::2]
    G_neg = G[:, 1::2]
    W_eff = G_pos - G_neg
    I = X_sub @ W_eff
    return beta * I, I


# ---------------------------------------------------------------
# 4) SANITY CHECK: la accuracy que sale de este forward reconstruido tiene
#    que coincidir con el acc_result YA guardado en su momento durante el
#    entrenamiento real (primera y ultima epoca guardada, los 5 folds).
# ---------------------------------------------------------------
sanity_rows = []
for fold in range(k_folds):
    train_idx, val_idx = splits[fold]
    X_val, y_val = X[val_idx], y[val_idx]
    for epoch_idx in (0, len(saved_epochs) - 1):  # primera y ultima epoca guardada
        epoch_num = int(saved_epochs[epoch_idx])
        G_arr = G_all[fold, epoch_idx]
        logits, _ = forward_logits(G_arr, X_val, beta_value)
        acc_reconstruida = (logits.argmax(1) == y_val).float().mean().item()
        acc_guardada = float(acc_result[SERIE, fold, epoch_num - 1])
        sanity_rows.append(dict(fold=fold, epoch=epoch_num, n_val=len(val_idx),
                                 acc_reconstruida=acc_reconstruida, acc_guardada=acc_guardada,
                                 diferencia=acc_reconstruida - acc_guardada))
        print(f"  fold={fold} epoch={epoch_num:>3}  acc_reconstruida={acc_reconstruida:.6f}  "
              f"acc_guardada={acc_guardada:.6f}  diff={acc_reconstruida - acc_guardada:+.2e}")

max_diff = max(abs(r["diferencia"]) for r in sanity_rows)
print(f"\nMaxima diferencia |reconstruida - guardada| sobre {len(sanity_rows)} chequeos: {max_diff:.2e}")
with open(os.path.join(OUT_DIR, "sanity_check_accuracy.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["fold", "epoch", "n_val", "acc_reconstruida", "acc_guardada", "diferencia"])
    w.writeheader()
    w.writerows(sanity_rows)

# ---------------------------------------------------------------
# 5) distribucion de corriente REAL (I, antes de beta) en la 1ra y ultima
#    epoca guardada, juntando los 5 folds (cubren el dataset completo)
# ---------------------------------------------------------------
def corrientes_reales(epoch_idx):
    todas = []
    for fold in range(k_folds):
        _, val_idx = splits[fold]
        X_val = X[val_idx]
        G_arr = G_all[fold, epoch_idx]
        _, I = forward_logits(G_arr, X_val, beta=1.0)  # beta=1 -> logits==I directamente
        todas.append(I.numpy().ravel())
    return np.concatenate(todas)


I_epoca1 = corrientes_reales(0)
I_epoca100 = corrientes_reales(len(saved_epochs) - 1)
print(f"\nCorriente real, epoca 1:   n={I_epoca1.size}  mean={I_epoca1.mean():.4e}  std={I_epoca1.std():.4e}  "
      f"rms={np.sqrt(np.mean(I_epoca1**2)):.4e}")
print(f"Corriente real, epoca 100: n={I_epoca100.size}  mean={I_epoca100.mean():.4e}  std={I_epoca100.std():.4e}  "
      f"rms={np.sqrt(np.mean(I_epoca100**2)):.4e}")

# ---------------------------------------------------------------
# 6) Monte Carlo de analisis_beta_optimo.py, replicado EXACTO para estos
#    mismos parametros fisicos (a_pot=a_dep=98.55, pulsos=100, Gmin/Gmax
#    iguales) -- mismo metodo: wij combinatorio pot[i]-dep[j], voltajes de
#    np.unique(X_train) (equipesado, NO por frecuencia real), suma de 784
#    terminos elegidos al azar, 1e6 repeticiones.
# ---------------------------------------------------------------
import sys
sys.path.insert(0, REPO_ROOT)
from MemCrossbarClass_beta_por_capa import generar_curvas_pot_dep  # noqa: E402

MUESTRAS = 1_000_000
CHUNK = 50_000
N_SUM = 28 * 28
rng = np.random.default_rng(35)

pot, dep, ratio, inl_pot, inl_dep = generar_curvas_pot_dep(
    pulsos_pot, cfg["curva_pd"]["pulsos_dep"], a_pot, a_dep, Gmin, Gmax,
    cfg["curva_pd"]["concavidad_dep"], cfg["curva_pd"]["concavidad_pot"])

wij = np.zeros([len(pot), len(dep)])
for i in range(len(pot)):
    wij[i] = pot[i] - dep
voltajes_equipesados = np.unique(X.numpy())
x_mc = np.multiply.outer(voltajes_equipesados, wij.flatten()).ravel()

corrientes_mc = np.empty(MUESTRAS)
for start in range(0, MUESTRAS, CHUNK):
    end = min(start + CHUNK, MUESTRAS)
    idx = rng.integers(0, x_mc.size, size=(end - start, N_SUM))
    corrientes_mc[start:end] = x_mc[idx].sum(axis=1)

print(f"\nCorriente Monte Carlo (analisis_beta_optimo.py): n={corrientes_mc.size}  "
      f"mean={corrientes_mc.mean():.4e}  std={corrientes_mc.std():.4e}  "
      f"rms={np.sqrt(np.mean(corrientes_mc**2)):.4e}")

# ---------------------------------------------------------------
# 7) heuristicas de "beta optimo" de analisis_beta_optimo.py, aplicadas a
#    las 3 distribuciones de corriente (MC, real epoca 1, real epoca 100),
#    comparadas contra el beta optimo EMPIRICO real de esta curva (350,
#    de resumen_por_INL/resumen_optimos.csv)
# ---------------------------------------------------------------
BETA_OPTIMO_EMPIRICO = 350.0  # data_beta_grid_search_SP/resumen_por_INL/resumen_optimos.csv


def heuristicas_beta(corrientes, nombre):
    b1 = abs(1 / np.mean(corrientes))
    b2 = abs(1 / (np.std(corrientes) + np.mean(corrientes)))
    b3 = 1 / np.sqrt(np.mean(corrientes ** 2))
    print(f"  [{nombre:<28}] beta~1/mean={b1:>9.2f}   beta~1/(std+mean)={b2:>9.2f}   "
          f"beta~1/RMS={b3:>9.2f}")
    return dict(fuente=nombre, mean=np.mean(corrientes), std=np.std(corrientes),
                rms=np.sqrt(np.mean(corrientes ** 2)),
                beta_1_sobre_mean=b1, beta_1_sobre_std_mas_mean=b2, beta_1_sobre_rms=b3)


print("\nHeuristicas de beta optimo (analisis_beta_optimo.py) por fuente de corriente:")
filas_heur = [
    heuristicas_beta(corrientes_mc, "Monte Carlo (sintetico)"),
    heuristicas_beta(I_epoca1, "Real, epoca 1 (~init)"),
    heuristicas_beta(I_epoca100, "Real, epoca 100 (convergida)"),
]
for f in filas_heur:
    f["beta_optimo_empirico"] = BETA_OPTIMO_EMPIRICO
print(f"\n  beta_optimo EMPIRICO real de esta curva (grid-search completo): {BETA_OPTIMO_EMPIRICO:.0f}")

with open(os.path.join(OUT_DIR, "resultados_reconstruccion.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["fuente", "mean", "std", "rms", "beta_1_sobre_mean",
                                       "beta_1_sobre_std_mas_mean", "beta_1_sobre_rms",
                                       "beta_optimo_empirico"])
    w.writeheader()
    w.writerows(filas_heur)
print(f"\nGuardado: resultados_reconstruccion.csv")

# ---------------------------------------------------------------
# 8) evolucion de mean/std de la corriente real a lo largo de las 100
#    epocas guardadas (para ver como se aleja del regimen "tipo MC")
# ---------------------------------------------------------------
mean_por_epoca = np.empty(len(saved_epochs))
std_por_epoca = np.empty(len(saved_epochs))
for epoch_idx in range(len(saved_epochs)):
    I_ep = corrientes_reales(epoch_idx)
    mean_por_epoca[epoch_idx] = I_ep.mean()
    std_por_epoca[epoch_idx] = I_ep.std()

# ---------------------------------------------------------------
# figuras
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.5), dpi=140)
bins = np.linspace(
    min(corrientes_mc.min(), I_epoca1.min(), I_epoca100.min()),
    max(corrientes_mc.max(), I_epoca1.max(), I_epoca100.max()), 80)
ax.hist(corrientes_mc, bins=bins, density=True, alpha=0.5, color="#7a7a7a",
        label=f"Monte Carlo sintetico (std={corrientes_mc.std():.2e})")
ax.hist(I_epoca1, bins=bins, density=True, alpha=0.55, color="#2a78d6",
        label=f"Real, epoca 1 (std={I_epoca1.std():.2e})")
ax.hist(I_epoca100, bins=bins, density=True, alpha=0.55, color="#eb6834",
        label=f"Real, epoca 100 (std={I_epoca100.std():.2e})")
ax.set_xlabel("corriente I (por muestra x neurona de salida)")
ax.set_ylabel("densidad")
ax.set_title("INL=1e-2, p=100, beta=1 -- corriente real vs. Monte Carlo sintetico")
ax.legend(frameon=False, fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "comparacion_corrientes_MC_vs_real.png"))
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=140)
axes[0].plot(saved_epochs, mean_por_epoca, color="#3aa66b")
axes[0].axhline(corrientes_mc.mean(), color="#7a7a7a", linestyle="--", label="Monte Carlo")
axes[0].set_xlabel("epoca"); axes[0].set_ylabel("mean(I)"); axes[0].legend(frameon=False, fontsize=8)
axes[1].plot(saved_epochs, std_por_epoca, color="#a34ad6")
axes[1].axhline(corrientes_mc.std(), color="#7a7a7a", linestyle="--", label="Monte Carlo")
axes[1].set_xlabel("epoca"); axes[1].set_ylabel("std(I)"); axes[1].legend(frameon=False, fontsize=8)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Evolucion de la corriente real durante el entrenamiento (serie 0, 5 folds)")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "evolucion_corriente_por_epoca.png"))
plt.close(fig)

print("\nGuardadas figuras: comparacion_corrientes_MC_vs_real.png, evolucion_corriente_por_epoca.png")
