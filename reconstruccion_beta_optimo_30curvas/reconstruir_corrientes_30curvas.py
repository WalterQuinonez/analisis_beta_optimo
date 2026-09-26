#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reconstruir_corrientes_30curvas.py

Extiende reconstruccion_beta_optimo_18curvas/reconstruir_corrientes_18curvas.py
a las 30 curvas que hay ahora en `data_beta_grid_search_SP/` (10 INL nominal
x 3 pulsos: se agregaron 5e-2, 5e-3, 5e-4, 5e-5 a las 6 originales
2e-1, 1e-2, 1e-3, 1e-4, 1e-5, 4.82e-6). Mismo metodo exacto, sin cambios de
fondo -- codigo identico salvo la paleta de colores (10 INL en vez de 6) y
el tamano del panel de trayectorias.

Para cada curva, con la corrida real de beta=1 ("beta00_b1", presente en
las 30 carpetas):

  1) Reconstruye la corriente REAL (I, antes de beta) en la PRIMERA y
     ULTIMA epoca guardada, juntando los 5 folds de la serie 0 (cubren el
     dataset completo), a partir de los pesos reales guardados
     (G_history_..._serie000.npz) y la particion de folds reconstruida con
     el fold_seed guardado en el config (100% reproducible sin reentrenar).
  2) Calcula el Monte Carlo CORREGIDO de `formula_beta_optimo_propuesta/
     estimar_beta_optimo_propuesta_v2.py` (G0_initialization real + pixeles
     reales de X_train_mnist.npy) para la misma curva P/D.
  3) Calcula el Monte Carlo SESGADO de `analisis_beta_optimo.py` (pesos
     combinatorios pot[i]-dep[j] + voltajes equipesados de np.unique) para
     la misma curva P/D.

Salida (en esta misma carpeta):
  resultados_30curvas.csv       -- una fila por curva, con las 3 fuentes de
                                    sigma/mean de corriente + metadata.
  trayectorias_std_30curvas.png -- panel 10x3 (INL x pulsos) de std(I) real
                                    vs epoca, con lineas de referencia MC.
  comparacion_sigma_fuentes_30curvas.png -- sigma por curva, las 3 fuentes juntas.
"""
import os
import sys
import csv
import glob
import json

import numpy as np
import torch
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO_ROOT, "data_beta_grid_search_SP")
RESUMEN_CSV = os.path.join(DATA_DIR, "resumen_por_INL", "resumen_optimos.csv")
X_TRAIN_PATH = os.path.join(REPO_ROOT, "X_train_mnist.npy")
Y_TRAIN_PATH = os.path.join(REPO_ROOT, "y_train_mnist.npy")

sys.path.insert(0, REPO_ROOT)
from MemCrossbarClass_beta_por_capa import generar_curvas_pot_dep, G0_initialization  # noqa: E402

SERIE = 0
N_IN = 784
SEED = 35
MC_PESOS_CORREGIDO = 200_000
MC_CORRIENTES_CORREGIDO = 300_000
CHUNK_CORREGIDO = 30_000
MUESTRAS_SESGADO = 1_000_000
CHUNK_SESGADO = 50_000

# paleta consistente con plot_accuracy_vs_beta_por_pulsos.py (mismo INL, mismo color en todo el proyecto)
COLOR_POR_INL = {
    "2e-1": "#d64a6a", "1e-2": "#c9a227", "5e-2": "#e8823f", "5e-3": "#8bbf3f",
    "1e-3": "#3aa66b", "5e-4": "#3fbfa0", "1e-4": "#4ac9c2", "5e-5": "#4a8fd6",
    "1e-5": "#2a78d6", "4.82e-6": "#a34ad6",
}

# ---------------------------------------------------------------
# dataset real (una sola vez)
# ---------------------------------------------------------------
X = torch.from_numpy(np.load(X_TRAIN_PATH)).float()
y = torch.from_numpy(np.load(Y_TRAIN_PATH)).long()
X_flat = X.numpy().ravel().astype(np.float64)
V_rms_real = float(np.sqrt(np.mean(X_flat ** 2)))
voltajes_equipesados = np.unique(X_flat)
print(f"X_train_mnist.npy: {tuple(X.shape)}   V_rms real = {V_rms_real:.6f}")


def forward_I(G_arr, X_sub):
    """I = x @ (G_pos - G_neg), identico a MemDNN.forward() con beta=1."""
    G = torch.from_numpy(G_arr).float()
    W_eff = G[:, 0::2] - G[:, 1::2]
    return (X_sub @ W_eff).numpy()


def corriente_mc_corregida(pot, dep, rng):
    """Metodo de estimar_beta_optimo_propuesta_v2.py: pesos via
    G0_initialization real (pool combinado, D_in sintetico grande) + voltajes
    muestreados de PIXELES REALES (respetando frecuencia real)."""
    pot_t = np.asarray(pot, dtype=np.float64)
    dep_t = np.asarray(dep, dtype=np.float64)
    torch.manual_seed(SEED)
    G = G0_initialization("random", torch.tensor(pot_t), torch.tensor(dep_t), MC_PESOS_CORREGIDO, 1)
    W = (G[:, 0] - G[:, 1]).numpy()
    corrientes = np.empty(MC_CORRIENTES_CORREGIDO)
    for start in range(0, MC_CORRIENTES_CORREGIDO, CHUNK_CORREGIDO):
        end = min(start + CHUNK_CORREGIDO, MC_CORRIENTES_CORREGIDO)
        n = end - start
        v_idx = rng.integers(0, X_flat.size, size=(n, N_IN))
        w_idx = rng.integers(0, W.size, size=(n, N_IN))
        corrientes[start:end] = (X_flat[v_idx] * W[w_idx]).sum(axis=1)
    return W.std(), W.mean(), corrientes


def corriente_mc_sesgada(pot, dep, rng):
    """Metodo original de analisis_beta_optimo.py: wij combinatorio
    pot[i]-dep[j] + voltajes de np.unique(X_train) (equipesado)."""
    wij = np.subtract.outer(pot, dep)  # (len(pot), len(dep))
    x_mc = np.multiply.outer(voltajes_equipesados, wij.flatten()).ravel()
    corrientes = np.empty(MUESTRAS_SESGADO)
    for start in range(0, MUESTRAS_SESGADO, CHUNK_SESGADO):
        end = min(start + CHUNK_SESGADO, MUESTRAS_SESGADO)
        n = end - start
        idx = rng.integers(0, x_mc.size, size=(n, N_IN))
        corrientes[start:end] = x_mc[idx].sum(axis=1)
    return corrientes


def main():
    empiricos = {}
    with open(RESUMEN_CSV, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            empiricos[row["carpeta"]] = row

    carpetas = sorted(glob.glob(os.path.join(DATA_DIR, "beta_grid_search_SP_INL_*")))
    filas = []
    trayectorias = {}

    for carpeta in carpetas:
        nombre = os.path.basename(carpeta)
        if nombre not in empiricos:
            continue
        row_emp = empiricos[nombre]

        res_paths = glob.glob(os.path.join(carpeta, "resultados_*_beta00_b1.npz"))
        if len(res_paths) != 1:
            print(f"  ({nombre}: esperaba 1 resultados_*_beta00_b1.npz, hay {len(res_paths)}, se omite)")
            continue
        exp_id = os.path.basename(res_paths[0])[len("resultados_"):-len(".npz")]

        with open(os.path.join(carpeta, f"config_{exp_id}.json"), "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        resultados = np.load(res_paths[0], allow_pickle=True)
        g_hist = np.load(os.path.join(carpeta, f"G_history_{exp_id}_serie{SERIE:03d}.npz"))
        G_all = g_hist["G_layer0_history"]         # (k_folds, n_saved, 784, 20)
        saved_epochs = resultados["saved_epochs"]
        acc_result = resultados["acc_result"]

        k_folds = cfg["entrenamiento"]["k_folds"]
        fold_seed = cfg["folds"]["fold_seeds_por_serie"][SERIE]
        beta_value = cfg["beta"]["valor"]
        assert beta_value == 1.0, f"{nombre}: se esperaba beta=1.0, hay {beta_value}"
        Gmin, Gmax = cfg["curva_pd"]["Gmin"], cfg["curva_pd"]["Gmax"]
        a_pot, a_dep = cfg["curva_pd"]["a_pot"], cfg["curva_pd"]["a_dep"]
        pulsos_pot, pulsos_dep = cfg["curva_pd"]["pulsos_pot"], cfg["curva_pd"]["pulsos_dep"]

        kf = KFold(n_splits=k_folds, shuffle=True, random_state=fold_seed)
        splits = list(kf.split(np.arange(len(X))))

        # --- sanity check rapido (1 fold, 1a y ultima epoca) ---
        train_idx, val_idx = splits[0]
        for epoch_idx_chk in (0, len(saved_epochs) - 1):
            I_chk = forward_I(G_all[0, epoch_idx_chk], X[val_idx])
            acc_chk = (I_chk.argmax(1) == y[val_idx].numpy()).mean()
            acc_guardada = float(acc_result[SERIE, 0, int(saved_epochs[epoch_idx_chk]) - 1])
            assert abs(acc_chk - acc_guardada) < 1e-6, f"{nombre}: sanity check fallo ({acc_chk} vs {acc_guardada})"

        # --- corriente real: 1ra y ultima epoca, todos los folds ---
        def corrientes_reales(epoch_idx):
            piezas = []
            for fold in range(k_folds):
                _, v_idx = splits[fold]
                piezas.append(forward_I(G_all[fold, epoch_idx], X[v_idx]).ravel())
            return np.concatenate(piezas)

        I_ep1 = corrientes_reales(0)
        I_epLast = corrientes_reales(len(saved_epochs) - 1)

        mean_traj = np.empty(len(saved_epochs))
        std_traj = np.empty(len(saved_epochs))
        for i in range(len(saved_epochs)):
            I_i = corrientes_reales(i)
            mean_traj[i] = I_i.mean()
            std_traj[i] = I_i.std()
        trayectorias[nombre] = dict(saved_epochs=saved_epochs, mean_traj=mean_traj, std_traj=std_traj)

        # --- MC corregido y MC sesgado ---
        pot, dep, ratio, inl_pot, inl_dep = generar_curvas_pot_dep(
            pulsos_pot, pulsos_dep, a_pot, a_dep, Gmin, Gmax,
            cfg["curva_pd"]["concavidad_dep"], cfg["curva_pd"]["concavidad_pot"])
        paso_medio_pot = float(np.mean(np.abs(np.diff(pot))))

        rng = np.random.default_rng(SEED)
        sigma_W_corregido, mean_W_corregido, corrientes_mc_corr = corriente_mc_corregida(pot, dep, rng)
        corrientes_mc_ses = corriente_mc_sesgada(pot, dep, rng)

        fila = dict(
            carpeta=nombre, inl_tag=row_emp["inl_tag"], inl_pot_real=inl_pot,
            pulsos_pot=pulsos_pot, a_pot=a_pot, paso_medio_pot=paso_medio_pot,
            beta_optimo_empirico=float(row_emp["beta_optimo"]),
            sigma_W_corregido=sigma_W_corregido, mean_W_corregido=mean_W_corregido,
            mean_I_mc_corregido=corrientes_mc_corr.mean(), sigma_I_mc_corregido=corrientes_mc_corr.std(),
            mean_I_mc_sesgado=corrientes_mc_ses.mean(), sigma_I_mc_sesgado=corrientes_mc_ses.std(),
            mean_I_real_ep1=I_ep1.mean(), sigma_I_real_ep1=I_ep1.std(),
            mean_I_real_epLast=I_epLast.mean(), sigma_I_real_epLast=I_epLast.std(),
            n_epoca_last=int(saved_epochs[-1]),
        )
        filas.append(fila)
        print(f"{nombre:<40} sigma: MC_corr={fila['sigma_I_mc_corregido']:.4e}  "
              f"MC_sesg={fila['sigma_I_mc_sesgado']:.4e}  real_ep1={fila['sigma_I_real_ep1']:.4e}  "
              f"real_epLast={fila['sigma_I_real_epLast']:.4e}   beta_emp={fila['beta_optimo_empirico']:.0f}")

    print(f"\nTotal curvas procesadas: {len(filas)}")

    # ---------------------------------------------------------------
    # CSV
    # ---------------------------------------------------------------
    campos = ["carpeta", "inl_tag", "inl_pot_real", "pulsos_pot", "a_pot", "paso_medio_pot",
              "beta_optimo_empirico", "sigma_W_corregido", "mean_W_corregido",
              "mean_I_mc_corregido", "sigma_I_mc_corregido", "mean_I_mc_sesgado", "sigma_I_mc_sesgado",
              "mean_I_real_ep1", "sigma_I_real_ep1", "mean_I_real_epLast", "sigma_I_real_epLast",
              "n_epoca_last"]
    csv_out = os.path.join(OUT_DIR, "resultados_30curvas.csv")
    with open(csv_out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        for f in filas:
            w.writerow({k: f[k] for k in campos})
    print(f"\nGuardado: {csv_out}")

    # ---------------------------------------------------------------
    # figura 1: trayectorias std(I) vs epoca, panel 10 INL x 3 pulsos
    # ---------------------------------------------------------------
    inl_tags_orden = ["2e-1", "5e-2", "1e-2", "5e-3", "1e-3", "5e-4", "1e-4", "5e-5", "1e-5", "4.82e-6"]
    pulsos_orden = [50, 100, 200]
    fig, axes = plt.subplots(len(inl_tags_orden), len(pulsos_orden), figsize=(11, 24), dpi=110, sharex=True)
    filas_por_nombre = {f["carpeta"]: f for f in filas}
    for i, inl_tag in enumerate(inl_tags_orden):
        for j, p in enumerate(pulsos_orden):
            nombre = f"beta_grid_search_SP_INL_{inl_tag}_p_{p}"
            ax = axes[i, j]
            if nombre not in trayectorias:
                ax.set_visible(False)
                continue
            tr = trayectorias[nombre]
            f = filas_por_nombre[nombre]
            ax.plot(tr["saved_epochs"], tr["std_traj"], color=COLOR_POR_INL[inl_tag], linewidth=1.6)
            ax.axhline(f["sigma_I_mc_corregido"], color="#3aa66b", linestyle="--", linewidth=1, alpha=0.8)
            ax.axhline(f["sigma_I_mc_sesgado"], color="#7a7a7a", linestyle=":", linewidth=1, alpha=0.8)
            ax.set_title(f"INL={inl_tag}, p={p}", fontsize=8)
            ax.tick_params(labelsize=7)
            if i == len(inl_tags_orden) - 1:
                ax.set_xlabel("epoca", fontsize=8)
            if j == 0:
                ax.set_ylabel("std(I)", fontsize=8)
    fig.suptitle("std(I) real vs epoca (linea solida) -- MC corregido (verde discontinua) vs "
                 "MC sesgado (gris punteada)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "trayectorias_std_30curvas.png"))
    plt.close(fig)

    # ---------------------------------------------------------------
    # figura 2: comparacion de sigma por curva, las fuentes juntas
    # ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(18, 6), dpi=130)
    nombres_orden = [f"beta_grid_search_SP_INL_{t}_p_{p}" for t in inl_tags_orden for p in pulsos_orden]
    xs = np.arange(len(nombres_orden))
    ancho = 0.2
    fuentes = [("sigma_I_mc_sesgado", "MC sesgado (analisis_beta_optimo.py)", "#7a7a7a"),
               ("sigma_I_mc_corregido", "MC corregido (G0_init + pixeles reales)", "#3aa66b"),
               ("sigma_I_real_ep1", "Real, epoca 1", "#2a78d6"),
               ("sigma_I_real_epLast", "Real, epoca final", "#eb6834")]
    for k, (campo, etiqueta, color) in enumerate(fuentes):
        valores = [filas_por_nombre[n][campo] for n in nombres_orden if n in filas_por_nombre]
        ax.bar(xs[:len(valores)] + (k - 1.5) * ancho, valores, width=ancho, color=color, label=etiqueta)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([n.replace("beta_grid_search_SP_INL_", "") for n in nombres_orden],
                        rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("sigma(I) [S]")
    ax.set_title("sigma(I): Monte Carlo (sesgado y corregido) vs. reconstruccion real, 30 curvas")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "comparacion_sigma_fuentes_30curvas.png"))
    plt.close(fig)

    print("\nGuardadas figuras: trayectorias_std_30curvas.png, comparacion_sigma_fuentes_30curvas.png")


if __name__ == "__main__":
    main()
