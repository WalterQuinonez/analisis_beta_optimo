#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validar_beta_optimo.py

Objetivo (pedido en prompt.txt): validar, usando EXCLUSIVAMENTE datos ya
generados por corridas de entrenamiento COMPLETAS (100 epocas, k=5 folds, 20
series por beta -carpeta `data_beta_grid_search_SP/`, resumida en
`resumen_por_INL/resumen_optimos.csv`), la formula analitica de beta_optimo
derivada en `papers/Reporte_beta_optimo_analitico.pdf` a partir del argumento
de Prezioso et al. 2015 / LeCun et al. "Efficient Backprop":

    beta_optimo ~ kappa / ( sqrt(N) * sigma_W * V_rms )      (*)

    N       = numero de entradas de la capa (784 pixeles, arquitectura SP
              [784, 10], sin bias)
    sigma_W = desvio estandar de W_ij = G+_ij - G-_ij bajo la inicializacion
              aleatoria ya provista (G0_initialization('random', ...)),
              propia de cada curva P/D (depende de Gmin, Gmax, pulsos, a)
    V_rms   = sqrt(E[V^2]) de los voltajes de entrada, calculado sobre el
              mismo dataset ya usado en el grid search (X_train_mnist.npy)
    kappa   = constante O(1-10) (dispersion objetivo del argumento de la
              no linealidad, para que softmax(beta*I) no sature ni quede
              demasiado chato)

NO se corre ningun entrenamiento nuevo y no se agrega ningun metodo nuevo al
proyecto: se usan tal cual `generar_curvas_pot_dep` y `G0_initialization` de
MemCrossbarClass_beta_por_capa.py (import directo, sin copiar su codigo)
sobre los mismos parametros de curva P/D que ya us el grid search real (leidos
de los config_*.json de cada corrida), y sobre el mismo X_train_mnist.npy.
sigma_W se estima por Monte Carlo (una unica llamada barata a
G0_initialization con una muestra grande) porque no hay forma cerrada simple
para el desvio de la union de dos exponenciales truncadas asimetricas; esto
NO es una simulacion de entrenamiento, es una estadistica sobre una funcion
ya provista.

Uso:
    python validar_beta_optimo.py
Salidas (en esta misma carpeta):
    resultados_validacion_beta_optimo.csv
    beta_pred_vs_empirico.png
    beta_pred_vs_sigmaW.png
"""

import csv
import glob
import json
import os
import sys

import numpy as np
import torch

# --- Importar EXCLUSIVAMENTE funciones ya provistas por el proyecto ---
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from MemCrossbarClass_beta_por_capa import generar_curvas_pot_dep, G0_initialization  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO_ROOT, "data_beta_grid_search_SP")
RESUMEN_CSV = os.path.join(DATA_DIR, "resumen_por_INL", "resumen_optimos.csv")
X_TRAIN_PATH = os.path.join(REPO_ROOT, "X_train_mnist.npy")

# Parametros fisicos comunes a las 6 curvas del grid search (idem
# Crossbar_train_experimento_beta_por_capa.py / los config_*.json de cada
# corrida: Rlow=1000 ohm, Rhigh=10000 ohm, concavidad pot='pos'/dep='neg',
# arquitectura SP [784, 10], amplitud_imagen=1).
RLOW = 1000.0
RHIGH = 10000.0
GMIN = 1.0 / RHIGH
GMAX = 1.0 / RLOW
CONCAVIDAD_POT = "pos"
CONCAVIDAD_DEP = "neg"
N_IN = 784   # sizes[0] del MemDNN usado en el grid search (sin bias)
D_OUT = 10

SEED_MC = 35
N_MC_ROWS = 20000  # filas "D_in" ficticias para la corrida Monte Carlo de sigma_W


def cargar_configs_reales():
    """Para cada carpeta beta_grid_search_SP_INL_*_p_* de data_beta_grid_search_SP,
    lee pulsos_pot/a_pot desde su primer config_*.json real (no se reescriben
    a mano para evitar transcribir mal un numero)."""
    filas = {}
    for carpeta in sorted(glob.glob(os.path.join(DATA_DIR, "beta_grid_search_SP_INL_*"))):
        if not os.path.isdir(carpeta):
            continue
        config_paths = sorted(glob.glob(os.path.join(carpeta, "config_*.json")))
        if not config_paths:
            continue
        with open(config_paths[0], "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        cpd = cfg["curva_pd"]
        nombre = os.path.basename(carpeta)
        filas[nombre] = {
            "pulsos_pot": cpd["pulsos_pot"],
            "pulsos_dep": cpd["pulsos_dep"],
            "a_pot": cpd["a_pot"],
            "a_dep": cpd["a_dep"],
            "Gmin": cpd["Gmin"],
            "Gmax": cpd["Gmax"],
            "concavidad_pot": cpd["concavidad_pot"],
            "concavidad_dep": cpd["concavidad_dep"],
            "inl_pot": cpd["inl_pot"],
            "N_in": cfg["arquitectura"]["sizes"][0],
        }
    return filas


def sigma_W_montecarlo(pulsos_pot, pulsos_dep, a_pot, a_dep, Gmin, Gmax,
                        concavidad_pot, concavidad_dep, seed=SEED_MC,
                        n_rows=N_MC_ROWS, D_out=D_OUT):
    """Monte Carlo de sigma_W = std(G+_ij - G-_ij) usando EXCLUSIVAMENTE
    generar_curvas_pot_dep + G0_initialization('random', ...) ya provistas.
    De paso, devuelve tambien el "paso medio" |pot[i+1]-pot[i]| de la curva
    P/D generada: es un diagnostico EXPLORATORIO (no forma parte de la
    formula (*) del reporte analitico) que se usa en la Seccion de
    limitaciones para chequear la sugerencia del propio
    Reporte_beta_optimo_analitico.pdf (Sec. 5, punto 7): que la pendiente
    local de la curva P/D -no solo sigma_W- podria explicar mejor el efecto
    del numero de pulsos sobre beta_optimo."""
    pot, dep, ratio, inl_pot, inl_dep = generar_curvas_pot_dep(
        pulsos_pot, pulsos_dep, a_pot, a_dep, Gmin, Gmax,
        concavidad_dep, concavidad_pot)
    pot_t = torch.tensor(pot, dtype=torch.float64)
    dep_t = torch.tensor(dep, dtype=torch.float64)

    torch.manual_seed(seed)  # G0_initialization usa torch.randint global, sin generator explicito
    G = G0_initialization("random", pot_t, dep_t, n_rows, D_out)
    W = G[:, 0::2] - G[:, 1::2]
    sigma_W = float(W.flatten().std(unbiased=True))
    paso_medio_pot = float(np.mean(np.abs(np.diff(pot))))
    return sigma_W, inl_pot, inl_dep, paso_medio_pot


def main():
    configs = cargar_configs_reales()

    X_train = np.load(X_TRAIN_PATH)
    V_rms = float(np.sqrt(np.mean(X_train.astype(np.float64) ** 2)))
    print(f"V_rms (X_train_mnist.npy) = {V_rms:.6f}")

    empiricos = []
    with open(RESUMEN_CSV, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            empiricos.append(row)

    filas_out = []
    for row in empiricos:
        carpeta = row["carpeta"]
        cfg = configs[carpeta]
        N_in = cfg["N_in"]
        sigma_W, inl_pot, inl_dep, paso_medio_pot = sigma_W_montecarlo(
            cfg["pulsos_pot"], cfg["pulsos_dep"], cfg["a_pot"], cfg["a_dep"],
            cfg["Gmin"], cfg["Gmax"], cfg["concavidad_pot"], cfg["concavidad_dep"])

        beta_emp = float(row["beta_optimo"])
        acc_emp = float(row["acc_optima"])
        denom = np.sqrt(N_in) * sigma_W * V_rms
        beta_pred_unit = 1.0 / denom  # beta predicho con kappa=1
        kappa_implicado = beta_emp * denom
        # diagnostico exploratorio (ver docstring de sigma_W_montecarlo)
        kappa2_implicado_paso = beta_emp * np.sqrt(N_in) * paso_medio_pot * V_rms

        filas_out.append({
            "carpeta": carpeta,
            "inl_tag": row["inl_tag"],
            "pulsos_pot": cfg["pulsos_pot"],
            "a_pot": cfg["a_pot"],
            "inl_pot": inl_pot,
            "N_in": N_in,
            "sigma_W": sigma_W,
            "paso_medio_pot": paso_medio_pot,
            "V_rms": V_rms,
            "beta_pred_kappa1": beta_pred_unit,
            "beta_optimo_empirico": beta_emp,
            "acc_optima_empirico": acc_emp,
            "kappa_implicado": kappa_implicado,
            "kappa2_implicado_paso_medio": kappa2_implicado_paso,
        })

    kappas = np.array([f["kappa_implicado"] for f in filas_out])
    kappa_geomean = float(np.exp(np.mean(np.log(kappas))))
    print("\ncarpeta                                   INL     sigma_W(S)   beta_pred(k=1)  beta_emp   kappa_impl")
    for f in filas_out:
        print(f"{f['carpeta']:<42} {f['inl_tag']:<7} {f['sigma_W']:.4e}   {f['beta_pred_kappa1']:>10.2f}   "
              f"{f['beta_optimo_empirico']:>8.1f}   {f['kappa_implicado']:>8.2f}")
    print(f"\nkappa implicado (formula oficial, sigma_W): media geometrica = {kappa_geomean:.3f}, "
          f"rango = [{kappas.min():.2f}, {kappas.max():.2f}], "
          f"cociente max/min = {kappas.max()/kappas.min():.2f}")

    # --- diagnostico exploratorio: paso medio de la curva P/D en vez de sigma_W ---
    kappas2 = np.array([f["kappa2_implicado_paso_medio"] for f in filas_out])
    print("\n[diagnostico exploratorio, no forma parte de la formula (*)] "
          "kappa implicado usando paso_medio_pot en vez de sigma_W:")
    for f in filas_out:
        print(f"  {f['carpeta']:<42} paso_medio_pot={f['paso_medio_pot']:.3e}  "
              f"kappa2={f['kappa2_implicado_paso_medio']:.4f}")
    print(f"  rango = [{kappas2.min():.4f}, {kappas2.max():.4f}], "
          f"cociente max/min = {kappas2.max()/kappas2.min():.2f}")

    for f in filas_out:
        f["beta_pred_kappa_geomean"] = kappa_geomean / (np.sqrt(f["N_in"]) * f["sigma_W"] * f["V_rms"])

    # ---- ajuste log-log beta_emp vs 1/(sqrt(N)*sigma_W*V_rms) (pendiente esperada = 1) ----
    x = np.log(np.array([1.0 / (np.sqrt(f["N_in"]) * f["sigma_W"] * f["V_rms"]) for f in filas_out]))
    y = np.log(np.array([f["beta_optimo_empirico"] for f in filas_out]))
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    print(f"\najuste log-log beta_emp ~ (1/(sqrt(N) sigma_W V_rms))^slope: "
          f"slope={slope:.3f} (teoria=1), kappa_ajustado=exp(intercept)={np.exp(intercept):.3f}, R2={r2:.3f}")

    csv_out = os.path.join(HERE, "resultados_validacion_beta_optimo.csv")
    campos = ["carpeta", "inl_tag", "pulsos_pot", "a_pot", "inl_pot", "N_in",
              "sigma_W", "paso_medio_pot", "V_rms", "beta_pred_kappa1", "beta_pred_kappa_geomean",
              "beta_optimo_empirico", "acc_optima_empirico", "kappa_implicado",
              "kappa2_implicado_paso_medio"]
    with open(csv_out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        w.writerows(filas_out)
    print(f"\nGuardado: {csv_out}")

    # ---- figuras ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colores = {"1e-2": "#2a78d6", "2e-1": "#eb6834"}

    fig, ax = plt.subplots(figsize=(6.2, 5.6), dpi=140)
    lo = min(min(f["beta_optimo_empirico"] for f in filas_out),
              min(f["beta_pred_kappa_geomean"] for f in filas_out)) * 0.7
    hi = max(max(f["beta_optimo_empirico"] for f in filas_out),
              max(f["beta_pred_kappa_geomean"] for f in filas_out)) * 1.4
    ax.plot([lo, hi], [lo, hi], color="#999", linestyle="--", linewidth=1, label="identidad")
    for inl_tag, color in colores.items():
        pts = [f for f in filas_out if f["inl_tag"] == inl_tag]
        ax.scatter([f["beta_pred_kappa_geomean"] for f in pts],
                   [f["beta_optimo_empirico"] for f in pts],
                   s=70, color=color, label=f"INL nominal = {inl_tag}", zorder=3)
        for f in pts:
            ax.annotate(f"p={f['pulsos_pot']}", (f["beta_pred_kappa_geomean"], f["beta_optimo_empirico"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8, color=color)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\beta_{pred}$ = $\kappa$ / ($\sqrt{N}\,\sigma_W\,V_{rms}$),  $\kappa$=" + f"{kappa_geomean:.2f} (media geom.)")
    ax.set_ylabel(r"$\beta_{optimo}$ empirico (grid search real, 100 epocas)")
    ax.set_title("Formula analitica vs beta óptimo empírico (datos reales)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, which="both", color="#e8e7e0", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "beta_pred_vs_empirico.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=140)
    for inl_tag, color in colores.items():
        pts = [f for f in filas_out if f["inl_tag"] == inl_tag]
        ax.scatter([f["sigma_W"] for f in pts], [f["beta_optimo_empirico"] for f in pts],
                   s=70, color=color, label=f"INL nominal = {inl_tag}", zorder=3)
        for f in pts:
            ax.annotate(f"p={f['pulsos_pot']}", (f["sigma_W"], f["beta_optimo_empirico"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8, color=color)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\sigma_W$ (S) -- Monte Carlo sobre G0_initialization")
    ax.set_ylabel(r"$\beta_{optimo}$ empirico")
    ax.set_title(r"$\beta_{optimo}$ empirico vs $\sigma_W$ de cada curva P/D")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, which="both", color="#e8e7e0", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "beta_pred_vs_sigmaW.png"))
    plt.close(fig)

    print("Guardado: beta_pred_vs_empirico.png, beta_pred_vs_sigmaW.png")


if __name__ == "__main__":
    main()
