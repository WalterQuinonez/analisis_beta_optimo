#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
estimar_beta_optimo_propuesta.py

Busca una ecuacion candidata para beta_optimo combinando:
  (a) la IDEA de `analisis_beta_optimo.py` de estudiar la distribucion de
      pesos W y de la corriente Monte Carlo I = sum_j W_ij V_j para acotar
      la escala de beta, mejorada para no repetir sus dos sesgos (ver
      `analisis_corrientes_beta_optimo/reporte_correlacion_corrientes_beta.pdf`):
        - los pesos W se generan igual que en la red real, sorteando G+ y
          G- INDEPENDIENTEMENTE del mismo pool combinado {pot}U{dep}, via
          `G0_initialization('random', ...)` (ya provista) -- no como
          `pot[i]-dep[j]` combinatorio.
        - los voltajes V se muestrean de las imagenes REALES de
          X_train_mnist.npy (respetando su frecuencia real: ~81% de los
          pixeles son 0), no de `np.unique(X_train)` equipesado.
  (b) el "paso medio" de la curva P/D (mean(|pot[i+1]-pot[i]|)), que el
      reporte previo (Reporte_beta_optimo_analitico.pdf, Sec.5 punto 7 y
      validacion_formula_beta_optimo/, Sec.5.3) identifico como el
      descriptor que SI explica la dependencia de beta_optimo con el
      numero de pulsos p, algo que sigma_W (una propiedad estatica de la
      distribucion agrupada de conductancias) no puede explicar por
      construccion (es casi independiente de p).
  (c) los 6 beta_optimo EMPIRICOS reales (grid-search con entrenamiento
      completo: 100 epocas, k=5 folds, 20 series -
      data_beta_grid_search_SP/resumen_por_INL/resumen_optimos.csv), para
      AJUSTAR (regresion log-log) una ecuacion combinada y compararla
      contra usar sigma_W solo o paso_medio solo.

No se modifica ningun archivo existente del proyecto. Solo se importan
(sin copiar ni alterar) `generar_curvas_pot_dep` y `G0_initialization` de
`MemCrossbarClass_beta_por_capa.py`. No se corre ningun entrenamiento de
red neuronal: todo el computo es estadistica/Monte Carlo barata + una
regresion lineal en escala log-log sobre 6 puntos ya existentes.
"""
import os
import sys
import csv
import glob
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO_ROOT, "data_beta_grid_search_SP")
RESUMEN_CSV = os.path.join(DATA_DIR, "resumen_por_INL", "resumen_optimos.csv")
X_TRAIN_PATH = os.path.join(REPO_ROOT, "X_train_mnist.npy")

sys.path.insert(0, REPO_ROOT)
from MemCrossbarClass_beta_por_capa import generar_curvas_pot_dep, G0_initialization  # noqa: E402

N_IN = 784
D_OUT = 10
SEED = 35
MC_PESOS = 200_000     # tamano de la muestra Monte Carlo de pesos (D_in ficticio)
MC_CORRIENTES = 300_000  # cantidad de "corrientes" I simuladas (sumas de 784 terminos)
CHUNK = 30_000


def cargar_configs_reales():
    """Lee pulsos_pot/a_pot/Gmin/Gmax/concavidad de los config_*.json REALES
    de cada una de las 6 corridas de grid-search (no se transcriben a mano)."""
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
        filas[os.path.basename(carpeta)] = dict(
            pulsos_pot=cpd["pulsos_pot"], pulsos_dep=cpd["pulsos_dep"],
            a_pot=cpd["a_pot"], a_dep=cpd["a_dep"],
            Gmin=cpd["Gmin"], Gmax=cpd["Gmax"],
            concavidad_pot=cpd["concavidad_pot"], concavidad_dep=cpd["concavidad_dep"],
        )
    return filas


def estudiar_pesos_y_corriente(cfg, X_train_flat, seed):
    """Reproduce la IDEA de analisis_beta_optimo.py (distribucion de pesos W
    y Monte Carlo de la corriente I=sum V*W) pero corrigiendo sus 2 sesgos
    (ver docstring del modulo)."""
    pot, dep, ratio, inl_pot, inl_dep = generar_curvas_pot_dep(
        cfg["pulsos_pot"], cfg["pulsos_dep"], cfg["a_pot"], cfg["a_dep"],
        cfg["Gmin"], cfg["Gmax"], cfg["concavidad_dep"], cfg["concavidad_pot"])

    # --- distribucion de PESOS: G0_initialization real (G+ y G- ambos del
    # pool combinado {pot}U{dep}, como en la red real), no pot[i]-dep[j] ---
    pot_t = np.asarray(pot, dtype=np.float64)
    dep_t = np.asarray(dep, dtype=np.float64)
    import torch
    torch.manual_seed(seed)
    G = G0_initialization("random", torch.tensor(pot_t), torch.tensor(dep_t), MC_PESOS, 1)
    W = (G[:, 0] - G[:, 1]).numpy()
    sigma_W = W.std()
    mean_W = W.mean()
    paso_medio_pot = float(np.mean(np.abs(np.diff(pot))))

    # --- distribucion de CORRIENTE: Monte Carlo de I = sum_{j=1}^{784} Vj*Wj,
    # con Vj muestreado de PIXELES REALES (respeta frecuencia real) y Wj
    # muestreado del mismo G0_initialization de arriba (no de un histograma
    # equipesado) ---
    rng = np.random.default_rng(seed)
    corrientes = np.empty(MC_CORRIENTES)
    for start in range(0, MC_CORRIENTES, CHUNK):
        end = min(start + CHUNK, MC_CORRIENTES)
        n = end - start
        v_idx = rng.integers(0, X_train_flat.size, size=(n, N_IN))
        w_idx = rng.integers(0, W.size, size=(n, N_IN))
        corrientes[start:end] = (X_train_flat[v_idx] * W[w_idx]).sum(axis=1)

    return dict(
        pot=pot, dep=dep, inl_pot=inl_pot,
        sigma_W=sigma_W, mean_W=mean_W, paso_medio_pot=paso_medio_pot,
        mean_I=corrientes.mean(), sigma_I=corrientes.std(),
        corrientes_sample=corrientes[:20000],  # para graficar sin guardar 300k puntos x 6
    )


def main():
    configs = cargar_configs_reales()
    X_train = np.load(X_TRAIN_PATH).astype(np.float64)
    X_flat = X_train.ravel()
    V_rms_real = float(np.sqrt(np.mean(X_flat ** 2)))
    print(f"V_rms real (X_train_mnist.npy, pesado por frecuencia) = {V_rms_real:.6f}")

    empiricos = []
    with open(RESUMEN_CSV, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            empiricos.append(row)

    filas = []
    for row in empiricos:
        carpeta = row["carpeta"]
        cfg = configs[carpeta]
        r = estudiar_pesos_y_corriente(cfg, X_flat, seed=SEED)
        beta_emp = float(row["beta_optimo"])

        sigma_I_clt = np.sqrt(N_IN) * r["sigma_W"] * V_rms_real  # prediccion CLT, referencia
        kappa_mc = beta_emp * r["sigma_I"]       # kappa implicado usando sigma_I MEDIDO por MC (corregido)
        kappa_clt = beta_emp * sigma_I_clt        # kappa implicado usando la formula CLT de siempre

        filas.append(dict(
            carpeta=carpeta, inl_tag=row["inl_tag"], pulsos_pot=cfg["pulsos_pot"], a_pot=cfg["a_pot"],
            sigma_W=r["sigma_W"], mean_W=r["mean_W"], paso_medio_pot=r["paso_medio_pot"],
            mean_I_mc=r["mean_I"], sigma_I_mc=r["sigma_I"], sigma_I_clt=sigma_I_clt,
            beta_optimo_empirico=beta_emp,
            kappa_mc=kappa_mc, kappa_clt=kappa_clt,
            corrientes_sample=r["corrientes_sample"],
        ))
        print(f"{carpeta:<42} sigma_W={r['sigma_W']:.4e}  paso_medio={r['paso_medio_pot']:.3e}  "
              f"sigma_I_mc={r['sigma_I']:.4e} (CLT={sigma_I_clt:.4e})  mean_I_mc={r['mean_I']:+.2e}  "
              f"beta_emp={beta_emp:.0f}  kappa_mc={kappa_mc:.3f}")

    # ---------------------------------------------------------------
    # Regresiones log-log: beta_optimo ~ sigma_W^b * paso_medio^d
    # ---------------------------------------------------------------
    sigmaW = np.array([f["sigma_W"] for f in filas])
    paso = np.array([f["paso_medio_pot"] for f in filas])
    beta = np.array([f["beta_optimo_empirico"] for f in filas])
    y = np.log(beta)

    def ajustar(X, nombre):
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ coef
        r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
        print(f"\n[{nombre}] coef={coef}  R2={r2:.4f}")
        return coef, pred, r2

    X_sw = np.column_stack([np.ones(6), np.log(sigmaW)])
    coef_sw, pred_sw, r2_sw = ajustar(X_sw, "solo sigma_W")

    X_pm = np.column_stack([np.ones(6), np.log(paso)])
    coef_pm, pred_pm, r2_pm = ajustar(X_pm, "solo paso_medio")

    X_comb = np.column_stack([np.ones(6), np.log(sigmaW), np.log(paso)])
    coef_comb, pred_comb, r2_comb = ajustar(X_comb, "combinado (sigma_W y paso_medio)")

    beta_pred_comb = np.exp(pred_comb)
    print("\nbeta_optimo empirico vs. predicho por la ecuacion combinada:")
    for f, bp in zip(filas, beta_pred_comb):
        f["beta_pred_combinada"] = bp
        print(f"  {f['carpeta']:<42} emp={f['beta_optimo_empirico']:>5.0f}  pred={bp:>7.1f}  "
              f"error={100*(bp/f['beta_optimo_empirico']-1):+6.1f}%")

    # ---------------------------------------------------------------
    # guardar CSV de resultados
    # ---------------------------------------------------------------
    csv_out = os.path.join(OUT_DIR, "resultados_formula_propuesta.csv")
    campos = ["carpeta", "inl_tag", "pulsos_pot", "a_pot", "sigma_W", "mean_W", "paso_medio_pot",
              "mean_I_mc", "sigma_I_mc", "sigma_I_clt", "beta_optimo_empirico",
              "kappa_mc", "kappa_clt", "beta_pred_combinada"]
    with open(csv_out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        for f in filas:
            w.writerow({k: f[k] for k in campos})
    print(f"\nGuardado: {csv_out}")

    coef_path = os.path.join(OUT_DIR, "coeficientes_regresion.csv")
    with open(coef_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["modelo", "intercept", "coef_sigma_W", "coef_paso_medio", "R2"])
        w.writerow(["solo_sigma_W", coef_sw[0], coef_sw[1], "", r2_sw])
        w.writerow(["solo_paso_medio", coef_pm[0], "", coef_pm[1], r2_pm])
        w.writerow(["combinado", coef_comb[0], coef_comb[1], coef_comb[2], r2_comb])
    print(f"Guardado: {coef_path}")

    # ---------------------------------------------------------------
    # figuras
    # ---------------------------------------------------------------
    colores = {"1e-2": "#2a78d6", "2e-1": "#eb6834"}

    # (1) corrientes MC corregidas, antes de escalar (deberian estar ~centradas en 0 ahora)
    fig, axes = plt.subplots(1, 6, figsize=(20, 3.4), dpi=130, sharey=True)
    for ax, f in zip(axes, filas):
        ax.hist(f["corrientes_sample"], bins=60, color=colores[f["inl_tag"]], alpha=0.85)
        ax.axvline(0, color="#333", linewidth=0.8)
        ax.set_title(f"{f['inl_tag']}, p={f['pulsos_pot']}\nmean={f['mean_I_mc']:.1e}", fontsize=9)
        ax.set_xlabel("I (corregida)")
    fig.suptitle("Corriente Monte Carlo CORREGIDA (pixeles reales + G0_initialization real)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "corrientes_corregidas.png"))
    plt.close(fig)

    # (2) beta predicho (combinado) vs empirico
    fig, ax = plt.subplots(figsize=(6, 5.5), dpi=140)
    lo = min(beta.min(), beta_pred_comb.min()) * 0.7
    hi = max(beta.max(), beta_pred_comb.max()) * 1.3
    ax.plot([lo, hi], [lo, hi], "--", color="#999", label="identidad")
    for f, bp in zip(filas, beta_pred_comb):
        ax.scatter(bp, f["beta_optimo_empirico"], s=70, color=colores[f["inl_tag"]], zorder=3)
        ax.annotate(f"p={f['pulsos_pot']}", (bp, f["beta_optimo_empirico"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8, color=colores[f["inl_tag"]])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\beta_{pred}$ (ecuación combinada: $\sigma_W^{b}\cdot$paso$_{medio}^{d}$)")
    ax.set_ylabel(r"$\beta_{óptimo}$ empírico (grid-search real)")
    ax.set_title(f"Ecuación combinada vs. β óptimo empírico  (R²={r2_comb:.3f})")
    for tag, c in colores.items():
        ax.scatter([], [], color=c, label=f"INL nominal {tag}")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, which="both", color="#e8e7e0", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "beta_pred_combinada_vs_empirico.png"))
    plt.close(fig)

    # (3) comparacion R2 de los 3 modelos
    fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=140)
    modelos = ["solo\nsigma_W", "solo\npaso_medio", "combinado"]
    r2s = [r2_sw, r2_pm, r2_comb]
    ax.bar(modelos, r2s, color=["#2a78d6", "#eb6834", "#3f7d54"])
    for i, v in enumerate(r2s):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("R² (ajuste log-log a los 6 β óptimo reales)")
    ax.set_title("Comparación de modelos")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "comparacion_R2.png"))
    plt.close(fig)

    print("\nGuardadas figuras: corrientes_corregidas.png, "
          "beta_pred_combinada_vs_empirico.png, comparacion_R2.png")


if __name__ == "__main__":
    main()
