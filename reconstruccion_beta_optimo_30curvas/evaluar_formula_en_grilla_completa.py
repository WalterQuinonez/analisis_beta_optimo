#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluar_formula_en_grilla_completa.py

Analisis critico (paralelo al de reconstruccion_beta_optimo_18curvas/
reporte_reconstruccion_beta_optimo.tex, Seccion "Analisis critico"),
extendido a las 30 curvas: si se usara beta_pred de la ecuacion final
(coeficientes_todos_los_modelos_30curvas.csv, modelo
combinado_sigma_I_mc_corregido) SIN ninguna busqueda, en vez de
search_best_beta(), cuanta accuracy se pierde respecto del beta optimo
real de la grilla de 40 puntos?

Para cada una de las 30 curvas:
  1) Carga TODOS los resultados_*.npz de esa carpeta (accuracy final
     promediada sobre todas las series x folds disponibles -- mismo
     criterio que plot_accuracy_vs_beta_por_INL.py).
  2) Calcula beta_pred con la ecuacion final (usa sigma_I_mc_corregido y
     paso_medio_pot, ya calculados en resultados_30curvas.csv -- no hace
     falta Monte Carlo de nuevo).
  3) Busca el punto de la GRILLA REAL (1,50,100,...,1950) mas cercano a
     beta_pred, y compara su accuracy contra el pico real de esa curva.

Salida (en esta misma carpeta):
    evaluacion_formula_en_grilla.csv
    formula_vs_grilla_real.png
"""
import os
import csv
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(REPO_ROOT, "data_beta_grid_search_SP")

# ecuacion final (30 curvas): log(beta) = intercept + a*log(sigma_I_mc_corregido) + b*log(paso_medio_pot)
INTERCEPT, A_SIGMA, B_PASO = -18.07921592, -3.00239247, -0.61887854


def beta_pred_formula(sigma_i_mc, paso_medio):
    log_beta = INTERCEPT + A_SIGMA * np.log(sigma_i_mc) + B_PASO * np.log(paso_medio)
    return np.exp(log_beta)


def accuracy_por_beta(carpeta):
    """Mismo criterio que plot_accuracy_vs_beta_por_INL.py: accuracy final
    (ultima epoca) promediada sobre todas las series x folds disponibles,
    para cada beta de la grilla."""
    npz_paths = sorted(glob.glob(os.path.join(carpeta, "resultados_*.npz")))
    betas, medias = [], []
    for path in npz_paths:
        d = np.load(path, allow_pickle=True)
        acc = d["acc_result"]  # (series_completadas, k_folds, epocas)
        if acc.size == 0:
            continue
        beta = float(d["beta"])
        final_por_serie_fold = acc[:, :, -1]
        betas.append(beta)
        medias.append(final_por_serie_fold.reshape(-1).mean())
    orden = np.argsort(betas)
    return np.array(betas)[orden], np.array(medias)[orden]


def main():
    with open(os.path.join(HERE, "resultados_30curvas.csv"), encoding="utf-8") as fh:
        filas_30 = list(csv.DictReader(fh))

    resultados = []
    for f in filas_30:
        carpeta = os.path.join(DATA_DIR, f["carpeta"])
        betas_grid, acc_grid = accuracy_por_beta(carpeta)
        if len(betas_grid) == 0:
            print(f"  ({f['carpeta']}: sin datos de grilla, se omite)")
            continue

        sigma_i_mc = float(f["sigma_I_mc_corregido"])
        paso = float(f["paso_medio_pot"])
        beta_pred = beta_pred_formula(sigma_i_mc, paso)

        idx_pred = int(np.argmin(np.abs(betas_grid - beta_pred)))
        beta_grid_mas_cercano = betas_grid[idx_pred]
        acc_en_pred = acc_grid[idx_pred]

        idx_pico = int(np.argmax(acc_grid))
        beta_pico = betas_grid[idx_pico]
        acc_pico = acc_grid[idx_pico]

        pct_del_pico = 100 * acc_en_pred / acc_pico
        resultados.append(dict(
            carpeta=f["carpeta"], inl_tag=f["inl_tag"], pulsos_pot=int(float(f["pulsos_pot"])),
            beta_optimo_grilla=beta_pico, acc_pico=acc_pico,
            beta_pred_formula=beta_pred, beta_grid_mas_cercano=beta_grid_mas_cercano,
            acc_en_beta_pred=acc_en_pred, pct_del_pico=pct_del_pico,
        ))
        print(f"{f['carpeta']:<40} beta_optimo(grilla)={beta_pico:>5.0f}  "
              f"beta_pred(formula)={beta_pred:>7.1f} -> grilla mas cercana={beta_grid_mas_cercano:>5.0f}  "
              f"acc_pico={acc_pico*100:.2f}%  acc_en_pred={acc_en_pred*100:.2f}%  "
              f"({pct_del_pico:.1f}% del pico)")

    n_dentro_99 = sum(1 for r in resultados if r["pct_del_pico"] >= 99.0)
    n_dentro_95 = sum(1 for r in resultados if r["pct_del_pico"] >= 95.0)
    print(f"\n{n_dentro_99}/{len(resultados)} curvas: beta_pred cae en un punto de grilla con "
          f">=99% del accuracy pico")
    print(f"{n_dentro_95}/{len(resultados)} curvas: beta_pred cae en un punto de grilla con "
          f">=95% del accuracy pico")
    print(f"pct_del_pico: media={np.mean([r['pct_del_pico'] for r in resultados]):.2f}%  "
          f"minimo={np.min([r['pct_del_pico'] for r in resultados]):.2f}%")

    csv_path = os.path.join(HERE, "evaluacion_formula_en_grilla.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["carpeta", "inl_tag", "pulsos_pot", "beta_optimo_grilla",
                                           "acc_pico", "beta_pred_formula", "beta_grid_mas_cercano",
                                           "acc_en_beta_pred", "pct_del_pico"])
        w.writeheader()
        w.writerows(resultados)
    print(f"\nGuardado: {csv_path}")

    # figura: pct_del_pico por curva, ordenado
    resultados_ord = sorted(resultados, key=lambda r: r["pct_del_pico"])
    etiquetas = [f"{r['inl_tag']}, p={r['pulsos_pot']}" for r in resultados_ord]
    valores = [r["pct_del_pico"] for r in resultados_ord]
    colores = ["#d64a6a" if v < 95 else ("#c9a227" if v < 99 else "#3aa66b") for v in valores]
    fig, ax = plt.subplots(figsize=(9, 9), dpi=140)
    ax.barh(etiquetas, valores, color=colores)
    ax.axvline(99, color="#3aa66b", linestyle="--", linewidth=1, label="99% del pico")
    ax.axvline(95, color="#c9a227", linestyle="--", linewidth=1, label="95% del pico")
    ax.set_xlim(min(95, min(valores) - 0.3), 100.3)
    ax.set_xlabel("accuracy en beta_pred, como % del accuracy pico de la grilla")
    ax.set_title("¿Cuanta accuracy se pierde usando beta_pred de la formula,\nsin ninguna busqueda? (30 curvas)")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.tick_params(labelsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "formula_vs_grilla_real.png"))
    plt.close(fig)
    print("Guardado: formula_vs_grilla_real.png")


if __name__ == "__main__":
    main()
