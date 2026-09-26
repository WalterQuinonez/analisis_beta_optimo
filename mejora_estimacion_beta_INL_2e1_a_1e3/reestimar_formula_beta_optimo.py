#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reestimar_formula_beta_optimo.py

Re-estudio de la formula analitica de beta_optimo (`formula_beta_optimo_propuesta/`,
usada hoy en produccion via `beta_pred_formula()` en MemCrossbarClass_beta_por_capa.py),
pedido porque la estimacion parecia estar fallando para el regimen de INL alto
(INL~2e-1, a_pot=5.85, p=50): la busqueda automatica (`search_best_beta_mejorado`)
eligio beta=1389 mientras la formula predecia 86.8.

Este script NO corre ningun entrenamiento nuevo (Fuerza_bruta): reutiliza el
ground truth de grid-search por fuerza bruta que YA EXISTE en
`data_beta_grid_search_SP/` (30 combos: 10 valores de INL nominal, de 2e-1 a
4.82e-6, x 3 cantidades de pulsos), igual que hizo
`formula_beta_optimo_propuesta/estimar_beta_optimo_propuesta_v2.py`, pero:

  1) usa las 30 combinaciones (esa v2 uso solo 18, las unicas que existian en
     ese momento; las otras 12 -5e-2,5e-3,5e-4,5e-5- se agregaron despues, ver
     `mejora_search_best_beta/validacion_formula_v2_30combos.csv`, que SI las
     valida pero no las usa para *reajustar* los coeficientes).
  2) evalua el ajuste restringido al RANGO pedido, INL en [1e-3, 2e-1] (5 de
     los 10 valores de INL nominal x 3 pulsos = 15 combos), en vez de mirar
     solo el error agregado sobre las 30.
  3) usa validacion cruzada leave-one-out (LOOCV) para el error "fuera de
     muestra" de cada modelo -no solo R2 dentro de muestra, que siempre
     mejora al agregar mas terminos aunque no generalice mejor-.

Documenta ademas (Seccion "batch_number", ver informe) el hallazgo por el que
en realidad se disparo este pedido: beta_optimo depende fuertemente de
`batch_number` (pasos de gradiente por epoca), una variable que NI la formula
NI el ground truth de fuerza bruta (siempre corrido con batch_number=32)
contemplan. La formula predijo bien el optimo real para batch_number=32
(86.8 vs 100, ratio 0.868, dentro de lo esperado); el numero "raro" (1389) lo
eligio la BUSQUEDA automatica corriendo con batch_number=1, un regimen para
el que no existe ningun ground truth (ver `estudio_batch_number.py`, en esta
misma carpeta, chequeo empirico chico y barato para esto).
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

sys.path.insert(0, REPO_ROOT)
from MemCrossbarClass_beta_por_capa import generar_curvas_pot_dep, G0_initialization  # noqa: E402

N_IN = 784
SEED = 35
MC_PESOS = 200_000  # tamano de la muestra Monte Carlo de pesos (D_in ficticio)

# Rango de INL pedido para este re-estudio.
INL_TAGS_RANGO = ["2e-1", "5e-2", "1e-2", "5e-3", "1e-3"]

COLOR_POR_INL = {
    "2e-1": "#d64a6a", "5e-2": "#e08a2c", "1e-2": "#c9a227", "5e-3": "#8fae2a",
    "1e-3": "#3aa66b", "1e-4": "#4ac9c2", "5e-4": "#2ab0a0", "1e-5": "#2a78d6",
    "5e-5": "#5566d6", "4.82e-6": "#a34ad6",
}
MARKER_POR_P = {50: "o", 100: "s", 200: "^"}


def cargar_configs_reales():
    """Lee pulsos_pot/a_pot/Gmin/Gmax/concavidad de los config_*.json REALES
    de cada corrida de grid-search (no se transcriben a mano)."""
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
            batch_number=cfg["entrenamiento"]["batch_number"],
            epochs=cfg["entrenamiento"]["epochs"],
        )
    return filas


def sigma_W_y_paso_medio(cfg, seed=SEED):
    """Mismo Monte Carlo (barato, sin entrenar nada) que
    `estimar_beta_optimo_propuesta_v2.py` / `estimar_sigma_W_y_paso_medio()`
    en MemCrossbarClass_beta_por_capa.py."""
    import torch
    pot, dep, ratio, inl_pot, inl_dep = generar_curvas_pot_dep(
        cfg["pulsos_pot"], cfg["pulsos_dep"], cfg["a_pot"], cfg["a_dep"],
        cfg["Gmin"], cfg["Gmax"], cfg["concavidad_dep"], cfg["concavidad_pot"])
    torch.manual_seed(seed)
    G = G0_initialization("random", torch.tensor(np.asarray(pot, dtype=np.float32)),
                           torch.tensor(np.asarray(dep, dtype=np.float32)), MC_PESOS, 1)
    W = (G[:, 0] - G[:, 1]).numpy()
    sigma_W = float(W.std())
    paso_medio = float(np.mean(np.abs(np.diff(pot))))
    return sigma_W, paso_medio, inl_pot


def ajustar_loocv(X, y):
    """Ajuste log-log por minimos cuadrados + LOOCV manual (n chico, se
    recalcula el ajuste n veces sin el punto i-esimo). Devuelve:
      coef (ajuste con TODOS los puntos), pred_in (dentro de muestra),
      pred_loo (para cada punto, prediccion SIN usarlo para ajustar), R2_in,
      R2_loo."""
    n = len(y)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred_in = X @ coef

    pred_loo = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        coef_i, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
        pred_loo[i] = X[i] @ coef_i

    r2_in = 1 - np.sum((y - pred_in) ** 2) / np.sum((y - y.mean()) ** 2)
    r2_loo = 1 - np.sum((y - pred_loo) ** 2) / np.sum((y - y.mean()) ** 2)
    return coef, pred_in, pred_loo, r2_in, r2_loo


def resumen_error(nombre, beta_emp, beta_pred):
    ratio = beta_pred / beta_emp
    log_ratio = np.log(ratio)
    print(f"  [{nombre}] ratio mediana={np.median(ratio):.3f}  "
          f"min={ratio.min():.3f}  max={ratio.max():.3f}  "
          f"RMSE(log)={np.sqrt(np.mean(log_ratio**2)):.3f}  "
          f"% dentro de 1.5x={100*np.mean((ratio>=1/1.5)&(ratio<=1.5)):.0f}%  "
          f"% dentro de 2x={100*np.mean((ratio>=0.5)&(ratio<=2)):.0f}%")
    return dict(nombre=nombre, ratio_mediana=float(np.median(ratio)),
                ratio_min=float(ratio.min()), ratio_max=float(ratio.max()),
                rmse_log=float(np.sqrt(np.mean(log_ratio ** 2))))


def main():
    configs = cargar_configs_reales()
    empiricos = []
    with open(RESUMEN_CSV, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            empiricos.append(row)
    print(f"Combos encontrados en resumen_optimos.csv: {len(empiricos)}")

    filas = []
    for row in empiricos:
        carpeta = row["carpeta"]
        cfg = configs[carpeta]
        sigma_W, paso_medio, inl_pot = sigma_W_y_paso_medio(cfg)
        filas.append(dict(
            carpeta=carpeta, inl_tag=row["inl_tag"], inl_pot_real=float(row["inl_pot_real"]),
            pulsos_pot=cfg["pulsos_pot"], a_pot=cfg["a_pot"],
            batch_number=cfg["batch_number"], epochs=cfg["epochs"],
            sigma_W=sigma_W, paso_medio=paso_medio,
            beta_optimo_empirico=float(row["beta_optimo"]),
        ))

    filas.sort(key=lambda f: (f["inl_pot_real"], f["pulsos_pot"]))
    for f in filas:
        print(f"  {f['carpeta']:<42} sigma_W={f['sigma_W']:.4e}  paso_medio={f['paso_medio']:.3e}  "
              f"beta_emp={f['beta_optimo_empirico']:.0f}")

    sigmaW_all = np.array([f["sigma_W"] for f in filas])
    paso_all = np.array([f["paso_medio"] for f in filas])
    beta_all = np.array([f["beta_optimo_empirico"] for f in filas])
    y_all = np.log(beta_all)
    en_rango = np.array([f["inl_tag"] in INL_TAGS_RANGO for f in filas])
    print(f"\nCombos en el rango pedido (INL 1e-3 .. 2e-1): {en_rango.sum()} de {len(filas)}")

    def X_comb(sw, pm):
        return np.column_stack([np.ones(len(sw)), np.log(sw), np.log(pm)])

    # ------------------------------------------------------------------
    # Modelo A: formula HOY EN PRODUCCION (coeficientes de
    # coeficientes_regresion_v2.csv, ajustada sobre 18 curvas en su momento).
    # Se evalua tal cual (sin reajustar) sobre el rango pedido.
    # ------------------------------------------------------------------
    INTERCEPT_V2 = -25.047163289759965
    COEF_LOG_SIGMA_W_V2 = -3.0474631066320534
    COEF_LOG_PASO_MEDIO_V2 = -0.6071708626364334

    def beta_pred_v2(sw, pm):
        return np.exp(INTERCEPT_V2 + COEF_LOG_SIGMA_W_V2 * np.log(sw) + COEF_LOG_PASO_MEDIO_V2 * np.log(pm))

    print("\n=== Modelo A: formula v2 actual (produccion), SIN reajustar ===")
    beta_pred_A_rango = beta_pred_v2(sigmaW_all[en_rango], paso_all[en_rango])
    err_A = resumen_error("v2_produccion (evaluada en rango 1e-3..2e-1)",
                           beta_all[en_rango], beta_pred_A_rango)

    # ------------------------------------------------------------------
    # Modelo B: reajuste con las 30 combinaciones (todas las que hay hoy),
    # LOOCV global, evaluado en el sub-rango pedido.
    # ------------------------------------------------------------------
    print("\n=== Modelo B: reajuste con las 30 combinaciones completas (LOOCV) ===")
    Xb = X_comb(sigmaW_all, paso_all)
    coef_b, pred_in_b, pred_loo_b, r2_in_b, r2_loo_b = ajustar_loocv(Xb, y_all)
    print(f"  coef={coef_b}  R2_in={r2_in_b:.4f}  R2_loocv={r2_loo_b:.4f}")
    err_B_rango = resumen_error("reajuste_30combos (LOOCV, evaluado en rango)",
                                 beta_all[en_rango], np.exp(pred_loo_b[en_rango]))

    # ------------------------------------------------------------------
    # Modelo C: reajuste SOLO con las 15 combinaciones del rango pedido
    # (INL 1e-3..2e-1), LOOCV dentro del rango -especializado en vez de
    # global-.
    # ------------------------------------------------------------------
    print("\n=== Modelo C: reajuste SOLO con las 15 combinaciones del rango (LOOCV) ===")
    Xc = X_comb(sigmaW_all[en_rango], paso_all[en_rango])
    yc = y_all[en_rango]
    coef_c, pred_in_c, pred_loo_c, r2_in_c, r2_loo_c = ajustar_loocv(Xc, yc)
    print(f"  coef={coef_c}  R2_in={r2_in_c:.4f}  R2_loocv={r2_loo_c:.4f}")
    err_C = resumen_error("reajuste_rango_1e-3_a_2e-1 (LOOCV)",
                           beta_all[en_rango], np.exp(pred_loo_c))

    # ------------------------------------------------------------------
    # Comparacion final + guardado
    # ------------------------------------------------------------------
    print("\n=== Resumen comparativo (todos evaluados/validados SOLO en el rango 1e-3..2e-1) ===")
    resumen = [err_A, err_B_rango, err_C]
    for r in resumen:
        print(f"  {r['nombre']:<45} mediana={r['ratio_mediana']:.3f}  "
              f"rango=[{r['ratio_min']:.3f}, {r['ratio_max']:.3f}]  RMSE(log)={r['rmse_log']:.3f}")

    with open(os.path.join(OUT_DIR, "comparacion_modelos_rango.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["nombre", "ratio_mediana", "ratio_min", "ratio_max", "rmse_log"])
        w.writeheader()
        for r in resumen:
            w.writerow(r)

    with open(os.path.join(OUT_DIR, "coeficientes_v3_30combos.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["modelo", "intercept", "coef_log_sigma_W", "coef_log_paso_medio", "R2_in", "R2_loocv", "n"])
        w.writerow(["v3_30combos", coef_b[0], coef_b[1], coef_b[2], r2_in_b, r2_loo_b, len(filas)])
        w.writerow(["v3_rango_1e-3_a_2e-1", coef_c[0], coef_c[1], coef_c[2], r2_in_c, r2_loo_c, int(en_rango.sum())])

    with open(os.path.join(OUT_DIR, "detalle_30combos.csv"), "w", newline="", encoding="utf-8") as fh:
        campos = ["carpeta", "inl_tag", "inl_pot_real", "pulsos_pot", "batch_number", "epochs",
                  "sigma_W", "paso_medio", "beta_optimo_empirico"]
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        for f in filas:
            w.writerow({k: f[k] for k in campos})

    # --- figura: beta_pred (3 modelos) vs beta_emp, solo en el rango pedido ---
    fig, ax = plt.subplots(figsize=(6.5, 6), dpi=140)
    beta_rango = beta_all[en_rango]
    preds = {
        "v2 producción (sin reajustar)": beta_pred_A_rango,
        "reajuste 30 combos (LOOCV)": np.exp(pred_loo_b[en_rango]),
        "reajuste solo rango (LOOCV)": np.exp(pred_loo_c),
    }
    lo = min(beta_rango.min(), *[p.min() for p in preds.values()]) * 0.7
    hi = max(beta_rango.max(), *[p.max() for p in preds.values()]) * 1.3
    ax.plot([lo, hi], [lo, hi], "--", color="#999", label="identidad")
    markers = {"v2 producción (sin reajustar)": "x", "reajuste 30 combos (LOOCV)": "o",
               "reajuste solo rango (LOOCV)": "^"}
    colors = {"v2 producción (sin reajustar)": "#999", "reajuste 30 combos (LOOCV)": "#2a78d6",
              "reajuste solo rango (LOOCV)": "#d64a6a"}
    for nombre, pred in preds.items():
        ax.scatter(pred, beta_rango, s=70, marker=markers[nombre], color=colors[nombre],
                   label=nombre, zorder=3, alpha=0.85)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\beta_{pred}$")
    ax.set_ylabel(r"$\beta_{óptimo}$ empírico (grid-search real, batch_number=32)")
    ax.set_title("Comparación de modelos, evaluados SOLO en INL 1e-3..2e-1\n(15 combos, batch_number=32)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, which="both", color="#e8e7e0", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "comparacion_modelos_rango.png"))
    plt.close(fig)

    print("\nGuardados: comparacion_modelos_rango.csv, coeficientes_v3_30combos.csv, "
          "detalle_30combos.csv, comparacion_modelos_rango.png")


if __name__ == "__main__":
    main()
