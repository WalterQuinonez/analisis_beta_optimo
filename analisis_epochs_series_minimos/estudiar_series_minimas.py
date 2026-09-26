"""
Fase 2: cuantas series (y cuantos folds) por entrenamiento hacen falta para
identificar un beta "suficientemente bueno", usando UNICAMENTE los datos ya
generados por el grid search de fuerza bruta (data_beta_grid_search_SP/, 20
series x 5 folds x 100 epocas por beta). No corre ninguna simulacion nueva:
se subsamplean (sin reemplazo) las series/folds ya guardados, muchas veces,
para estimar que tan confiable es la eleccion de beta con menos repeticiones.

Metodologia (por combo INL x pulsos):
  1. Ground truth: beta_optimo usando las 20 series x 5 folds x epoca=100.
  2. Para cada n_series in {1,2,3,5,8,12,16,20} (con folds=5 fijo, que es lo
     que pide el usuario: "cantidad de series por entrenamiento"):
       - Repetir N_REPETICIONES veces: elegir n_series series al azar (sin
         reemplazo) de las 20 disponibles, promediar acc(beta, epoca=E) sobre
         esas series x 5 folds, elegir beta_hat = argmax_beta.
       - Evaluar el "regret" de esa eleccion con el mismo criterio de la Fase 1:
         regret = acc_optimo_final - acc_mean_TODAS_LAS_SERIES[beta_hat, epoca=100]
         (es decir: si me quedo con beta_hat, pero LUEGO esa red se entrena
         con normalidad hasta convergencia, cuanto pierdo respecto del optimo).
       - Se usa E = EPOCH_PARA_DECISION (por defecto, el epochs_min recomendado
         en la Fase 1) porque asi se evalua el protocolo reducido COMPLETO
         (menos series Y menos epocas), que es el que de verdad se piensa usar.
     Reportar: fraccion de repeticiones con regret < TOL_ACC (confiabilidad),
     y percentiles del regret.
  3. Igual analisis pero variando n_folds in {1,2,3,5} con series=20 fijo, para
     chequear sensibilidad a k_folds (secundario).

Salidas:
    series_minimas_vs_precision.csv
    series_minimas_vs_precision.png
    folds_minimos_vs_precision.csv
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from comun_carga_datos import listar_combos, cargar_combo, acc_mean_por_beta_epoca

DIR_SALIDA = os.path.dirname(os.path.abspath(__file__))
TOL_ACC = 0.01
N_REPETICIONES = 300
SERIES_A_PROBAR = [1, 2, 3, 5, 8, 12, 16, 20]
FOLDS_A_PROBAR = [1, 2, 3, 5]
EPOCH_PARA_DECISION = 8  # = epochs_min recomendado en la Fase 1 (ver reporte)
SEED = 12345


def evaluar_subsample(acc, idx_final, epoca_idx, n_series, n_folds, rng):
    """
    acc: (n_betas, series=20, folds=5, epochs)
    Devuelve el regret (en accuracy final, epoca=100) de elegir beta con un
    subsample aleatorio de n_series series x n_folds folds, evaluado en la
    epoca `epoca_idx` (0-indexed).
    """
    n_betas, n_series_disp, n_folds_disp, n_epochs = acc.shape
    series_idx = rng.choice(n_series_disp, size=n_series, replace=False)
    folds_idx = rng.choice(n_folds_disp, size=n_folds, replace=False)
    sub = acc[np.ix_(np.arange(n_betas), series_idx, folds_idx, [epoca_idx])]
    acc_mean_sub = sub.mean(axis=(1, 2, 3))  # (n_betas,)
    beta_hat_idx = int(np.argmax(acc_mean_sub))

    acc_final_verdadero = acc[:, :, :, -1].mean(axis=(1, 2))  # (n_betas,) con TODOS los datos
    acc_opt_final = acc_final_verdadero[idx_final]
    regret = acc_opt_final - acc_final_verdadero[beta_hat_idx]
    return regret


def estudiar_combo_series(combo, rng):
    betas, acc = cargar_combo(combo["path"])
    acc_mean = acc_mean_por_beta_epoca(acc)
    idx_final = int(np.argmax(acc_mean[:, -1]))
    epoca_idx = min(EPOCH_PARA_DECISION, acc.shape[3]) - 1

    filas = []
    for n_series in SERIES_A_PROBAR:
        regrets = [
            evaluar_subsample(acc, idx_final, epoca_idx, n_series, acc.shape[2], rng)
            for _ in range(N_REPETICIONES)
        ]
        regrets = np.array(regrets)
        filas.append({
            "combo": combo["nombre"], "inl_tag": combo["inl_tag"], "pulsos": combo["pulsos"],
            "n_series": n_series, "n_folds": acc.shape[2],
            "prob_regret_ok": float(np.mean(regrets < TOL_ACC)),
            "regret_mediana": float(np.median(regrets)),
            "regret_p90": float(np.quantile(regrets, 0.90)),
        })
    return filas


def estudiar_combo_folds(combo, rng):
    betas, acc = cargar_combo(combo["path"])
    acc_mean = acc_mean_por_beta_epoca(acc)
    idx_final = int(np.argmax(acc_mean[:, -1]))
    epoca_idx = min(EPOCH_PARA_DECISION, acc.shape[3]) - 1

    filas = []
    for n_folds in FOLDS_A_PROBAR:
        if n_folds > acc.shape[2]:
            continue
        regrets = [
            evaluar_subsample(acc, idx_final, epoca_idx, acc.shape[1], n_folds, rng)
            for _ in range(N_REPETICIONES)
        ]
        regrets = np.array(regrets)
        filas.append({
            "combo": combo["nombre"], "inl_tag": combo["inl_tag"], "pulsos": combo["pulsos"],
            "n_series": acc.shape[1], "n_folds": n_folds,
            "prob_regret_ok": float(np.mean(regrets < TOL_ACC)),
            "regret_mediana": float(np.median(regrets)),
            "regret_p90": float(np.quantile(regrets, 0.90)),
        })
    return filas


def main():
    rng = np.random.default_rng(SEED)
    combos = listar_combos()

    filas_series = []
    filas_folds = []
    for c in combos:
        filas_series.extend(estudiar_combo_series(c, rng))
        filas_folds.extend(estudiar_combo_folds(c, rng))

    df_series = pd.DataFrame(filas_series)
    df_folds = pd.DataFrame(filas_folds)

    df_series.to_csv(os.path.join(DIR_SALIDA, "series_minimas_vs_precision.csv"), index=False)
    df_folds.to_csv(os.path.join(DIR_SALIDA, "folds_minimos_vs_precision.csv"), index=False)

    print(f"epoca usada para la decision (EPOCH_PARA_DECISION={EPOCH_PARA_DECISION}), tol={TOL_ACC*100:.1f}pp\n")
    print("--- Confiabilidad (peor combo) vs n_series, folds=5 ---")
    agg = df_series.groupby("n_series").agg(
        prob_ok_media=("prob_regret_ok", "mean"),
        prob_ok_peor=("prob_regret_ok", "min"),
        regret_p90_peor=("regret_p90", "max"),
    ).reset_index()
    print(agg.to_string(index=False))

    print("\n--- Confiabilidad (peor combo) vs n_folds, series=20 ---")
    agg_f = df_folds.groupby("n_folds").agg(
        prob_ok_media=("prob_regret_ok", "mean"),
        prob_ok_peor=("prob_regret_ok", "min"),
        regret_p90_peor=("regret_p90", "max"),
    ).reset_index()
    print(agg_f.to_string(index=False))

    # recomendacion: minimo n_series tal que el PEOR combo tenga prob_ok >= 0.90
    candidatos = agg[agg["prob_ok_peor"] >= 0.90]
    series_min_recomendado = int(candidatos["n_series"].min()) if len(candidatos) else max(SERIES_A_PROBAR)
    print(f"\n>>> series_min recomendado (peor combo con prob_regret_ok >= 90%): {series_min_recomendado}")

    # --- grafico ---
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for combo_nombre, sub in df_series.groupby("combo"):
        sub = sub.sort_values("n_series")
        ax.plot(sub["n_series"], sub["prob_regret_ok"] * 100, color="steelblue", alpha=0.25, linewidth=1)
    ax.plot(agg["n_series"], agg["prob_ok_media"] * 100, color="steelblue", linewidth=2.5, marker="o", label="media sobre 30 combos")
    ax.plot(agg["n_series"], agg["prob_ok_peor"] * 100, color="crimson", linewidth=2.5, marker="s", label="peor combo")
    ax.axhline(90, color="gray", linestyle="--", label="umbral 90%")
    ax.axvline(series_min_recomendado, color="crimson", linestyle=":", label=f"series_min = {series_min_recomendado}")
    ax.set_xlabel("n_series usadas (de 20 disponibles), k_folds=5, epoca de decision=%d" % EPOCH_PARA_DECISION)
    ax.set_ylabel(f"P(regret < {TOL_ACC*100:.0f}pp) [%]")
    ax.set_title("Confiabilidad de elegir beta_optimo segun cantidad de series")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(DIR_SALIDA, "series_minimas_vs_precision.png"), dpi=150)
    print(f"Guardado {os.path.join(DIR_SALIDA, 'series_minimas_vs_precision.png')}")


if __name__ == "__main__":
    main()
