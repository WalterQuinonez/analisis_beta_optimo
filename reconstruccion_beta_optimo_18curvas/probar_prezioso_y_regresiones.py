#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probar_prezioso_y_regresiones.py

Usa `resultados_18curvas.csv` (generado por reconstruir_corrientes_18curvas.py)
para:

  1) Probar la ecuacion de Prezioso/LeCun LITERAL (exponente -1 fijo, sin
     ajustar):  beta_optimo = kappa / sigma_I  -- con 3 fuentes distintas de
     sigma_I: el Monte Carlo corregido (G0_initialization + pixeles reales),
     la corriente REAL reconstruida en la 1ra epoca guardada (~ cerca de la
     inicializacion real), y la corriente REAL en la ultima epoca (converged).
     Para cada fuente: cuanto varia el kappa implicado entre las 18 curvas
     (si la ecuacion fuera valida, deberia ser ~constante), y que R^2 da la
     prediccion de beta con exponente fijo -1.

  2) Re-ajustar (exponente LIBRE, regresion log-log) beta_optimo vs cada
     sigma_I sola y combinada con paso_medio, para las 3 fuentes de sigma_I,
     y comparar contra el modelo ya existente de
     `formula_beta_optimo_propuesta/estimar_beta_optimo_propuesta_v2.py`
     (sigma_W del Monte Carlo corregido + paso_medio, R^2=0.938).

Salida (en esta misma carpeta):
  coeficientes_todos_los_modelos.csv, resultados_prezioso.csv,
  comparacion_R2_todos_los_modelos.png, prezioso_literal_vs_empirico.png,
  ecuacion_final_vs_empirico.png
"""
import os
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_IN = os.path.join(HERE, "resultados_18curvas.csv")

COLOR_POR_INL = {
    "2e-1": "#d64a6a", "1e-2": "#c9a227", "1e-3": "#3aa66b",
    "1e-4": "#4ac9c2", "1e-5": "#2a78d6", "4.82e-6": "#a34ad6",
}
MARKER_POR_P = {50: "o", 100: "s", 200: "^"}


def cargar():
    filas = []
    with open(CSV_IN, "r", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            filas.append({k: (float(v) if k not in ("carpeta", "inl_tag") else v)
                          for k, v in r.items()})
    return filas


def main():
    filas = cargar()
    n = len(filas)
    print(f"Curvas cargadas: {n}")

    beta = np.array([f["beta_optimo_empirico"] for f in filas])
    paso = np.array([f["paso_medio_pot"] for f in filas])
    pulsos = np.array([f["pulsos_pot"] for f in filas])
    inl_real = np.array([f["inl_pot_real"] for f in filas])
    y_log_beta = np.log(beta)

    fuentes_sigma = [
        ("sigma_I_mc_corregido", "MC corregido (G0_init+pixeles reales)"),
        ("sigma_I_real_ep1", "Real, epoca 1"),
        ("sigma_I_real_epLast", "Real, epoca final"),
    ]

    # -----------------------------------------------------------
    # 1) Prezioso LITERAL: beta = kappa / sigma_I  (exponente -1 fijo)
    # -----------------------------------------------------------
    print("\n=== Ecuacion de Prezioso LITERAL (beta = kappa/sigma_I, exponente -1 fijo) ===")
    filas_prezioso = []
    for campo, etiqueta in fuentes_sigma:
        sigma = np.array([f[campo] for f in filas])
        kappa_i = beta * sigma
        log_kappa = np.log(kappa_i)
        kappa_geo = np.exp(log_kappa.mean())
        pred = kappa_geo / sigma
        pred_log = np.log(pred)
        r2_forzado = 1 - np.sum((y_log_beta - pred_log) ** 2) / np.sum((y_log_beta - y_log_beta.mean()) ** 2)
        ratio = kappa_i.max() / kappa_i.min()
        print(f"[{etiqueta}] kappa: min={kappa_i.min():.3f} max={kappa_i.max():.3f} "
              f"(razon max/min={ratio:.1f}x)  kappa_geo={kappa_geo:.3f}  R2(exp=-1 fijo)={r2_forzado:.3f}")
        for f, s, k, p in zip(filas, sigma, kappa_i, pred):
            filas_prezioso.append(dict(carpeta=f["carpeta"], inl_tag=f["inl_tag"], pulsos_pot=f["pulsos_pot"],
                                        fuente=etiqueta, sigma_I=s, kappa_implicado=k,
                                        beta_pred_prezioso_literal=p, beta_optimo_empirico=f["beta_optimo_empirico"]))

    csv_prezioso = os.path.join(HERE, "resultados_prezioso.csv")
    with open(csv_prezioso, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["carpeta", "inl_tag", "pulsos_pot", "fuente", "sigma_I",
                                           "kappa_implicado", "beta_pred_prezioso_literal", "beta_optimo_empirico"])
        w.writeheader()
        w.writerows(filas_prezioso)
    print(f"Guardado: {csv_prezioso}")

    # -----------------------------------------------------------
    # 2) regresiones log-log (exponente LIBRE): solo sigma_I, y combinado
    #    con paso_medio, para las 3 fuentes de sigma_I
    # -----------------------------------------------------------
    def ajustar(X, nombre):
        coef, *_ = np.linalg.lstsq(X, y_log_beta, rcond=None)
        pred = X @ coef
        r2 = 1 - np.sum((y_log_beta - pred) ** 2) / np.sum((y_log_beta - y_log_beta.mean()) ** 2)
        print(f"[{nombre}] coef={coef}  R2={r2:.4f}")
        return coef, pred, r2

    print("\n=== Regresiones log-log, exponente LIBRE ===")
    ones = np.ones(n)
    modelos = {}
    for campo, etiqueta in fuentes_sigma:
        sigma = np.array([f[campo] for f in filas])
        X_solo = np.column_stack([ones, np.log(sigma)])
        coef, pred, r2 = ajustar(X_solo, f"solo sigma_I ({etiqueta})")
        modelos[f"solo_{campo}"] = dict(coef=coef, pred=pred, r2=r2, etiqueta=f"solo sigma_I\n({etiqueta})")

        X_comb = np.column_stack([ones, np.log(sigma), np.log(paso)])
        coef, pred, r2 = ajustar(X_comb, f"combinado sigma_I ({etiqueta}) + paso_medio")
        modelos[f"combinado_{campo}"] = dict(coef=coef, pred=pred, r2=r2,
                                              etiqueta=f"combinado\n({etiqueta}+paso)")

    # -----------------------------------------------------------
    # elegir el mejor modelo combinado (mayor R2) como ecuacion final
    # -----------------------------------------------------------
    combinados = {k: v for k, v in modelos.items() if k.startswith("combinado_")}
    mejor_key = max(combinados, key=lambda k: combinados[k]["r2"])
    mejor = combinados[mejor_key]
    campo_mejor = mejor_key[len("combinado_"):]
    sigma_mejor = np.array([f[campo_mejor] for f in filas])
    beta_pred_mejor = np.exp(mejor["pred"])
    print(f"\nMejor modelo combinado: {mejor_key}  R2={mejor['r2']:.4f}  coef={mejor['coef']}")

    coef_path = os.path.join(HERE, "coeficientes_todos_los_modelos.csv")
    with open(coef_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["modelo", "intercept", "coef_log_sigma_I", "coef_log_paso_medio", "R2"])
        for k, v in modelos.items():
            coef = v["coef"]
            if len(coef) == 2:
                w.writerow([k, coef[0], coef[1], "", v["r2"]])
            else:
                w.writerow([k, coef[0], coef[1], coef[2], v["r2"]])
    print(f"Guardado: {coef_path}")

    # -----------------------------------------------------------
    # figuras
    # -----------------------------------------------------------
    # (a) Prezioso literal (mejor fuente: MC corregido, la mas cercana a la
    #     teoria) vs empirico
    sigma_mc = np.array([f["sigma_I_mc_corregido"] for f in filas])
    kappa_geo_mc = np.exp(np.mean(np.log(beta * sigma_mc)))
    beta_pred_prezioso = kappa_geo_mc / sigma_mc
    fig, ax = plt.subplots(figsize=(6.5, 6), dpi=140)
    lo = min(beta.min(), beta_pred_prezioso.min()) * 0.5
    hi = max(beta.max(), beta_pred_prezioso.max()) * 1.5
    ax.plot([lo, hi], [lo, hi], "--", color="#999", label="identidad")
    for f, bp in zip(filas, beta_pred_prezioso):
        c = COLOR_POR_INL[f["inl_tag"]]; m = MARKER_POR_P[f["pulsos_pot"]]
        ax.scatter(bp, f["beta_optimo_empirico"], s=80, color=c, marker=m, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\beta_{pred}$ (Prezioso literal: $\kappa/\sigma_I$, exponente $-1$ fijo)")
    ax.set_ylabel(r"$\beta_{óptimo}$ empírico")
    r2_lit = 1 - np.sum((y_log_beta - np.log(beta_pred_prezioso)) ** 2) / np.sum((y_log_beta - y_log_beta.mean()) ** 2)
    ax.set_title(f"Ecuación de Prezioso literal vs. empírico (18 curvas, R²={r2_lit:.3f})")
    for tag, c in COLOR_POR_INL.items():
        ax.scatter([], [], color=c, label=f"INL {tag}")
    for p, m in MARKER_POR_P.items():
        ax.scatter([], [], color="#555", marker=m, label=f"p={p}")
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="best")
    ax.grid(True, which="both", color="#e8e7e0", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "prezioso_literal_vs_empirico.png"))
    plt.close(fig)

    # (b) comparacion R2 de todos los modelos
    orden = ["solo_sigma_I_mc_corregido", "combinado_sigma_I_mc_corregido",
             "solo_sigma_I_real_ep1", "combinado_sigma_I_real_ep1",
             "solo_sigma_I_real_epLast", "combinado_sigma_I_real_epLast"]
    etiquetas = [modelos[k]["etiqueta"] for k in orden]
    r2s = [modelos[k]["r2"] for k in orden]
    colores_barras = ["#2a78d6", "#1c4f8f", "#eb6834", "#a8431a", "#3aa66b", "#256b45"]
    fig, ax = plt.subplots(figsize=(11, 5), dpi=140)
    ax.axhline(r2_lit, color="#999", linestyle="--", linewidth=1,
               label=f"Prezioso literal (exp.=-1 fijo): R²={r2_lit:.3f}")
    ax.bar(etiquetas, r2s, color=colores_barras)
    for i, v in enumerate(r2s):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylim(min(0, r2_lit) - 0.05, 1.05)
    ax.set_ylabel("R² (ajuste log-log a los 18 β óptimo reales)")
    ax.set_title("Comparación de todos los modelos probados")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    plt.setp(ax.get_xticklabels(), fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "comparacion_R2_todos_los_modelos.png"))
    plt.close(fig)

    # (c) ecuacion final (mejor combinado) vs empirico
    fig, ax = plt.subplots(figsize=(6.5, 6), dpi=140)
    lo = min(beta.min(), beta_pred_mejor.min()) * 0.7
    hi = max(beta.max(), beta_pred_mejor.max()) * 1.3
    ax.plot([lo, hi], [lo, hi], "--", color="#999", label="identidad")
    for f, bp in zip(filas, beta_pred_mejor):
        c = COLOR_POR_INL[f["inl_tag"]]; m = MARKER_POR_P[f["pulsos_pot"]]
        ax.scatter(bp, f["beta_optimo_empirico"], s=80, color=c, marker=m, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\beta_{pred}$ (ecuación final combinada)")
    ax.set_ylabel(r"$\beta_{óptimo}$ empírico")
    ax.set_title(f"Ecuación final vs. empírico (18 curvas, R²={mejor['r2']:.3f})")
    for tag, c in COLOR_POR_INL.items():
        ax.scatter([], [], color=c, label=f"INL {tag}")
    for p, m in MARKER_POR_P.items():
        ax.scatter([], [], color="#555", marker=m, label=f"p={p}")
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="best")
    ax.grid(True, which="both", color="#e8e7e0", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "ecuacion_final_vs_empirico.png"))
    plt.close(fig)

    print("\nErrores de la ecuacion final por curva:")
    for f, bp in zip(filas, beta_pred_mejor):
        err = 100 * (bp / f["beta_optimo_empirico"] - 1)
        print(f"  {f['carpeta']:<40} emp={f['beta_optimo_empirico']:>5.0f}  pred={bp:>7.1f}  error={err:+6.1f}%")

    print("\nGuardadas figuras: prezioso_literal_vs_empirico.png, "
          "comparacion_R2_todos_los_modelos.png, ecuacion_final_vs_empirico.png")


if __name__ == "__main__":
    main()
