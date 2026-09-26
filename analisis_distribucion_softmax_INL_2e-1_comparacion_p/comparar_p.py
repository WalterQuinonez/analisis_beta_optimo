#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
comparar_p.py

Compara, para INL=2e-1, los tres analisis "hermanos" ya hechos
(analisis_distribucion_softmax_INL_2e-1_p_{50,100,200}/resumen_por_beta.csv)
para ver que tienen en comun (y que no) al cambiar la cantidad de pulsos de
la curva P/D.

No recalcula nada de cero: solo lee los resumen_por_beta.csv ya generados
por analizar_distribucion_softmax.py en cada una de esas tres carpetas.

Salida (en esta misma carpeta):
    comparacion_en_beta_optimo.csv
    accuracy_vs_beta_comparado.png
    logit_std_vs_beta_comparado.png
"""
import os
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

CASOS = [
    dict(p=50, beta_optimo=100.0, color="#2a78d6",
         carpeta=os.path.join(REPO_ROOT, "analisis_distribucion_softmax_INL_2e-1_p_50")),
    dict(p=100, beta_optimo=100.0, color="#eb6834",
         carpeta=os.path.join(REPO_ROOT, "analisis_distribucion_softmax_INL_2e-1_p_100")),
    dict(p=200, beta_optimo=200.0, color="#3aa66b",
         carpeta=os.path.join(REPO_ROOT, "analisis_distribucion_softmax_INL_2e-1_p_200")),
]

for caso in CASOS:
    with open(os.path.join(caso["carpeta"], "resumen_por_beta.csv"), encoding="utf-8") as fh:
        filas = list(csv.DictReader(fh))
    filas = sorted(filas, key=lambda r: float(r["beta"]))
    caso["filas"] = filas
    caso["betas"] = np.array([float(r["beta"]) for r in filas])

# ---------------------------------------------------------------------------
# tabla comparativa en el beta optimo de cada curva
# ---------------------------------------------------------------------------
campos_tabla = ["p", "beta_optimo", "acc_reconstruida", "conf_media", "entropia_media",
                "ece", "brecha_confianza", "logit_std", "frac_saturada"]
filas_tabla = []
for caso in CASOS:
    idx = int(np.argmin(np.abs(caso["betas"] - caso["beta_optimo"])))
    r = caso["filas"][idx]
    filas_tabla.append(dict(
        p=caso["p"], beta_optimo=caso["beta_optimo"],
        acc_reconstruida=float(r["acc_reconstruida"]),
        conf_media=float(r["conf_media"]), entropia_media=float(r["entropia_media"]),
        ece=float(r["ece"]), brecha_confianza=float(r["brecha_confianza"]),
        logit_std=float(r["logit_std"]), frac_saturada=float(r["frac_saturada"]),
    ))

csv_path = os.path.join(HERE, "comparacion_en_beta_optimo.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=campos_tabla)
    w.writeheader()
    w.writerows(filas_tabla)
print(f"Guardado: {csv_path}")
print("\nComparacion en el beta optimo de cada curva (INL=2e-1):")
for r in filas_tabla:
    print(f"  p={r['p']:>3}  beta*={r['beta_optimo']:>4.0f}  acc={r['acc_reconstruida']:.4f}  "
          f"conf_media={r['conf_media']:.4f}  entropia={r['entropia_media']:.4f}  "
          f"ece={r['ece']:.4f}  brecha_conf={r['brecha_confianza']:.4f}  "
          f"logit_std={r['logit_std']:.3f}  frac_sat={r['frac_saturada']:.3f}")

# ---------------------------------------------------------------------------
# figura 1: accuracy vs beta, las 3 curvas superpuestas
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=140)
for caso in CASOS:
    acc = [float(r["acc_reconstruida"]) for r in caso["filas"]]
    ax.plot(caso["betas"], acc, color=caso["color"], marker="o", markersize=4,
            linewidth=1.8, label=f"p={caso['p']}")
    ax.axvline(caso["beta_optimo"], color=caso["color"], linestyle=":", linewidth=1.2, alpha=0.7)
ax.set_xlabel("beta")
ax.set_ylabel("accuracy reconstruida (serie 0)")
ax.set_title("INL=2e-1 — accuracy vs beta, comparado entre p (lineas punteadas = beta optimo)")
ax.legend(frameon=False, fontsize=9, title="pulsos")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "accuracy_vs_beta_comparado.png"))
plt.close(fig)

# ---------------------------------------------------------------------------
# figura 2: logit_std vs beta, las 3 curvas superpuestas (escala log en y)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=140)
for caso in CASOS:
    logit_std = [float(r["logit_std"]) for r in caso["filas"]]
    ax.plot(caso["betas"], logit_std, color=caso["color"], marker="o", markersize=4,
            linewidth=1.8, label=f"p={caso['p']}")
    idx = int(np.argmin(np.abs(caso["betas"] - caso["beta_optimo"])))
    ax.plot(caso["beta_optimo"], logit_std[idx], marker="*", markersize=16,
            color=caso["color"], markeredgecolor="black", markeredgewidth=0.6, zorder=5)
ax.axhspan(1.7, 2.8, color="#7a7a7a", alpha=0.12, zorder=0,
           label="banda de logit_std en los 3 beta optimos (1.8-2.7)")
ax.set_yscale("log")
ax.set_xlabel("beta")
ax.set_ylabel("desvio estandar de los logits crudos (beta*I)")
ax.set_title("INL=2e-1 — dispersion de logits vs beta, comparado entre p\n"
              "(estrella = beta optimo de cada curva)")
ax.legend(frameon=False, fontsize=8.5)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "logit_std_vs_beta_comparado.png"))
plt.close(fig)

print("\nGuardadas figuras: accuracy_vs_beta_comparado.png, logit_std_vs_beta_comparado.png")
