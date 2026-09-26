#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
comparar_INL.py

Extiende comparar_p.py: ahora compara tambien entre INL, no solo entre p.
Responde el "proximo paso" del reporte original (reporte_distribucion_softmax_INL_2e-1.md,
Seccion 6): usando la data ya generada para INL=1e-2 (curva P/D casi
lineal, analisis_distribucion_softmax_INL_1e-2_p_{50,100,200}/) contra la
ya conocida de INL=2e-1 (curva muy no lineal), ver si:
  (a) el "acantilado de estabilidad" post-beta-optimo es propio de INL
      grande (muy no lineal), y
  (b) el rango de logit_std en el beta optimo se mantiene parecido entre
      INL, o depende tambien del INL (no solo de p).

No recalcula nada de cero: solo lee los resumen_por_beta.csv ya generados
por analizar_distribucion_softmax.py en las 6 carpetas
(INL x p = {2e-1,1e-2} x {50,100,200}).

Salida (en esta misma carpeta):
    comparacion_en_beta_optimo_INL_y_p.csv
    accuracy_vs_beta_comparado_INL_y_p.png
    logit_std_vs_beta_comparado_INL_y_p.png
"""
import os
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

# (INL, p, beta_optimo, color, estilo de linea)
CASOS = [
    dict(inl="2e-1", p=50, beta_optimo=100.0, color="#2a78d6", ls="-"),
    dict(inl="2e-1", p=100, beta_optimo=100.0, color="#eb6834", ls="-"),
    dict(inl="2e-1", p=200, beta_optimo=200.0, color="#3aa66b", ls="-"),
    dict(inl="1e-2", p=50, beta_optimo=250.0, color="#2a78d6", ls="--"),
    dict(inl="1e-2", p=100, beta_optimo=350.0, color="#eb6834", ls="--"),
    dict(inl="1e-2", p=200, beta_optimo=750.0, color="#3aa66b", ls="--"),
]

for caso in CASOS:
    carpeta = os.path.join(REPO_ROOT, f"analisis_distribucion_softmax_INL_{caso['inl']}_p_{caso['p']}")
    with open(os.path.join(carpeta, "resumen_por_beta.csv"), encoding="utf-8") as fh:
        filas = list(csv.DictReader(fh))
    filas = sorted(filas, key=lambda r: float(r["beta"]))
    caso["filas"] = filas
    caso["betas"] = np.array([float(r["beta"]) for r in filas])

# ---------------------------------------------------------------------------
# tabla comparativa en el beta optimo de cada curva
# ---------------------------------------------------------------------------
campos_tabla = ["inl", "p", "beta_optimo", "acc_reconstruida", "conf_media", "entropia_media",
                "ece", "brecha_confianza", "logit_std", "frac_saturada"]
filas_tabla = []
for caso in CASOS:
    idx = int(np.argmin(np.abs(caso["betas"] - caso["beta_optimo"])))
    r = caso["filas"][idx]
    filas_tabla.append(dict(
        inl=caso["inl"], p=caso["p"], beta_optimo=caso["beta_optimo"],
        acc_reconstruida=float(r["acc_reconstruida"]),
        conf_media=float(r["conf_media"]), entropia_media=float(r["entropia_media"]),
        ece=float(r["ece"]), brecha_confianza=float(r["brecha_confianza"]),
        logit_std=float(r["logit_std"]), frac_saturada=float(r["frac_saturada"]),
    ))

csv_path = os.path.join(HERE, "comparacion_en_beta_optimo_INL_y_p.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=campos_tabla)
    w.writeheader()
    w.writerows(filas_tabla)
print(f"Guardado: {csv_path}")
print("\nComparacion en el beta optimo de cada curva (INL x p):")
for r in filas_tabla:
    print(f"  INL={r['inl']:>5}  p={r['p']:>3}  beta*={r['beta_optimo']:>4.0f}  "
          f"acc={r['acc_reconstruida']:.4f}  ece={r['ece']:.4f}  "
          f"logit_std={r['logit_std']:.3f}  frac_sat={r['frac_saturada']:.3f}")

# ---------------------------------------------------------------------------
# figura 1: accuracy vs beta, las 6 curvas (2 INL x 3 p)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 6), dpi=140)
for caso in CASOS:
    acc = [float(r["acc_reconstruida"]) for r in caso["filas"]]
    ax.plot(caso["betas"], acc, color=caso["color"], linestyle=caso["ls"], marker="o",
            markersize=3.5, linewidth=1.8,
            label=f"INL={caso['inl']}, p={caso['p']}")
    ax.axvline(caso["beta_optimo"], color=caso["color"], linestyle=":", linewidth=1.0, alpha=0.5)
ax.set_xlabel("beta")
ax.set_ylabel("accuracy reconstruida (serie 0)")
ax.set_title("Accuracy vs beta — INL=2e-1 (lineas solidas) vs INL=1e-2 (lineas punteadas)")
ax.legend(frameon=False, fontsize=8.5, ncol=2)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "accuracy_vs_beta_comparado_INL_y_p.png"))
plt.close(fig)

# ---------------------------------------------------------------------------
# figura 2: logit_std vs beta, las 6 curvas (escala log en y)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 6), dpi=140)
for caso in CASOS:
    logit_std = [float(r["logit_std"]) for r in caso["filas"]]
    ax.plot(caso["betas"], logit_std, color=caso["color"], linestyle=caso["ls"], marker="o",
            markersize=3.5, linewidth=1.8, label=f"INL={caso['inl']}, p={caso['p']}")
    idx = int(np.argmin(np.abs(caso["betas"] - caso["beta_optimo"])))
    ax.plot(caso["beta_optimo"], logit_std[idx], marker="*", markersize=16,
            color=caso["color"], markeredgecolor="black", markeredgewidth=0.6, zorder=5)
ax.axhspan(1.7, 2.8, color="#2a78d6", alpha=0.08, zorder=0, label="banda INL=2e-1 (1.8-2.7)")
ax.axhspan(3.4, 6.5, color="#eb6834", alpha=0.08, zorder=0, label="banda INL=1e-2 (3.5-6.4)")
ax.set_yscale("log")
ax.set_xlabel("beta")
ax.set_ylabel("desvio estandar de los logits crudos (beta*I)")
ax.set_title("Dispersion de logits vs beta — INL=2e-1 (solidas) vs INL=1e-2 (punteadas)\n"
              "(estrella = beta optimo de cada curva)")
ax.legend(frameon=False, fontsize=8, ncol=2)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "logit_std_vs_beta_comparado_INL_y_p.png"))
plt.close(fig)

print("\nGuardadas figuras: accuracy_vs_beta_comparado_INL_y_p.png, logit_std_vs_beta_comparado_INL_y_p.png")
