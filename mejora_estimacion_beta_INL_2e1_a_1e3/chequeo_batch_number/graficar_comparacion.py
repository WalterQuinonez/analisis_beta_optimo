#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Grafica accuracy vs beta a batch_number=32 (ground truth completo, 20
series x 5 folds x 100 epocas) contra el chequeo barato a batch_number=1
(1 serie x 5 folds x 100 epocas, 4 puntos de beta), para INL=2e-1, p=50
(a_pot=5.85), el caso que disparo este re-estudio."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("curva_bn32.json") as fh:
    d = json.load(fh)
betas32 = np.array(d["betas"])
medias32 = np.array(d["medias"])

betas1 = np.array([50, 100, 300, 1389])
medias1 = np.array([0.6313, 0.6356, 0.6505, 0.7004])

fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=140)
ax.plot(betas32, medias32, "-o", color="#2a78d6", markersize=4, linewidth=1.5,
        label="batch_number=32 (ground truth: 20 series x 5 folds)")
idx32 = int(np.argmax(medias32))
ax.scatter([betas32[idx32]], [medias32[idx32]], color="#2a78d6", s=140,
           marker="*", zorder=5, label=f"óptimo bn=32: β={betas32[idx32]:.0f}")

ax.plot(betas1, medias1, "-o", color="#d64a6a", markersize=7, linewidth=1.5,
        label="batch_number=1 (chequeo: 1 serie x 5 folds, 4 puntos)")
ax.annotate("sigue subiendo:\nno se llegó al pico",
            xy=(1389, 0.7004), xytext=(600, 0.74),
            arrowprops=dict(arrowstyle="->", color="#d64a6a"), color="#d64a6a", fontsize=9)

ax.axvline(86.8, color="#888", linestyle="--", linewidth=1)
ax.text(86.8 * 1.05, 0.615, r"$\beta_{pred}$ fórmula = 86.8", color="#888", fontsize=8, rotation=90, va="bottom")

ax.set_xscale("log")
ax.set_xlabel(r"$\beta$")
ax.set_ylabel("accuracy final (media sobre folds)")
ax.set_title("INL=2e-1, p=50 (a_pot=5.85): el óptimo de β depende fuerte de batch_number")
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.grid(True, which="both", color="#e8e7e0", linewidth=0.6)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("comparacion_bn1_vs_bn32.png")
print("Guardado: comparacion_bn1_vs_bn32.png")
