#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reproducir_analisis_beta_optimo.py

Reproduccion FIEL del metodo de `analisis_beta_optimo.py` (mismos
parametros, misma construccion de wij/histograma, mismo Monte Carlo de la
corriente), para poder:
  (a) describir con precision el metodo que usa ese script para estimar
      beta_optimo, y
  (b) correlacionar la distribucion Monte Carlo de la corriente "antes" (I)
      y "despues" (beta*I) de escalarla por los beta_optimo REALES
      (encontrados por grid-search + entrenamiento completo, no por este
      script), para las 3 curvas P/D del grupo INL nominal 1e-2
      (`data_beta_grid_search_SP/resumen_por_INL/resumen_optimos.csv`).

Diferencias respecto de `analisis_beta_optimo.py` (NO se modifica ese
archivo; esto es una copia adaptada para poder correrla sin bloquear en
ventanas de matplotlib y para poder extraer numeros):
  - backend "Agg" + savefig en vez de figuras interactivas sin `plt.show()`.
  - el Monte Carlo de 1.000.000 de corrientes se vectoriza por chunks
    (`np.random.randint` + indexado) en vez de 1.000.000 de llamadas a
    `np.random.choice` en un for-loop de Python; el METODO estadistico es
    el mismo (muestreo con reposicion de 784 terminos del mismo pool de
    productos V*W), solo cambia la implementacion para que sea rapida.
  - ademas de imprimir/graficar, guarda un CSV con las estadisticas de
    "antes" y "despues", y un segundo CSV de diagnostico que descompone
    mean(I) = 784 * E[V] * E[W] para explicar por que esa distribucion no
    esta centrada en cero (ver reporte).

No se corre ningun entrenamiento de red neuronal aca: todo esto es
estadistica sobre el mismo Monte Carlo que ya proponia el archivo original.
"""
import os
import sys
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
from MemCrossbarClass_beta_por_capa import generar_curvas_pot_dep  # noqa: E402

# ---- parametros IDENTICOS a analisis_beta_optimo.py ----
a_pot_list = [48.83, 98.55, 197.99]
a_dep_list = [48.83, 98.55, 197.99]
# beta_optimo empiricos (grid-search real, 100 epocas, k=5 folds, ver
# data_beta_grid_search_SP/resumen_por_INL/resumen_optimos.csv) para
# p=50,100,200 dentro del grupo INL nominal 1e-2. En analisis_beta_optimo.py
# estos mismos 3 valores estan hardcodeados como `betas`.
betas = [250, 350, 750]
pulsos_pot_list = [50, 100, 200]
pulsos_dep_list = [50, 100, 200]
concavidad_pot = 'pos'
concavidad_dep = 'neg'
Rhigh = 10000
Rlow = 1000
Gmin = 1 / Rhigh
Gmax = 1 / Rlow

amplitud_imagen = 1
X_train = np.load(os.path.join(REPO_ROOT, "X_train_mnist.npy")) * amplitud_imagen
voltajes = np.unique(X_train)  # tal cual analisis_beta_optimo.py: valores UNICOS, sin pesar por frecuencia real

np.random.seed(35)
MUESTRAS = 1_000_000
CHUNK = 50_000
N_SUM = 28 * 28  # 784 terminos por corriente (una entrada por pixel)

# ---- diagnostico: por que E[V] y E[W] no son cero aca ----
diag_rows = [{
    "cantidad": "E[V] (voltajes unicos, equipesados, como en el script)",
    "valor": voltajes.mean(),
}, {
    "cantidad": "E[V] (X_train real, todos los pixeles, pesado por frecuencia)",
    "valor": X_train.mean(),
}, {
    "cantidad": "V_rms (voltajes unicos, equipesados)",
    "valor": float(np.sqrt(np.mean(voltajes ** 2))),
}, {
    "cantidad": "V_rms (X_train real)",
    "valor": float(np.sqrt(np.mean(X_train ** 2))),
}, {
    "cantidad": "fraccion de pixeles == 0 en X_train real",
    "valor": float(np.mean(X_train == 0)),
}]

fig, axes = plt.subplots(2, 3, figsize=(15.5, 8), dpi=130, sharex=False)
filas_csv = []

for r in range(3):
    a_pot, a_dep = a_pot_list[r], a_dep_list[r]
    pulsos_pot, pulsos_dep = pulsos_pot_list[r], pulsos_dep_list[r]
    beta = betas[r]

    pot, dep, ratio, inl_pot, inl_dep = generar_curvas_pot_dep(
        pulsos_pot, pulsos_dep, a_pot, a_dep, Gmin, Gmax, concavidad_dep, concavidad_pot)

    # wij[i,j] = pot[i] - dep[j], TODAS las combinaciones (igual que el
    # for-loop `wij[i] = pot[i] - dep` de analisis_beta_optimo.py)
    wij = pot[:, None] - dep[None, :]
    sigma_W = wij.flatten().std()
    mean_W = wij.mean()

    # histograma = todos los posibles V_j * W_ij (outer product), igual que
    # `histograma[i, j] = voltajes[i] * wij.flatten()[j]`
    x = np.multiply.outer(voltajes, wij.flatten()).ravel()

    # Monte Carlo de I = sum_{j=1}^{784} V_j*W_ij, muestreando 784 terminos
    # CON REPOSICION del pool x -- equivalente a
    # `np.random.choice(x, size=784, replace=True)` repetido MUESTRAS veces,
    # vectorizado por chunks para que sea rapido.
    corrientes = np.empty(MUESTRAS)
    for start in range(0, MUESTRAS, CHUNK):
        end = min(start + CHUNK, MUESTRAS)
        idx = np.random.randint(0, x.size, size=(end - start, N_SUM))
        corrientes[start:end] = x[idx].sum(axis=1)

    mean_I, std_I = corrientes.mean(), corrientes.std()
    rms_I = np.sqrt(np.mean(corrientes ** 2))
    # los 3 candidatos de beta_optimo tal cual los imprime analisis_beta_optimo.py
    beta_1_mean = abs(1 / mean_I)
    beta_1_std_plus_mean = abs(1 / std_I + mean_I)  # posible bug de parentesis, ver reporte
    beta_1_rms = 1 / rms_I

    escaladas = beta * corrientes
    mean_S, std_S = escaladas.mean(), escaladas.std()
    rms_S = np.sqrt(np.mean(escaladas ** 2))
    frac_gt1 = float(np.mean(np.abs(escaladas) > 1))
    frac_gt3 = float(np.mean(np.abs(escaladas) > 3))
    frac_gt10 = float(np.mean(np.abs(escaladas) > 10))

    print(f"\n=== p={pulsos_pot} (a={a_pot}), INL_pot={inl_pot:.4f}, beta_optimo_empirico={beta} ===")
    print(f"  mean(pot)={pot.mean():.4e}  mean(dep)={dep.mean():.4e}  mean(W)={mean_W:.4e}  sigma_W={sigma_W:.4e}")
    print(f"  ANTES  (I):     mean={mean_I:.4e}  std={std_I:.4e}  rms={rms_I:.4e}  mean/std={mean_I/std_I:.2f}")
    print(f"  candidatos beta_optimo del script: 1/mean={beta_1_mean:.2f}  "
          f"|1/std+mean|={beta_1_std_plus_mean:.2f}  1/rms={beta_1_rms:.2f}")
    print(f"  DESPUES(beta*I): mean={mean_S:.4f}  std={std_S:.4f}  rms={rms_S:.4f}")
    print(f"  fraccion |beta*I|>1: {frac_gt1*100:.1f}%   >3: {frac_gt3*100:.1f}%   >10: {frac_gt10*100:.1f}%")

    filas_csv.append(dict(
        p=pulsos_pot, a=a_pot, beta_optimo_empirico=beta, inl_pot=inl_pot,
        mean_pot=pot.mean(), mean_dep=dep.mean(), mean_W=mean_W, sigma_W=sigma_W,
        mean_I=mean_I, std_I=std_I, rms_I=rms_I, mean_over_std_I=mean_I / std_I,
        beta_1_over_mean=beta_1_mean, beta_1_over_std_plus_mean=beta_1_std_plus_mean, beta_1_over_rms=beta_1_rms,
        mean_betaI=mean_S, std_betaI=std_S, rms_betaI=rms_S,
        frac_betaI_gt1=frac_gt1, frac_betaI_gt3=frac_gt3, frac_betaI_gt10=frac_gt10,
    ))

    ax_a = axes[0, r]
    ax_a.hist(corrientes, bins=80, color="#2a78d6", alpha=0.85)
    ax_a.axvline(0, color="#333", linewidth=0.8)
    ax_a.set_title(f"ANTES: I  (p={pulsos_pot}, mean={mean_I:.3f}, std={std_I:.3f})")
    ax_a.set_xlabel("corriente I (u. arb.)")

    ax_b = axes[1, r]
    ax_b.hist(escaladas, bins=80, color="#eb6834", alpha=0.85)
    for v in (-3, -1, 0, 1, 3):
        ax_b.axvline(v, color="#333", linewidth=0.7, linestyle=":" if v else "-")
    ax_b.set_title(f"DESPUES: beta*I  (beta={beta}, mean={mean_S:.1f}, std={std_S:.2f})")
    ax_b.set_xlabel("beta * I")

fig.tight_layout()
out_png = os.path.join(OUT_DIR, "corrientes_antes_despues_beta.png")
fig.savefig(out_png)
print(f"\nGuardado: {out_png}")

csv_path = os.path.join(OUT_DIR, "corrientes_antes_despues_beta.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(filas_csv[0].keys()))
    w.writeheader()
    w.writerows(filas_csv)
print(f"Guardado: {csv_path}")

diag_path = os.path.join(OUT_DIR, "diagnostico_voltajes.csv")
with open(diag_path, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["cantidad", "valor"])
    w.writeheader()
    w.writerows(diag_rows)
print(f"Guardado: {diag_path}")
