#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
estimar_beta_optimo_propuesta_v2.py

Repite el metodo de `estimar_beta_optimo_propuesta.py` (mismo Monte Carlo
corregido de pesos W y corriente I, mismo "paso medio" de la curva P/D,
misma regresion log-log para beta_optimo) pero con la data nueva: ya NO son
6 curvas (2 INL nominal x 3 pulsos), son **18 curvas (6 INL nominal x 3
pulsos)**, con INL nominal cubriendo casi 4.4 ordenes de magnitud
(2e-1 .. 4.82e-6) en vez de solo 2 valores. Esto permite recien ahora poner
a prueba de verdad la FORMA de la ley de potencia en sigma_W (antes estaba
determinada por un solo contraste entre 2 grupos de INL, ver Seccion 5 del
reporte v1).

Ademas, agrega el analisis pedido: "las curvas de accuracy vs beta tienden
a aplanarse (mas anchas/menos picudas) a medida que el INL se achica,
porque la curva P/D se vuelve mas lineal". Se separan las DOS partes de esa
idea:
  (a) "la curva P/D se vuelve mas lineal cuando el INL se achica" -- esto es
      CASI TAUTOLOGICO: `calcular_indice_nl()` (la funcion que define INL)
      es exactamente el exceso de longitud de la curva sobre la distancia
      recta normalizada, i.e. INL = 0 significa curva perfectamente lineal
      por DEFINICION. Este script solo lo grafica para dejarlo concreto,
      no lo "prueba" (no hay nada que probar: es la definicion de INL).
  (b) "la curva de accuracy vs beta se aplana (el optimo se vuelve menos
      picudo/mas ancho) cuando el INL se achica" -- esto SI es una
      afirmacion empirica no trivial. Se mide con dos descriptores de
      "planitud" por curva (ancho de la meseta al 99% del pico, y
      curvatura local en el pico via un ajuste cuadratico) y se correlaciona
      contra INL nominal a lo largo de las 18 curvas reales.

No se modifica ningun archivo existente del proyecto. Solo se importan
(sin copiar ni alterar) `generar_curvas_pot_dep` y `G0_initialization` de
`MemCrossbarClass_beta_por_capa.py`, y `cargar_curva` de
`data_beta_grid_search_SP/plot_accuracy_vs_beta_por_INL.py`.
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
sys.path.insert(0, DATA_DIR)
from MemCrossbarClass_beta_por_capa import generar_curvas_pot_dep, G0_initialization  # noqa: E402
from plot_accuracy_vs_beta_por_INL import cargar_curva  # noqa: E402

N_IN = 784
SEED = 35
MC_PESOS = 200_000       # tamano de la muestra Monte Carlo de pesos (D_in ficticio)
MC_CORRIENTES = 300_000  # cantidad de "corrientes" I simuladas (sumas de 784 terminos)
CHUNK = 30_000

# paleta fija por INL nominal (misma que usan los otros scripts del proyecto,
# asi los colores son consistentes entre todas las figuras)
COLOR_POR_INL = {
    "2e-1": "#d64a6a", "1e-2": "#c9a227", "1e-3": "#3aa66b",
    "1e-4": "#4ac9c2", "1e-5": "#2a78d6", "4.82e-6": "#a34ad6",
}
MARKER_POR_P = {50: "o", 100: "s", 200: "^"}


def cargar_configs_reales():
    """Lee pulsos_pot/a_pot/Gmin/Gmax/concavidad de los config_*.json REALES
    de cada una de las 18 corridas de grid-search (no se transcriben a mano)."""
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
    """Igual que en estimar_beta_optimo_propuesta.py (v1): Monte Carlo
    corregido de pesos (G0_initialization real) y corriente (pixeles reales
    de MNIST, no equipesados)."""
    pot, dep, ratio, inl_pot, inl_dep = generar_curvas_pot_dep(
        cfg["pulsos_pot"], cfg["pulsos_dep"], cfg["a_pot"], cfg["a_dep"],
        cfg["Gmin"], cfg["Gmax"], cfg["concavidad_dep"], cfg["concavidad_pot"])

    pot_t = np.asarray(pot, dtype=np.float64)
    dep_t = np.asarray(dep, dtype=np.float64)
    import torch
    torch.manual_seed(seed)
    G = G0_initialization("random", torch.tensor(pot_t), torch.tensor(dep_t), MC_PESOS, 1)
    W = (G[:, 0] - G[:, 1]).numpy()
    sigma_W = W.std()
    mean_W = W.mean()
    paso_medio_pot = float(np.mean(np.abs(np.diff(pot))))

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
    )


def planitud_de_curva(betas, medias):
    """Dos descriptores de 'que tan picudo/plano' es el optimo de una curva
    accuracy vs beta:
      - ancho99: ancho (en beta) de la meseta contigua alrededor del pico
        donde accuracy >= 99% del pico, con interpolacion lineal entre
        puntos de la grilla para el borde exacto (no solo el punto de
        grilla mas cercano).
      - curvatura: coeficiente cuadratico de un ajuste local (parabola)
        centrado en el pico, sobre una ventana de +/-3 puntos de grilla
        alrededor del maximo (o menos si el pico esta en el borde).
        Mas negativo = pico mas agudo. Mas cerca de 0 = mas plano.
    """
    idx_pico = int(np.argmax(medias))
    beta_pico = betas[idx_pico]
    acc_pico = medias[idx_pico]
    umbral = 0.99 * acc_pico

    # --- ancho de meseta al 99%, con interpolacion lineal en los bordes ---
    def cruce_interpolado(i0, i1):
        b0, b1 = betas[i0], betas[i1]
        a0, a1 = medias[i0], medias[i1]
        if a1 == a0:
            return b1
        t = (umbral - a0) / (a1 - a0)
        return b0 + t * (b1 - b0)

    # borde izquierdo
    i = idx_pico
    while i > 0 and medias[i - 1] >= umbral:
        i -= 1
    if i > 0:
        beta_izq = cruce_interpolado(i - 1, i)
    else:
        beta_izq = betas[0]

    # borde derecho
    j = idx_pico
    n = len(medias)
    while j < n - 1 and medias[j + 1] >= umbral:
        j += 1
    if j < n - 1:
        beta_der = cruce_interpolado(j, j + 1)
    else:
        beta_der = betas[-1]

    ancho99 = beta_der - beta_izq

    # --- curvatura local (parabola) alrededor del pico ---
    lo = max(0, idx_pico - 3)
    hi = min(n, idx_pico + 4)
    bx = betas[lo:hi]
    by = medias[lo:hi]
    if len(bx) >= 3:
        coef = np.polyfit(bx - beta_pico, by, 2)
        curvatura = coef[0]
    else:
        curvatura = np.nan

    return dict(beta_pico=beta_pico, acc_pico=acc_pico, ancho99=ancho99,
                ancho99_frac=ancho99 / beta_pico, curvatura=curvatura)


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
    n_curvas = len(empiricos)
    print(f"Curvas encontradas en resumen_optimos.csv: {n_curvas}")

    filas = []
    for row in empiricos:
        carpeta = row["carpeta"]
        cfg = configs[carpeta]
        r = estudiar_pesos_y_corriente(cfg, X_flat, seed=SEED)
        beta_emp = float(row["beta_optimo"])

        # planitud de la curva accuracy vs beta real (para el analisis (b))
        curva = cargar_curva(os.path.join(DATA_DIR, carpeta))
        betas_c = np.array([f["beta"] for f in curva["filas"]])
        medias_c = np.array([f["media"] for f in curva["filas"]])
        plan = planitud_de_curva(betas_c, medias_c)

        sigma_I_clt = np.sqrt(N_IN) * r["sigma_W"] * V_rms_real
        kappa_mc = beta_emp * r["sigma_I"]
        kappa_clt = beta_emp * sigma_I_clt

        filas.append(dict(
            carpeta=carpeta, inl_tag=row["inl_tag"], inl_pot_real=float(row["inl_pot_real"]),
            pulsos_pot=cfg["pulsos_pot"], a_pot=cfg["a_pot"],
            sigma_W=r["sigma_W"], mean_W=r["mean_W"], paso_medio_pot=r["paso_medio_pot"],
            mean_I_mc=r["mean_I"], sigma_I_mc=r["sigma_I"], sigma_I_clt=sigma_I_clt,
            beta_optimo_empirico=beta_emp,
            kappa_mc=kappa_mc, kappa_clt=kappa_clt,
            ancho99=plan["ancho99"], ancho99_frac=plan["ancho99_frac"], curvatura=plan["curvatura"],
            pot=r["pot"],
        ))
        print(f"{carpeta:<42} sigma_W={r['sigma_W']:.4e}  paso_medio={r['paso_medio_pot']:.3e}  "
              f"beta_emp={beta_emp:.0f}  ancho99={plan['ancho99']:.0f}  curvatura={plan['curvatura']:.3e}")

    n = len(filas)
    sigmaW = np.array([f["sigma_W"] for f in filas])
    paso = np.array([f["paso_medio_pot"] for f in filas])
    inl_real = np.array([f["inl_pot_real"] for f in filas])
    pulsos = np.array([f["pulsos_pot"] for f in filas], dtype=float)
    beta = np.array([f["beta_optimo_empirico"] for f in filas])
    y = np.log(beta)

    def ajustar(X, nombre):
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ coef
        r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
        gl = len(y) - X.shape[1]
        print(f"\n[{nombre}] coef={coef}  R2={r2:.4f}  (n={len(y)}, gl_residuales={gl})")
        return coef, pred, r2

    ones = np.ones(n)
    X_sw = np.column_stack([ones, np.log(sigmaW)])
    coef_sw, pred_sw, r2_sw = ajustar(X_sw, "solo sigma_W")

    X_pm = np.column_stack([ones, np.log(paso)])
    coef_pm, pred_pm, r2_pm = ajustar(X_pm, "solo paso_medio")

    X_comb = np.column_stack([ones, np.log(sigmaW), np.log(paso)])
    coef_comb, pred_comb, r2_comb = ajustar(X_comb, "combinado (sigma_W y paso_medio)")

    X_directo = np.column_stack([ones, np.log(inl_real), np.log(pulsos)])
    coef_directo, pred_directo, r2_directo = ajustar(X_directo, "directo (INL_real y pulsos)")

    beta_pred_comb = np.exp(pred_comb)
    beta_pred_directo = np.exp(pred_directo)
    print("\nbeta_optimo empirico vs. predicho (ecuacion combinada sigma_W+paso_medio):")
    for f, bp in zip(filas, beta_pred_comb):
        f["beta_pred_combinada"] = bp
        print(f"  {f['carpeta']:<42} emp={f['beta_optimo_empirico']:>5.0f}  pred={bp:>7.1f}  "
              f"error={100*(bp/f['beta_optimo_empirico']-1):+6.1f}%")
    for f, bp in zip(filas, beta_pred_directo):
        f["beta_pred_directo"] = bp

    # ---------------------------------------------------------------
    # analisis (b): planitud de la curva de accuracy vs INL
    # ---------------------------------------------------------------
    log_inl = np.log(inl_real)
    log_ancho = np.log(np.array([f["ancho99_frac"] for f in filas]))
    log_curv = np.log(np.abs(np.array([f["curvatura"] for f in filas])))
    r_ancho = np.corrcoef(log_inl, log_ancho)[0, 1]
    r_curv = np.corrcoef(log_inl, log_curv)[0, 1]
    print(f"\nCorrelacion (log INL, log ancho99_frac) = {r_ancho:+.3f}  (n={n})")
    print(f"Correlacion (log INL, log |curvatura|)   = {r_curv:+.3f}  (n={n})")

    # ---------------------------------------------------------------
    # guardar CSVs
    # ---------------------------------------------------------------
    csv_out = os.path.join(OUT_DIR, "resultados_formula_propuesta_v2.csv")
    campos = ["carpeta", "inl_tag", "inl_pot_real", "pulsos_pot", "a_pot", "sigma_W", "mean_W",
              "paso_medio_pot", "mean_I_mc", "sigma_I_mc", "sigma_I_clt", "beta_optimo_empirico",
              "kappa_mc", "kappa_clt", "beta_pred_combinada", "beta_pred_directo",
              "ancho99", "ancho99_frac", "curvatura"]
    with open(csv_out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        for f in filas:
            w.writerow({k: f[k] for k in campos})
    print(f"\nGuardado: {csv_out}")

    coef_path = os.path.join(OUT_DIR, "coeficientes_regresion_v2.csv")
    with open(coef_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["modelo", "intercept", "coef_log_sigma_W", "coef_log_paso_medio",
                    "coef_log_INL", "coef_log_pulsos", "R2", "n", "gl_residuales"])
        w.writerow(["solo_sigma_W", coef_sw[0], coef_sw[1], "", "", "", r2_sw, n, n - 2])
        w.writerow(["solo_paso_medio", coef_pm[0], "", coef_pm[1], "", "", r2_pm, n, n - 2])
        w.writerow(["combinado_sigmaW_pasomedio", coef_comb[0], coef_comb[1], coef_comb[2],
                    "", "", r2_comb, n, n - 3])
        w.writerow(["directo_INL_pulsos", coef_directo[0], "", "", coef_directo[1],
                    coef_directo[2], r2_directo, n, n - 3])
    print(f"Guardado: {coef_path}")

    # ---------------------------------------------------------------
    # figuras
    # ---------------------------------------------------------------

    # (1) curvas P/D normalizadas, una por INL nominal (p=100), para dejar
    # concreto que INL chico == curva mas cercana a la recta (por definicion)
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=140)
    for f in filas:
        if f["pulsos_pot"] != 100:
            continue
        pot = np.asarray(f["pot"])
        pot_norm = (pot - pot[0]) / (pot[-1] - pot[0])
        x_norm = np.linspace(0, 1, len(pot_norm))
        ax.plot(x_norm, pot_norm, color=COLOR_POR_INL[f["inl_tag"]], linewidth=2,
                label=f"INL={f['inl_tag']} (a={f['a_pot']:.1f})")
    ax.plot([0, 1], [0, 1], "--", color="#999", linewidth=1, label="recta (INL=0)")
    ax.set_xlabel("pulso normalizado")
    ax.set_ylabel("conductancia normalizada")
    ax.set_title("Curva P/D normalizada por INL nominal (p=100)\n"
                  "INL chico -> curva mas cercana a la recta (por definicion de INL)")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "curvas_PD_normalizadas_vs_INL_v2.png"))
    plt.close(fig)

    # (2) planitud del optimo (ancho99_frac y curvatura) vs INL
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=140)
    for f in filas:
        c = COLOR_POR_INL[f["inl_tag"]]
        m = MARKER_POR_P[f["pulsos_pot"]]
        axes[0].scatter(f["inl_pot_real"], f["ancho99_frac"], color=c, marker=m, s=70, zorder=3)
        axes[1].scatter(f["inl_pot_real"], abs(f["curvatura"]), color=c, marker=m, s=70, zorder=3)
    for ax, ylab, r in zip(axes, ["ancho meseta @99% / beta_optimo", "|curvatura| en el pico"],
                            [r_ancho, r_curv]):
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("INL real")
        ax.set_ylabel(ylab)
        ax.set_title(f"r(log-log) = {r:+.2f}")
        ax.grid(True, which="both", color="#e8e7e0", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    for tag, c in COLOR_POR_INL.items():
        axes[0].scatter([], [], color=c, label=f"INL {tag}")
    for p, m in MARKER_POR_P.items():
        axes[0].scatter([], [], color="#555", marker=m, label=f"p={p}")
    axes[0].legend(frameon=False, fontsize=7.5, ncol=2, loc="best")
    fig.suptitle("¿Se aplana el optimo de accuracy-vs-beta cuando el INL se achica? (18 curvas)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "planitud_vs_INL_v2.png"))
    plt.close(fig)

    # (3) beta predicho (combinado) vs empirico, ahora con 18 puntos
    fig, ax = plt.subplots(figsize=(6.5, 6), dpi=140)
    lo = min(beta.min(), beta_pred_comb.min()) * 0.7
    hi = max(beta.max(), beta_pred_comb.max()) * 1.3
    ax.plot([lo, hi], [lo, hi], "--", color="#999", label="identidad")
    for f, bp in zip(filas, beta_pred_comb):
        c = COLOR_POR_INL[f["inl_tag"]]
        m = MARKER_POR_P[f["pulsos_pot"]]
        ax.scatter(bp, f["beta_optimo_empirico"], s=80, color=c, marker=m, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\beta_{pred}$ (ecuación combinada: $\sigma_W^{b}\cdot$paso$_{medio}^{d}$)")
    ax.set_ylabel(r"$\beta_{óptimo}$ empírico (grid-search real)")
    ax.set_title(f"Ecuación combinada vs. β óptimo empírico (18 curvas, R²={r2_comb:.3f})")
    for tag, c in COLOR_POR_INL.items():
        ax.scatter([], [], color=c, label=f"INL {tag}")
    for p, m in MARKER_POR_P.items():
        ax.scatter([], [], color="#555", marker=m, label=f"p={p}")
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="best")
    ax.grid(True, which="both", color="#e8e7e0", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "beta_pred_combinada_vs_empirico_v2.png"))
    plt.close(fig)

    # (4) comparacion R2 de los 4 modelos
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
    modelos = ["solo\nsigma_W", "solo\npaso_medio", "combinado\n(sigma_W+paso)", "directo\n(INL+pulsos)"]
    r2s = [r2_sw, r2_pm, r2_comb, r2_directo]
    ax.bar(modelos, r2s, color=["#2a78d6", "#eb6834", "#3f7d54", "#a34ad6"])
    for i, v in enumerate(r2s):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(f"R² (ajuste log-log a los {n} β óptimo reales)")
    ax.set_title("Comparación de modelos (18 curvas: 6 INL nominal x 3 pulsos)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "comparacion_R2_v2.png"))
    plt.close(fig)

    print("\nGuardadas figuras: curvas_PD_normalizadas_vs_INL_v2.png, planitud_vs_INL_v2.png, "
          "beta_pred_combinada_vs_empirico_v2.png, comparacion_R2_v2.png")


if __name__ == "__main__":
    main()
