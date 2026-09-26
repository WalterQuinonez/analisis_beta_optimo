#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analizar_distribucion_softmax.py

Analiza la distribucion de los logits y de las probabilidades
softmax(beta * I) de VALIDACION, reconstruidas EXACTAMENTE (sin reentrenar
nada) a partir de los G_history_*.npz ya guardados del grid-search de beta
para INL=2e-1, p=200 (data_beta_grid_search_SP/beta_grid_search_SP_INL_2e-1_p_200/),
para cada uno de los 40 valores de beta de la grilla.

Version "hermana" de analisis_distribucion_softmax_INL_2e-1_p_50/ (mismo
metodo exacto), para poder comparar contra p=200 si el patron
encontrado en p=50 (acantilado de estabilidad + logit_std de orden 1-2 en
el beta optimo) se repite al cambiar la cantidad de pulsos.

Objetivo: ver si la distribucion softmax en el beta OPTIMO empirico
(beta=200, acc=88.96%, ver
data_beta_grid_search_SP/resumen_por_INL/resumen_optimos.csv) tiene algo
particular frente a betas mas chicos (softmax casi uniforme, subconfiado) o
mas grandes (softmax saturado, sobreconfiado).

Metodo de reconstruccion (identico a
reconstruccion_logits_val_INL_1e-2_p_100_beta1/reconstruir_logits_val.py y a
analisis_distribucion_softmax_INL_2e-1_p_50/analizar_distribucion_softmax.py):
  - forward() de MemDNN es determinista: I = X @ (G_pos - G_neg), sin ruido
    (config de esta corrida tiene ruido=0%), logits = beta * I.
  - particion train/val 100% reproducible via
    KFold(shuffle=True, random_state=fold_seed), con fold_seed guardado en
    cada config (folds.fold_seeds_por_serie). El modo de esta grilla es
    "particion_distinta_por_serie_reusada_entre_betas": la serie 0 usa LA
    MISMA particion de folds en las 40 corridas de beta, asi que los 40
    conjuntos de validacion reconstruidos son directamente comparables.
  - se usa la SERIE=0 y la ULTIMA epoca guardada (epoca 100, modelo
    convergido). Los 5 folds de una serie cubren el dataset completo
    (60000 imagenes) sin solaparse, asi que agregarlos da la distribucion
    softmax sobre TODO MNIST para esa corrida.

Salida (en esta misma carpeta):
    resumen_por_beta.csv
    distribuciones_representativas.npz
    accuracy_vs_beta_reconstruida.png
    confianza_entropia_vs_beta.png
    calibracion_vs_beta.png
    histogramas_maxprob.png
    histogramas_logits.png
    diagramas_reliability.png
"""
import os
import csv
import glob
import json

import numpy as np
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
CARPETA = os.path.join(REPO_ROOT, "data_beta_grid_search_SP", "beta_grid_search_SP_INL_2e-1_p_200")
SERIE = 0
BETA_OPTIMO_EMPIRICO = 200.0  # resumen_por_INL/resumen_optimos.csv: INL=2e-1, p=200
BETAS_REPRESENTATIVOS = [1.0, 50.0, 100.0, 200.0, 300.0, 800.0, 1950.0]
TITULO_CURVA = "INL=2e-1, p=200"

# ---------------------------------------------------------------------------
# dataset (igual que Crossbar_train_experimento_beta_por_capa.py)
# ---------------------------------------------------------------------------
X = np.load(os.path.join(REPO_ROOT, "X_train_mnist.npy")).astype(np.float32)
y = np.load(os.path.join(REPO_ROOT, "y_train_mnist.npy")).astype(np.int64)
print(f"Dataset: X={X.shape}  y={y.shape}")


def softmax(logits):
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=1, keepdims=True)


def entropia(probs, eps=1e-12):
    return -np.sum(probs * np.log(probs + eps), axis=1)


def calibration_error(conf, correcto, n_bins=15):
    """ECE (Expected Calibration Error) + detalle por bin, para diagramas de
    reliability."""
    bordes = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(conf)
    ece_val = 0.0
    centros, accs_bin, confs_bin, pesos_bin = [], [], [], []
    for i in range(n_bins):
        lo, hi = bordes[i], bordes[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        centros.append((lo + hi) / 2)
        if m.sum() == 0:
            accs_bin.append(np.nan)
            confs_bin.append(np.nan)
            pesos_bin.append(0)
            continue
        acc_bin = correcto[m].mean()
        conf_bin = conf[m].mean()
        ece_val += (m.sum() / n) * abs(acc_bin - conf_bin)
        accs_bin.append(acc_bin)
        confs_bin.append(conf_bin)
        pesos_bin.append(m.sum())
    return ece_val, np.array(centros), np.array(accs_bin), np.array(confs_bin), np.array(pesos_bin)


# ---------------------------------------------------------------------------
# recorrer las 40 corridas de beta de esta curva
# ---------------------------------------------------------------------------
configs = sorted(glob.glob(os.path.join(CARPETA, "config_*.json")))
print(f"{len(configs)} configs encontradas en {CARPETA}")

filas = []
distribuciones = {}  # beta -> dict(conf, entropia, logits, correcto)

for cpath in configs:
    with open(cpath, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    beta_val = float(cfg["beta"]["valor"])
    exp_id = cfg["experiment_id"]
    k_folds = cfg["entrenamiento"]["k_folds"]
    fold_seed = cfg["folds"]["fold_seeds_por_serie"][SERIE]

    g_path = os.path.join(CARPETA, f"G_history_{exp_id}_serie{SERIE:03d}.npz")
    r_path = os.path.join(CARPETA, f"resultados_{exp_id}.npz")
    if not (os.path.isfile(g_path) and os.path.isfile(r_path)):
        print(f"  (falta G_history o resultados para beta={beta_val:.0f}, se omite)")
        continue

    g_hist = np.load(g_path)
    G_all = g_hist["G_layer0_history"]  # (k_folds, n_saved, 784, 20)
    epoch_idx = G_all.shape[1] - 1      # ultima epoca guardada (convergida)

    resultados = np.load(r_path, allow_pickle=True)
    acc_result = resultados["acc_result"]        # (series, k_folds, epocas)
    saved_epochs = resultados["saved_epochs"]
    epoch_num = int(saved_epochs[epoch_idx])

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=fold_seed)
    splits = list(kf.split(np.arange(len(X))))

    logits_folds, y_folds = [], []
    for fold in range(k_folds):
        _, val_idx = splits[fold]
        G_arr = G_all[fold, epoch_idx]           # (784, 20)
        W_eff = G_arr[:, 0::2] - G_arr[:, 1::2]  # (784, 10)
        I = X[val_idx] @ W_eff                    # (n_val, 10)
        logits_folds.append(beta_val * I)
        y_folds.append(y[val_idx])

    logits = np.concatenate(logits_folds, axis=0)
    y_val_todos = np.concatenate(y_folds, axis=0)

    probs = softmax(logits)
    pred = probs.argmax(axis=1)
    correcto = (pred == y_val_todos)
    conf = probs.max(axis=1)
    ent = entropia(probs)

    acc_reconstruida = correcto.mean()
    acc_guardada = float(acc_result[SERIE, :, epoch_num - 1].mean())

    conf_correcta = conf[correcto].mean() if correcto.any() else np.nan
    conf_incorrecta = conf[~correcto].mean() if (~correcto).any() else np.nan
    ece_val, *_ = calibration_error(conf, correcto)

    filas.append(dict(
        beta=beta_val,
        acc_reconstruida=acc_reconstruida,
        acc_guardada=acc_guardada,
        conf_media=conf.mean(), conf_std=conf.std(),
        entropia_media=ent.mean(), entropia_std=ent.std(),
        conf_correcta=conf_correcta, conf_incorrecta=conf_incorrecta,
        brecha_confianza=(conf_correcta - conf_incorrecta
                           if not (np.isnan(conf_correcta) or np.isnan(conf_incorrecta))
                           else np.nan),
        ece=ece_val,
        logit_std=logits.std(), logit_mean_abs=np.abs(logits).mean(),
        frac_saturada=(conf > 0.999).mean(),
    ))

    if beta_val in BETAS_REPRESENTATIVOS:
        distribuciones[beta_val] = dict(
            conf=conf.astype(np.float32), entropia=ent.astype(np.float32),
            logits=logits.astype(np.float32).ravel(), correcto=correcto,
        )

    print(f"beta={beta_val:>7.0f}  acc_recon={acc_reconstruida:.4f} (guardada={acc_guardada:.4f}, "
          f"diff={acc_reconstruida - acc_guardada:+.2e})  conf_media={conf.mean():.4f}  "
          f"entropia_media={ent.mean():.4f}  ece={ece_val:.4f}  frac_saturada={(conf > 0.999).mean():.3f}")

filas.sort(key=lambda f: f["beta"])
max_diff_sanity = max(abs(f["acc_reconstruida"] - f["acc_guardada"]) for f in filas)
print(f"\nSanity check: maxima diferencia |acc_reconstruida - acc_guardada| sobre "
      f"{len(filas)} betas: {max_diff_sanity:.2e}")

# ---------------------------------------------------------------------------
# guardar resumen_por_beta.csv
# ---------------------------------------------------------------------------
campos = ["beta", "acc_reconstruida", "acc_guardada", "conf_media", "conf_std",
          "entropia_media", "entropia_std", "conf_correcta", "conf_incorrecta",
          "brecha_confianza", "ece", "logit_std", "logit_mean_abs", "frac_saturada"]
csv_path = os.path.join(HERE, "resumen_por_beta.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=campos)
    w.writeheader()
    w.writerows(filas)
print(f"Guardado: {csv_path}")

# guardar distribuciones representativas para poder rehacer las figuras sin
# recomputar todo
npz_path = os.path.join(HERE, "distribuciones_representativas.npz")
np.savez_compressed(npz_path, betas=np.array(sorted(distribuciones)), **{
    f"conf_b{int(b)}": d["conf"] for b, d in distribuciones.items()
}, **{
    f"entropia_b{int(b)}": d["entropia"] for b, d in distribuciones.items()
}, **{
    f"logits_b{int(b)}": d["logits"] for b, d in distribuciones.items()
}, **{
    f"correcto_b{int(b)}": d["correcto"] for b, d in distribuciones.items()
})
print(f"Guardado: {npz_path}")

# ---------------------------------------------------------------------------
# figuras
# ---------------------------------------------------------------------------
betas_arr = np.array([f["beta"] for f in filas])
idx_optimo = int(np.argmin(np.abs(betas_arr - BETA_OPTIMO_EMPIRICO)))


def marcar_optimo(ax):
    ax.axvline(BETA_OPTIMO_EMPIRICO, color="#d64a6a", linestyle=":", linewidth=1.3,
               label=f"beta optimo = {BETA_OPTIMO_EMPIRICO:.0f}")


# 1) accuracy reconstruida vs beta (sanity check contra la curva ya conocida)
fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
ax.plot(betas_arr, [f["acc_reconstruida"] for f in filas], color="#2a78d6", marker="o",
        markersize=4, linewidth=1.6, label="accuracy reconstruida (serie 0)")
ax.plot(betas_arr, [f["acc_guardada"] for f in filas], color="#7a7a7a", linestyle="--",
        linewidth=1.2, label="accuracy guardada (serie 0, sanity check)")
marcar_optimo(ax)
ax.set_xlabel("beta"); ax.set_ylabel("accuracy (serie 0)")
ax.set_title(f"{TITULO_CURVA} — accuracy reconstruida vs beta")
ax.legend(frameon=False, fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "accuracy_vs_beta_reconstruida.png"))
plt.close(fig)

# 2) confianza media y entropia media vs beta
fig, ax1 = plt.subplots(figsize=(8.5, 5.5), dpi=140)
ax2 = ax1.twinx()
ax1.plot(betas_arr, [f["conf_media"] for f in filas], color="#2a78d6", marker="o",
         markersize=4, linewidth=1.6, label="confianza media (max softmax)")
ax1.fill_between(betas_arr,
                  [f["conf_media"] - f["conf_std"] for f in filas],
                  [f["conf_media"] + f["conf_std"] for f in filas],
                  color="#2a78d6", alpha=0.12)
ax2.plot(betas_arr, [f["entropia_media"] for f in filas], color="#eb6834", marker="s",
         markersize=4, linewidth=1.6, label="entropia media")
marcar_optimo(ax1)
ax1.set_xlabel("beta"); ax1.set_ylabel("confianza media (max softmax)", color="#2a78d6")
ax2.set_ylabel("entropia media (nats)", color="#eb6834")
ax1.tick_params(axis="y", labelcolor="#2a78d6")
ax2.tick_params(axis="y", labelcolor="#eb6834")
ax1.set_title(f"{TITULO_CURVA} — confianza y entropia del softmax vs beta")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=8.5, loc="center right")
ax1.spines["top"].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "confianza_entropia_vs_beta.png"))
plt.close(fig)

# 3) calibracion vs beta: ECE, brecha de confianza correcta-incorrecta, frac saturada
fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), dpi=140)
axes[0].plot(betas_arr, [f["ece"] for f in filas], color="#a34ad6", marker="o", markersize=4)
axes[0].set_ylabel("ECE (error de calibracion)"); axes[0].set_title("ECE vs beta")
axes[1].plot(betas_arr, [f["brecha_confianza"] for f in filas], color="#3aa66b", marker="o", markersize=4)
axes[1].set_ylabel("confianza(correctos) - confianza(incorrectos)")
axes[1].set_title("Separacion de confianza vs beta")
axes[2].plot(betas_arr, [f["frac_saturada"] for f in filas], color="#c9a227", marker="o", markersize=4)
axes[2].set_ylabel("fraccion con max-prob > 0.999"); axes[2].set_title("Saturacion del softmax vs beta")
for ax in axes:
    marcar_optimo(ax)
    ax.set_xlabel("beta")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
fig.suptitle(f"{TITULO_CURVA} — metricas de calibracion vs beta")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "calibracion_vs_beta.png"))
plt.close(fig)

# 4) histogramas de max-prob (confianza por muestra) para betas representativos
colores_rep = {1.0: "#7a7a7a", 50.0: "#4ac9c2", 100.0: "#d64a6a", 200.0: "#c9a227",
               300.0: "#3aa66b", 800.0: "#eb6834", 1950.0: "#2a78d6"}
fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=140)
bins = np.linspace(0.1, 1.0, 60)
for b in BETAS_REPRESENTATIVOS:
    if b not in distribuciones:
        continue
    etiqueta = f"beta={b:.0f}" + ("  (OPTIMO)" if b == BETA_OPTIMO_EMPIRICO else "")
    ax.hist(distribuciones[b]["conf"], bins=bins, density=True, histtype="step",
            linewidth=2.2 if b == BETA_OPTIMO_EMPIRICO else 1.4,
            color=colores_rep.get(b, "#000000"), label=etiqueta)
ax.set_xlabel("max softmax (confianza de la prediccion)")
ax.set_ylabel("densidad")
ax.set_title(f"{TITULO_CURVA} — distribucion de confianza softmax por beta")
ax.legend(frameon=False, fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "histogramas_maxprob.png"))
plt.close(fig)

# 5) histogramas de logits crudos (beta*I, las 10 salidas juntas) por beta representativo
fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=140)
for b in BETAS_REPRESENTATIVOS:
    if b not in distribuciones:
        continue
    logits_b = distribuciones[b]["logits"]
    rango = np.percentile(np.abs(logits_b), 99)
    bins = np.linspace(-rango, rango, 80)
    etiqueta = f"beta={b:.0f} (std={logits_b.std():.2f})" + ("  (OPTIMO)" if b == BETA_OPTIMO_EMPIRICO else "")
    ax.hist(logits_b, bins=bins, density=True, histtype="step",
            linewidth=2.2 if b == BETA_OPTIMO_EMPIRICO else 1.4,
            color=colores_rep.get(b, "#000000"), label=etiqueta)
ax.set_xlabel("logit = beta * I (las 10 salidas, todas las muestras)")
ax.set_ylabel("densidad")
ax.set_title(f"{TITULO_CURVA} — distribucion de logits crudos por beta\n"
              "(eje x recortado al percentil 99 de cada beta para poder comparar formas)")
ax.legend(frameon=False, fontsize=8.5)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "histogramas_logits.png"))
plt.close(fig)

# 6) diagramas de reliability (confianza esperada vs accuracy real, por bin)
n_rep = len(BETAS_REPRESENTATIVOS)
n_cols = 4
n_rows = int(np.ceil(n_rep / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 3.6 * n_rows), dpi=140,
                          sharex=True, sharey=True)
axes_flat = np.array(axes).reshape(-1)
for ax, b in zip(axes_flat, BETAS_REPRESENTATIVOS):
    if b not in distribuciones:
        ax.axis("off")
        continue
    conf = distribuciones[b]["conf"]
    correcto = distribuciones[b]["correcto"]
    ece_val, centros, accs_bin, confs_bin, pesos_bin = calibration_error(conf, correcto)
    ax.plot([0, 1], [0, 1], color="#7a7a7a", linestyle="--", linewidth=1)
    ax.bar(centros, accs_bin, width=1 / 15 * 0.9, color=colores_rep.get(b, "#2a78d6"), alpha=0.75,
           edgecolor="white")
    titulo = f"beta={b:.0f}  (ECE={ece_val:.3f})"
    if b == BETA_OPTIMO_EMPIRICO:
        titulo += "  <- OPTIMO"
    ax.set_title(titulo, fontsize=9.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
for ax in axes_flat[len(BETAS_REPRESENTATIVOS):]:
    ax.axis("off")
for ax in axes[-1, :] if n_rows > 1 else [axes_flat[-1]]:
    ax.set_xlabel("confianza (max softmax)")
for ax in (axes[:, 0] if n_rows > 1 else [axes_flat[0]]):
    ax.set_ylabel("accuracy real")
fig.suptitle(f"{TITULO_CURVA} — diagramas de reliability por beta\n"
              "(diagonal = calibracion perfecta; barra por debajo = sobreconfiado)")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "diagramas_reliability.png"))
plt.close(fig)

print("\nGuardadas figuras: accuracy_vs_beta_reconstruida.png, confianza_entropia_vs_beta.png, "
      "calibracion_vs_beta.png, histogramas_maxprob.png, histogramas_logits.png, "
      "diagramas_reliability.png")
