"""
Plotea accuracy final vs beta, promediando sobre TODAS las series y folds
disponibles para cada beta (no solo la ultima serie), leyendo directamente
los 'resultados_*.npz' de esta carpeta (acc_result: series x folds x epocas).

Para betas con corridas "en_curso" se usan las series ya completadas
(series_completadas < series_totales); el grafico indica cuantas series
entraron en cada punto.

Uso:
    python plot_accuracy_vs_beta.py
"""

import glob
import os

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTADOS_GLOB = os.path.join(HERE, "resultados_SP_1_09_2026_*.npz")
OUT_PNG = os.path.join(HERE, "accuracy_vs_beta.png")


def cargar_datos():
    filas = []
    for path in sorted(glob.glob(RESULTADOS_GLOB)):
        d = np.load(path, allow_pickle=True)
        acc = d["acc_result"]  # (series_completadas, k_folds, epocas)
        beta = float(d["beta"])
        series_completadas = int(d["series_completadas"])
        series_totales = acc.shape[0]  # el array ya viene recortado a las series completadas

        final_por_serie_fold = acc[:, :, -1]  # (series, folds) -- ultima epoca guardada
        valores = final_por_serie_fold.reshape(-1)  # aplana series*folds

        filas.append({
            "beta": beta,
            "path": os.path.basename(path),
            "series_completadas": series_completadas,
            "n_folds": acc.shape[1],
            "n_puntos": valores.size,
            "media": valores.mean(),
            "min": valores.min(),
            "max": valores.max(),
            "std": valores.std(),
        })
    filas.sort(key=lambda f: f["beta"])
    return filas


def main():
    filas = cargar_datos()

    betas = np.array([f["beta"] for f in filas])
    medias = np.array([f["media"] for f in filas])
    minimos = np.array([f["min"] for f in filas])
    maximos = np.array([f["max"] for f in filas])
    series_totales_esperadas = 20  # de la config (entrenamiento.series)

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=140)

    # rango min-max por beta (sobre todas las series*folds disponibles)
    ax.vlines(betas, minimos, maximos, color="#2a78d6", alpha=0.30, linewidth=3, zorder=1)

    # media conectada
    ax.plot(betas, medias, color="#2a78d6", marker="o", markersize=6,
            markerfacecolor="#2a78d6", markeredgecolor="white", markeredgewidth=1.2,
            linewidth=2, zorder=3, label="media (todas las series×folds disponibles)")

    # Etiquetar cada punto no escala: con grillas grandes (>10-12 betas) las
    # anotaciones se pisan y quedan ilegibles. En cambio se etiquetan
    # selectivamente: el pico (mejor beta), los extremos de la grilla, y
    # cualquier corrida incompleta (esas SI importa marcarlas siempre).
    n = len(filas)
    idx_pico = int(np.argmax(medias))
    indices_a_etiquetar = {0, n - 1, idx_pico}

    for i, f in enumerate(filas):
        completo = f["series_completadas"] >= series_totales_esperadas
        if i not in indices_a_etiquetar and completo:
            continue
        etiqueta = f"{f['media']*100:.2f}%"
        if i == idx_pico and completo:
            etiqueta = f"pico: {etiqueta}\n(beta={f['beta']:.0f})"
        if not completo:
            etiqueta += f"\n({f['series_completadas']}/{series_totales_esperadas} series)"
        color = "#52514e" if completo else "#898781"
        ax.annotate(etiqueta, (f["beta"], f["media"]), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=8, color=color, linespacing=1.3)

    # marca visual (borde punteado) para los puntos con corrida aun en curso
    incompletos = [f for f in filas if f["series_completadas"] < series_totales_esperadas]
    if incompletos:
        ax.scatter([f["beta"] for f in incompletos], [f["media"] for f in incompletos],
                   s=110, facecolors="none", edgecolors="#898781", linewidths=1.3,
                   linestyles=(0, (2, 2)), zorder=4, label="beta aún en curso (series parciales)")

    ax.set_xlabel("beta")
    ax.set_ylabel("accuracy final (última época de cada serie)")
    ax.set_title("Accuracy vs beta — media sobre todas las series×folds disponibles\nSP_1_09_2026")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v*100:.1f}%")

    # Ticks del eje x: con pocas betas se muestran todas derechas; con
    # grillas grandes se rotan y se aligeran (una de cada `step`) para que
    # no se superpongan.
    ax.set_xticks(betas)
    ax.set_xticklabels([f"{b:.0f}" for b in betas])
    if n > 15:
        step = max(1, round(n / 18))
        for i, tick_label in enumerate(ax.get_xticklabels()):
            tick_label.set_rotation(90)
            tick_label.set_fontsize(8)
            if i % step != 0 and i != n - 1:
                tick_label.set_visible(False)

    ax.grid(axis="y", color="#e1e0d9", linewidth=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    fig.tight_layout()
    fig.savefig(OUT_PNG)
    print(f"Guardado: {OUT_PNG}")

    print("\nbeta\tseries\tn(series*folds)\tmedia\tstd\tmin\tmax")
    for f in filas:
        print(f"{f['beta']:.0f}\t{f['series_completadas']}/20\t{f['n_puntos']}\t"
              f"{f['media']*100:.2f}%\t{f['std']*100:.2f}%\t{f['min']*100:.2f}%\t{f['max']*100:.2f}%")


if __name__ == "__main__":
    main()
