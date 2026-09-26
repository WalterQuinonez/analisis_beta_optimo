"""
Estudia como cambia la DISTRIBUCION de accuracy (no solo la media) con beta.

Para cada beta junta accuracy final (ultima epoca guardada) de todas las
combinaciones serie x fold disponibles en 'resultados_*.npz', y calcula:
  - estadisticos descriptivos (media, std, IQR, skew, curtosis)
  - test de normalidad (Shapiro-Wilk) por beta
  - test de homogeneidad de varianzas entre betas (Levene)
  - correlacion de Pearson entre beta y (std, skew) para ver si la forma
    de la distribucion se mueve de manera sistematica con beta

Genera:
  - distribucion_accuracy_vs_beta.png  (violin plot + std/IQR vs beta + skew vs beta)
  - imprime la tabla de estadisticos y los tests por consola

Uso:
    python analizar_distribucion_beta.py
"""

import glob
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTADOS_GLOB = os.path.join(HERE, "resultados_SP_1_09_2026_*.npz")
OUT_PNG = os.path.join(HERE, "distribucion_accuracy_vs_beta.png")
SERIES_TOTALES_ESPERADAS = 20


def cargar_datos():
    filas = []
    for path in sorted(glob.glob(RESULTADOS_GLOB)):
        d = np.load(path, allow_pickle=True)
        acc = d["acc_result"]  # (series_completadas, k_folds, epocas)
        beta = float(d["beta"])
        series_completadas = int(d["series_completadas"])

        valores = acc[:, :, -1].reshape(-1)  # ultima epoca guardada, aplanado series*folds

        filas.append({
            "beta": beta,
            "series_completadas": series_completadas,
            "valores": valores,
        })
    filas.sort(key=lambda f: f["beta"])
    return filas


def resumir(filas):
    resumen = []
    for f in filas:
        v = f["valores"]
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        shapiro_p = stats.shapiro(v).pvalue if len(v) >= 3 else np.nan
        resumen.append({
            "beta": f["beta"],
            "n": v.size,
            "series": f["series_completadas"],
            "media": v.mean(),
            "std": v.std(ddof=1),
            "iqr": q3 - q1,
            "skew": stats.skew(v),
            "kurtosis": stats.kurtosis(v),  # exceso (normal = 0)
            "shapiro_p": shapiro_p,
            "min": v.min(),
            "max": v.max(),
        })
    return resumen


def main():
    filas = cargar_datos()
    resumen = resumir(filas)

    print(f"{'beta':>5} {'n':>4} {'series':>7} {'media%':>8} {'std%':>7} {'iqr%':>7} "
          f"{'skew':>7} {'exc.kurt':>9} {'shapiro_p':>10}")
    for r in resumen:
        print(f"{r['beta']:5.0f} {r['n']:4d} {r['series']:4d}/20 "
              f"{r['media']*100:8.2f} {r['std']*100:7.3f} {r['iqr']*100:7.3f} "
              f"{r['skew']:7.2f} {r['kurtosis']:9.2f} {r['shapiro_p']:10.3f}")

    betas = np.array([r["beta"] for r in resumen])
    stds = np.array([r["std"] for r in resumen])
    iqrs = np.array([r["iqr"] for r in resumen])
    skews = np.array([r["skew"] for r in resumen])

    # tendencia: correlacion de pearson beta vs std, beta vs skew
    r_std, p_std = stats.pearsonr(betas, stds)
    r_skew, p_skew = stats.pearsonr(betas, skews)
    print(f"\nCorrelacion Pearson beta vs std:   r={r_std:+.3f}  p={p_std:.4f}")
    print(f"Correlacion Pearson beta vs skew:  r={r_skew:+.3f}  p={p_skew:.4f}")

    # homogeneidad de varianza entre TODAS las betas (Levene, robusto a no-normalidad)
    grupos = [f["valores"] for f in filas]
    levene_stat, levene_p = stats.levene(*grupos)
    print(f"\nLevene (igualdad de varianzas entre las {len(filas)} betas): "
          f"stat={levene_stat:.3f}  p={levene_p:.4f}  "
          f"({'rechaza homogeneidad' if levene_p < 0.05 else 'no rechaza homogeneidad'} al 5%)")

    n = len(filas)

    def _aligerar_ticks_x(ax_):
        """Con pocas betas se muestran todas las etiquetas derechas; con
        grillas grandes se rotan y se aligera una de cada `step` para que
        no se superpongan."""
        ax_.set_xticks(betas)
        ax_.set_xticklabels([f"{b:.0f}" for b in betas])
        if n > 15:
            step = max(1, round(n / 18))
            for i, tick_label in enumerate(ax_.get_xticklabels()):
                tick_label.set_rotation(90)
                tick_label.set_fontsize(8)
                if i % step != 0 and i != n - 1:
                    tick_label.set_visible(False)

    # --- figura ---
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), dpi=140,
                              gridspec_kw={"height_ratios": [2.2, 1, 1]})

    ax = axes[0]
    datos_pct = [f["valores"] * 100 for f in filas]
    ancho_violin = max(1.5, min(6, 0.6 * (betas[1] - betas[0]) if n > 1 else 6))
    parts = ax.violinplot(datos_pct, positions=betas, widths=ancho_violin,
                           showmedians=False, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("#2a78d6")
        pc.set_alpha(0.28)
        pc.set_edgecolor("none")
    bp = ax.boxplot(datos_pct, positions=betas, widths=2.2, patch_artist=True,
                     showfliers=True, manage_ticks=False,
                     boxprops=dict(facecolor="#2a78d6", alpha=0.75, edgecolor="#184f95"),
                     medianprops=dict(color="white", linewidth=1.6),
                     whiskerprops=dict(color="#184f95"),
                     capprops=dict(color="#184f95"),
                     flierprops=dict(marker="o", markersize=3, markerfacecolor="#898781",
                                      markeredgecolor="none", alpha=0.5))
    incompletos = {f["beta"] for f in filas if f["series_completadas"] < SERIES_TOTALES_ESPERADAS}
    for b in incompletos:
        ax.axvline(b, color="#898781", linestyle=":", linewidth=1, alpha=0.5, zorder=0)
    _aligerar_ticks_x(ax)
    ax.set_ylabel("accuracy final (%)")
    ax.set_title("Distribución de accuracy por beta (violin + boxplot, serie×fold individuales)\n"
                  "líneas punteadas = beta con corrida aún en curso (menos series)")
    ax.spines[["top", "right"]].set_visible(False)

    ax2 = axes[1]
    ax2.plot(betas, stds * 100, marker="o", color="#2a78d6", label="std")
    ax2.plot(betas, iqrs * 100, marker="s", color="#eb6834", label="IQR")
    _aligerar_ticks_x(ax2)
    ax2.set_ylabel("dispersión (%)")
    ax2.set_title(f"Dispersión vs beta  (Pearson r={r_std:+.2f}, p={p_std:.3f} para std)")
    ax2.legend(frameon=False, fontsize=9)
    ax2.grid(axis="y", color="#e1e0d9", linewidth=1)
    ax2.spines[["top", "right"]].set_visible(False)

    ax3 = axes[2]
    ax3.axhline(0, color="#c3c2b7", linewidth=1)
    ax3.plot(betas, skews, marker="o", color="#4a3aa7")
    _aligerar_ticks_x(ax3)
    ax3.set_xlabel("beta")
    ax3.set_ylabel("asimetría (skew)")
    ax3.set_title(f"Asimetría vs beta  (Pearson r={r_skew:+.2f}, p={p_skew:.3f})")
    ax3.grid(axis="y", color="#e1e0d9", linewidth=1)
    ax3.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_PNG)
    print(f"\nGuardado: {OUT_PNG}")


if __name__ == "__main__":
    main()
