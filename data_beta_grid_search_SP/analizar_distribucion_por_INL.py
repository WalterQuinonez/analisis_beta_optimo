"""
Generaliza analizar_distribucion_beta.py (que era para UNA sola carpeta) a
TODAS las subcarpetas de grid-search de beta que hay bajo esta carpeta.

Para cada subcarpeta (una curva P/D: cierta cantidad de pulsos p y su 'a',
pensada para dar cierto INL nominal) calcula, por beta, sobre TODAS las
combinaciones serie x fold disponibles de la ULTIMA epoca guardada:
  - media, std, IQR, skew, curtosis (exceso)
  - test de normalidad por beta (Shapiro-Wilk)
  - test de homogeneidad de varianzas entre betas de esa curva (Levene)
  - correlacion de Pearson beta vs std y beta vs skew (¿la dispersion /
    asimetria se mueve de forma sistematica con beta?)

Salidas (en resumen_por_INL/):
  - distribucion_<carpeta>.png       -- violin+box de esa curva sola
                                         (una por cada subcarpeta)
  - dispersión_vs_beta_INL_<tag>.png -- std e IQR vs beta, TODAS las curvas
                                         del mismo INL superpuestas
  - asimetria_vs_beta_INL_<tag>.png  -- skew vs beta, idem superpuestas
  - resumen_distribucion.csv         -- tabla larga (carpeta, beta, media,
                                         std, iqr, skew, kurtosis, shapiro_p)
  - imprime por consola los tests de Levene y las correlaciones de Pearson
    de cada curva

Usa el mismo color por curva que plot_accuracy_vs_beta_por_INL.py (mismo
orden alfabetico de subcarpetas dentro de cada INL) para poder comparar
visualmente entre ambos juegos de figuras.

Uso:
    python analizar_distribucion_por_INL.py
"""

import csv
import glob
import json
import os
import re

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "resumen_por_INL")
NOMBRE_RE = re.compile(r"INL_(?P<inl>.+?)_p_(?P<p>\d+)")

PALETA = ["#2a78d6", "#eb6834", "#3aa66b", "#a34ad6", "#c9a227",
          "#d64a6a", "#4ac9c2", "#7a7a7a"]


def cargar_curva(subcarpeta):
    npz_paths = sorted(glob.glob(os.path.join(subcarpeta, "resultados_*.npz")))
    if not npz_paths:
        return None

    filas = []
    for path in npz_paths:
        d = np.load(path, allow_pickle=True)
        acc = d["acc_result"]
        if acc.size == 0:
            continue
        beta = float(d["beta"])
        series_completadas = int(d["series_completadas"])
        valores = acc[:, :, -1].reshape(-1)
        filas.append({"beta": beta, "series_completadas": series_completadas, "valores": valores})
    filas.sort(key=lambda f: f["beta"])
    if not filas:
        return None

    config_paths = sorted(glob.glob(os.path.join(subcarpeta, "config_*.json")))
    meta = {}
    if config_paths:
        with open(config_paths[0], "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        cpd = cfg.get("curva_pd", {})
        meta = {
            "pulsos_pot": cpd.get("pulsos_pot"),
            "a_pot": cpd.get("a_pot"),
            "inl_pot": cpd.get("inl_pot"),
            "series_totales": cfg.get("entrenamiento", {}).get("series"),
        }

    return {"filas": filas, "meta": meta, "nombre": os.path.basename(subcarpeta)}


def resumir(filas):
    resumen = []
    for f in filas:
        v = f["valores"]
        q1, q3 = np.percentile(v, [25, 75])
        shapiro_p = stats.shapiro(v).pvalue if 3 <= len(v) <= 5000 else np.nan
        resumen.append({
            "beta": f["beta"], "n": v.size, "series": f["series_completadas"],
            "media": v.mean(), "std": v.std(ddof=1), "iqr": q3 - q1,
            "skew": stats.skew(v), "kurtosis": stats.kurtosis(v),
            "shapiro_p": shapiro_p, "min": v.min(), "max": v.max(),
        })
    return resumen


def _aligerar_ticks_x(ax_, betas):
    n = len(betas)
    ax_.set_xticks(betas)
    ax_.set_xticklabels([f"{b:.0f}" for b in betas])
    if n > 15:
        step = max(1, round(n / 18))
        for i, tick_label in enumerate(ax_.get_xticklabels()):
            tick_label.set_rotation(90)
            tick_label.set_fontsize(8)
            if i % step != 0 and i != n - 1:
                tick_label.set_visible(False)


def figura_individual(curva, resumen, color, out_png):
    filas = curva["filas"]
    betas = np.array([r["beta"] for r in resumen])
    stds = np.array([r["std"] for r in resumen])
    iqrs = np.array([r["iqr"] for r in resumen])
    skews = np.array([r["skew"] for r in resumen])
    n = len(betas)

    fig, axes = plt.subplots(3, 1, figsize=(9, 11), dpi=140,
                              gridspec_kw={"height_ratios": [2.2, 1, 1]})

    ax = axes[0]
    datos_pct = [f["valores"] * 100 for f in filas]
    ancho_violin = max(1.5, min(6, 0.6 * (betas[1] - betas[0]) if n > 1 else 6))
    parts = ax.violinplot(datos_pct, positions=betas, widths=ancho_violin,
                           showmedians=False, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(color)
        pc.set_alpha(0.28)
        pc.set_edgecolor("none")
    ax.boxplot(datos_pct, positions=betas, widths=2.2, patch_artist=True,
               showfliers=True, manage_ticks=False,
               boxprops=dict(facecolor=color, alpha=0.75, edgecolor="#184f95"),
               medianprops=dict(color="white", linewidth=1.6),
               whiskerprops=dict(color="#184f95"), capprops=dict(color="#184f95"),
               flierprops=dict(marker="o", markersize=3, markerfacecolor="#898781",
                                markeredgecolor="none", alpha=0.5))
    series_totales = curva["meta"].get("series_totales")
    if series_totales:
        for r in resumen:
            if r["series"] < series_totales:
                ax.axvline(r["beta"], color="#898781", linestyle=":", linewidth=1, alpha=0.5, zorder=0)
    _aligerar_ticks_x(ax, betas)
    ax.set_ylabel("accuracy final (%)")
    ax.set_title(f"{curva['nombre']}\nDistribución de accuracy por beta (violin+boxplot, serie×fold)")
    ax.spines[["top", "right"]].set_visible(False)

    ax2 = axes[1]
    ax2.plot(betas, stds * 100, marker="o", color=color, label="std")
    ax2.plot(betas, iqrs * 100, marker="s", color="#eb6834", label="IQR")
    _aligerar_ticks_x(ax2, betas)
    ax2.set_ylabel("dispersión (%)")
    ax2.legend(frameon=False, fontsize=9)
    ax2.grid(axis="y", color="#e1e0d9", linewidth=1)
    ax2.spines[["top", "right"]].set_visible(False)

    ax3 = axes[2]
    ax3.axhline(0, color="#c3c2b7", linewidth=1)
    ax3.plot(betas, skews, marker="o", color="#4a3aa7")
    _aligerar_ticks_x(ax3, betas)
    ax3.set_xlabel("beta")
    ax3.set_ylabel("asimetría (skew)")
    ax3.grid(axis="y", color="#e1e0d9", linewidth=1)
    ax3.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main():
    subcarpetas = sorted(
        d for d in glob.glob(os.path.join(HERE, "beta_grid_search_SP_INL_*"))
        if os.path.isdir(d)
    )
    if not subcarpetas:
        print("No se encontraron subcarpetas 'beta_grid_search_SP_INL_*' bajo", HERE)
        return
    os.makedirs(OUT_DIR, exist_ok=True)

    grupos = {}
    filas_csv = []

    for sc in subcarpetas:
        nombre = os.path.basename(sc)
        m = NOMBRE_RE.search(nombre)
        inl_tag = m.group("inl") if m else "desconocido"
        curva = cargar_curva(sc)
        if curva is None:
            print(f"  (sin resultados_*.npz todavia en {nombre}, se omite)")
            continue
        resumen = resumir(curva["filas"])
        curva["resumen"] = resumen
        grupos.setdefault(inl_tag, []).append(curva)

        betas = np.array([r["beta"] for r in resumen])
        stds = np.array([r["std"] for r in resumen])
        skews = np.array([r["skew"] for r in resumen])
        r_std, p_std = stats.pearsonr(betas, stds)
        r_skew, p_skew = stats.pearsonr(betas, skews)
        grupos_levene = [f["valores"] for f in curva["filas"]]
        levene_stat, levene_p = stats.levene(*grupos_levene)

        print(f"\n=== {nombre} ===")
        print(f"  Pearson beta vs std:  r={r_std:+.3f}  p={p_std:.4f}")
        print(f"  Pearson beta vs skew: r={r_skew:+.3f}  p={p_skew:.4f}")
        print(f"  Levene ({len(curva['filas'])} betas): stat={levene_stat:.3f}  p={levene_p:.4f}  "
              f"({'rechaza homogeneidad' if levene_p < 0.05 else 'no rechaza homogeneidad'} al 5%)")

        for r in resumen:
            filas_csv.append({
                "inl_tag": inl_tag, "carpeta": nombre,
                "pulsos_pot": curva["meta"].get("pulsos_pot"),
                "beta": r["beta"], "n": r["n"], "series": r["series"],
                "media": r["media"], "std": r["std"], "iqr": r["iqr"],
                "skew": r["skew"], "kurtosis": r["kurtosis"], "shapiro_p": r["shapiro_p"],
            })

    if not grupos:
        print("Ninguna subcarpeta tiene resultados_*.npz cargables todavia.")
        return

    # figuras individuales (una por curva), color = mismo orden/paleta que
    # plot_accuracy_vs_beta_por_INL.py dentro de cada grupo de INL
    for inl_tag, curvas in sorted(grupos.items()):
        for i, curva in enumerate(curvas):
            color = PALETA[i % len(PALETA)]
            out_png = os.path.join(OUT_DIR, f"distribucion_{curva['nombre']}.png")
            figura_individual(curva, curva["resumen"], color, out_png)
            print(f"Guardado: {out_png}")

    # comparaciones por INL: dispersion y asimetria, todas las curvas superpuestas
    for inl_tag, curvas in sorted(grupos.items()):
        fig_std, ax_std = plt.subplots(figsize=(9, 5), dpi=140)
        fig_skew, ax_skew = plt.subplots(figsize=(9, 5), dpi=140)

        for i, curva in enumerate(curvas):
            color = PALETA[i % len(PALETA)]
            resumen = curva["resumen"]
            betas = np.array([r["beta"] for r in resumen])
            stds = np.array([r["std"] for r in resumen]) * 100
            skews = np.array([r["skew"] for r in resumen])
            p_pot = curva["meta"].get("pulsos_pot")
            a_pot = curva["meta"].get("a_pot")
            etiqueta = f"p={p_pot}, a={a_pot:.2f}" if p_pot is not None else curva["nombre"]

            ax_std.plot(betas, stds, marker="o", color=color, markersize=5, linewidth=1.8, label=etiqueta)
            ax_skew.plot(betas, skews, marker="o", color=color, markersize=5, linewidth=1.8, label=etiqueta)

        ax_std.set_xlabel("beta")
        ax_std.set_ylabel("std accuracy (%)")
        ax_std.set_title(f"Dispersión (std) vs beta — INL nominal = {inl_tag}")
        ax_std.grid(axis="y", color="#e1e0d9", linewidth=1)
        ax_std.spines[["top", "right"]].set_visible(False)
        ax_std.legend(frameon=False, fontsize=8.5)
        fig_std.tight_layout()
        out_std = os.path.join(OUT_DIR, f"dispersion_vs_beta_INL_{inl_tag}.png")
        fig_std.savefig(out_std)
        plt.close(fig_std)
        print(f"Guardado: {out_std}")

        ax_skew.axhline(0, color="#c3c2b7", linewidth=1)
        ax_skew.set_xlabel("beta")
        ax_skew.set_ylabel("asimetría (skew)")
        ax_skew.set_title(f"Asimetría vs beta — INL nominal = {inl_tag}")
        ax_skew.grid(axis="y", color="#e1e0d9", linewidth=1)
        ax_skew.spines[["top", "right"]].set_visible(False)
        ax_skew.legend(frameon=False, fontsize=8.5)
        fig_skew.tight_layout()
        out_skew = os.path.join(OUT_DIR, f"asimetria_vs_beta_INL_{inl_tag}.png")
        fig_skew.savefig(out_skew)
        plt.close(fig_skew)
        print(f"Guardado: {out_skew}")

    csv_path = os.path.join(OUT_DIR, "resumen_distribucion.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "inl_tag", "carpeta", "pulsos_pot", "beta", "n", "series",
            "media", "std", "iqr", "skew", "kurtosis", "shapiro_p"])
        w.writeheader()
        w.writerows(filas_csv)
    print(f"\nGuardado: {csv_path}")


if __name__ == "__main__":
    main()
