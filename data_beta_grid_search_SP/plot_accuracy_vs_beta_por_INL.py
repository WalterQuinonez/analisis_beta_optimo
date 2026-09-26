"""
Junta TODAS las corridas de grid-search de beta que hay bajo esta carpeta
(una subcarpeta por curva P/D: distinta cantidad de pulsos y/o distinto
parametro 'a', pensados para dar el mismo INL nominal) y arma, para cada
INL nominal, UNA figura con todas esas curvas superpuestas (accuracy media
vs beta), para ver si el beta optimo coincide entre curvas aunque cambien
la cantidad de pasos y el parametro 'a'.

Cada subcarpeta se identifica por su nombre:
    beta_grid_search_SP_INL_<inl_tag>_p_<pulsos>[...]
El INL nominal (para agrupar) sale del <inl_tag> del nombre de carpeta.
Los valores reales (pulsos_pot, a_pot, inl_pot, inl_dep) se leen de un
config_*.json de esa subcarpeta.

Para cada beta dentro de una subcarpeta, la accuracy usada es la media de
la ULTIMA epoca guardada, sobre todas las series x folds ya completados
(igual criterio que plot_accuracy_vs_beta.py original).

Uso:
    python plot_accuracy_vs_beta_por_INL.py
Salida:
    resumen_por_INL/accuracy_vs_beta_INL_<tag>.png   (una por cada INL)
    resumen_por_INL/resumen_optimos.csv              (tabla de optimos)
"""

import csv
import glob
import json
import os
import re

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "resumen_por_INL")
NOMBRE_RE = re.compile(r"INL_(?P<inl>.+?)_p_(?P<p>\d+)")

# Paleta fija para que el mismo p/a siempre tenga el mismo color entre
# figuras (mas facil de comparar a simple vista).
PALETA = ["#2a78d6", "#eb6834", "#3aa66b", "#a34ad6", "#c9a227",
          "#d64a6a", "#4ac9c2", "#7a7a7a"]


def cargar_curva(subcarpeta):
    """Lee todos los resultados_*.npz de una subcarpeta y arma la curva
    accuracy media vs beta (mismo criterio que plot_accuracy_vs_beta.py)."""
    npz_paths = sorted(glob.glob(os.path.join(subcarpeta, "resultados_*.npz")))
    if not npz_paths:
        return None

    filas = []
    for path in npz_paths:
        d = np.load(path, allow_pickle=True)
        acc = d["acc_result"]  # (series_completadas, k_folds, epocas)
        if acc.size == 0:
            continue
        beta = float(d["beta"])
        series_completadas = int(d["series_completadas"])
        final_por_serie_fold = acc[:, :, -1]
        valores = final_por_serie_fold.reshape(-1)
        filas.append({
            "beta": beta,
            "series_completadas": series_completadas,
            "media": valores.mean(),
            "min": valores.min(),
            "max": valores.max(),
            "n_puntos": valores.size,
        })
    filas.sort(key=lambda f: f["beta"])
    if not filas:
        return None

    # metadata de la curva P/D: de cualquier config_*.json de la subcarpeta
    config_paths = sorted(glob.glob(os.path.join(subcarpeta, "config_*.json")))
    meta = {}
    if config_paths:
        with open(config_paths[0], "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        cpd = cfg.get("curva_pd", {})
        meta = {
            "pulsos_pot": cpd.get("pulsos_pot"),
            "pulsos_dep": cpd.get("pulsos_dep"),
            "a_pot": cpd.get("a_pot"),
            "a_dep": cpd.get("a_dep"),
            "inl_pot": cpd.get("inl_pot"),
            "inl_dep": cpd.get("inl_dep"),
            "series_totales": cfg.get("entrenamiento", {}).get("series"),
        }

    return {"filas": filas, "meta": meta, "nombre": os.path.basename(subcarpeta)}


def main():
    subcarpetas = sorted(
        d for d in glob.glob(os.path.join(HERE, "beta_grid_search_SP_INL_*"))
        if os.path.isdir(d)
    )
    if not subcarpetas:
        print("No se encontraron subcarpetas 'beta_grid_search_SP_INL_*' bajo", HERE)
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    # agrupar por INL nominal (tag del nombre de carpeta)
    grupos = {}
    for sc in subcarpetas:
        nombre = os.path.basename(sc)
        m = NOMBRE_RE.search(nombre)
        inl_tag = m.group("inl") if m else "desconocido"
        curva = cargar_curva(sc)
        if curva is None:
            print(f"  (sin resultados_*.npz todavia en {nombre}, se omite)")
            continue
        grupos.setdefault(inl_tag, []).append(curva)

    if not grupos:
        print("Ninguna subcarpeta tiene resultados_*.npz cargables todavia.")
        return

    resumen_optimos = []

    for inl_tag, curvas in sorted(grupos.items()):
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=140)

        for i, curva in enumerate(curvas):
            filas = curva["filas"]
            meta = curva["meta"]
            color = PALETA[i % len(PALETA)]

            betas = np.array([f["beta"] for f in filas])
            medias = np.array([f["media"] for f in filas])
            minimos = np.array([f["min"] for f in filas])
            maximos = np.array([f["max"] for f in filas])

            p_pot = meta.get("pulsos_pot")
            a_pot = meta.get("a_pot")
            inl_pot = meta.get("inl_pot")
            etiqueta = f"p={p_pot}, a={a_pot:.2f}" if p_pot is not None else curva["nombre"]
            if inl_pot is not None:
                etiqueta += f" (INL={inl_pot:.4f})"

            ax.vlines(betas, minimos, maximos, color=color, alpha=0.15, linewidth=3, zorder=1)
            ax.plot(betas, medias, color=color, marker="o", markersize=5,
                    markerfacecolor=color, markeredgecolor="white", markeredgewidth=1.0,
                    linewidth=1.8, zorder=3, label=etiqueta)

            # marcar corridas incompletas (series_completadas < series_totales)
            series_totales = meta.get("series_totales")
            if series_totales:
                incompletos = [f for f in filas if f["series_completadas"] < series_totales]
                if incompletos:
                    ax.scatter([f["beta"] for f in incompletos], [f["media"] for f in incompletos],
                               s=90, facecolors="none", edgecolors=color, linewidths=1.2,
                               linestyles=(0, (2, 2)), zorder=4)

            # optimo de esta curva
            idx_pico = int(np.argmax(medias))
            beta_pico = betas[idx_pico]
            acc_pico = medias[idx_pico]
            ax.annotate(f"{acc_pico*100:.2f}%\nbeta={beta_pico:.0f}",
                        (beta_pico, acc_pico), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8, color=color,
                        fontweight="bold", linespacing=1.2)
            ax.axvline(beta_pico, color=color, linestyle=":", linewidth=1, alpha=0.5, zorder=0)

            resumen_optimos.append({
                "inl_tag": inl_tag,
                "carpeta": curva["nombre"],
                "pulsos_pot": p_pot,
                "a_pot": a_pot,
                "inl_pot_real": inl_pot,
                "beta_optimo": beta_pico,
                "acc_optima": acc_pico,
            })

        ax.set_xlabel("beta")
        ax.set_ylabel("accuracy final (media de todas las series×folds disponibles)")
        ax.set_title(f"Accuracy vs beta — curvas con INL nominal = {inl_tag}\n"
                      "lineas punteadas verticales = beta optimo de cada curva")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v*100:.1f}%")
        ax.grid(axis="y", color="#e1e0d9", linewidth=1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8.5, loc="best")

        fig.tight_layout()
        out_png = os.path.join(OUT_DIR, f"accuracy_vs_beta_INL_{inl_tag}.png")
        fig.savefig(out_png)
        plt.close(fig)
        print(f"Guardado: {out_png}  ({len(curvas)} curva/s)")

    # tabla resumen de optimos, para comparar numericamente entre curvas del mismo INL
    csv_path = os.path.join(OUT_DIR, "resumen_optimos.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "inl_tag", "carpeta", "pulsos_pot", "a_pot", "inl_pot_real",
            "beta_optimo", "acc_optima"])
        w.writeheader()
        w.writerows(resumen_optimos)
    print(f"Guardado: {csv_path}")

    print("\nResumen de beta optimo por curva (agrupado por INL nominal):")
    print(f"{'INL':>8} {'carpeta':<45} {'pulsos':>7} {'a':>9} {'INL real':>10} "
          f"{'beta*':>7} {'acc*':>8}")
    for r in sorted(resumen_optimos, key=lambda r: (r["inl_tag"], r["pulsos_pot"] or 0)):
        print(f"{r['inl_tag']:>8} {r['carpeta']:<45} {str(r['pulsos_pot']):>7} "
              f"{r['a_pot']:>9.2f} {r['inl_pot_real']:>10.5f} "
              f"{r['beta_optimo']:>7.0f} {r['acc_optima']*100:>7.2f}%")


if __name__ == "__main__":
    main()
