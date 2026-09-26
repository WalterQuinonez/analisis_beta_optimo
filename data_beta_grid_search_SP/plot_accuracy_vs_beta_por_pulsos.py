"""
Version "por pulsos" de plot_accuracy_vs_beta_por_INL.py: en vez de agrupar
las subcarpetas por INL nominal (una figura por INL, con las distintas
cantidades de pulsos superpuestas), las agrupa por cantidad de pulsos (una
figura por p, con las distintas curvas de INL superpuestas). Sirve para
comparar, a igual cantidad de pulsos, como cambia la curva accuracy vs beta
(y el beta optimo) segun el INL.

Cada subcarpeta se identifica por su nombre:
    beta_grid_search_SP_INL_<inl_tag>_p_<pulsos>[...]
Los valores reales (pulsos_pot, a_pot, inl_pot, inl_dep) se leen de un
config_*.json de esa subcarpeta.

Para cada beta dentro de una subcarpeta, la accuracy usada es la media de
la ULTIMA epoca guardada, sobre todas las series x folds ya completados
(igual criterio que plot_accuracy_vs_beta.py / plot_accuracy_vs_beta_por_INL.py).

Uso:
    python plot_accuracy_vs_beta_por_pulsos.py
Salida:
    resumen_por_INL/accuracy_vs_beta_p_<pulsos>.png   (una por cada p)
"""

import glob
import json
import os
import re

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "resumen_por_INL")
NOMBRE_RE = re.compile(r"INL_(?P<inl>.+?)_p_(?P<p>\d+)")

# paleta fija por INL nominal, para que el mismo INL siempre tenga el mismo
# color entre figuras (mas facil de comparar a simple vista).
COLOR_POR_INL = {
    "2e-1": "#d64a6a",
    "1e-2": "#c9a227",
    "5e-2": "#e8823f",
    "5e-3": "#8bbf3f",
    "1e-3": "#3aa66b",
    "5e-4": "#3fbfa0",
    "1e-4": "#4ac9c2",
    "5e-5": "#4a8fd6",
    "1e-5": "#2a78d6",
    "4.82e-6": "#a34ad6",
}
PALETA_EXTRA = ["#eb6834", "#7a7a7a"]


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

    # agrupar por cantidad de pulsos
    grupos = {}
    for sc in subcarpetas:
        nombre = os.path.basename(sc)
        m = NOMBRE_RE.search(nombre)
        if not m:
            continue
        inl_tag = m.group("inl")
        p_tag = m.group("p")
        curva = cargar_curva(sc)
        if curva is None:
            print(f"  (sin resultados_*.npz todavia en {nombre}, se omite)")
            continue
        grupos.setdefault(p_tag, []).append((inl_tag, curva))

    if not grupos:
        print("Ninguna subcarpeta tiene resultados_*.npz cargables todavia.")
        return

    extra_idx = 0
    color_asignado = dict(COLOR_POR_INL)

    for p_tag, curvas in sorted(grupos.items(), key=lambda kv: int(kv[0])):
        # orden por INL real (de mas alto a mas bajo, como en el resto de figuras)
        curvas.sort(key=lambda ic: ic[1]["meta"].get("inl_pot") or 0, reverse=True)

        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=140)

        for inl_tag, curva in curvas:
            filas = curva["filas"]
            meta = curva["meta"]
            color = color_asignado.get(inl_tag)
            if color is None:
                color = PALETA_EXTRA[extra_idx % len(PALETA_EXTRA)]
                color_asignado[inl_tag] = color
                extra_idx += 1

            betas = np.array([f["beta"] for f in filas])
            medias = np.array([f["media"] for f in filas])
            minimos = np.array([f["min"] for f in filas])
            maximos = np.array([f["max"] for f in filas])

            inl_pot = meta.get("inl_pot")
            etiqueta = f"INL={inl_pot:.5f}" if inl_pot is not None else f"INL={inl_tag}"

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

            # optimo de esta curva (sin texto flotante: con 6 curvas juntas se
            # superponen; el valor exacto queda en resumen_optimos.csv)
            idx_pico = int(np.argmax(medias))
            beta_pico = betas[idx_pico]
            ax.axvline(beta_pico, color=color, linestyle=":", linewidth=1, alpha=0.5, zorder=0)

        ax.set_xlabel("beta")
        ax.set_ylabel("accuracy final (media de todas las series×folds disponibles)")
        ax.set_title(f"Accuracy vs beta — curvas con p = {p_tag}\n"
                      "lineas punteadas verticales = beta optimo de cada curva")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v*100:.1f}%")
        ax.grid(axis="y", color="#e1e0d9", linewidth=1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8.5, loc="best")

        fig.tight_layout()
        out_png = os.path.join(OUT_DIR, f"accuracy_vs_beta_p_{p_tag}.png")
        fig.savefig(out_png)
        plt.close(fig)
        print(f"Guardado: {out_png}  ({len(curvas)} curva/s)")


if __name__ == "__main__":
    main()
