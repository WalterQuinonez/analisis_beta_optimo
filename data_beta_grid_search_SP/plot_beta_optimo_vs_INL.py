"""
Grafica el beta optimo (el que maximiza la accuracy final) en funcion del INL,
con una curva separada para cada cantidad de pulsos (p=50, 100, 200).

Toma como entrada resumen_por_INL/resumen_optimos.csv, que genera
plot_accuracy_vs_beta_por_INL.py (correr ese script primero si la carpeta
resumen_por_INL no existe o esta desactualizada).

Uso:
    python plot_beta_optimo_vs_INL.py
Salida:
    resumen_por_INL/beta_optimo_vs_INL.png
"""

import csv
import os

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "resumen_por_INL")
CSV_PATH = os.path.join(OUT_DIR, "resumen_optimos.csv")

# mismos colores que plot_accuracy_vs_beta_por_INL.py, pero indexados por
# pulsos (no por orden de aparicion) para que sean consistentes entre
# figuras.
COLOR_POR_PULSOS = {
    50: "#2a78d6",
    100: "#eb6834",
    200: "#3aa66b",
}
PALETA_EXTRA = ["#a34ad6", "#c9a227", "#d64a6a", "#4ac9c2", "#7a7a7a"]


def main():
    if not os.path.isfile(CSV_PATH):
        print(f"No existe {CSV_PATH}. Corre primero plot_accuracy_vs_beta_por_INL.py")
        return

    filas = []
    with open(CSV_PATH, "r", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            filas.append({
                "inl_tag": r["inl_tag"],
                "pulsos_pot": int(r["pulsos_pot"]),
                "inl_pot_real": float(r["inl_pot_real"]),
                "beta_optimo": float(r["beta_optimo"]),
                "acc_optima": float(r["acc_optima"]),
            })

    if not filas:
        print(f"{CSV_PATH} esta vacio.")
        return

    # agrupar por cantidad de pulsos
    por_pulsos = {}
    for f in filas:
        por_pulsos.setdefault(f["pulsos_pot"], []).append(f)

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=140)

    extra_idx = 0
    for pulsos in sorted(por_pulsos):
        puntos = sorted(por_pulsos[pulsos], key=lambda f: f["inl_pot_real"])
        inl = np.array([p["inl_pot_real"] for p in puntos])
        beta = np.array([p["beta_optimo"] for p in puntos])

        color = COLOR_POR_PULSOS.get(pulsos)
        if color is None:
            color = PALETA_EXTRA[extra_idx % len(PALETA_EXTRA)]
            extra_idx += 1

        ax.plot(inl, beta, color=color, marker="o", markersize=6,
                markerfacecolor=color, markeredgecolor="white", markeredgewidth=1.0,
                linewidth=1.8, label=f"p = {pulsos}")

    ax.set_xscale("log")
    ax.set_xlabel("INL")
    ax.set_ylabel("beta optimo")
    ax.set_title("Beta optimo vs INL, por cantidad de pulsos")
    ax.grid(True, which="both", color="#e1e0d9", linewidth=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="best", title="pulsos")

    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "beta_optimo_vs_INL.png")
    fig.savefig(out_png)
    plt.close(fig)
    print(f"Guardado: {out_png}")


if __name__ == "__main__":
    main()
