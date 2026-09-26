"""
Fase 1: cuantas epocas hacen falta para identificar el mismo beta_optimo que con
epochs=100, usando UNICAMENTE los datos ya generados por el grid search de
fuerza bruta (data_beta_grid_search_SP/). No corre ninguna simulacion nueva.

Para cada combo (INL, pulsos) y cada epoca e (1..100):
    beta_opt(e) = argmax_beta  mean_{series,folds} acc(beta, e)

Se prueban tres criterios de "cuantas epocas alcanzan", de mas estricto a mas
relevante en la practica (el landscape accuracy-vs-beta es MUY plano cerca del
optimo, asi que exigir que el indice de beta coincida exactamente es enganoso:
distintos betas vecinos dan practicamente la misma accuracy final, y el ruido
estadistico hace que el argmax salte entre ellos aunque el resultado practico
sea identico):

  1) coincidencia_indice: dist(idx_opt(e), idx_opt(100)) <= 1 paso de grilla.
  2) acc_ganador: |acc(beta_opt(e), epoca=e) - acc(beta_opt(100), epoca=100)| < tol
     (converge la accuracy del propio beta ganador a medida que entrena mas).
  3) regret (el criterio realmente relevante): si yo cortara la busqueda en la
     epoca e y me quedara con beta_opt(e), pero ese modelo se entrenara igual
     hasta la epoca 100, cuanta accuracy final pierdo respecto del optimo real?
        regret(e) = acc_opt_final - acc_mean[idx_opt(e), epoca=100]
     Este es el que importa para decidir "epochs_min de la BUSQUEDA de beta"
     (no epochs_min del entrenamiento final, que es otra cosa).

En los tres casos se busca la minima epoca e* a partir de la cual la condicion
se sostiene ininterrumpidamente hasta la epoca 100.

Salidas (en esta misma carpeta):
    epoca_minima_por_combo.csv
    epocas_minimas_vs_INL_pulsos.png
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from comun_carga_datos import listar_combos, cargar_combo, acc_mean_por_beta_epoca

DIR_SALIDA = os.path.dirname(os.path.abspath(__file__))
TOL_ACC = 0.01          # 1% de accuracy absoluto (tolerancia principal reportada)
TOLERANCIAS_SENSIBILIDAD = [0.02, 0.01, 0.005, 0.002]  # pp: para tabla de sensibilidad
SOSTENIDO_EPOCAS = 10    # la condicion debe mantenerse por N epocas seguidas hasta el final
PASO_GRILLA_TOLERADO = 1  # se admite terminar a <=1 indice de distancia del beta optimo real


def indice_mas_cercano(betas, beta_valor):
    return int(np.argmin(np.abs(betas - beta_valor)))


def primera_epoca_sostenida(condicion_por_epoca, n_epochs, sostenido=SOSTENIDO_EPOCAS):
    """
    condicion_por_epoca: array booleano (n_epochs,), True si la condicion se
    cumple en esa epoca (comparando contra el estado final).
    Devuelve la primera epoca (1-indexed) a partir de la cual condicion es True
    de forma ININTERRUMPIDA hasta la ultima epoca. Si nunca se sostiene,
    devuelve n_epochs (la ultima).
    """
    for e in range(n_epochs):
        if np.all(condicion_por_epoca[e:]):
            return e + 1  # 1-indexed
    return n_epochs


def analizar_combo(combo):
    betas, acc = cargar_combo(combo["path"])  # acc: (n_betas, series, folds, epochs)
    acc_mean = acc_mean_por_beta_epoca(acc)     # (n_betas, epochs)
    n_betas, n_epochs = acc_mean.shape

    idx_opt_final = int(np.argmax(acc_mean[:, -1]))
    beta_opt_final = betas[idx_opt_final]
    acc_opt_final = acc_mean[idx_opt_final, -1]

    idx_opt_por_epoca = np.argmax(acc_mean, axis=0)  # (epochs,), indice de beta elegido en cada e

    # --- criterio 1: coincidencia de indice de beta (estricto, referencia) ---
    dist_grilla = np.abs(idx_opt_por_epoca - idx_opt_final)
    condicion_indice = dist_grilla <= PASO_GRILLA_TOLERADO
    e_min_indice = primera_epoca_sostenida(condicion_indice, n_epochs)

    # --- criterio 2: accuracy del propio beta ganador de esa epoca ---
    acc_del_ganador_por_epoca = np.array([acc_mean[idx_opt_por_epoca[e], e] for e in range(n_epochs)])
    condicion_acc = np.abs(acc_del_ganador_por_epoca - acc_opt_final) < TOL_ACC
    e_min_acc = primera_epoca_sostenida(condicion_acc, n_epochs)

    # --- criterio 3 (el relevante): regret en accuracy FINAL (epoca 100) del
    # beta que se hubiera elegido usando solo datos hasta la epoca e ---
    acc_final_del_elegido_en_e = acc_mean[idx_opt_por_epoca, -1]  # (epochs,)
    regret_por_epoca = acc_opt_final - acc_final_del_elegido_en_e
    condicion_regret = regret_por_epoca < TOL_ACC
    e_min_regret = primera_epoca_sostenida(condicion_regret, n_epochs)

    fila = {
        "combo": combo["nombre"],
        "inl_tag": combo["inl_tag"],
        "pulsos": combo["pulsos"],
        "n_betas": n_betas,
        "beta_optimo_100ep": beta_opt_final,
        "acc_optimo_100ep": acc_opt_final,
        "epoca_min_indice_estable": e_min_indice,
        "epoca_min_acc_ganador": e_min_acc,
        "epoca_min_regret": e_min_regret,
        "regret_max_observado": float(regret_por_epoca.max()),
    }
    for tol in TOLERANCIAS_SENSIBILIDAD:
        cond = regret_por_epoca < tol
        fila[f"epoca_min_regret_tol{tol}"] = primera_epoca_sostenida(cond, n_epochs)
    return fila


def main():
    combos = listar_combos()
    filas = [analizar_combo(c) for c in combos]
    df = pd.DataFrame(filas).sort_values(["pulsos", "inl_tag"]).reset_index(drop=True)

    ruta_csv = os.path.join(DIR_SALIDA, "epoca_minima_por_combo.csv")
    df.to_csv(ruta_csv, index=False)
    print(f"Guardado {ruta_csv}")
    print(df.to_string(index=False))

    print("\n--- Resumen ---")
    for col in ["epoca_min_indice_estable", "epoca_min_acc_ganador", "epoca_min_regret"]:
        print(
            f"{col}: media={df[col].mean():.1f} "
            f"mediana={df[col].median():.1f} "
            f"p90={df[col].quantile(0.90):.1f} "
            f"p95={df[col].quantile(0.95):.1f} "
            f"max={df[col].max()}"
        )
    print(f"regret_max_observado (peor caso, todas las epocas/combos): {df['regret_max_observado'].max()*100:.3f} pp")

    print("\n--- Sensibilidad a la tolerancia de regret ---")
    for tol in TOLERANCIAS_SENSIBILIDAD:
        col = f"epoca_min_regret_tol{tol}"
        print(
            f"tol={tol*100:.2f}pp -> mediana={df[col].median():.1f} "
            f"p95={df[col].quantile(0.95):.1f} max={df[col].max()}"
        )

    epochs_min_recomendado = int(max(5, np.ceil(df["epoca_min_regret"].max())))
    print(f"\n>>> epochs_min recomendado (max observado del criterio de regret<{TOL_ACC*100:.1f}pp,")
    print(f"    con piso de seguridad de 5): {epochs_min_recomendado}")
    print("    (criterio de indice exacto de beta es demasiado estricto: el landscape")
    print("     accuracy-vs-beta es muy plano cerca del optimo, asi que el argmax salta")
    print("     entre betas vecinos con practicamente la misma accuracy; ver columna")
    print("     epoca_min_indice_estable como referencia/diagnostico, no como recomendacion)")

    # --- grafico ---
    fig, ax = plt.subplots(figsize=(9, 5.5))
    pulsos_unicos = sorted(df["pulsos"].unique())
    marcadores = {50: "o", 100: "s", 200: "^"}
    for p in pulsos_unicos:
        sub = df[df["pulsos"] == p].copy()
        # orden por INL numerico real (inl_tag es string tipo "1e-2")
        sub["inl_num"] = sub["inl_tag"].astype(float)
        sub = sub.sort_values("inl_num")
        ax.plot(
            sub["inl_num"], sub["epoca_min_regret"],
            marker=marcadores.get(p, "o"), label=f"pulsos={p}",
        )
    ax.axhline(epochs_min_recomendado, color="crimson", linestyle="--",
               label=f"recomendado (p95) = {epochs_min_recomendado}")
    ax.set_xscale("log")
    ax.set_xlabel("INL (nominal)")
    ax.set_ylabel(f"epoca minima con regret < {TOL_ACC*100:.1f}pp sostenido")
    ax.set_title("Epocas minimas para elegir un beta que no pierda accuracy final (criterio de regret)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    ruta_png = os.path.join(DIR_SALIDA, "epocas_minimas_vs_INL_pulsos.png")
    fig.savefig(ruta_png, dpi=150)
    print(f"Guardado {ruta_png}")


if __name__ == "__main__":
    main()
