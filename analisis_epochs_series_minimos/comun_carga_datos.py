"""
Utilidades de carga de datos para el estudio de epochs/series minimos.

Este modulo SOLO LEE los resultados ya generados por el grid search de fuerza
bruta en `data_beta_grid_search_SP/` (no modifica ni re-corre nada). No importa
ni depende de `Crossbar_train_experimento_beta_por_capa.py` /
`MemCrossbarClass_beta_por_capa.py` (los codigos provistos por el usuario no se
tocan en ningun punto de este estudio).

Estructura esperada de cada archivo `resultados_*_beta*.npz`:
    acc_result:  float32 (series=20, folds=5, epochs=100)
    beta:        escalar (valor de beta de esa corrida)
    series_completadas: escalar (cuantas series realmente terminaron)
"""

import glob
import os
import re

import numpy as np

RAIZ_PROYECTO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_GRID = os.path.join(RAIZ_PROYECTO, "data_beta_grid_search_SP")

PATRON_COMBO = re.compile(r"beta_grid_search_SP_INL_(?P<inl>[^_]+)_p_(?P<pulsos>\d+)$")


def listar_combos(dir_grid=DIR_GRID):
    """Devuelve lista de dicts {inl, pulsos, path} para cada carpeta INLxpulsos."""
    combos = []
    for nombre in sorted(os.listdir(dir_grid)):
        ruta = os.path.join(dir_grid, nombre)
        if not os.path.isdir(ruta):
            continue
        m = PATRON_COMBO.match(nombre)
        if not m:
            continue
        combos.append({
            "inl_tag": m.group("inl"),
            "pulsos": int(m.group("pulsos")),
            "nombre": nombre,
            "path": ruta,
        })
    return combos


def cargar_combo(path_combo, series_max=None, folds_max=None):
    """
    Carga todos los archivos resultados_*_beta*.npz de una carpeta combo.

    Devuelve:
        betas: array (n_betas,) ordenado ascendente
        acc:   array (n_betas, series, folds, epochs) float32
               (recortado a series_max/folds_max si se piden submuestras;
               por defecto usa todo lo disponible = 20 series x 5 folds)
    """
    archivos = sorted(glob.glob(os.path.join(path_combo, "resultados_*_beta*.npz")))
    if not archivos:
        raise FileNotFoundError(f"No se encontraron resultados_*.npz en {path_combo}")

    registros = []
    for f in archivos:
        d = np.load(f)
        beta = float(d["beta"])
        acc = d["acc_result"]  # (series, folds, epochs)
        n_series_ok = int(d["series_completadas"]) if "series_completadas" in d else acc.shape[0]
        registros.append((beta, acc, n_series_ok))

    registros.sort(key=lambda r: r[0])
    betas = np.array([r[0] for r in registros], dtype=float)

    n_series_disponibles = min(r[2] for r in registros)
    n_series_disponibles = min(n_series_disponibles, min(r[1].shape[0] for r in registros))
    n_folds_disponibles = min(r[1].shape[1] for r in registros)
    n_epochs = min(r[1].shape[2] for r in registros)

    if series_max is not None:
        n_series_disponibles = min(n_series_disponibles, series_max)
    if folds_max is not None:
        n_folds_disponibles = min(n_folds_disponibles, folds_max)

    acc = np.stack(
        [r[1][:n_series_disponibles, :n_folds_disponibles, :n_epochs] for r in registros],
        axis=0,
    )  # (n_betas, series, folds, epochs)

    return betas, acc


def acc_mean_por_beta_epoca(acc):
    """acc: (n_betas, series, folds, epochs) -> media sobre (series,folds): (n_betas, epochs)"""
    return acc.mean(axis=(1, 2))


if __name__ == "__main__":
    combos = listar_combos()
    print(f"Combos encontrados: {len(combos)}")
    for c in combos[:3]:
        betas, acc = cargar_combo(c["path"])
        print(c["nombre"], "betas:", betas.shape, "acc:", acc.shape)
