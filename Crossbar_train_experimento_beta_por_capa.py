#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 17:33:07 2026
Adaptado el 29/08/2026 para guardar resultados de forma compacta.
Adaptado el 30/08/2026 para guardar (checkpoint) cada vez que termina una
serie, de forma que un corte por tiempo de cola del cluster o por falta de
memoria no haga perder las series ya completadas.
Adaptado el 01/09/2026 para admitir un beta DISTINTO POR CAPA (usa
MemCrossbarClass_beta_por_capa.py). `beta` puede ser:
  - un escalar: se replica en todas las capas (mismo comportamiento de
    siempre).
  - una lista/tupla de largo len(sizes) - 1: un beta por capa, para estudiar
    el impacto de tener betas distintos por capa en el entrenamiento.
Adaptado el 01/09/2026 (2) para agregar un GRID SEARCH manual de beta:
`beta_grid` es una lista de valores de beta (cada uno a su vez un escalar o
una lista por capa) con los que se corre un entrenamiento COMPLETO e
independiente por cada valor. Cada corrida de la grilla guarda sus
resultados en sus PROPIOS archivos .npz/.json/G_history (nunca se junta todo
en un unico archivo grande), para poder comparar el impacto de cada beta.
Adaptado el 01/09/2026 (3): cada SERIE usa ahora una particion de folds
DISTINTA (no siempre la misma), para poder ver si el beta optimo es sensible
a que datos cayeron en train/validacion y no solo a la estocasticidad de
entrenamiento (init de G, orden de batches, cuantizacion de pulsos). Las
`series` particiones se generan UNA sola vez (con semillas fijas y
reproducibles, derivadas de `seed`) y se REUSAN igual para todos los betas de
la grilla, asi la comparacion entre betas para una serie dada sigue siendo
justa (todos ven los mismos datos) y lo unico que varia entre series es la
particion del dataset.

Adaptado el 02/09/2026 para poder correr como TAREA de un SLURM job array
(un solo `sbatch train_job.sh` lanza N tareas independientes, una por beta
de una lista -ver train_job.sh-). Se agregaron argumentos de linea de
comando opcionales:
  --beta               un beta (o una lista por capa separada por comas,
                        ej. "50,150") para ESTA tarea. Si se pasa, se corre
                        SOLO ese beta (se ignora `beta_grid` de mas abajo).
  --beta-idx           indice de esta corrida dentro de la grilla completa
                        (para nombrar archivos igual que antes); pensado
                        para pasar $SLURM_ARRAY_TASK_ID.
  --experiment-id-base prefijo compartido por TODAS las tareas de la misma
                        grilla, para que sus archivos de salida queden
                        agrupados con el mismo prefijo aunque cada tarea
                        arranque en un proceso/nodo distinto.
  --n-betas-totales    tamano total de la grilla (solo informativo, para el
                        manifest de esta tarea).
Si no se pasa --beta (uso manual, sin SLURM), el script se comporta EXACTO
que antes: corre `beta_grid` (o `beta`, si beta_grid es None) de forma
secuencial en un unico proceso.

NOTA (01/09/2026): la version que ejecuta la grilla de beta EN PARALELO
(varios procesos, GPU/CPU) esta en un archivo aparte,
`Crossbar_train_experimento_beta_por_capa_paralelo.py`, para no complicar
esta version secuencial de referencia. Para correr la grilla en paralelo
EN EL CLUSTER (un job array de SLURM, cada beta en su propia tarea/nodo) no
hace falta ese archivo aparte: alcanza con este script + `--beta` (ver
train_job.sh), que es mas simple y deja el checkpoint por serie de cada
tarea funcionando igual que en la version secuencial.

@author: walter

Copia de Crossbar_train.py que:
  1) Junta el resultado de CADA CORRIDA (una por beta de la grilla) en un
     UNICO archivo comprimido (.npz), en lugar de un .npy por cada
     combinacion de fold/epoca.
  2) Actualiza ese .npz (y el .json de configuracion/progreso) CADA VEZ QUE
     TERMINA UNA SERIE -no solo al final de la corrida-, escribiendo primero
     a un archivo temporal y reemplazando de forma atomica (os.replace) para
     no dejar nunca un archivo a medio escribir si el proceso se corta justo
     durante el guardado.
  3) IMPORTANTE (RAM): los estados G de la red (lo mas pesado, D_in x 2*D_out
     valores por "foto") NO se acumulan en un unico arreglo para todas las
     series -eso haria crecer el uso de RAM linealmente con `series` sin
     limite, aunque se guarde seguido a disco-. En cambio, se arma un arreglo
     nuevo SOLO para la serie que se esta corriendo, se escribe a su propio
     archivo (G_history_..._serieNNN.npz) apenas esa serie termina, y esa
     memoria se libera antes de arrancar la serie siguiente. Asi el pico de
     RAM que usa el guardado de G queda acotado por UNA serie, sin importar
     cuantas series (ni cuantos betas de la grilla) tenga el experimento en
     total.
  4) Al terminar cada corrida (o si se corta y se la vuelve a mirar a mitad
     de camino), el .json de esa corrida siempre tiene ademas todos los
     parametros necesarios para repetirla (incluye la semilla aleatoria, el
     beta usado y la lista de archivos de historial de G escritos hasta ese
     momento).
  5) Ademas se mantiene un .json "manifest" de la grilla completa (uno por
     todo el experimento, liviano) que lista, para cada beta de la grilla,
     su archivo de resultados/configuracion y si esa corrida ya termino -
     util para ver de un vistazo el progreso de la grilla sin abrir cada
     corrida.

No se modifica MemCrossbarClass_beta_por_capa.py aca: se usan solo las
clases/funciones ya provistas (generar_curvas_pot_dep, MemDNN, DataSetLoader).
"""

import argparse
import json
import os
import platform
import time

import numpy as np
import torch
import torch.nn as nn

from MemCrossbarClass_beta_por_capa import MemDNN, generar_curvas_pot_dep, DataSetLoader

# =============================================================================
# 0) SEMILLA (para poder repetir el experimento con el .json guardado)
# =============================================================================
seed = 35
torch.manual_seed(seed)
np.random.seed(seed)

# =============================================================================
# 1) PARAMETROS DEL EXPERIMENTO (identicos en espiritu a Crossbar_train.py)
# =============================================================================
experimento = "beta_grid_search"
sistema = 'SP'
fecha = '1_09_2026'

# parametros de las curvas P/D
a_pot = 197.99
a_dep = 197.99
G0_distrbtn = 'random'
fixed = False
optimize_beta = False
pulsos_pot = 200
pulsos_dep = 200
concavidad_pot = 'pos'
concavidad_dep = 'neg'
Rhigh = 10000
Rlow = 1000
Gmin = 1 / Rhigh
Gmax = 1 / Rlow
device = torch.device("cpu")

# entrenamiento
sizes = [784,  10]
series = 20
epochs = 100
lr = 1
k_folds = 5
batch_number = 32
criterion = nn.CrossEntropyLoss()
n_layers = len(sizes) - 1

# protocolo electrico
# beta: escalar (mismo valor para todas las capas) o lista con un valor por
# capa, de largo len(sizes) - 1. Solo se usa si `beta_grid` (mas abajo) queda
# en None, o como base para la busqueda automatica si optimize_beta=True.
beta = 260
beta_max = 300
points_beta = 2
delta_t_forward = 10e-9
delta_t_pulse = 10e-9
amplitud_imagen = 1
Vs = 1
Vr = -1

# -----------------------------------------------------------------------
# GRID SEARCH MANUAL DE BETA
# -----------------------------------------------------------------------
# Lista de valores de beta a explorar, cada uno con un entrenamiento
# COMPLETO e independiente (series x folds x epochs). Cada elemento de la
# lista puede ser:
#   - un escalar: mismo beta para todas las capas en esa corrida
#   - una lista/tupla de largo n_layers (= len(sizes) - 1): un beta por capa
#     en esa corrida
#
# Ejemplo con sizes = [784, 128, 10] (n_layers = 2):
#   beta_grid = [50, 100, 150, [50, 150], [150, 50]]
#
# Si se deja en None, se corre UNA SOLA corrida usando `beta` (o el beta
# encontrado por optimize_beta si optimize_beta=True), igual que antes de
# agregar el grid search.
beta_grid = None

# ruido (no implementado 29/01/2026)
porcentaje_maximo = 0
porcentaje_minimo = 0

# cada cuantas epocas se guarda una "foto" completa de G dentro de una serie
# (1 = todas, como hacia el codigo original). Siempre se guarda ademas la
# ultima epoca de cada serie.
save_G_every = 1

dataset_X = 'X_train_mnist.npy'
dataset_y = 'y_train_mnist.npy'


# =============================================================================
# 1.5) ARGUMENTOS DE LINEA DE COMANDO (uso: una tarea de un SLURM job array)
# =============================================================================
def _parse_beta_cli(texto):
    """'260' -> 260.0 ; '50,150' -> [50.0, 150.0] (beta por capa)."""
    partes = texto.split(",")
    if len(partes) == 1:
        return float(partes[0])
    return [float(p) for p in partes]


_parser = argparse.ArgumentParser(
    description="Entrena la crossbar para uno o varios valores de beta. "
                "Sin argumentos corre beta_grid (o beta) de forma secuencial, "
                "igual que antes; con --beta corre SOLO ese beta -pensado "
                "para una tarea de un SLURM job array, ver train_job.sh-.")
_parser.add_argument(
    "--beta", type=_parse_beta_cli, default=None,
    help="Beta de esta corrida: un numero, o una lista separada por comas "
         "(un valor por capa, ej. '50,150'). Si se pasa, reemplaza a "
         "beta_grid y se corre unicamente este beta.")
_parser.add_argument(
    "--beta-idx", type=int, default=0,
    help="Indice de esta corrida dentro de la grilla completa (para nombrar "
         "archivos de forma consistente). Pasar $SLURM_ARRAY_TASK_ID.")
_parser.add_argument(
    "--experiment-id-base", type=str, default=None,
    help="Prefijo compartido por todas las tareas de la misma grilla, para "
         "que sus archivos de salida queden agrupados aunque cada una "
         "corra en un proceso/nodo distinto. Si no se pasa, se genera con "
         "la fecha/hora actual del proceso (uso manual/secuencial).")
_parser.add_argument(
    "--n-betas-totales", type=int, default=None,
    help="Tamano total de la grilla (solo informativo, para el manifest de "
         "esta tarea). Pasar len(BETAS) del job array.")
cli_args, _cli_unknown = _parser.parse_known_args()

if cli_args.beta is not None:
    # Modo "tarea de job array": se corre UN SOLO beta en este proceso.
    beta_grid = [cli_args.beta]
    beta_grid_start_idx = cli_args.beta_idx
    n_betas_totales_manifest = cli_args.n_betas_totales or 1
else:
    # Modo original: secuencial, un proceso corre todo beta_grid (o `beta`
    # si beta_grid quedo en None, ver mas abajo).
    beta_grid_start_idx = 0
    n_betas_totales_manifest = None  # se completa con len(beta_grid) mas abajo


# =============================================================================
# 2) VALIDACION Y HELPERS PARA BETA (escalar o lista por capa)
# =============================================================================
def _validar_beta(beta_value, n_layers):
    """Verifica que, si beta_value es una lista/tupla, tenga largo n_layers.
    Un escalar siempre es valido (se replica en todas las capas)."""
    if isinstance(beta_value, (list, tuple, np.ndarray)):
        assert len(beta_value) == n_layers, (
            f"beta {beta_value} tiene {len(beta_value)} elementos pero "
            f"sizes define {n_layers} capas; debe ser un escalar o una "
            f"lista/tupla de largo {n_layers}."
        )


def _beta_tag(beta_value):
    """Genera un tag corto y valido para nombres de archivo/carpeta a partir
    de un beta (escalar o lista por capa). Ej: 148.015 -> 'b148p015',
    [50, 150] -> 'b50-150'."""
    def _fmt(b):
        s = f"{float(b):.4g}"
        return s.replace(".", "p").replace("-", "neg")

    if isinstance(beta_value, (list, tuple, np.ndarray)):
        return "b" + "-".join(_fmt(b) for b in beta_value)
    return "b" + _fmt(beta_value)


def _beta_jsonable(beta_value):
    """Convierte beta (escalar o lista/tupla/np.ndarray) a algo serializable
    por json.dump tal cual (float o lista de floats)."""
    if isinstance(beta_value, (list, tuple, np.ndarray)):
        return [float(b) for b in beta_value]
    return float(beta_value)


_validar_beta(beta, n_layers)
if beta_grid is not None:
    for b in beta_grid:
        _validar_beta(b, n_layers)

# =============================================================================
# 3) CARPETA DEL EXPERIMENTO (compartida por todas las corridas de la grilla)
# =============================================================================
folder_resultados = (f"{experimento}_{sistema}")
os.makedirs(folder_resultados, exist_ok=True)

run_timestamp = time.strftime("%Y%m%d_%H%M%S")
experiment_id_base = cli_args.experiment_id_base or f"{sistema}_{fecha}_{run_timestamp}"

# En modo "tarea de job array" puede haber varias tareas de la MISMA grilla
# (mismo experiment_id_base) escribiendo en paralelo, cada una en su propio
# proceso/nodo: si todas escribieran el mismo manifest.json se pisarian entre
# si (condicion de carrera). Por eso cada tarea escribe su PROPIO manifest,
# con su indice de grilla en el nombre; no hace falta coordinarlas porque el
# progreso real de cada corrida ya vive en su config_*.json/resultados_*.npz
# (con nombres unicos por beta, sin colision).
if cli_args.beta is not None:
    path_json_grid_manifest = os.path.join(
        folder_resultados,
        f"beta_grid_manifest_{experiment_id_base}_tarea{cli_args.beta_idx:02d}.json")
else:
    path_json_grid_manifest = os.path.join(
        folder_resultados, f"beta_grid_manifest_{experiment_id_base}.json")

# =============================================================================
# 4) CURVAS P/D (compartidas por todas las corridas de la grilla)
# =============================================================================
pot, dep, ratio, inl_pot, inl_dep = generar_curvas_pot_dep(
    pulsos_pot, pulsos_dep, a_pot, a_dep, Gmin, Gmax, concavidad_dep, concavidad_pot)

pot = torch.tensor(pot, dtype=torch.float32)
dep = torch.tensor(dep, dtype=torch.float32)

# =============================================================================
# 5) DATASET Y FOLDS (compartidos por todas las corridas de la grilla)
# =============================================================================
X_train = np.load(dataset_X) * amplitud_imagen
y_train = np.load(dataset_y)
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).long()

# Una particion de folds DISTINTA por serie (semillas fijas y reproducibles,
# derivadas de `seed`), generada UNA sola vez y reusada tal cual para cada
# beta de la grilla. Asi:
#   - para una serie fija, todos los betas entrenan/validan sobre exactamente
#     los mismos datos -> comparar accuracy entre betas sigue siendo justo.
#   - entre series, lo unico que cambia es que datos cayeron en train vs.
#     validacion -> permite ver si el accuracy (o el mejor beta) es sensible
#     a la particion del dataset, y no solo a la estocasticidad del
#     entrenamiento (init de G, orden de batches, cuantizacion de pulsos).
fold_seeds_por_serie = [seed + s for s in range(series)]
folds_por_serie = [
    DataSetLoader(X_train, y_train, k_folds, batch_number, random_state=fold_seed)
    for fold_seed in fold_seeds_por_serie
]

# =============================================================================
# 6) BUSQUEDA AUTOMATICA DE BETA (opcional, identica a Crossbar_train.py)
# =============================================================================
if optimize_beta:
    model = MemDNN(
        sizes=sizes, beta=beta, pot=pot, dep=dep,
        G0_distribution=G0_distrbtn, fixed=fixed, device=device
    )

    # Usa la particion de folds de la primera serie (fold_seeds_por_serie[0]);
    # esta busqueda automatica es solo una estimacion previa, no forma parte
    # del grid search en si.
    train_loaders_search, val_loaders_search = folds_por_serie[0]
    best_beta_search, best_acc_search = model.search_best_beta(
        train_loaders_search, val_loaders_search, criterion, a_pot, lr,
        delta_t_forward, delta_t_pulse, Vr, Vs,
        beta_max=beta_max, points=points_beta
    )

    print("\n===================================")
    print(f"Mejor beta encontrado (busqueda automatica) = {best_beta_search}")
    print(f"Accuracy promedio CV                        = {best_acc_search}")
    print("===================================\n")
else:
    best_beta_search, best_acc_search = beta, 0

# Si no se paso una grilla explicita, se corre una unica corrida (mismo
# comportamiento que antes de agregar el grid search).
if beta_grid is None:
    beta_grid = [best_beta_search]

# =============================================================================
# 7) FUNCIONES DE GUARDADO COMPACTO Y ATOMICO
# =============================================================================
def snapshot_G(model):
    """Copia en CPU/numpy del estado actual de cada capa G_layers[i]."""
    return [G.detach().cpu().numpy().copy() for G in model.G_layers]


def _atomic_write_npz(path_npz, **arrays):
    """Escribe a un archivo temporal y recien al final lo renombra al nombre
    definitivo (os.replace es atomico). Asi, si el proceso se corta justo
    mientras se esta guardando, el .npz anterior (del checkpoint previo)
    queda intacto en vez de quedar a medio escribir/corrupto.

    OJO: np.savez_compressed le agrega ".npz" al nombre de archivo si el
    string que recibe no termina en ".npz" (asi que pasarle "algo.npz.tmp"
    terminaria escribiendo "algo.npz.tmp.npz"). Para evitarlo, se le pasa un
    file handle ya abierto en vez de un string -ahi numpy escribe tal cual,
    sin tocar el nombre."""
    tmp_path = path_npz + ".tmp"
    with open(tmp_path, "wb") as f:
        np.savez_compressed(f, **arrays)
    _replace_with_retry(tmp_path, path_npz)


def _atomic_write_json(path_json, obj):
    tmp_path = path_json + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    _replace_with_retry(tmp_path, path_json)


def _replace_with_retry(tmp_path, final_path, intentos=5, espera_s=0.3):
    """os.replace() es atomico tanto en Linux (donde va a correr esto en el
    cluster) como en Windows, pero en Windows puede fallar con
    PermissionError de forma transitoria si algo (un antivirus, un script
    que esta mirando el progreso, etc.) tiene el archivo destino abierto en
    ese instante. Se reintenta un par de veces antes de darse por vencido."""
    for intento in range(intentos):
        try:
            os.replace(tmp_path, final_path)
            return
        except PermissionError:
            if intento == intentos - 1:
                raise
            time.sleep(espera_s)


def save_experiment_results(path_npz, series, acc_result, train_loss_result, val_loss_result,
                             energy_forward_result, energy_pulse_result,
                             saved_epochs, beta_value, series_completadas, g_history_files):
    """
    Guarda las metricas (accuracy/perdida/energia) disponibles HASTA AHORA de
    UNA corrida (un beta de la grilla) en un unico archivo .npz comprimido
    -liviano: son a lo sumo un puñado de arreglos de `series x k_folds x
    epochs` floats, series de cientos no representan un problema de RAM ni
    de tamaño en disco-. Se llama una vez por serie completada; cada llamada
    REEMPLAZA -de forma atomica- el archivo anterior con la version mas
    actualizada.

    Los estados G (lo pesado) NO estan aca: viven en archivos aparte, uno por
    serie (ver save_G_history_serie), y este .npz solo guarda la LISTA de
    esos archivos (g_history_files) para poder encontrarlos.

    Formas de los arrays guardados (dimension 0 = serie, solo se guardan las
    series ya completadas, es decir arr.shape[0] == series_completadas):
      acc_result, train_loss_result, val_loss_result,
      energy_forward_result, energy_pulse_result : (series_completadas, k_folds, epochs)
      saved_epochs                                : (len(saved_epochs),)
                                                       numero de epoca (1-indexado)
                                                       de cada "foto" de G guardada
    """
    save_dict = dict(
        acc_result=acc_result[:series_completadas],
        train_loss_result=train_loss_result[:series_completadas],
        val_loss_result=val_loss_result[:series_completadas],
        energy_forward_result=energy_forward_result[:series_completadas],
        energy_pulse_result=energy_pulse_result[:series_completadas],
        saved_epochs=np.array(saved_epochs),
        beta=np.array(_beta_jsonable(beta_value)),
        series_completadas=np.array(series_completadas),
        g_history_files=np.array(g_history_files),
    )
    _atomic_write_npz(path_npz, **save_dict)
    print(f"[checkpoint] Resultados actualizados en: {path_npz} "
          f"({series_completadas}/{series} series completas)")


def save_G_history_serie(folder_resultados, experiment_id, serie, saved_epochs,
                          G_history_this_serie):
    """
    Guarda el historial de G de UNA SOLA serie (de UNA corrida/beta) en su
    propio archivo .npz comprimido: (k_folds, len(saved_epochs), D_in,
    2*D_out) por capa. Se escribe una unica vez, apenas esa serie termina, y
    nunca se vuelve a reescribir -a diferencia del checkpoint de metricas,
    que se reemplaza en cada serie-, por eso no hace falta escritura atomica
    con reintentos (pero se usa igual, es barato y no duele).
    """
    filename = f"G_history_{experiment_id}_serie{serie:03d}.npz"
    path = os.path.join(folder_resultados, filename)
    save_dict = {f"G_layer{i}_history": arr for i, arr in enumerate(G_history_this_serie)}
    save_dict["saved_epochs"] = np.array(saved_epochs)
    save_dict["serie"] = np.array(serie)
    _atomic_write_npz(path, **save_dict)
    print(f"[checkpoint] Historial de G de la serie {serie} guardado en: {path}")
    return filename


def save_experiment_config(path_json, progreso, **kwargs):
    """Escribe/actualiza el .json de UNA corrida con todos los parametros
    necesarios para repetirla (curvas P/D, arquitectura, protocolo
    electrico, entrenamiento, semilla, dataset, el beta de esa corrida y
    archivos de salida), mas un bloque 'progreso' que se actualiza en cada
    checkpoint (util para ver, sin cargar el .npz, hasta donde llego una
    corrida cortada por el cluster)."""
    obj = dict(kwargs)
    obj["progreso"] = progreso
    _atomic_write_json(path_json, obj)
    print(f"[checkpoint] Configuracion/progreso actualizados en: {path_json}")


def save_grid_manifest(path_json_grid_manifest, grid_progress):
    """Escribe/actualiza el .json manifest de TODA la grilla: lista, para
    cada beta de `beta_grid`, su archivo de resultados/configuracion y el
    estado de esa corrida. Se reescribe (atomico) cada vez que una corrida
    de la grilla termina, asi que sirve para ver de un vistazo el progreso
    global sin tener que abrir cada corrida."""
    obj = dict(
        experiment_id_base=experiment_id_base,
        timestamp_inicio=run_timestamp,
        seed=seed,
        # en modo tarea de job array esto es el tamano de la grilla COMPLETA
        # (pasado por --n-betas-totales), no solo el beta de esta tarea;
        # si no se paso, se informa el tamano de lo que corre ESTE proceso.
        n_betas_totales=n_betas_totales_manifest or len(beta_grid),
        # grid_progress es un dict {indice_de_grilla: {...}} (para poder
        # actualizar por indice real aunque este proceso solo corra un
        # subconjunto de la grilla, ver seccion 8); se serializa como lista
        # ordenada por indice para mantener el mismo formato de manifest
        # que antes (un array de corridas).
        corridas=[grid_progress[idx] for idx in sorted(grid_progress)],
        ultima_actualizacion=time.strftime("%Y%m%d_%H%M%S"),
    )
    _atomic_write_json(path_json_grid_manifest, obj)
    print(f"[checkpoint] Manifest de la grilla actualizado en: {path_json_grid_manifest}")


# Parametros fijos del experimento que NO dependen del beta de cada corrida
# (arquitectura, curvas P/D, dataset, protocolo electrico, etc.): se arman
# una sola vez aca para no repetirlos en cada corrida/checkpoint.
config_fija_grilla = dict(
    seed=seed,
    sistema=sistema,
    fecha=fecha,
    dataset={"X": dataset_X, "y": dataset_y, "amplitud_imagen": amplitud_imagen},
    arquitectura={"sizes": sizes, "fixed": fixed, "G0_distribution": G0_distrbtn,
                  "device": str(device)},
    curva_pd={"pulsos_pot": pulsos_pot, "pulsos_dep": pulsos_dep,
              "a_pot": a_pot, "a_dep": a_dep,
              "concavidad_pot": concavidad_pot, "concavidad_dep": concavidad_dep,
              "Rlow": Rlow, "Rhigh": Rhigh, "Gmin": Gmin, "Gmax": Gmax,
              "ratio": ratio, "inl_pot": inl_pot, "inl_dep": inl_dep},
    entrenamiento={"series": series, "epochs": epochs, "k_folds": k_folds,
                   "batch_number": batch_number, "lr": lr,
                   "criterion": criterion.__class__.__name__},
    folds={"modo": "particion_distinta_por_serie_reusada_entre_betas",
           "fold_seeds_por_serie": fold_seeds_por_serie},
    protocolo_electrico={"delta_t_forward": delta_t_forward,
                         "delta_t_pulse": delta_t_pulse, "Vr": Vr, "Vs": Vs},
    ruido={"porcentaje_maximo": porcentaje_maximo,
           "porcentaje_minimo": porcentaje_minimo},
    guardado={"save_G_every": save_G_every},
    busqueda_beta_automatica={"optimize_beta": optimize_beta,
                              "beta_inicial": _beta_jsonable(beta),
                              "beta_max": beta_max, "points_beta": points_beta,
                              "best_beta_search": _beta_jsonable(best_beta_search),
                              "best_acc_search": best_acc_search},
    software={"python": platform.python_version(), "torch": torch.__version__,
              "numpy": np.__version__, "platform": platform.platform()},
)

# formas de G por capa, calculadas de `sizes` (sin necesidad de instanciar un
# modelo solo para esto): capa i va de sizes[i] a sizes[i+1] neuronas, y
# MemDNN.G0_initialization arma G con forma (D_in, 2*D_out).
G_shapes = [(sizes[i], 2 * sizes[i + 1]) for i in range(len(sizes) - 1)]

# indices de epoca (1-indexados) que se van a "fotografiar" por completo
saved_epochs = [e for e in range(1, epochs + 1)
                if (e % save_G_every == 0) or (e == epochs)]
n_saved = len(saved_epochs)

# =============================================================================
# 8) GRID SEARCH: un entrenamiento COMPLETO e independiente por cada beta,
#    cada uno con sus propios archivos .npz/.json/G_history (nunca se junta
#    todo en un unico archivo grande).
# =============================================================================
# dict {indice_de_grilla: {...}}, no lista: en modo tarea de job array
# `beta_idx` (mas abajo) es el indice REAL dentro de la grilla completa
# (p.ej. 7), no la posicion dentro del beta_grid de ESTE proceso (que tiene
# largo 1) -con una lista, grid_progress[beta_idx] se saldria de rango.
grid_progress = {
    idx: {"indice": idx, "beta": _beta_jsonable(b), "beta_tag": _beta_tag(b),
          "estado": "pendiente", "experiment_id": None,
          "archivo_resultados": None, "archivo_configuracion": None}
    for idx, b in enumerate(beta_grid, start=beta_grid_start_idx)
}
save_grid_manifest(path_json_grid_manifest, grid_progress)

grid_t0 = time.time()

# start=beta_grid_start_idx: en modo secuencial es 0 (idem comportamiento
# anterior); en modo tarea de job array es $SLURM_ARRAY_TASK_ID, asi el
# nombre de archivo de ESTA tarea (beta{beta_idx:02d}_...) coincide con su
# indice real dentro de la grilla completa, aunque este proceso solo corra
# un beta_grid de largo 1.
for beta_idx, beta_value in enumerate(beta_grid, start=beta_grid_start_idx):
    _validar_beta(beta_value, n_layers)
    beta_tag = _beta_tag(beta_value)

    # Cada corrida de la grilla tiene su propio experiment_id (y por lo
    # tanto sus propios archivos .npz/.json/G_history) dentro de la MISMA
    # carpeta de resultados.
    experiment_id = f"{experiment_id_base}_beta{beta_idx:02d}_{beta_tag}"
    path_npz = os.path.join(folder_resultados, f"resultados_{experiment_id}.npz")
    path_json = os.path.join(folder_resultados, f"config_{experiment_id}.json")

    print(f"\n########## Beta {beta_idx + 1}/{len(beta_grid)}: {beta_value} "
          f"({experiment_id}) ##########")

    grid_progress[beta_idx].update(
        estado="en_curso",
        experiment_id=experiment_id,
        archivo_resultados=os.path.basename(path_npz),
        archivo_configuracion=os.path.basename(path_json),
    )
    save_grid_manifest(path_json_grid_manifest, grid_progress)

    config_fija = dict(
        config_fija_grilla,
        experiment_id=experiment_id,
        timestamp_inicio=run_timestamp,
        beta={"valor": _beta_jsonable(beta_value), "indice_grilla": beta_idx,
              "tag": beta_tag},
        guardado=dict(
            config_fija_grilla["guardado"],
            archivo_resultados=os.path.basename(path_npz),
            archivo_configuracion=os.path.basename(path_json),
        ),
    )

    # Config inicial (progreso=0), asi que incluso si el proceso se corta
    # durante la primera serie de esta corrida, ya queda documentado en
    # disco que se estaba corriendo y con que parametros.
    save_experiment_config(
        path_json,
        progreso={"series_completadas": 0, "series_totales": series,
                  "estado": "en_curso", "ultima_actualizacion": time.strftime("%Y%m%d_%H%M%S")},
        **config_fija,
    )

    # -------------------------------------------------------------------
    # ENTRENAMIENTO (series x folds x epocas) de esta corrida, con
    # checkpoint al final de cada serie.
    # -------------------------------------------------------------------
    acc_all = torch.zeros([series, k_folds, epochs])
    train_loss_all = torch.zeros([series, k_folds, epochs])
    val_loss_all = torch.zeros([series, k_folds, epochs])
    energy_forward_all = torch.zeros([series, k_folds, epochs])
    energy_pulse_all = torch.zeros([series, k_folds, epochs])

    # lista (creciente) de archivos de historial de G ya escritos, uno por
    # serie completada de ESTA corrida.
    g_history_files = []

    run_t0 = time.time()

    for serie in range(series):
        # Particion de folds de ESTA serie (distinta por serie, ver seccion 5;
        # identica para todos los betas de la grilla en esta misma serie).
        train_loaders, val_loaders = folds_por_serie[serie]

        # Arreglo de historial de G SOLO para esta serie (no para todas las
        # series juntas): esto es lo que mantiene acotado el uso de RAM sin
        # importar cuan largo sea `series`.
        G_history_this_serie = [
            np.zeros((k_folds, n_saved, *shape), dtype=np.float32) for shape in G_shapes
        ]

        for fold in range(k_folds):
            model = MemDNN(
                sizes=sizes, beta=beta_value, pot=pot, dep=dep,
                G0_distribution=G0_distrbtn, fixed=fixed, device=device
            )

            train_loader = train_loaders[fold]
            val_loader = val_loaders[fold]

            save_idx = 0
            for e in range(1, epochs + 1):
                time_stamp = time.time()
                total_loss, total_energy_forward, total_energy_pulse = model.train_epoch(
                    train_loader, lr, criterion, delta_t_forward, delta_t_pulse, Vr, Vs)

                acc_val, loss_val = model.evaluate_model(val_loader, criterion)

                acc_all[serie, fold, e - 1] = acc_val
                train_loss_all[serie, fold, e - 1] = total_loss / batch_number
                val_loss_all[serie, fold, e - 1] = loss_val
                energy_forward_all[serie, fold, e - 1] = total_energy_forward
                energy_pulse_all[serie, fold, e - 1] = total_energy_pulse

                if e in saved_epochs:
                    for i, G in enumerate(snapshot_G(model)):
                        G_history_this_serie[i][fold, save_idx] = G
                    save_idx += 1

                print(f"Beta {beta_value} Serie {serie} Fold {fold} Epoch {e}/{epochs} "
                      f"--- Tiempo {time.time() - time_stamp:.3f}s "
                      f"--- Loss train {total_loss / batch_number:.5f} "
                      f"--- Loss val {loss_val:.5f} --- Acc {acc_val:.3f} "
                      f"--- Energy forward {total_energy_forward:.3f} "
                      f"--- Energy pulse {total_energy_pulse:.3f}")

        # ---------------------------------------------------------------
        # CHECKPOINT: la serie `serie` termino (todos sus folds y epocas ya
        # se corrieron).
        #   1) el historial de G de ESTA serie se escribe a su propio
        #      archivo (una vez, no se reescribe mas) y se libera de RAM.
        #   2) las metricas (livianas) de todas las series 0..serie de ESTA
        #      corrida se reescriben de forma atomica en su propio .npz.
        # ---------------------------------------------------------------
        series_completadas = serie + 1

        g_history_files.append(
            save_G_history_serie(folder_resultados, experiment_id, serie,
                                  saved_epochs, G_history_this_serie)
        )
        del G_history_this_serie  # liberar RAM antes de arrancar la proxima serie

        save_experiment_results(
            path_npz,
            series,
            acc_result=acc_all.numpy(),
            train_loss_result=train_loss_all.numpy(),
            val_loss_result=val_loss_all.numpy(),
            energy_forward_result=energy_forward_all.numpy(),
            energy_pulse_result=energy_pulse_all.numpy(),
            saved_epochs=saved_epochs,
            beta_value=beta_value,
            series_completadas=series_completadas,
            g_history_files=g_history_files,
        )

        acc_final_por_fold = acc_all[serie, :, -1].tolist()
        save_experiment_config(
            path_json,
            progreso={
                "series_completadas": series_completadas,
                "series_totales": series,
                "estado": "en_curso" if series_completadas < series else "completo",
                "ultima_actualizacion": time.strftime("%Y%m%d_%H%M%S"),
                "tiempo_transcurrido_s": round(time.time() - run_t0, 1),
                "acc_final_ultima_serie_por_fold": acc_final_por_fold,
                "archivos_G_history": g_history_files,
            },
            **config_fija,
        )

    # Esta corrida (este beta) termino: se actualiza el manifest de la
    # grilla completa.
    grid_progress[beta_idx].update(
        estado="completo",
        acc_final_ultima_serie_por_fold=acc_all[-1, :, -1].tolist(),
        tiempo_transcurrido_s=round(time.time() - run_t0, 1),
    )
    save_grid_manifest(path_json_grid_manifest, grid_progress)

    print(f"Corrida de beta={beta_value} finalizada.")
    print(f"  -> {path_npz}")
    print(f"  -> {path_json}")

if cli_args.beta is not None:
    print(f"\nTarea {cli_args.beta_idx} de la grilla finalizada "
          f"({time.time() - grid_t0:.1f}s).")
else:
    print("\nGrid search de beta finalizado "
          f"({len(beta_grid)} corridas, {time.time() - grid_t0:.1f}s en total).")
print(f"  -> Manifest: {path_json_grid_manifest}")
