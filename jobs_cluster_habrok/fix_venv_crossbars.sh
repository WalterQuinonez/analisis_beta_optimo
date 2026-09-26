#!/bin/bash
# =============================================================================
# fix_venv_crossbars.sh
#
# Arregla el venv $HOME/venvs/crossbars, que quedo con `bin/python3`
# enlazado al Python DEL SISTEMA OPERATIVO (/usr/bin/python3) en vez de al
# Python del modulo Python/3.9.6-GCCcore-11.2.0 (servido via CVMFS). Ese
# `/usr/bin/python3` es LOCAL a cada nodo -no es una ruta compartida como
# $HOME o /cvmfs/...-, y con la migracion de Habrok de AlmaLinux 8 a 9 en
# curso, distintos nodos de computo pueden tener un `/usr/bin/python3`
# distinto (o directamente sin el Python 3.6 del que dependia el venv
# viejo). Por eso numpy/torch "desaparecian" segun en que nodo caia la
# tarea (ver diagnostico_python_cluster_habrok.txt).
#
# Este script recrea el venv con el modulo cargado, para que quede atado a
# la ruta de CVMFS (uniforme en TODOS los nodos, migrados o no) en vez de
# al OS local de cada nodo.
#
# COMO CORRERLO:
#   1) Subi este archivo al cluster (scp / git pull / etc.)
#   2) Conectate al LOGIN NODE de Habrok (no lo sometas con sbatch: los
#      nodos de computo normalmente no tienen salida a internet, y esto
#      necesita bajar paquetes de PyPI).
#   3) chmod +x fix_venv_crossbars.sh
#   4) ./fix_venv_crossbars.sh
#
# Es seguro correrlo mas de una vez: si ya existe un backup del venv viejo
# con el mismo nombre, agrega un sufijo de fecha/hora distinto en vez de
# pisarlo.
# =============================================================================

# (sin -u/nounset: los scripts de activate/deactivate de virtualenv suelen
# referenciar variables sin default seguro; con -u esto puede abortar el
# script en un punto inesperado. Ver el mismo comentario en train_job.sh)
set -eo pipefail

VENV_DIR="$HOME/venvs/crossbars"
MODULE="Python/3.9.6-GCCcore-11.2.0"

echo "=== 1) Cargando el modulo $MODULE ==="
module purge
module load "$MODULE"

MODULE_PYTHON="$(which python3)"
echo "    python3 del modulo: $MODULE_PYTHON"
case "$MODULE_PYTHON" in
    /cvmfs/*) : ;;  # esperado
    *)
        echo "ATENCION: 'which python3' no aparenta venir de /cvmfs/..." >&2
        echo "          (dio: $MODULE_PYTHON). Revisa que el modulo" >&2
        echo "          '$MODULE' exista en este nodo antes de seguir" >&2
        echo "          (module spider $MODULE)." >&2
        exit 1
        ;;
esac

echo
echo "=== 2) Backup del venv viejo (si existe) ==="
if [ -d "$VENV_DIR" ]; then
    BACKUP_DIR="${VENV_DIR}_broken_$(date +%Y%m%d_%H%M%S)"
    echo "    Moviendo $VENV_DIR -> $BACKUP_DIR"
    mv "$VENV_DIR" "$BACKUP_DIR"
else
    echo "    No existia $VENV_DIR, no hay nada que respaldar."
fi

echo
echo "=== 3) Creando el venv nuevo con el Python del modulo ==="
python3 -m venv "$VENV_DIR"

NEW_PYTHON_LINK="$(readlink -f "$VENV_DIR/bin/python3")"
echo "    $VENV_DIR/bin/python3 -> $NEW_PYTHON_LINK"
case "$NEW_PYTHON_LINK" in
    /cvmfs/*) echo "    OK: quedo enlazado a CVMFS, no al sistema operativo." ;;
    *)
        echo "ATENCION: el venv nuevo tampoco quedo enlazado a /cvmfs/..." >&2
        echo "          (dio: $NEW_PYTHON_LINK). Algo no esta bien; no" >&2
        echo "          sigo instalando paquetes." >&2
        exit 1
        ;;
esac

echo
echo "=== 4) Activando el venv nuevo e instalando paquetes ==="
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install numpy torch

echo
echo "=== 5) Verificacion final ==="
python3 -c "
import sys, numpy, torch
print('sys.executable :', sys.executable)
print('numpy version  :', numpy.__version__)
print('torch version  :', torch.__version__)
"

deactivate || true

echo
echo "=== Listo ==="
echo "El venv $VENV_DIR quedo recreado y enlazado al modulo $MODULE."
echo "Si todo lo de arriba salio bien (sin ATENCION/ERROR), ya podes"
echo "volver a lanzar tus jobs con train_job.sh / train_job_curva2.sh."
