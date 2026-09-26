#!/bin/bash
# =============================================================================
# install_deps_crossbars.sh
#
# Instala/verifica, DENTRO del venv $HOME/venvs/crossbars, todas las
# dependencias de terceros que usan Crossbar_train_experimento_beta_por_capa.py
# y MemCrossbarClass_beta_por_capa.py (revisadas a mano: torch, numpy,
# scikit-learn). Incluye matplotlib por si tambien corres plot_PDcurves.py
# en el cluster.
#
# Correr esto NO reemplaza fix_venv_crossbars.sh: este script asume que el
# venv YA esta bien enlazado al modulo Python (si todavia no lo arreglaste,
# corre primero fix_venv_crossbars.sh).
#
# COMO CORRERLO (en el LOGIN NODE, no con sbatch -necesita internet-):
#   chmod +x install_deps_crossbars.sh
#   ./install_deps_crossbars.sh
#
# Es seguro correrlo mas de una vez: pip no reinstala lo que ya esta al dia,
# y el chequeo de imports al final solo informa, no rompe nada si ya estaba
# todo instalado.
# =============================================================================

set -eo pipefail

VENV_DIR="$HOME/venvs/crossbars"
MODULE="Python/3.9.6-GCCcore-11.2.0"

# pip_package:modulo_a_importar (separados por ':')
PAQUETES=(
    "numpy:numpy"
    "torch:torch"
    "scikit-learn:sklearn"
    "matplotlib:matplotlib"
)

echo "=== 1) Cargando el modulo $MODULE y activando el venv ==="
module purge
module load "$MODULE"
source "$VENV_DIR/bin/activate"
echo "    python3 -> $(readlink -f "$VENV_DIR/bin/python3")"

echo
echo "=== 2) Instalando paquetes ==="
pip install --upgrade pip
for par in "${PAQUETES[@]}"; do
    pip_pkg="${par%%:*}"
    echo "    pip install $pip_pkg"
    pip install "$pip_pkg"
done

echo
echo "=== 3) Verificacion (import de cada paquete) ==="
FALLo=0
for par in "${PAQUETES[@]}"; do
    pip_pkg="${par%%:*}"
    mod="${par##*:}"
    if ver="$(python3 -c "import ${mod}; print(${mod}.__version__)" 2>/dev/null)"; then
        echo "    OK   $pip_pkg  (import $mod -> $ver)"
    else
        echo "    FALTA  $pip_pkg  (no se pudo 'import $mod')" >&2
        FALLo=1
    fi
done

deactivate || true

echo
if [ "$FALLo" -eq 0 ]; then
    echo "=== Listo: todas las dependencias estan disponibles en $VENV_DIR ==="
else
    echo "=== Hay paquetes que siguen fallando, revisa los mensajes de arriba ===" >&2
    exit 1
fi
