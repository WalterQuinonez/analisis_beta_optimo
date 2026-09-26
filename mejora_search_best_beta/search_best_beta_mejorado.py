#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Version mejorada de `MemDNN.search_best_beta()`.

NO SE MODIFICA `Fuerza_bruta/MemCrossbarClass_beta_por_capa.py`: este modulo
importa `MemDNN` sin tocarlo y define una SUBCLASE, `MemDNNMejorado`, que
sobreescribe unicamente `search_best_beta`. Es un reemplazo "drop-in": donde
antes se usaba `MemDNN(...)`, alcanza con usar `MemDNNMejorado(...)` y el
resto del codigo (train_epoch, evaluate_model, forward, etc.) sigue siendo
exactamente el original, heredado sin cambios.

=============================================================================
ERRORES METODOLOGICOS ENCONTRADOS EN EL ORIGINAL (search_best_beta, lineas
268-370 de MemCrossbarClass_beta_por_capa.py) Y COMO SE CORRIGEN ACA:
=============================================================================

1) BUG: rama muerta sin `else`.
   El cuerpo entero esta adentro de `if a <= 10000:`. Si se llama con
   a_pot > 10000 (un valor de curva P/D perfectamente valido: `a_pot` grande
   simplemente significa curva mas lineal/menos no ideal), la funcion no
   hace nada y devuelve `None` implicito. No hay ninguna razon fisica o
   matematica documentada para ese umbral -no aparece en los papers de
   `Fuerza_bruta/papers/` ni en ningun comentario- y es facil de disparar:
   basta pedir una curva P/D muy lineal (a_pot alto) para que la busqueda de
   beta se rompa en silencio.
   FIX: se elimina la condicion; la busqueda corre siempre.

2) FLACO: solo 1 epoca de entrenamiento por candidato de beta.
   `model.train_epoch(...)` se llama una unica vez por (beta, fold) antes de
   evaluar. El estudio empirico en
   `analisis_beta_optimo/analisis_epochs_series_minimos/` (mismo repositorio,
   usa el GROUND TRUTH real de 100 epocas guardado en
   `data_beta_grid_search_SP/`, 30 combinaciones INL x pulsos) midio
   directamente el costo de elegir beta con pocas epocas: con 1 sola epoca,
   el peor caso de los 30 combos pierde hasta 1.48 puntos porcentuales de
   accuracy final respecto del beta verdaderamente optimo. Con 8 epocas, el
   regret cae por debajo de 1pp en el peor de los 30 casos (ver
   `epoca_minima_por_combo.csv`, columna `epoca_min_regret`, max=8).
   FIX: `epochs_busqueda` configurable, default 8 (en vez de 1), en base a
   ese hallazgo empirico documentado.

3) FLACO (el mas serio): no hay control de aleatoriedad entre candidatos de
   beta -> ruido de inicializacion confunde la comparacion.
   Para cada beta y cada fold se crea un `MemDNN` nuevo con
   `G0_distribution='random'`, que sortea la conductancia inicial con
   `torch.randint(...)` (ver `G0_initialization`, linea ~562 del archivo
   original) usando el estado GLOBAL del generador aleatorio de torch. Ese
   estado sigue avanzando de una llamada a la siguiente, asi que cada beta de
   la grilla ve una inicializacion de G (y un orden de batches, porque los
   DataLoader tambien usan `shuffle=True` con el generador global) TOTALMENTE
   DISTINTA e independiente de la de los demas betas. El estudio de este
   mismo proyecto (Fase 1 de `analisis_epochs_series_minimos/`) mostro que el
   paisaje accuracy-vs-beta es MUY plano cerca del optimo: con esa cantidad
   de ruido de inicializacion sin controlar, y sin promediar sobre series
   (search_best_beta no tiene bucle de series, solo de folds), es facil que
   el argmax elija un beta "ganador" solo por una inicializacion afortunada,
   no porque ese beta sea mejor.
   FIX: "common random numbers" (tecnica estandar de reduccion de varianza en
   comparaciones de Monte Carlo): antes de crear el modelo de CADA beta para
   un fold `k` dado, se resetea la semilla de torch a un valor que depende
   SOLO de `k` (no de beta). Asi, para un fold fijo, TODOS los betas de la
   grilla arrancan con exactamente la misma inicializacion de G y ven
   exactamente el mismo orden de batches -> la diferencia de accuracy entre
   betas refleja el efecto de beta, no el de la semilla que le toco.

4) FLACO: grilla de beta mal elegida (rango insuficiente + resolucion
   insuficiente en la practica) Y CIEGA -no usa nada de lo que ya se sabe
   sobre beta_optimo-.
   La firma por defecto ya es angosta (`beta_max=400`, `points=20` ->
   paso de ~21 en el rango [1,400]), pero la llamada REAL en
   `Crossbar_train_experimento_beta_por_capa.py` (linea ~396) la angosta
   todavia mas: `beta_max=300, points_beta=2` -> `np.linspace(1,300,2)` son
   SOLO DOS puntos, beta=1 y beta=300. Ademas, el rango [1,300] o [1,400] se
   queda corto: el ground truth de fuerza bruta (`data_beta_grid_search_SP/`,
   30 combos, grilla real de 1 a 1950) encuentra betas optimos de hasta 750
   en varios regimenes (ver `epoca_minima_por_combo.csv`,
   columna `beta_optimo_100ep`) -- fuera del rango que la busqueda automatica
   puede siquiera considerar.

   FIX (dos partes):

   4a) Grilla logaritmica en vez de lineal: beta_optimo escala como una
   potencia de sigma_W/paso_medio/N (ver `formula_beta_optimo_propuesta/`),
   asi que un espaciado logaritmico da resolucion relativa pareja en todo
   el rango en vez de sobre-muestrear los valores altos como haria una
   grilla lineal.

   4b) LA GRILLA SE CENTRA EN LA FORMULA ANALITICA YA ENCONTRADA en este
   proyecto (`formula_beta_optimo_propuesta/estimar_beta_optimo_propuesta_v2.py`,
   coeficientes en `coeficientes_regresion_v2.csv`, modelo
   `combinado_sigmaW_pasomedio`, R²=0.938 sobre 18 curvas):

       beta_pred = exp(-25.0472) * sigma_W**(-3.0475) * paso_medio**(-0.6072)

   con `sigma_W` (desvio estandar de W=G+-G- bajo `G0_initialization`,
   Monte Carlo con el `pot`/`dep` de ESTE modelo) y `paso_medio` (paso medio
   de la curva de potenciacion, `mean(|diff(pot)|)`) calculados en el
   momento, sin necesidad de correr ningun entrenamiento -es aritmetica
   sobre arrays que ya existen, tarda milisegundos-.

   Validacion nueva hecha para esta mejora (no estaba en el reporte
   original, que solo probaba 18 curvas): se corrio esta formula contra las
   30 combinaciones INL x pulsos que HOY existen en
   `data_beta_grid_search_SP/` (12 mas que las 18 usadas para ajustarla) y
   el cociente beta_predicho/beta_optimo_real da mediana=0.95, rango
   [0.72, 1.26] -> el 100% de los 30 casos cae dentro de un factor de 2x
   (de hecho, dentro de un factor de 1.3x). Ver
   `validacion_formula_v2_30combos.csv` en esta misma carpeta.

   Con una prediccion tan buena, la grilla de busqueda ya NO necesita cubrir
   [1,2000] a ciegas: alcanza con centrarla en `beta_pred` y cubrir un
   margen multiplicativo generoso (`margen=4x` por defecto, bastante mas
   ancho que el error maximo observado de 1.26x, para tener colchon en
   regimenes todavia no explorados -otro rango de conductancia, otra
   arquitectura-). Con el rango mucho mas angosto alcanza con MENOS puntos
   para la misma resolucion relativa (`n_puntos=10` en vez de 16).

   RED DE SEGURIDAD: si el optimo empirico encontrado cae en el borde de
   esa grilla angosta (senal de que la formula fallo en este regimen, como
   ya paso con la version anterior de la formula -ver
   `Reporte_beta_optimo_analitico.pdf`, "Barrido B"-), se expande
   automaticamente el rango (duplicando el margen) y se agregan solo los
   puntos NUEVOS que hacen falta -no se re-evaluan los que ya se probaron-,
   hasta que el optimo quede lejos del borde o se llegue a un tope de
   expansiones (`max_expansiones=2`).

5) CODIGO MUERTO: el parametro `std_range=(2.5,3.5)` esta en la firma pero
   nunca se usa en el cuerpo de la funcion.
   FIX: se remueve el parametro sin uso y, en su lugar, se agrega un
   diagnostico nuevo y realmente usado: la funcion tambien devuelve el
   "meseta" de betas cuyo accuracy medio cae dentro de 1 error estandar del
   mejor (regla del 1-SE, estandar en seleccion de modelos, ej. glmnet). Esto
   documenta explicitamente cuando el optimo encontrado es parte de un tramo
   plano (grupo de betas estadisticamente indistinguibles) en vez de un pico
   nitido, algo que el codigo original no reportaba y que, segun el mismo
   estudio de este proyecto, es la norma cerca del optimo real.

=============================================================================
COSTO COMPUTACIONAL (para que quede explicito que sigue siendo barato):
=============================================================================
Con los defaults de aca (n_puntos=10, k_folds_busqueda=3, epochs_busqueda=8):
  10 betas x 3 folds x 8 epocas = 240 "epoca-fold" unidades (sin expansion).
Contra el grid search de fuerza bruta real del proyecto:
  40 betas x 20 series x 5 folds x 100 epocas = 400 000 unidades.
-> mas de 1600x mas barato que el grid search completo. Medido en esta
   maquina (CPU, arquitectura SP 784->10): ~218s (~3.6 min) por llamada a
   `search_best_beta`, promediado sobre 10 corridas reales de validacion
   (ver `validar_mejora.py` / `resultados_validacion.json`).
"""

import sys
import os

FUERZA_BRUTA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Fuerza_bruta")
)
sys.path.insert(0, FUERZA_BRUTA_DIR)  # para importar el codigo ORIGINAL, sin copiarlo ni modificarlo

import numpy as np
import torch

from MemCrossbarClass_beta_por_capa import MemDNN, G0_initialization  # noqa: E402  (original, sin tocar)


# =============================================================================
# Formula analitica ya encontrada en este proyecto (no se re-deriva aca), ver
# `formula_beta_optimo_propuesta/coeficientes_regresion_v2.csv`, modelo
# `combinado_sigmaW_pasomedio`, R²=0.938 sobre 18 curvas, validada aca contra
# 30 (ver docstring del modulo y `validacion_formula_v2_30combos.csv`).
# =============================================================================
_INTERCEPT = -25.047163289759965
_COEF_LOG_SIGMA_W = -3.0474631066320534
_COEF_LOG_PASO_MEDIO = -0.6071708626364334


def beta_pred_formula(sigma_W, paso_medio):
    """beta_óptimo predicho por la fórmula empírica (ver arriba). `sigma_W` y
    `paso_medio` en las mismas unidades (Siemens) que usa el resto del
    código de este proyecto."""
    log_beta = _INTERCEPT + _COEF_LOG_SIGMA_W * np.log(sigma_W) + _COEF_LOG_PASO_MEDIO * np.log(paso_medio)
    return float(np.exp(log_beta))


def estimar_sigma_W_y_paso_medio(pot, dep, n_muestras=784, seed=35):
    """Estima sigma_W (Monte Carlo, vía la G0_initialization ORIGINAL sin
    modificar) y paso_medio (de la curva de potenciación) para un juego de
    curvas pot/dep dado. Tarda milisegundos (no entrena nada)."""
    gen_state = torch.get_rng_state()
    try:
        torch.manual_seed(seed)
        G = G0_initialization('random', pot, dep, n_muestras, 1)
    finally:
        torch.set_rng_state(gen_state)  # no perturbar el RNG global del resto de la busqueda
    W = (G[:, 0] - G[:, 1]).cpu().numpy()
    sigma_W = float(W.std())
    pot_np = pot.cpu().numpy() if torch.is_tensor(pot) else np.asarray(pot)
    paso_medio = float(np.mean(np.abs(np.diff(pot_np))))
    return sigma_W, paso_medio


class MemDNNMejorado(MemDNN):
    """Subclase de MemDNN que unicamente sobreescribe `search_best_beta`.
    Todo lo demas (forward, train_epoch, evaluate_model, G0_initialization,
    etc.) se hereda TAL CUAL del original."""

    def search_best_beta(
        self,
        train_loaders,
        val_loaders,
        loss_fn,
        a: float,  # se mantiene por compatibilidad de firma; ya no se usa
                   # para decidir si correr o no la busqueda (ver error 1)
        lr,
        delta_t_forward,
        delta_t_pulse,
        Vr, Vs,
        margen: float = 4.0,
        n_puntos: int = 10,
        k_folds_busqueda: int = 3,
        epochs_busqueda: int = 8,
        seed_base: int = 12345,
        max_expansiones: int = 2,
        verbose: bool = True,
    ):
        """
        Busqueda de beta mejorada (ver docstring del modulo para el detalle
        de cada correccion respecto de la version original). En vez de
        barrer una grilla fija y ciega, primero calcula `beta_pred` con la
        formula analitica ya encontrada en este proyecto (sigma_W y
        paso_medio de las curvas P/D de ESTE modelo) y centra ahi una
        grilla log-espaciada angosta (`beta_pred/margen .. beta_pred*margen`).
        Si el optimo empirico cae en el borde de esa grilla (la formula
        fallo en este regimen), expande el rango automaticamente.

        Devuelve: best_beta, best_acc_mean, info

        `info` (dict) incluye, ademas del diagnostico de la version anterior:
            beta_pred_formula : estimacion inicial (sin entrenar nada)
            sigma_W, paso_medio : insumos de esa estimacion
            expansiones        : cuantas veces hubo que agrandar el rango
        """
        assert len(train_loaders) == len(val_loaders)
        n_folds_disponibles = len(train_loaders)
        k_folds_busqueda = min(k_folds_busqueda, n_folds_disponibles)

        # --- FIX 4b: primero estimar, despues buscar cerca de la estimacion ---
        sigma_W, paso_medio = estimar_sigma_W_y_paso_medio(self.pot, self.dep, n_muestras=self.sizes[0])
        beta_pred = beta_pred_formula(sigma_W, paso_medio)
        if verbose:
            print(f"[formula] sigma_W={sigma_W:.4e} paso_medio={paso_medio:.4e} "
                  f"-> beta_pred={beta_pred:.1f} (centro de la grilla de busqueda)")

        betas_evaluados = {}  # beta -> (acc_mean, acc_sem), cache para no re-evaluar en una expansion
        rango = margen

        def evaluar_beta(beta):
            if beta in betas_evaluados:
                return betas_evaluados[beta]
            fold_accs = []
            for k in range(k_folds_busqueda):
                # --- FIX 3: common random numbers ---
                # La semilla depende SOLO del fold `k` (no del beta), asi
                # todos los betas ven la MISMA inicializacion de G0 y el
                # MISMO orden de batches para un fold dado. Esto aisla el
                # efecto de beta del ruido de inicializacion.
                torch.manual_seed(seed_base + k)
                np.random.seed(seed_base + k)

                model = MemDNN(
                    sizes=self.sizes,
                    beta=float(beta),
                    pot=self.pot,
                    dep=self.dep,
                    G0_distribution='random',
                    fixed=self.fixed,
                    device=self.device,
                )
                model.train()

                # --- FIX 2: varias epocas, no 1 sola ---
                for _ in range(epochs_busqueda):
                    model.train_epoch(
                        train_loaders[k], lr, loss_fn,
                        delta_t_forward, delta_t_pulse, Vr, Vs,
                    )

                acc, _ = model.evaluate_model(val_loaders[k], loss_fn)
                fold_accs.append(acc)

            acc_mean = float(np.mean(fold_accs))
            acc_sem = float(np.std(fold_accs) / np.sqrt(len(fold_accs)))
            betas_evaluados[beta] = (acc_mean, acc_sem)
            if verbose:
                print(f"[beta={beta:9.2f}] acc_mean={acc_mean:.4f} +- {acc_sem:.4f} "
                      f"(SEM, {k_folds_busqueda} folds)")
            return acc_mean, acc_sem

        expansiones = 0
        while True:
            betas = np.geomspace(beta_pred / rango, beta_pred * rango, n_puntos)
            for beta in betas:
                evaluar_beta(float(beta))

            betas_ordenados = np.array(sorted(betas_evaluados.keys()))
            accs_ordenados = np.array([betas_evaluados[b][0] for b in betas_ordenados])
            idx_best = int(np.argmax(accs_ordenados))
            idx_en_esta_grilla = int(np.argmin(np.abs(betas - betas_ordenados[idx_best])))

            en_el_borde = idx_en_esta_grilla in (0, len(betas) - 1)
            if en_el_borde and expansiones < max_expansiones:
                expansiones += 1
                rango *= 2
                if verbose:
                    print(f"[aviso] el optimo cayo en el borde de la grilla centrada en la formula "
                          f"-> la formula parece fallar en este regimen; expandiendo margen a {rango:.0f}x "
                          f"(expansion {expansiones}/{max_expansiones})")
                continue
            break

        best_beta = float(betas_ordenados[idx_best])
        best_acc_mean = float(accs_ordenados[idx_best])
        best_sem = betas_evaluados[best_beta][1]

        # --- FIX 5: diagnostico de "meseta" (regla del 1-SE) en vez de un parametro sin usar ---
        umbral_meseta = best_acc_mean - best_sem
        meseta_1se = betas_ordenados[accs_ordenados >= umbral_meseta].tolist()

        self.beta = self._expand_beta(best_beta, len(self.G_layers))

        info = {
            "beta_pred_formula": beta_pred,
            "sigma_W": sigma_W,
            "paso_medio": paso_medio,
            "betas": betas_ordenados,
            "acc_mean_por_beta": accs_ordenados,
            "meseta_1se": meseta_1se,
            "k_folds_busqueda": k_folds_busqueda,
            "epochs_busqueda": epochs_busqueda,
            "expansiones": expansiones,
        }
        if verbose:
            print(f"[resultado] beta_pred(formula)={beta_pred:.1f} -> beta_elegido(busqueda)={best_beta:.1f} "
                  f"(factor {best_beta/beta_pred:.2f}x sobre la prediccion)")
            if len(meseta_1se) > 1:
                print(f"[aviso] {len(meseta_1se)} betas quedan dentro de 1 error estandar "
                      f"del mejor (meseta plana): {[round(b) for b in meseta_1se]}")

        return best_beta, best_acc_mean, info
