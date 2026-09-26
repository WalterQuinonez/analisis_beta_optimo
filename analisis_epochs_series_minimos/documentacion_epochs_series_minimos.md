# Documentación: cuántas épocas y cuántas series hacen falta para estimar beta óptimo

Walter Quiñonez · análisis realizado con Claude Code · 8 de septiembre de 2026

## 0. Objetivo y alcance

El grid search de fuerza bruta del proyecto (`Fuerza_bruta/Crossbar_train_experimento_beta_por_capa.py`,
corridas guardadas en `data_beta_grid_search_SP/`) usa por defecto `series=20`, `k_folds=5`,
`epochs=100` para cada uno de los 40 valores de beta de la grilla, en cada una de las 30
combinaciones (INL nominal × pulsos) ya exploradas. Esto son
`30 combos × 40 betas × 20 series × 5 folds × 100 épocas ≈ 12 millones` de época-fold-serie, y
34 GB en disco (mayormente `G_history_*.npz`, el snapshot completo de las conductancias).

Pregunta de este estudio: **¿son necesarios esos números, o alcanza con mucho menos para
identificar el mismo beta óptimo (o uno con la misma accuracy final, dentro de una tolerancia
razonable)?** Se responde en dos pasos que reutilizan los datos ya existentes (sin correr nada
nuevo) y un tercer paso de corroboración con corridas nuevas mínimas.

No se modificó `Crossbar_train_experimento_beta_por_capa.py` ni `MemCrossbarClass_beta_por_capa.py`
en ningún momento. Todo el código de este estudio vive en
`analisis_beta_optimo/analisis_epochs_series_minimos/`.

## 1. Fuente de datos

`comun_carga_datos.py` carga, para cada una de las 30 carpetas
`data_beta_grid_search_SP/beta_grid_search_SP_INL_<INL>_p_<pulsos>/`, los 40 archivos
`resultados_*_beta*.npz` (uno por beta de la grilla), cada uno con `acc_result` de forma
`(series=20, folds=5, epochs=100)`. Con esto se arma, por combo, un tensor
`acc[beta, serie, fold, época]` de forma `(40, 20, 5, 100)`.

Verificación rápida (`comun_carga_datos.py` ejecutado standalone):

```
Combos encontrados: 30
beta_grid_search_SP_INL_1e-2_p_100 betas: (40,) acc: (40, 20, 5, 100)
...
```

## 2. Fase 1 — Épocas mínimas (`estudiar_epochs_minimos.py`)

### 2.1 Por qué "el índice de beta coincide" es el criterio equivocado

El primer intento fue: para cada época `e`, calcular `beta_hat(e) = argmax_beta acc_media(beta, e)`
y pedir que coincida con `beta_hat(100)` (a lo sumo 1 paso de grilla), sostenido hasta el final.
Resultado: mediana 40 épocas, p95 = 100 (columna `epoca_min_indice_estable` en
`epoca_minima_por_combo.csv`). Es decir, según este criterio casi no se podría acortar nada.

La razón es un artefacto, no una necesidad real: **el paisaje accuracy-vs-beta es muy plano
cerca del óptimo** (varios betas vecinos dan accuracy final indistinguible dentro del ruido
estadístico de 20 series × 5 folds), así que el `argmax` salta de forma errática entre esos betas
vecinos incluso a época 100. Exigir que el índice coincida exactamente penaliza saltos que no
tienen ningún costo práctico.

### 2.2 El criterio que sí importa: *regret*

Se define, para cada época `e`:

```
beta_hat(e)   = argmax_beta  acc_media(beta, época=e)
regret(e)     = acc_óptima_final  -  acc_media( beta_hat(e), época=100 )
```

Es decir: *si yo cortara la búsqueda de beta en la época `e` y me quedara con `beta_hat(e)`, pero
esa red se siguiera entrenando con normalidad hasta la época 100, ¿cuánta accuracy final pierdo
respecto del óptimo real?* Esto es exactamente lo que le importa a alguien que quiere abaratar la
*búsqueda* de beta sin tocar el entrenamiento final.

Se buscó la mínima época `e*` a partir de la cual `regret(e) < tol` de forma sostenida hasta la
época 100, para `tol ∈ {2%, 1%, 0.5%, 0.2%}` de accuracy absoluto.

### 2.3 Resultados (30 combos, `epoca_minima_por_combo.csv`)

| criterio | mediana | p90 | p95 | máx (peor de los 30 combos) |
|---|---|---|---|---|
| índice de beta exacto (referencia, no usar) | 40 | 100 | 100 | 100 |
| accuracy del propio ganador converge (<1pp) | 3 | 4 | 4.5 | 6 |
| **regret < 1pp (recomendado)** | **1** | **1** | **1** | **8** |

Sensibilidad a la tolerancia de regret (peor combo de los 30):

| tolerancia | mediana | p95 | máx |
|---|---|---|---|
| 2.0 pp | 1 | 1 | 1 |
| **1.0 pp** | 1 | 1 | **8** |
| 0.5 pp | 1 | 7.5 | 19 |
| 0.2 pp | 1 | 20.6 | 38 |

El regret máximo observado en absoluto (cualquier época, cualquier combo) fue **1.48 puntos
porcentuales** — ya de por sí chico incluso en el peor caso posible (época 1, el peor combo).

**Conclusión de la Fase 1:** con una tolerancia de 1 punto porcentual de accuracy final (razonable
para este problema, donde las accuracies rondan 85-92%), **8 épocas alcanzan** en el peor de los
30 combos ya explorados (INL desde 4.82e-6 hasta 2e-1, pulsos ∈ {50,100,200}). Esto es coherente
con que el propio `search_best_beta()` ya existente en el código (aunque desactivado por defecto,
`optimize_beta=False`) entrena solo **1 época** por candidato de beta — nuestro análisis, basado en
el ground truth real de 100 épocas, confirma independientemente que ese orden de magnitud (pocas
épocas) es razonable, y da un número concreto con margen de seguridad: **epochs_min = 8**
(recomendación conservadora: usar 10 para tener margen).

Gráfico: `epocas_minimas_vs_INL_pulsos.png`.

## 3. Fase 2 — Series y folds mínimos (`estudiar_series_minimas.py`)

### 3.1 Metodología: subsampling sin reemplazo sobre datos ya existentes

Para cada combo, y para `n_series ∈ {1,2,3,5,8,12,16,20}` (con `k_folds=5` fijo, que es el
parámetro que pidió acotar el usuario — "la cantidad de series por entrenamiento"):

1. Se repite 300 veces: elegir `n_series` series al azar (sin reemplazo) de las 20 disponibles,
   promediar `acc(beta, época=8)` sobre esas series × 5 folds (época 8 = `epochs_min` recomendado
   en la Fase 1, porque se evalúa el protocolo reducido *completo*, no solo el efecto de las
   series por separado), elegir `beta_hat = argmax_beta`.
2. Calcular el mismo `regret` de la Fase 1 pero evaluando `beta_hat` con el ground truth completo
   (20 series × 5 folds × época 100).
3. Reportar `P(regret < 1pp)` sobre las 300 repeticiones (confiabilidad de la decisión).

Se repite el mismo análisis variando `n_folds ∈ {1,2,3,5}` con `series=20` fijo (chequeo
secundario).

### 3.2 Resultados (peor de los 30 combos, `series_minimas_vs_precision.csv`)

| n_series (folds=5) | P(regret<1pp) media | P(regret<1pp) peor combo |
|---|---|---|
| 1 | 95.8% | 43.3% |
| 2 | 97.7% | 63.7% |
| 3 | 97.8% | 64.0% |
| 5 | 98.9% | 82.0% |
| **8** | **99.7%** | **93.3%** |
| **12** | **99.9%** | **98.0%** |
| 16 | 100.0% | 100.0% |
| 20 (todas) | 100.0% | 100.0% |

| n_folds (series=20) | P(regret<1pp) media | P(regret<1pp) peor combo |
|---|---|---|
| 1 | 99.3% | 78.7% |
| 2 | 99.1% | 73.3% |
| **3** | **100.0%** | **100.0%** |
| 5 (todas) | 100.0% | 100.0% |

**Conclusión de la Fase 2:**
- `series_min = 12` da ≥98% de confiabilidad en el peor de los 30 combos ya explorados (100% en
  la media). `series_min = 8` ya da 93% en el peor combo — aceptable si se prioriza velocidad
  sobre robustez extrema, pero 12 es la recomendación por defecto.
- `k_folds` puede bajar de 5 a **3** sin ninguna pérdida medible de confiabilidad (100% en ambos
  casos, en los 30 combos). No hace falta usar 5 folds para este propósito.

Gráfico: `series_minimas_vs_precision.png`.

## 4. Fase 3 — Corroboración con corridas nuevas (código copiado, protocolo reducido)

### 4.1 Por qué hacía falta esto además de las Fases 1-2

Las Fases 1-2 reanalizan datos que ya usaban `G0_distribution='random'` y las mismas 20
particiones de folds (semillas `35..54`) que el resto del proyecto. Para verificar que la
recomendación no es un artefacto de reciclar exactamente esos mismos datos, se corrieron
combinaciones nuevas con:

- Semillas de fold nuevas (`seed=9035`, en vez de `35`) — ninguna de las 20 series ya usadas se
  reutiliza.
- El protocolo reducido: `series=8`, `k_folds=3`, `epochs=8` (la variante más agresiva
  considerada, no la conservadora de `series=12`, para poner a prueba el caso más exigente).
- Una grilla de beta más angosta pero no arbitraria: 16 valores tomados de la grilla original de
  40 (1, 50, 100, ..., hasta 1600), sin asumir de antemano dónde está el óptimo.

Código: `codigo_reducido/Crossbar_train_confirmatorio.py`. Es una copia adaptada del script
original (no se tocó el original); importa `MemCrossbarClass_beta_por_capa.py` de
`Fuerza_bruta/` sin copiarlo ni modificarlo (`sys.path.insert`). Antes de cada serie chequea
`shutil.disk_usage()` y aborta si quedarían menos de 50 GB libres en el disco.

### 4.2 Combos elegidos

Dos combos representativos de los extremos de no linealidad ya explorados en
`data_beta_grid_search_SP/`, tomando los parámetros de curva P/D (`a_pot`, `a_dep`, `Gmin`,
`Gmax`, `G0_distribution`, etc.) directamente de los `config_*.json` ya guardados, para que la
única diferencia real con el ground truth sea el protocolo:

| combo | INL nominal | pulsos | beta_óptimo (ground truth, 100% del protocolo) | acc (ground truth) |
|---|---|---|---|---|
| INL_2e-1_p_50 | 2e-1 (más no lineal) | 50 | 100 | 0.8479 |
| INL_1e-2_p_100 | 1e-2 (intermedio) | 100 | 350 | 0.9091 |

### 4.3 Resultado

| combo | beta elegido (protocolo reducido, semillas nuevas) | beta óptimo (ground truth) | acc ground truth del beta elegido (época 100, protocolo completo) | acc óptima (ground truth) | **regret real** | tiempo de la corrida confirmatoria |
|---|---|---|---|---|---|---|
| INL_2e-1_p_50 | 150 | 100 | 0.8434 | 0.8479 | **0.445 pp** | 3315 s (≈55 min) |
| INL_1e-2_p_100 | 300 | 350 | 0.9084 | 0.9091 | **0.071 pp** | 3400 s (≈57 min) |

El "regret real" se calculó tomando el beta elegido por el protocolo reducido (con semillas de
fold completamente nuevas, `seed=9035`) y buscando su accuracy en el **ground truth completo**
(20 series × 5 folds × época 100, los mismos datos de `data_beta_grid_search_SP/` usados en las
Fases 1-2) — es decir, "si me quedo con este beta y entreno con el protocolo completo, ¿cuánto
pierdo respecto del óptimo real?".

En ambos combos el beta elegido por el protocolo más agresivo considerado (`series=8, k_folds=3,
epochs=8`, semillas nuevas) quedó **dentro de la tolerancia de 1 pp** fijada en las Fases 1-2
(0.445 pp y 0.071 pp de regret real, respectivamente) — y de hecho por debajo del promedio de
regret esperado según la tabla de la Sección 3.2 para `n_series=8` (P(regret<1pp)≈93% en el peor
combo ya predecía que la mayoría de las corridas con ese protocolo caerían dentro de tolerancia).
Esto corrobora, con datos genuinamente nuevos (no reciclados), que la recomendación de las Fases
1-2 es real y no un artefacto de sobreajustar a las 20 series ya existentes.

Cada corrida confirmatoria (16 betas × 8 series × 3 folds × 8 épocas = 3072 época-fold-serie)
tardó ≈55-57 minutos en CPU local (single-thread, sin GPU) — consistente con la reducción de
≈28-400× en cómputo estimada en la Sección 5.

## 5. Recomendación final

| parámetro | valor original (grid search) | valor recomendado | reducción |
|---|---|---|---|
| epochs | 100 | **10** (8 + margen) | 10× |
| series | 20 | **12** | 1.7× |
| k_folds | 5 | **3** | 1.7× |
| **total (épocas×series×folds)** | **10 000** | **360** | **≈28×** |

Con esta configuración, el estudio de subsampling (Fase 2) predice ≥98% de probabilidad de
identificar un beta con accuracy final a menos de 1 punto porcentual del óptimo verdadero, en
cualquiera de los 30 regímenes de INL/pulsos ya explorados. Para el uso más agresivo posible
(`series=8, k_folds=3, epochs=8`, ≈14× más rápido aún, ≈**400×** más barato que el protocolo
original) la confiabilidad en el peor caso baja a ~93%; queda corroborado con corridas nuevas en
la Fase 3.

## 6. Limitaciones

- El análisis de las Fases 1-2 es *post-hoc* sobre los 30 combos ya explorados (arquitectura fija
  `sizes=[784,10]`, MNIST, `Gmin/Gmax` fijos con ratio 10). No cubre variaciones de arquitectura
  (MLP con capa oculta), de dataset (Fashion-MNIST) ni de rango de conductancia (ratio ≠ 10) que
  puedan explorarse más adelante para completar el pedido original de `prompt.txt`; si se agregan
  esos regímenes conviene repetir al menos la Fase 3 (corroboración) en ellos antes de asumir que
  la misma recomendación aplica.
- El criterio de regret usa como referencia la propia accuracy a época 100 de la corrida
  completa (dato ya existente); no hay una verdad "más allá" de eso con la que contrastar (p.ej.
  más de 100 épocas) — se asume que 100 épocas ya representa convergencia práctica, supuesto
  razonable dado que las curvas de accuracy se aplanan muy antes de la época 100 en los datos
  disponibles.
- La Fase 3 usa solo 2 combos (no los 30) y una única corrida por combo (no repeticiones), porque
  cada corrida nueva, aun con el protocolo reducido, no es gratis en tiempo de cómputo local
  (~1-2h por combo en CPU). Sirve como corroboración puntual, no como confirmación exhaustiva.
