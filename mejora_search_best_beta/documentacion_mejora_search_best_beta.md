# Documentación: mejora de `search_best_beta()`

Walter Quiñonez · análisis realizado con Claude Code · 9 de septiembre de 2026

## 0. Pedido

> Analiza el método `search_best_beta()`. Si hay algún error metodológico en la búsqueda,
> mejoralo para que sea más preciso. Usa cualquier método que te parezca adecuado, pero que
> no sea costoso computacionalmente.

No se modificó `Fuerza_bruta/Crossbar_train_experimento_beta_por_capa.py` ni
`Fuerza_bruta/MemCrossbarClass_beta_por_capa.py`. La mejora vive en una subclase,
`MemDNNMejorado`, en `search_best_beta_mejorado.py` (esta carpeta), que hereda todo de
`MemDNN` y solo sobreescribe `search_best_beta`. Es un reemplazo "drop-in": donde se usaba
`MemDNN(...)` alcanza con usar `MemDNNMejorado(...)`.

## 1. El método original y sus errores

Código completo original: `MemCrossbarClass_beta_por_capa.py:268-370`. Por cada beta de una
grilla, entrena 1 época en cada fold (creando un modelo nuevo cada vez) y promedia el accuracy;
se queda con el beta de mayor accuracy promedio.

### 1.1 Bug: rama muerta sin `else`

Todo el cuerpo está adentro de `if a <= 10000:`. Si se llama con `a_pot > 10000` (una curva
P/D perfectamente válida — más lineal, menos no ideal), la función no hace nada y devuelve
`None` en silencio. No hay ninguna razón documentada para ese umbral en el código ni en los
papers de `Fuerza_bruta/papers/`.

### 1.2 Solo 1 época de entrenamiento por candidato

El estudio de este mismo proyecto en `analisis_epochs_series_minimos/` (que usa el ground
truth real de 100 épocas guardado en `data_beta_grid_search_SP/`, 30 combinaciones INL×pulsos)
midió el costo de elegir beta con pocas épocas: con 1 sola época, el peor de los 30 combos
pierde hasta 1.48 puntos porcentuales de accuracy final respecto del beta verdaderamente
óptimo. Con 8 épocas, ese regret cae por debajo de 1pp en el peor caso.

### 1.3 El error más serio: sin control de aleatoriedad entre candidatos de beta

Para cada beta y cada fold se crea un `MemDNN` nuevo con `G0_distribution='random'`, que
sortea la conductancia inicial con `torch.randint(...)` usando el estado **global** del
generador aleatorio de torch. Ese estado sigue avanzando de una llamada a la siguiente, así
que cada beta de la grilla ve una inicialización de G (y un orden de batches — los
`DataLoader` también usan `shuffle=True` con el generador global) **totalmente distinta e
independiente** de la de los demás betas.

Como el paisaje accuracy-vs-beta es muy plano cerca del óptimo (hallazgo de la Fase 1 de
`analisis_epochs_series_minimos/`), y `search_best_beta` no promedia sobre series (solo sobre
folds), es fácil que el `argmax` elija un beta "ganador" solo por una inicialización
afortunada, no porque ese beta sea mejor.

### 1.4 Grilla de beta mal elegida, y ciega

La firma por defecto ya es angosta (`beta_max=400, points=20`), pero la llamada real en
`Crossbar_train_experimento_beta_por_capa.py:396` la angosta más todavía:
`beta_max=300, points_beta=2` → `np.linspace(1,300,2)` son **solo dos puntos**, β=1 y β=300.
Además el rango se queda corto: el ground truth de fuerza bruta encuentra betas óptimos de
hasta 750 en varios regímenes — fuera del rango que la búsqueda automática puede considerar.

### 1.5 Código muerto

El parámetro `std_range=(2.5,3.5)` está en la firma y nunca se usa en el cuerpo.

## 2. Las correcciones

| # | Problema | Corrección |
|---|---|---|
| 1 | rama muerta sin `else` | se elimina la condición, la búsqueda corre siempre |
| 2 | 1 época por candidato | `epochs_busqueda=8` (configurable), en base al hallazgo de `analisis_epochs_series_minimos/` |
| 3 | sin control de aleatoriedad | *common random numbers*: la semilla depende solo del fold, no del beta |
| 4 | grilla angosta/ciega | grilla log-espaciada **centrada en la fórmula analítica ya encontrada** (ver §3), con red de seguridad que expande el rango si la fórmula falla |
| 5 | `std_range` sin usar | se reemplaza por un diagnóstico real: la "meseta" de betas dentro de 1 error estándar del mejor (regla del 1-SE) |

## 3. Usar la fórmula analítica como primera estimación

En `formula_beta_optimo_propuesta/` (trabajo previo de este proyecto) se había ajustado, por
regresión log-log sobre 18 curvas P/D reales:

```
β_óptimo ≈ 1.34×10⁻¹¹ · σ_W^(−3.05) · paso_medio^(−0.61)      (R² = 0.938, n = 18)
```

con `σ_W` = desvío estándar de W=G⁺−G⁻ (Monte Carlo vía `G0_initialization`, sin modificar) y
`paso_medio` = paso medio de la curva de potenciación (`mean(|diff(pot)|)`).

**Validación nueva, hecha para esta mejora** (no estaba en el reporte original, que solo
probaba 18 curvas): se corrió esta fórmula contra las **30 combinaciones** que hoy existen en
`data_beta_grid_search_SP/` (12 más que las usadas para ajustarla). Resultado
(`validacion_formula_v2_30combos.csv`):

| | valor |
|---|---|
| cociente β_predicho / β_óptimo_real — mediana | 0.948 |
| cociente — mínimo | 0.720 |
| cociente — máximo | 1.261 |
| % de los 30 casos dentro de un factor de 2× | **100%** |

La fórmula predice el óptimo real dentro de un factor de 1.26× en el peor de los 30 casos ya
explorados — mucho mejor de lo que se pensaba necesario. Con una predicción así de buena, la
búsqueda ya no necesita cubrir un rango amplio [1,2000] a ciegas: alcanza con centrarla en
`beta_pred` (calculado en milisegundos, sin entrenar nada) y cubrir un margen multiplicativo
de seguridad (`margen=4×` por defecto, con bastante colchón sobre el error máximo observado).

Si el óptimo empírico cae en el borde de esa grilla angosta (señal de que la fórmula falló en
ese régimen — como ya le pasó a una versión anterior de la fórmula con el efecto de la no
linealidad, ver `Reporte_beta_optimo_analitico.pdf`), la búsqueda expande automáticamente el
rango (hasta `max_expansiones=2` veces), reutilizando los puntos ya evaluados.

## 4. Validación empírica: original vs. mejorado

Script: `validar_mejora.py`. Metodología: 2 combos representativos (extremos de no linealidad:
INL nominal 2×10⁻¹ y 1×10⁻²), 5 repeticiones cada uno con semillas de fold **nuevas**
(`seed=20000+rep`, no relacionadas con las 20 series del grid search original), comparando 3
métodos y midiendo el *regret* real contra el ground truth completo (100 épocas, 20 series, 5
folds ya guardado):

- **(a)** original tal como se usa hoy en producción (`beta_max=300, points=2, k_folds=5`, 1 época).
- **(b)** original con su firma "de fábrica" (`beta_max=400, points=20, k_folds=5`, 1 época).
- **(c)** mejorado (`MemDNNMejorado`, defaults: margen=4, n_puntos=10, k_folds_busqueda=3,
  epochs_busqueda=8).

### Resultado (10 corridas por método, `resultados_validacion.json`)

| Método | regret medio | regret mediana | **regret peor caso** | tiempo promedio |
|---|---|---|---|---|
| (a) original (como se usa hoy) | 10.19 pp | 10.19 pp | **20.30 pp** | 11.8 s |
| (b) original (defaults de fábrica) | 0.24 pp | 0.03 pp | 2.16 pp | 118.2 s |
| **(c) mejorado** | **0.08 pp** | **0.04 pp** | **0.45 pp** | 218.2 s |

El método (a) —el que realmente corre hoy en `Crossbar_train_experimento_beta_por_capa.py`—
elige siempre β=1 en el combo más no lineal (INL=2e-1), perdiendo sistemáticamente ~20 puntos
porcentuales de accuracy: con solo 2 puntos de grilla, cuando ninguno cae cerca del óptimo real
(100, en ese combo) no hay forma de recuperarse. El método (b) mejora mucho con más puntos de
grilla, pero sigue siendo errático (hasta 2.16pp) por la falta de control de semillas. El
método mejorado (c) nunca superó 0.45pp de regret en las 10 corridas, a un costo de ~3.6
minutos por búsqueda (1.8× más lento que (b), pero sigue siendo despreciable frente a un
entrenamiento completo de horas).

### Detalle por corrida

| combo | rep | (a) β elegido → regret | (b) β elegido → regret | (c) β elegido → regret |
|---|---|---|---|---|
| INL_2e-1_p_50 (óptimo real=100) | 0 | 1 → 20.30pp | 85 → 0.00pp | 137.8 → 0.44pp |
| | 1 | 1 → 20.30pp | 64 → 2.16pp | 101.3 → 0.00pp |
| | 2 | 1 → 20.30pp | 106 → 0.00pp | 101.3 → 0.00pp |
| | 3 | 1 → 20.30pp | 106 → 0.00pp | 101.3 → 0.00pp |
| | 4 | 1 → 20.30pp | 85 → 0.00pp | 101.3 → 0.00pp |
| INL_1e-2_p_100 (óptimo real=350) | 0 | 300 → 0.07pp | 316 → 0.07pp | 329.6 → 0.00pp |
| | 1 | 300 → 0.07pp | 358 → 0.00pp | 448.6 → 0.09pp |
| | 2 | 300 → 0.07pp | 400 → 0.06pp | 448.6 → 0.09pp |
| | 3 | 300 → 0.07pp | 400 → 0.06pp | 448.6 → 0.09pp |
| | 4 | 300 → 0.07pp | 400 → 0.06pp | 448.6 → 0.09pp |

Nota honesta: en el combo INL_1e-2_p_100, (c) queda ligeramente por encima de (a)/(b) en
regret absoluto (0.09pp vs 0.06-0.07pp) — la diferencia es minúscula (menos de un décimo de
punto porcentual) y (a) acierta ahí por casualidad (300 quedó cerca de 350 en esa grilla de 2
puntos); en el otro combo (a) pierde 20pp. El mejorado es el único que nunca fue mal en
ninguno de los 20 casos.

## 5. Diagnóstico de la "meseta" (regla del 1-SE)

Cerca del óptimo, varios betas dan accuracy estadísticamente indistinguible. El método
mejorado ahora reporta explícitamente ese grupo (`info["meseta_1se"]`), algo que el original
no hacía (el parámetro `std_range` estaba ahí para algo parecido a esto, pero nunca se usó).
Ejemplo real de una corrida:

```
[resultado] beta_pred(formula)=86.8 -> beta_elegido(busqueda)=137.8 (factor 1.59x sobre la prediccion)
[aviso] 2 betas quedan dentro de 1 error estandar del mejor (meseta plana): [55, 138]
```

## 6. Cómo usarlo

```python
from search_best_beta_mejorado import MemDNNMejorado

model = MemDNNMejorado(sizes=sizes, beta=beta_inicial, pot=pot, dep=dep,
                        G0_distribution=G0_distrbtn, fixed=fixed, device=device)

best_beta, best_acc, info = model.search_best_beta(
    train_loaders, val_loaders, criterion, a_pot, lr,
    delta_t_forward, delta_t_pulse, Vr, Vs,
)
```

La firma acepta los mismos primeros 9 argumentos posicionales que el original (por eso `a`
se mantiene, aunque ya no se usa para decidir si correr la búsqueda); los parámetros nuevos
(`margen`, `n_puntos`, `k_folds_busqueda`, `epochs_busqueda`, `seed_base`,
`max_expansiones`) tienen defaults razonables y no hace falta tocarlos para el uso normal.

## 7. Limitaciones

- La fórmula usada para centrar la grilla (§3) fue ajustada y validada solo para la
  arquitectura SP de este proyecto (`sizes=[784,10]`) y ratio Gmax/Gmin=10. Si se usa con
  otra arquitectura o rango de conductancia, la red de seguridad (expansión automática si el
  óptimo cae en el borde) es la que sostiene la precisión — conviene revisar los prints de
  `[aviso] ... expandiendo margen` la primera vez que se use en un régimen nuevo.
  `estimar_sigma_W_y_paso_medio` usa `n_muestras=self.sizes[0]`, así que para otra
  arquitectura sigue siendo internamente consistente con el tamaño de la crossbar real.
  Además, ni la fórmula ni la búsqueda mejorada optimizan betas *distintos por capa* (mismo
  alcance que el método original: buscan un único beta global).
- La validación de la Sección 4 usa 2 combos y 5 repeticiones cada uno (no los 30 combos
  completos ni cientos de repeticiones), por costo de tiempo; es una validación puntual, no
  exhaustiva, aunque cubre los dos extremos de no linealidad ya explorados.
- El costo de la búsqueda mejorada (~3.6 min) es mayor que el del método (a) tal como se usa
  hoy (~12s) — es la contrapartida directa de ya no fallar catastróficamente; sigue siendo
  despreciable frente a un entrenamiento completo real (horas).
