# Valores de a_pot / a_dep para INL objetivo ≈ 4.827834e-06

Fecha: 2026-09-05

## Contexto

Script de referencia: `plot_PDcurves.py`
Función usada: `generar_curvas_pot_dep()` en `MemCrossbarClass_beta_por_capa.py`
INL calculado con: `calcular_indice_nl()`

Parámetros fijos usados en la búsqueda (iguales a los de `plot_PDcurves.py`):
- Rhigh = 10000 Ω, Rlow = 1000 Ω → Gmin = 1e-4 S, Gmax = 1e-3 S
- concavidad_pot = 'pos'
- concavidad_dep = 'neg'
- a_pot = a_dep (mismo valor para ambas curvas)

## Origen del valor objetivo

El valor objetivo (4.827834e-06) corresponde al `inl_dep` ya registrado en:
`MLP_test_SP/config_SP_1_09_2026_20260905_151722_beta00_b800.json`
para `pulsos=100`, `a_pot=a_dep=4567.9`:
- inl_pot = 4.837374997954803e-06
- inl_dep = 4.827834398556257e-06

## Método

Búsqueda de raíz (bisección + `brentq`, SciPy) sobre `a_pot=a_dep` para que
`inl_dep(a, pulsos)` reproduzca el valor objetivo, replicando en NumPy puro
las funciones `generar_curvas_pot_dep` y `calcular_indice_nl`.
Validación: para pulsos=100 la raíz encontrada fue a=4567.900000199595,
coincidiendo con el valor ya conocido del config → confirma el método.

## Resultados (ajustando por inl_dep)

| pulsos | a_pot = a_dep | INL resultante |
|---|---|---|
| 50  | 2257.81 | 4.827834e-06 |
| 100 | 4567.9  | 4.827834e-06 (valor de referencia, ya conocido) |
| 200 | 9187.43 | 4.827834e-06 |

## Resultados alternativos (ajustando por inl_pot en vez de inl_dep)

inl_pot e inl_dep no son exactamente iguales para el mismo `a`
(pequeña asimetría en `calcular_indice_nl`). Si el objetivo fuera igualar
`inl_pot` en lugar de `inl_dep`:

| pulsos | a_pot = a_dep | INL resultante |
|---|---|---|
| 50  | 2262.51 | 4.827834e-06 |
| 100 | 4572.41 | 4.827834e-06 |
| 200 | 9191.85 | 4.827834e-06 |

---

## INL objetivo = 0.001

Mismo método y mismos parámetros fijos (Gmin=1e-4, Gmax=1e-3, concavidad_pot='pos', concavidad_dep='neg').

| pulsos | a_pot = a_dep (ajustando por inl_dep) | a_pot = a_dep (ajustando por inl_pot) |
|---|---|---|
| 50  | 156.62 | 156.90 |
| 100 | 316.81 | 317.08 |
| 200 | 637.16 | 637.42 |

## INL objetivo = 0.0001

| pulsos | a_pot = a_dep (ajustando por inl_dep) | a_pot = a_dep (ajustando por inl_pot) |
|---|---|---|
| 50  | 496.05 | 497.04 |
| 100 | 1003.53 | 1004.48 |
| 200 | 2018.36 | 2019.29 |

## INL objetivo = 0.00001

| pulsos | a_pot = a_dep (ajustando por inl_dep) | a_pot = a_dep (ajustando por inl_pot) |
|---|---|---|
| 50  | 1568.79 | 1572.04 |
| 100 | 3173.88 | 3177.00 |
| 200 | 6383.62 | 6386.67 |
