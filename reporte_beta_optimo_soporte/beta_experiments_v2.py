#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2: fixes a circularity in v1 (beta search grid range was derived from the very
analytic formula being tested, biasing the result). Here each experiment uses ONE
common, fixed beta grid (independent of sigma_W/N/V_rms) shared across all
configurations in that sweep, and each configuration is repeated with several
random subsamples/seeds and the median best_beta is reported, to reduce the
noise from the reduced-scale (single-epoch, small-subsample) search.
Still uses ONLY functions/classes already provided in MemCrossbarClass.py.
"""
import sys, os, json, time
PROJECT = r"C:\Users\walte\Documents\pesos_sinapticos_crossbar"
OUT = r"C:\Users\walte\AppData\Local\Temp\claude\C--Users-walte-Documents-pesos-sinapticos-crossbar\1a906e48-0654-416f-a8f5-112296b0c6ec\scratchpad"
sys.path.insert(0, PROJECT)

import numpy as np
import torch
import torch.nn as nn
from MemCrossbarClass import (generar_curvas_pot_dep, calcular_indice_nl,
                               G0_initialization, MemDNN, DataSetLoader)

X_full = np.load(os.path.join(PROJECT, "X_train_mnist.npy"))
y_full = np.load(os.path.join(PROJECT, "y_train_mnist.npy"))
amplitud_imagen = 1.0

def voltage_stats(X_cols):
    V = X_cols * amplitud_imagen
    return dict(V_rms=float(np.sqrt(np.mean(V**2))), V_mean=float(np.mean(V)))

def sigma_W_from_curves(pot, dep, n_samples=200000):
    pot_t = torch.tensor(pot, dtype=torch.float32)
    dep_t = torch.tensor(dep, dtype=torch.float32)
    G = G0_initialization('random', pot_t, dep_t, D_in=n_samples, D_out=1, device='cpu')
    W = (G[:, 0] - G[:, 1]).numpy()
    return dict(sigma_W=float(np.std(W)), mean_W=float(np.mean(W)))

def beta_analytic(N, sigma_W, V_rms, kappa=3.0):
    return kappa / (np.sqrt(N) * sigma_W * V_rms)

def run_one(X_cols, y, N_in, pot, dep, beta_max, points, seed,
            n_subsample=3000, k_folds=3, batch_number=15,
            lr=1.0, dt=10e-9, Vr=-1.0, Vs=1.0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X_cols), size=min(n_subsample, len(X_cols)), replace=False)
    Xs = torch.from_numpy(X_cols[idx]).float()
    ys = torch.from_numpy(y[idx]).long()
    train_loaders, val_loaders = DataSetLoader(Xs, ys, k_folds, batch_number)
    pot_t = torch.tensor(pot, dtype=torch.float32)
    dep_t = torch.tensor(dep, dtype=torch.float32)
    model = MemDNN(sizes=[N_in, 10], beta=1.0, pot=pot_t, dep=dep_t,
                    G0_distribution='random', fixed=False, device='cpu')
    best_beta, best_acc = model.search_best_beta(
        train_loaders, val_loaders, nn.CrossEntropyLoss(),
        a=5.85, lr=lr, delta_t_forward=dt, delta_t_pulse=dt,
        Vr=Vr, Vs=Vs, beta_max=beta_max, points=points)
    return float(best_beta), float(best_acc)

def run_repeated(X_cols, y, N_in, pot, dep, beta_max, points, n_repeats=3, seed0=100):
    betas, accs = [], []
    for r in range(n_repeats):
        b, a = run_one(X_cols, y, N_in, pot, dep, beta_max, points, seed=seed0 + r)
        betas.append(b); accs.append(a)
    return dict(best_beta_repeats=betas, best_acc_repeats=accs,
                best_beta_median=float(np.median(betas)),
                best_beta_mean=float(np.mean(betas)),
                best_beta_std=float(np.std(betas)),
                best_acc_mean=float(np.mean(accs)))

Rhigh_base, Rlow_base = 10000.0, 1000.0
Gmin_base, Gmax_base = 1/Rhigh_base, 1/Rlow_base
a_base = 5.85
pulsos_base = 100
vs784 = voltage_stats(X_full)

pot_b, dep_b, ratio_b, inl_pot_b, inl_dep_b = generar_curvas_pot_dep(
    pulsos_base, pulsos_base, a_base, a_base, Gmin_base, Gmax_base, 'neg', 'neg')
sw_base = sigma_W_from_curves(pot_b, dep_b)

N_REPEATS = 3

# =================================================================
# EXPERIMENT A: crossbar-size scan -- FIXED common grid for all N
# =================================================================
print("\n=== Experiment A: crossbar-size scan (fixed common grid) ===")
N_list = [784, 392, 196, 98]
BETA_MAX_A, POINTS_A = 6000.0, 20
expA = []
t0 = time.time()
for N_in in N_list:
    col_idx = np.linspace(0, 783, N_in).astype(int)
    Xc = X_full[:, col_idx]
    vs = voltage_stats(Xc)
    pred = beta_analytic(N_in, sw_base['sigma_W'], vs['V_rms'], kappa=3.0)
    res = run_repeated(Xc, y_full, N_in, pot_b, dep_b, BETA_MAX_A, POINTS_A, n_repeats=N_REPEATS)
    res.update(N_in=N_in, sigma_W=sw_base['sigma_W'], V_rms=vs['V_rms'],
                beta_pred_kappa3=pred, INL=inl_pot_b)
    print(N_in, res, "elapsed", time.time() - t0)
    expA.append(res)

# =================================================================
# EXPERIMENT B: nonlinearity scan -- FIXED common grid
# =================================================================
print("\n=== Experiment B: nonlinearity (a) scan (fixed common grid) ===")
a_list = [2.0, 5.85, 20.0, 200.0, 1000.0]
BETA_MAX_B, POINTS_B = 4000.0, 20
expB = []
for a in a_list:
    pot, dep, ratio, inl_pot, inl_dep = generar_curvas_pot_dep(
        pulsos_base, pulsos_base, a, a, Gmin_base, Gmax_base, 'neg', 'neg')
    sw = sigma_W_from_curves(pot, dep)
    pred = beta_analytic(784, sw['sigma_W'], vs784['V_rms'], kappa=3.0)
    res = run_repeated(X_full, y_full, 784, pot, dep, BETA_MAX_B, POINTS_B, n_repeats=N_REPEATS)
    res.update(a=a, sigma_W=sw['sigma_W'], V_rms=vs784['V_rms'],
                beta_pred_kappa3=pred, INL=inl_pot)
    print(a, res, "elapsed", time.time() - t0)
    expB.append(res)

# =================================================================
# EXPERIMENT C: conductance-range scan -- FIXED common grid
# =================================================================
print("\n=== Experiment C: conductance-range scan (fixed common grid) ===")
Rhigh_list = [2000.0, 5000.0, 10000.0, 50000.0, 100000.0]
BETA_MAX_C, POINTS_C = 5000.0, 20
expC = []
for Rhigh in Rhigh_list:
    Gmin = 1/Rhigh
    Gmax = 1/Rlow_base
    pot, dep, ratio, inl_pot, inl_dep = generar_curvas_pot_dep(
        pulsos_base, pulsos_base, a_base, a_base, Gmin, Gmax, 'neg', 'neg')
    sw = sigma_W_from_curves(pot, dep)
    pred = beta_analytic(784, sw['sigma_W'], vs784['V_rms'], kappa=3.0)
    res = run_repeated(X_full, y_full, 784, pot, dep, BETA_MAX_C, POINTS_C, n_repeats=N_REPEATS)
    res.update(Rhigh=Rhigh, ratio=ratio, sigma_W=sw['sigma_W'], V_rms=vs784['V_rms'],
                beta_pred_kappa3=pred, INL=inl_pot)
    print(Rhigh, res, "elapsed", time.time() - t0)
    expC.append(res)

out = dict(full_vstats=vs784, base_curve=dict(INL=inl_pot_b, sigma_W=sw_base['sigma_W']),
           grids=dict(A=[BETA_MAX_A, POINTS_A], B=[BETA_MAX_B, POINTS_B], C=[BETA_MAX_C, POINTS_C]),
           n_repeats=N_REPEATS, expA=expA, expB=expB, expC=expC)
with open(os.path.join(OUT, "beta_results_v2.json"), "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved to beta_results_v2.json. Total elapsed:", time.time() - t0)
