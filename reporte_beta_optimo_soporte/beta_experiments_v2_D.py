#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment D: voltage-encoding-amplitude scan (tests the V_rms dependence of
beta_optimo directly, using the 'amplitud_imagen' parameter already present in
Crossbar_train.py). Same method/fixed-grid discipline as beta_experiments_v2.py.
"""
import sys, os, json, time
PROJECT = r"C:\Users\walte\Documents\pesos_sinapticos_crossbar"
OUT = r"C:\Users\walte\AppData\Local\Temp\claude\C--Users-walte-Documents-pesos-sinapticos-crossbar\1a906e48-0654-416f-a8f5-112296b0c6ec\scratchpad"
sys.path.insert(0, PROJECT)

import numpy as np
import torch
import torch.nn as nn
from MemCrossbarClass import generar_curvas_pot_dep, G0_initialization, MemDNN, DataSetLoader

X_full = np.load(os.path.join(PROJECT, "X_train_mnist.npy"))
y_full = np.load(os.path.join(PROJECT, "y_train_mnist.npy"))

def sigma_W_from_curves(pot, dep, n_samples=200000):
    pot_t = torch.tensor(pot, dtype=torch.float32)
    dep_t = torch.tensor(dep, dtype=torch.float32)
    G = G0_initialization('random', pot_t, dep_t, D_in=n_samples, D_out=1, device='cpu')
    W = (G[:, 0] - G[:, 1]).numpy()
    return dict(sigma_W=float(np.std(W)))

def beta_analytic(N, sigma_W, V_rms, kappa=3.0):
    return kappa / (np.sqrt(N) * sigma_W * V_rms)

def run_one(X_cols, y, N_in, pot, dep, beta_max, points, seed,
            n_subsample=3000, k_folds=3, batch_number=15,
            lr=1.0, dt=10e-9, Vr=-1.0, Vs=1.0):
    torch.manual_seed(seed); np.random.seed(seed)
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

def run_repeated(X_cols, y, N_in, pot, dep, beta_max, points, n_repeats=3, seed0=500):
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
pot_b, dep_b, ratio_b, inl_pot_b, inl_dep_b = generar_curvas_pot_dep(
    100, 100, 5.85, 5.85, Gmin_base, Gmax_base, 'neg', 'neg')
sw_base = sigma_W_from_curves(pot_b, dep_b)

print("\n=== Experiment D: voltage-amplitude scan (fixed common grid) ===")
amp_list = [0.25, 0.5, 1.0, 2.0, 4.0]
BETA_MAX_D, POINTS_D = 8000.0, 20
N_REPEATS = 3
expD = []
t0 = time.time()
for amp in amp_list:
    Xc = X_full * amp
    V_rms = float(np.sqrt(np.mean(Xc**2)))
    pred = beta_analytic(784, sw_base['sigma_W'], V_rms, kappa=3.0)
    res = run_repeated(Xc, y_full, 784, pot_b, dep_b, BETA_MAX_D, POINTS_D, n_repeats=N_REPEATS)
    res.update(amp=amp, sigma_W=sw_base['sigma_W'], V_rms=V_rms, beta_pred_kappa3=pred, INL=inl_pot_b)
    print(amp, res, "elapsed", time.time() - t0)
    expD.append(res)

with open(os.path.join(OUT, "beta_results_v2_D.json"), "w") as f:
    json.dump(dict(expD=expD, grid=[BETA_MAX_D, POINTS_D], n_repeats=N_REPEATS), f, indent=2)
print("Saved. elapsed", time.time() - t0)
