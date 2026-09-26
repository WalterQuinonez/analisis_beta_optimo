#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`MemDNNTanh`: subclase de `MemDNN` (de `MemCrossbarClass_beta_por_capa.py`,
raíz del repo) que reemplaza la ReLU de las capas ocultas por tangente
hiperbólica. Verifica numéricamente la Sección 5 ("Extensión analítica") de
`reporte_beta_capa_oculta.pdf`: la predicción de que, a diferencia de ReLU,
con tanh SÍ debería importar cómo se reparte beta entre capas (no sólo el
producto), sobre todo cuando beta_oculta satura la tanh.

NO SE MODIFICA `MemCrossbarClass_beta_por_capa.py`: esta subclase sólo
sobreescribe `forward` y `training_step` para cambiar la no linealidad de
las capas ocultas (ReLU -> tanh; la capa de salida sigue siendo lineal en
ambos casos). Todo lo demás (G0_initialization, la regla física de
actualización `Nmanhattan_vectorizado`/`Spulse`/`Rpulse`, `evaluate_model`,
`search_best_beta_mejorado`, etc.) se hereda sin cambios de `MemDNN`.
"""

import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from MemCrossbarClass_beta_por_capa import MemDNN, Nmanhattan_vectorizado  # noqa: E402


class MemDNNTanh(MemDNN):
    """Igual a `MemDNN`, pero con `tanh` en vez de `ReLU` en las capas
    ocultas (la última capa sigue siendo lineal, `logits = beta[-1] * I`,
    igual que en `MemDNN`)."""

    def forward(self, x):
        h = x
        for i, G in enumerate(self.G_layers):
            G_pos = G[:, 0::2]
            G_neg = G[:, 1::2]
            W_eff = G_pos - G_neg
            I = h @ W_eff
            if i < len(self.G_layers) - 1:
                h = torch.tanh(self.beta[i] * I)
            else:
                h = self.beta[i] * I
        return h

    def training_step(self, x, y, lr, loss_fn, delta_t_forward, delta_t_pulse, Vr, Vs):
        x = x.to(self.device)
        y = y.to(self.device)

        # -------- Build W_eff list --------
        W_list = []
        for G in self.G_layers:
            G_pos = G[:, 0::2]
            G_neg = G[:, 1::2]
            W = (G_pos - G_neg).detach().clone()
            W.requires_grad_(True)
            W_list.append(W)

        # -------- Forward --------
        h = x
        energy_forward = 0.0
        logits = None

        for i, W in enumerate(W_list):
            energy_forward += torch.sum(
                delta_t_forward * (h.to(torch.float64) ** 2) @ self.G_layers[i].to(torch.float64)
            )

            I = h @ W
            if i < len(W_list) - 1:
                h = torch.tanh(self.beta[i] * I)
            else:
                logits = self.beta[i] * I

        # -------- Loss --------
        loss = loss_fn(logits, y)
        loss.backward()

        # -------- Physical update (identica a MemDNN.training_step) --------
        energy_pulse = 0.0
        with torch.no_grad():
            for i, W in enumerate(W_list):
                dW = -lr * W.grad
                D_in, D_out = dW.shape

                G_new, E = Nmanhattan_vectorizado(
                    dW, self.G_layers[i], self.pot, self.dep,
                    D_in, D_out, delta_t_pulse, Vr, Vs, self.fixed,
                )

                self.G_layers[i].data = G_new
                energy_pulse += torch.sum(E)
                W.grad.zero_()

        return loss.item(), energy_forward.item(), energy_pulse.item()
