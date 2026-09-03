#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 18:36:49 2026

@author: walter
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.model_selection import KFold


class MemDNN(nn.Module):
    """
    Red feedforward multicapa con crossbars memristivos diferenciales.
    - Sin bias
    - G ∈ (D_in, 2*D_out)
    - W_eff = G+ - G-
    - Beta por capa: acepta un escalar (se replica en todas las capas, igual
      que la version original) o una lista/tupla/array con un beta distinto
      por capa (largo == len(sizes) - 1).
    """

    def __init__(self,
                 sizes,
                 beta,
                 pot,
                 dep,
                 fixed,
                 G0_distribution='random',
                 device='cpu'):
        """
        sizes: [D0, D1, D2, ..., DL]
        beta: escalar (mismo beta para todas las capas) o lista/array con un
              beta por capa (largo == len(sizes) - 1)
        """
        super().__init__()

        self.fixed = fixed
        self.device = device
        self.pot = pot
        self.dep = dep
        self.G_layers = nn.ParameterList()
        self.sizes = sizes
        self.beta = self._expand_beta(beta, len(sizes) - 1)

        for i in range(len(sizes) - 1):
            D_in = sizes[i]
            D_out = sizes[i + 1]

            G = G0_initialization(
                G0_distribution,
                pot, dep,
                D_in, D_out,
                device=device
            )

            self.G_layers.append(nn.Parameter(G.float()))


    # -------------------------------------------------------------------------

    @staticmethod
    def _expand_beta(beta, n_layers):
        """
        Normaliza `beta` a una lista de floats de largo `n_layers` (una
        entrada por capa).

        - Si `beta` es un escalar (int/float): se replica para todas las
          capas (comportamiento equivalente al beta global original).
        - Si `beta` es una lista/tupla/np.ndarray/torch.Tensor: se usa un
          valor por capa, debe tener largo == n_layers.
        """
        if isinstance(beta, (list, tuple, np.ndarray, torch.Tensor)):
            beta_list = [float(b) for b in beta]
            if len(beta_list) != n_layers:
                raise ValueError(
                    f"beta tiene largo {len(beta_list)} pero el modelo tiene "
                    f"{n_layers} capas. Pasar un escalar (mismo beta para "
                    f"todas las capas) o una lista de largo {n_layers}."
                )
            return beta_list
        else:
            return [float(beta)] * n_layers


    # -------------------------------------------------------------------------

    def forward(self, x):
        """
        x: (batch, D0)
        """
        h = x

        for i, G in enumerate(self.G_layers):

            # Pesos diferenciales
            G_pos = G[:, 0::2]
            G_neg = G[:, 1::2]
            W_eff = G_pos - G_neg

            I = h @ W_eff

            if i < len(self.G_layers) - 1:
                h = F.relu(self.beta[i] * I)
            else:
                # logits
                h = self.beta[i] * I

        return h
    
    
    # -------------------------------------------------------------------------
    
    
    
    def training_step(self,
                         x, y,
                         lr,
                         loss_fn,
                         delta_t_forward,
                         delta_t_pulse,
                         Vr, Vs):
   
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
   
           for i, W in enumerate(W_list):
               energy_forward += torch.sum(
                   delta_t_forward * (h.to(torch.float64) ** 2) @ self.G_layers[i].to(torch.float64)
               )
   
               I = h @ W
               if i < len(W_list) - 1:
                   h = F.relu(self.beta[i] * I)
               else:
                   logits = self.beta[i] * I
   
           # -------- Loss --------
           loss = loss_fn(logits, y)
           loss.backward()
   
           # -------- Physical update --------
           energy_pulse = 0.0
   
           with torch.no_grad():
               for i, W in enumerate(W_list):
                   dW = -lr * W.grad
   
                   D_in, D_out = dW.shape
   
                   G_new, E = Nmanhattan_vectorizado(
                       dW,
                       self.G_layers[i],
                       self.pot, self.dep,
                       D_in, D_out,
                       delta_t_pulse,
                       Vr, Vs,self.fixed 
                   )
   
                   self.G_layers[i].data = G_new
                   energy_pulse += torch.sum(E)
   
                   W.grad.zero_()
   
           return loss.item(), energy_forward.item(), energy_pulse.item()
   
    
   
    # -------------------------------------------------------------------------

    
    # --------------------------------------------------
    # Train full epoch
    # --------------------------------------------------
    def train_epoch(self,
                    train_loader,
                    lr,
                    loss_fn,
                    delta_t_forward,
                    delta_t_pulse,
                    Vr, Vs):

        self.train()

        total_loss = 0.0
        total_energy_forward = 0.0
        total_energy_pulse = 0.0

        for x, y in train_loader:
            l, ef, ep = self.training_step(
                x, y,
                lr,
                loss_fn,
                delta_t_forward,
                delta_t_pulse,
                Vr, Vs
            )

            total_loss += l
            total_energy_forward += ef
            total_energy_pulse += ep

        return total_loss, total_energy_forward, total_energy_pulse


    # -------------------------------------------------------------------------

    def save_G(self, fold , epoch, folder="checkpoints"):
        os.makedirs(folder, exist_ok=True)
    
        for i, G in enumerate(self.G_layers):
            filename = f"fold_{fold}_G_layer{i}_epoch{epoch}.npy"
            path = os.path.join(folder, filename)
    
            np.save(path, G.detach().cpu().numpy())


    # -------------------------------------------------------------------------
    
    @torch.no_grad()
    def evaluate_model(self, val_loader, loss_fn):
        self.eval()
    
        correct = 0
        total = 0
        val_loss = 0.0
    
        for x, y in val_loader:
            x = x.to(self.device)
            y = y.to(self.device)
    
            logits = self(x)
            loss = loss_fn(logits, y)
    
            val_loss += loss.item() * x.size(0)
    
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    
        val_loss /= total
        accuracy = correct / total   # ← ENTRE 0 y 1
    
        return accuracy, val_loss
    # -------------------------------------------------------------------------

    def search_best_beta(
        self,
        train_loaders,   # lista de train_loader (uno por fold)
        val_loaders,     # lista de val_loader   (uno por fold)
        loss_fn,
        a: float,
        lr,
        delta_t_forward,
        delta_t_pulse,
        Vr, Vs,
        beta_max: float = 400,
        points: int = 20,
        std_range=(2.5, 3.5)
    ):
        """
        Busca el mejor beta usando K-fold cross-validation.

        Para cada beta:
          - se entrena desde el MISMO estado base
          - se evalúa en todos los folds
          - se usa el accuracy promedio

        NOTA: esta búsqueda sigue barriendo un único beta GLOBAL (el mismo
        valor se replica en todas las capas vía `_expand_beta`), igual que la
        version original. No optimiza betas distintos por capa.
        """
    
        assert len(train_loaders) == len(val_loaders)
        n_folds = len(train_loaders)
    
        # -----------------------------
        # Estado base (modelo sin entrenar)
        # -----------------------------
        base_state = self._save_state()
    
        # ============================================================
        # CASO: criterio por ACCURACY
        # ============================================================
        if a <= 10000:
            betas = np.linspace(1, beta_max, points)

            best_beta = self.beta[0]
            best_acc_mean = -np.inf
    
            for beta in betas:
                print(f"\n[a={a}] Probando beta = {beta}")
    
                fold_accs = []
    
                for k in range(n_folds):
                    print(f"  Fold {k+1}/{n_folds}")
                    # ======================================================
                    #CREAR MODELO NUEVO SIEMPRE (G0_distribution = random)
                    # ======================================================
                    model = MemDNN(
                        sizes=self.sizes,
                        beta=float(beta),
                        pot=self.pot,
                        dep=self.dep,
                        G0_distribution='random',   # ← FORZADO
                        fixed=self.fixed,
                        device=self.device
                    )

    
                    # Restaurar modelo base SIEMPRE
                    # self._restore_state(base_state)
                    # self.beta = float(beta)
    
                    # -------- Entrenar 1 epoch en este fold --------
                    model.train()
                    model.train_epoch(
                        train_loaders[k],
                        lr,
                        loss_fn,
                        delta_t_forward,
                        delta_t_pulse,
                        Vr, Vs
                    )
    
                    # -------- Evaluar en validación del fold --------
                    acc, _ = model.evaluate_model(val_loaders[k], loss_fn)
                    fold_accs.append(acc)
    
                    print(f"    accuracy fold = {acc}")
    
                acc_mean = np.mean(fold_accs)
                acc_std  = np.std(fold_accs)
    
                print(
                    f"  beta = {beta} → "
                    f"acc_mean = {acc_mean:.4f}, acc_std = {acc_std:.4f}"
                )
    
                if acc_mean > best_acc_mean:
                    best_acc_mean = acc_mean
                    best_beta = beta
    
            # Fijar mejor beta y restaurar estado base
            # self._restore_state(base_state)
            self.beta = self._expand_beta(best_beta, len(self.G_layers))

            return best_beta, best_acc_mean


    # -------------------------------------------------------------------------
    
    def _save_state(self):
        return {
            "beta": list(self.beta),
            "G": [G.detach().clone() for G in self.G_layers]
        }

    # -------------------------------------------------------------------------

    def _restore_state(self, state):
        self.beta = list(state["beta"])
        for G, G0 in zip(self.G_layers, state["G"]):
            G.data.copy_(G0)


############################################################################################
############################################################################################


from typing import Tuple
import torch

@torch.jit.script
def Spulse_tensor_vectorizado(
    g: torch.Tensor,
    pot: torch.Tensor,
    dep: torch.Tensor,
    delta_t_pulse: float,
    Vs: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    
    
    g = g.float()
    pot = pot.float()
    dep = dep.float()
    g0 = g.clone()
    g_new = g.clone()

    pot_match = (pot.view(1, -1) == g.view(-1, 1))
    in_pot = pot_match.any(dim=1)

    pot_indices = pot_match.float().argmax(dim=1)
    next_pot_index = pot_indices + 1
    next_pot_index = torch.where(
        next_pot_index < pot.size(0),
        next_pot_index,
        pot_indices
    )
    g_new[in_pot] = pot[next_pot_index[in_pot]]

    not_in_pot = ~in_pot
    g_np = g[not_in_pot]

    mask = pot.view(1, -1) > g_np.view(-1, 1)
    any_match = mask.any(dim=1)
    idx = mask.float().argmax(dim=1)
    updated = pot[idx]
    g_np_new = torch.where(any_match, updated, dep[0])

    g_new[not_in_pot] = g_np_new

    energy = 0.5 * delta_t_pulse * (g0 * Vs**2 + g_new * Vs**2)
    return g_new, energy

############################################################################################
############################################################################################

@torch.jit.script
def Rpulse_tensor_vectorizado(
    g: torch.Tensor,
    pot: torch.Tensor,
    dep: torch.Tensor,
    delta_t_pulse: float,
    Vr: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    
    g = g.float()
    pot = pot.float()
    dep = dep.float()
    g0 = g.clone()
    g_new = g.clone()

    dep_match = (dep.view(1, -1) == g.view(-1, 1))
    in_dep = dep_match.any(dim=1)

    dep_indices = dep_match.float().argmax(dim=1)
    next_dep_index = dep_indices + 1
    next_dep_index = torch.where(
        next_dep_index < dep.size(0),
        next_dep_index,
        dep_indices
    )
    g_new[in_dep] = dep[next_dep_index[in_dep]]

    not_in_dep = ~in_dep
    g_nd = g[not_in_dep]

    mask = dep.view(1, -1) < g_nd.view(-1, 1)
    any_match = mask.any(dim=1)
    idx = mask.float().argmax(dim=1)
    updated = dep[idx]
    g_nd_new = torch.where(any_match, updated, dep[0])

    g_new[not_in_dep] = g_nd_new

    energy = 0.5 * delta_t_pulse * (g0 * Vr**2 + g_new * Vr**2)
    return g_new, energy


############################################################################################
############################################################################################


@torch.jit.script
def Nmanhattan_vectorizado(
    d_w: torch.Tensor,
    G: torch.Tensor,
    pot: torch.Tensor,
    dep: torch.Tensor,
    D_in: int,
    D_out: int,
    delta_t_pulse: float,
    Vr: float,
    Vs: float,
    fixed: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    
    
    
    d_w = d_w.reshape(-1).float()
    G = G.reshape(-1).float()
    pot = pot.float()
    dep = dep.float()

    E = torch.zeros_like(G)
    
    idx_all = torch.arange(d_w.size(0), device=d_w.device)

    # Parte positiva
    pos_mask = d_w >= 0
    idx0 = (2 * idx_all)[pos_mask]       # pares
    idx1 = idx0 + 1                      # impares

    G0, E0 = Spulse_tensor_vectorizado(G[idx0], pot, dep, delta_t_pulse, Vs)
    if fixed:
        G1 = G[idx1].clone()
        E1 = torch.zeros_like(G1)
    else:
        G1, E1 = Rpulse_tensor_vectorizado(G[idx1], pot, dep, delta_t_pulse, Vr)

    G[idx0] = G0
    G[idx1] = G1
    E[idx0] = E0
    E[idx1] = E1

    # Parte negativa
    neg_mask = d_w < 0
    idx0 = (2 * idx_all)[neg_mask]       # pares
    idx1 = idx0 + 1                      # impares

    G0, E0 = Rpulse_tensor_vectorizado(G[idx0], pot, dep, delta_t_pulse, Vr)
    if fixed:
        G1 = G[idx1].clone()
        E1 = torch.zeros_like(G1)
    else:
        G1, E1 = Spulse_tensor_vectorizado(G[idx1], pot, dep, delta_t_pulse, Vs)

    G[idx0] = G0
    G[idx1] = G1
    E[idx0] = E0
    E[idx1] = E1

    return G.reshape(D_in, 2 * D_out), E.reshape(D_in, 2 * D_out)

############################################################################################
############################################################################################



def G0_initialization(distribution, pot, dep, D_in, D_out, device="cpu"):
    if distribution == 'random':
        # Concatenate `pot` and `dep`, then sort
        conductancia = torch.sort(torch.cat((pot, dep)))[0]

        # Randomly select indices for `conductancia`
        indexes = torch.randint(0, len(conductancia), (D_in, 2 * D_out), device=device)

        # Use advanced indexing to create G based on the random indexes
        G = conductancia[indexes].to(device)

    elif distribution == 'HRS':
        # Set G to be filled with the first value of `pot` in the desired shape
        G = torch.full((D_in, 2 * D_out), pot[0], device=device, dtype=torch.float64)

    elif distribution == 'LRS':
        # Set G to be filled with the last value of `pot` in the desired shape
        G = torch.full((D_in, 2 * D_out), pot[-1], device=device, dtype=torch.float64)

    elif distribution == 'Gfix_random':
        # Concatenate `pot` and `dep`, then sort
        conductancia = torch.sort(torch.cat((pot, dep)))[0]

        # Randomly select indices for `conductancia`
        indexes = torch.randint(0, len(conductancia), (D_in, 2 * D_out), device=device)

        # Use advanced indexing to create G based on the random indexes
        G = conductancia[indexes].to(device)
        for i in range(D_out):
            G[:,2*i+1] = torch.mean(pot )
        
    return G


############################################################################################
############################################################################################

def longitud_curva (pulsos  , curve     ) :
    '''La curva y la lista de pulsos que tomo tiene que estar normalizadsa'''
    d_curve = 0
    for i in range(len (curve) - 1):
       d_curve = d_curve + np.sqrt (  (pulsos[i+1] - pulsos[i] )**2 + (curve[i+1] - curve[i])**2 )
    return d_curve

############################################################################################
############################################################################################

def calcular_indice_nl (curve):
    ''' Toma una curva de pot/dep, calcula la cantidad de puslsos, normaliza ambas 
    cantidades para que la curva quede en el cuadrado (0,0) (0,1) (1,1) (1,0) y calcula 
    el indice. Con el primer y segundo punto chequea si la curva es creciente o decreciente para
    saber si es pot o dep. RECORDAR: las curvas que genero son monótamente crecientes o decrecientes.'''
    
    if (curve[0] < curve[1]) == True: #esto es una curva de pot
        curve_norm = curve/curve[-1] 
        pulsos_norm = np.linspace(0 , 1, len (curve_norm))
        d = np.sqrt (  (pulsos_norm[-1] - pulsos_norm[0] )**2 + (curve_norm[-1] - curve_norm[0])**2 )
        #d = np.sqrt(2)
        L = longitud_curva (pulsos_norm , curve_norm   )
        return (L - d) /d
    else: # esto es una curve de dep
        curve_norm = curve/curve[0] 
        pulsos_norm = np.linspace(0 , 1, len (curve_norm))
        d = np.sqrt (  (pulsos_norm[-1] - pulsos_norm[0] )**2 + (curve_norm[-1] - curve_norm[0])**2 )
        #d = np.sqrt(2)
        L = longitud_curva (pulsos_norm , curve_norm   )
        return (L - d) /d


############################################################################################
############################################################################################

#def generar_curvas_pot_dep (  pulsos_pot : int, pulsos_dep : int, a_pot :float, a_dep :float , G_min : float , G_max :float = 0.001  ):
#    """
#    Función que genera y devuelve las curvas de pot, dep y ratio a partir de la cantidad de puntos, parametro a y
#    valores maximos y minimos para la conductancia (esto termina determinando el ratio HRS/LRS)

 #   """
 #   #G_max = 0.001 #este queda fijo porque tomo como Rlow 1kohm siempre
 #   "POT"
 #   ratio = round(G_max/G_min)
 #   p = np.arange(pulsos_pot)
 #   b = (G_max - G_min)/(1-np.exp(-pulsos_pot/a_pot))
 #   pot = b*(1 -np.exp(-p/a_pot)) + G_min
 #   inl_pot = calcular_indice_nl (pot)

  #  "DEP"
  #  dep = -b*(1 -np.exp((p-pulsos_dep)/a_dep)) + G_max #esto no me da valores decrecientes
  #  dep = dep[::-1].copy() #ahora es decreciente. uso copy() porque Torch da problemas (strides invertidos)
  #  inl_dep = calcular_indice_nl (dep)
  #  return pot, dep , round(ratio) , inl_pot, inl_dep



def generar_curvas_pot_dep (  pulsos_pot : int, pulsos_dep : int, 
                            a_pot :float, 
                            a_dep :float , 
                            G_min : float , 
                            G_max :float = 0.001, 
                            concavidad_dep : str = 'neg' , 
                            concavidad_pot : str = 'pos' ) :
    
    """
    Función que genera y devuelve las curvas de pot, dep y ratio a partir de la cantidad de puntos, parametro a y
    valores maximos y minimos para la conductancia (esto termina determinando el ratio HRS/LRS)

    """
    #G_max = 0.001 #este queda fijo porque tomo como Rlow 1kohm siempre
    "POT"
    ratio = round(G_max/G_min)
    pp = np.arange(pulsos_pot)
    b = (G_max - G_min)/(1-np.exp(-pulsos_pot/a_pot))
    pot = b*(1 -np.exp(-pp/a_pot)) + G_min
   

    "DEP"
    pd = np.arange(pulsos_dep)
    b = (G_max - G_min)/(1-np.exp(-pulsos_dep/a_dep))
    dep = b*(1 -np.exp(-pd/a_dep)) + G_min
    dep = dep[::-1].copy() #ahora es decreciente. uso copy() porque Torch da problemas (strides invertidos)

    if concavidad_pot == 'pos':
        inl_pot = calcular_indice_nl (pot)
        if concavidad_dep == 'neg':
            dep =( - (dep - (G_max - G_min)) + 2*G_min  )[::-1].copy()
            inl_dep = calcular_indice_nl (dep)
            return pot, dep , round(ratio) , inl_pot, inl_dep
        else: 
            inl_dep = calcular_indice_nl (dep)
            return pot, dep , round(ratio) , inl_pot, inl_dep
    
    else: 
        #concavidad potenciacion negativa
        pot =( - (pot - (G_max - G_min)) + 2*G_min  )[::-1].copy()
        inl_pot = calcular_indice_nl (pot)
        if concavidad_dep == 'neg':
            dep =( - (dep - (G_max - G_min)) + 2*G_min  )[::-1].copy()
            inl_dep = calcular_indice_nl (dep)
            return pot, dep , round(ratio)  , inl_pot, inl_dep
        else: 
            inl_dep = calcular_indice_nl (dep)
            return pot, dep , round(ratio)  , inl_pot, inl_dep

############################################################################################
############################################################################################


# def mapGnew(G, pot, dep):
#     #   pesos = model_new.G_layers[0].detach().cpu().numpy().flatten()
#     dimensiones_originales = G.shape
#     G = G.detach().cpu().numpy().flatten()
#     j = 0 
#     for i , gij in enumerate( G ) :
#         if np.sum(gij == pot.detach().cpu().numpy().flatten()) ==  0:
#             idx = torch.argmin(torch.abs(dep - gij ))
#             #print(f"Elemento viejo {gij}, elemento nuevo {dep[idx]}")
#             G[i] =  dep[idx]
#             j = j +1
#     print(f"{j} elementos cambiados en la matrix de pesos G")
#     G = torch.from_numpy(G)
#     return G.view(dimensiones_originales)



def mapGnew(G, pot, dep, tol=1e-8):

    shape = G.shape

    G_flat = G.view(-1)
    pot_flat = pot.view(-1)
    dep_flat = dep.view(-1)

    # -----------------------------------
    # máscara: qué valores NO están en pot
    # -----------------------------------
    diff_pot = torch.abs(G_flat.unsqueeze(1) - pot_flat.unsqueeze(0))
    in_pot = torch.any(diff_pot < tol, dim=1)

    mask = ~in_pot   # los que hay que mapear

    # -----------------------------------
    # nearest neighbor en dep
    # -----------------------------------
    diff_dep = torch.abs(G_flat.unsqueeze(1) - dep_flat.unsqueeze(0))
    idx = torch.argmin(diff_dep, dim=1)

    G_new_flat = G_flat.clone()
    G_new_flat[mask] = dep_flat[idx[mask]]

    print(f"{mask.sum().item()} elementos cambiados en la matriz de pesos G")

    return G_new_flat.view(shape)




############################################################################################
############################################################################################


def DataSetLoader(X_train, y_train, k_folds, batch_number, random_state=None):
    """
    random_state: semilla para el shuffle de KFold. Pasarla explicitamente
    (en vez de dejarla en None) permite generar particiones de folds
    reproducibles y DISTINTAS entre llamadas -por ejemplo, una particion
    distinta por serie- sin depender del estado global de numpy/torch.
    """

    mnist_dataset = TensorDataset(X_train, y_train)
    images_per_fold = len(X_train)/k_folds
    batch_size = int((len(X_train) - images_per_fold)/batch_number)
    batch_size_val = int(images_per_fold/5)
    # KFold from sklearn
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    
    
    train_loaders = []
    val_loaders   = []

    
    for fold, (train_idx, val_idx) in enumerate(kf.split(mnist_dataset)):
        train_subsampler = Subset(mnist_dataset, train_idx)
        val_subsampler = Subset(mnist_dataset, val_idx)
        
        #armo los dataloaders para aplicar batchs
        train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subsampler, batch_size=batch_size, shuffle=False)
            
        
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

    return    train_loaders ,  val_loaders


############################################################################################
############################################################################################





