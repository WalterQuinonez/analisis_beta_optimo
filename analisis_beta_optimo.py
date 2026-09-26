#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 19:02:25 2026

@author: walter
"""

"""21/08/2026: Analizar todas las posbilidades del argumento 
de la funcion de activacion y de la funcion de activacion mismo 
no muestra nada definito para relacionar con un beta optimo. 
Hice los histogramas y trate de relacionar media, dispersión
y cantidad maxima de cuentas de los histogramas de x y softmax

El detalle es que estuve analisando termino a termino. Ahora 
voy a tratar de analizar las posibles sumas que pueden tomar los 
argumentos de la función de activación. 

"""


import numpy as np
from MemCrossbarClass_beta_por_capa import *
from matplotlib import pyplot  as plt
import torch.nn as nn




#parametros de las curvas
a_pot_list  = [5.85, 5.85  ]
a_dep_list = [5.85 , 5.85 ]
betas = [100 , 500 ]
G0_distrbtn='random'
pulsos_pot_list = [50, 50]
pulsos_dep_list = [50 , 50]
concavidad_pot = 'pos'
concavidad_dep = 'neg'
Rhigh = 10000
Rlow = 1000
Gmin = 1/Rhigh
Gmax = 1/Rlow


MUESTRAS = 1_000_000
CHUNK = 50_000
N_SUM = 28 * 28  # 784 terminos por corriente (una entrada por pixel)


amplitud_imagen  = 1
X_train = (np.load('X_train_mnist.npy')/1)*amplitud_imagen
y_train = np.load('y_train_mnist.npy')

X_train = torch.from_numpy(X_train).float()


fig, ax = plt.subplots()
fig, ax2 = plt.subplots()
fig, ax3 = plt.subplots()
fig, ax4 = plt.subplots()
fig, ax5 = plt.subplots()

for r in range(len (betas)):
    a_pot = a_pot_list[r]
    a_dep = a_dep_list[r]
    pulsos_pot = pulsos_pot_list[r]
    pulsos_dep = pulsos_dep_list[r]


    #genero curvas pot/dep
    pot,dep, ratio , inl_pot , inl_dep = generar_curvas_pot_dep (  
                                pulsos_pot, 
                                pulsos_dep, 
                                a_pot, 
                                a_dep, 
                                Gmin, 
                                Gmax, 
                                concavidad_dep, 
                                concavidad_pot)
    
    print(f"inl pot {inl_pot}")
    print(f"inl dep {inl_dep}")
    
    ax.plot(np.arange(len(pot)) , pot)
    ax.plot(np.arange(len(pot)) + len(pot) , dep)    

    wij = np.zeros([len(pot) , len(dep)])
    for i in range(len(pot)):
        wij[i] = pot[i] - dep
    
    
    voltajes = np.unique (X_train)

            
    x = np.multiply.outer(voltajes, wij.flatten()).ravel()        
            
    """wij tienen todas las combinaciones posibles de los pesos"""
    ax2.hist( wij.flatten(), 
              bins=50,
              alpha = 0.8,
              label = f'media {np.mean(wij.flatten())}')  
    ax2.set_xlabel(r'w_ij posibles')
    ax2.set_ylabel('counts')
    ax2.legend()
    
    
    """Histograma tienen todas las combinaciones posibles de V_j* W_ij"""
    ax3.hist( x, 
              bins=50,
              alpha = 0.8,
              label = f'media {np.mean(x)}')  
    ax3.set_xlabel(r'V_j * W_ij posibles')
    ax3.set_ylabel('counts')
    ax3.legend()
    
    

    
    corrientes = np.empty(MUESTRAS)
    for start in range(0, MUESTRAS, CHUNK):
        end = min(start + CHUNK, MUESTRAS)
        idx = np.random.randint(0, x.size, size=(end - start, N_SUM))
        corrientes[start:end] = x[idx].sum(axis=1)
           
    """corrientes lo construyo con terminos tomados al azar de todas los posibles V_j* W_ij
        en histograma"""
    ax4.hist(corrientes, 
              bins=50,
              alpha = 0.8,
              label = f'MC inocente beta {betas[r]} ')  
    ax4.set_xlabel(r'corrientes (suma random)')
    ax4.set_ylabel('counts')

    ax5.hist(betas[r] * corrientes, 
              bins=50,
              alpha = 0.8,
              label = f'MC inocente beta {betas[r]} ')  
    ax5.set_xlabel(r'beta * corrientes (suma random)')
    ax5.set_ylabel('counts')

    print(f"desviacon Wij posibles {np.std(wij.flatten())}")
    print(f"media corriente {np.mean(corrientes)}")
    print(f"std corriente {np.std(corrientes)}")
    print(f"beta optimo {np.abs(1/np.mean(corrientes))} ")
    print(f"beta optimo {np.abs(1/(np.std(corrientes)) + np.mean(corrientes) ) } ")
    print("#######################################################")
    print(f"RMS corriente {np.sqrt(np.mean(corrientes**2))}")
    print(f"otro beta optimo {1/np.sqrt(np.mean(corrientes**2))}")


    
    
    D_in = 28*28
    D_out = 10
    pot = torch.tensor(pot, dtype=torch.float32)
    dep = torch.tensor(dep, dtype=torch.float32)

    for k in range(5):
        G = G0_initialization('random', pot, dep, D_in, D_out, device="cpu")
        G_pos = G[:, 0::2]
        G_neg = G[:, 1::2]
        W_eff = G_pos - G_neg
        
        currents_MC = X_train  @ W_eff
        
        ax4.hist(currents_MC.flatten(), 
                  bins=50,
                  alpha = 0.8,
                  label = f'MC beta beta {betas[r]}  ') 
        ax4.legend() 
        
        ax5.hist(betas[r]* currents_MC.flatten(), 
                  bins=50,
                  alpha = 0.8,
                  label = f'MC beta beta {betas[r]}  ') 
        ax5.legend() 
    
        
    mean_current =  torch.zeros(10000) 
    for k in range(10000):
        print(k)
        G = G0_initialization('random', pot, dep, D_in, D_out, device="cpu")
        G_pos = G[:, 0::2]
        G_neg = G[:, 1::2]
        W_eff = G_pos - G_neg
        currents_MC = X_train  @ W_eff
        mean_current[k] = torch.mean(currents_MC)
        
    mean_current = mean_current.numpy()    
    
    
    fig, ax6 = plt.subplots()
    ax6.hist(mean_current, 
              bins=50,
              alpha = 0.8,
              label = f'MC beta {betas[r]} ') 
    ax6.legend() 
    ax6.set_xlabel('mean_current')
    ax6.set_ylabel('counts')
    
    ax6.hist(betas[r]*mean_current, 
              bins=50,
              alpha = 0.8,
              label = 'beta * MC beta {betas[r]}') 
    ax6.legend() 
    ax6.set_xlabel('mean_current')
    ax6.set_ylabel('counts')
    
    
    softmax_curr = torch.softmax(currents_MC, dim= 1) 
    softmax_beta = torch.softmax(betas[r] * currents_MC, dim= 1) 
    
    fig, ax7 = plt.subplots()
    ax7.hist(softmax_curr.flatten(), 
              bins=50,
              alpha = 0.8,
              label = f'softmax_current_MC beta {betas[r]}') 
    
    ax7.hist(softmax_beta.flatten(), 
              bins=50,
              alpha = 0.8,
              label = f'softmax_current*beta_MC beta {betas[r]}') 
    ax7.legend() 
    ax7.set_xlabel('softmax_current ')
    ax7.set_ylabel('counts')