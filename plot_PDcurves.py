# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 14:59:11 2026

@author: walte
"""



import numpy as np
from MemCrossbarClass_beta_por_capa import *
from matplotlib import pyplot  as plt
import torch.nn as nn




#parametros de las curvas
a_pot  =  496.05
a_dep =   496.05
G0_distrbtn='random'
pulsos_pot = 50
pulsos_dep = 50
concavidad_pot = 'pos'
concavidad_dep = 'neg'
Rhigh = 10000
Rlow = 1000
Gmin = 1/Rhigh
Gmax = 1/Rlow


MUESTRAS = 1_000_000
CHUNK = 50_000
N_SUM = 28 * 28  # 784 terminos por corriente (una entrada por pixel)




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

fig, ax = plt.subplots()

ax.plot(np.arange(len(pot)) , pot)
ax.plot(np.arange(len(pot)) + len(pot) , dep)    
ax.set_xlabel("pulses")
ax.set_ylabel("G (S)")
plt.show()

