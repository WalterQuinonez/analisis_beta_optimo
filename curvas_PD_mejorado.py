# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 20:45:58 2026

@author: walte
"""

import numpy as np
from matplotlib import pyplot as plt 
import os
############################################################################################
############################################################################################

def longitud_curva (pulsos  , curve     ) :
    '''La yurva y la lista de pulsos que tomo tiene que estar normalizadsa'''
    d_curve = 0
    for i in range(len (curve) - 1):
       d_curve = d_curve + np.sqrt (  (pulsos[i+1] - pulsos[i] )**2 + (curve[i+1] - curve[i])**2 )
    return d_curve



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
    
    



def construir_plano_corregido(pot, dep):
    """Grilla completa (2N)x(2N): ambos elementos del par diferencial pueden
    tomar cualquier valor de S = pot U dep, en cualquier orden.
    Devuelve tambien S para referencia.
    """
    S = np.concatenate([np.asarray(pot), np.asarray(dep)])
    Wij_full = S[:, None] - S[None, :]
    return Wij_full, S

    
################################################################################

pulsos_pot = 50
pulsos_dep = 50
a_pot = 2257.81
a_dep = 2257.81
Gmin = 1/10000
Gmax = 1/1000
concavidad_pot = 'pos'
concavidad_dep = 'neg'

pot, dep , rat , aa , cc = generar_curvas_pot_dep (  pulsos_pot , 
                                           pulsos_dep , 
                                           a_pot , 
                                           a_dep , 
                                           Gmin, 
                                           Gmax , 
                                           concavidad_dep,
                                           concavidad_pot  )

plt.figure(figsize=(8, 5))
plt.scatter(np.arange(0,pulsos_pot), pot )
plt.scatter (pulsos_pot + np.arange(0,pulsos_dep ) , dep )


Wij = np.zeros([pulsos_pot,pulsos_dep])
for i in range(pulsos_pot):
    Wij[i] = pot[i] - dep 


plt.figure(figsize=(8, 5))
plt.hist(Wij.flatten(), bins=50)    
plt.xlabel('Wij')
plt.ylabel('Cantidad de elementos')
plt.title('Distribución de los pesos Wij')

################################################################################
################################################################################

Wij_full, S = construir_plano_corregido(pot, dep)

# Todas las combinaciones posibles (G_neg, G_pos)
Gn_plano, Gp_plano = np.meshgrid(S, S)

epoch = 3

folder_resultados = os.getcwd() + "\\data_beta_grid_search_SP\\beta_grid_search_SP_INL_4.82e-6_p_50"

file = folder_resultados + \
    "\\G_history_beta_grid_search_SP_INL_4.82e-6_p_50_job31226072_beta00_b1_serie000.npz"

G_history = np.load(file)

    
for e in range(10):    
    G = G_history['G_layer0_history'][0][e]
    
    Gp = G[:, 0::2]
    Gn = G[:, 1::2]
    
    
    # ============================================================
    # PLOT
    # ============================================================
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Estados permitidos por las curvas pot/dep
    ax.scatter(
        Gn_plano.flatten(),
        Gp_plano.flatten(),
        s=8,
        c="gray",
        alpha=0.35,
        label="Estados posibles"
    )
    
    # Estados reales de la matriz G
    ax.scatter(
        Gn.flatten(),
        Gp.flatten(),
        s=10,
        c="red",
        alpha=0.5,
        label=f"Matriz G - epoch {epoch}"
    )
    
    ax.set_xlabel("G_neg")
    ax.set_ylabel("G_pos")
    
    ax.set_title(f"Estados de conductancia - epoch {e*10}")
    
    ax.legend()
    
    fig.tight_layout()
    plt.show()
    
    
    







# # Coordenadas de cada elemento
# X, Y = np.meshgrid(dep, pot)
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')

# scatter = ax.scatter(
#     X.flatten(),
#     Y.flatten(),
#     Wij.flatten(),
#     c=Wij.flatten(),
#     cmap='viridis',
#     s=0.5,
#     alpha=0.6
# )

# ax.set_xlabel('dep')
# ax.set_ylabel('pot')
# ax.set_zlabel('Wij')

# fig.colorbar(scatter, ax=ax, label='Wij')

# plt.show()



# epochs = 100
# Gp = np.zeros(epochs)
# Gm = np.zeros(epochs)
# salidas = 6





# for entradas in  np.arange(0,28*28,2): 
#     for i in range(epochs):
#         file  = folder_resultados + "\\G_history_beta_grid_search_SP_INL_4.82e-6_p_50_job31226072_beta00_b1_serie000.npz" 
#         G = np.load (file) 
        
        
        
        
#         Gp[i] = G[entradas,2*salidas]
#         Gm[i] = G[entradas ,2*salidas+1]

    





#     scatter = ax.scatter(
#         Gm[:-2].flatten(),
#         Gp[:-2].flatten(),
#         (Gp[:-2]- Gm[:-2]).flatten(),
#         #c=Wij.flatten(),
#         color='red',
#         s = 1, 
#         alpha=0.5
#     )
    
    
#     scatter = ax.scatter(
#         Gm[-1].flatten(),
#         Gp[-1].flatten(),
#         (Gp[-1]- Gm[-1]).flatten(),
#         #c=Wij.flatten(),
#         color='blue',
#         s = 10, 
#         alpha=1
#     )
        
        
    




# tolerancia = 1e-8
# for k in range(epochs):     
#     print(np.any(np.isclose(Gp[k], pot, atol=tolerancia)) , np.any(np.isclose(Gp[k], dep, atol=tolerancia))   )
    
#     idxp = np.where(np.isclose(Gp[k], pot, atol=tolerancia))[0]
#     idxd = np.where(np.isclose(Gp[k], dep, atol=tolerancia))[0]

#     print(idxp , idxd)
    

# plt.figure(figsize=(8, 5))

# plt.plot(Gp)
# plt.plot(Gm)    
# plt.plot(Gp- Gm)





# # Coordenadas de cada elemento
# X, Y = np.meshgrid(dep, pot)
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')

# scatter = ax.scatter(
#     X.flatten(),
#     Y.flatten(),
#     Wij.flatten(),
#     color = 'gray',  
#     s=0.5,
#     alpha=0.5
# )

# ax.set_xlabel('dep')
# ax.set_ylabel('pot')
# ax.set_zlabel('Wij')

# fig.colorbar(scatter, ax=ax, label='Wij')

# plt.show()



# Gp = np.zeros(epochs)
# Gm = np.zeros(epochs)
# colors = ['red', 'green', 'blue' , 'orange' , 'purple', 'gray']
# for salidas in range(6):
#     for entradas in np.arange(0,28*28,2):    
#         file = folder_resultados+"/"+f"fold_0_G_layer0_epoch{epochs}.npy" 
#         G = np.load (file) 
#         Gp = G[entradas,2*salidas]
#         Gm = G[entradas,2*salidas+1]
        
        
#         scatter = ax.scatter(
#             Gm.flatten(),
#             Gp.flatten(),
#             (Gp- Gm).flatten(),
#             #c=Wij.flatten(),
#             color= colors[salidas],
#             s = 10, 
#             alpha=1
#         )
    


# ################################################################################

# # Coordenadas de cada elemento
# X, Y = np.meshgrid(dep, pot)
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')

# betas = [1 , 100]
# for beta in betas: 
#     Wij_scaleado = Wij*beta 
#     scatter = ax.scatter(
#         X.flatten(),
#         Y.flatten(),
#         Wij_scaleado.flatten(),
#         c=Wij_scaleado.flatten(),
#         cmap='viridis',
#         s=5,
#         alpha=0.6
#     )
    
#     ax.set_xlabel('dep')
#     ax.set_ylabel('pot')
#     ax.set_zlabel('Wij')
    
# fig.colorbar(scatter, ax=ax, label='Wij')

# plt.show()
################################################################################





