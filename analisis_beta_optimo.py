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
from MemCrossbarClass import *
from matplotlib import pyplot  as plt
import torch.nn as nn




#parametros de las curvas
a_pot_list  = [48.83,98.55,197.99 ]
a_dep_list = [48.83,98.55,197.99 ]
betas = [250,350,750]
G0_distrbtn='random'
pulsos_pot_list = [50,100,200]
pulsos_dep_list = [50,100,200]
concavidad_pot = 'pos'
concavidad_dep = 'neg'
Rhigh = 10000
Rlow = 1000
Gmin = 1/Rhigh
Gmax = 1/Rlow




amplitud_imagen  = 1
X_train = np.load('X_train_mnist.npy')*amplitud_imagen
y_train = np.load('y_train_mnist.npy')




fig, ax2 = plt.subplots()
ax2.hist(X_train.flatten(), 
          bins=50,
          alpha = 0.8)  
ax2.set_xlabel(r'pixeles imagenes')
ax2.set_ylabel('counts')

fig, ax = plt.subplots()
fig, ax2 = plt.subplots()
fig, ax3 = plt.subplots()
fig, ax4 = plt.subplots()
fig, ax5 = plt.subplots()

for r in range(3):
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
    histograma = np.zeros([len(voltajes) , len(wij.flatten())])
    for i in range(len(voltajes)):
        for j in range(len(wij.flatten())):
            histograma[i, j] = voltajes[i]* (wij.flatten())[j]     
            
            
            
    """wij tienen todas las combinaciones posibles de los pesos"""
    ax2.hist(wij.flatten(), 
              bins=50,
              alpha = 0.8)  
    ax2.set_xlabel(r'w_ij posibles')
    ax2.set_ylabel('counts')
    ax2.legend()
    print(np.sqrt(np.mean(wij.flatten()**2)))
    
    
    """Histograma tienen todas las combinaciones posibles de V_j* W_ij"""
    ax3.hist(histograma.flatten(), 
              bins=50,
              alpha = 0.8)  
    ax3.set_xlabel(r'V_j * W_ij posibles')
    ax3.set_ylabel('counts')
    ax3.legend()
    print(np.sqrt(np.mean(histograma.flatten()**2)))
    
    
    
    
    muestras = 1000000
    
    mean_x =  []
    softmax = []
    softmax_min= []
    softmax_max= []
    
    x = histograma.flatten()
    corrientes = np.zeros(muestras)
    for k in range(muestras):   
        corrientes [k]= np.sum( np.random.choice(x, size=28*28, replace=True) )  
           
    """corrientes lo construyo con terminos tomados al azar de todas los posibles V_j* W_ij
        en histograma"""
    ax4.hist(corrientes, 
              bins=50,
              alpha = 0.8)  
    ax4.set_xlabel(r'corrientes (suma random)')
    ax4.set_ylabel('counts')

    ax5.hist(betas[r] * corrientes, 
              bins=50,
              alpha = 0.8)  
    ax4.set_xlabel(r'beta * corrientes (suma random)')
    ax4.set_ylabel('counts')


    print(f"desviacon Wij posibles {np.std(wij.flatten())}")
    print(f"media corriente {np.mean(corrientes)}")
    print(f"std corriente {np.std(corrientes)}")
    print(f"beta optimo {np.abs(1/np.mean(corrientes))} ")
    print(f"beta optimo {np.abs(1/(np.std(corrientes)) + np.mean(corrientes) ) } ")
    print("#######################################################")
    print(f"RMS corriente {np.sqrt(np.mean(corrientes**2))}")
    print(f"otro beta optimo {1/np.sqrt(np.mean(corrientes**2))}")




# betas =np.arange(1,2000,100)


# fig, ax2 = plt.subplots()
# corriente_rms = np.zeros(len(betas))
# for i , beta in enumerate (betas) : 
#     xx = beta*corrientes
#     ax2.hist(xx, 
#               bins=50,
#               alpha = 0.8, 
#               label= f'beta = {beta}')  
#     corriente_rms[i] = np.sqrt(np.mean(xx**2))


# ax2.set_xlabel(r'corriente (A)')
# ax2.set_ylabel('counts')
# ax2.legend()




# fig, ax2 = plt.subplots()
# salidas_rms = np.zeros(len(betas))
# for i , beta in enumerate (betas) :
#     salidas = np.exp(beta*corrientes)/sum(np.exp(beta*corrientes))
#     ax2.hist(salidas, 
#               bins=50,
#               alpha = 0.8, 
#               label= f'beta = {beta}')  
#     salidas_rms[i] =  np.sqrt(np.mean(salidas**2))

# ax2.set_xlabel(r'salidas')
# ax2.set_ylabel('counts')
# ax2.legend()





# fig, ax2 = plt.subplots()
# ax2.plot(betas , corriente_rms)
# ax2.set_xlabel("beta")
# ax2.set_ylabel("corriente_rms")

# fig, ax2 = plt.subplots()
# ax2.plot(betas , salidas_rms)
# ax2.set_xlabel("beta")
# ax2.set_ylabel("salidas_rms")


# fig, ax2 = plt.subplots()
# ax2.plot(betas[:-1] , np.diff(salidas_rms))
# ax2.set_xlabel("beta")
# ax2.set_ylabel("derivada salidas_rms")



# fig, ax2 = plt.subplots()
# ax2.plot(betas[:-2] , np.diff (np.diff(salidas_rms)))
# ax2.set_xlabel("beta")
# ax2.set_ylabel("d^2 salidas_rms")


# fig, ax2 = plt.subplots()
# ax2.plot(betas[:-3] , np.diff(np.diff (np.diff(salidas_rms))))
# ax2.set_xlabel("beta")
# ax2.set_ylabel("d^3 salidas_rms")

# fig, ax2 = plt.subplots()
# ax2.plot(betas,mean_x)
# ax2.set_xlabel("beta")       
# ax2.set_ylabel("mean_current")       


# fig, ax3 = plt.subplots()
# ax3.plot(betas,softmax, label = 'media')
# ax3.plot(betas,softmax_min , label = 'min')
# ax3.plot(betas,softmax_max , label = 'max')
# ax3.set_xlabel("beta")      
# ax3.set_ylabel("softmax_mean")     
# ax3.legend()



# betas = np.arange(1,3000,5)
# fig, ax = plt.subplots()
# #fig2, ax2 = plt.subplots()
# max_counts = []
# max_x= []
# min_x= []
# mean_x = []
# for beta in betas : 
#     print(beta)
#     x = beta * histograma.flatten()
#     softmax = np.exp(x)/sum(np.exp(x))
     
#     counts, bins, _ = ax.hist(
#         softmax,
#         bins=50,
#         alpha=0.8,
#         label=f'beta = {beta}'
#     )
    
#     max_counts.append(np.max(counts))
#     max_x.append(np.max(softmax))
#     min_x.append(np.min(softmax))    
#     mean_x.append(np.mean(softmax))
#     ax.set_xlabel(r'$\beta$ $V_j^k$ $W_{ij}$')
#     ax.set_ylabel('counts')
#     ax.legend()

# fig, ax = plt.subplots()
# ax.plot(betas,max_counts)
# ax.set_xlabel("beta")
# ax.set_ylabel("max counts x")


# fig, ax = plt.subplots()
# ax.plot(betas,min_x)
# ax.plot(betas,max_x)
# ax.plot(betas,mean_x)
# ax.set_xlabel("beta")


    
    # arg_maximo = beta * np.max(voltajes) * np.max(wij) * 28*28
    # arg_minimo = beta * np.max(voltajes) * np.min(wij) * 28*28
    # salida_maxima = np.exp(arg_maximo)/sum(np.exp(x))
    # salida_minima = np.exp(arg_minimo)/sum(np.exp(x))
    
    # print(f'--------- Beta = {beta} ---------')
    # print(f'argMax {arg_maximo} --- salidaMax {salida_maxima}') 
    # print(f'argMin {arg_minimo} --- salidaMin {salida_minima}') 


    
    
    # muestras = 100000
    # suma = np.zeros(muestras)
    # for k in range(muestras):   
    #     suma [k]= np.sum( np.random.choice(x, size=28*28, replace=True) )  
       
    
    # ax2.hist(suma, 
    #           bins=50,
    #           alpha = 0.8, 
    #           label= f'beta = {beta}')  
    
    # ax2.set_xlabel(r'argumento softmax')
    # ax2.set_ylabel('counts')
    # ax2.legend()
       
    
    # fig3, ax3 = plt.subplots()

    # salidas = np.exp(suma)/sum(np.exp(suma))
    # ax3.hist(salidas, 
    #           bins=100,
    #           alpha = 0.8, 
    #           label= f'beta = {beta}')  
    
    # ax3.set_xlabel(r'Salida_i')
    # ax3.set_ylabel('counts')
    # ax3.legend()
    
    
   
    
   
    
   
    
   
    
   
    




# betas = [1,10,100,150,200,250,260,270,280,290,300]
# valores_unicos = np.zeros_like(betas)    
# for  k , beta  in enumerate(betas): 
#     x = beta * histograma.flatten()
#     softmax = np.exp(x)/sum(np.exp(x))
#     softmax_3= round_sig(softmax, sig=3)
#     valores_unicos[k] = len(np.unique(softmax_3))    
    
# fig, ax = plt.subplots()    
# ax.plot(betas , valores_unicos)



# # beta  = 191.95959595959596
# beta = 100
# fig, ax = plt.subplots()
# plt.hist(beta * histograma.flatten(), bins=50,alpha = 0.8)  # arguments are passed to np.histogram
# ax.set_xlabel('valores argumento activacion')
# ax.set_ylabel('counts')
# plt.legend()




# np.max (np.unique(beta * histograma.flatten()) ) * 28*28

