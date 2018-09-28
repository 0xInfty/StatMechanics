# -*- coding: utf-8 -*-
"""
This script includes some examples on how to use 'markov' module.

Some resolved exercises:
   Ej. 1: Problema con estados absorventes del borracho discreto

@author: 0xInfty
@date: 09/2018
"""

import numpy as np
import markov as mkv

#%% Ej. 1: Problema con estados absorventes del borracho discreto

"""Problema del borracho discreto con 3 pasos intermedios.

Cuatro cuadras separan de un bar la casa de un hombre.

casa____1_____2_____3_____bar

Si el hombre se encuentra en cualquiera de las tres esquinas 
intermedias, se dirigirá hacia su casa o hacia el bar con igual 
probabilidad. Pero si llega, se quedará en uno de los dos sitios.

a) Encuentre la matriz  de transición del problema. Exprésela de forma
canónica.
b) ¿Cuánto tiempo caminará el hombre antes de detenerse?
c) ¿Con qué probabilidad terminará en el bar?

"""

M0 = np.zeros([5,5]) # transition matrix (0 is home, 4 is the bar)
for i in [0,4]:
    M0[i,i] = 1
for i in [1,2,3]:
    M0[i-1,i] = 0.5
    M0[i+1,i] = 0.5

Mc = np.zeros([5,5]) # canonical matrix
Q = .5 * (np.diag(np.ones(2), -1) + np.diag(np.ones(2), 1))
R = np.zeros([2,3]) # 2 absorvant states and 3 transitory states
R[0,0] = .5
R[1,2] = .5
Mc[:3,:3] = Q # the right upper part of Mc is Q
Mc[3:,:3] = R # the right lower part of Mc is R
Mc[3:,3:] = np.diag(np.ones(2)) # the left lower part of Mc is 1
  
N, RN = mkv.canonic_solver(Q, R) # apply the N = (1-Q)^(-1) method

ans_b = [mkv.get_canonic_data(N, RN, "ab", 1),
         mkv.get_canonic_data(N, RN, "ab", 2),
         mkv.get_canonic_data(N, RN, "ab", 3)]


  
#%% Ej.2: Resolver un problema más grande de estados absorventes.

M = np.diag(np.ones(8),-1)*.6 + .4*np.diag(np.ones(8),1)
M[:,0] = np.zeros(9)
M[:,8] = np.zeros(9)
M[0,0] = 1
M[8,8] = 1
 
Mc = np.zeros([9,9])
Mc[0:6,0:6] = np.diag(np.)