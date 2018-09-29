# -*- coding: utf-8 -*-
"""
This script includes some examples on how to use 'markov' module.

Some solved exercises:

Ex.1: 
    3-steps drunk man's absorvant states problem.
Ex.2:
    Prisoner's 7-steps bet absorvant states problem.
   
See also 'markov.py' on https://github.com/0xInfty/StatMechanics.git

@author: 0xInfty
@date: 09/2018
"""

import numpy as np
import markov as mkv

#%% Ex.1: 3-steps drunk man's absorvant states problem

"""Drunk man's absorvant states problem (3 mid steps).

There are four blocks between the bar and the drunk man's home.

home____1_____2_____3_____bar

If the man is at any of the three middle corners, he will walk towards 
the bar or his home with the same probability. But if he arrives at 
either the bar or his home, he will stay there.

a) Find the transition matrix and its associated canonic matrix.
b) How much will the man walk befor he stops?
c) What is the probability that he will end at the bar?

"""

M = np.zeros([5,5]) # transition matrix (0 is home, 4 is the bar)
for i in [0,4]:
    M[i,i] = 1
for i in [1,2,3]:
    M[i-1,i] = 0.5
    M[i+1,i] = 0.5


Mc, ab = mkv.canonic_finder(M) # canonic matrix with ab absorvant states
Q, R = mkv.canonic_decomposer(Mc, ab) # QR decomposition
N, RN = mkv.canonic_solver(Q, R) # apply the N = (1-Q)^(-1) method

print("\n")
ans_b = [mkv.canonic_data(N, RN, "abs", 1),
         mkv.canonic_data(N, RN, "abs", 2),
         mkv.canonic_data(N, RN, "abs", 3)]

print("\n")
ans_c = [mkv.canonic_data(N, RN, "pr", 1, 2),
         mkv.canonic_data(N, RN, "pr", 2, 2),
         mkv.canonic_data(N, RN, "pr", 3, 2)]

  
#%% Ex.2: Prisoner's 7-steps bet absorvant states problem.

M = np.diag(np.ones(8),-1)*.6 + .4*np.diag(np.ones(8),1)
M[:,0] = np.zeros(9)
M[:,8] = np.zeros(9)
M[0,0] = 1
M[8,8] = 1
