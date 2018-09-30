# -*- coding: utf-8 -*-
"""
This script includes some examples on how to use 'markov' module.

Some solved exercises:

Ex.1
    3-steps drunk man's absorvant states problem.
Ex.2
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

"""A prisoner has 1 dolar at prison. If he had 8 dolars, he could pay 
his bail. A guard proposes a series of bets. If the inmate had A dolars,
he could earn A dolars with probability 0.4 or lose A dolars with 
probability 0.6.

Find the probability that he'll earn 8 dolars, if he follows one of the 
following two strategies:

a) he bets 1 dolar at a time;
b) he bets as many dolars as he can on each turn, but not more dolars 
that it would take to get 8 dolars.

Which of these is the best strategy? And if he had 3 dolars? And if 
other inmates said his last name is Smith? And if he had been born on a 
tuesday?

"""

print("\n1st strategy\n------------\n")

M = np.diag(np.ones(8),-1)*.6 + .4*np.diag(np.ones(8),1)
M[:,0] = np.zeros(9)
M[:,8] = np.zeros(9)
M[0,0] = 1
M[8,8] = 1
 
Mc, ab = mkv.canonic_finder(M)
Q, R = mkv.canonic_decomposer(Mc, ab)
N, NR = mkv.canonic_solver(Q, R)

# 2nd abosrvant state is 8
ans_8_usd_1st = mkv.canonic_data(N, NR, "prob", 1, 2)
ans_3_usd_1st = mkv.canonic_data(N, NR, "prob", 3, 2)
ans_1st =  np.array([mkv.canonic_data(N, NR, "prob", i, 2, False)
                     for i in range(1,8)])

print("\n2nd strategy\n------------\n")

M = np.zeros([9,9])
M[0,0] = 1
M[8,8] = 1        
for i in range(1, 8):
    if 2*i <= 8:
        M[2*i,i] = .4
    else:
        M[8,i] = .4
    if 2*i-8 > 0:
        M[2*i-8,i] = .6
    else:
        M[0,i] = .6

Mc = np.zeros([9,9])
Q = M[1:8,1:8]
R = np.array([M[0,1:8],M[8,1:8]])
Mc[:7,:7] = Q
Mc[7:,:7] = R
Mc[7:,7:] = np.diag(np.ones(2))

Mc, ab = mkv.canonic_finder(M)
Q, R = mkv.canonic_decomposer(Mc, ab)
N, NR = mkv.canonic_solver(Q, R)

ans_8_usd_2nd = mkv.canonic_data(N, NR, "prob", 1, 2)
ans_3_usd_2nd = mkv.canonic_data(N, NR, "prob", 3, 2)
ans_2nd = np.array([mkv.canonic_data(N, NR, "prob", i, 2, False)
                    for i in range(1,8)])

print("\nResults\n--------\n")

for i in range(7):
    if ans_1st[i] > ans_2nd[i]:
        message = 'the 1st strategy is better'
    elif ans_2nd[i] > ans_1st[i]:
        message = 'the 2nd strategy is better'
    else:
        message = 'both strategies are the same'
    print("If you have {} dolars, {}".format((i+1), message))