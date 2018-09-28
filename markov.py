# -*- coding: utf-8 -*-
"""
This module contains functions to solve Markov's probability problems.

Some of its useful functions are:
    
solve_canonic():
    Returns N and R.N from Q, R extracted from M canonical matrix.
get_canonic_data():
    Searches data on 'N', 'RN' of Markov's 'M' with absorvant states.

Some of examples:
    
1) Solve a Markov problem with absorvant states.


@author: 0xInfty
@date: 09/2018
"""

import numpy as np

#%% Auxiliar functions

def counting_suffix(number):
    
    """Returns a number's suffix string to use as counting.
    
    Parameters
    ----------
    number: int, float
        Any number, though it is designed to work with integers.
    
    Returns
    -------
    ans: str
        A string representing the integer number plus a suffix.
    
    Examples
    --------
    >> counting_sufix(1)
    '1st'
    >> counting_sufix(22)
    '22nd'
    >> counting_sufix(1.56)
    '2nd'
    
    """
    
    number = round(number)
    unit = int(str(number)[-1])
    
    if unit == 1:
        ans = '{}st'.format(number)
    if unit == 2:
        ans = '{}nd'.format(number)
    if unit == 3:
        ans = '{}rd'.format(number)
    else:
        ans = '{}th'.format(number)
    
    return ans
 
#%% Module corpus

def canonic_solver(Q, R):
    
    """Returns N and R.N from Q, R extracted from M canonical matrix.
    
    It takes the Q and R matrix from M canonical matrix of transition 
    probabilities. Then, it solves the Markov problem with absorvant 
    states. Because of that, it returns the matrix N = sum(Q^n, 0, inf) 
    using N = inv(1-Q) whose convergence is guarenteed by det(Q)<1. It 
    also returns RN = dot(R, N) matrix product.
    
    Parameters
    ----------
    Q: np.array with size (nt,nt)
        The square matrix of transition probabilities between transitory 
        states. It has size 'nt'x'nt', where nt is the number of 
        transitory states.
    R: np.array with size (na,nt)
        The non-square matrix of transition probabilities from 
        transitory states to absorvant states. It has size 'na'x'nt', 
        where 'na' is the number of absorvant states.
    
    Returns
    -------
    N: np.array with size (nt,nt)
        The square matrix of transition probabilites raised to N power 
        where N-->inf. It has size 'nt'x'nt', same as 'Q'. Its elements 
        'N[i,j]' represent the mean number of steps it takes to get to 
        'i' transitory state from 'j'th transitory state.
    RN: np.array with size (na,nt)
        The non-square matrix of probabilities whose elements 'NR[i,j]' 
        say the probability of getting to 'i' absorvant state from 'j' 
        transitory state.
    
    Raises
    ------
    TypeError("Q should be a matrix"):
        if Q is not a matrix.
    TypeError("Q should be a square matrix"):
        if Q is not a square matrix.
    ValueError("Q's determinat is D>=1"):
        if Q's determinant isn't less than one, because in that case the 
        method of N = inv(1-Q) is not correct.
    TypeError("R should be a matrix")
        if R is not a matrix.
    TypeError("R should have as many columns as Q"):
        if R dimensions are not what should be expected from Q's 
        dimensions.
    
    """
    
    
    # numpy.linalg.inv
    # numpy.linalg.det
    # numpy.dot
       
    try:
        m = len(Q[:,0])
    except IndexError:
        raise TypeError("Q should be a matrix")
    
    if m != len(Q[0,:]):
        raise TypeError("Q should be a square matrix")

    if np.linalg.det(Q) >= 1:
        raise ValueError("Q's determinat is D>=1")
        return
    
    try:
        if m != len(R[0,:]):
            raise TypeError("R should have as many columns as Q")
            return
    except IndexError:
        raise TypeError("R should be a matrix")
        return    
    
    N = np.linalg.inv(np.diag(np.ones(m))-Q)
    RN = np.dot(R, N)
    
    return N, RN


def get_canonic_data(N, RN, datastring, index_from_1, index_to_1=None):
    
    """Searches data on 'N', 'RN' of Markov's 'M' with absorvant states.
    
    This function takes 'N', 'RN' matrix of Markov's 'M' transition 
    probabilities' matrix with absorvant states, which can be 
    decomposed on 'Q', 'R' matrix and solved by a canonical method. 
    Then it returns a specific piece of information indicated by
    'datastring' using 'index_from_1' and 'index_to_1' as 1-indexed 
    states.
    
    When 'datastring' includes 'pr', this function returns the 
    probability of getting to 'index_to_1' absorvant state from 
    'index_from_1' transitory state.
    
    When 'index_from_1' includes 'tr', this function returns the mean 
    number of steps it takes to get to 'index_to_1' transitory state 
    from 'index_from_1' transitory state.
    
    Otherwise, if 'datastring' includes 'ab', this function returns the 
    mean number of steps it takes to get absorved from 'index_from_1' 
    transitory state.
    
    Parameters
    ----------
    N: np.array with size (nt,nt)
        The square matrix of transition probabilites raised to N power 
        where N-->inf. Its elements 'N[i,j]' represent the mean number 
        of steps it takes to get to 'i'th transitory state from 'j'th 
        transitory state.
    R: np.array with size (na,nt)
        The non-square matrix of probabilities whose elements 'NR[i,j]' 
        say the probability of getting to 'i' absorvant state from 'j' 
        transitory state.

    Returns
    -------
    ans: int, float
        The required piece of information.

    Raises
    ------
    TypeError("N should be a matrix"):
        if N is not a matrix.
    TypeError("N should be a square matrix"):
        if N is not a square matrix.
    TypeError("NR should be a matrix")
        if NR is not a matrix.
    TypeError("NR should have as many columns as N"):
        if NR dimensions are not what should be expected from N's 
        dimensions.
    
    """
    
    # have to count from 1 the transitory or absorvant states
    
    # N has dimensions TxT, same as Q
    # R has dimensions AXT
    
    try:
        m = len(N[:,0])
    except IndexError:
        raise TypeError("N should be a matrix")
    
    if m != len(N[0,:]):
        raise TypeError("N should be a square matrix")

    try:
        if m != len(RN[0,:]):
            raise TypeError("RN should have as many columns as N")
            return
    except IndexError:
        raise TypeError("RN should be a matrix")
        return   

    dic = {'tr': 'transitory steps',
           'pr': 'probability',
           'abs': 'absorvant steps'}
    
    for key, value in dic.items():
        if key in datastring:
            datastring = value
    
    if datastring == 'transitory steps':
        ans = N[index_from_1-1, index_to_1-1]
        print("It takes {:.2f} steps to get to {} transitory \
state starting at {} transitory state".format(
                      ans,
                      counting_suffix(index_to_1),
                      counting_suffix(index_from_1)))
    elif datastring == 'absorvant steps':
        ans = np.dot(np.ones(m),N)[index_from_1-1]
        print("It takes {:.2f} steps to get absorved from \
{} transitory state".format(
                      ans,
                      counting_suffix(index_from_1)))
    elif datastring == 'probability':
        ans = RN[index_to_1-1, index_from_1-1]
        print("Starting on {} transitory state, will get to \
{} absorvant state with {:.3f} probability ({:.0f}%)".format(
                      counting_suffix(index_to_1),
                      counting_suffix(index_from_1),
                      ans,
                      ans*100))
    
    return ans

    
    