# -*- coding: utf-8 -*-
"""
This module contains functions to solve Markov's probability problems.

Some of its useful functions are:

canonic_finder()
    Finds 'Mc' canonic matrix from 'M' transition probabilities matrix.
canonic_decomposer()
    Returns 'Q, R' matrix from 'Mc' Markov canonic transition matrix.
canonic_solver()
    Returns 'N' and 'R.N' from 'Q, R' partial Markov transition matrix.
canonic_data()
    Searches data on 'N', 'RN' of Markov's 'M' with absorvant states.

There are some examples of its use available at 'markov_ex.py' from 
https://github.com/0xInfty/StatMechanics.git

@author: 0xInfty
@date: 09/2018
"""

import numpy as np

#%% Auxiliar functions

def counting_suffix(number):
    
    """Returns a number's suffix string to use as counting.
    
    Parameters
    ----------
    number : int, float
        Any number, though it is designed to work with integers.
    
    Returns
    -------
    ans : str
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
    elif unit == 2:
        ans = '{}nd'.format(number)
    elif unit == 3:
        ans = '{}rd'.format(number)
    else:
        ans = '{}th'.format(number)
    
    return ans
 
#%% Module corpus

def canonic_finder(M):

    """From 'M' with 'ab' absorvant states, returns 'Mc' canonic matrix.
    
    It takes 'M' as the Markov matrix of transition probabilities, whose 
    elements 'M[i,j]' hold the transition probability of going from 'j' 
    state to 'i' state. It identifies the absorvant states and 
    constructs the change of base matrix 'C' from a list of permutation 
    pairs. Then, it defines 'Mc' as the Markov canonic matrix of 
    transition. Finally, it returns 'Mc' and 'tr'.
    
    Parameters
    ----------
    M : numpy.array of size 'm'
        Square matrix of Markov transition probabilities. It has size 
        'mxm', where 'm' is the number of states.
    
    Returns
    -------
    Mc : numpy.array of size 'm'
        Square Markov canonic matrix of transition probabilities. It 
        also has size 'mxm', same as M.
    ab : int
        Number of absorvant states.
    
    Raises
    ------
    TypeError : "M should be a matrix"
        if M is not a matrix.
    TypeError : "M should be a square matrix"
        if M is not a square matrix.
    ValueError : "M's determinat should be 0"
        if M's determinant is not zero.
    ValueError : "M's columns should each add up to 1"
        if any of M's column doesn't add up to 1, because in that case 
        the transition probabilities from a single state, ordered on M's 
        columns, don't add up to 1.
    ValueError : "No absorvant states found on M"
        if ab=0, since in that case Mc can't be defined.
        
    """
    
    try:
        m = len(M[:,0])
    except IndexError:
        raise TypeError("M should be a matrix")
        return
    
    if m != len(M[0,:]):
        raise TypeError("M should be a square matrix")
        return

    if np.linalg.det(M) != 0:
        raise ValueError("M's determinat should be 0")
        return    
    
    if not (np.dot(np.ones(m),M) == np.ones(m)).all():
        raise ValueError("M's columns should each add up to 1")
        return
        
    absorvant_states_list = []
    ab = 0 # number of absorvant states
    for i in range(m):
        versor = np.zeros(m)
        versor[i] = 1
        if (M[:,i] == versor).all():
            print("State 'M{}' is {} absorvant state".format(
                  i, counting_suffix(ab+1)))
            absorvant_states_list.append(i)
            ab = ab + 1
    
    tr = 0 # number of transient states
    for i in range(m):
        if i not in absorvant_states_list:
            print("State 'M{}' is {} transient state".format(
                  i, counting_suffix(tr+1)))
            tr = tr + 1
    
    if ab != 0:
                
        j_ab = tr
        j_tr = 0
        change_of_base_pairs = []
        for i in range(m):
            if i in absorvant_states_list:
                change_of_base_pairs.append((i,j_ab))
                j_ab = j_ab + 1
            else:
                change_of_base_pairs.append((i,j_tr))
                j_tr = j_tr + 1
                
        C = np.zeros([m,m]) # change of base matrix
        
        for (i, j) in change_of_base_pairs:
            print("State 'M{}' will be 'C{}'".format(i,j))
            C[i,j] = 1
        
        Mc = np.dot(np.transpose(C), np.dot(M,C)) # canonic matrix
        
    else:
        
        raise ValueError("No absorvant states found on M")
        return               
    
    return Mc, ab

def canonic_decomposer(Mc, ab):
    
    """Decomposes 'Mc' Markov canonic matrix with 'ab' absorvant states.
    
    It takes 'Mc' as the Markov canonic matrix of transition 
    probabilities with 'ab' absorvant states. Then, it decomposes it on 
    'Q' and 'R' matrix. 'Q' only holds the transition probabilities from 
    and to transient states. 'R' only holds the transition 
    probabilities from transient states to absorvant states.
    
    Parameters
    ----------
    Mc : numpy.array of size 'm'
        Square Markov canonic matrix of transition probabilities. It has 
        size 'mxm', where 'm' is the total number of states.
    ab : int
        Number of absorvant states.
    
    Returns
    -------
    Q : numpy.array of size '(tr,tr)'
        Square matrix of transition probabilities from and to transient 
        states. It has size 'trxtr', where 'tr' is the number of 
        transient states.
    R : numpy.array of size '(ab,tr)'
        Partial matrix of transition probabilities from transient states 
        to absorvant states. It has size 'abxtr', where 'ab' is the 
        number of absorvant states and 'tr' is the number of transient 
        states.
    
    Raises
    ------
    TypeError : "Mc should be a matrix"
        if Mc is not a matrix.
    TypeError : "Mc should be a square matrix"
        if Mc is not a square matrix.
    ValueError : "Mc's determinat should be 0"
        if Mc's determinant is not zero.
    ValueError : "Mc's columns should each add up to 1"
        if any of Mc's column doesn't add up to 1, because in that case 
        the transition probabilities from a single state, ordered on 
        Mc's columns, don't add up to 1.
        
    """
    
    try:
        m = len(Mc[:,0])
    except IndexError:
        raise TypeError("Mc should be a matrix")
        return
    
    if m != len(Mc[0,:]):
        raise TypeError("Mc should be a square matrix")
        return

    if np.linalg.det(Mc) != 0:
        raise ValueError("Mc's determinat should be 0")
        return    
    
    if not (np.dot(np.ones(m),Mc) == np.ones(m)).all():
        raise ValueError("Mc's columns should each add up to 1")
        return
    
    tr = m - ab # number of transient states

    Q = Mc[:tr,:tr] # the right upper part of Mc is Q
    R = Mc[tr:,:tr] # the right lower part of Mc is R
    # the left lower part of Mc is 1
    # the left upper part of Mc is 0
    
    return Q, R

def canonic_solver(Q, R):
    
    """Returns 'N' and 'R.N' from 'Q, R' extracted from canonic matrix.
    
    It takes the Q and R matrix from M canonical matrix of transition 
    probabilities. Then, it solves the Markov problem with absorvant 
    states. Because of that, it returns the matrix N = sum(Q^n, 0, inf) 
    using N = inv(1-Q) whose convergence is guarenteed by det(Q)<1. It 
    also returns RN = dot(R, N) matrix product.
    
    Parameters
    ----------
    Q : np.array with size (nt,nt)
        The square matrix of transition probabilities between transient 
        states. It has size 'nt'x'nt', where nt is the number of 
        transient states.
    R : np.array with size (na,nt)
        The non-square matrix of transition probabilities from 
        transient states to absorvant states. It has size 'na'x'nt', 
        where 'na' is the number of absorvant states.
    
    Returns
    -------
    N : np.array with size (nt,nt)
        The square matrix of transition probabilites raised to N power 
        where N-->inf. It has size 'nt'x'nt', same as 'Q'. Its elements 
        'N[i,j]' represent the mean number of steps it takes to get to 
        'i' transient state from 'j'th transient state.
    RN : np.array with size (na,nt)
        The non-square matrix of probabilities whose elements 'NR[i,j]' 
        say the probability of getting to 'i' absorvant state from 'j' 
        transient state.
    
    Raises
    ------
    TypeError : "Q should be a matrix"
        if Q is not a matrix.
    TypeError : "Q should be a square matrix"
        if Q is not a square matrix.
    ValueError : "Q's determinat is D>=1"
        if Q's determinant isn't less than one, because in that case the 
        method of N = inv(1-Q) is not correct.
    TypeError : "R should be a matrix"
        if R is not a matrix.
    TypeError : "R should have as many columns as Q"
        if R dimensions are not what should be expected from Q's 
        dimensions.
    
    """
       
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


def canonic_data(N, RN, datastring, index_from_1, 
                 index_to_1=None, print_result=True):
    
    """Searches data on 'N', 'RN' of Markov's 'M' with absorvant states.
    
    This function takes 'N', 'RN' matrix of Markov's 'M' transition 
    probabilities' matrix with absorvant states, which can be 
    decomposed on 'Q', 'R' matrix and solved by a canonical method. 
    Then it returns a specific piece of information indicated by
    'datastring' using 'index_from_1' and 'index_to_1' as 1-indexed 
    states.
    
    When 'datastring' includes 'pr', this function returns the 
    probability of getting to 'index_to_1' absorvant state from 
    'index_from_1' transient state.
    
    When 'index_from_1' includes 'tr', this function returns the mean 
    number of steps it takes to get to 'index_to_1' transient state 
    from 'index_from_1' transient state.
    
    Otherwise, if 'datastring' includes 'ab', this function returns the 
    mean number of steps it takes to get absorved from 'index_from_1' 
    transient state.
    
    Parameters
    ----------
    N : np.array with size (nt,nt)
        The square matrix of transition probabilites raised to N power 
        where N-->inf. Its elements 'N[i,j]' represent the mean number 
        of steps it takes to get to 'i'th transient state from 'j'th 
        transient state.
    R : np.array with size (na,nt)
        The non-square matrix of probabilities whose elements 'NR[i,j]' 
        say the probability of getting to 'i' absorvant state from 'j' 
        transient state.
    datastring : str including one of {'pr', 'tr', 'abs'}
        The string that indicates what information is required.
    index_from_1 : int
        The index of the 1-indexed state the transition starts from.
    index_to_1=None : int, optional.
        The index of the 1-indexed state the transition goes to.
    print_result=True : bool, optional.
        A parameter that decides whether to print the result or not.

    Returns
    -------
    ans : int, float
        The required piece of information.

    Raises
    ------
    TypeError : "N should be a matrix"
        if N is not a matrix.
    TypeError : "N should be a square matrix"
        if N is not a square matrix.
    TypeError : "NR should be a matrix"
        if NR is not a matrix.
    TypeError : "NR should have as many columns as N"
        if NR dimensions are not what should be expected from N's 
        dimensions.
    
    """
        
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

    dic = {'tr': 'transient steps',
           'pr': 'probability',
           'abs': 'absorvant steps'}
    
    for key, value in dic.items():
        if key in datastring:
            datastring = value
    
    if datastring == 'transient steps':
        ans = N[index_from_1-1, index_to_1-1]
        msg = "It takes {:.2f} steps to get to {} transient \
state starting at {} transient state".format(
                      ans,
                      counting_suffix(index_to_1),
                      counting_suffix(index_from_1))
    elif datastring == 'absorvant steps':
        ans = np.dot(np.ones(m),N)[index_from_1-1]
        msg = "It takes {:.2f} steps to get absorved from \
{} transient state".format(
                      ans,
                      counting_suffix(index_from_1))
    elif datastring == 'probability':
        ans = RN[index_to_1-1, index_from_1-1]
        msg = "Starting on {} transient state, will get to \
{} absorvant state with {:.3f} probability ({:.0f}%)".format(
                      counting_suffix(index_from_1),
                      counting_suffix(index_to_1),
                      ans,
                      ans*100)
    
    if print_result:
        print(msg)

    return ans
