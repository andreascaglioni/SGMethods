import numpy as np
from math import exp, floor, log2
from numpy.linalg import lstsq


def coordUnitVector(dim, entry):
    en = np.zeros(dim, dtype=int)
    en[entry]=1
    return en


def lexicSort(I):
    return I[np.lexsort(np.rot90(I))]

def rate(err, nDofs):
    err = np.array(err)
    nDofs = np.array(nDofs)
    return -np.log(err[1::]/err[:-1:])/np.log(nDofs[1::]/nDofs[:-1:])

def LSFitLoglog(x, y):
    X = np.log(x)
    Y = np.log(y)
    A = np.vstack([X, np.ones(len(X))]).T
    alpha, gamma = np.linalg.lstsq(A, Y, rcond=None)[0]
    return alpha, exp(gamma)

def compute_level_LC(i):
    '''Compte level and index in LC expansion given linear index
        INPUT i int number of basis function
        OUTPUT l int level of i-th basis function NB starts from 0
        j int index of i-th basis function within l-th level NB starts from 1'''
    
    if i == 0:
        l=0
        j=1
    elif i == 1:
        l=1
        j=1
    else:
        l = floor(log2(i))+1
        j = i - 2**(l-1)
    return l, j


def findMid(mid, midSet):
    """Find position of multi-index in multi-index set

    Args:
        mid (array int): the multi-index we search for
        midSet (2D array int): multi-index set (each row is a multi-index) 

    Returns:
        tuple(bool,int): first vaue tells if mid is in midSet, second value is 
        the position of mid in midSet (-1 if not found)
    """
    # TODO make faster expliting lexicographic order
    if(midSet.size==0):
        return False, -1
    assert(midSet.shape[1] == mid.size)
    boolVec = np.all(midSet==mid, axis=1)
    pos = np.where(boolVec)[0]
    assert(pos.size<2)
    if len(pos)==0:
        return False, -1
    else:
        return True, pos[0]
    
    def find_idx_in_margin(margin, mid):
        idx = np.all(margin == mid, axis=1)
        idx = np.where(idx == True)
        idx = idx[0][0]
        return idx