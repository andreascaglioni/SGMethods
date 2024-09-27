import numpy as np
from math import exp, floor, log2



################################ LC expansion #################################
# TODO move to repo SLLG
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


################################ Multi-indices ################################
def coordUnitVector(dim, entry):
    en = np.zeros(dim, dtype=int)
    en[entry]=1
    return en


def lexicSort(I):
    return I[np.lexsort(np.rot90(I))]


# TODO the functions findMid, find_idx_in_margin, checkIfElement all do similar
# tasks. Refactor and keep only one. 
# TODO make one version that exploits lexicographic order for afster search in 
# O(log(number mids)*dim) by useing binary search for each component
# TODO make findMid independent of trailing 0s in either margin or midSet
def findMid(mid, midSet):
    """Find position of multi-index in multi-index set

    Args:
        mid (array int): the multi-index we search for
        midSet (2D array int): multi-index set (each row is a multi-index) 

    Returns:
        tuple(bool,int): first vaue tells if mid is in midSet, second value is 
        the position of mid in midSet (-1 if not found)
    """
    # TODO make faster exploiting lexicographic order
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

def checkIfElement(mid, midSet):
    """_summary_

    Args:
        mid (_type_): _description_
        midSet (_type_): _description_

    Returns:
        _type_: _description_
    """
    assert (mid.size == midSet.shape[1])
    assert (len(mid.shape) == 1)
    assert (len(midSet.shape) == 2)
    return (midSet == mid).all(axis=1).any()

# TODO substittue checkIfElement with findMid
def midIsInReducedMargin(mid, mid_set):
    """
    Check if a given multi-index mid belongs to the reduced margin of a 
    multi-index set mid_set. This is the case if and only if
    forall n = 1,...,N : mid_n = 0 or mid - e_n in mid_set
    
    Args:
        mid (np.ndarray): A 1D numpy array representing the multi-index.
        mid_set (np.ndarray): A 2D numpy array representing the multi-index set.
    Returns:
        bool: True if the midpoint is in the reduced margin, False otherwise.
    """

    dimMids = mid_set.shape[1]
    assert (dimMids == mid.size)
    condition = np.zeros(dimMids, dtype=bool)
    for n in range(dimMids):
        en = coordUnitVector(dimMids, n)
        condition[n] = ((mid[n] == 0) or (checkIfElement(mid-en, mid_set)))
    return np.all(condition)


