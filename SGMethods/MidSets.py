import numpy as np
from math import floor


def TPMidSet(w, N):
    """ each row is a multi-index
    define with base knot 0
    codnition: max_i=0^{d-1} x_i leq w
    NBB leq so w=0 onyl mid 0 included """

    ITP = ()
    for i in range(N):
        ITP += (np.linspace(0, w, w+1),)
    IMat = np.meshgrid(*ITP, indexing='ij')
    I = IMat[0].flatten()
    for n in range(1, len(ITP)):
        tmp = IMat[n].flatten()
        I = np.vstack((I, tmp))
    I = I.T
    return I


def anisoSmolyakMidSet(w, N, a):
    """ now a is a np array containing weights so that I = {i : \sum_n i_n a_n \leq w}"""
    assert N >= 1
    assert(np.all(a>0))
    assert(a.shape[0] == N)

    if w == 0:
        return np.zeros([1, N], dtype=int)
    elif N == 1:
        v = np.linspace(0, floor(w/a[0]), floor(w/a[0])+1, dtype=int)
        v = np.reshape(v, (1,)+v.shape)
        return v.T
    else: # w>0 & N >1
        I = np.zeros((0, N), dtype=int)
        for comp1 in range(floor(w/a[0]) + 1): # 0...floor(w/a[0])
            rest = anisoSmolyakMidSet(w-comp1*a[0], N-1, a[1::])
            newFirstCol = comp1*np.ones((rest.shape[0], 1), dtype=int)
            newRows = np.concatenate((newFirstCol, rest), 1)
            I = np.concatenate((I, newRows), 0)
        return I

def SmolyakMidSet(w, N):
    """ Each row is a multi-index define with base knot 0, condition: \sum_i=0^{d-1} x_i \leq w
    NBB leq so w=0 onyl mid 0 included """
    assert N >= 1
    a = np.repeat(1., N)
    return anisoSmolyakMidSet(w, N, a)