import numpy as np
from math import floor, log, inf

# DEPRECATED using np.meshgrid for N>32 not allowed
# def TPMidSet(w, N):
    # """ each row is a multi-index
    # define with base knot 0
    # codnition: max_i=0^{d-1} x_i leq w
    # NBB leq so w=0 onyl mid 0 included """
#     ITP = ()
#     for i in range(N):
#         ITP += (np.linspace(0, w, w+1),)
#     IMat = np.meshgrid(*ITP, indexing='ij')
#     I = IMat[0].flatten()
#     for n in range(1, len(ITP)):
#         tmp = IMat[n].flatten()
#         I = np.vstack((I, tmp))
#     I = I.T
#     return I

def TPMidSet(w, N):
    """ each row is a multi-index
    define with base knot 0
    codnition: max_i=0^{d-1} x_i leq w
    NBB leq so w=0 onyl mid 0 included """
    assert(N>=1)
    assert(w>=0)
    if w == 0:
        return np.zeros([1, N], dtype=int)
    elif N == 1:
        v = np.linspace(0, w, w+1, dtype=int)
        v = np.reshape(v, (1,)+v.shape)
        return v.T
    else: # w>0 & N >1
        I = np.zeros((0, N), dtype=int)
        for comp1 in range(w+1):
            rest = TPMidSet(w, N-1)
            newFirstCol = comp1*np.ones((rest.shape[0], 1), dtype=int)
            newRows = np.concatenate((newFirstCol, rest), 1)
            I = np.concatenate((I, newRows), 0)
        return I

def anisoSmolyakMidSet_freeN(w,a):
    """as following function, but here dimension is determined as maximum possible effective dimension
    input w int or float
          a FUNCTION a(L) L length of sequence"""
    # find the maximum N for which a \cdot e_N \leq w
    N = 1
    while(a(N+1)[-1] <= w):
        N+=1
    return anisoSmolyakMidSet(w, N, a(N))

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

def LLG_optimal_midset_freeN(T, rhoFun):
    """Determine effective dimension, fix rho to vector rather than function"""
    maxLev = max(0, 2*(T-2))
    N = int(2**maxLev)
    rho = rhoFun(N)
    assert(np.all(rho>0))
    assert(rho.size == N)
    return LLG_optimal_midset_recursive(N, T, rho)

# def LLG_optimal_midset_recursive(N, T, rho):
#     """INPUT N int number parametric dimensions
#              T int threshold parameter midset
#              rhoFun function; rhoFun(N) outputs a length N increasing list of positive doubles of 
#        OUTPUT \Lambda_T np array shape (# SG nodes, N) for kanpsack-type optimal multi-index set i.e.
#        Lambda_T = \left\{ \nu\in N^N : 2\vert\nu\vert_1 + \sum_{\nu_i>0} \log_2(\rho_i) + N \log_2(deg-1) \leq T}
#        NBB term  + N \log_2(deg-1) omitted because independent of \nu """
#     if T <= 0 :
#         return np.zeros((1, N), dtype=int)
#     elif N == 1:
#         b = max(0, floor(0.5*(T-log(rho[0], 2))) )
#         I = np.linspace(0, b, b+1)
#         I = np.reshape(I, (1,)+I.shape)
#         return I.T.astype(int)
#     else: # T>0 and N>1. recursion by elimination of first coordinate
#         I = np.zeros((0, N), dtype=int)
#         LIM = max(0, floor(0.5*(T-log(rho[0], 2))))
#         # case nu_1 = 0
#         nu1=0
#         rest = LLG_optimal_midset_recursive(N-1, T, rho[1::])
#         newFirstCol = nu1*np.ones((rest.shape[0], 1), dtype=int)
#         newRows = np.concatenate((newFirstCol, rest), 1)
#         I = np.concatenate((I, newRows), 0)
#         # case nu_1>0
#         for nu1 in range(1, LIM+1):  # nbb last int in range() is NOT included!!
#             rest = LLG_optimal_midset_recursive(N-1, T - 2*nu1 - log(rho[0], 2), rho[1::])
#             newFirstCol = nu1*np.ones((rest.shape[0], 1), dtype=int)
#             newRows = np.concatenate((newFirstCol, rest), 1)
#             I = np.concatenate((I, newRows), 0)
#         return I.astype(int)


def count_cps(I, r):
    """INPUT: I np array shape  number of nodes x number of dimensions
              r function that counts nodes"""
    ncps = 0
    for i in range(I.shape[0]):
        ncps += r(I[i, :])
    return ncps


"""Multi-index set class; can be grwon by adding mids from reduced margin
Start from {0}
New dimensions are also added as follows: I keep an empty dimenision as buffer, when a fully fimensional mid is added, increase the buffer by 1"""
class midSet():
    def __init__(self, maxN=inf):
        self.N = 0  ## NBB self.midset.shape[1] is always N+1 (not N) because of buffer
        self.maxN = maxN  # maximum number of allowed dimensions
        self.midSet = np.zeros((1, 1)).astype(int)
        self.numMids = self.midSet.shape[0]
        self.margin = np.identity(1).astype(int)
    def update(self, idx_margin):
        mid = self.margin[idx_margin,:]
        # assert(mid[-1] >= 0 and mid[-1] <= 1) # sanity check for dimensinality (buffer dim margin should always be empty)
        # check if mid is in reduced margin
        for n in range(self.N+1):
            en = np.zeros(self.N+1).astype(int)
            en[n] = 1
            assert(mid[n] == 0 or ((mid-en).tolist() in self.midSet.tolist()))
        #update midset
        self.midSet = np.row_stack((self.midSet, self.margin[idx_margin, :]))
        self.midSet =self.midSet[np.lexsort(np.rot90(self.midSet))]
        self.numMids += 1
        #update margin 
        self.margin = np.delete(self.margin, idx_margin,  0)  # remove added mid 
        for n in range(self.N+1): # add those in forward margin (if they are in not already in margin!)
            en = np.zeros(self.N+1).astype(int)
            en[n] = 1
            en.astype(int)
            newMid = mid + en
            if(not(newMid.tolist() in self.margin.tolist())):
                self.margin = np.row_stack((self.margin, newMid))
        # update number of dimensions N of midset and margin if added mid had nonzero last dimension. Also add (I,1) to margin
        if(mid[-1] == 1 and self.N < self.maxN):
            self.N = self.N + 1
            marginNewDim = np.hstack((self.midSet, np.ones((self.midSet.shape[0], 1))))
            self.midSet = np.hstack((self.midSet, np.zeros((self.midSet.shape[0], 1)))).astype(int)
            self.margin = np.hstack((self.margin, np.zeros((self.margin.shape[0], 1))))
            self.margin = np.vstack((self.margin, marginNewDim)).astype(int)
        # finally sort the margin in lexicographic order
        self.margin = self.margin[np.lexsort(np.rot90(self.margin))]