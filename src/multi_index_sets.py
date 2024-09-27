import numpy as np
from math import floor, inf
from src.utils import coordUnitVector as unitVector
from src.utils import lexicSort, midIsInReducedMargin, find_idx_in_margin, \
    findMid, lexicSort


"""A collection of functions to generate multi-index sets"""


def TPMidSet(w, N):
    """Generate a tensor product multi-index set such that
        max_i=0^{N-1} nu_i leq w
        NB leq so for w=0 only multi-index 0 included

    Args:
        w (int): maximum l^infty norm of multi-index 
        N (int): number of dimensions

    Returns:
        np.array: multi-index set (each row is a multi-index)
    """

    assert (N >= 1)
    assert (w >= 0)
    
    if w == 0:
        return np.zeros([1, N], dtype=int)
    elif N == 1:
        v = np.linspace(0, w, w+1, dtype=int)
        v = np.reshape(v, (1,)+v.shape)
        return v.T
    else:  # w>0 & N >1
        I = np.zeros((0, N), dtype=int)
        for comp1 in range(w+1):
            rest = TPMidSet(w, N-1)
            newFirstCol = comp1*np.ones((rest.shape[0], 1), dtype=int)
            newRows = np.concatenate((newFirstCol, rest), 1)
            I = np.concatenate((I, newRows), 0)
        return I


def anisoSmolyakMidSet_freeN(w, a):
    """Generate the anisotropic Smolyak multi-index set
        set{ bnu in N_0^N : sum_i=1^N a_i*nu_i leq w}
        . The dimension (N) is 
        determined as: max {nin N : a_N leq w} 

    Args:
        w (double): maximum l^infty norm of multi-index 
        a (function): a(N) gives anisotropy vector of length N

    Returns:
        np.array: multi-index set (each row is a multi-index)
    """

    # Find the maximum N for which a cdot e_N leq w
    N = 1
    while (a(N+1)[-1] <= w):
        N += 1
    return anisoSmolyakMidSet(w, N, a(N))


def anisoSmolyakMidSet(w, N, a):
    """Generate the anisotropic Smolyak multi-index set:
        set{ bnu in N_0^N : sum_i=1^N a_i*nu_i leq w}

    Args:
        w (double): Maximum l^infty norm of multi-index 
        N (int): Number of dimensions
        a (array double): Anisotropy vector (of length N)

    Returns:
        np.array: Multi-index set (each row is a multi-index)
    """

    assert N >= 1
    assert w>=0
    if type(a) == list:
        a = np.array(a)
    assert (a.size == N)
    assert (np.all(a > 0))
    
    if w == 0:
        return np.zeros([1, N], dtype=int)
    elif N == 1:
        v = np.linspace(0, floor(w/a[0]), floor(w/a[0])+1, dtype=int)
        v = np.reshape(v, (1,)+v.shape)
        return v.T
    else:  # w>0 & N >1
        I = np.zeros((0, N), dtype=int)
        for comp1 in range(floor(w/a[0]) + 1):  # 0...floor(w/a[0])
            rest = anisoSmolyakMidSet(w-comp1*a[0], N-1, a[1::])
            newFirstCol = comp1*np.ones((rest.shape[0], 1), dtype=int)
            newRows = np.concatenate((newFirstCol, rest), 1)
            I = np.concatenate((I, newRows), 0)
        return I


def SmolyakMidSet(w, N):
    """Generate the (isotropic) Smolyak multi-index set.
        {{bnu in N_0^N : sum_i=1^N nu_i leq w}

    Args:
        w (double or int): maximum l^infty norm of multi-index 
        N (int): number of dimensions

    Returns:
        np.array: multi-index set (each row is a multi-index)
    """

    assert N >= 1
    a = np.repeat(1., N)
    return anisoSmolyakMidSet(w, N, a)



# TODO remove now in module MidSet
# class midSet():
#     """Multi-index set that can be grown by adding mids from reduced margin.
#        Can be used for adaptivity, where one does not know a-priori the next 
#        bestmulti-index set or for a priori profit-based enlargment if the defi-
#        nition of the profit is not monotonic.
#        Start from {0}.
#        New dimensions are added as follows: I keep an empty buffer dimenision.
#        When a mid is added to last dim, increase buffer by 1 . 
#        If maximum dimension maxN is reached, then there is no buffer anymore.
#     """

#     def __init__(self, maxN=inf, trackReducedMargin=False):
#         """Initialize basic variables; then build default midset {0}  with its 
#         margin, and reduced margin (if tracked).

#         Args:
#             maxN (int, optional): Maximum number of dimensions. Defaults to inf.
#             trackReducedMargin (bool, optional): whether or not to keep track of
#                 reduced margin (costs more). Defaults to False.
#         """
#         self.maxN = maxN  # maximum number of allowed dimensions
#         self.N = 0
#         # NB the number of columns of the minset is always the same as for the 
#         # margin, not self.N. This is for better compatibility later.
#         self.midSet = np.zeros((1, 1), dtype=int)
#         self.numMids = self.midSet.shape[0]  # TODO remove and use getCardinality instead
#         # NB always have midset.shape[1] = self.margin.shape[1] = min(N+1, maxN)
#         self.dimMargin = min(self.N+1, maxN)
#         self.margin = np.identity(self.dimMargin).astype(int)
#         self.trackReducedMargin = trackReducedMargin
#         self.reducedMargin = np.identity(self.dimMargin).astype(int)

#     def getMidSet(self):
#         """Get the midset as an array without the buffer dimension, if there.
#             Otherwose simply return the midset.
        
#         Returns:
#             2D array int: multi-index set
#         """

#         if (self.midSet.shape == (1, 1)):  # the case of trivial midset [0]
#             return self.midSet
#         if (np.linalg.norm(self.midSet[:, -1], ord=1) < 1.e-4):
#             return self.midSet[:, :-1:]
#         return self.midSet

#     def getN(self):
#         """Get the number of dimensions in the multi-index set. The trivial 
#         multi-index set {0} has N=0.

#         Returns:
#             int: number of dimensions
#         """

#         return self.N
    
#     def get_cardinality(self):
#         """Get the cardinality of the multi-index set.

#         Returns:
#             int: number of multi-indices in the set
#         """

#         return self.midSet.shape[0]

#     def update(self, idx_margin):
#         """ Add a multi-index in the margin to the multi-index set.
#          A recursive function keeps the multi-index set downward-closed. 

#         Args:
#             idx_margin (int): Position in the margin of the multi-index to be 
#              added to the multi-index set
#         """

#         assert(idx_margin >= 0 and idx_margin < self.margin.shape[0])

#         # midOriginal = self.margin[idx_margin, :]  # TODO why need this?
#         mid = self.margin[idx_margin, :]

#         # Check if mid is in reduced margin. If not, recursively add mids from
#         # backward margin. Check one dim at a time.
#         for n in range(self.dimMargin):
#             en = unitVector(self.dimMargin, n)
#             checkMember, _ = findMid(mid-en, self.midSet)
#             if (not (mid[n] == 0 or checkMember)):
#                 mid_tmp = mid - en
#                 idx_mid_tmp = find_idx_in_margin(self.margin, mid_tmp)
#                 self.update(idx_mid_tmp)  # Recursive call!
#                 # Dimension may have grown by 1
#                 if (mid.size < self.dimMargin):
#                     mid = np.append(mid, 0)
#                 idx_margin = find_idx_in_margin(self.margin, mid)

#         # Because of the previous loop, we can assume the multi-index mid to be
#         # in the reduced margin of the multi-index set midSet.

#         # Update midSet
#         self.midSet = np.row_stack((self.midSet, mid))
#         self.midSet = lexicSort(self.midSet)
#         self.numMids += 1

#         # Update margin within first dimMargin dimensions (see below for 
#         # addition in buffer)
#         self.margin = np.delete(self.margin, idx_margin,  0)  # remove added mid
#         self.margin = np.row_stack(
#             (self.margin, mid + np.identity(self.dimMargin, dtype=int)))
#         # mids just added may already have been in midset
#         self.margin = np.unique(self.margin, axis=0)

        
#         # Dimension adaptivity
        
#         # increase N by 1 if mid is last coordinate unit vector
#         increaseN = (mid[-1] == 1 and np.all(mid[0:-1:]
#                      == np.zeros(self.dimMargin-1)))
#         if increaseN:
#             self.N += 1
#             # if N < nMax, add (I,1) to margin. Otherwise midset has saturated 
#             # all available dimensions, so we stop including buffer dimension
#             if (self.N < self.maxN):  # NBB self.N already updated 2 lines above
#                 mid = np.append(mid, 0)  # needed in reduced margin computation
#                 marginNewDim = np.hstack(
#                     (self.midSet, \
#                      np.ones((self.midSet.shape[0], 1), dtype=int)))
#                 self.midSet = np.hstack(
#                     (self.midSet, \
#                      np.zeros((self.midSet.shape[0], 1), dtype=int)))
#                 self.margin = np.hstack(
#                     (self.margin, \
#                      np.zeros((self.margin.shape[0], 1), dtype=int)))
#                 self.margin = np.vstack((self.margin, marginNewDim))
#                 self.dimMargin += 1
#         # Finally sort it lexicograhycally
#         self.margin = lexicSort(self.margin)

#         # Update reduced margin
#         if (self.trackReducedMargin):
#             # 1 delete element just added to margin. NB because of recursion, we
#             # can be sure it was in reduced margin
#             idx_rm = np.where(
#                 (self.reducedMargin == mid).all(axis=1))[0][0]
#             self.reducedMargin = np.delete(self.reducedMargin, idx_rm,  0)

#             # 2 update dimension reduced margin, if needed
#             if (self.dimMargin > self.reducedMargin.shape[1]):
#                 dimDiff = self.dimMargin - self.reducedMargin.shape[1]
#                 self.reducedMargin = np.hstack((self.reducedMargin, np.zeros(
#                     (self.reducedMargin.shape[0], dimDiff), dtype=int)))
#                 en = unitVector(self.dimMargin, self.dimMargin-1)
#                 # if margin increased dimension, there is a new unit vector in 
#                 #   it and it is also reduced
#                 self.reducedMargin = np.vstack((self.reducedMargin, en))

#             # 3 add reduced margin elements among forward margin of mid
#             for n in range(self.dimMargin):
#                 fwMid = mid + unitVector(self.dimMargin, n)
#                 if midIsInReducedMargin(fwMid, self.midSet):
#                     self.reducedMargin = np.vstack(
#                         (self.reducedMargin, np.reshape(fwMid, (1, -1))))
#             self.reducedMargin = np.unique(self.reducedMargin, axis=0)
#             self.reducedMargin = lexicSort(self.reducedMargin)


# TODO what if the user inputs a profit such that P_{e_n} rightarrow 0 as n rightarrow infty?
def computeMidSetFast_freeDim(Profit, PMin):
    """Compute midset based on a profit function and a minimum profit threshold:
     Lambda = set{bnu in N^{infty} : PP_{bnu} geq PMin}.
    NB Profit should be monotone (decreasing) wrt to the dimensions (otherwise
    it is not possible to determine it) and the multi-indices (to obtain a
        downward-closed multi-index set).
    NB The present function also determines the maximum dimension possible, then 
    calls the recursive part (next function) with fixed dimension.

    Args:
        Profit (function): Get profit (positive double) for a multi-index. 
            NB this function is required to vanish for increasing dimensions: 
            P_{e_n} rightarrow 0 as n rightarrow infty
            otherwise, the algorithm cannot terminate
        PMin (double): minimum profit threshold for multi-index to be included

    Returns:
        2D array int: multi-index set (each row is a multi-index)
    """

    dMax = 1
    nu = unitVector(dMax+1, dMax)
    while (Profit(nu) > PMin):
        dMax += 1
        nu = unitVector(dMax+1, dMax)
    # dMax is last that allows profit(e_{dMax}) > PMin
    return computeMidSetFast(Profit, PMin, dMax)

def computeMidSetFast(Profit, PMin, N):
    """ Computes recrursively over N dimensions the multi-index set such that 
    profit is above the threshold PMin:
    Lambda = set{bnu in N^{infty} : PP_{bnu} geq PMin}.
    NB Profit function should be monotone (decreasing) wrt to the multi-indices
    to obtain a downward-closed multi-index set.

    Args:
        Profit (function): Compute profit (positive double) for a multi-index
        PMin (double): minimum profit threshold for multi-index to be included
        N (int): number of dimensions

    Returns:
        2D array int: multi-index set (each row is a multi-index of length N)
    """

    assert (N >= 0)
    if N == 0:
        return np.array([0], dtype=int)
    elif N == 1:  # this is the final case of the recursion
        nu = 0
        assert (Profit(nu) > 0)
        while (Profit(nu+1) > PMin):
            nu = nu+1
        # at this point, nu is the largest mid with P > Pmin
        return np.linspace(0, nu, nu+1, dtype=int).reshape((-1, 1))
    else:  # N>1
        I = np.zeros((0, N), dtype=int)
        # check how large the LAST component can be
        nuNExtra = unitVector(N, N-1)  # has only a 1 in the last component
        while (Profit(nuNExtra) > PMin):
            nuNExtra[-1] += 1
        maxLastComp = nuNExtra[-1] - 1
        # Recursively compute the other dimensions. Since I started with the 
        # LAST components, I can simply call again the function Profit that will
        # look onyl at the first N-1 components
        for lastComponentNu in range(0, maxLastComp+1):
            nuCurr = np.zeros(N, dtype=int)
            nuCurr[-1] = lastComponentNu
            PRec = PMin / Profit(nuCurr)
            INewComps = computeMidSetFast(Profit, PRec, N-1)
            INewMids = np.hstack(
                (INewComps, np.full((INewComps.shape[0], 1), lastComponentNu)))
            I = np.vstack((I, INewMids))
        return lexicSort(I)