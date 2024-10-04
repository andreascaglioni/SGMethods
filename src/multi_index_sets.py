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


# TODO what if the user inputs a profit such that P_{e_n} rightarrow 0 as n rightarrow infty?
# TODO rename it, to generic, must meniton OPTIMALITY WRT GAUSSIAN RFS
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