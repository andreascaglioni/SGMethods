"""A collection of functions that carry out efficiently various basic tasks
needed in the core functions or that may come in handy when using SGMethods for
applications.
"""

from math import exp, floor, log2
import numpy as np
import matplotlib.pyplot as plt

# TODO move to module LC.
def compute_level_lc(i):
    r"""Converts from linear to hiearchical indices for a wavelet explansion 
    like the Levy-Ciesielski expansion of the Browninan motion. The change of
    indices is:

    .. math::

        l &= \left\lfloor \log_2(i+1) \right\rfloor + 1 \\
        j &= i - 2^{l-1}

    Args:
        i (int): Linear index. The fist basis function (identity) corresponds 
            to i=0.

    Returns:
        tuple[int, int]: Pair of hiearahical indices (level, index in level). 
        The levels start from 0, the index in level from 1.
            
    """

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


# Functions for working with multi-indices.
def coord_unit_vector(dim, entry):
    """Shortcut to make a coordinate unit vector. It automacally neglects all
    the trailin zeros after the given dimension.

    Args:
        dim (int): Dimension of the vector.
        entry (int): The unique entry equal to one.

    Returns:
        numpy.ndarray[int]: The deisred coordinate unit vector.
    """

    en = np.zeros(dim, dtype=int)
    en[entry]=1
    return en

def lexic_sort(mid_set):
    """Sort given mutli-index set in lexicographic order. This means that one
    vector comes before another if the first non-equal entry (from the left) is
    smaller.

    Args:
        mid_set (numpy.ndarray[int]): The multi-index set to test. Must be a 2D
            array, each row is a multi-index.

    Returns:
        bool: The sorted mutli-index set.
    """
    return mid_set[np.lexsort(np.rot90(mid_set))]

# TODO the functions findMid, find_idx_in_margin, checkIfElement all do similar
# tasks. Refactor and keep only one.
# TODO make one version that exploits lexicographic order for afster search in
# O(log(number mids)*dim) by useing binary search for each component
# TODO make findMid independent of trailing 0s in either margin or midSet
def find_mid(mid, mid_set):
    """Find out whether a given multi-index appears in a multi index set and 
    determine the position of its first occurance.

    Args:
        mid (numpy.ndarray[int]): The multi-index to search for.
        midSet (2D array int): The multi-index set. Must be a 2D array, each row
            is a multi-index.

    Returns:
        tuple(bool,int): First value tells whether mid was found inside mid_set.
            Second value is the position (-1 if not found).
    """

    # TODO make faster exploiting lexicographic order
    if mid_set.size==0:
        return False, -1
    assert mid_set.shape[1] == mid.size
    bool_vec = np.all(mid_set==mid, axis=1)
    pos = np.where(bool_vec)[0]
    assert pos.size<2
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
        en = coord_unit_vector(dimMids, n)
        condition[n] = ((mid[n] == 0) or (checkIfElement(mid-en, mid_set)))
    return np.all(condition)


# TODO: implement
def is_downward(mid_set):
    """Check if the multi-index set is downward-closed. This means that:

        .. math::

            \forall\nu\in\Lambda, \forall n\in\mathbb{N},\nu_n=0\text{ or }
            \nu-e_n\in\Lambda

        where :math:`e_n` is the n-th coordinate unit vecotr.

    Returns:
        bool: True if the multi-index set is downward-closed, False otherwise.
    """

    Warning("is_downward not yet implemented yet")
    condition = False
    if condition:
        return True
    else:
        return False


# Basic plotting functions for convergence plots, midsets

def rate(err, n_dofs):
    r"""Compute the rate of logarithmic convergence of a sequence.

    Args:
        err (numpy.ndarray[float]): 1D array of errors (positive floats).
        n_nofs (numpy.ndarray[float]): Number of degrees of freedom that give 
            the corresponding error in the array err.

    Returns:
        numpy.ndarray[float]: 1D array containing rates of convergence computed
        between subsequence pars of ``n_dofs`` and ``err`` as

        .. math::

            -\frac{\log(err_{i+1}/err_i)}
            {\log(n_{\text{dofs},i+1}/n_{\text{dofs},i})}.
    """
    err = np.array(err)
    n_dofs = np.array(n_dofs)
    return -np.log(err[1::]/err[:-1:])/np.log(n_dofs[1::]/n_dofs[:-1:])

def ls_fit_loglog(x, y):
    r"""Fit samples of a function :math:`f:\mathbb{R}\rightarrow\mathbb{R}` to a
    linear function in the least-squares sense in the log-log scale.

    Args:
        x (numpy.ndarray[float]): Independent variable samples.
        y (numpy.ndarray[float]): Dependent variable samples.

    Returns:
        tuple[float, float]: The slope and offset of the linear fit in the log-
    log scale.
    """
    X = np.log(x)
    Y = np.log(y)
    A = np.vstack([X, np.ones(len(X))]).T
    alpha, gamma = np.linalg.lstsq(A, Y, rcond=None)[0]
    return alpha, exp(gamma)

def plot_2d_midset(midset):
    """Plot the 2D multi-index set in the plane."""

    assert midset.shape[1] <= 2
    if midset.shape[1] == 1:
        plt.plot(midset, np.zeros(midset.shape[0]), 's')
    else:
        plt.plot(midset[:,0], midset[:,1], 's')
