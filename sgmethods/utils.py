"""A collection of functions that carry out efficiently various basic tasks
needed in the core functions or that may come in handy when using SGMethods in
numerical aexperiments and applications.
"""

from math import exp, floor, log2
import numpy as np
import matplotlib.pyplot as plt


def compute_effective_dim_mid_set(mid_set):
    """Find max dimension with non-zero indices.

    Args:
        mid_set (numpy.ndarray[float]): The multi-index set. Each row is a multi-index.

    Returns:
        int: The maximum dimension with at least 1 non-zero index.
    """
    s = np.linalg.norm(mid_set, axis=0)
    w = np.asarray(s>0).nonzero()[0]
    if w.size == 0:
        return 0
    else:
        return np.amax(w)+1  # +1 to go from array idx to number


def float_f(x):
    return "{:.4e}".format(x)


# TODO move to module LC.
def compute_level_lc(i):
    r"""Converts from linear to hierarchical indices for a wavelet expansion 
    like the Levy-Ciesielski expansion of the Brownian motion. The change of
    indices is, for a linear index :math:`i\in\mathbb{N}_0`:

    .. math::

        l &= \left\lfloor \log_2(i+1) \right\rfloor + 1 \\
        j &= i - 2^{l-1}

    Args:
        i (int): Linear index. The fist basis function corresponds to i=0.

    Returns:
        tuple[int, int]: Pair of hierarchical indices (level :math:`l`, index 
        in level :math:`j`). 
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


def float_f(x):
    """
    Standart SGMethods formatting (scientific, with 4 decimals).
    Args:
        x (float): The number to format.
    Returns:
        str: The formatted string in scientific notation.
    """

    return "{:.4e}".format(x)


# Functions for working with multi-indices.
def coord_unit_vector(dim, entry):
    """Returns a coordinate unit vector, i.e. one with all zeros except for one
    entry equal to one.

    Args:
        dim (int): Length of the vector.
        entry (int): The unique entry equal to one.

    Returns:
        numpy.ndarray[int]: The desired coordinate unit vector.
    """

    en = np.zeros(dim, dtype=int)
    en[entry]=1
    return en

def lexic_sort(mid_set):
    """Sort given multi-index set in lexicographic order. This means that one
    vector comes before another if the first non-equal entry (from the left) is
    smaller.

    Args:
        mid_set (numpy.ndarray[int]): The multi-index set to test. Must be a 2D
            array, each row is a multi-index.

    Returns:
        bool: The sorted multi-index set.
    """
    return mid_set[np.lexsort(np.rot90(mid_set))]

# TODO make one version that exploits lexicographic order for afster search in
# O(log(number mids)*dim) by useing binary search for each component
# TODO make findMid independent of trailing 0s in either margin or midSet
def find_mid(mid, mid_set):
    """Find out whether a given multi-index appears in a multi index set and 
    determine the position of its first occurrence.

    Args:
        mid (numpy.ndarray[int]): The multi-index to search for.
        mid_set (numpy.ndarray[int]): The multi-index set. Must be a 2D array, each row
            is a multi-index.

    Returns:
        tuple(bool,int): First value tells whether ``mid`` was found inside 
        ``mid_set``. Second value is the position (-1 if not found).
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
    """DEPRECATED. To be removed soon and substitute with 
    :py:func:`find_mid`."""
    idx = np.all(margin == mid, axis=1)
    idx = np.where(idx == True)
    idx = idx[0][0]
    return idx

def checkIfElement(mid, mid_set):
    """DEPRECATED. To be removed soon and substitute with 
    :py:func:`find_mid`."""

    assert (mid.size == mid_set.shape[1])
    assert (len(mid.shape) == 1)
    assert (len(mid_set.shape) == 2)
    return (mid_set == mid).all(axis=1).any()

# TODO substittue checkIfElement with findMid
def mid_is_in_reduced_margin(mid, mid_set):
    r"""
    Check if ``mid`` belongs to the reduced margin of ``mid_set``. This is the 
    case if and only if 
    
    .. math::
        
        \forall n = 1,...,N, \ \nu_n = 0 
        \textrm{ or } \nu - e_n \in \Lambda,

    where we denoted ``mid_set`` and ``mid`` by :math:`\Lambda` and :math:`\nu`
    respectively. Here :math:`e_n` is the n-th coordinate unit vector,
    :math:`N` is the dimension of ``mid_set`` and ``mid``.

    Args:
        mid (numpy.ndarray[float]): 1D array for the mutli-index.
        mid_set (numpy.ndarray[float]): A 2D numpy array for the multi-index
            set. Each wor is a multi-index. 
    Returns:
        bool: True if ``mid`` is in the reduced margin of ``mid_set``, False 
        otherwise.
    """

    dim_mids = mid_set.shape[1]
    assert dim_mids == mid.size
    condition = np.zeros(dim_mids, dtype=bool)
    for n in range(dim_mids):
        en = coord_unit_vector(dim_mids, n)
        condition[n] = ((mid[n] == 0) or (checkIfElement(mid-en, mid_set)))
    return np.all(condition)

def is_downward(mid_set):
    r"""Check if the multi-index set is downward-closed. This means that:

        .. math::

            \forall\nu\in\Lambda, \forall n\in\mathbb{N},\nu_n=0\text{ or }
            \nu-e_n\in\Lambda,

        where :math:`e_n` is the n-th coordinate unit vector.

    Returns:
        bool: True if the multi-index set is downward-closed, False otherwise.
    """

    for mid in mid_set:
        for i in range(mid.size):
            if mid[i] != 0 and not\
                find_mid(mid-coord_unit_vector(mid.size, i), mid_set)[0]:
                return False
    return True

# TODO implement
def add_multi_index(mids, mid_set):
    """Add the multi-indices in ``mids`` to the multi-index set ``mid_set``.

    Args:
        mids (numpy.ndarray[float]): The multi-indices to add. Can be a float 
            (1 multi-index of length 1), 1D array (1 multi-index), or 2D array
            (each row is a multi-index).
        mid_set (numpy.ndarray[float]): 2D array, where each row represents a 
            multi-index. The multi-index set.

    Returns:
        numpy.ndarray[float]: The resulting multi-index set, lexicographically
        sorted.
    """
    return None

# Basic plotting functions for convergence plots, midsets

def rate(err, n_dofs):
    r"""Compute the rate of logarithmic convergence of a sequence.

    Args:
        err (numpy.ndarray[float]): 1D array of errors (positive).
        n_nofs (numpy.ndarray[int]): Number of degrees of freedom that give 
            the corresponding error in ``err``.

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
        plt.plot(midset, np.zeros(midset.shape[0]), 's', color='black')
    else:
        plt.plot(midset[:,0], midset[:,1], 's', color='black')
    plt.xticks(np.arange(int(midset[:,0].min()), int(midset[:,0].max())+1, 1))
    plt.yticks(np.arange(int(midset[:,1].min()), int(midset[:,1].max())+1, 1))
    plt.xlabel('Index 1')
    plt.ylabel('Index 2')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
