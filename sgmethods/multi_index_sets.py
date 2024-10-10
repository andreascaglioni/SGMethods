"""A collection of functions to generate dowward-closed multi-index sets."""

from math import floor
import numpy as np
from sgmethods.utils import coord_unit_vector as unitVector
from sgmethods.utils import lexic_sort


def tensor_product_mid_set(w, N):
    r"""Generate a tensor product multi-index set, i.e.

    .. math::

        \Lambda = \{\nu\in\mathbb{N}_0^N : \sum_{n=1}^N \nu_n \leq w\},
        
    where we denoted :math:`\Lambda` the multi-index set, and :math:`w, N` the 
    input ``w``, ``N``.

    Args:
        w (float): Sizing parameter fo the multi-index set.
        N (int): Number of dimensions.

    Returns:
        numpy.ndarray[int]: Multi-index set (each row is a multi-index of length
        ``N``).
    """

    assert N >= 1
    assert w >= 0

    if w == 0:
        return np.zeros([1, N], dtype=int)
    elif N == 1:
        v = np.linspace(0, w, w+1, dtype=int)
        v = np.reshape(v, (1,)+v.shape)
        return v.T
    else:  # w>0 & N >1
        mid_set = np.zeros((0, N), dtype=int)
        for comp1 in range(w+1):
            rest = tensor_product_mid_set(w, N-1)
            new_first_col = comp1*np.ones((rest.shape[0], 1), dtype=int)
            new_rows = np.concatenate((new_first_col, rest), 1)
            mid_set = np.concatenate((mid_set, new_rows), 0)
        return mid_set


def aniso_smolyak_mid_set_free_N(w, a):
    r"""Generate the anisotropic Smolyak multi-index set:

    .. math::

        \Lambda = \{\nu\in\mathbb{N}_0^N : \sum_{i=1}^N a_i\nu_i \leq w\},

    where we denoted :math:`\Lambda` the multi-index set, and :math:`w, N` the 
    input ``w``, ``N``. 
    The dimensionality is determined as: 

    .. math::

        N = \min \{ N\in\mathbb{N} : 
        \forall \nu\in\Lambda,\ \sum_{n=1}^N a_n\nu_n \leq w \}.

    Args:
        w (float): Sizing parameter fo the multi-index set.
        a (Callable[[int], numpy.ndarray[float]]): a(N) gives the anisotropy 
            vector of length :math:`N`. The anisotropy vector is a vector of
            positive floats.

    Returns:
        numpy.ndarray[int]: Multi-index set (each row is a multi-index).    
    """

    # Find the maximum N for which a cdot e_N leq w
    N = 1
    while a(N+1)[-1] <= w:
        N += 1
    return aniso_smolyak_mid_set(w, N, a(N))


def aniso_smolyak_mid_set(w, N, a):
    r"""Generate the anisotropic Smolyak multi-index set:

    .. math::
    
        \Lambda = \{\nu \in \mathbb{N}_0^N : \sum_{i=1}^N a_i\nu_i \leq w \}.

    where we denoted :math:`\Lambda` the multi-index set, and :math:`w, N` the 
    input ``w``, ``N``.
    
    Args:
        w (float): Sizing parameter fo the multi-index set.
        N (int): Number of dimensions.
        a (Callable[[int], numpy.array[float]]): The resuting :math:`a(N)` gives
            the anisotropy vector of length :math:`N`. The anisotropy vector is
            a vector of positive floats.

    Returns:
        numpy.ndarray[int]: Multi-index set (each row is a multi-index of length
        ``N``).
    """

    assert N >= 1
    assert w>=0
    if type(a) == list:
        a = np.array(a)
    assert a.size == N
    assert np.all(a > 0)

    if w == 0:
        return np.zeros([1, N], dtype=int)
    elif N == 1:
        v = np.linspace(0, floor(w/a[0]), floor(w/a[0])+1, dtype=int)
        v = np.reshape(v, (1,)+v.shape)
        return v.T
    else:  # w>0 & N >1
        mid_set = np.zeros((0, N), dtype=int)
        for comp1 in range(floor(w/a[0]) + 1):  # 0...floor(w/a[0])
            rest = aniso_smolyak_mid_set(w-comp1*a[0], N-1, a[1::])
            new_first_col = comp1*np.ones((rest.shape[0], 1), dtype=int)
            new_rows = np.concatenate((new_first_col, rest), 1)
            mid_set = np.concatenate((mid_set, new_rows), 0)
        return mid_set


def smolyak_mid_set(w, N):
    r"""Generate the (isotropic) Smolyak multi-index set, i.e.

    .. math::
        
        \Lambda = \{\nu \in \mathbb{N}_0^N : \sum_{i=1}^N \nu_i \leq w\},

    where we denoted :math:`\Lambda` the multi-index set, and :math:`w, N` the 
    input ``w``, ``N``.
    
    Args:
        w (float): Sizing parameter fo the multi-index set.
        N (int): Number of dimensions.
        
    Returns:
        numpy.ndarray[int]: Multi-index set (each row is a multi-index of length
        ``N``).
    """

    assert N >= 1
    a = np.repeat(1., N)
    return aniso_smolyak_mid_set(w, N, a)

def mid_set_profit_free_dim(profit_fun, minimum_profit):
    r"""Compute a midset based on a profit function ``profit_fun`` and a minimum 
    profit threshold ``profit_fun`` as:
    
    .. math:: 
    
        \Lambda = \{\nu\in\mathbb{N}_0^{\infty}:
        \mathcal{P}_{\nu}\geq P_{\textrm{min}}\},
    
    where we denoted
    :math:`\Lambda` the multi-index set,
    :math:`\mathcal{P}_{\nu}` the profit functiona pploed to a multi-index, and
    :math:`P_{\textrm{min}}` the minimum profit threshold.
    The profit should be monotone (decreasing) wrt to the dimensions (otherwise
    it is not possible to determine it) and the multi-indices (to obtain a 
    downward-closed multi-index set). 
    This condition cannot be checked by the library, the user must ensure it.
    The present function also determines the maximum dimension possible, then 
    calls the :py:func:`compute_mid_set_fast` with fixed dimension.

    Args:
        profit_fun (Callable[[numpy.ndarray[int]], float]): Get profit 
            (positive) for a multi-index. 
            This function is required to vanish for increasing dimensions: 
            :math:`\mathcal{P}_{\mathbf{e}_n} \rightarrow 0` as 
            :math:`n \rightarrow \infty`. Otherwise, the algorithm cannot 
            terminate. This condition cannot be verified by the library and the
            user must ensure it.
        minimum_profit (float): minimum profit threshold for multi-index to be
            included

    Returns:
        numpy.ndarray[int]: Multi-index set (each row is a multi-index).
    """

    d_max = 1
    nu = unitVector(d_max+1, d_max)
    while profit_fun(nu) > minimum_profit:
        d_max += 1
        nu = unitVector(d_max+1, d_max)
    # dMax is last that allows profit(e_{dMax}) > PMin
    return compute_mid_set_fast(profit_fun, minimum_profit, d_max)

def compute_mid_set_fast(profit_fun, minimum_profit, N):
    r""" Computes recrursively over ``N`` dimensions the multi-index set such 
    that the profit ``profit_fun`` is above the threshold ``minimum_profit`` for
    all its multi-indices. The multi-index set is defined as:

    .. math:: 
    
        \Lambda 
        = \{\nu\in\mathbb{N}_0^{N}:\mathcal{P}_{\nu}\geq P_{\textrm{min}}\}
        
    where we denoted
    :math:`\Lambda` the multi-index set,
    :math:`\mathcal{P}_{\nu}` the profit functiona pploed to a multi-index, and
    :math:`P_{\textrm{min}}` the minimum profit threshold.
    The profit function should be monotone (decreasing) wrt to the multi-indices
    to obtain a downward-closed multi-index set.

    Args:
        profit_fun (Callable[[numpy.ndarray[int]], float]): Compute profit 
            (positive float) for a multi-index.
        minimum_profit (float): minimum profit threshold for multi-index to be
            included N (int): number of dimensions.

    Returns:
        numpy.ndarray[int]: Multi-index set (each row is a multi-index of length
        ``N``).
    """

    assert N >= 0
    if N == 0:
        return np.array([0], dtype=int)
    elif N == 1:  # this is the final case of the recursion
        nu = 0
        assert profit_fun(nu) > 0
        while profit_fun(nu+1) > minimum_profit:
            nu = nu+1
        # at this point, nu is the largest mid with P > Pmin
        return np.linspace(0, nu, nu+1, dtype=int).reshape((-1, 1))
    else:  # N>1
        mid_set = np.zeros((0, N), dtype=int)
        # check how large the LAST component can be
        nu_n__extra = unitVector(N, N-1)  # has only a 1 in the last component
        while profit_fun(nu_n__extra) > minimum_profit:
            nu_n__extra[-1] += 1
        max_last_comp = nu_n__extra[-1] - 1
        # Recursively compute the other dimensions. Since I started with the
        # LAST components, I can simply call again the function Profit that will
        # look onyl at the first N-1 components
        for last_comp_nu in range(0, max_last_comp+1):
            nu_curr = np.zeros(N, dtype=int)
            nu_curr[-1] = last_comp_nu
            profit_rec = minimum_profit / profit_fun(nu_curr)
            mid_set_new_comps = \
                compute_mid_set_fast(profit_fun, profit_rec, N-1)
            mid_set_new_mids = np.hstack(
                (mid_set_new_comps,\
                 np.full((mid_set_new_comps.shape[0], 1), last_comp_nu))\
                    )
            mid_set = np.vstack((mid_set, mid_set_new_mids))
        return lexic_sort(mid_set)
    