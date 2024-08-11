from scipy.stats import norm
from scipy.special import erfinv
from math import log, sqrt, log2, exp, pi
import numpy as np
from numpy.polynomial.hermite import hermroots

"""Module for generating 1D interpolation nodes"""


def unboundKnotsNonNest(n):
    """Generate nodes that eventually cover R, not nested

    Args:
        n (int): number of nodes

    Returns:
        array: n interpolation nodes
    """
    assert (n > 0)
    assert (n % 2 == 1)
    if n == 0:
        return np.array([0])
    knotsExtrema = norm.cdf([-sqrt(2 * log(n)), sqrt(2 * log(n))])
    knotsBounded = knotsExtrema[0] + \
        (knotsExtrema[1] - knotsExtrema[0]) * np.linspace(0, 1, int(n))
    return norm.ppf(knotsBounded)


def unboundedNodesOptimal(n, p=2):
    """Generate desired number of nodes that eventually cover R with a 
    distribution that is optimal for L2_mu (mu=gaussian density) error of degree
    p piecewise polynomial interpolant

    Args:
        n (int): number of nodes
        p (int, optional): Piecewise polynomial interpolant degree + 1 (=number
        of needed nodes to detemrine piecewise poly in each interval). Defaults
        to 2.
        NB n must be odd because one node is 0 and nodes are symmetric around 0
    Returns:
        array: n interpolation nodes
    """

    assert (n % 2 == 1)
    m = int((n-1)/2)
    xx = np.linspace(-m, m, n)/(m+1)
    c = sqrt(2*(2*p+1))
    return c * erfinv(xx)


def unboundedKnotsNested(n, p=2):
    """Use previous function to generated NESTED nodes that eventually conver R 
    and are optimal for L^2_mu (mu = normal density) error of degree p piecewise
    polynomial interpolant

    Args:
        n (int): number of nodes
        p (int, optional): Piecewise polynomial interpolant degree + 1 (=number 
        of needed nodes to detemrine piecewise poly in each interval). Defaults 
        to 2.
        NB n must be odd because one node is 0 and nodes are symmetric around 0
        NB n must have right value for nestedness: n = 2^(i+1)-1 for i=0,1,...

    Returns:
        _type_: _description_
    """
    assert (log2(n+1)-1 >= 0 & (abs(log2(n+1) - int(log2(n+1)) < 1.e-10)))
    return unboundedNodesOptimal(n, p)


def equispacedNodes(n):
    """Gnerate n equispaced nodes on [-1,1] with first and last node on boundary

    Args:
        n (int): number of nodes

    Returns:
        array of double: n  nodes
    """

    if (n == 1):
        return 0
    else:
        return np.linspace(-1, 1, n)


def equispacedInteriorNodes(n):
    """Generates n equispaced nodes on (-1,1) with the first and last node in 
    the interior at same distance from -1,1

    Args:
        n (int): number of nodes

    Returns:
        array of double: n  nodes
    """

    if (n == 1):
        return 0
    else:
        xx = np.linspace(-1, 1, n+2)
        return xx[1:-1:]


def CCNodes(n):
    """Generate n Clenshaw-Curtis nodes, i.e. extrema of (n-1)-th Chebyshev polynomial

    Args:
        n (int): number of nodes >=1

    Returns:
        array of double: n nodes; if n=1, return [0]
    """

    if (n == 1):
        return 0
    else:
        return -np.cos(pi*(np.linspace(0, n-1, n))/(n-1))


def HermiteNodes(n):
    """Hermite interpolation nodes (roots of the n-th Hermite polynomial)

    Args:
        n (int): number of nodes >=1

    Returns:
        array of double: n nodes; if n=1, return [0]
    """

    en = np.zeros(n+1)
    en[-1] = 1
    return hermroots(en)
