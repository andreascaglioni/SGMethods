"""
This module provides various functions to generate nodes for interpolation and 
numerical integration. The nodes are generated for different types of 
distributions and polynomial interpolants.
"""

from math import sqrt, log2, pi
import numpy as np
from numpy.polynomial.hermite import hermroots
from scipy.special import erfinv


def equispaced_nodes(n):
    """Gnerate n equispaced nodes on [-1,1] with first and last node on
    boundary.

    Args:
        n (int): number of nodes.

    Returns:
        numpy.ndarray[float]: n  nodes.
    """

    if n == 1:
        return 0  # TODO consider np.array([0]) instead, and for other functions
    else:
        return np.linspace(-1, 1, n)


def equispaced_interior_nodes(n):
    """Generates n equispaced nodes on (-1,1) with the first and last node in 
    the interior at same distance from -1 and 1.
    They conicide with the n+2 equispaced nodes on [-1,1] with first and last
    node left out.

    Args:
        n (int): number of nodes.

    Returns:
        numpy.ndarray[float]: n nodes.
    """

    if n == 1:
        return 0
    xx = np.linspace(-1, 1, n+2)
    return xx[1:-1:]


def cc_nodes(n):
    r"""Generate n Clenshaw-Curtis nodes, i.e. extrema of (n-1)-th Chebyshev 
    polynomial on [-1,1]. They read

    .. math::

        x_i = -\cos\left(\frac{i\pi}{n-1}\right), \quad i=0,1,...,n-1.

    Args:
        n (int): number of nodes >=1.

    Returns:
        numpy.ndarray[float]: n nodes; if n=1, return 0.
    """

    if n == 1:
        return 0
    return -np.cos(pi*(np.linspace(0, n-1, n))/(n-1))


def hermite_nodes(n):
    """Hermite interpolation nodes (roots of the n-th Hermite polynomial).

    Args:
        n (int): number of nodes >=1.

    Returns:
        numpy.ndarray[float]: n nodes; if n=1, return 0.
    """

    en = np.zeros(n+1)
    en[-1] = 1
    return hermroots(en)


def optimal_gaussian_nodes(n, p=2):
    """Generate ``n`` interpolation nodes that eventually cover 
    :math:`\mathbb{R}` and give optimal degree ``p+1`` piecewise poynomial 
    interpolation in the :math:`L^2_{\mu}(\mathbb{R})` norm, where :math:`\mu`
    denote the Gaussian density.

    Args:
        n (int): number of nodes
        p (int, optional): Piecewise polynomial interpolant degree + 1 (i.e. the
            number of nodes needed to determine a piecewise polynomial of degree
            p in each interval). Defaults to 2. n must be odd because nodes are 
            symmetric around 0 and one node is 0.
    
    Returns:
        numpy.ndarray[float]: n interpolation nodes.
    """

    assert n % 2 == 1
    m = int((n-1)/2)
    xx = np.linspace(-m, m, n)/(m+1)
    c = sqrt(2*(2*p+1))
    return c * erfinv(xx)


def unbounded_nodes_nested(n, p=2):
    """Use :py:func:`optimal_gaussian_nodes` to generated ``n`` *nested* nodes
    that eventually conver :math:`\mathbb{R}` and give optimal degree ``p+1`` 
    piecewise poynomial interpolation in the :math:`L^2_{\mu}(\mathbb{R})` norm,
    where :math:`\mu` denote the Gaussian density. 
    ``n`` must be odd because one node is 0 and nodes are symmetric around 0. 
    ``n`` must have the right value for the sequence of ndoes to be nested, i.e. 
    :math:`n = 2^{i+1}-1` for some :math:`i\in\mathbb{N}_0`.

    Args:
        n (int): number of nodes.
        p (int, optional): Piecewise polynomial interpolant degree + 1 (i.e. the
            number of nodes needed to determine a piecewise polynomial of degree
            p in each interval). Defaults to 2.

    Returns:
        numpy.ndarray[float]: n interpolation nodes.
    """
    assert (log2(n+1)-1 >= 0 & (abs(log2(n+1) - int(log2(n+1)) < 1.e-10)))
    return optimal_gaussian_nodes(n, p)