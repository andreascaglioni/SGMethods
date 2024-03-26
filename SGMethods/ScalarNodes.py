from scipy.stats import norm
from scipy.special import erfinv
from math import log, sqrt, log2, exp, pi
import numpy as np
from numpy.polynomial.hermite import hermroots


def unboundKnotsNonNest(n):
    assert (n > 0)
    assert (n % 2 == 1)
    if n == 0:
        return np.array([0])
    knotsExtrema = norm.cdf([-sqrt(2 * log(n)), sqrt(2 * log(n))])
    knotsBounded = knotsExtrema[0] + (knotsExtrema[1] - knotsExtrema[0]) * np.linspace(0, 1, int(n))
    return norm.ppf(knotsBounded)

def unboundedNodesOptimal(n, p=2):
    """ n number of nodes
        p int number of interpolation pts, so degree+1 !!
        return np.array length n of optimal nodes for order p interpolant"""    
    assert(n%2==1)
    m = int((n-1)/2)
    xx = np.linspace(-m,m,n)/(m+1)
    c = sqrt(2*(2*p+1))
    return c * erfinv(xx)
    

def unboundedKnotsNested(n, p=2):
    """ n number of nodes must be of form 2^(i+1) - 1
        p int number of interpolation pts, so degree+1 !!
        return np.array length n of optimal nodes for order p interpolant """
    assert (log2(n + 1) - 1 >= 0 & (abs(log2(n + 1) - int(log2(n + 1)) < 1.e-10)))
    return unboundedNodesOptimal(n, p)


def equispacedNodes(n):
    """n number of nodes
    return n equispaced nodes on [-1,1] with the first and last node on the closure"""
    if(n==1):
        return 0
    else:
        return np.linspace(-1,1, n)

def equispacedInteriorNodes(n):
    """n number of nodes
    return n equispaced nodes on (-1,1) with the first and last node on the closure"""
    if(n==1):
        return 0
    else:
        xx = np.linspace(-1,1, n+2)
        return xx[1:-1:]

def CCNodes(n):
    """n number of nodes
    return n Clenshw-CUrtis nodes, i.e. extrema of (n-1)-th Chebyshev polynomial"""
    if(n==1):
        return 0
    else:
        return  -np.cos(pi*(np.linspace(0,n-1,n))/(n-1))
        
def HermiteNodes(n):
    """n number of nodes
    return n Hermite nodes, i.e. roots of the n-th Hermite polynomial"""
    en = np.zeros(n+1)
    en[-1]=1
    return hermroots(en)