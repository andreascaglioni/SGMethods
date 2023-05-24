from scipy.stats import norm
from scipy.special import erfinv
from math import log, sqrt, log2, exp, pi
import numpy as np


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



def unboundedKnotsTest(n):
    assert(n%2==1)
    m = int((n-1)/2)
    xx = np.linspace(-m,m,n)/(m+1)
    return 2*sqrt(5) * erfinv(xx)

def unboundedKnotsTest2(n):
    assert(n%2==1)
    m = int((n-1)/2)
    xx = np.linspace(1, m, m)/(m+1)
    v = 2*sqrt(2) * erfinv(xx)
    vr = np.sort(-v)
    return np.hstack((vr, np.array([0.]), v))