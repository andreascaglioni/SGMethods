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


def unboundedNodesOptimal(n):
    assert(n%2==1)
    m = int((n-1)/2)
    xx = np.linspace(-m,m,n)/(m+1)
    return 2**(3/2) * erfinv(xx)

def unboundedKnotsNested(n):
    """n must be of form 2^(i+1) - 1"""
    assert (log2(n + 1) - 1 >= 0 & (abs(log2(n + 1) - int(log2(n + 1)) < 1.e-10)))
    return unboundedNodesOptimal(n)

# wrong distribution of nodes, thoruw away soon
def unboundedKnotsNestedDEPRECATED(n):
    """n must be of form 2^(i+1) - 1"""
    assert (log2(n + 1) - 1 >= 0 & (abs(log2(n + 1) - int(log2(n + 1)) < 1.e-10)))
    knotsBounded = np.linspace(0, 1, n+2)
    knotsBounded = knotsBounded[1:-1:]
    return norm.ppf(knotsBounded)