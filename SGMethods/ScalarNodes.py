from scipy.stats import norm
from math import log, sqrt, log2
import numpy as np


def unboundedKnotsNested(n):
    """n must be of form 2^(i+1) - 1"""
    assert (log2(n + 1) - 1 >= 0 & (abs(log2(n + 1) - int(log2(n + 1)) < 1.e-10)))
    knotsBounded = np.linspace(0, 1, n+2)
    knotsBounded = knotsBounded[1:-1:]
    return norm.ppf(knotsBounded)

def unboundKnotsNonNest(n):
    assert (n > 0)
    assert (n % 2 == 1)
    if n == 0:
        return np.array([0])
    knotsExtrema = norm.cdf([-sqrt(2 * log(n)), sqrt(2 * log(n))])
    knotsBounded = knotsExtrema[0] + (knotsExtrema[1] - knotsExtrema[0]) * np.linspace(0, 1, int(n))
    return norm.ppf(knotsBounded)


