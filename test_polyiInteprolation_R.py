import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from SGMethods.ScalarNodes import unboundedNodesOptimal
from math import sqrt
"""test interpolation on R degrees from 1 to 3"""

def mean_approx(x):
    return sqrt(np.mean(np.square(x)))

def rate(nNodes, err):
    err = np.array(err)
    nNodes = np.array(nNodes)
    return -np.log(err[1::]/err[0:-1:])/np.log(nNodes[1::]/nNodes[0:-1:])

fun = lambda x: np.sin(2**0*x**2)
yyRnd = np.sort(np.random.normal(0, 1, int(1.e6)))
uExa = fun(yyRnd)
lev2knots = lambda nu : 2*nu+1
err1 = []
err2 = []
err3 = []
nNodes = []
for w in range(16):
    print(w)
    # deg 1
    nodes = unboundedNodesOptimal(lev2knots(2**w+1), p=1)
    uNodes = fun(nodes)
    I = interp1d(nodes, uNodes, kind='slinear', fill_value='extrapolate')
    uInterp = I(yyRnd)
    errFun = uExa - uInterp
    err1.append(mean_approx(errFun))
    nNodes.append(lev2knots(2**w+1))

    # deg 2
    nodes = unboundedNodesOptimal(lev2knots(2**w+1), p=2)
    uNodes = fun(nodes)
    I = interp1d(nodes, uNodes, kind='quadratic', fill_value='extrapolate')
    uInterp = I(yyRnd)
    errFun = uExa - uInterp
    err2.append(mean_approx(errFun))

    # deg 3
    nodes = unboundedNodesOptimal(lev2knots(2**w+1), p=3)
    uNodes = fun(nodes)
    I = interp1d(nodes, uNodes, kind='cubic', fill_value='extrapolate')
    uInterp = I(yyRnd)
    errFun = uExa - uInterp
    err3.append(mean_approx(errFun))
print(rate(nNodes, err1))
print(rate(nNodes, err2))
print(rate(nNodes, err3))
plt.loglog(nNodes, err1, label="1")
plt.loglog(nNodes, err2, label="2")
plt.loglog(nNodes, err3, label="3")
plt.loglog(nNodes, 1/np.array(nNodes)**2, '-k')
plt.loglog(nNodes, 1/np.array(nNodes)**3, '-k')
plt.loglog(nNodes, 1/np.array(nNodes)**4, '-k')
plt.legend()
plt.show()