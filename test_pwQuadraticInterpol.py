from cProfile import label
from codecs import latin_1_decode
from re import S
import numpy as np
import matplotlib.pyplot as plt
from math import sin, sqrt, pi
from SGMethods.ScalarNodes import unboundedKnotsNested
from scipy.interpolate import interp1d
from scipy.stats import norm 
from SGMethods.ScalarNodes import unboundedNodesOptimal


# 1D pwquadratic interpolation
class pwQuadraticInterpol:
    def __init__(self, x, y):
        assert(x.size%2 == 1)
        self.nStencilPoints = int(0.5*(x.size-1))
        assert(x.size == y.size)
        self.x=x
        self.y=y
        self.halfx = x[0::2]

    def __call__(self, z):
        if(self.y.size==1):
            return np.ones_like(z)*self.y[0]
        # sort z
        sorting_z = np.argsort(z)
        sz = z[sorting_z]
        unsort_sz = np.argsort(sorting_z)
        # for each point in z, find stencil index, i.e. j=0,...,m/2 : x_{2j} < z < x_{2(j+1)}
        jj = np.zeros(z.size, dtype=int)
        pPrev = -1
        for n in range(1, self.halfx.size):
            p = np.searchsorted(sz, self.halfx[n], side='right')
            jj[pPrev:p] = n-1
            pPrev = p
        jj[p:] = n-1
        # the value of the interpolant in z_i is a sum over the stencil only
        # Iu(z_i) = u(y_{2j-1})L_{2j-1}(y) + u(y_{2j})L_{2j}(y) + u(y_{2j+1})L_{2j+1}(y)
        x0 = self.x[jj*2]
        x1 = self.x[jj*2+1]
        x2 = self.x[jj*2+2]
        L0 = ((sz-x1)*(sz-x2))/((x0-x1)*(x0-x2))
        L1 = ((sz-x0)*(sz-x2))/((x1-x0)*(x1-x2))
        L2 = ((sz-x0)*(sz-x1))/((x2-x0)*(x2-x1))
        out = self.y[jj*2] * L0
        out += self.y[jj*2+1] * L1
        out += self.y[(jj+1)*2] * L2
        return out[unsort_sz]
        
        
def mean_approx(x):
    return sqrt(np.mean(np.square(x)))

def rate(nNodes, err):
    err = np.array(err)
    nNodes = np.array(nNodes)
    return -np.log(err[1::]/err[0:-1:])/np.log(nNodes[1::]/nNodes[0:-1:])


u = lambda x : np.sin(x-0.5)
xx = np.sort(np.random.normal(0,1,int(1.e7)))
lev2knot = lambda nu : 2**(nu+1)-1

# nodes = unboundedNodesOptimal(9, p=3)  # it is always degree+1
# uNodes = u(nodes)
# I=pwQuadraticInterpol(nodes, uNodes)
# z = np.array([nodes[1]])
# print(u(z))
# print(I(z))
# Iu = I(xx)
# uExa = u(xx)
# plt.plot(xx, uExa)
# plt.plot(xx, Iu)
# plt.plot(nodes, uNodes, 'x')
# plt.show()
    
err = []
nNodes = []
for nu in range(16):
    nodes = unboundedNodesOptimal(lev2knot(nu), p=3)  # it is always degree+1
    uNodes = u(nodes)
    I=pwQuadraticInterpol(nodes, uNodes)
    Iu = I(xx)
    uExa = u(xx)
    err.append(mean_approx(uExa-Iu))
    nNodes.append(lev2knot(nu))
    # plt.plot(xx, uExa)
    # plt.plot(xx, Iu)
    # plt.plot(nodes, uNodes, 'x')
    # plt.show()
print(rate(nNodes, err))
plt.loglog(nNodes, err)
plt.loglog(nNodes, 1/np.array(nNodes)**2, '-k')
plt.show()
