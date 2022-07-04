from __future__ import division
import sys
sys.path.append('/home/ascaglio/workspace/SGMethods')
from tests.ParametricPoisson import sampleParametricPoisson
import matplotlib.pyplot as plt
from fenics import *
import numpy as np

# Convergence test
yy = [1., 0.5, 0.1]
nHref = 64
uRef, Vref, meshRef = sampleParametricPoisson(yy, nHref)
nH = np.array([2, 4, 8, 16, 32])
err = []
errFun = Function(Vref)
for nIter, nHCurr in enumerate(nH):
    u, V, mesh = sampleParametricPoisson(yy, nHCurr)
    M = PETScDMCollection.create_transfer_matrix(V, Vref)
    # plot(u)
    # plt.show()
    # error
    v = M * u.vector()
    errFun.vector()[:] = uRef.vector()[:] - v[:]
    err.append(norm(errFun, 'H1'))
print(err)
plt.loglog(nH, err, '.-')
plt.loglog(nH, 1./nH, '-k')
plt.show()