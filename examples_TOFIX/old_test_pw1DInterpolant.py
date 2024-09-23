import numpy as np
from scipy.interpolate import interp1d
import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
from SGMethods.nodes_1d import unboundedKnotsNested
from SGMethods.sparse_grid_interpolant import SGInterpolant
from SGMethods.multi_index_sets import midSet
import matplotlib.pyplot as plt
from math import pi, sqrt, exp
from scipy.stats import norm


F = lambda x: np.sin(1*x)

nRNDSamples = 100
yyRND = np.sort(np.random.normal(0, 1, nRNDSamples))
FOnyy = F(yyRND)

maxNumNodes = 1025

# choose interpolant
p=2
interpolationType = "linear"
lev2knots = lambda nu: 2**(nu+1)-1
knots = lambda m : unboundedKnotsNested(m,p=p)

err = np.array([])
nNodes = np.array([])
I = midSet(maxN=1)
w=0
while(True):
    interpolant = SGInterpolant(I.midSet, knots, lev2knots, interpolationType=interpolationType, NParallel=1)
    if(interpolant.numNodes > maxNumNodes):
        break
    print("# nodes:", interpolant.numNodes, "\nNumber effective dimensions:", I.N)
    uOnSG = interpolant.sampleOnSG(F, 1)
    uInterp = interpolant.interpolate(yyRND, uOnSG)

    # compute error
    errSamples = np.array([ abs(uInterp[n]-FOnyy[n]) for n in range(nRNDSamples) ])
    errCurr = sqrt(np.mean(np.square(errSamples)))
    err = np.append(err, errCurr)
    nNodes = np.append(nNodes, interpolant.numNodes)
    
    print("Error:", err[w])
    I.update(0)
    w=w+1

print(err)
rates = -np.log(err[1::]/err[0:-1:])/np.log(nNodes[1::]/nNodes[0:-1:])
print("Rate:",  rates)
plt.loglog(nNodes, err, '.-', nNodes, np.power(nNodes, -2), '-k')
plt.show()
