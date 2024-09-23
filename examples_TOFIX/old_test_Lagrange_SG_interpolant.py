from __future__ import division
import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from SGMethods.multi_index_sets import anisoSmolyakMidSet, SmolyakMidSet
from SGMethods.sparse_grid_interpolant import SGInterpolant
from SGMethods.nodes_1d import unboundedKnotsNested
import scipy.io



def convergenceTest(maxNumNodes, N, anisoVector, knots, lev2knots, F, dimF, xxRND, fOnxx, weight):
    # convergence test
    err = np.array([])
    nNodes = np.array([])
    w=0
    while True:
        print("Computing n = ", w)
        I = anisoSmolyakMidSet(w, N, anisoVector)
        # plt.scatter(*zip(*I))
        # plt.show()
        interpolant = SGInterpolant(I, knots, lev2knots)
        SG = interpolant.SG
        # plt.scatter(*zip(*SG))
        # plt.show()

        if(interpolant.numNodes>maxNumNodes):
            break
        if w ==0 :
            oldSG = None
            FOnSG = None
        FOnSG = interpolant.sampleOnSG(F, dimF, oldSG, FOnSG)
        
        fInterp = interpolant.interpolate(xxRND, FOnSG)
        err = np.append(err, np.amax(spaceNorm(fOnxx - fInterp) * weight))
        nNodes = np.append(nNodes, interpolant.numNodes)

        oldSG = interpolant.SG
        w+=1
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(xxRND[:, 0], xxRND[:, 1], fInterp[:,0], marker='.')
        # ax.scatter(xxRND[:, 0], xxRND[:, 1], fOnxx[:,0], marker='.')
        # plt.show()
    return err, nNodes

# choose function
N = 2
dimF = 3
def F(x): 
    return np.sin(x[0]+0.01*x[1]) * np.array([1., 2., 3.])
# choose interpolant
lev2knots = lambda n: n+1
mat = scipy.io.loadmat('SGMethods/knots_weighted_leja_2.mat')
wLejaArray = np.ndarray.flatten(mat['X'])
knots = lambda n : wLejaArray[0:n:]

# error computations
spaceNorm = lambda x: np.linalg.norm(x, ord=2, axis=1)
NRNDSamples = 1000
xxRND = np.random.normal(0, 1, [NRNDSamples, N])
fOnxx = np.zeros((NRNDSamples, dimF))
for w in range(NRNDSamples):
    fOnxx[w] = F(xxRND[w, :])
weight = np.exp(- np.square(np.linalg.norm(xxRND, 2, axis=1)))


maxNumNodes = 100
err, nNodes = convergenceTest(maxNumNodes, N, np.array([1., 1]), knots, lev2knots, F, dimF, xxRND, fOnxx, weight)
print(err)
rates = -np.log(err[1::]/err[0:-1:])/np.log(nNodes[1::]/nNodes[0:-1:])
print("Rate:",  rates)
plt.loglog(nNodes, err, '.-', label="1")

err, nNodes = convergenceTest(maxNumNodes, N, np.array([1., 2]), knots, lev2knots, F, dimF, xxRND, fOnxx, weight)
print(err)
rates = -np.log(err[1::]/err[0:-1:])/np.log(nNodes[1::]/nNodes[0:-1:])
print("Rate:",  rates)
plt.loglog(nNodes, err, '.-', label="2")

err, nNodes = convergenceTest(maxNumNodes, N, np.array([1., 4]), knots, lev2knots, F, dimF, xxRND, fOnxx, weight)
print(err)
rates = -np.log(err[1::]/err[0:-1:])/np.log(nNodes[1::]/nNodes[0:-1:])
print("Rate:",  rates)
plt.loglog(nNodes, err, '.-', label="4")
plt.legend()
plt.show()