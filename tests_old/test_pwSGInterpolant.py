from __future__ import division
import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from SGMethods.MidSets import anisoSmolyakMidSet, SmolyakMidSet
from SGMethods.SGInterpolant import SGInterpolant
from SGMethods.ScalarNodes import unboundedKnotsNested


# choose function
N = 2
dimF = 3
def F(x):
    return np.sin(np.sum(x))+1 * np.array([1., 2., 3.])

anisoVector = np.array([1., 2.])
# choose interpolant
lev2knots = lambda n: 2**(n+1)-1
knots = unboundedKnotsNested

# error computations
spaceNorm = lambda x: np.linalg.norm(x, ord=2, axis=1)
NRNDSamples = 1000
xxRND = np.random.normal(0, 1, [NRNDSamples, N])
fOnxx = np.zeros((NRNDSamples, dimF))
for w in range(NRNDSamples):
    fOnxx[w] = F(xxRND[w, :])
weight = np.exp(- np.square(np.linalg.norm(xxRND, 2, axis=1)))

# convergence test
nLevels = 8
err = np.zeros(nLevels)
nNodes = np.zeros(nLevels)
for w in range(0, len(err)):
    print("Computing n = ", w)
    I = SmolyakMidSet(w, N)
    # plt.scatter(*zip(*I))
    # plt.show()
    interpolant = SGInterpolant(I, knots, lev2knots, pieceWise=True)
    SG = interpolant.SG
    # plt.scatter(*zip(*SG))
    # plt.show()

    nNodes[w] = interpolant.numNodes
    if w ==0 :
        oldSg = None
        FOnSG = None
    FOnSG = interpolant.sampleOnSG(F, dimF)
    
    fInterp = interpolant.interpolate(xxRND, FOnSG)
    err[w] = np.amax(spaceNorm(fOnxx - fInterp) * weight)  # np.mean(spaceNorm(fOnxx-fInterp))  # 

    oldSG = interpolant.SG
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(xxRND[:, 0], xxRND[:, 1], fInterp[:,0], marker='.')
    # ax.scatter(xxRND[:, 0], xxRND[:, 1], fOnxx[:,0], marker='.')
    # plt.show()

print(err)
rates = -np.log(err[1::]/err[0:-1:])/np.log(nNodes[1::]/nNodes[0:-1:])
print("Rate:",  rates)
## result
#[9.06765844e-01 2.38283819e-01 7.05886377e-02 1.46946275e-02
# 4.32681306e-03 1.14810192e-03 3.10653690e-04 8.51160727e-05]
#Rate: [0.83036553 0.99413116 1.4825023  1.2630792  1.45533061 1.49623628
# 1.52933197]

plt.loglog(nNodes, err, '.-', nNodes, 1/nNodes, '-k')
plt.show()
