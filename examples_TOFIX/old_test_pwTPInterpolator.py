import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
from SGMethods.tp_interpolant_wrapper import TPInterpolatorWrapper
from SGMethods.nodes_tp import TPKnots
from SGMethods.nodes_1d import unboundedKnotsNested


N = 2
dimF = 2
F = lambda x, y: np.sin(x+y) * np.array([1., 2.])
nLevels = 5

NRNDSamples = 1000
nNodesV = (np.power(2, np.linspace(1, nLevels, nLevels)) - 1).astype(int)
nNodesVND = np.power(nNodesV, N)
xxRND = np.random.normal(0, 1, [NRNDSamples, N])
FOnxx = np.zeros((NRNDSamples, dimF))
for n in range(NRNDSamples):
    FOnxx[n] = F(*(list(xxRND[n, :])))
spaceNorm = lambda x: np.linalg.norm(x, ord=2, axis=1)
ww = np.exp(- np.square(np.linalg.norm(xxRND, 2, axis=1)))

err = np.zeros([len(nNodesV)])
for n in range(len(nNodesV)):
    print("Computing n =", n)
    # nodes, TPNodes = TPKnots(unboundedKnotsNested, np.array([nNodesV[n], 1]))
    nodes, TPNodes = TPKnots(unboundedKnotsNested, np.ones(N, dtype=int)*nNodesV[n])
    nodesGrid = np.meshgrid(*nodes, indexing='ij')
    fOnNodes = np.zeros(nodesGrid[0].shape + (dimF,))
    it = np.nditer(nodesGrid[0], flags=['multi_index'])
    for x in it:
        currNode = [nodesGrid[n][it.multi_index] for n in range(N)]
        fOnNodes[it.multi_index] = F(*tuple(currNode))  # the lhs is the remaining section of the tensor

    # interpolate
    interp = TPInterpolatorWrapper(nodes, fOnNodes)
    fInterp = interp(xxRND)
    # error
    err[n] = np.amax(spaceNorm(FOnxx - fInterp) * ww)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xxRND[:,0], xxRND[:,1], fInterp[:,0])
    ax.scatter(xxRND[:,0], xxRND[:,1], FOnxx[:,0])
    plt.show()

print(err)
rates = -np.log(err[1::]/err[0:-1:])/np.log(nNodesVND[1::]/nNodesVND[0:-1:])
print("Rate:",  rates)