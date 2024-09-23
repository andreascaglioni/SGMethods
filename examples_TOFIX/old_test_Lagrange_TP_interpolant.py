import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
from SGMethods.tp_lagrange import TPLagrangeInterpolator
from SGMethods.nodes_1d import unboundedKnotsNested
from SGMethods.nodes_tp import TPKnots
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import scipy.io



# choose function
N = 3
dimF = 3
F = lambda x,y,z: np.sin(x+y+z)*np.array([1.,2.,3.])

# error computations
spaceNorm = lambda x: np.linalg.norm(x, ord=2, axis=1)
NRNDSamples = 1000
xxRND = np.random.normal(0, 1, [NRNDSamples, N])
fOnxx = np.zeros((NRNDSamples, dimF))
for w in range(NRNDSamples):
    fOnxx[w] = F(*list(xxRND[w, :]))
weight = np.exp(- np.square(np.linalg.norm(xxRND, 2, axis=1)))
nLevels = 6
nNodesV = (np.power(2, np.linspace(1, nLevels, nLevels)) - 1).astype(int)

mat = scipy.io.loadmat('SGMethods/knots_weighted_leja_2.mat')
wLejaArray = np.ndarray.flatten(mat['X'])
WLejaNodes1D = lambda n : wLejaArray[0:n:]

err = np.zeros([len(nNodesV)])
for n in range(len(nNodesV)):
    print("Computing n =", n)
    nodes, TPNodes = TPKnots(WLejaNodes1D, np.ones(N, dtype=int)*nNodesV[n])
    nodesGrid = np.meshgrid(*nodes, indexing='ij')
    fOnNodes = np.zeros(nodesGrid[0].shape + (dimF,))
    it = np.nditer(nodesGrid[0], flags=['multi_index'])
    for x in it:
        currNode = [nodesGrid[n][it.multi_index] for n in range(N)]
        fOnNodes[it.multi_index] = F(*tuple(currNode))  # the lhs is the remaining section of the tensor

    I = TPLagrangeInterpolator(nodes, fOnNodes)
    yy = I(xxRND)
    err[n] = np.amax(np.multiply(spaceNorm(yy - fOnxx), weight))

    # plt.plot(xxRND, np.multiply(spaceNorm(yy - fOnxx), weight), '.')
    # plt.semilogy(xxRND, yy, '.')
    # plt.plot(xxRND, fOnxx, '.')
    # plt.plot(xxRND, 1/weight, '.')
    plt.show() 

print(err)
nNodesVND = np.power(nNodesV, N)
rates = -np.log(err[1::]/err[0:-1:])/np.log(nNodesVND[1::]/nNodesVND[0:-1:])
print("Rate:",  rates)
plt.loglog(nNodesVND, err, '.-')
plt.show()