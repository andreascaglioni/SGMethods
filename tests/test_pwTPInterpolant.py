import matplotlib.pyplot as plt
import numpy as np
from pwSG.TPInterpolant import TPInterpolant
from pwSG.TPKnots import TPKnots
from pwSG.ScalarNodes import unboundedKnotsNested
from mpl_toolkits.mplot3d import Axes3D


N = 2
F = lambda x, y: np.sin(x+y)
nLevels = 6
NRNDSamples = 1000
nNodesV = np.power(2, np.linspace(1, nLevels, nLevels)) - 1
nNodesVND = np.power(nNodesV, N)
xxRND = np.random.normal(0, 1, [N, NRNDSamples])
FOnxx = np.zeros(NRNDSamples)
for n in range(NRNDSamples):
    FOnxx[n] = F(*(list(xxRND[:, n])))
ww = np.exp(- np.square(np.linalg.norm(xxRND, 2, axis=0)))
err = np.zeros([len(nNodesV)])
for n in range(len(nNodesV)):
    print("Computing n =", n)
    # nodes, TPNodes = TPKnots(unboundedKnotsNested, np.array([nNodesV[n], 1]))
    nodes, TPNodes = TPKnots(unboundedKnotsNested, np.ones(N)*nNodesV[n])
    # samples of F
    nodesGrid = np.meshgrid(*nodes)
    fOnNodes = F(*nodesGrid)
    # interpolate
    interp = TPInterpolant(nodes, fOnNodes)
    fInterp = interp(xxRND.T)
    # error
    err[n] = np.amax((FOnxx - fInterp)*ww)

print(err)
rates = -np.log(err[1::]/err[0:-1:])/np.log(nNodesVND[1::]/nNodesVND[0:-1:])
print("Rate:",  rates)