import numpy as np
from scipy.interpolate import interp1d
import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
from SGMethods.ScalarNodes import unboundedKnotsNested
from SGMethods.SGInterpolant import SGInterpolant
import matplotlib.pyplot as plt
from math import pi, sqrt, exp
from scipy.stats import norm


ff=1
F = lambda x: np.sin(10*x)
lev2knot = lambda nu : 2**(nu+1+2)-1
levMax = 20
levs = np.linspace(0,levMax,levMax+1).astype(int)
nNodesV = lev2knot(levs)
nRNDSamples = 100000
xxRND = np.sort(np.random.normal(0, 1, nRNDSamples))
ww = 1/sqrt(2*pi)*np.exp(-np.square(xxRND)/2)
FOnxx = F(xxRND)
err = np.zeros(levs.size)
for levCurr in levs:
    nodes = unboundedKnotsNested(lev2knot(levCurr), p=4)
    fOnNodes = F(nodes)
    interp = interp1d(nodes, fOnNodes, kind="cubic", fill_value="extrapolate")
    fInterp = interp(xxRND)
    errSamples = FOnxx - fInterp
    err[levCurr] =  sqrt(1/nRNDSamples * np.sum(np.square(errSamples))) # np.amax(np.abs(errSamples * ww))
    # print(xxRND[np.argmax(np.abs(errSamples * ww))])
    # plt.plot(xxRND, fInterp)
    # plt.plot(nodes, fOnNodes, '*')
    # plt.show()
    # plt.plot(xxRND, np.abs(errSamples * ww))
    # idxMax = np.argmax(np.abs(errSamples * ww))
    # plt.plot(xxRND[idxMax], np.abs(errSamples[idxMax] * ww[idxMax]), '*')
    # plt.show()
print(err)
rates = -np.log(err[1::]/err[0:-1:])/np.log(nNodesV[1::]/nNodesV[0:-1:])
print("Rate:",  rates)
plt.loglog(nNodesV, err, '.-')
plt.loglog(nNodesV, ff*np.power(nNodesV.astype(float), -1), '-k')
plt.show()


#################### mapped 1D interpolation
# ff=10
# F = lambda x: np.sin(ff*x)
# FOnX = lambda x: F(norm.ppf(x))
# lev2knot = lambda nu : 2**(nu+1+1)-1
# levMax = 20
# levs = np.linspace(0,levMax,levMax+1).astype(int)
# nNodesV = lev2knot(levs)

# xxRND = np.sort(np.random.uniform(0,1,100000))
# FOnxxRND = FOnX(xxRND)
# ww = (2*pi)**(-0.5)*np.exp(-0.5*np.square(norm.ppf(xxRND)))

# err = np.zeros(levs.size)
# for levCurr in levs:
#     nNodes =  nNodesV[levCurr]
#     nodes = np.linspace(0,1,nNodes+2)[1:-1:]
    
#     fOnNodes = FOnX(nodes)
#     interp = interp1d(nodes, fOnNodes, kind="linear", fill_value="extrapolate")
#     fInterp = interp(xxRND)
#     errSamples = FOnxxRND - fInterp
#     err[levCurr] = np.amax(np.abs(errSamples * ww))
#     # plt.plot(xxRND, np.abs(errSamples * ww))
#     # idxMax = np.argmax(np.abs(errSamples * ww))
#     # plt.plot(xxRND[idxMax], np.abs(errSamples[idxMax] * ww[idxMax]), '*')
#     # plt.show()
    
# print(err)
# rates = -np.log(err[1::]/err[0:-1:])/np.log(nNodesV[1::]/nNodesV[0:-1:])
# print("Rate:",  rates)
# plt.loglog(nNodesV, err, '.-')
# plt.loglog(nNodesV, ff*np.power(nNodesV.astype(float), -1), '-k')
# plt.loglog(nNodesV, ff*np.power(nNodesV.astype(float), -2), '-k')
# plt.show()

################# classical on (0,1)
# ff=1
# F = lambda x: np.sin(ff*x)
# lev2knot = lambda nu : 2**(nu+1+1)-1
# levMax = 20
# levs = np.linspace(0,levMax,levMax+1).astype(int)
# nNodesV = lev2knot(levs)
# xxRND = np.sort(np.random.uniform(0,1,100000))
# FOnxxRND = F(xxRND)
# err = np.zeros(levs.size)
# for levCurr in levs:
#     nodes = np.linspace(0,1, nNodesV[levCurr])
#     print(nodes)
#     fOnNodes = F(nodes)
#     interp = interp1d(nodes, fOnNodes, kind="linear", fill_value="extrapolate")
#     fInterp = interp(xxRND)
#     errSamples = FOnxxRND - fInterp
#     err[levCurr] = np.amax(np.abs(errSamples ))
        
# print(err)
# rates = -np.log(err[1::]/err[0:-1:])/np.log(nNodesV[1::]/nNodesV[0:-1:])
# print("Rate:",  rates)
# plt.loglog(nNodesV, err, '.-')
# plt.loglog(nNodesV, ff**2*np.power(nNodesV.astype(float), -2), '-k')
# plt.show()