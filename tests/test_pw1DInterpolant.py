import numpy as np
from scipy.interpolate import interp1d
import sys
sys.path.append('/home/ascaglio/workspace/SGMethods')
from SGMethods.ScalarNodes import unboundedKnotsNested


F = lambda x: np.sin(x)
nNodesV = (np.power(2, (np.linspace(2, 7, 7))) - 1).astype(int)
NRNDSamples = 1000
xxRND = np.sort(np.random.normal(0, 1, [NRNDSamples]))
FOnxx = F(xxRND)
ww = np.exp(-np.square(xxRND))
err = np.zeros([len(nNodesV)])
for n in range(len(nNodesV)):
    nodes = unboundedKnotsNested(nNodesV[n])
    fOnNodes = F(nodes)
    interp = interp1d(nodes, fOnNodes, kind="linear", fill_value="extrapolate")
    fInterp = interp(xxRND)
    errSamples = FOnxx - fInterp
    err[n] = np.amax(np.abs(errSamples * ww))
print(err)
rates = -np.log(err[1::]/err[0:-1:])/np.log(nNodesV[1::]/nNodesV[0:-1:])
print("Rate:",  rates)
