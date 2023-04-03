import numpy as np
import matplotlib.pyplot as plt
from math import sin, sqrt, cos


def sample_mean_and_variance(xx):
    # INPUT xx shaped #samples
    # OUTPUT mean double
    #        variance double
    assert(len(xx.shape)==1)
    nSamples = xx.size
    mean = np.mean(xx)
    variance = 1/(nSamples-1)*(np.sum(np.square(xx-mean)))
    return (mean, variance)
    
def sampleBrownianMotion(tt):
    dt = tt[1]-tt[0]
    increments = sqrt(dt)*np.random.normal(size=tt.size-1)
    WW = np.cumsum(increments)
    return np.concatenate(([0.], WW))

b = lambda x : np.cos(x)
T = 1
nSamplesUnitTime = 400
dt = 1/nSamplesUnitTime
tt = np.linspace(0,T, int(T*nSamplesUnitTime))
NMCSamples = 1000
intBWW = np.zeros((NMCSamples, tt.size))
mean = np.zeros_like(tt)
variance = np.zeros_like(tt)
for nSample in range(NMCSamples):
    WW = sampleBrownianMotion(tt)
    intBWW[nSample, :] = dt*np.cumsum(b(WW)-b(WW[0]))
for nt in range(tt.size):
    (mean[nt], variance[nt]) = sample_mean_and_variance(intBWW[:, nt])
plt.plot(tt, mean)
plt.plot(tt, variance)
plt.plot(tt, -tt**2/2/2,5, 'k-')
plt.show()