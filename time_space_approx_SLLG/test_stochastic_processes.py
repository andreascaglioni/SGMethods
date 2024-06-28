import numpy as np
import matplotlib.pyplot as plt
from math import sin, sqrt, cos
import matplotlib.pyplot as plt


def sampleBrownianMotion(tt):
    dt = tt[1]-tt[0]
    increments = sqrt(dt)*np.random.normal(size=tt.size-1)
    WW = np.cumsum(increments)
    return np.concatenate(([0.], WW))

# def L2ProjectionPwConstant(f, tt):
#     nLocalIntegral = 100
#     k = tt[1]-tt[0]
#     PP = np.zeros(tt.size-1)
#     for i in range(PP.size):
#         xx = np.linspace(tt[i], tt[i+1], nLocalIntegral)
#         ffSamples = f(xx)
#         PP[i] = 1/k * np.sum(ffSamples) * k/nLocalIntegral
#     return PP

def L2ProjectionPwConstantWSamples(FF, tt):
    # Assume FF samples equspaced, taken at times nested with tt by an integer factor
    nLocalIntegral = int((FF.size-1) / (tt.size-1))
    assert(nLocalIntegral - (FF.size-1) / (tt.size-1) < 1.e-2)
    k = tt[1]-tt[0]
    PP = np.zeros(tt.size-1)
    err = np.zeros_like(PP)
    for i in range(PP.size):
        # xx = np.linspace(tt[i], tt[i+1], nLocalIntegral)
        ffSamples = FF[i*nLocalIntegral:(i+1)*nLocalIntegral]
        PP[i] = 1/k * np.sum(ffSamples) * k/nLocalIntegral

        err[i] = k/nLocalIntegral * np.sum(np.square(ffSamples - PP[i]))
    L2err = sqrt(np.sum(err))
    return PP, L2err

def plotL2Projection(tt, PP):
    for i in range(tt.size-1):
        plt.plot([tt[i], tt[i+1]], [PP[i], PP[i]], '-')

def convTestP0L2ProjW(T, ttSamples, nKK):
    err=[]
    FF = sampleBrownianMotion(ttSamples)
    for nK in nKK:
        k = T/nK
        tt = np.linspace(0, 1, nK)
        PP, errC = L2ProjectionPwConstantWSamples(FF, tt)
        err = np.append(err, errC)
    return err

# T = 1
# nK = 20
# samplesMult = 10
# k = 1/nK
# tt = np.linspace(0, 1, nK)
# ttSamples = np.linspace(0, T, (nK-1)*samplesMult+1)
# FF = sampleBrownianMotion(ttSamples)
# PP, err = L2ProjectionPwConstantWSamples(FF, tt)
# print(PP)
# print(err)
# plt.plot(ttSamples, FF)
# plotL2Projection(tt, PP)
# plt.show()

# Convergence test in L2 norm
T = 1
nKSamples = 2**10+1
ttSamples = np.linspace(0, T, nKSamples)
nKK = 2**np.array([1,2,3,4,5,6,7,8], dtype=int)+1
err = convTestP0L2ProjW(T, ttSamples, nKK)
print(err)
kk = T/nKK
plt.loglog(kk ,err, '.-', kk, kk**(0.5), 'k-', kk, kk, 'k-')
plt.show()




