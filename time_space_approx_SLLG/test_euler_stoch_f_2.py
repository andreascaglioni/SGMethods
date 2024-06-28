import numpy as np
import matplotlib.pyplot as plt
from math import sin, sqrt, cos


F = lambda x : -x
b = lambda x : np.cos(5*x)


def Euler(y0, tt, WW):
    k = tt[1]-tt[0]
    yy = np.zeros_like(tt)
    yy[0] = y0
    for i in range(1, tt.size):
        fCurr = F(yy[i-1])*b(WW[i-1])
        # fCurr = -yy[i-1]*cos(WW[i-1])
        # fCurr = np.sin(yy[i-1]) + yy[i-1] *  WW[i]
        yy[i] = yy[i-1] + k * fCurr 
    return yy

def sampleBrownianMotion(tt):
    dt = tt[1]-tt[0]
    increments = sqrt(dt)*np.random.normal(size=tt.size-1)
    WW = np.cumsum(increments)
    return np.concatenate(([0.], WW))
    
NMCSamples = 10
y0 = 1
################## convergence test of RMSE
nTime = 2**np.linspace(1,13,13,dtype=int)+1
dt = 1/(nTime-1)
nTimeExa = 2**14+1
ttExa = np.linspace(0,1, nTimeExa)
dtExa = 1/(nTimeExa-1)
statError = []
AbsErrorSamples = np.zeros((len(nTime), NMCSamples))
ppQuantSamples = np.zeros((len(nTime), NMCSamples))

for nSample in range(NMCSamples):
    WW = sampleBrownianMotion(ttExa)
    # EXACT SOLUTION
    vals_bW_tt = np.concatenate(([0.], b(WW)))
    int_bW_tt = np.cumsum(vals_bW_tt) * dtExa
    yyExaCurr = y0 * np.exp(-int_bW_tt)

    for nRefine in range(len(nTime)):
        nTimeCurr = nTime[nRefine]
        tt = np.linspace(0,1, nTimeCurr)
        ntt = np.linspace(0,nTimeCurr-1, nTimeCurr)
        
        assert(np.linalg.norm(tt - ntt*dt[nRefine]) < 1.e-10)

        jump = (nTimeExa-1)//(nTimeCurr-1)
        yyCurr = Euler(y0, tt, WW[::jump])
        AbsErrorSamples[nRefine, nSample] = np.amax(np.abs(yyCurr - yyExaCurr[::jump]))
        
    plt.loglog(dt, AbsErrorSamples[:, nSample], 'k.-', alpha=0.1)
      

statError = np.mean(AbsErrorSamples, axis=1)  # np.amax(AbsErrorSamples, axis=1)  # 
print("Error: ", statError)
statError=np.array(statError)
print("Rates ", -np.log(statError[1::]/statError[:-1:])/np.log(nTime[1::]/nTime[:-1:]))
plt.loglog(dt, statError, '.-', label='Error (mean L^{\infty})')
plt.loglog(dt, dt, '-k', label='k')
plt.loglog(dt, dt**0.5, '-k', label='k^{0.5}')
plt.xlabel('k')
plt.legend()
plt.show()