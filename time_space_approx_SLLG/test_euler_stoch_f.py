import numpy as np
import matplotlib.pyplot as plt
from math import sin, sqrt, cos, exp
from scipy.stats import norm

def Euler(y0, tt, WW):
    k = tt[1]-tt[0]
    yy = np.zeros_like(tt)
    yy[0] = y0
    for i in range(1, tt.size):
        fCurr = -yy[i-1]*cos(WW[i-1])
        # fCurr = np.sin(yy[i-1]) + yy[i-1] *  WW[i]
        yy[i] = yy[i-1] + k * fCurr 
    return yy

def sampleBrownianMotion(tt):
    dt = tt[1]-tt[0]
    increments = sqrt(dt)*np.random.normal(size=tt.size-1)
    WW = np.cumsum(increments)
    return np.concatenate(([0.], WW))
    
NMCSamples = 100
y0 = 0.2
################## convergence test of RMSE
nTime = 2**np.linspace(1,11,11,dtype=int)+1
dt = 1/(nTime-1)
nTimeExa = 2**12+1
ttExa = np.linspace(0, 1, nTimeExa)
dtExa = 1/(nTimeExa-1)
statError = []
AbsErrorSamples = np.zeros((len(nTime), NMCSamples))
ppQuantSamples = np.zeros((len(nTime), NMCSamples))
ppQ2 = np.zeros((len(nTime), NMCSamples))
mu = np.zeros((len(nTime), NMCSamples))
cdf= np.zeros((len(nTime), NMCSamples))

F = lambda x : -x
b = lambda x : np.cos(x)  # x  # 
Db = lambda x : np.sin(x)

for nSample in range(NMCSamples):
    WW = sampleBrownianMotion(ttExa)
    yyExaCurr = Euler(y0, ttExa, WW)
    for nRefine in range(len(nTime)):
        nTimeCurr = nTime[nRefine]
        tt = np.linspace(0,1, nTimeCurr)
        ntt = np.linspace(0,nTimeCurr-1, nTimeCurr)
        
        assert(np.linalg.norm(tt - ntt*dt[nRefine]) < 1.e-10)

        jump = (nTimeExa-1)//(nTimeCurr-1)
        WWCurr =  WW[::jump]
        yyCurr = Euler(y0, tt, WWCurr)
        AbsErrorSamples[nRefine, nSample] = np.amax(np.abs(yyCurr - yyExaCurr[::jump]))
        
        #### compute post-processed quantity
        # intInc = np.zeros(nTimeCurr-1)
        # for l in range(intInc.size):
        #     tjCurrIdx = jump*l
        #     ttIdx = np.linspace(tjCurrIdx, jump*(l+1)-1, jump).astype(int)
        #     intInc[l] = dtExa * np.sum(b(WW[ttIdx]) - b(WW[tjCurrIdx]))
        # ppQuantSamples[nRefine, nSample] = abs(np.sum(F(yyCurr[0:-1:]) * intInc))

        #### compute another post-processed quantity ***at final time***
        dtcurr = dt[nRefine]
        RR = F(yyCurr[0:-1:]) * Db(WWCurr[0:-1:])
        WIncs = WWCurr[1::] - WWCurr[0:-1:]
        muk = 0.5*dtcurr * np.sum(RR * WIncs)
        # sigmak = sqrt(dtcurr*3/12*np.sum(RR**2))
        # ppQ2[nRefine, nSample] = muk * (1-2*norm.cdf(-muk/sigmak))
        mu[nRefine, nSample] = muk
        # cdf[nRefine, nSample] = (1-2*norm.cdf(muk/sigmak))

    # plt.loglog(dt, AbsErrorSamples[:, nSample], 'k.-', alpha=0.1)
        
# ppQuant = np.mean(ppQuantSamples, axis=1)
# print("Rates ", -np.log(ppQuant[1::]/ppQuant[:-1:])/np.log(nTime[1::]/nTime[:-1:]))
# plt.loglog(dt, ppQuant, '.-', dt, dt, 'k-', dt, dt**0.5, 'k-')
# plt.show()

# ppQ2 = np.mean(ppQ2, axis=1)
mu = np.mean(mu, axis=1)
# cdf = np.mean(cdf, axis=1)
# print(ppQ2)
# print("Rates ", -np.log(ppQ2[1::]/ppQ2[:-1:])/np.log(nTime[1::]/nTime[:-1:]))
print(mu)
print("Rates mu", -np.log(mu[1::]/mu[:-1:])/np.log(nTime[1::]/nTime[:-1:]))
# print(cdf)
# print("Rates cdf", -np.log(cdf[1::]/cdf[:-1:])/np.log(nTime[1::]/nTime[:-1:]))

# plt.loglog(dt, np.abs(ppQ2), '.-', dt, dt, 'k-', dt, dt**2, 'k-')
plt.loglog(dt, np.abs(mu), 'o-')
# plt.loglog(dt, np.abs(cdf), '*-')
plt.show()


# statError = np.mean(AbsErrorSamples, axis=1)  # np.amax(AbsErrorSamples, axis=1)  # 
# print("Error: ", statError)
# statError=np.array(statError)
# print("Rates ", -np.log(statError[1::]/statError[:-1:])/np.log(nTime[1::]/nTime[:-1:]))
# plt.loglog(dt, statError, '.-', label='Error (mean L^{\infty})')
# plt.loglog(dt, dt, '-k', label='k')
# plt.loglog(dt, dt**0.5, '-k', label='k^{0.5}')
# plt.xlabel('k')
# plt.legend()
# plt.show()