import numpy as np
from math import sin, sqrt
import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
from SGMethods.SGInterpolant import SGInterpolant
from SGMethods.ScalarNodes import unboundedKnotsNested
from SGMethods.MidSets import midSet
from SGMethods.MLInterpolant import MLInterpolant
from SLLG.profits import ProfitMix
from SLLG.expansions_Brownian_motion import param_LC_Brownian_motion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
from scipy.interpolate import interp1d

""""Test simulatentous refinement SG and Physical space"""




################################ PARAMETERS ################################
T = 1
p = 2  # NBB degrere + 1
interpolationType = "linear"
lev2knots = lambda nu: 2**(nu+1)-1
knots = lambda m : unboundedKnotsNested(m,p=p)
Profit = lambda nu: ProfitMix(nu, p)
Levels = [0,1,2,3, 4]
nNodes = 2**(2*np.array(Levels)+1)
NRNDSamples = 5
################################ EXACT AND APPROXIMATE SAMPLERS ################################
nTau = 2**(1+np.array(Levels,dtype=int))
tau = 1/nTau
nTT = nTau+1
tt = []
for k in range(len(nTau)):
    tt.append(np.linspace(0,T, nTT[k]))
def FApprox(y, k):  # piecewise const approximation at level in position k in list of Levels (e.g. k+1)
    W = param_LC_Brownian_motion(tt[k], y, T)
    return np.sin(W)

nTauRef = nTau[-1]*2
nTTRef = nTauRef+1
ttRef = np.linspace(0,T,nTTRef)
def FExa(y):  # reference
    WRef = param_LC_Brownian_motion(ttRef, y, T)
    return np.sin(WRef)






################################ TEST CONVERGENCE SG ################################
yyRnd = np.random.normal(0, 1, [NRNDSamples, 1000])
print("Sample reference...")
samplesFExa = list(map(FExa, yyRnd))

err = []
nNodesEff = []
MidSetObj = midSet()
interpolant = SGInterpolant(MidSetObj.midSet, knots, lev2knots, interpolationType)
for currLevel in Levels:
    print("Computeing level", currLevel)
    while(interpolant.numNodes < nNodes[currLevel]):
        P = Profit(MidSetObj.margin)
        idMax = np.argmax(P)
        MidSetObj.update(idMax)
        interpolant = SGInterpolant(MidSetObj.getMidSet(), knots, lev2knots, interpolationType, NParallel=1)
    
    uOnSG = interpolant.sampleOnSG(FExa)
    uInterp = interpolant.interpolate(yyRnd, uOnSG)

    errSamples = np.linalg.norm(samplesFExa - uInterp, 2, axis=1) * sqrt(1/nTauRef)
    errCurr = sqrt(np.mean(np.square(errSamples)))
    print("error:", errCurr)
    err.append(errCurr)
    nNodesEff.append(interpolant.numNodes)

    # plt.plot(ttRef, uInterp[0], '.-')
    # plt.plot(ttRef, samplesFExa[0], 'k-')
    # plt.show()
    # for i in range(interpolant.SG.shape[0]):
    #     plt.plot(ttRef, uOnSG[i], '.-')
    #     plt.plot(ttRef, FExa(interpolant.SG[i]), 'k-')
    #     plt.show()


plt.loglog(nNodesEff, err, '.-')
plt.loglog(nNodesEff, np.power(nNodesEff, -1/2), 'k-')
plt.show()




exit()








################################ ERROR COMPUTATION ################################
yyRnd = np.random.normal(0, 1, [NRNDSamples, 1000])
print("Sample reference...")
samplesFExa = list(map(FExa, yyRnd))

def computeErrorSingleLevel(SLApprox, currLevel):
    dtRef = T/nTTRef
    errSamples = np.zeros(len(samplesFExa))
    for ny in range(len(samplesFExa)):
        I = interp1d(tt[currLevel], SLApprox[ny], kind='previous') 
        uInterpCurr = I(ttRef)
        errSamples[ny] = sqrt(dtRef)*np.linalg.norm(samplesFExa[ny] - uInterpCurr, ord=2)
    return sqrt(np.mean(np.square(errSamples)))

################################ CONVERGENCE TEST ################################
err = []
cost = []
MidSetObj = midSet()
interpolant = SGInterpolant(MidSetObj.midSet, knots, lev2knots, interpolationType)
for currLevel in Levels:
    print("Computeing level", currLevel)
    while(interpolant.numNodes < nNodes[currLevel]):
        P = Profit(MidSetObj.margin)
        idMax = np.argmax(P)
        MidSetObj.update(idMax)
        interpolant = SGInterpolant(MidSetObj.getMidSet(), knots, lev2knots, interpolationType, NParallel=1)
    
    FCurr = lambda y: FApprox(y, currLevel)
    uOnSG = interpolant.sampleOnSG(FCurr)
    uInterp = interpolant.interpolate(yyRnd, uOnSG)
    errCurr = computeErrorSingleLevel(uInterp, currLevel)
    print("error:", errCurr)
    err.append(errCurr)
    cost.append(interpolant.numNodes * 2**(3*currLevel))

    plt.plot(tt[currLevel], uInterp[0], '.-')
    plt.plot(ttRef, samplesFExa[0], 'k-')
    plt.show()

plt.loglog(cost, err, '.-')
plt.loglog(cost, np.power(cost, -1/3), 'k-')
plt.show()