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


################################ PARAMETERS ################################
T = 1
NParallel = 1
p = 2  # NBB degrere + 1
interpolationType = "linear"
lev2knots = lambda nu: 2**(nu+1)-1
knots = lambda m : unboundedKnotsNested(m,p=p)
Profit = lambda nu: ProfitMix(nu, p)
NNLevels = [1,2,3,4,5,6]
SGCards = [[1],
           [1, 1],
           [10,3,1],
           [82,18,4,1],
           [602, 131, 27, 7, 2],
           [1500, 887, 193, 42, 10, 2]] 

################################ EXACT AND APPROXIMATE SAMPLERS ################################
nTau = 2**np.array(NNLevels,dtype=int)
tau = 1/nTau
nTT = nTau+1
tt = []
for k in range(len(nTau)):
    tt.append(np.linspace(0,T, nTT[k]))
def FApprox(y, k):  # piecewise const approximation DISCRETE
    W = param_LC_Brownian_motion(tt[k], y, T)
    return np.sin(W)

nTauRef = nTau[-1]*2
nTTRef = nTauRef+1
ttRef = np.linspace(0,T,nTTRef)
def FExa(y):  # piecewise const approximation REFERENCE
    WRef = param_LC_Brownian_motion(ttRef, y, T)
    return np.sin(WRef)


################################ ERROR COMPUTATION ################################
NSamples = 5
yyRnd = np.random.normal(0, 1, [NSamples, 1000])
print("Sample reference...")
samplesFExa = list(map(FExa, yyRnd))

def computeErrorMultilevel(MLTerms):
    dtRef = T/nTTRef
    errSamples = np.zeros(len(samplesFExa))
    for ny in range(len(samplesFExa)):
        uInterpCurr = np.zeros(samplesFExa[0].size)
        for k in range(len(MLTerms)):
            # inteprolate in time
            I = interp1d(tt[k], MLTerms[k][ny], kind='previous')   # current dofs at level k inteprolated in reference space
            dofscurr = I(ttRef)
            uInterpCurr += dofscurr
        errSamples[ny] = sqrt(dtRef)*np.linalg.norm(samplesFExa[ny] - uInterpCurr, ord=2)  # L2 norm in time 
    return sqrt(np.mean(np.square(errSamples)))

################################ CONVERGENCE TEST ################################
err = []
totalCost = []
for i in range(len(NNLevels)):
    nLevels = NNLevels[i]
    print(nLevels, "level(s)")

    # Generate single level interpolants
    SLIL = []
    MidSetObj = midSet()
    SGICurr = SGInterpolant(MidSetObj.midSet, knots, lev2knots, interpolationType, NParallel=NParallel)
    SLIL.append(SGICurr)
    print("generating single level inteprlants...")
    for k in range(1, nLevels):
        SGICurr = SGInterpolant(MidSetObj.midSet, knots, lev2knots, interpolationType, NParallel=NParallel)
        while(SGICurr.numNodes < SGCards[i][k]):
            P = Profit(MidSetObj.margin)
            idMax = np.argmax(P)
            MidSetObj.update(idMax)
            SGICurr = SGInterpolant(MidSetObj.getMidSet(), knots, lev2knots, interpolationType, NParallel=NParallel)
        SLIL.append(SGICurr)
    print("define ML inteprolant...")
    # define interpolant ML
    interpolant = MLInterpolant(SLIL)
    print("sampleML inteprolant...")
    FOnSGML = interpolant.sample(FApprox)
    
    # compute error
    print("error computation...")
    MLTerms = interpolant.getMLTerms(yyRnd, FOnSGML)
    errCurr = computeErrorMultilevel(MLTerms)
    err.append(errCurr)
    print("error", errCurr)
    # compute cost
    kk = np.linspace(0,nLevels-1, nLevels)
    hh = 2**(-kk)
    costKK = np.power(hh, -3)
    totalCost.append(interpolant.totalCost(costKK))
    

plt.loglog(totalCost, err, '.-')
plt.loglog(totalCost, np.power(totalCost, -1/3), 'k-')
plt.show()