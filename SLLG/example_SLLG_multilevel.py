import numpy as np
from math import sin, sqrt
import sys, os
from multiprocessing import Pool
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 

sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
from SGMethods.SGInterpolant import SGInterpolant
from SGMethods.ScalarNodes import unboundedKnotsNested
from SGMethods.MidSets import midSet
from SGMethods.MLInterpolant import MLInterpolant
from SLLG.sample_LLG_function_noise_2 import sample_LLG_function_noise_2
from SLLG.error_sample_pb2_L2norm import error_sample_pb2_L2norm
from SLLG.profits import PorfitL1Holo

from dolfin import UnitSquareMesh, VectorElement, FunctionSpace, PETScDMCollection, Function
from scipy.interpolate import interp1d


"""Multileve example for the parametric LLG problem. """

T=1
FEMOrder = BDFOrder = 1
nRefs = 5  # 
NParallel = 32
d = 3
NRNDSamples = 1024  #   4  # 
NLCExpansion = 2**10
p=2
interpolationType = "linear"

#reference solution
NhRef = 2**(2+nRefs+1)
NtauRef = NhRef*8
def F(x):
    return sample_LLG_function_noise_2(x, NhRef, NtauRef, T, FEMOrder, BDFOrder)

#approximate solution
kk = np.linspace(0, nRefs-1, nRefs, dtype=int)
Nh = 2**(2+kk)
Ntau = Nh*8
def FApprox(x, k):
    NhC = Nh[k]
    NtauC = Ntau[k]
    return sample_LLG_function_noise_2(x, NhC, NtauC, T, FEMOrder, BDFOrder)

# Error estimation

def physicalError(u, uExa, NhRef, NtauRef, T, FEMOrder, BDFOrder):
    return error_sample_pb2_L2norm(u, uExa, NhRef, NtauRef, NhRef, NtauRef, T, FEMOrder, BDFOrder)

def computeErrorMultilevel(uExa, MLTerms, NhRef, NtauRef, Nh, Ntau, FEMOrder, BDFOrder):
    """INPUT uExa 2D array
             Nh and Ntau are arrays of lenght nLevevls!
             MLTERMS list of 2D arrays; k-th entry has shape nY x kth physical space size """
    # uExa is ready to be used, The MLTerms must be interpolate indo reference space.
    # 1 pre-processing: generate all needed for the interpolation
    meshRef = UnitSquareMesh(NhRef, NhRef)
    ttref = np.linspace(0, T, NtauRef+1)
    Pr3 = VectorElement('Lagrange', meshRef.ufl_cell(), FEMOrder, dim=3)
    V3Ref = FunctionSpace(meshRef, Pr3)
    dimV3Ref = V3Ref.dim()
    tt = []
    mesh = []
    Pr3 = []
    V3 = []
    dimV3 = []
    M = []
    Ntt = Ntau + 1  # by definition, Ntau = T/tau; Ntt is the number of timesteps!
    for k in range(len(MLTerms)):
        tt.append(np.linspace(0, T, Ntt[k]))
        mesh.append(UnitSquareMesh(Nh[k], Nh[k]))
        Pr3.append(VectorElement('Lagrange', mesh[k].ufl_cell(), FEMOrder, dim=3))
        V3.append(FunctionSpace(mesh[k], Pr3[k]))
        dimV3.append(V3[k].dim())
        if dimV3[k] != dimV3Ref:
                M.append(PETScDMCollection.create_transfer_matrix(V3[k], V3Ref))
    # 2 interpolate and compute error sample
    errSamples = np.zeros(len(uExa))
    for ny in range(len(uExa)):
        # uInterpCurr is 1D array of size of reference physical space
        uInterpCurr = 0*uExa[0]
        for k in range(len(MLTerms)):
            dofs = MLTerms[k][ny]  # current dofs at level k to be inteprolated in reference space
            if dofs.size != Ntt[k]*dimV3[k]:
                print("Error loading ML expnasion, level ", k, ":", dofs.size, " != ", Ntt[k], " * ", dimV3[k])
                assert(len(dofs) == Ntt[k]*dimV3[k])
            # 1 inteprolate in space
            assert(FEMOrder==1)
            dofsref2 = np.ndarray((dimV3Ref, Ntt[k]))
            tmp = Function(V3[k])
            for niter in range(Ntt[k]):
                if V3[k].dim() == V3Ref.dim():  # here assume FE space are NESTED
                    dofsref2[:, niter] = dofs[niter*dimV3[k]:(niter+1)*dimV3[k]:]
                else:
                    tmp.vector()[:] = dofs[niter*dimV3[k]:(niter+1)*dimV3[k]:]
                    v = M[k]* tmp.vector()
                    dofsref2[:, niter] = v[:]
            # 2 interpolate time
            assert(BDFOrder==1)
            F = interp1d(tt[k], dofsref2, assume_sorted=True)
            dofscurr = F(ttref)
            dofscurr = dofscurr.flatten('F')
            uInterpCurr += dofscurr
        errSamples[ny] = physicalError(uInterpCurr, uExa[ny], NhRef, NtauRef, T, FEMOrder, BDFOrder)
    # 3 compute gaussina norm in parameter space
    errSamples = np.array(errSamples)
    return sqrt(np.mean(np.square(errSamples)))
#########################################################################

yyRnd = np.random.normal(0, 1, [NRNDSamples, NLCExpansion])
print("Parallel random sampling")
pool = Pool(NParallel)
uExa = pool.map(F, yyRnd)

#compute sequence of approximations, error
lev2knots = lambda nu: 2**(nu+1)-1
knots = lambda m : unboundedKnotsNested(m,p=p)
totalCost = []
err = []
for nLevels in range(1, 1+ nRefs):
    # generate the list of SG interpolants "single level". I select them iteratively; Each has to have twoce the numer of codes than the previous one
    SLIL = []
    MidSetObj = midSet()
    SGICurr = SGInterpolant(MidSetObj.midSet, knots, lev2knots, interpolationType)  # has only 1 collocation node
    SLIL.append(SGICurr)
    for k in range(1, nLevels):
        SGICurr = SGInterpolant(MidSetObj.midSet, knots, lev2knots, interpolationType)
        while(SGICurr.numNodes < 2**k):
            P = PorfitL1Holo(MidSetObj.margin, p)
            idMax = np.argmax(P)
            MidSetObj.update(idMax)
            SGICurr = SGInterpolant(MidSetObj.midSet, knots, lev2knots, interpolationType)
        SLIL.append(SGICurr)

    interpolant = MLInterpolant(SLIL)
    FOnSGML = interpolant.sample(FApprox)
    MLTerms = interpolant.getMLTerms(yyRnd, FOnSGML)

    errCurr = computeErrorMultilevel(uExa, MLTerms, NhRef, NtauRef, Nh, Ntau, FEMOrder, BDFOrder)
    err.append(errCurr)
    
    # compute cost
    kk = np.linspace(0,nLevels-1, nLevels)
    hh = 2**(-kk-3)
    tautau = hh*8
    costKK = np.power(hh, -d)*tautau  # the cost of the optimal elliptic FE solver, times the number of timesteps
    totalCost.append(interpolant.totalCost(costKK))
    np.savetxt('convergenge_multilevel_pb2.csv',np.transpose(np.array([totalCost, err])), delimiter=',')

# converngence
print("error:", err)
print("cost:", totalCost)
plt.loglog(totalCost, err, '.-')
plt.loglog(totalCost, np.power(totalCost, -1/d), 'k-')
plt.plot()