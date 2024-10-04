import numpy as np 
import matplotlib.pyplot as plt
from math import pi, sqrt
import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
from SGMethods.nodes_1d import CCNodes
from SGMethods.tp_lagrange import TPLagrangeInterpolator
from SGMethods.multi_index_sets import anisoSmolyakMidSet
from SGMethods.sparse_grid_interpolant import SGInterpolant
from tutorials.parametric_affine_poisson import sampleParametricPoisson, computeErrorSamplesPoisson

"""Tutorial on sparse grid interpolation with SGMethods. 
We see how to use SGMethods to approximate the parametric affine diffusion Poisson problem."""


# Problem parameters
np.random.seed(36157)
N = 10  # number of parametric dimensions
C = pi**2/6 * 1.1  # normalization diffusion coefficient
a_min = 1 - -pi**2/(6*C)  # minimum diffusion coefficient
nH = 16 # mesh resolution i.e. # edges per side of square domain mesh
def F(y):  # target function sampler
    return sampleParametricPoisson(y, nH=nH, orderDecayCoefficients=2)  # Sample the parametric Poisson problem

# Error computation
yyRnd = np.random.uniform(-1, 1, (128, 1000))  # Random parameters for MC error estimation
uExa = np.squeeze(np.array(list(map(F, yyRnd))))  # Random sample of exact function

# Sparse grid parameters
TPInterpolant = lambda nodesTuple, fOnNodes : TPLagrangeInterpolator(nodesTuple, fOnNodes)
lev2knots = lambda n: np.where(n==0, 1, 2**n+1)  # doubling rule:  1 if n==0 else 2**(n)-1
knots = lambda n : CCNodes(n)  # Clenshaw-Curtis nodes

# Anisotrpopy vector for parametric Poisson
gamma = lambda N : 1/(C*a_min*np.linspace(1, N, N)**2)
tau = lambda N : 1/(2*gamma(N))
anisoVec = lambda N : 0.5* np.log(1+tau(N))  
midSet = anisoSmolyakMidSet(w=2, N=N, a=anisoVec(N))  # Anisotropic Smolyak multi-index set
interpolant = SGInterpolant(midSet, knots, lev2knots, TPInterpolant, NParallel=8)  # Spars grid interpolant
uOnSG = interpolant.sampleOnSG(F)  # Sample the function on the sparse grid
uInterp = interpolant.interpolate(yyRnd, uOnSG)  # Compute the interpolated value
errSamples = computeErrorSamplesPoisson(uInterp, uExa, nH)  # Compute error samples in H^1_0(D)
errorL2 = np.mean(errSamples)  # L^2 error in parameter space
print("Error:", errorL2)
print("Sparse grid:\n", interpolant.SG)

# refine SG and compute refinement
midSet2 = anisoSmolyakMidSet(w=4, N=N, a=anisoVec(N))
interpolant2 = SGInterpolant(midSet2, knots, lev2knots, TPInterpolant=TPInterpolant, NParallel=8)
print("Number of collocation nodes:", interpolant.numNodes)
# Sample on refined sparse grid and recycle old values
uOnSG = interpolant.sampleOnSG(F, oldXx=interpolant.SG, oldSamples=uOnSG)
uInterp = interpolant.interpolate(yyRnd, uOnSG,)
errSamples = computeErrorSamplesPoisson(uInterp, uExa, nH)
errorL2 = sqrt(np.mean(np.square(errSamples)))
print("Error:", errorL2)
print("Sparse grid:\n", interpolant2.SG)