import numpy as np
from math import sin, sqrt

def compute_error(Fexa, FApprox):
    
    yyRnd = np.random.normal(0, 1, [NSamples, dimParam])
    
    

K = 4  # number of levels -1
FExa = lambda y : sin(np.sum(y))
FApprox = lambda y, N : FExa(y) + 1/N

# error computation
NSamples = 100
dimParam = 1000
yyRnd = np.random.normal(0, 1, [NSamples, dimParam])
samplesFExa = map(FExa, yyRnd)
def compute_error(FAppr):
    errSamples = FAppr - samplesFExa
    return sqrt(1/NSamples * np.sum(np.square(errSamples)))

# compute approximation
interpolant = MLInterpolant()
FOnSGML = interpolant.sample(FApprox)
# interpolate at some point
FInterp = interpolant.interpolate(yyRnd, FOnSGML)
# compute error
err = compute_error(FInterp)
print(err)