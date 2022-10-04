from math import pi
import numpy as np


NLCExpansion = 2**10
NRNDSamples=10
yyRnd = np.random.normal(0, 1, [NRNDSamples, NLCExpansion])
ww = (2*pi)**(-0.5*NLCExpansion) * np.exp(-0.5*(np.linalg.norm(yyRnd, 2, axis=1)**2))
print(ww)