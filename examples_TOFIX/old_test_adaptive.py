import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
from sgmethods.nodes_1d import unbounded_nodes_nested
from sgmethods.multi_index_sets import midSet
from sgmethods.compute_error_indicators import compute_aposteriori_estimator_reduced_margin
from sgmethods.sparse_grid_interpolant import SGInterpolant

"""An example using the [Gerstner-Griebel] adaptivity"""


N = 2
dimF = 3
def F(x):
    return np.sin(np.sum(x * 10**(-np.linspace(0,x.size, x.size)) ))+1 * np.array([1., 2., 3.])
def physicalNorm(x):
    return np.linalg.norm(x, 2)

# interpolant parameters
lev2knots = lambda n: 2**(n+1)-1
knots = unbounded_nodes_nested
interpolationType = "linear"
NParallel = 1

# adaptive loop
NLCExpansion = 2**10
NRNDSamples = 100
yyRnd = np.random.normal(0, 1, [NRNDSamples, NLCExpansion])
maxNumNodes = 500
I = midSet(maxN=2)
oldSG = None
uOnSG = None
w=0
while True:
    # COMPUTE
    interpolant = SGInterpolant(I.midSet, knots, lev2knots, interpolationType=interpolationType, NParallel=NParallel)
    if(interpolant.numNodes > maxNumNodes):
        break
    print("# nodes:", interpolant.numNodes, "\nNumber effective dimensions:", I.N)
    uOnSG = interpolant.sampleOnSG(F, dimF, oldSG, uOnSG)
    uInterp = interpolant.interpolate(yyRnd, uOnSG)
    # ESTIMATE: 
    estimator_reduced_margin = compute_aposteriori_estimator_reduced_margin(I, knots, lev2knots, F, dimF, interpolant.SG, uOnSG, yyRnd, physicalNorm, NRNDSamples, uInterp, interpolationType=interpolationType, NParallel=NParallel)
    # MARK: Maximum profit
    idMax = np.argmax(estimator_reduced_margin)  # NBB index in REDUCED margin
    mid = I.reducedMargin[idMax, :]
    idxMargin = np.where(( I.margin==mid).all(axis=1) )[0][0]
    # REFINE:
    I.update(idxMargin)
    oldSG = interpolant.SG
    w+=1

# plot in 2D
if w>0:
    plt.plot(I.midSet[:,0], I.midSet[:,1], '.')
    plt.plot(I.margin[:,0], I.margin[:,1], 'go')
    plt.plot(I.reducedMargin[:,0], I.reducedMargin[:,1], 'ro')
    for i in range(I.reducedMargin.shape[0]):
        m = I.reducedMargin[i, :]
        plt.text(m[0], m[1], str(estimator_reduced_margin[i]))
    plt.show()