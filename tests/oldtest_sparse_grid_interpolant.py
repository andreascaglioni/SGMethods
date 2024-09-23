import unittest
import numpy as np
from math import sqrt
from multiprocessing import Pool
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
from SGMethods.multi_index_sets import midSet
from SGMethods.sparse_grid_interpolant import SGInterpolant
from SGMethods.nodes_1d import unboundedKnotsNested
from SLLG.expansions_Brownian_motion import param_LC_Brownian_motion
from SLLG.profits import ProfitMix
from scipy.interpolate import RegularGridInterpolator

class TestInterpolatedValue(unittest.TestCase):
    def test_list_int(self):
        """
        Test that sin(W) is interpolated correctly.
        I use ProfitMix, even if better could be done for this simple function
        """
        # EXPECTED ERROR SAVED FROM PREVIOUS RUN
        savedErr = np.array([0.54047893, 0.35196493, 0.27934259, 0.22020262, 0.20271274, 0.17607892, 0.16306789, 0.11849408, 0.10750849, 0.1028781,  0.08775988])

        # NUMERICS PARAMETERS
        NParallel=8
        NRNDSamples = 256
        np.random.seed(1607)
        maxNumNodes = 128
        p=2
        # interpolationType =  "linear"
        TPInterpolant = lambda activeNodesTuple, fOnNodes : RegularGridInterpolator(activeNodesTuple, fOnNodes, method='linear', bounds_error=False, fill_value=None)
        lev2knots = lambda n: 2**(n+1)-1
        knots = lambda n : unboundedKnotsNested(n, p=p)
        Profit = lambda nu : ProfitMix(nu, p)

        # FUNCTION SAMPLER
        Nt = 100
        tt = np.linspace(0, 1, Nt)
        dt = 1/Nt
        def F(x):
            return np.sin(param_LC_Brownian_motion(tt, x, 1))

        # ERROR COPMUTATION
        def computeL2Error(uExa, Iu):
            assert(uExa.shape == Iu.shape)
            spaceNorm = lambda x : sqrt(dt)*np.linalg.norm(x, ord=2, axis=1)  # L2 norm in time 
            errSample = spaceNorm(uExa-Iu)
            return sqrt(np.mean(np.square(errSample)))
        yyRnd = np.random.normal(0, 1, [NRNDSamples, 1000])  # infinite paramter vector
        uExa = np.array(list(map(F, yyRnd)))


        # CONVERGENCE TEST
        err = np.array([])
        nNodes = np.array([])
        nDims = np.array([])
        w=0
        I = midSet()
        oldSG = None
        uOnSG = None
        while True:
            print("Computing w  = ", w)
            interpolant = SGInterpolant(I.midSet, knots, lev2knots, TPInterpolant=TPInterpolant, NParallel=1)
            if(interpolant.numNodes > maxNumNodes):
                break
            midSetMaxDim = np.amax(I.midSet, axis=0)
            print("# nodes:", interpolant.numNodes, "\nNumber effective dimensions:", I.N, "\nMax midcomponents:", midSetMaxDim)
            uOnSG = interpolant.sampleOnSG(F, oldXx=oldSG, oldSamples=uOnSG)
            uInterp = interpolant.interpolate(yyRnd, uOnSG)
            # COMPUTE ERROR
            err = np.append(err, computeL2Error(uExa, uInterp))
            print("Error:", err[-1])
            nNodes = np.append(nNodes, interpolant.numNodes)
            nDims = np.append(nDims, I.N)
            oldSG = interpolant.SG
            ncps = interpolant.numNodes
            while interpolant.numNodes < sqrt(2)*ncps:
                P = Profit(I.margin)
                idMax = np.argmax(P)
                I.update(idMax)
                interpolant = SGInterpolant(I.midSet, knots, lev2knots, TPInterpolant==TPInterpolant, NParallel=NParallel)
                if(interpolant.numNodes > maxNumNodes):
                    break
            w+=1
        print("Error: ", err)
        rates = -np.log(err[1::]/err[:-1:])/np.log(nNodes[1::]/nNodes[:-1:])
        print("Rate:",  rates)
        
        normDiffErr = np.linalg.norm(err-savedErr)
        self.assertAlmostEqual(normDiffErr, 0.)
        

if __name__ == '__main__':
    unittest.main()