import numpy as np


class TPLagrangeInterpolator:
    def __init__(self, nodesTuple, F):

        # formatting input
        if(len(F.shape) == len(nodesTuple)):  # F is scalar field
            F = np.reshape(F, F.shape + (1,))
        
        self.nodesTuple = nodesTuple  # tuple of length N, each entry is a vector of 1D interpolation nodes
        self.F = F  # N+1 Dimensional tensor containing values of data to interpolate (each data point may be vector of length dimData, so +1)
        
        self.nDims = len(nodesTuple)
        self.nNodesDims = tuple(len(x) for x in self.nodesTuple)
        self.dimF = F.shape[-1]
        
        # sanity check
        assert(self.nNodesDims == F.shape[:-1:])

        # compute \lambda_i coefficients
        self.lambdaCoeffs = ()  # tuple of length N, each entry is 1/( \prod_{j\neq i} x_i-x_j}) i.e. coeffcieints appearing in barycentric interpolation formula [Trefetten]
        for n in range(self.nDims):
            lCurrD = np.ones(self.nNodesDims[n])
            currNodes = self.nodesTuple[n]
            for i in range(self.nNodesDims[n]):
                wCurr = np.repeat(True, currNodes.shape)
                wCurr[i] = False
                lCurrD[i] = np.prod(currNodes[i] - currNodes, where=wCurr) 
            lCurrD = 1/lCurrD
            self.lambdaCoeffs = self.lambdaCoeffs + (lCurrD,)

    def __call__(self, xNew):
        # xNew array shaped (numX, N)
        # OUTPUT 2D array of size (numX, dimData)

        numX = xNew.shape[0]
        assert(xNew.shape[1] == self.nDims)

        # change shape F to cardY1 x ... x cardY_N x 1 x dimData for later product with another tensor
        self.F = np.reshape(self.F, (self.nNodesDims) + (1,) + (self.dimF,))
        # compute tensor Lagrange basis functions
            
            
        lagrangeBasis = np.ones((self.nNodesDims)+(numX,))
        for n in range(self.nDims):
            # compute 1D basis functions: for each get is 2D array shaped cardYn x numX
            xCurr = np.reshape(xNew[:, n], (1,-1))
            nodesCurr = np.reshape(self.nodesTuple[n], (-1,1))
            LCurr = np. prod(xCurr - nodesCurr, axis=0)
            currLambda = np.reshape(self.lambdaCoeffs[n], (-1,1))
            lagBasisCurr = LCurr * currLambda / (xCurr - nodesCurr)
            # reshape it to 1 x ... x cardYn x ... x numX
            shapeCurr = list(1 for _ in range(self.nDims))
            shapeCurr[n] = self.nNodesDims[n]
            shapeCurr.append(numX)
            lagBasisCurr = np.reshape(lagBasisCurr, tuple(shapeCurr))
            # exterior product ot get tensor of basis functions shaped cardY1 x ... x cardY_N x numX
            lagrangeBasis = np.multiply(lagrangeBasis, lagBasisCurr)
        lagrangeBasis = np.reshape(lagrangeBasis, lagrangeBasis.shape+(1,))
        # compute product tensor data and Lagrange basis functions
        p = np.multiply( self.F, lagrangeBasis )
        # reduce first N dimensions and reshape
        s = np.sum(p, axis=tuple(n for n in range(self.nDims)))
        return np.reshape( s,(numX, self.dimF))