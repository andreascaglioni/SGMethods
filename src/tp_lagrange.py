import numpy as np


class TPLagrangeInterpolator:
    """Given function samples on tensor product nodes, interpolates with tensor 
    product Lagrange interpolation computed with barycentric interpolation 
    formula, see 
    Trefetten-Approximation Theory and Approximation Practice('19)
    """    
    def __init__(self, nodesTuple, F):
        """Initializes the interpolator with nodes and function values.

        Args:
            nodesTuple (tuple of 1D array double): k-th entry contains 1D nodes
                in direction k
            F (Function): Given paramter vector, returns function value
        """

        # Format input
        if(len(F.shape) == len(nodesTuple)):  # If F is scalar, turn to N x 1
            F = np.reshape(F, F.shape + (1,))
        self.nodesTuple = nodesTuple 
        self.F = F
        self.nDims = len(nodesTuple)
        self.nNodesDims = tuple(len(x) for x in self.nodesTuple)
        self.dimF = F.shape[-1]

        # Sanity check
        assert(self.nNodesDims == F.shape[:-1:])

        # Compute \lambda_i coefficients
        # \labda_i = (\prod_{j\neqi} x_i-x_j})^{-1}
        # Put them in tuple of length N
        self.lambdaCoeffs = ()
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
        """Sample the tensor product Lagrange interpolant on new points xNew

        Args:
            xNew (1D array double): Nodes where to sample the interpolant

        Returns:
            2D array double: Each row is the value of the interpolant on the 
            corresponding row of xNew
        """        

        numX = xNew.shape[0]
        assert(xNew.shape[1] == self.nDims)
        # Change shape F to cardY1 x ... x cardY_N x 1 x dimData 
        self.F = np.reshape(self.F, (self.nNodesDims) + (1,) + (self.dimF,))
        # Compute tensor Lagrange basis functions
        lagrangeBasis = np.ones((self.nNodesDims)+(numX,))
        for n in range(self.nDims):
            # Compute 1D basis functions: for each, get 2D array(cardYn x numX)
            xCurr = np.reshape(xNew[:, n], (1,-1))
            nodesCurr = np.reshape(self.nodesTuple[n], (-1,1))
            LCurr = np. prod(xCurr - nodesCurr, axis=0)
            currLambda = np.reshape(self.lambdaCoeffs[n], (-1,1))
            lagBasisCurr = LCurr * currLambda / (xCurr - nodesCurr)
            # Reshape it to 1 x ... x 1 x cardYn x 1 x ... x 1 x numX
            shapeCurr = list(1 for _ in range(self.nDims))
            shapeCurr[n] = self.nNodesDims[n]
            shapeCurr.append(numX)
            lagBasisCurr = np.reshape(lagBasisCurr, tuple(shapeCurr))
            # Exterior product ot get tensor of basis functions shaped 
            # cardY1 x ... x cardY_N x numX
            lagrangeBasis = np.multiply(lagrangeBasis, lagBasisCurr)
        lagrangeBasis = np.reshape(lagrangeBasis, lagrangeBasis.shape+(1,))
        # Compute product tensor data and Lagrange basis functions
        p = np.multiply( self.F, lagrangeBasis )
        # Reduce first N dimensions and reshape
        s = np.sum(p, axis=tuple(n for n in range(self.nDims)))
        return np.reshape( s,(numX, self.dimF))