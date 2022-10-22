import numpy as np


class TPPwQuadraticInterpolator:
    def __init__(self, nodesTuple, F):
        # formatting input if F is scalar field
        if(len(F.shape) == len(nodesTuple)): 
            F = np.reshape(F, F.shape + (1,))
        self.nodesTuple = nodesTuple  # tuple of length N, each entry is a vector of 1D interpolation nodes
        self.F = F  # N+1 Dimensional tensor containing values of data to interpolate (each data point may be vector of length dimData, so +1)
        self.nDims = len(nodesTuple)
        self.nNodesDims = tuple(len(x) for x in self.nodesTuple)
        self.dimF = F.shape[-1]
        # sanity check
        assert(self.nNodesDims == F.shape[:-1:])

    def __call__(self, xNew):
        # xNew array shaped (numX, N)
        # OUTPUT 2D array of size (numX, dimData)
        
        numX = xNew.shape[0]
        assert(xNew.shape[1] == self.nDims)
        

        # 1 compute tensor basis functions and format interpolation data
        # 2 format collocation samples so they are ready for prodcut and reduction
        LL = np.ones(3*np.ones(self.nDims)+(numX, 1))
        JJ = np.ones(tuple(3*np.ones(self.nDims))+(numX,))
        for n in range(self.nDims):
            zn = xNew[:,n]
            xn = self.nodesTuple[n]
            halfxn = xn[0::2]
            assert(np.all(zn[1:]<=zn[0:-1]))
            # find stencil of each element in nth component xNew. Stencil jj is np.array length numX
            jj = np.zeros(zn.size, dtype=int)
            pPrev = -1
            for i in range(1, halfxn.size):
                p = np.searchsorted(zn, halfxn[i], side='right')
                jj[pPrev:p] = i-1
                pPrev = p
            jj[p:] = i-1
            # compute 3 corresponding basis functions in dim n
            x0 = xn[jj*2]
            x1 = xn[jj*2+1]
            x2 = xn[jj*2+2]
            L0 = ((zn-x1)*(zn-x2))/((x0-x1)*(x0-x2))
            L1 = ((zn-x0)*(zn-x2))/((x1-x0)*(x1-x2))
            L2 = ((zn-x0)*(zn-x1))/((x2-x0)*(x2-x1))

            # compute 1D basis functions: for each get 2D array shaped cardYn x numX
            xCurr = np.reshape(xNew[:, n], (1,-1))
            nodesCurr = np.reshape(self.nodesTuple[n], (-1,1))
            LCurr = np. prod(xCurr - nodesCurr, axis=0)
            currLambda = np.reshape(self.lambdaCoeffs[n], (-1,1))
            lagBasisCurr = LCurr * currLambda / (xCurr - nodesCurr)

            # reshape it to 1 x ... x 1 x 3 x 1 x ... x 1 x numX
            shapeCurr = list(1 for _ in range(self.nDims))
            shapeCurr[n] = self.nNodesDims[n]
            shapeCurr.append(numX)
            lagBasisCurr = np.reshape(lagBasisCurr, tuple(shapeCurr))

            # exterior product ot get tensor of basis functions shaped 3 x ... x 3 x numX
            lagrangeBasis = np.multiply(lagrangeBasis, lagBasisCurr)

        lagrangeBasis = np.reshape(lagrangeBasis, lagrangeBasis.shape+(1,))

        # 3 format inteprolation data
        self.F = np.reshape(self.F, (self.nNodesDims) + (1,) + (self.dimF,))
        FF = np.zeros()
        F = np.zeros(tuple(3*np.ones(self.nDims))+(numX, self.dimF))
        FF = self.F[JJ,:]
        
        
        # 4 compute product tensor data and Lagrange basis functions
        p = np.multiply(FF, LL)
        
        # 5 reduce first N dimensions and reshape
        s = np.sum(p, axis=tuple(n for n in range(self.nDims)))
        return np.reshape( s,(numX, self.dimF))