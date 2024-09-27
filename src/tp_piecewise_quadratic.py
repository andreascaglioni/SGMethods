import numpy as np


class TPPwQuadraticInterpolator:
    """Given function samples on tensor product nodes, interpolates with tensor 
        product piecewise quadratic interpolation.
    """    
    def __init__(self, nodesTuple, F):
        """Initializes the interpolator with nodes and function values.

        Args:
            nodesTuple (tuple of 1D array double): k-th entry contains 1D nodes 
                in direction k.
            F (Function): given paramter vector, returns function value.
        """

        # Format input if F is scalar field
        if(len(F.shape) == len(nodesTuple)): 
            F = np.reshape(F, F.shape + (1,))

        self.nodesTuple = nodesTuple
        self.F = F
        self.nDims = len(nodesTuple)
        self.nNodesDims = tuple(len(x) for x in self.nodesTuple)
        self.dimF = F.shape[-1]
        # sanity check
        assert(self.nNodesDims == F.shape[:-1:])

    def __call__(self, xNew):
        """Sample the tensor product interpolant on new points xNew

        Args:
            xNew (1D array double): Nodes where to sample the interpolant

        Returns:
            2D array double: Each row is the value of the interpolant on the 
                corresponding row of xNew.
        """   
        
        numX = xNew.shape[0]
        assert(xNew.shape[1] == self.nDims)

        # 1 compute stencil SS corresonding to every node 
        # 2 compute tensor basis functions and format into LL
        LL = np.ones(tuple(3*np.ones(self.nDims, dtype=int))+(numX,1))
        SS = np.ones((self.nDims, numX), dtype=int)
        for n in range(self.nDims):
            # stencil is computed 1 dimneison at a time. 
            # assume x scalar. To identify its stencil, think of the the first 
            #   even collocation node to the left.
            # For many x, it is faster to determine the stencil to which each x
            #   belongs iff the array of xs is sorted. 
            # so 
            #   1. sort x;
            #   2. find corresponding stencil of sorted sx; 
            #   3. sort list of stencil indices by reverse sorting of x.
            znOriginal = xNew[:,n]
            sorting = np.argsort(znOriginal)
            revSorting = np.argsort(sorting)
            zn = znOriginal[sorting]
            xn = self.nodesTuple[n]
            halfxn = xn[0::2]
            # jj are the indices, for aeach element in scalar interpol point
            # zn, of the knot to the left in the scalar knots sequence xn
            jj = np.zeros(zn.size, dtype=int)
            pPrev = -1
            p=0
            for i in range(1, halfxn.size):
                p = np.searchsorted(zn, halfxn[i], side='right')
                jj[pPrev:p] = i-1
                pPrev = p
            jj[p:] = halfxn.size-1
            jj = jj[revSorting]
            SS[n, :] = jj
            # compute 3 corresponding basis functions in dim n
            x0 = xn[jj*2]
            x1 = xn[jj*2+1]
            x2 = xn[jj*2+2]
            L0 = ((znOriginal-x1)*(znOriginal-x2))/((x0-x1)*(x0-x2))
            L1 = ((znOriginal-x0)*(znOriginal-x2))/((x1-x0)*(x1-x2))
            L2 = ((znOriginal-x0)*(znOriginal-x1))/((x2-x0)*(x2-x1))

            Ln = np.vstack((L0, L1, L2))
            shapen = np.ones(self.nDims+2, dtype=int)
            shapen[n] = 3
            shapen[-2] = numX
            Ln = np.reshape(Ln, tuple(shapen))
            LL *= Ln

        # 3 format interpolation data
        FF = np.zeros(tuple(3*np.ones(self.nDims, dtype=int))+(numX, self.dimF))
        it = np.nditer(FF[...,0,0],flags=['multi_index'])
        for i in it:
            vecIt = np.asarray(it.multi_index, dtype=int)
            vecIt = vecIt.reshape(-1,1)
            position = 2*SS+vecIt
            FF[it.multi_index] = self.F[tuple(position)] # numX x dimF
        
        # 4 compute product tensor data and Lagrange basis functions
        p = np.multiply(FF, LL)
        
        # 5 reduce first N dimensions and reshape
        s = np.sum(p, axis=tuple(n for n in range(self.nDims)))
        return np.reshape( s,(numX, self.dimF))