import numpy as np


class TPPwCubicInterpolator:
    """Given function samples on tensor product nodes, interpolates with tensor product piecewise cubic polynomial interpolation
    """
    def __init__(self, nodesTuple, F):
        """Initializes the interpolator with nodes and function values

        Args:
            nodesTuple (tuple of 1D array double): k-th entry contains 1D nodes in direction k
            F (Function): given paramter vector, returns function value
        """

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
        """Sample the tensor product Lagrange interpolant on new points xNew

        Args:
            xNew (1D array double): Nodes where to sample the interpolant

        Returns:
            2D array double: Each row is the value of the interpolant on the corresponding row of xNew
        """   

        numX = xNew.shape[0]
        assert(xNew.shape[1] == self.nDims)

        # 1 compute stencil SS corresonding to every node 
        # 2 compute tensor basis functions and format into LL
        SS = np.ones((self.nDims, numX), dtype=int)
        LL = np.ones(tuple(4*np.ones(self.nDims, dtype=int))+(numX,1))
        for n in range(self.nDims):
            # stencil is computed 1 dimension at a time. 
            # assume x to be scalar. to identify its stencil, think of the the first *even* collocation node.
            # If you have many xs, it is faster to determine the stencil to which each x belongs if the array of xs is sorted. 
            # so 1. sort x: 2. find corresponding stencil of sorted sx; 3. sort list of stencil indices by reverse sorting of x
            # NBB first stencil (-\infty, y_2] second collocation nodes; last stencil [y_{n-1}, infty) (y_{N} last collocation node)
            znOriginal = xNew[:,n]
            sorting = np.argsort(znOriginal)
            revSorting = np.argsort(sorting)  #  znOriginal = zn[revSorting]
            zn = znOriginal[sorting]
            xn = self.nodesTuple[n]
            halfxn = xn[0::3]  # stencils' boundaries
            jj = np.zeros(zn.size, dtype=int)
            pPrev = -1
            for i in range(1, halfxn.size):  # start with first stencil; examine position of xs wrt its right boundary
                p = np.searchsorted(zn, halfxn[i], side='right')  # zn[p] is first node to the right of current stencil
                jj[pPrev:p] = i-1  # assign tensil index to current nodes
                pPrev = p
            jj[p:] = i-1  # remaining nodes live in last stencil
            jj = jj[revSorting]
            SS[n, :] = jj

            # compute 4 corresponding basis functions in dim n
            k = (xn.size-1)//3
            nRed = 3*k+1
            xnRed = xn[:nRed:] # "regular" part of nodes array; i.e. largest subarray with 3k+1 length
            x0 = xnRed[jj*3]
            x1 = xnRed[jj*3+1]
            x2 = xnRed[jj*3+2]
            x3 = xnRed[jj*3+3]
            # lastjj = jj[-1]
            # if(xn.size == nRed+1):
            #     x0 = np.append(x0, xn[lastjj*3+1])
            #     x1 = np.append(x1, xn[lastjj*3+2])
            #     x2 = np.append(x2, xn[lastjj*3+3])
            #     x3 = np.append(x3, xn[lastjj*3+4])
            # elif (xn.size == nRed+2):
            #     x0 = np.append(x0, xn[lastjj*3+2])
            #     x1 = np.append(x1, xn[lastjj*3+3])
            #     x2 = np.append(x2, xn[lastjj*3+4])
            #     x3 = np.append(x3, xn[lastjj*3+5])
            # else:
            #     print("what happened? Inconsistent number of nodes for cubic interpolation")
            #     exit

            L0 = ((znOriginal-x1)*(znOriginal-x2)*(znOriginal-x3))/((x0-x1)*(x0-x2)*(x0-x3))  # 1D w length numX
            L1 = ((znOriginal-x0)*(znOriginal-x2)*(znOriginal-x3))/((x1-x0)*(x1-x2)*(x1-x3))
            L2 = ((znOriginal-x0)*(znOriginal-x1)*(znOriginal-x3))/((x2-x0)*(x2-x1)*(x2-x3))
            L3 = ((znOriginal-x0)*(znOriginal-x1)*(znOriginal-x2))/((x3-x0)*(x3-x1)*(x3-x2))

            Ln = np.vstack((L0, L1, L2, L3))
            shapen = np.ones(self.nDims+2, dtype=int)
            shapen[n] = 4
            shapen[-2] = numX
            Ln = np.reshape(Ln, tuple(shapen))
            LL *= Ln

        # 3 format interpolation data
        FF = np.zeros(tuple(4*np.ones(self.nDims, dtype=int))+(numX, self.dimF))
        it = np.nditer(FF[...,0,0],flags=['multi_index'])
        for i in it:
            vecIt = np.asarray(it.multi_index, dtype=int)
            vecIt = vecIt.reshape(-1,1)
            position = 3*SS+vecIt
            FF[it.multi_index] = self.F[tuple(position)] # size numX x dimF
        
        # 4 compute product tensor data and Lagrange basis functions
        p = np.multiply(FF, LL)
        
        # 5 reduce first N dimensions and reshape
        s = np.sum(p, axis=tuple(n for n in range(self.nDims)))
        return np.reshape(s, (numX, self.dimF))