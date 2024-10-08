"""Examples of tenosr products interpolatio operators. They will be wrapped
in the interponat wrapper class and used for sparse grid interpolation through
the inclusion-exclusion formula.
The user can implement new interpolation operators ina alalgogous classes.
IMPORTANT: The input and output must be compatible with the wrapper, so same
as the examples in this file! In particular the following methods are needed:
- __init__(self, nodes_tuple, f_on_nodes)
- __call__(self, x_new): Input is 2D array double (each row is a parameter 
    vector on which to interpolate), output is 2D array double (each row is the 
    value of the interpolant on the corresponding parameter in x_new).
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

class TPPwLinearInterpolator:
    """Given function samples on tensor product nodes, interpolates with tensor 
        product piecewise linear interpolation.
    """
    def __init__(self, nodes_tuple, f_on_nodes):
        """Initializes the interpolator with nodes and function values.

        Args:
            nodesTuple (tuple of 1D array double): k-th entry contains 1D nodes 
                in direction k.
            F (Function): given paramter vector, returns function value.
        """
        self.nodes_tuple = nodes_tuple
        self.f_on_nodes = f_on_nodes

    def __call__(self, x_new):
        """Sample the tensor product interpolant on new points ``x_new``.

        Args:
            x_new (nump.ndarray[float]): Nodes where to sample the interpolant.
                Each row is one node.

        Returns:
            nump.ndarray[float]: Each row is the value of the interpolant on the 
                corresponding node of x_new.
        """
        inteprolant = RegularGridInterpolator(self.nodes_tuple, self.f_on_nodes,
                                              method='linear',
                                              bounds_error=False,
                                              fill_value=None)
        return inteprolant(x_new)

class TPPwQuadraticInterpolator:
    """Given function samples on tensor product nodes, interpolates with tensor 
        product piecewise quadratic interpolation.
    """
    def __init__(self, nodes_tuple, f):
        """Initializes the interpolator with nodes and function values.

        Args:
            nodes_tuple (tuple[nump.ndarray[float]]): k-th entry contains 1D 
                nodes in direction k.
            F (Function): Given paramter vector, returns function value.
        """

        # Format input if F is scalar field
        if len(f.shape) == len(nodes_tuple):
            f = np.reshape(f, f.shape + (1,))

        self.nodes_tuple = nodes_tuple
        self.f = f
        self.n_dims = len(nodes_tuple)
        self.n_nodes_dims = tuple(len(x) for x in self.nodes_tuple)
        self.dim_f = f.shape[-1]

        # Sanity check
        assert self.n_nodes_dims == f.shape[:-1:]

    def __call__(self, x_new):
        """Sample the tensor product interpolant on new points x_new

        Args:
            x_new (2D array double): Nodes where to sample the interpolant. Each 
            row is one node

        Returns:
            2D array double: Each row is the value of the interpolant on the 
                corresponding row of x_new.
        """

        numX = x_new.shape[0]
        assert x_new.shape[1] == self.n_dims

        # 1 compute stencil SS corresonding to every node
        # 2 compute tensor basis functions and format into LL
        LL = np.ones(tuple(3*np.ones(self.n_dims, dtype=int))+(numX,1))
        SS = np.ones((self.n_dims, numX), dtype=int)
        for n in range(self.n_dims):
            # stencil is computed 1 dimneison at a time.
            # Assume x scalar. To identify its stencil, think of the the first
            # even collocation node to the left.
            # For many x, it is faster to determine the stencil to which each x
            # belongs iff the array of xs is sorted.
            # so, we proceed as follows:
            #   1. sort x;
            #   2. find corresponding stencil of sorted sx;
            #   3. sort list of stencil indices by reverse sorting of x.
            zn_original = x_new[:,n]
            sorting = np.argsort(zn_original)
            rev_sorting = np.argsort(sorting)
            zn = zn_original[sorting]
            xn = self.nodes_tuple[n]
            halfxn = xn[0::2]
            # jj are the indices, for aeach element in scalar interpol point
            # zn, of the knot to the left in the scalar knots sequence xn
            jj = np.zeros(zn.size, dtype=int)
            p_prev = -1
            p=0
            for i in range(1, halfxn.size):
                p = np.searchsorted(zn, halfxn[i], side='right')
                jj[p_prev:p] = i-1
                p_prev = p
            jj[p:] = halfxn.size-1
            jj = jj[rev_sorting]
            SS[n, :] = jj
            # compute 3 corresponding basis functions in dim n
            x0 = xn[jj*2]
            x1 = xn[jj*2+1]
            x2 = xn[jj*2+2]
            L0 = ((zn_original-x1)*(zn_original-x2))/((x0-x1)*(x0-x2))
            L1 = ((zn_original-x0)*(zn_original-x2))/((x1-x0)*(x1-x2))
            L2 = ((zn_original-x0)*(zn_original-x1))/((x2-x0)*(x2-x1))

            Ln = np.vstack((L0, L1, L2))
            shapen = np.ones(self.n_dims+2, dtype=int)
            shapen[n] = 3
            shapen[-2] = numX
            Ln = np.reshape(Ln, tuple(shapen))
            LL *= Ln

        # 3 format interpolation data
        ff = np.zeros(tuple(3*np.ones(self.n_dims, dtype=int))+\
                      (numX, self.dim_f))
        it = np.nditer(ff[...,0,0],flags=['multi_index'])
        for i in it:
            vec_it = np.asarray(it.multi_index, dtype=int)
            vec_it = vec_it.reshape(-1,1)
            position = 2*SS+vec_it
            ff[it.multi_index] = self.f[tuple(position)] # numX x dimF

        # 4 compute product tensor data and Lagrange basis functions
        p = np.multiply(ff, LL)

        # 5 reduce first N dimensions and reshape
        s = np.sum(p, axis=tuple(n for n in range(self.n_dims)))
        return np.reshape( s,(numX, self.dim_f))

class TPPwCubicInterpolator:
    """Given function samples on tensor product nodes, interpolates with tensor 
    product piecewise cubic polynomial interpolation
    """

    def __init__(self, nodes_tuple, f):
        """Initializes the interpolator with nodes and function values.

        Args:
            nodesTuple (tuple[numpy.ndarray[float]]): k-th entry contains 1D 
                nodes in direction k.
            F (Callable[[numpy.ndarray[float], numpy.ndarray]]): Function to
                interpolate.
        """

        # formatting input if F is scalar field
        if len(f.shape) == len(nodes_tuple) :
            f = np.reshape(f, f.shape + (1,))
        self.nodes_tuple = nodes_tuple
        self.f = f
        self.n_dims = len(nodes_tuple)
        self.n_nodes_dims = tuple(len(x) for x in self.nodes_tuple)
        self.dim_f = f.shape[-1]

        # sanity check
        assert self.n_nodes_dims == f.shape[:-1:] 

    def __call__(self, x_new):
        """Sample the tensor product Lagrange interpolant on new points x_new

        Args:
            x_new (2D array double): Nodes where to sample the interpolant. Each 
            row is one node

        Returns:
            2D array double: Each row is the value of the interpolant on the 
            corresponding row of x_new
        """

        numX = x_new.shape[0]
        assert(x_new.shape[1] == self.n_dims)

        # 1 compute stencil SS corresonding to every node 
        # 2 compute tensor basis functions and format into LL
        SS = np.ones((self.n_dims, numX), dtype=int)
        LL = np.ones(tuple(4*np.ones(self.n_dims, dtype=int))+(numX,1))
        for n in range(self.n_dims):
            # Stencil is computed 1 dimension at a time. 
            # Assume x to be scalar. To identify its stencil, think of the the 
            #   first *even* collocation node.
            # If you have many xs, it is faster to determine the stencil to 
            #   which each x belongs if the array of xs is sorted. 
            # so 
            #   1. sort x; 
            #   2. find corresponding stencil of sorted xs; 
            #   3. sort list of stencil indices by reverse sorting of x.
            # NBB first stencil (-infty, y_2] second collocation nodes; last 
            #   stencil [y_{n-1}, infty) (y_{N} last collocation node).
            znOriginal = x_new[:,n]
            sorting = np.argsort(znOriginal)
            revSorting = np.argsort(sorting)  # znOriginal = zn[revSorting]
            zn = znOriginal[sorting]
            xn = self.nodes_tuple[n]
            halfxn = xn[0::3]  # stencils' boundaries
            jj = np.zeros(zn.size, dtype=int)
            pPrev = -1
            for i in range(1, halfxn.size): 
                p = np.searchsorted(zn, halfxn[i], side='right')
                jj[pPrev:p] = i-1  # assign tensil index to current nodes
                pPrev = p
            jj[p:] = i-1  # remaining nodes live in last stencil
            jj = jj[revSorting]
            SS[n, :] = jj

            # Compute 4 corresponding basis functions in dim n
            k = (xn.size-1)//3
            nRed = 3*k+1
            xnRed = xn[:nRed:] 
            # "regular" part of nodes array (largest subarray with length 3k+1)
            x0 = xnRed[jj*3]
            x1 = xnRed[jj*3+1]
            x2 = xnRed[jj*3+2]
            x3 = xnRed[jj*3+3]

            L0 = ((znOriginal-x1)*(znOriginal-x2)*(znOriginal-x3))/((x0-x1)*(x0-x2)*(x0-x3))
            L1 = ((znOriginal-x0)*(znOriginal-x2)*(znOriginal-x3))/((x1-x0)*(x1-x2)*(x1-x3))
            L2 = ((znOriginal-x0)*(znOriginal-x1)*(znOriginal-x3))/((x2-x0)*(x2-x1)*(x2-x3))
            L3 = ((znOriginal-x0)*(znOriginal-x1)*(znOriginal-x2))/((x3-x0)*(x3-x1)*(x3-x2))

            Ln = np.vstack((L0, L1, L2, L3))
            shapen = np.ones(self.n_dims+2, dtype=int)
            shapen[n] = 4
            shapen[-2] = numX
            Ln = np.reshape(Ln, tuple(shapen))
            LL *= Ln

        # 3 format interpolation data
        FF = np.zeros(tuple(4*np.ones(self.n_dims, dtype=int))+(numX, self.dim_f))
        it = np.nditer(FF[...,0,0],flags=['multi_index'])
        for i in it:
            vecIt = np.asarray(it.multi_index, dtype=int)
            vecIt = vecIt.reshape(-1,1)
            position = 3*SS+vecIt
            FF[it.multi_index] = self.f[tuple(position)] # size numX x dimF
        
        # 4 compute product tensor data and Lagrange basis functions
        p = np.multiply(FF, LL)
        
        # 5 reduce first N dimensions and reshape
        s = np.sum(p, axis=tuple(n for n in range(self.n_dims)))
        return np.reshape(s, (numX, self.dim_f))

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

    def __call__(self, x_new):
        """Sample the tensor product Lagrange interpolant on new points x_new

        Args:
            x_new (2D array double): Nodes where to sample the interpolant. Each 
            row is one node

        Returns:
            2D array double: Each row is the value of the interpolant on the 
            corresponding row of x_new
        """     

        numX = x_new.shape[0]
        assert(x_new.shape[1] == self.nDims)
        # Change shape F to cardY1 x ... x cardY_N x 1 x dimData 
        self.F = np.reshape(self.F, (self.nNodesDims) + (1,) + (self.dimF,))
        # Compute tensor Lagrange basis functions
        lagrangeBasis = np.ones((self.nNodesDims)+(numX,))
        for n in range(self.nDims):
            # Compute 1D basis functions: for each, get 2D array(cardYn x numX)
            xCurr = np.reshape(x_new[:, n], (1,-1))
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