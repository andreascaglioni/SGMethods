"""Examples of tenosr products interpolatio operators. They will be wrapped
in the interponat wrapper class and used for sparse grid interpolation through
the inclusion-exclusion formula.
The user can implement new interpolation operators ina alalgogous classes.
IMPORTANT: The input and output must be compatible with the wrapper, so same
as the examples in this file! In particular the following methods are needed:
- __init__(self, nodes_tuple, f_on_nodes);
- __call__(self, x_new): Input is numpy.ndarray[float] (each row is a parameter 
vector on which to interpolate), output is numpy.ndarray[float] (each row is
the value of the interpolant on the corresponding parameter in x_new).
"""

# TODO consider inheritance to define all these function. It could laso be
#   useful to users. Just inherit from the class.
# TODO Also consider, instead of having an additional wrapper, to implement
# those functionalities inside the mother class

import numpy as np
from scipy.interpolate import RegularGridInterpolator

class TPPwLinearInterpolator:
    """Given function samples on tensor product nodes, interpolates with tensor 
        product piecewise linear interpolation.
    """

    def __init__(self, nodes_tuple, f_on_nodes):
        """Initializes the interpolator with nodes and function values.

        Args:
            nodes_tuple (tuple[nump.ndarray[float]]): k-th entry contains 1D 
                nodes in direction k.
            f (Callable[[numpy.ndarray[float]], numpy.ndarray[float]]): Function
                to interpolate.
        """
        self.nodes_tuple = nodes_tuple
        self.f_on_nodes = f_on_nodes

    def __call__(self, x_new):
        """Sample the interpolant on new points x_new.

        Args:
            x_new (numpy.ndarray[float]): Nodes where to sample the interpolant.
                Each row is one node.

        Returns:
            numpy.ndarray[float]: Each row is the value of the interpolant on
            the corresponding row of x_new.
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
            f (Callable[[numpy.ndarray[float]], numpy.ndarray[float]]): Function
                to interpolate.
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
        """Sample the interpolant on new points x_new.

        Args:
            x_new (numpy.ndarray[float]): Nodes where to sample the interpolant.
                Each row is one node.

        Returns:
            numpy.ndarray[float]: Each row is the value of the interpolant on
            the corresponding row of x_new.
        """

        num_x = x_new.shape[0]
        assert x_new.shape[1] == self.n_dims

        # 1 compute stencil SS corresonding to every node
        # 2 compute tensor basis functions and format into LL
        ll = np.ones(tuple(3*np.ones(self.n_dims, dtype=int))+(num_x,1))
        ss = np.ones((self.n_dims, num_x), dtype=int)
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
            ss[n, :] = jj
            # compute 3 corresponding basis functions in dim n
            x0 = xn[jj*2]
            x1 = xn[jj*2+1]
            x2 = xn[jj*2+2]
            l0 = ((zn_original-x1)*(zn_original-x2))/((x0-x1)*(x0-x2))
            l1 = ((zn_original-x0)*(zn_original-x2))/((x1-x0)*(x1-x2))
            l2 = ((zn_original-x0)*(zn_original-x1))/((x2-x0)*(x2-x1))

            ln = np.vstack((l0, l1, l2))
            shapen = np.ones(self.n_dims+2, dtype=int)
            shapen[n] = 3
            shapen[-2] = num_x
            ln = np.reshape(ln, tuple(shapen))
            ll *= ln

        # 3 format interpolation data
        ff = np.zeros(tuple(3*np.ones(self.n_dims, dtype=int))+\
                      (num_x, self.dim_f))
        it = np.nditer(ff[...,0,0],flags=['multi_index'])
        for i in it:
            vec_it = np.asarray(it.multi_index, dtype=int)
            vec_it = vec_it.reshape(-1,1)
            position = 2*ss+vec_it
            ff[it.multi_index] = self.f[tuple(position)] # numX x dimF

        # 4 compute product tensor data and Lagrange basis functions
        p = np.multiply(ff, ll)

        # 5 reduce first N dimensions and reshape
        s = np.sum(p, axis=tuple(n for n in range(self.n_dims)))
        return np.reshape( s,(num_x, self.dim_f))

class TPPwCubicInterpolator:
    """Given function samples on tensor product nodes, interpolates with tensor 
        product piecewise cubic interpolation.
    """

    def __init__(self, nodes_tuple, f):
        """Initializes the interpolator with nodes and function values.

        Args:
            nodes_tuple (tuple[nump.ndarray[float]]): k-th entry contains 1D 
                nodes in direction k.
            f (Callable[[numpy.ndarray[float]], numpy.ndarray[float]]): Function
                to interpolate.
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
        """Sample the interpolant on new points x_new.

        Args:
            x_new (numpy.ndarray[float]): Nodes where to sample the interpolant.
                Each row is one node.

        Returns:
            numpy.ndarray[float]: Each row is the value of the interpolant on
            the corresponding row of x_new.
        """

        num_x = x_new.shape[0]
        assert x_new.shape[1] == self.n_dims

        # 1 compute stencil SS corresonding to every node
        # 2 compute tensor basis functions and format into LL
        ss = np.ones((self.n_dims, num_x), dtype=int)
        ll = np.ones(tuple(4*np.ones(self.n_dims, dtype=int))+(num_x,1))
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
            zn_original = x_new[:,n]
            sorting = np.argsort(zn_original)
            rev_sorting = np.argsort(sorting)  # znOriginal = zn[revSorting]
            zn = zn_original[sorting]
            xn = self.nodes_tuple[n]
            halfxn = xn[0::3]  # stencils' boundaries
            jj = np.zeros(zn.size, dtype=int)
            p_prev = -1
            for i in range(1, halfxn.size):
                p = np.searchsorted(zn, halfxn[i], side='right')
                jj[p_prev:p] = i-1  # assign tensil index to current nodes
                p_prev = p
            jj[p:] = i-1  # remaining nodes live in last stencil
            jj = jj[rev_sorting]
            ss[n, :] = jj

            # Compute 4 corresponding basis functions in dim n
            k = (xn.size-1)//3
            n_red = 3*k+1
            xn_red = xn[:n_red:]
            # "regular" part of nodes array (largest subarray with length 3k+1)
            x0 = xn_red[jj*3]
            x1 = xn_red[jj*3+1]
            x2 = xn_red[jj*3+2]
            x3 = xn_red[jj*3+3]

            l0 = ((zn_original-x1)*(zn_original-x2)*(zn_original-x3))\
                /((x0-x1)*(x0-x2)*(x0-x3))
            l1 = ((zn_original-x0)*(zn_original-x2)*(zn_original-x3))\
                /((x1-x0)*(x1-x2)*(x1-x3))
            l2 = ((zn_original-x0)*(zn_original-x1)*(zn_original-x3))\
                /((x2-x0)*(x2-x1)*(x2-x3))
            l3 = ((zn_original-x0)*(zn_original-x1)*(zn_original-x2))\
                /((x3-x0)*(x3-x1)*(x3-x2))

            ln = np.vstack((l0, l1, l2, l3))
            shapen = np.ones(self.n_dims+2, dtype=int)
            shapen[n] = 4
            shapen[-2] = num_x
            ln = np.reshape(ln, tuple(shapen))
            ll *= ln

        # 3 format interpolation data
        ff = np.zeros(tuple(4*np.ones(self.n_dims, dtype=int))\
                      +(num_x, self.dim_f))
        it = np.nditer(ff[...,0,0],flags=['multi_index'])
        for i in it:
            vec_it = np.asarray(it.multi_index, dtype=int)
            vec_it = vec_it.reshape(-1,1)
            position = 3*ss+vec_it
            ff[it.multi_index] = self.f[tuple(position)] # size numX x dimF

        # 4 compute product tensor data and Lagrange basis functions
        p = np.multiply(ff, ll)

        # 5 reduce first N dimensions and reshape
        s = np.sum(p, axis=tuple(n for n in range(self.n_dims)))
        return np.reshape(s, (num_x, self.dim_f))

class TPLagrangeInterpolator:
    """Given function samples on tensor product nodes, interpolates with tensor 
        product Lagrange interpolation computed with barycentric interpolation 
        formula, see 
        Trefetten-*Approximation Theory and Approximation Practice* (2019)
    """

    def __init__(self, nodes_tuple, f):
        """Initializes the interpolator with nodes and function values.

        Args:
            nodes_tuple (tuple[nump.ndarray[float]]): k-th entry contains 1D 
                nodes in direction k.
            f (Callable[[numpy.ndarray[float]], numpy.ndarray[float]]): Function
                to interpolate.
        """

        # Format input
        if len(f.shape) == len(nodes_tuple):  # If F is scalar, turn to N x 1
            f = np.reshape(f, f.shape + (1,))
        self.nodes_tuple = nodes_tuple
        self.f = f
        self.n_dims = len(nodes_tuple)
        self.n_nodes_dims = tuple(len(x) for x in self.nodes_tuple)
        self.dim_f = f.shape[-1]

        # Sanity check
        assert self.n_nodes_dims == f.shape[:-1:]

        # Compute \lambda_i coefficients
        # \labda_i = (\prod_{j\neqi} x_i-x_j})^{-1}
        # Put them in tuple of length N
        self.lambda_coeffs = ()
        for n in range(self.n_dims):
            l_curr_d = np.ones(self.n_nodes_dims[n])
            curr_nodes = self.nodes_tuple[n]
            for i in range(self.n_nodes_dims[n]):
                w_curr = np.repeat(True, curr_nodes.shape)
                w_curr[i] = False
                l_curr_d[i] = np.prod(curr_nodes[i] - curr_nodes, where=w_curr)
            l_curr_d = 1/l_curr_d
            self.lambda_coeffs = self.lambda_coeffs + (l_curr_d,)

    def __call__(self, x_new):
        """Sample the interpolant on new points x_new.

        Args:
            x_new (numpy.ndarray[float]): Nodes where to sample the interpolant.
                Each row is one node.

        Returns:
            numpy.ndarray[float]: Each row is the value of the interpolant on
            the corresponding row of x_new.
        """

        num_x = x_new.shape[0]
        assert x_new.shape[1] == self.n_dims
        # Change shape F to cardY1 x ... x cardY_N x 1 x dimData
        self.f = np.reshape(self.f, (self.n_nodes_dims) + (1,) + (self.dim_f,))
        # Compute tensor Lagrange basis functions
        lagrange_basis = np.ones((self.n_nodes_dims)+(num_x,))
        for n in range(self.n_dims):
            # Compute 1D basis functions: for each, get 2D array(cardYn x numX)
            x_curr = np.reshape(x_new[:, n], (1,-1))
            nodes_curr = np.reshape(self.nodes_tuple[n], (-1,1))
            l_curr = np. prod(x_curr - nodes_curr, axis=0)
            curr_lambda = np.reshape(self.lambda_coeffs[n], (-1,1))
            lag_basis_curr = l_curr * curr_lambda / (x_curr - nodes_curr)
            # Reshape it to 1 x ... x 1 x cardYn x 1 x ... x 1 x numX
            shape_curr = list(1 for _ in range(self.n_dims))
            shape_curr[n] = self.n_nodes_dims[n]
            shape_curr.append(num_x)
            lag_basis_curr = np.reshape(lag_basis_curr, tuple(shape_curr))
            # Exterior product ot get tensor of basis functions shaped
            # cardY1 x ... x cardY_N x numX
            lagrange_basis = np.multiply(lagrange_basis, lag_basis_curr)
        lagrange_basis = np.reshape(lagrange_basis, lagrange_basis.shape+(1,))
        # Compute product tensor data and Lagrange basis functions
        p = np.multiply( self.f, lagrange_basis )
        # Reduce first N dimensions and reshape
        s = np.sum(p, axis=tuple(n for n in range(self.n_dims)))
        return np.reshape( s,(num_x, self.dim_f))
    