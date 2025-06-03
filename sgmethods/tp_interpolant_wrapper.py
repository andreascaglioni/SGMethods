"""Module with the implementation of the wrapper class of tensor product 
    interpolants. Important to enforce the compatibility of user-defined 
    interpolants.
    """

import numpy as np

class TPInterpolatorWrapper:
    """ Wrapper for several tensor product interpolants. 

    If a direction has only 1 collocation node, it should be 0 and the 
    1-node approximation is constant in that direction. This is the case of any
    of the directions in which the multi-index does not have its support (no
    non-zero index).

    The user can define new tensor prodcut interplants following the
    instructions written in the module tp_interpolants and look at the examples
    given there.
    """

    def __init__(self, active_nodes_tuple, active_dims, f_on_nodes, \
                 tp_interpolant):
        r"""Import data on nodes, function to interpolate, and interpolation
        method.

        Args:
            active_nodes_tuple (tuple[numpy.ndarray[float]]): Tuple of 1D nodes
                in each direction for whihc there is more than 1 node.
            active_dims (numpy.ndarray[int]): Dimensions with more that 1 node.
            f_on_nodes (numpy.ndarray[double]): Values of data to interpolate
            (each data point may be vector of some lenght).
            tp_interpolant (Class): An appropriate class for tensor product 
                itnerpolation. See e.g. those implemented in the module
                tp_interpolants.
        """

        self.f_on_nodes = f_on_nodes
        self.active_dims = active_dims  # dimensions with more than one node
        self.l = tp_interpolant(active_nodes_tuple, self.f_on_nodes)

    def __call__(self, x_new):
        """Interpolate on desired new points in paramter space with method 
        chosen in constructor. It will first purge all directions with 1 
        node because in these directions the interpolant is constant.

        Args:
            xNew (ND array double): new parameter vectors to evaluate. 1 per row

        Returns:
            array of double: output of f on xNew. One per row.
        """

        # Handle shape of x_new
        if x_new.size == 1:  # n_new is made of 1 parametric point of dimension 1
            x_new = x_new.reshape((1, -1))

        # if xNew has len(shape)=1, reshape it to have 1 column ()
        if len(x_new.shape)==1:
            x_new = np.reshape(x_new, (-1,1))
        # Purge components of x in inactive dimensions (interpolant is constant
        # in those dimensions)
        x_new = x_new[:, self.active_dims]

        if x_new.shape[1] == 0: # No active dimensions
            assert self.f_on_nodes.shape[0] == 1
            self.f_on_nodes = np.reshape(self.f_on_nodes, (1, -1))
            return np.repeat(self.f_on_nodes, x_new.shape[0], axis=0)
        return self.l(x_new)
    