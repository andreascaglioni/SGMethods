import numpy as np

class TPInterpolatorWrapper:
    """ Wrapper for several tensor product interpolants. 
    If a direction has only 1 colllocation node, it should be 0. In this 
    directions, the interpolation approximation is a constant extrapolation.
    Example of interpolants that can be wrapped are ginev in the module
    src.tp_interpolants. 
    The user can define new interplants following the instructions written in 
    the module src.tp_interpolants.
    """

    def __init__(self, activeNodesTuple, activeDims, fOnNodes, TPInterpolant):
        """Take parameters to define tensor product interpolant

        Args:
            activeNodesTuple (tuple of 1D array double): Tuple of 1D nodes in 
                each direction for whihc there is more than 1 node
            activeDims (array): Dimensions with more that 1 node
            fOnNodes (array): Values of data to interpolate (each data point may
                be vector of some lenght)
            TPInterpolant (Class): One of the classes in the module 
                src.tp_inteprolants. (e.g. TPPwLinearInterpolator). The
                user can also define their own class following the instructions
                in the module src.tp_inteprolants. 
        """

        self.fOnNodes = fOnNodes
        self.activeDims = activeDims  # dimensions with more than one node
        self.L = TPInterpolant(activeNodesTuple, self.fOnNodes)

    def __call__(self, xNew):
        """Interpolate on desired new points in paramter space with method 
        chosen in constructor. It will first purge all directions with 1 
        node because in these directions the interpolant is constant.

        Args:
            xNew (ND array double): new parameter vectors to evaluate. 1 per row

        Returns:
            array of double: output of f on xNew. One per row.
        """    
            
        # if xNew has len(shape)=1, reshape it
        if(len(xNew.shape)==1):
            xNew = np.reshape(xNew, (-1,1))
        # Purge components of x in inactive dimensions (interpolant is constant 
        # in those dimensions)
        xNew = xNew[:, self.activeDims]
        
        if xNew.shape[1] == 0: # No active dimensions
            assert(self.fOnNodes.shape[0] == 1)
            self.fOnNodes = np.reshape(self.fOnNodes, (1, -1))
            return np.repeat(self.fOnNodes, xNew.shape[0], axis=0)
        else:
            return self.L(xNew)