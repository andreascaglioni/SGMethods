from distutils.log import error
from matplotlib.image import interpolations_names
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from src.tp_lagrange import TPLagrangeInterpolator
from src.tp_piecewise_quadratic import TPPwQuadraticInterpolator
from src.tp_piecewise_cubic import TPPwCubicInterpolator


class TPInterpolatorWrapper:
    """ Wrapper for many Tensor Product interpolation methods.
    If a direction has only 1 colllocation node, it should be 0. In this 
    directions, the interpolation approximation is a constant extrapolation
    """    
    def __init__(self, activeNodesTuple, activeDims, fOnNodes, TPInterpolant):
        """Take parameters to define tensor product interpolant

        Args:
            activeNodesTuple (tuple of 1D array double): Tuple of 1D nodes in 
                each direction for whihc there is more than 1 node
            activeDims (array): Dimensions with more that 1 node
            fOnNodes (array): Values of data to interpolate (each data point may
                be vector of some lenght)
            TPInterpolant (function): Function to create the "raw" interpolant
                opeartor. Resulting interpolant has __call__(nNew) method
                to sample in new points.
                Some interpolants are available either in standard packages or 
                were implemented in this project. For example:
                - RegularGridInterpolator (Scipy)
                - TPPwQuadraticInterpolator (SGMethods)
                - TPPwCubicInterpolator (SGMethods) 
                - TPLagrangeInterpolator (SGMethods) """

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
        # Purge components of x in inactive dimensions
        # (interpolant is constant in those dimensions)
        xNew = xNew[:, self.activeDims]
        
        if xNew.shape[1] == 0: # No active dimensions
            assert(self.fOnNodes.shape[0] == 1)
            self.fOnNodes = np.reshape(self.fOnNodes, (1, -1))
            return np.repeat(self.fOnNodes, xNew.shape[0], axis=0)
        else:
            return self.L(xNew)