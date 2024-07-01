from distutils.log import error
from matplotlib.image import interpolations_names
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from SGMethods.TPLagrangeInterpolator import TPLagrangeInterpolator
from SGMethods.TPPwQuadraticInterpolator import TPPwQuadraticInterpolator
from SGMethods.TPPwCubicInterpolator import TPPwCubicInterpolator


class TPInterpolatorWrapper:
    """ Wrapper for many Tensor Product interpolation methods.
    If a direction has only 1 colllocation node, it should be 0. In this directions, the interpolation approximation is a constant extrapolation
    """    
    def __init__(self, activeNodesTuple, activeDims, fOnNodes, TPInterpolant):
        """Take parameters to define tensor product interpolant

        Args:
            activeNodesTuple (tuple of 1D array double): Tople of 1D nodes in each direction for whihc there is more than 1 node
            activeDims (array): Dimensions with more that 1 node
            fOnNodes (N+1 array double): Values of data to interpolate (each data point may be vector of some lenght, so +1)
            TPInterpolant (function): Function to create the "raw" interpolant opeartor. Input: activeNodesTuple, fOnNodes as input 
                Resulting interpolant can be called with __call__(nNew), whre xNew is 2D array, each row is a new point to interpolate.
            Some interpolants are available either in standard packages or were implemented in this project. For example:
                - RegularGridInterpolator(activeNodesTuple, self.fOnNodes, method='linear', bounds_error=False, fill_value=None)  (From scipy)
                - TPPwQuadraticInterpolator(activeNodesTuple, self.fOnNodes) (this project)
                - TPPwCubicInterpolator(activeNodesTuple, self.fOnNodes) (this project)
                - TPLagrangeInterpolator(activeNodesTuple, self.fOnNodes) (this project)
                - TPInterpolant (this project)
        """        
        self.fOnNodes = fOnNodes
        self.activeDims = activeDims  # dimensions with more than one node
        self.L = TPInterpolant(activeNodesTuple, self.fOnNodes)

    def __call__(self, xNew):
        """Interpolate on desired new points in paramter space with method chosen in constructor.
        It will first purge all directions with 1 collocation nodes because in these directions the interpolant is constant

        Args:
            xNew (ND array double): new parameter vectors to evaluate. One per row

        Returns:
            array of double: output of f on xNew (new nodes). One per row.
        """    
            
        # if xNew has len(shape)=1, reshape it
        if(len(xNew.shape)==1):
            xNew = np.reshape(xNew, (-1,1))
        # purge components of x in inactive dimensions, because the interpolating function will be constant in those dimensions
        xNew = xNew[:, self.activeDims]
        # handle case with no active dimensions (1 cp)
        if xNew.shape[1] == 0:
            assert(self.fOnNodes.shape[0] == 1)
            self.fOnNodes = np.reshape(self.fOnNodes, (1, -1))
            return np.repeat(self.fOnNodes, xNew.shape[0], axis=0)
        else:  # call regular scipy interpolant
            return self.L(xNew)