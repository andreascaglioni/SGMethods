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
    def __init__(self, activeNodesTuple, activeDims, fOnNodes, interpolationType="polynomial"):
        """Take parameters to define tensor product interpolant

        Args:
            activeNodesTuple (tuple of 1D array double): Tople of 1D nodes in each direction for whihc there is more than 1 node
            activeDims (array): Dimensions with more that 1 node
            fOnNodes (N+1 array double): Values of data to interpolate (each data point may be vector of some lenght, so +1)
            interpolationType (str, optional): Choose between;
                - linear: Tensor product linear piecewise polynomail interpolant;
                - quadratic: Tensor product quadratic piecewise polynomial interpolant;
                - cubic: Tensor product cubic piecewise polynomial interpolant;
                - polynomial: Tensor product Lagrange interpolant.
            Defaults to "polynomial".
        """        
        self.fOnNodes = fOnNodes
        self.activeDims = activeDims  # dimensions with more than one node
        if(interpolationType == "linear"):
            self.L = RegularGridInterpolator(activeNodesTuple, self.fOnNodes, method='linear', bounds_error=False, fill_value=None)
        elif(interpolationType == "quadratic"):
            self.L = TPPwQuadraticInterpolator(activeNodesTuple, self.fOnNodes)
        elif(interpolationType == "cubic"):
            self.L = TPPwCubicInterpolator(activeNodesTuple, self.fOnNodes)
        elif(interpolationType == "polynomial"):
            self.L = TPLagrangeInterpolator(activeNodesTuple, self.fOnNodes)
        else:
            error("interpolation type: " , interpolationType, "not recognized")

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