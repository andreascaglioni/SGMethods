from distutils.log import error
from matplotlib.image import interpolations_names
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from SGMethods.TPLagrangeInterpolator import TPLagrangeInterpolator
from SGMethods.TPPwQuadraticInterpolator import TPPwQuadraticInterpolator

'''wrap RegularGridInterpolator to handle case of 1 collocation point in some direction. In this directions, 
the ROM is a constant extrapolation'''

class TPInterpolatorWrapper:
    def __init__(self, activeNodesTuple, activeDims, fOnNodes, interpolationType="polynomial"):
        self.fOnNodes = fOnNodes
        self.activeDims = activeDims  # dimensions with more than one node
        if(interpolationType == "linear"):
            self.L = RegularGridInterpolator(activeNodesTuple, self.fOnNodes, method='linear', bounds_error=False, fill_value=None)
        elif(interpolationType == "quadratic"):
            self.L = TPPwQuadraticInterpolator(activeNodesTuple, self.fOnNodes)
        elif(interpolationType == "polynomial"):
            self.L = TPLagrangeInterpolator(activeNodesTuple, self.fOnNodes)
        else:
            error("interpolation type: " , interpolationType, "not recognized")

    def __call__(self, xNew):
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