import numpy as np
from scipy.interpolate import RegularGridInterpolator


'''wrap RegularGridInterpolator to handle case of 1 collocation point in some direction. In this directions, 
the ROM is a constant extrapolation'''


class TPInterpolatorWrapper:
    def __init__(self, nodesTuple, fOnNodes):
        self.nodesTuple = nodesTuple
        self.fOnNodes = fOnNodes
        self.activeDims = []  # dimensions with more than one node
        self.activeNodesTuple = tuple()
        for n in range(len(self.nodesTuple)):
            if self.nodesTuple[n].size > 1:
                self.activeDims.append(n)
                self.activeNodesTuple += (self.nodesTuple[n], )
        self.L = RegularGridInterpolator(self.activeNodesTuple, np.squeeze(self.fOnNodes), method='linear', bounds_error=False, fill_value=None)

    def __call__(self, xNew):
        # purge components of x in inactive dimensions, because the interpolating function will be constant in those dimensions
        xNew = xNew[:, self.activeDims]
        # handle case with no active dimensions (1 cp)
        if xNew.shape[1] == 0:
            assert(self.fOnNodes.shape[0] == 1)
            self.fOnNodes = np.reshape(self.fOnNodes, (1, -1))
            return np.repeat(self.fOnNodes, xNew.shape[0], axis=0)
        else:  # call regular scipy interpolant
            return self.L(xNew)
