from scipy.interpolate import RegularGridInterpolator


# TODO document
class TPPwLinearInterpolator:
    def __init__(self, nodes_tuple, f_on_nodes):
        self.nodes_tuple = nodes_tuple
        self.f_on_nodes = f_on_nodes

    def __call__(self, xNew):
        inteprolant = RegularGridInterpolator(self.nodes_tuple, self.f_on_nodes,
                                              method='linear',
                                              bounds_error=False, 
                                              fill_value=None)
        return inteprolant(xNew)