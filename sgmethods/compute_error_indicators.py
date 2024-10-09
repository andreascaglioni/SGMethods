"""Module with the function to compute the a-posteiori error indicators as in
[Gerstner, Griebel (2003)] for adaptive sparse grid interpolation.
"""

import numpy as np
from sgmethods.sparse_grid_interpolant import SGInterpolant
from sgmethods.utils import find_mid, lexic_sort
from sgmethods.tp_inteprolants import TPPwLinearInterpolator


# TODO: Write tests
def compute_GG_indicators_RM(old_RM, old_estimators, mid_set, knots, lev2knots,\
                            f, SG, u_on_SG, yy_rnd, L2_err_param, u_interp,\
                            TP_inteporlant=TPPwLinearInterpolator,\
                            n_parallel=1):

    r"""Compute refinement error indicators for adaptive sparse grid 
    interpolation as in [Gerstner, Griebel (2003)]. The error indicators 
    corresponding to :math:`\nu\in\mathcal{R}\mathcal{M}_{\Lambda}` is given by:

    .. math::

        \left\vert \Delta_{\nu} u\right\vert_{L^2_{\mu}(\Gamma)},
    
    where :math:`\Delta_{\nu} u` is the hieararchical surplus oeprator, 
    :math:`u:\Gamma\rightarrow V` is the function we are interpolating, and 
    :math:`\mu` is the Gaussian measure.    
    We compute the indicators only on the reduced margin of the current 
    multi-index set. 
    NB Recall that their sum (even on the whole maring) is NOT an error 
    estimator.
    This function recycles the previously computed values if the margin did not
    change in a neighbourhood.

    Args:
        old_RM (numpy.ndarray[int]): Old reduced margin. It is a 2D array
            where each row is a multi-index.
        old_estimators (numpy.ndarray[float]): Error indicators computed on the
            old reduced margin.
        mid_set (Class MidSet): Current multi-index set.
        knots (Callable[[int], numpy.ndarray[float]]): Returns the nodes vector
            of input length.
        lev2knots (Callable[[int], int]): Given a level >=0, returns a
            correspondingnumber number >0 of nodes.
        f (Callable[[numpy.ndarray[float]], numpy.ndarray[float]): Function to
            interpolate.
        SG (numpy.ndarray[float]): Current sparse grid. A 2D array.
        u_on_SG (numpy.ndarray[float]): Values F on current sparse grid. 2D 
            array, each row is a value of F on a sparse grid node.
        yy_rnd (numpy.ndarray[float]): Random parameters in parametric domain.
            A N+1-dimensional array where each row is a parameter vector
            (dimension N).
        L2_err_param (Callable[[numpy.ndarray[float], numpy.ndarray[float]], float]):
            Compute the :math:`L^2(\Gamma)` distance of the given
            functions using Monte Carlo quadrature.
            The functions are given through their value on the same random
            points in :math:`\Gamma'.
        u_interp (numpy.ndarray[float]): Values of the function to interpolate
            on the current sparse grid. 2D array, each row is the value in the
            corresponding row of SG.
        TP_inteporlant (Object TPInterpolatorWrapper, optional): Desired
            interpolation method as in tp_interpolants. Defaults to piecewise
            linear.
        n_parallel (int, optional): Number of parallel computations. Default 1.

    Returns:
        numpy.ndarray[float]: 1D array. Each entry is the error indicator
        corresponding to a multi-index in the reduced margin.
    """

    # Check input
    assert old_RM.shape[0] == old_estimators.size

    cardinality_reduced_margin = mid_set.reducedMargin.shape[0]
    estimator_reduced_margin = np.zeros(cardinality_reduced_margin)

    # Recycle from previous list of estimators, if any
    if old_RM.size==0:
        to_compute = range(mid_set.reducedMargin.shape[0])
    else:
        # Adapt if margin dimension grew
        dim_diff = mid_set.reducedMargin.shape[1] - old_RM.shape[1]
        assert dim_diff >= 0  # TODO: haldle general case? Usefull?
        if dim_diff>0:
            old_RM = np.hstack( (old_RM, np.zeros((old_RM.shape[0], dim_diff), \
                                                dtype=int)) )

        # Compute later all other mids
        to_compute = []
        for i in range(cardinality_reduced_margin):
            curr_mid = mid_set.reducedMargin[i]
            is_in_old_RM, pos = find_mid(curr_mid, old_RM)  # TODO change
            if is_in_old_RM:
                estimator_reduced_margin[i] = old_estimators[pos]
            else:
                to_compute.append(i)

    # Compute the rest
    for i in to_compute:
        curr_mid = mid_set.reducedMargin[i, :]
        t_ext = lexic_sort( np.vstack((mid_set.midSet, curr_mid)) )
        # TODO write and use here lexSortInsert(mid, midSet)
        # TODO Pass only lambda I : interpolant_give_midset(I)
        interpol_ext = SGInterpolant(t_ext, knots, lev2knots, \
                                       tp_interpolant=TP_inteporlant, \
                                        n_parallel=n_parallel, verbose=False)
        u_on_SG_ext = interpol_ext.sample_on_SG(f, old_xx=SG, old_samples=u_on_SG)
        u_interp_ext = interpol_ext.interpolate(yy_rnd, u_on_SG_ext)
        estimator_reduced_margin[i] = L2_err_param(u_interp_ext, u_interp)
        # TODO pass only more general norm
    return estimator_reduced_margin
