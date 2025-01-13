"""Module with functions to compute the a-posteriori error *indicators* for
adaptive enlargement of the multi-index set (equivalently, enlargement of the
sparse grid or enrichment of the polynomial approximation space).

Coming soon: A-posteriori error estimator from *Guignard, Nobile, SIAM JNA 
(2018)*.
"""

import numpy as np
from sgmethods.sparse_grid_interpolant import SGInterpolant
from sgmethods.utils import find_mid, lexic_sort
from sgmethods.tp_interpolants import TPPwLinearInterpolator


# TODO: Write tests
def compute_GG_indicators_RM(old_RM, old_estimators, mid_set, knots, lev2knots,\
                            f, SG, u_on_SG, yy_rnd, L2_err_param, u_interp,\
                            TP_inteporlant=TPPwLinearInterpolator,\
                            n_parallel=1):

    r"""Compute refinement error indicators for adaptive sparse grid 
    interpolation as in 
    
    *Gerstner, T., Griebel, M. Dimension-Adaptive Tensor-Product Quadrature. 
    Computing 71, 65-87 (2003). https://doi.org/10.1007/s00607-003-0015-5*

    The error indicator corresponding to
    :math:`\nu\in\mathcal{R}\mathcal{M}_{\Lambda}` is given by:

    .. math::

        \left\vert \Delta_{\nu} u\right\vert_{L^2_{\mu}(\Gamma)},
    
    where :math:`\Delta_{\nu} u` is the hierarchical surplus operator, 
    :math:`u:\Gamma\rightarrow V` is the function we are interpolating, and 
    :math:`\mu` is the Gaussian measure.    
    We compute the indicators only on the reduced margin of the current 
    multi-index set. 
    NB Recall that their sum (even on the whole margin) is NOT an error 
    estimator.
    This function recycles the previously computed values if the margin did not
    change in a neighbourhood.

    Args:
        old_RM (numpy.ndarray[int]): Old reduced margin. It is a 2D array
            where each row is a multi-index.
        old_estimators (numpy.ndarray[float]): Error indicators computed on the
            old reduced margin.
        mid_set (:py:class:`~sgmethods.mid_set.MidSet`): Current multi-index 
            set.
        knots (Callable[[int], numpy.ndarray[float]]): Returns the nodes vector
            of input length.
        lev2knots (Callable[[int], int]): Given a level >=0, returns a
            corresponding number number >0 of nodes.
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
            points in :math:`\Gamma`.
        u_interp (numpy.ndarray[float]): Values of the function to interpolate
            on the current sparse grid. 2D array, each row is the value in the
            corresponding row of SG.
        TP_inteporlant (Appropriate tensor product interpolant class, optional):
            Desired interpolation method as in tp_interpolants. Defaults to 
            piecewise linear.
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

def _compute_indicator_function():
    pass

def compute_GN_indicators_estimator(interpolant, samples_u, mid_set,\
                                    compute_norm, diffusion):
    
    """Compute error indicator for the sparse grid interpolant of the 
    parameter-to-solution map of the affine diffusion Poisson problem, as in

    *Guignard, D., Nobile, F. A-posteriori error estimation for the stocahstic
    collocation finite element method. SIAM J. NUMER. ANAL.  56, No. 5, 
    pp. 3121--3143 (2018). https://doi.org/10.1007/s00607-003-0015-5*

    Additionally, compute an error estimator that slightly varies from the one
    defined in the prebious publication: Rather than the sum of indicators 
    (reliable but possibly not efficieint), compute the norm of the sum.

    Args:
        interpolant (:py:class:`~sgmethods`): The sparse grid interpolant.
        samples_u (numpy.ndarray[float]): The values of the function to 
            interpolate on the sparse grid. Each row is the value of the 
            function on a sparse grid node.
        mid_set (:py:class:`~sgmethods`): The multi-index set used to define
            the sparse grid interpolant.
        compute_norm (Callable[[numpy.ndarray[float]], float]): Compute the norm
        diffusion (Callable[[numpy.ndarray[float]], numpy.ndarray[float]]): The
            parametric affine diffusion operator. It takes a parameter vector
            and a value of the space variable and returns a scalar.

    Returns:
        _type_: _description_
    """
    error_indicators = np.zeros(interpolant.mid_set.shape[1])
    error_estimator = 0
    estimator_fun = 0  # The error estimator is its norm
    for i, nu in enumerate(mid_set.margin):  # nu is a multi-index in margin
        # compute error indicator corresponding to nu
        indicator_fun = 0
        error_indicators[i] = compute_norm(indicator_fun)
        
        # add to estimator functio (NB Norm must be computed after the loop!)
        estimator_fun = estimator_fun + indicator_fun

    error_estimator = compute_norm(estimator_fun)
    return error_indicators, error_estimator
    


