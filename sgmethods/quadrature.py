import numpy as np
from sgmethods.sparse_grid_interpolant import SGInterpolant
from sgmethods.nodes_1d import opt_guass_nodes_nest
from sgmethods.multi_index_sets import compute_mid_set_fast
import numbers
import warnings


def profit_sllg(nu, p=2):
    """Compute multi-index profits for sparse grid interpolant of parametric
    SLLG.

    Args:
        nu (numpy.ndarray[float]): 2D array, each row is a multi-index.
        p (int): Integer >= 2. Piecewise polynomial interpolation degree + 1.

    Returns:
        numpy.ndarray[float]: 1D array of (non-negative) profits. Length equals
        number of rows of nu.
    """

    #  Check input
    if not (isinstance(p, int) and p > 1):
        raise TypeError("p is not int >= 2")
    if not isinstance(nu, np.ndarray):
        raise TypeError("nu must be a numpy.ndarray.")
    if nu.ndim != 2:
        raise ValueError("nu must be a 2D array.")
    if not np.issubdtype(nu.dtype, int):
        raise TypeError("nu must be an array of int.")
    if np.any(nu < 0):
        raise ValueError("All entries of nu must be non-negative.")

    N_nu, D = nu.shape
    C1, C2 = 1, 1

    w1 = np.asarray(nu == 1).nonzero()  # np.where(nu == 1)
    w2 = np.asarray(nu > 1).nonzero()  # np.where(nu > 1)

    # Levels Levy-Ciesielski expansion (same shape as nuwith np.repeat)
    nn = np.arange(1, D + 1, 1)  # linear indices mid in nu from (1)
    ell = np.ceil(np.log2(nn))  # log-indices mid in nu (from 0)
    ell = np.reshape(ell, (1, ell.size))
    ell = np.repeat(ell, N_nu, axis=0)

    # The regularity weights (radious of domain of holomorphic extension)
    rho = np.zeros_like(nu, dtype=float)
    if not (w1[0].size == 0):  # if 1 tuple element is empty, the other is too
        rho[w1] = 2.0 ** (3.0 / 2.0 * ell[w1])
    if not (w2[0].size == 0):
        rho[w2] = 2.0 ** (1.0 / 2.0 * ell[w2])

    # Compute value of components of each multi-index (multiply -> value multi-index)
    V_comps = np.ones_like(rho)
    V_comps[w1] = C1 * np.power(rho[w1], -1.0)
    V_comps[w2] = C2 * np.power(2.0, -p * nu[w2] * rho[w2])
    value = np.prod(V_comps, axis=1)

    # Work of a multi-index. Computed with product structure, as for work.
    W_comps = (2 ** (nu + 1) - 2) * (p - 1) + 1
    work = np.prod(W_comps, axis=1)

    return value / work


# TODO Currently assuming Gaussian samples for SLLG. Add more options/make an input
# TODO improve efficiency computation integral: instead of MC, do exact computation based on inclusion-exclsion and TP structure
def compute_quadrature_params(
    min_n_samples, dim_samples, distrbution="gauss", eps=1.0e-2
):
    """
    Computes quadrature nodes and weights for sparse grid quadrature using a Gaussian distribution.
    Args:
        min_n_samples (int): Minimum number of quadrature nodes requested.
        dim_samples (int): Number of dimensions for the quadrature.
        distrbution (str, optional): Integral measure. Only "gauss" is supported for now. Defaults to "gauss".
        eps (float, optional): Accuracy integration basis functions. Defaults to 1.0e-4.
    Returns:
        tuple:
            quad_nodes (np.ndarray): Array of quadrature nodes (sparse grid points).
            quad_weights (np.ndarray): Array of quadrature weights corresponding to the nodes.
    Raises:
        ValueError: If `min_n_samples` is not a positive number.
        ValueError: If `dim_samples` is not a positive integer.
        ValueError: If `distrbution` is not "gauss".
    Notes:
        - The quadrature weights are computed as the mean of the interpolated Lagrange basis functions over a large Monte Carlo sample.
        - The function relies on external functions/classes: `opt_guass_nodes_nest`, `profit_sllg`, `compute_mid_set_fast`, and `SGInterpolant`.
    """

    # Check input
    if not isinstance(min_n_samples, numbers.Number) or min_n_samples <= 0:
        raise ValueError("n_samples must be a positive integer.")
    if not isinstance(dim_samples, int) or dim_samples <= 0:
        raise ValueError("dim_samples must be a positive integer.")
    if distrbution != "gauss":
        raise ValueError("Only 'gauss' distribution is supported.")

    knots = lambda n: opt_guass_nodes_nest(n)  # noqa: E731
    lev2knots = lambda i: np.where(i > 0, 2 ** (i + 1) - 1, 1)  # noqa: E731
    P = profit_sllg

    # Find mid_set such that: #SG > min_n_samples
    min_p = P(np.zeros((1, 1), dtype=int))[0]  # include all mids : profit > min_p
    decrease_min_p = True
    while decrease_min_p:
        mid_set = compute_mid_set_fast(P, min_p, dim_samples)
        I = SGInterpolant(mid_set, knots, lev2knots)
        if I.num_nodes > min_n_samples:
            decrease_min_p = False
        else:
            min_p *= 0.5

    # Quadarature samples = current sparse grid
    quad_nodes = I.SG

    quad_weights = compute_quadarture_wights_MC(dim_samples, eps, I)

    return quad_nodes, quad_weights


def compute_quadrature_wights_incl_excl(dim_samples, eps, sg_interp):
    """Compute quadrature weights for sparse grid quadrature using the
    inclusion-exclusion formula.

    Args:
        dim_samples (int): Number of dimensions.
        eps (float): Accuracy parameter (not used in this implementation).
        sg_interp (SGInterpolant): Sparse grid interpolant object.

    Returns:
        np.ndarray: Quadrature weights for each sparse grid node.
    """

    # Fetch data
    mid_set = sg_interp.mid_set
    SG = sg_interp.SG
    aalpha = sg_interp.combination_coeffs  # 1d array (#Lambda, ) incl-excl coeffs

    ww = np.zeros(sg_interp.num_nodes)  # weights to return

    # pre-compute indices of mids which give TP grid containing each SG node
    idx_mids_y = tuple(ww)


    # pe-compute scalar integrals \int_{R} l_y^{\nu} \text{d}\mu, where
    # l_y^{\nu}: 1D Lagrange basis fun correposdnig to y \in \YY_{\nu}
    # \mu Standard Gaussian measure
    
    yy_1d_unique = np.unique(SG.flatten())
    ww_1d = np.zeros(np.amax(mid_set))

    # Pre-compute whights tensor-product quadrature (only basis function corresponding to \by in SG)
    # \by -> \bnu s.t. \by\in \YY_{\bnu} -> TP weight
    
    #  combine pre-computed data


    # Find, for each $\by \in \HH_{\Lambda}$, the multi indices $\bnu\in\Lambda$ such that 
    #   * \by \in \YY_{\bnu}
    #   * \alpha_{\bnu} \neq 0
    where_y_in_YY = np.array(np.array([], dtype=int), dtype=object)
    for i in range(mid_set.shape[0]):
        nu = mid_set[i]
        

        

    # aalpha: (#SG, # {nu\in\Lambda : by \in YY_nu})
    # ww_tp:  (#SG, # {nu\in\Lambda : by \in YY_nu}) 
    nu_select_y = None
    aalpha_rep = np.array(np.array([]), dtype=object)
    for i in aalpha_rep.size:
        aalpha_rep[i] = aalpha[nu_select_y]
    
    ww = np.sum(aalpha_rep * ww_tp, axis = 1)
    return ww


    # The inclusion-exclusion principle for sparse grid quadrature weights:
    # For each node, sum the contributions from all multi-indices containing it,
    # with alternating signs according to the inclusion-exclusion principle.

    # Get the multi-index set and the corresponding levels
    mid_set = sg_interp.mid_set
    nodes = sg_interp.SG
    num_nodes = sg_interp.num_nodes

    # Get the hierarchical surpluses (coefficients) for each node
    # For quadrature, the weight for each node is the sum of the weights
    # of the tensor-product quadrature rules that include it, with the
    # appropriate inclusion-exclusion sign.

    # For Gaussian quadrature, the 1D weights for each level
    # are given by the standard Gauss-Hermite quadrature weights.
    # We need to compute the tensor-product weights for each multi-index,
    # and then sum their contributions to each node.

    # Precompute 1D nodes and weights for all levels used
    max_level = np.max(mid_set)
    one_d_weights = {}
    one_d_nodes = {}
    for l in range(max_level + 1):
        n_knots = sg_interp.lev2knots(l)
        x, w = np.polynomial.hermite.hermgauss(n_knots)
        one_d_nodes[l] = x
        one_d_weights[l] = w / np.sqrt(np.pi)  # normalize for standard normal

    # Map each node to its multi-index and local index in the tensor grid
    node_to_multi = sg_interp.node_to_multi  # (node_idx) -> (multi_idx, local_idx)

    # For each node, sum the contributions from all multi-indices
    quad_weights = np.zeros(num_nodes)
    for node_idx in range(num_nodes):
        # Find all multi-indices that include this node
        multi_indices = node_to_multi[node_idx]
        total_weight = 0.0
        for multi_idx, local_idx in multi_indices:
            # Inclusion-exclusion sign: (-1)^(sum of multi_idx - min_level)
            sign = (-1) ** (np.sum(mid_set[multi_idx]) - np.min(mid_set[multi_idx]))
            # Tensor-product weight for this node in this multi-index
            w = 1.0
            for d in range(dim_samples):
                l = mid_set[multi_idx, d]
                w *= one_d_weights[l][local_idx[d]]
            total_weight += sign * w
        quad_weights[node_idx] = total_weight

    return quad_weights


# DEPRECATED (inefficient)
def compute_quadarture_wights_MC(dim_samples, eps, sg_interp):
    """Compute quadrature weights of sparse grid quadrature as :
        weights[i] = \int_{\Gamma} L_{y_i} d\mu,
    where L_{y} denotes a Lagrange basis function of I.
    Observe that:
        L_{y_i} = I[delta_{y_i}],
    where delta_{y_i}(y) = 1 if y = y_i, 0 otherwise.

    Args:
        dim_samples (int): Number of dimensions.
        eps (float): Accuracy parameter for MC integration.
        sg_interp (SGInterpolant): Sparse grid interpolant object.

    Returns:
        np.ndarray: Quadrature weights for each sparse grid node.
    """

    warnings.warn("compute_quadarture_wights_MC is deprecated.", DeprecationWarning)

    # 1. Define a function with #SG components. Each is 1 in only 1 SG node.
    Delta_sg = np.eye(sg_interp.num_nodes)

    # 2. Interpolate over a large Monte Carlo sample.
    n_mc_samples = int(1 / eps) ** 2  #  10000
    zz_mc = np.random.standard_normal((n_mc_samples, dim_samples))
    ww_samples = sg_interp.interpolate(zz_mc, Delta_sg)

    # 3. Compute the quadrature weights as the mean over the MC samples.
    quad_weights = np.mean(ww_samples, axis=0)
    return quad_weights
