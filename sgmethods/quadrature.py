from math import sqrt, pi, exp, erf
import numpy as np
from sgmethods.sparse_grid_interpolant import SGInterpolant
from sgmethods.nodes_1d import opt_guass_nodes_nest
from sgmethods.multi_index_sets import compute_mid_set_fast
import numbers
import warnings
from scipy.stats import norm


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
        - The quadrature weights are with inclusion-exclusion formula
        - The function relies on external functions/classes: `opt_guass_nodes_nest`, `profit_sllg`, `compute_mid_set_fast`, and `SGInterpolant`.
    """

    # Check input
    if not isinstance(min_n_samples, numbers.Number) or min_n_samples <= 0:
        raise ValueError("n_samples must be a positive integer.")
    if not isinstance(dim_samples, int) or dim_samples <= 0:
        raise ValueError("dim_samples must be a positive integer.")
    if distrbution != "gauss":
        raise ValueError("Only 'gauss' distribution is supported.")

    knots = opt_guass_nodes_nest
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

    sg = sg_interp.SG
    num_nodes = sg.interp.num_nodes
    combi_coeffs = sg.combination_coeffs
    num_mids, dim = sg_interp.mid_set.shape
    lev2knots = sg_interp.lev2knots

    ww = np.zeros(num_nodes)  # quadrature weights, to return
    
    # Change shape combi_coeffs to (1, -1) for breoadcasting
    combi_coeffs = combi_coeffs.reshape(1, -1)
    
    # ------------------------------------------------------------------------ #
    #                       Compute 1d quadrature weights                      #
    # ------------------------------------------------------------------------ #
    max_n = np.amax(sg_interp.mid_set.flatten())
    W_1d = compute_1d_quadrature_weights(max_n, sg_interp)
    
    

    # ------------------------------------------------------------------------ #
    #                       Compute TP quadrature weights                      #
    # ------------------------------------------------------------------------ #
    # For each *actuve* multi-index in the multi-index set
    # Compute a matrix W_tp_nu of shape (lev2knots(nu[0]), ..., lev2knots(nu[-1]))
    # W_tp_nu[i[0], ..., i[-1]] = \int_{\Gamma} l^{\bnu}_{\by} d\mu
    # where \by = [kk[0, i[0]], ..., k[-1, i[-1]]]
    # and k[j, :] are the 1d nodes corresponding to nu[j]
    # TODO how use 1d quadrature weights?
    
    W_tp = np.zeros(num_mids)
    for i in range(W_tp):
        W_1d = np.zeros(lev2knots())

    # TODO convert to SG shape:
    #     (W_tp)_ij = \int_{\Gamma} l_{\by_i}^{\bnu_j} d\mu
    #     W_tp = np.zeros(num_nodes, num_mids)
    W_tp = None
    
    #  Cmpute SG weights with inclusion-exclusion formula
    ww = np.sum(combi_coeffs * W_tp, axis=1)  # bradcasting combi_coeffs

    return ww

def compute_1d_quadrature_weights(max_n, lev2knots, knots, interpolant="pw_lin_extrapolated"):
    """ Compute all 1D interpolation weights associated to a quadrature rule, up to a maximum number of quadrature nodes for all smaller admissible number of nodes.

    Args:
        max_n (int): The maximum index (exclusive) for which to compute quadrature weights.
        knots (callable[int, [np.ndarray[float]]]): Function that returns the 1D nodes for a given number of knots. Should be compatible with `lev2knots`.
        lev2knots (callable[int, [int]]): Function mapping an index `n` to the number of knots/nodes to use.
        interpolant (str, optional): The type of 1D interpolant to use. Currently only supports "pw_lin_extrapolated" (piecewise linear interpolant with extrapolation). Defaults to "pw_lin_extrapolated".

    Returns:
        np.ndarray: 1D array of objects. Each entry corresponds to an index up to `max_n`. The j-th entry is an array of 1D quadrature weights corresponding to the 1D nodes for the j-th index.

    Notes:
        - The number of nodes for each index `n` is determined as `lev2knots(n)` for `n = 0, ..., max_n - 1`.
        - `knots(m)` returns the 1D nodes (works only if `m` is generated by `lev2knots`).
        - Currently implemented only for the piecewise linear interpolant with extrapolation ("pw_lin_extrapolated").
        - integral (a*x+b)* exp**(-x**2/2)/sqrt(2*pi) = 
            -a/sqrt(2*pi)*exp**(-x**2/2) + sqrt(pi/2)*b*erf(x/sqrt(2))
    """
    
    
    prim_integrand = lambda a, b, x : -a/sqrt(2*pi)*exp**(-x**2/2) + sqrt(pi/2)*b*erf(x/sqrt(2))

    W_1d = np.empty(max_n, dtype=object)
    for n in range(max_n):
        num_knots = lev2knots(n)
        nodes = knots(num_knots)
        W_1d[n] = np.zeros(num_knots)
        if interpolant == "pw_lin_extrapolated":
            if num_knots == 0:
                W_1d[0] = np.array([1.0])
            else:
                # Node 0
                y, yn = nodes[0], nodes[1]
                f = lambda x : prim_integrand(1/(y-yn), -yn/(y-yn), x)
                int_y_yn = f[yn]-f[y]
                W_1d[n][0] = norm.cdf(nodes[0]) + int_y_yn







            raise NotImplementedError("Only 'pw_lin_extrapolated' interpolant is supported.")
    return W_1d



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
