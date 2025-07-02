from math import sqrt, pi, exp, erf
import numpy as np
from sgmethods.sparse_grid_interpolant import SGInterpolant
from sgmethods.nodes_1d import opt_guass_nodes_nest
from sgmethods.multi_index_sets import compute_mid_set_fast
import numbers
import warnings
from scipy.special import erf


def profit_sllg(nu, p=2):
    """Compute the sparse grid profits of given multi-indices.
    For piecewise-polynomials of degree p-1 and parametric SLLG.

    Args:
        nu (numpy.ndarray[float]): 2D array, each row is a multi-index.
        p (int): Integer >= 2. Piecewise polynomial interpolation degree + 1.

    Returns:
        numpy.ndarray[float]: 1D array of (non-negative) profits. Length equals number of rows of nu.
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

    w1 = np.asarray(nu == 1).nonzero()
    w2 = np.asarray(nu > 1).nonzero()

    # Levels Levy-Ciesielski expansion (same shape as nu)
    nn = np.arange(1, D + 1, 1)  # linear indices mids
    ell = np.ceil(np.log2(nn))  # log-indices mids
    ell = np.reshape(ell, (1, ell.size))
    ell = np.repeat(ell, N_nu, axis=0)

    # The regularity weights (radii of C-disks of holomorphic extension)
    rho = np.zeros_like(nu, dtype=float)
    if not (w1[0].size == 0):  # if entry 0 is empty, the other is too
        rho[w1] = 2.0 ** (3.0 / 2.0 * ell[w1])
    if not (w2[0].size == 0):
        rho[w2] = 2.0 ** (1.0 / 2.0 * ell[w2])

    # Compute value
    V_comps = np.ones_like(rho)
    V_comps[w1] = C1 * np.power(rho[w1], -1.0)
    V_comps[w2] = C2 * np.power(2.0, -p * nu[w2] * rho[w2])
    value = np.prod(V_comps, axis=1)

    # Compute work
    W_comps = (2 ** (nu + 1) - 2) * (p - 1) + 1
    work = np.prod(W_comps, axis=1)

    return value / work


# TODO Currently assuming Gaussian samples for SLLG. Add more options/make an input
# TODO improve efficiency computation integral: instead of MC, do exact computation based on inclusion-exclsion and TP structure
def compute_quadrature_params(min_n_samples, dim, P, knots, lev2knots):
    """
    Computes quadrature nodes and weights for sparse grid quadrature using a Gaussian distribution.
    Args:
        min_n_samples (int): Minimum number of desired quadrature nodes.
        dim (int): Number of dimensions for the quadrature. 
        distrbution (str, optional): Integral measure. Only "gauss" is supported for now. Defaults to "gauss".
        eps (float, optional): Accuracy integration basis functions. Defaults to 1.0e-4.
    Returns:
        quad_nodes (np.ndarray): Array of quadrature nodes (sparse grid points of dimension `dim`).
        quad_weights (np.ndarray): Array of quadrature weights corresponding to the nodes.
    Notes:
        - The quadrature weights are with inclusion-exclusion formula
        - The function relies on external functions/classes: `opt_guass_nodes_nest`, `profit_sllg`, `compute_mid_set_fast`, and `SGInterpolant`.
    """

    # Check input
    if not isinstance(min_n_samples, numbers.Number) or min_n_samples <= 0:
        raise ValueError("n_samples must be a positive integer.")
    if not (isinstance(dim, numbers.Number) ) or dim <= 0:
        raise ValueError("dim_samples must be a positive integer.")
    

    # Find mid_set such that: #SG > min_n_samples
    if min_n_samples == 1:
        mid_set = np.array([[0]])
        I = SGInterpolant(mid_set, knots, lev2knots)
    else:
        mid0 = np.zeros((1, 1), dtype=int)
        min_p = P(mid0)[0]
        while True:
            # NB Settgin N=dim guarantees dimension does not grow beyoind dim
            mid_set = compute_mid_set_fast(P, min_p, dim)
            I = SGInterpolant(mid_set, knots, lev2knots)
            if I.num_nodes > min_n_samples:
                break
            min_p *= 0.5

    # Quadarature samples = current sparse grid + enforce dimensionality
    quad_nodes = np.zeros((I.num_nodes, dim))
    quad_nodes += I.SG

    quad_weights = compute_quadrature_wights_incl_excl(dim, I)

    return quad_nodes, quad_weights


def compute_quadrature_wights_incl_excl(dim, sg_interp):
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
    num_nodes = sg_interp.num_nodes
    max_nu = np.amax(sg_interp.mid_set.flatten())
    lev2knots = sg_interp.lev2knots
    knots = sg_interp.knots
    ww = np.zeros(num_nodes)  # quadrature weights, to return

    # Compute 1d quadrature weights
    W_1d = compute_1d_quadrature_weights(max_nu, lev2knots, knots)

    # Inclusion-exclusion formula: Loop over active mids
    for i, nu in enumerate(sg_interp.active_mids):
        # 1 Compute TP quadrature weight from I_{nu}
        dim_nu = nu.shape[0]
        w_1d = [W_1d[nu[d]] for d in range(dim_nu)]
        w_tp = np.array(np.meshgrid(*w_1d, indexing="ij"))
        w_tp = np.prod(w_tp, axis=0)
        w_tp_flat = w_tp.reshape(-1)

        # TODO sg_interp.where_nu[i]: where the TP nodes of level bnu_i (ACTIVE mid) are located in tp.interp.SG
        # Add to SG weights with inclusion-exclusion formula
        ww[sg_interp.where_tp[i]] += sg_interp.combination_coeffs[i] * w_tp_flat
    return ww


def compute_1d_quadrature_weights(
    max_nu, lev2knots, knots, interpolant="pw_lin_extr"
):
    if interpolant != "pw_lin_extr":
        raise NotImplementedError("For now only pw. lin.+extrapolation supported.")

    def prim_integrand(a, b, x):
        # With SimPy: Primitive of (ax+b)* e^(-x^2/2)
        p = -a * np.exp(-(x**2) / 2) + sqrt(2 * pi) * b * erf(sqrt(2) * x / 2) / 2
        p /= sqrt(2 * pi)
        return p

    ww_1d = []
    for nu in range(max_nu+1):  # include max_nu
        n_nodes = lev2knots(nu)
        ww_curr = np.zeros(n_nodes)
        nodes = knots(n_nodes)
        if n_nodes == 1:  # constant itnerpolation
            ww_curr = np.array([1.0])
        else:
            # Integrate to the LEFT of current node
            nodes_left = np.concatenate(([-np.inf], nodes[0:-1]))
            a = 1 / (nodes - nodes_left)
            a[0] = -1 / (nodes[1] - nodes[0])  # edit for leftmost integral (unbounded)
            b = -nodes_left * a
            b[0] = -nodes[1] / (nodes[1] - nodes[0])  # edit for leftmost integral
            nodes_left[1] = -np.inf  # the second (idx 1) basis function also unbounded
            int_left = prim_integrand(a, b, nodes) - prim_integrand(a, b, nodes_left)
            ww_curr += int_left
            # Integrate to the RIGHT of current node ASSUMING simmetry basis functions
            int_right = int_left[::-1]
            ww_curr += int_right
        ww_1d.append(ww_curr)
    return ww_1d








# ------------------------- DEPRECATED (inefficient) ------------------------ #
def compute_quadarture_wights_MC(dim_samples, eps, sg_interp):
    """Compute quadrature weights of sparse grid quadrature as :
        weights[i] = int_{Gamma} L_{y_i} d mu,
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
