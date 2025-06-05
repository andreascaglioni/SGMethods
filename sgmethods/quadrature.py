import numpy as np
from sgmethods.sparse_grid_interpolant import SGInterpolant
from sgmethods.nodes_1d import opt_guass_nodes_nest
from sgmethods.multi_index_sets import compute_mid_set_fast


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

    # Levels of the Levy-Ciesielski expansion.
    nn = np.arange(1, D + 1, 1)  # linear indices mid in nu from (1)
    # BUG nn = 1 + np.arange(0, D + 1, 1)
    ell = np.ceil(np.log2(nn))  # log-indices mid in nu (from 0)

    # Reshape to make the same shape as nu. Each entry of ell denotes the log index of the correpsoding entry of nu
    ell = np.reshape(ell, (1, ell.size))
    ell = np.repeat(ell, N_nu, axis=0)

    # The regularity weights (radious of domain of holomorphic extension);
    # We use them to define the value.
    # NB they depend on nu (even if the  regularity of the function does not)
    # because a maximal domain of holomorphy is unknown.
    rho = np.zeros_like(nu, dtype=float)
    if not (w1[0].size == 0):  # if 1 tuple element is empty, the other is too
        rho[w1] = 2.0 ** (3.0 / 2.0 * ell[w1])
    if not (w2[0].size == 0):
        rho[w2] = 2.0 ** (1.0 / 2.0 * ell[w2])

    # Compute value of each component of each multi-index. Then multiply them to
    # obtain the value of the whole multi-index
    V_comps = np.ones_like(rho)
    V_comps[w1] = C1 * np.power(rho[w1], -1.0)
    V_comps[w2] = C2 * np.power(2.0, -p * nu[w2] * rho[w2])
    value = np.prod(V_comps, axis=1)

    # Work of a multi-index. Computed with product structure, as for work.
    W_comps = (2 ** (nu + 1) - 2) * (p - 1) + 1
    work = np.prod(W_comps, axis=1)

    return value / work


def compute_quadrature_params(min_n_samples, dim_samples, distrbution="gauss"):
    # Check input
    if not isinstance(min_n_samples, int) or min_n_samples <= 0:
        raise ValueError("n_samples must be a positive integer.")
    if not isinstance(dim_samples, int) or dim_samples <= 0:
        raise ValueError("dim_samples must be a positive integer.")

    if distrbution != "gauss":
        raise ValueError("Only 'gauss' distribution is supported.")

    # Assuming Gaussian samples for SLLG
    # TODO make function more flexible

    knots = lambda n: opt_guass_nodes_nest(n)  # noqa: E731
    lev2knots = lambda i: np.where(i > 0, 2**(i + 1) - 1, 1)  # noqa: E731
    P = lambda nu: profit_sllg(nu)

    # At end of loop, # sparse grid > min_n_samples
    min_p = P(np.zeros((1, 1), dtype=int))[0]
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

    # Quadrature weights[i] = \int_{\Gamma} L_{\by_i}text{d}\mu,
    # where L_{\by} denotes a Lagrange basis function of I
    # I compute observing that 
    # L_{y_i} = I[delta_{y_i}], where delta_{y_i}(y_i) = 1., 0 oth.
    # So I 
    
    # 1. define a function with #SG components. Each is 1 in only 1 SG node.
    Delta_sg = np.eye(I.num_nodes)
    
    # 2. Interpolate over a large Monte Carlo sample
    zz_mc = np.random.standard_normal((10000, dim_samples))
    ww_samples = I.interpolate(zz_mc, Delta_sg)
    
    # 3. Compute the quadrature weights as the mean over the MC samples
    quad_weights = np.mean(ww_samples, axis=0)

    return quad_nodes, quad_weights
