import sys
import math

sys.path.append("./")
from sgmethods.quadrature import compute_quadrature_params


# test prodfit used for qudarature function
from sgmethods.quadrature import profit_sllg
from sgmethods.multi_index_sets import tensor_product_mid_set
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sgmethods.utils import float_f


def test_profit_definition():
    """
    - Check that the profit at the first index is close to 1.0.
    - Check that the profit values are monotonically non-increasing.
    - Visualizes the profit values as a 3D scatter plot with respect to Lambda parameters.
    """
    N_params = 2
    Lambda = tensor_product_mid_set(w=4, N=N_params)
    pp = profit_sllg(Lambda)

    # Test general propeties Profit
    assert math.isclose(pp[0], 1.0)
    for d_curr in range(1, N_params + 1):
        assert pp[d_curr] <= pp[0]  # Monotonic

    # Visual test
    for i, p in enumerate(pp):
        print(Lambda[i], "profit", p)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(Lambda[:, 0], Lambda[:, 1], np.log(pp), c="b", marker="o")
    ax.set_xlabel("Lambda[0]")
    ax.set_ylabel("Lambda[1]")
    ax.set_zlabel("Profit")
    plt.title("3D Scatter Plot of Lambda and Profit")
    plt.show()


def test_quadrature_execution():
    """Tests quadrature parameter computation and visualizes nodes and weights in 3D."""

    min_n = 10
    dim = 2
    n, w = compute_quadrature_params(min_n, dim)
    print("Nodes", n)
    print("Weights", w)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(n[:, 0], n[:, 1], w, c="b", marker="o")
    ax.set_xlabel("y_0")
    ax.set_ylabel("y_1")
    ax.set_zlabel("w(y)")
    plt.title("3D Scatter Plot of nodes and weights")
    plt.show()


def test_quadrature_simple_f():
    """
    Test simple quadrature rules for 1D integrals using different functions:
    - f(x) = 1
    - f(x) = x
    - f(x) = x**2
    """

    n_samples = 40
    d = 1

    # ----------------------- f Constant=1 -> int f = 1 ---------------------- #
    print("f = 1")
    f = lambda x: 1 + 0 * x[0]  # noqa: E731
    n, w = compute_quadrature_params(min_n_samples=n_samples, dim_samples=d)

    assert n.shape[0] >= n_samples, "not enough samples"

    If_approx = np.dot(w, f(n.T))
    If = 1
    print("If=", If, "; If approx =", If_approx, "; Error =", np.abs(If - If_approx))

    # -------------------------- f = x -> int f = 0 -------------------------- #
    print("f = x")
    f = lambda x: x  # noqa: E731
    n, w = compute_quadrature_params(min_n_samples=n_samples, dim_samples=d)
    If_approx = np.dot(w, f(n.T).squeeze())
    If = 0
    print("If=", If, "; If approx =", If_approx, "; Error =", np.abs(If - If_approx))

    # ------------------------- f = x**2 -> int f = 1 ------------------------ #
    print("f = x**2")
    f = lambda x: x**2  # noqa: E731
    n, w = compute_quadrature_params(min_n_samples=n_samples, dim_samples=d)
    If_approx = np.dot(w, f(n.T).squeeze())
    If = 1
    print("If=", If, "; If approx =", If_approx, "; Error =", np.abs(If - If_approx))


def test_conv_quadr_simple_f():
    """
    Tests the convergence of a quadrature method for integrating f(x) = x^2 over varying sample sizes and tolerances.
    """

    f = lambda x: x**2
    If = 1  # exact
    d = 1
    nns = 2 ** np.arange(10)
    err = np.ones_like(nns, dtype=float)
    for eps in [1.0e-1, 1.0e-2, 1.0e-3]:
        print("eps:", eps)
        for i, ns in enumerate(nns):
            n, w = compute_quadrature_params(min_n_samples=ns, dim_samples=d, eps=eps)
            If_approx = np.dot(w, f(n.T).squeeze())
            err[i] = np.abs(If - If_approx)
            print(
                "    ns",
                ns,
                "If_approx",
                float_f(If_approx),
                "err",
                float_f(If - If_approx),
            )

        # plot line
        plt.loglog(nns, err, ".-", label=str(eps))
        plt.legend()
    plt.loglog(nns, 1.0 / nns, "k-", label="n^(-1)")
    plt.loglog(nns, 1.0 / np.sqrt(nns), "k-", label="n^(-1/2)")
    plt.show()


# test_profit_definition()
test_quadrature_execution()
# test_quadrature_simple_f()
# test_conv_quadr_simple_f()
