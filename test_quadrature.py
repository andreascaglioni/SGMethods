import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append("./")
from sgmethods.quadrature import profit_sllg, compute_quadrature_params
from sgmethods.multi_index_sets import tensor_product_mid_set
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
        assert pp[d_curr] <= pp[0]  # Monotonicity

    # Visual test
    for i, p in enumerate(pp):
        print(Lambda[i], "profit", p)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(Lambda[:, 0], Lambda[:, 1], np.log(pp), c="b", marker="o")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.title("3D Scatter Plot of log(Profit)")
    plt.show()


def test_quadrature_execution():
    """Tests quadrature parameter computation and visualizes nodes and weights in 3D."""

    n, w = compute_quadrature_params(min_n_samples=10, dim_samples=2)

    print(f"{'Node':>11} | {'Weight':>8}")
    print("-" * 25)
    for node, weight in zip(n, w):
        node_str = " ".join(f"{x:5.2f}" for x in node)
        print(f"{node_str:>10} | {weight:>8.4f}")

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

    # --------------------------- f = 1 > int f = 1 -------------------------- #
    f = lambda x: 1 + 0 * x[0]  # noqa: E731
    If = 1
    print("f=1, If =", If)
    n, w = compute_quadrature_params(min_n_samples=n_samples, dim_samples=d)
    assert n.shape[0] >= n_samples, "not enough samples"
    If_approx = np.dot(w, f(n.T))
    print("If=", If, "; If approx =", If_approx, "; Error =", np.abs(If - If_approx))

    # -------------------------- f = x -> int f = 0 -------------------------- #
    f = lambda x: x  # noqa: E731
    If = 0
    print("If=", If, "; If approx =", If_approx, "; Error =", np.abs(If - If_approx))
    # TODO complete

    # ------------------------- f = x**2 -> int f = 1 ------------------------ #
    print("f = x**2")
    f = lambda x: x**2  # noqa: E731
    If = 1
    print("f=x**2, If =", If)
    n, w = compute_quadrature_params(min_n_samples=n_samples, dim_samples=d)
    assert n.shape[0] >= n_samples, "not enough samples"
    If_approx = np.dot(w, np.squeeze(f(n.T)))
    print(
        "If approx =", float_f(If_approx), "; Error =", float_f(np.abs(If - If_approx))
    )


def test_conv_quadr_simple_f():
    """
    Tests the convergence of a quadrature method for integrating f(x) = x^2 over varying sample sizes and tolerances.
    """

    f = lambda x: x**2  # noqa: E731
    If = 1  # exact
    N = 1
    nns = 2 ** np.arange(10)
    err = np.ones_like(nns, dtype=float)
    for eps in [1.0e-1, 1.0e-2, 1.0e-3]:
        print("eps:", eps)
        for i, ns in enumerate(nns):
            n, w = compute_quadrature_params(min_n_samples=ns, dim_samples=N, eps=eps)
            If_approx = np.dot(w, f(n.T).squeeze())
            err[i] = np.abs(If - If_approx)
            print(
                f"    ns {ns:4}",
                "If_approx " + float_f(If_approx),
                "err " + float_f(If - If_approx),
            )

        plt.loglog(nns, err, ".-", label=str(eps))

    plt.legend()
    plt.loglog(nns, 1.0 / nns, "k-", label="n^(-1)")
    plt.loglog(nns, 1.0 / np.sqrt(nns), "k-", label="n^(-1/2)")
    plt.show()


def test_convergence_mean_error():
    """
    Test convergence of the mean absolute error of the quadrature rule
    over multiple random runs, for f(x) = x^2 (mean = 1).
    """
    f = lambda x: x**2  # noqa: E731
    If = 1  # exact mean for standard normal
    N = 1
    nns = 2 ** np.arange(1, 8)
    n_trials = 30
    mean_err = np.zeros_like(nns, dtype=float)
    std_err = np.zeros_like(nns, dtype=float)
    eps = 1e-3

    for i, ns in enumerate(nns):
        errs = []
        for _ in range(n_trials):
            n, w = compute_quadrature_params(min_n_samples=ns, dim_samples=N, eps=eps)
            If_approx = np.dot(w, f(n.T).squeeze())
            errs.append(np.abs(If - If_approx))
        mean_err[i] = np.mean(errs)
        std_err[i] = np.std(errs)
        print(
            f"ns={ns}, mean_err={float_f(mean_err[i])}, std_err={float_f(std_err[i])}"
        )

    plt.errorbar(nns, mean_err, yerr=std_err, fmt="o-", label="Mean abs error")
    plt.loglog(nns, 1.0 / nns, "k--", label="n^(-1)")
    plt.loglog(nns, 1.0 / np.sqrt(nns), "k-.", label="n^(-1/2)")
    plt.xlabel("Number of nodes")
    plt.ylabel("Mean absolute error")
    plt.legend()
    plt.title("Convergence of mean quadrature error (f = x^2)")
    plt.show()


if __name__ == "main":
    # test_profit_definition()
    # test_quadrature_execution()
    # test_quadrature_constant_f()
    # test_quadrature_identity()
    # test_quadrature_square()
    # test_convergence()
    test_convergence_mean_error()
