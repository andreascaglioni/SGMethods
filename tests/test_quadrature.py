import sys

sys.path.append("./")
from sgmethods.quadrature import compute_quadrature_params


# test prodfit used for qudarature function
from sgmethods.quadrature import profit_sllg
from sgmethods.multi_index_sets import tensor_product_mid_set
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def test_profit_definition():
    Lambda = tensor_product_mid_set(w=4, N=2)
    pp = profit_sllg(Lambda)
    for (
        i,
        p,
    ) in enumerate(pp):
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
    min_n = 10
    dim = 2
    (
        n,
        w,
    ) = compute_quadrature_params(min_n, dim)
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


# def test_quadrature_simple_f():
ns = 400
d = 1

# f Constant -> int f = 1
f = lambda x: 1 + 0 * x[0]
(
    n,
    w,
) = compute_quadrature_params(min_n_samples=ns, dim_samples=d)
If_approx = np.dot(w, f(n.T))
If = 1
print("f=1; If=", If, "; If approx =", If_approx, "; Error =", np.abs(If - If_approx))

# f = x -> int f = 0
f = lambda x: x
(
    n,
    w,
) = compute_quadrature_params(min_n_samples=ns, dim_samples=d)
If_approx = np.dot(w, f(n.T).squeeze())
If = 0
print("f=x; If=", If, "; If approx =", If_approx, "; Error =", np.abs(If - If_approx))

# f = x**2 -> int f = 1
f = lambda x: x**2
(
    n,
    w,
) = compute_quadrature_params(min_n_samples=ns, dim_samples=d)
If_approx = np.dot(w, f(n.T).squeeze())
If = 1
print(
    "f=x**2; If=", If, "; If approx =", If_approx, "; Error =", np.abs(If - If_approx)
)


from sgmethods.sparse_grid_interpolant import SGInterpolant
from sgmethods.nodes_1d import opt_guass_nodes_nest
from sgmethods.multi_index_sets import compute_mid_set_fast

knots = lambda n: opt_guass_nodes_nest(n)  # noqa: E731
lev2knots = lambda i: np.where(i > 0, 2 ** (i + 1) - 1, 1)  # noqa: E731
P = lambda nu: profit_sllg(nu)

# At end of loop, # sparse grid > min_n_samples
min_p = P(np.zeros((1, 1), dtype=int))[0]
decrease_min_p = True
while decrease_min_p:
    mid_set = compute_mid_set_fast(P, min_p, d)
    I = SGInterpolant(mid_set, knots, lev2knots)
    if I.num_nodes > ns:
        decrease_min_p = False
    else:
        min_p *= 0.5

f_on_sg = I.sample_on_SG(f)
xx = np.random.standard_normal(20)
xx = np.sort(xx)
If = I.interpolate(xx, f_on_sg)

plt.plot(xx, If, ".", label="If")
xx = np.sort(np.random.standard_normal(1000))
plt.plot(xx, f(xx), label="f")
plt.legend()
plt.show()