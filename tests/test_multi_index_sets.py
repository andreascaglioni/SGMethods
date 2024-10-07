import pytest
import numpy as np
from sgmethods.multi_index_sets import tensor_product_mid_set, aniso_smolyak_mid_set, mid_set_profit_free_dim

@pytest.mark.parametrize("w, N, expected_mid_set", [
    (0, 1, np.array([[0]])),
    (1, 1, np.array([[0], [1]])),
    (1, 2, np.array([[0, 0], [0, 1], [1, 0], [1, 1]])),
    (2, 2, np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], 
                     [2, 1], [2, 2]]))
])
def test_TPMidSet(w, N, expected_mid_set):
    test_mid_set = tensor_product_mid_set(w, N)
    assert np.array_equal(test_mid_set, expected_mid_set)

@pytest.mark.parametrize("w, N, a, expected_mid_set", [
    (0, 1, [1], tensor_product_mid_set(0,1)),
    (1, 1, [1], tensor_product_mid_set(1,1)),
    (1, 2, [1, 1], [[0, 0], [0, 1], [1, 0]]),
    (1, 2, [1, 0.5], np.array([[0, 0], [0, 1], [0, 2], [1, 0]]))
])
def test_anisoSmolyakMidSet(w, N, a, expected_mid_set):
    test_mid_set = aniso_smolyak_mid_set(w, N, a)
    assert np.array_equal(test_mid_set, expected_mid_set)

def test_computeMidSetFast_freeDim():
    PMin = 1.e-2
    sparse_vec = lambda n : np.linspace(1, n, n)**2
    Profit = lambda bnu : 2.**(-np.dot(bnu, sparse_vec(np.atleast_1d(bnu).size)))
    Lambda = mid_set_profit_free_dim(profit_fun=Profit, minimum_profit=PMin)
    assert np.all(
        Lambda == np.array([[0, 0],
                            [0, 1],
                            [1, 0],
                            [1, 1],
                            [2, 0],
                            [2, 1],
                            [3, 0],
                            [4, 0],
                            [5, 0],
                            [6, 0]])
        )