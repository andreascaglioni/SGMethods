import unittest
import numpy as np
from math import sqrt
from scipy.special import erfinv
from src.nodes_1d import unboundedKnotsNested


class Test1Node(unittest.TestCase):
    def test_list_int(self):
        """ Test that the 1 point nodes family is just (0.)
        """
        y = unboundedKnotsNested(1)
        self.assertAlmostEqual(y, np.array([0.]))


if __name__ == '__main__':
    unittest.main()


def unboundedKnotsTest(n):
    assert (n % 2 == 1)
    m = int((n-1)/2)
    xx = np.linspace(-m, m, n)/(m+1)
    return 2*sqrt(5) * erfinv(xx)


def unboundedKnotsTest2(n):
    assert (n % 2 == 1)
    m = int((n-1)/2)
    xx = np.linspace(1, m, m)/(m+1)
    v = 2*sqrt(2) * erfinv(xx)
    vr = np.sort(-v)
    return np.hstack((vr, np.array([0.]), v))
