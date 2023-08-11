import unittest
import numpy as np
from SGMethods.ScalarNodes import unboundedKnotsNested


class Test1Node(unittest.TestCase):
    def test_list_int(self):
        """
        Test that the 1 point nodes family is just (0.)
        """
        y = unboundedKnotsNested(1)
        self.assertAlmostEqual(y, np.array([0.]))

if __name__ == '__main__':
    unittest.main()