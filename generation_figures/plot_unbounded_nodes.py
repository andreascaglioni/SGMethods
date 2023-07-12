import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))

from SGMethods.ScalarNodes import unboundedKnotsNested

lev2knots = lambda nu: 2**(nu+1)-1
for nu in range(5):
    n = lev2knots(nu)
    kk = unboundedKnotsNested(n,p=2)
    plt.plot(kk, nu*np.ones_like(kk), '.')
    print(kk)
plt.show()