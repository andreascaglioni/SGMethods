import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
import numpy as np
import matplotlib.pyplot as plt
from SGMethods.ScalarNodes import unboundedKnotsNested


# free example
mm = ['.', 'x', '+', 'o']
for i, n in enumerate([3, 7, 15, 31]):
    kk = unboundedKnotsNested(n)
    plt.plot(kk, i*np.ones(kk.size), mm[i])
plt.legend(["3", "7", "15", "31"])
plt.grid()
plt.show()

# 0 nodes are not allowed
try:
    unboundedKnotsNested(0)
except:
    print("correctly launched error for n = 0")
# check that 1st node is 0
assert(np.linalg.norm(unboundedKnotsNested(1) - np.array([0.])) < 1.e-10)
# check that output has size n
for n in range(1, 10):
    assert(np.size(unboundedKnotsNested(n)) == n)
# TODO check nestedness





