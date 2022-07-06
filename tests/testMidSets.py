import numpy as np
import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))
from SGMethods.MidSets import TPMidSet, SmolyakMidSet

############################# test smolyak midset #################################
# free example
I = SmolyakMidSet(3, 2)
print(I)
I = SmolyakMidSet(2, 3)
print(I)

# Check assert: N=0 gives error
N = 0
try:
    SmolyakMidSet(10, N)
    print("assert N = 0 not triggered")
except:
    print("assert N = 0 correctly launced")

# N = 1 (same as linspace)
N = 1
for w in range(0, 10):
    I = SmolyakMidSet(w, N)
    ITest = np.linspace(0, w, w + 1)
    ITest = np.reshape(ITest, (1,) + ITest.shape)
    ITest = ITest
    ITest = ITest.T
    assert (np.all(ITest == I))

# N = 2
N = 2
w = 0
I = SmolyakMidSet(w, N)
assert (np.all(I == np.zeros((1, N))))
w = 1
I = SmolyakMidSet(w, N)
ITest = np.array([[0, 0], [0, 1], [1, 0]])
assert (np.all(I == ITest))


############################# test TP midset #################################
# free example
I = TPMidSet(3, 2)
print(I)
I = TPMidSet(2, 3)
print(I)

# Check assert: N=0 gives error
N = 0
try:
    TPMidSet(10, N)
    print("assert N = 0 not triggered")
except:
    print("assert N = 0 correctly launced")

# N = 1 (same as linspace)
N = 1
for w in range(0, 10):
    I = TPMidSet(w, N)
    ITest = np.linspace(0, w, w + 1)
    ITest = np.reshape(ITest, (1,) + ITest.shape)
    assert (np.all(ITest == I))

# N = 2
N = 2
w = 0
I = TPMidSet(w, N)
assert (np.all(I == np.zeros((1, N))))
w = 1
I = TPMidSet(w, N)
ITest = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
assert (np.all(I == ITest))
