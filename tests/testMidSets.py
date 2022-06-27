import numpy as np
from pwSG.MidSets import TPMidSet, smolyakMidSet

############################# test smolyak midset #################################
# free example
I = smolyakMidSet(3, 2)
print(I)
I = smolyakMidSet(2, 3)
print(I)

# Check assert: N=0 gives error
N = 0
try:
    smolyakMidSet(10, N)
    print("assert N = 0 not triggered")
except:
    print("assert N = 0 correctly launced")

# N = 1 (same as linspace)
N = 1
for w in range(0, 10):
    I = smolyakMidSet(w, N)
    ITest = np.linspace(0, w, w + 1)
    ITest = np.reshape(ITest, (1,) + ITest.shape)
    ITest = ITest
    ITest = ITest.T
    assert (np.all(ITest == I))

# N = 2
N = 2
w = 0
I = smolyakMidSet(w, N)
assert (np.all(I == np.zeros((1, N))))
w = 1
I = smolyakMidSet(w, N)
ITest = np.array([[0, 0], [0, 1], [1, 0]])
assert (np.all(I == ITest))


############################# test TP midset #################################
# # free example
# I = TPMidSet(3, 2)
# print(I)
# I = TPMidSet(2, 3)
# print(I)
#
# # Check assert: N=0 gives error
# N = 0
# try:
#     TPMidSet(10, N)
#     print("assert N = 0 not triggered")
# except:
#     print("assert N = 0 correctly launced")
#
# # N = 1 (same as linspace)
# N = 1
# for w in range(0, 10):
#     I = TPMidSet(w, N)
#     ITest = np.linspace(0, w, w + 1)
#     ITest = np.reshape(ITest, (1,) + ITest.shape)
#     assert (np.all(ITest == I))
#
# # N = 2
# N = 2
# w = 0
# I = TPMidSet(w, N)
# assert (np.all(I == np.zeros((1, N))))
# w = 1
# I = TPMidSet(w, N)
# ITest = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# assert (np.all(I == ITest))
