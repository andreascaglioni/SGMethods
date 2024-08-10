import sys, os
sys.path.insert(1, os.path.join(os.path.expanduser("~"), 'workspace/SGMethods'))

from xml.etree.ElementTree import tostring
import numpy as np
from math import floor, log
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
from SGMethods.multi_index_sets import SmolyakMidSet

def dim_indep_midset(N, L, deg, rho):
    """INPUT N int number parametric dimensions
             L int size parameter
             deg int degree 1D interpolant operator
             rho list length N aniso coeffcients
       OUTPUT I np array shape (# SG nodes, N)"""
    
    assert(rho.shape == (N,))
    assert(deg>0)
    assert(N>0)
    assert(np.all(rho>0))

    if L <= 0 :
        return np.zeros((1, N), dtype=int)
    elif N == 1:
        b = max(0, floor(L-0.5*log(rho[0], 2)))
        I = np.linspace(0, b, b+1)
        I = np.reshape(I, (1,)+I.shape)
        return I.T
    else: # L>0 and N>1
        I = np.zeros((0, N), dtype=int)
        LIM = max(0,floor(L - 0.5*log(rho[0],2)))

        # case nu_1 = 0
        nu1=0
        rest = dim_indep_midset(N-1, L, deg, rho[1::])
        newFirstCol = nu1*np.ones((rest.shape[0], 1), dtype=int)
        newRows = np.concatenate((newFirstCol, rest), 1)
        I = np.concatenate((I, newRows), 0)
        # case nu_1>0
        for nu1 in range(1, LIM+1):  # nbb last int in range() excluded
            rest = dim_indep_midset(N-1, L - nu1 - 0.5*log(rho[0], 2), deg, rho[1::])
            newFirstCol = nu1*np.ones((rest.shape[0], 1), dtype=int)
            newRows = np.concatenate((newFirstCol, rest), 1)
            I = np.concatenate((I, newRows), 0)
        return I


def plot_sections(I, dim1, dim2):
    N = I.shape[1]
    assert(0<=dim1 and dim1<N)
    assert(0<=dim2 and dim2<N)
    assert(dim1 != dim2)
    plt.scatter(I[:,dim1], I[:,dim2])
    plt.xlabel(dim1)
    plt.ylabel(dim2)


# TEST
deg = 2
beta = 0.5

###################### TEST 1 visualize in 2D
# N = 2
# rho = 2**(beta*np.ceil(np.log2(np.linspace(1, N, N))))
# print("rho: ", rho)
# L = 10
# I = dim_indep_midset(N, L, deg, rho)
# print(I)

# plt.scatter(I[:,0], I[:,1])

# plt.figure(1)
# plot_sections(I,0,1)
# plt.figure(2)
# plot_sections(I,0,10)
# plt.figure(3)
# plot_sections(I,0,19)

# plt.show()

###################### TEST 2 visualize in 3D
# N=20
# rho = 2**(beta*np.ceil(np.log2(np.linspace(1, N, N))))
# L=6
# I = dim_indep_midset(N, L, deg, rho)
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(I[:,0], I[:,10], I[:,19])
# plt.show()


###################### TEST 3 dimension dependence: plot cardinality vs L for different N
# for N in range(1,10):
#     print("N=", N)
#     rho = 2**(beta*np.ceil(np.log2(np.linspace(1, N, N))))
#     nNodes = []
#     for L in range(30):
#         I = dim_indep_midset(N, L, deg, rho)
#         nNodes.append(I.shape[0])
#     plt.loglog(np.linspace(0,L, L+1), nNodes,'.-')
#     plt.xlabel("L")
#     plt.draw()

#     plt.pause(.001)

# plt.show()

###################### TEST 3 dimension dependence: plot cardinality vs n for different L
for L in range(1, 10):
    print("L=", L)
    nNodes = []
    # nNodes2 = []
    for N in range(1,10):
        rho = 16*2**(beta*np.ceil(np.log2(np.linspace(1, N, N))))
        I = dim_indep_midset(N, L, deg, rho)
        nNodes.append(I.shape[0])
        plt.loglog(np.linspace(1,N,N), nNodes,'.-')
        plt.xlabel("N")

        # compare w classical smolyak midset
        #     I2 = SmolyakMidSet(L, N)
        #     nNodes2.append(I2.shape[0])
        #     plt.semilogy(np.linspace(1,N,N), nNodes2,'.-')
    plt.draw()
    plt.pause(1)
plt.show()

# L=100
# nn = np.linspace(1,L,L)
# yy = np.cumsum(np.log(L/nn+1))
# plt.plot(nn, yy, '.-')
# plt.show()


#################### test which C makes the rho_i sum smaller than 1
# N = 10000
# alpha = 2
# rho = np.cumsum(2**(-beta*alpha*np.ceil(np.log2(np.linspace(1, N, N)))))
# print(rho)