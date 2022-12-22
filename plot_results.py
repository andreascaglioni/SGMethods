import matplotlib.pyplot as plt
import numpy as np



# A = np.loadtxt("results/freeN/convergenge_pwLin_aniso_SLLG.csv")
# print(A)
# plt.plot(A[0,:], A[1,:], '.-', label="N=1")
# plt.show()


# for n in [1,2,4, 8]:
#     A = np.loadtxt("results/pwQuadratic/convergenge_pwLin_aniso_SLLG_"+str(n)+".csv")
#     plt.loglog(A[0,:], A[1,:], '+-', label="p2 "+str(n))

for n in [1,2,4, 8, 16]:
    A = np.loadtxt("results/Lagrange/convergenge_SLLG_N_"+str(n)+".csv")
    plt.loglog(A[0,:], A[1,:], '.-', label="L "+str(n))
plt.legend()
plt.show()



######################## compare 2 leja nodes: w=e^-x^2 and e^{-x^2/2}
# A = np.loadtxt("results/Lagrange/convergenge_SLLG_N_1.csv")
# plt.loglog(A[0,:], A[1,:], '.-', label="1")
# A = np.loadtxt("results/convergenge_SLLG_N_1alt_leja.csv")
# plt.loglog(A[0,:], A[1,:], '.-', label="2")
# plt.legend()
# plt.show()