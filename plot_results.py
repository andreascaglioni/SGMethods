import matplotlib.pyplot as plt
import numpy as np



A = np.loadtxt("results/freeN/convergenge_pwLin_aniso_SLLG.csv")
print(A)
plt.plot(A[0,:], A[1,:], '.-', label="N=1")
plt.show()


# for n in [1,2,4,8,16]:
#     A = np.loadtxt("convergenge_pwLin_aniso_SLLG_"+str(n)+".csv")
#     print(A)
#     plt.plot(A[0,:], A[1,:], '.-', label="grown")
# plt.show()