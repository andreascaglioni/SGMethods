import numpy as np
import matplotlib.pyplot as plt


# PLOT ANISOTROPIC
A = np.loadtxt("results/convergenge_SLLG_aniso_Nh32_N16.csv")
nNodes = A[0]
err = A[1]
print("Error aniso:", err)
rates = -np.log(err[1::]/err[0:-1:])/np.log(nNodes[1::]/nNodes[0:-1:])
print("Rate aniso:",  rates)
plt.loglog(nNodes, err, '.-')

# PLOT ISOTROPIC
A = np.loadtxt("results/convergenge_SLLG_iso_Nh32_N16.csv")
nNodes = A[0]
err = A[1]
print("Error iso:", err)
rates = -np.log(err[1::]/err[0:-1:])/np.log(nNodes[1::]/nNodes[0:-1:])
print("Rate iso:",  rates)
plt.loglog(nNodes, err, '.-')
# plt.loglog(nNodes, 1.e-5/nNodes, '-k')
plt.show()