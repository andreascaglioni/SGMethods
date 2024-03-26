import matplotlib.pyplot as plt
import numpy as np


A = np.loadtxt("results/compare_dependence_spacetime_approx/mixProfits/convergenge_SG_ProfitL1MixSimple_COARSER.csv",delimiter=",")
print(A.size)
# A=A.reshape((18,3))
plt.loglog(A[:,0], A[:,1], '.-',label="8")
A = np.loadtxt("results/compare_dependence_spacetime_approx/mixProfits/convergenge_SG_ProfitL1MixSimple_COARSE.csv",delimiter=",")
print(A.size)
# A=A.reshape((18,3))
plt.loglog(A[:,0], A[:,1], '.-',label="16")
A = np.loadtxt("results/compare_dependence_spacetime_approx/mixProfits/convergenge_SG_PwLin_ProfitMix.csv",delimiter=",")
print(A.size)
# A=A.reshape((17,3))
plt.loglog(A[:,0], A[:,1], '.-',label="32")
xx = A[:,0]
plt.loglog(xx, xx**(-0.25), '-k')
plt.loglog(xx, xx**(-0.5), '-k')
plt.legend()
plt.show()
