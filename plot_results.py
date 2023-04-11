import matplotlib.pyplot as plt
import numpy as np


# A = np.loadtxt("results/pwQuadratic/convergenge_pwQuadr_aniso_SLLG.csv")
A = np.loadtxt("RESULTS_NEW_PROFIT_H0/convergenge_pwLinear_SLLG_NEWPROFIT.csv")
plt.loglog(A[0,:], A[1,:], '.-')
np.savetxt("results_new_profit.csv", np.transpose(A[0:2,:]))

A = np.loadtxt("RESULTS_OLD_PROFIT_H0/convergenge_pwLinear_SLLG.csv")
# A = np.loadtxt("RESULTS_NEW_PROFIT_H0_QUADRATIC/convergenge_pwQuadratic_SLLG_NEWPROFIT.csv")
# A = A[:,0:-1]
plt.loglog(A[0,:], A[1,:], '.-')
xx = A[0,:]
plt.loglog(xx, xx**(-0.25), '-k')
# plt.loglog(xx, xx**(-0.5), '-k')
plt.show()

np.savetxt("results_old_profit.csv", np.transpose(A))

# A = np.loadtxt("RESULTS_NEW_PROFIT_H0/convergenge_pwLinear_SLLG_NEWPROFIT.csv")
# plt.plot(A[0,:], A[2,:], 'o-')
# # A = np.loadtxt("RESULTS_OLD_PROFIT_H0/convergenge_pwLinear_SLLG.csv")
# A = np.loadtxt("RESULTS_NEW_PROFIT_H0_QUADRATIC/convergenge_pwQuadratic_SLLG_NEWPROFIT.csv")
# plt.plot(A[0,:], A[2,:], '.-')
# xx = A[0,:]
# # plt.loglog(xx, xx**(-0.25), '-k')
# # plt.loglog(xx, xx**(-0.5), '-k')
# plt.show()




# for n in [1,2,4, 8]:
#     A = np.loadtxt("results/pwQuadratic/convergenge_pwLin_aniso_SLLG_"+str(n)+".csv")
#     plt.loglog(A[0,:], A[1,:], '+-', label="p2 "+str(n))

# for n in [1,2,4, 8, 16]:
#     A = np.loadtxt("results/Lagrange/convergenge_SLLG_N_"+str(n)+".csv")
#     plt.loglog(A[0,:], A[1,:], '.-', label="L "+str(n))
# plt.legend()
# plt.show()



######################## compare 2 leja nodes: w=e^-x^2 and e^{-x^2/2}
# A = np.loadtxt("results/Lagrange/convergenge_SLLG_N_1.csv")
# plt.loglog(A[0,:], A[1,:], '.-', label="1")
# A = np.loadtxt("results/convergenge_SLLG_N_1alt_leja.csv")
# plt.loglog(A[0,:], A[1,:], '.-', label="2")
# plt.legend()
# plt.show()