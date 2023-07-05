import matplotlib.pyplot as plt
import numpy as np
from utils.LeastSquares_fit import LSFitLoglog
from dolfin import *


hh = 1/np.array([1,2,4,8,16,32])
FEConv = np.loadtxt("results/MULTILEVEL/example_NONConst_IC_g_02/multilevel_preprocessing_conv_FE_nonconstIC.csv",delimiter=",")
FEConv[:,0] = hh
plt.loglog(FEConv[:,0], FEConv[:,1], 'x-',label="FE")
[powFE, factorFE] = LSFitLoglog(FEConv[1:,0], FEConv[1:,1])
factorFE = factorFE*1.05
plt.loglog(FEConv[:,0], factorFE*FEConv[:,0]**powFE, '-k')

SGConv = np.loadtxt("results/MULTILEVEL/example_NONConst_IC_g_02/multilevel_preprocessing_conv_SG_nonconstIC.csv", delimiter=",")
plt.loglog(SGConv[:,0], SGConv[:,1], '.-',label="SG")
[powSG, factorSG] = LSFitLoglog(SGConv[:,0], SGConv[:,1])
factorSG = factorSG *1.1
plt.loglog(SGConv[:,0], factorSG*SGConv[:,0]**powSG, '-k')

plt.legend()
plt.xlabel("h / # SG nodes")
plt.show()

print("FE", powFE, factorFE)
print("SG", powSG, factorSG)

# compute sparse grid cardianlities as in [Teckentrup] for several number of levels
for nLevs in range(1,7):
    K = nLevs-1
    hhCurr = hh[:nLevs:]
    CardSG = np.ceil(np.power(factorFE/factorSG * np.power(hhCurr[-1]/hhCurr, powFE) * 1/(1+K), 1/powSG))
    print(nLevs, "levels:", CardSG)