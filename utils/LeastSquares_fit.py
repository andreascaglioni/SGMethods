import numpy as np
from numpy.linalg import lstsq
import matplotlib.pyplot as plt
from math import exp


table = np.loadtxt("results/smoothing_profit_deg_3/convergenge_pwLin_pb2_fasterSmoothProfitL2Long.csv", delimiter=",")
x = table[:,0]
y = table[:,1]

X = np.log(x)
Y = np.log(y)

A = np.vstack([X, np.ones(len(X))]).T
alpha, gamma = np.linalg.lstsq(A, Y, rcond=None)[0]

print(alpha, gamma)

plt.loglog(x, y, 'o', label='Original data', markersize=10)
plt.loglog(x, exp(gamma) * np.power(x, alpha), 'r', label='Fitted line')
plt.legend()
plt.show()