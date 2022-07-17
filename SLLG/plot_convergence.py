import numpy as np
import matplotlib.pyplot as plt
from math import pow

def plot_err(name, leg):
    # PLOT ANISOTROPIC
    A = np.loadtxt(name)
    nNodes = A[0]
    err = A[1]
    print("Error aniso:", err)
    rates = -np.log(err[1::]/err[0:-1:])/np.log(nNodes[1::]/nNodes[0:-1:])
    print("Rate aniso:",  rates)
    plt.loglog(nNodes, err, '.-', label=leg)
    
def exp_rate(N, xx):
    anisoVector = np.append(1, 2**(0.5*np.floor(np.log2(np.linspace(1,N-1, N-1)))))
    C = N * pow(np.prod(anisoVector), 1/N)
    return 1.e5*np.exp(- C * np.power(xx, 1/N))
    

plot_err("convergenge_SLLG_aniso_N8.csv", "a")
plot_err("convergenge_SLLG_iso_N8.csv", "i")

xx = np.linspace(1,180)
plt.loglog(xx, 1.e-1*exp_rate(8,xx), '-k')

plt.legend()
plt.show()