import numpy as np


import numpy as np
import matplotlib.pyplot as plt

def fitToExponential(x, y):
    y = np.log(y)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    expC = -m
    multC = np.exp(c)
    return multC, expC
    
A = np.loadtxt("probe_SLLG_regularity_parameters.csv")
print(A)
xx = np.linspace(1,15,15)
mC = np.zeros(A.shape[0])
eC = np.zeros(A.shape[0])
for n in range(A.shape[0]):
    # remove if smaller than 5.e-10
    ww = np.where(A[n]>5.e-10)
    xc =xx[ww]
    yc = np.squeeze(A[n, ww])

    
    plt.semilogy(xc, yc, '.-')
    mC[n], eC[n] = fitToExponential(xc, yc)
    xl=np.linspace(xc[0], xc[-1])
    ss = "e="+ ('%.2f' % eC[n] ) +" C=" + ('%.2f' % mC[n])
    plt.semilogy(xl, mC[n]*np.exp(-eC[n]*xl), '-k', label=ss)
plt.legend()
plt.show()

plt.plot(eC, '.-', label='eC')
plt.plot(mC, '.-', label='mC')
xx = np.linspace(1,A.shape[0]-1, A.shape[0]-1)
plt.plot(np.append(1, 2**(0.5*np.floor(np.log2(xx)))), '.-k', label="theory")
plt.legend()

plt.show()