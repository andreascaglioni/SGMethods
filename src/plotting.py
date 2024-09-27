import numpy as np
from math import exp
import matplotlib.pyplot as plt

def rate(err, nDofs):
    err = np.array(err)
    nDofs = np.array(nDofs)
    return -np.log(err[1::]/err[:-1:])/np.log(nDofs[1::]/nDofs[:-1:])

def LSFitLoglog(x, y):
    X = np.log(x)
    Y = np.log(y)
    A = np.vstack([X, np.ones(len(X))]).T
    alpha, gamma = np.linalg.lstsq(A, Y, rcond=None)[0]
    return alpha, exp(gamma)

def plot_2D_midset(midset):
    assert midset.shape[1] <= 2
    if midset.shape[1] == 1:
        plt.plot(midset, np.zeros(midset.shape[0]), 's')
    else:
        plt.plot(midset[:,0], midset[:,1], 's')

