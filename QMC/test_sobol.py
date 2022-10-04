from scipy.stats import qmc
import matplotlib.pyplot as plt
from math import sin
import numpy as np

def f(x):
    return sin(np.sum(x))

def compute_mc_mean(f, sample):
    mc_mean = 0
    for n in range(sample.shape[0]):
        mc_mean += f(sample[n,:])
    mc_mean /= sample.shape[0]
    return mc_mean


d = 10
m = 8
sampler = qmc.Sobol(d=d, scramble=False)
sample = sampler.random_base2(m=m)

sample_rnd = np.random.uniform(0,1,(2**m, d))

qmc_mean = []
mc_mean = []
for i in range(m):
    qmc_mean.append(compute_mc_mean(f, sample[0:2**i,:]))
    mc_mean.append(compute_mc_mean(f, sample_rnd[0:2**i,:]))


ref = qmc_mean[-1]
qmc_err = []
mc_err = []
for i in range(m-1):
    qmc_err.append(abs(ref-qmc_mean[i]))
    mc_err.append(abs(ref-mc_mean[i]))
print(qmc_mean)
print(mc_mean)
print(qmc_err)
print(mc_err)

plt.loglog(2**(np.linspace(0,m-1,m-1)), qmc_err)
plt.loglog(2**(np.linspace(0,m-1,m-1)), mc_err)
plt.show()
