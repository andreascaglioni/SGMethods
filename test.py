from cProfile import label
import numpy as np
import matplotlib.pyplot as plt

Nmax=50
NN = np.linspace(1, Nmax, Nmax)
alpha=2
tau = 0.9
r = 1/(1-2**(-(alpha+1)*tau+1))
Ctau1 = (r**NN * (alpha-1)**(NN*(1-tau)))**(1/tau)
print(Ctau1[1:5])

r2 = r-1 # 1/(2**((alpha+1)*tau-1))
Ctau2 = (2**(r2/(1-2**(-alpha*0.5*tau))) * NN**(np.log2(NN)/2)*NN**0.5 * (alpha-1)**(NN*(1-tau)))**(1/tau)
print(Ctau2[1:5])

plt.loglog(NN, Ctau1, label="naive")
plt.loglog(NN, Ctau2, label="dim")
plt.legend()
plt.show()


### look ad tependence convergence on tau
tt = np.linspace(1/(1+alpha), 1, 5)
for tau in tt:
    Ctau2 = (2**(r2/(1-2**(-alpha*0.5*tau))) * NN**(np.log2(NN)/2)*NN**0.5 * (alpha-1)**(NN*(1-tau)))**(1/tau)
    plt.plot(NN, Ctau2, label=tau)
plt.show()