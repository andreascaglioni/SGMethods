from SGMethods.MidSets import midSet, count_cps
import matplotlib.pyplot as plt
import numpy as np


######################################## test multi-index set for piecewise-linear SG 


beta=0.5
rhoFun = lambda N : 2**(beta*np.ceil(np.log2(np.linspace(1,N,N))))
ProfitFun = lambda x : 2*np.sum(x) +  np.sum(np.log2(rhoFun(x.size)) , where=(x>0))  ## actually it is 2**(-P) and I want the n mids with largest profit. this is equivalent ot pickigng the n mids with SMALLEST P

I = midSet()
print(I.midSet)
print("--")
print(I.margin)

cps = []
ndims = []
nIters = 65
for n in range(nIters):
    # print("-------------------------")
    P = np.apply_along_axis(ProfitFun, 1, I.margin)
    # print("log profit: ", P)
    idMax = np.argmin(P)
    # print("idmax:", idMax)
    I.update(idMax)
    # print(I.midSet)
    # print("--")
    # print(I.margin)
    cps.append(count_cps(I.midSet, lambda nu: 2**(np.sum(nu))))
    ndims.append(I.N)
    # print("iter", n, ": dim", I.N, "nuber cps:", count_cps(I.midSet, lambda nu: 2**(np.sum(nu))))
    print(np.amax(I.midSet[:,0]))

# plt.plot(np.linspace(1,nIters,nIters), cps, '.-')
# plt.show()
# plt.plot(np.linspace(1,nIters,nIters), ndims, '.-')
# plt.show()
plt.plot(cps, ndims, '.-')
plt.show()