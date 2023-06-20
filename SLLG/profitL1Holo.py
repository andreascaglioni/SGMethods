import numpy as np


def PorfitL1Holo(nu, p):
    if(len(nu.shape) == 1):
        nu= np.reshape(nu, (1,-1))
    nMids = nu.shape[0]
    nDims = nu.shape[1]
    rho = 2**(1.5*np.ceil(np.log2(np.linspace(1,nDims,nDims)))) 
    repRho = np.repeat(np.reshape(rho, (1, -1)), nMids, axis=0)
    C1 = 1
    C2 = 1
    v1 = np.prod(C1* np.power(repRho, -1), axis=1, where=(nu==1))
    v2 = np.prod(C2*np.power(2**nu * repRho, -p), axis=1, where=(nu>1))
    w = np.prod((2**(nu+1)-2)*(p-1)+1, axis=1)
    return v1 * v2 / w