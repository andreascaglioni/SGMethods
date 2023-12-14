import numpy as np

def ProfitHolderHolo(nu,p):
    """The basic profits. Definition based on holomorphy proof of param to sol map assuming wiener process and magnetization to be Holder.
    This will lead to a suboptimal dimension dependent ocnvergence"""
    if(len(nu.shape) == 1):
        nu= np.reshape(nu, (1,-1))
    nMids = nu.shape[0]
    nDims = nu.shape[1]
    rho = 2**(0.5*np.ceil(np.log2(np.linspace(1,nDims,nDims)))) 
    repRho = np.repeat(np.reshape(rho, (1, -1)), nMids, axis=0)
    C1 = 1
    C2 = 1
    v1 = np.prod(C1* np.power(repRho, -1), axis=1, where=(nu==1))
    v2 = np.prod(C2* np.power(2**nu * repRho, -p), axis=1, where=(nu>1))
    w = np.prod((2**(nu+1)-2)*(p-1)+1, axis=1)
    return v1 * v2 / w

def PorfitL1HoloOld(nu, p):
    """If you assume the parameter to solution map is holomoprhic with L1 as sapce for the Wiener process, you get this profit definition"""
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

def ProfitMix(nu, p):  
    """update"""
    # convert ints into 1D 1 entry arrays
    if(isinstance(nu, int)):
        nu = np.array([nu], dtype=int)
    if(len(nu.shape) == 1):
        nu= np.reshape(nu, (1,-1))
    nMids = nu.shape[0]
    nDims = nu.shape[1]
    rhoFast = 2**(1.5*np.ceil(np.log2(np.linspace(1,nDims,nDims)))) 
    repRhoFast = np.repeat(np.reshape(rhoFast, (1, -1)), nMids, axis=0)
    rho = 2**(0.5*np.ceil(np.log2(np.linspace(1,nDims,nDims)))) 
    repRho = np.repeat(np.reshape(rho, (1, -1)), nMids, axis=0)
    C1 = 1
    C2 = 1
    v1 = np.prod(C1* np.power(repRhoFast, -1), axis=1, where=(nu==1))  # from explicit estimate on first derivative
    v2 = np.prod(C2*np.power(2**nu * repRho, -p), axis=1, where=(nu>1))  # from holomorphy
    w = np.prod((2**(nu+1)-2)*(p-1)+1, axis=1)
    return v1 * v2 / w


def ProfitMixL1holo(nu, p):  
    """this complies with stuff proved in paper. In particular we assume holomprhy with W in L1. 
    when nu_i>1 I assume slow growth domain of holomorphy (I can increase polynomial interpolation degree p-1 instead)
    when nu_i=1, I assume faster growth unless many indices are 1 in the current LC level
    if use p>=3, i.e. at least piecewise quadratic interpolation, I get THEORETICAL dim-independent convergence."""
    # convert ints into 1D 1 entry arrays
    if(isinstance(nu, int)):
        nu = np.array([nu], dtype=int)
    if(len(nu.shape) == 1):
        nu= np.reshape(nu, (1,-1))
    nMids = nu.shape[0]
    nDims = nu.shape[1]
    
    # value corresponding to components nu_i=1
    C1 = 1
    idMid, pp = np.where(nu==1)
    ll = np.zeros_like(pp, dtype=int)  # array of corresponding levels to which the index belongs
    ll = np.floor(np.log2(pp)).astype(int)+1
    ll[np.where(pp==0)] = 0
    v1 = np.ones(nMids)
    # TODO get rid of loop
    for i in range(nMids):
        llUniqueAppearing, contLLAppearing = np.unique(ll[np.where(idMid==i)], return_counts=True)
        tmp = np.power(np.power(2, -3/2 * llUniqueAppearing) * contLLAppearing, contLLAppearing)
        v1[i] = C1 * np.prod(tmp)
    
    # value for nu_i>1    
    rho = 2**(0.5*np.ceil(np.log2(np.linspace(1,nDims,nDims)))) 
    repRho = np.repeat(np.reshape(rho, (1, -1)), nMids, axis=0)
    C2 = 1
    v2 = np.prod(C2*np.power(2**nu * repRho, -p), axis=1, where=(nu>1))  # from holomorphy
    w = np.prod((2**(nu+1)-2)*(p-1)+1, axis=1)
    return v1 * v2 / w
