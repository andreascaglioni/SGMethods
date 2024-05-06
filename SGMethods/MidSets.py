import numpy as np
from math import floor, log, inf
from utils.utils import coordUnitVector as unitVector
from utils.utils import lexicSort as lexicSort


"""A collection of functions to generate multi-index sets"""


def TPMidSet(w, N):
    """Generate a tensor product multi-index set such that
        max_i=0^{N-1} nu_i leq w
        NB leq so for w=0 only multi-index 0 included
    
    Args:
        w (int): maximum l^infty norm of multi-index 
        N (int): number of dimensions
    
    Returns:
        np.array: multi-index set (each row is a multi-index)
    """
    
    assert(N>=1)
    assert(w>=0)
    if w == 0:
        return np.zeros([1, N], dtype=int)
    elif N == 1:
        v = np.linspace(0, w, w+1, dtype=int)
        v = np.reshape(v, (1,)+v.shape)
        return v.T
    else: # w>0 & N >1
        I = np.zeros((0, N), dtype=int)
        for comp1 in range(w+1):
            rest = TPMidSet(w, N-1)
            newFirstCol = comp1*np.ones((rest.shape[0], 1), dtype=int)
            newRows = np.concatenate((newFirstCol, rest), 1)
            I = np.concatenate((I, newRows), 0)
        return I

def anisoSmolyakMidSet_freeN(w,a):
    """Generate the anisotropic Smolyak multi-index set. The dimension (N) is determined by value of a
        sum_i=0^{N-1} a_i*nu_i leq w
        NB leq so for w=0 only multi-index 0 included
        
    Args:
        w (double or int): maximum l^infty norm of multi-index 
        a (function): a(N) gives anisotropy vector of length N
    
    Returns:
        np.array: multi-index set (each row is a multi-index)
    """

    # find the maximum N for which a \cdot e_N \leq w
    N = 1
    while(a(N+1)[-1] <= w):
        N+=1
    return anisoSmolyakMidSet(w, N, a(N))

def anisoSmolyakMidSet(w, N, a):
    """Generate the anisotropic Smolyak multi-index set.
        sum_i=0^{N-1} a_i*nu_i leq w
        NB leq so for w=0 only multi-index 0 included
        
    Args:
        w (double or int): maximum l^infty norm of multi-index 
        N (int): number of dimensions
        a (array double): anisotropy vector (of length N)
    
    Returns:
        np.array: multi-index set (each row is a multi-index)
    """
    
    assert N >= 1
    assert(np.all(a>0))
    assert(a.shape[0] == N)
    if w == 0:
        return np.zeros([1, N], dtype=int)
    elif N == 1:
        v = np.linspace(0, floor(w/a[0]), floor(w/a[0])+1, dtype=int)
        v = np.reshape(v, (1,)+v.shape)
        return v.T
    else: # w>0 & N >1
        I = np.zeros((0, N), dtype=int)
        for comp1 in range(floor(w/a[0]) + 1): # 0...floor(w/a[0])
            rest = anisoSmolyakMidSet(w-comp1*a[0], N-1, a[1::])
            newFirstCol = comp1*np.ones((rest.shape[0], 1), dtype=int)
            newRows = np.concatenate((newFirstCol, rest), 1)
            I = np.concatenate((I, newRows), 0)
        return I

def SmolyakMidSet(w, N):
    """Generate the (isotropic) Smolyak multi-index set.
        sum_i=0^{N-1} nu_i leq w
        NB leq so for w=0 only multi-index 0 included
        
    Args:
        w (double or int): maximum l^infty norm of multi-index 
        N (int): number of dimensions
    
    Returns:
        np.array: multi-index set (each row is a multi-index)
    """
    # """ Each row is a multi-index define with base knot 0, condition: \sum_i=0^{d-1} x_i \leq w
    # NBB leq so w=0 onyl mid 0 included """
    assert N >= 1
    a = np.repeat(1., N)
    return anisoSmolyakMidSet(w, N, a)

# def LLG_optimal_midset_freeN(T, rhoFun):
#     """Determine effective dimension, fix rho to vector rather than function"""
#     maxLev = max(0, 2*(T-2))
#     N = int(2**maxLev)
#     rho = rhoFun(N)
#     assert(np.all(rho>0))
#     assert(rho.size == N)
#     return LLG_optimal_midset_recursive(N, T, rho)

# def LLG_optimal_midset_recursive(N, T, rho):
#     """INPUT N int number parametric dimensions
#              T int threshold parameter midset
#              rhoFun function; rhoFun(N) outputs a length N increasing list of positive doubles of 
#        OUTPUT \Lambda_T np array shape (# SG nodes, N) for kanpsack-type optimal multi-index set i.e.
#        Lambda_T = \left\{ \nu\in N^N : 2\vert\nu\vert_1 + \sum_{\nu_i>0} \log_2(\rho_i) + N \log_2(deg-1) \leq T}
#        NBB term  + N \log_2(deg-1) omitted because independent of \nu """
#     if T <= 0 :
#         return np.zeros((1, N), dtype=int)
#     elif N == 1:
#         b = max(0, floor(0.5*(T-log(rho[0], 2))) )
#         I = np.linspace(0, b, b+1)
#         I = np.reshape(I, (1,)+I.shape)
#         return I.T.astype(int)
#     else: # T>0 and N>1. recursion by elimination of first coordinate
#         I = np.zeros((0, N), dtype=int)
#         LIM = max(0, floor(0.5*(T-log(rho[0], 2))))
#         # case nu_1 = 0
#         nu1=0
#         rest = LLG_optimal_midset_recursive(N-1, T, rho[1::])
#         newFirstCol = nu1*np.ones((rest.shape[0], 1), dtype=int)
#         newRows = np.concatenate((newFirstCol, rest), 1)
#         I = np.concatenate((I, newRows), 0)
#         # case nu_1>0
#         for nu1 in range(1, LIM+1):  # nbb last int in range() is NOT included!!
#             rest = LLG_optimal_midset_recursive(N-1, T - 2*nu1 - log(rho[0], 2), rho[1::])
#             newFirstCol = nu1*np.ones((rest.shape[0], 1), dtype=int)
#             newRows = np.concatenate((newFirstCol, rest), 1)
#             I = np.concatenate((I, newRows), 0)
#         return I.astype(int)




def find_idx_in_margin(margin, mid):
    idx = np.all(margin == mid, axis=1)
    idx = np.where(idx == True)
    idx = idx[0][0]
    return idx

def midIsInReducedMargin(mid, mid_set):
    dimMids = mid_set.shape[1]
    assert(dimMids==mid.size)
    condition  = np.zeros(dimMids, dtype=bool)
    for n in range(dimMids):
        en = unitVector(dimMids, n)
        condition[n] = ( (mid[n]==0)  or (checkIfElement(mid-en, mid_set)))
    return np.all(condition)


def checkIfElement(mid, midSet):
    """_summary_

    Args:
        mid (_type_): _description_
        midSet (_type_): _description_

    Returns:
        _type_: _description_
    """
    assert(mid.size == midSet.shape[1])
    assert(len(mid.shape) == 1)
    assert(len(midSet.shape) == 2)
    return (midSet == mid).all(axis=1).any()

class midSet():
    """Multi-index set class. 
        Can be grown by adding mids from reduced margin
        Start from {0}
        New dimensions are also added as follows: I keep an empty dimenision as buffer, when a fully dimensional mid is added, increase the buffer by 1  
        If the maximum dimension maxN is reached, then there is no buffer anymore
    """    
    def __init__(self, maxN=inf, trackReducedMargin=False):
        """Initialize basic variables; then build default midset {0}  with its margin, reduced margin (if tracked)

        Args:
            maxN (int, optional): Maximum number of dimensions. Defaults to inf.
            trackReducedMargin (bool, optional): whether or not to keep track of reduced margin (costs more). Defaults to False.
        """
        self.maxN = maxN  # maximum number of allowed dimensions
        self.midSet = np.zeros((1,1), dtype=int)  # NB the number of columns is always the same as for the margin, not self.N! This is for better compatibility later
        self.N = 0  # start only with 0 in midset  
        self.numMids = self.midSet.shape[0]
        self.dimMargin = min(self.N+1, maxN) #  NBB always have midset.shape[1] = self.margin.shape[1] = min(N+1, maxN)
        self.margin = np.identity(self.dimMargin).astype(int)
        self.trackReducedMargin = trackReducedMargin
        self.reducedMargin = np.identity(self.dimMargin).astype(int)  # store mids in reduced margin

    def getMidSet(self):
        """Get the midset as an array without the buffer dimension, if it is there. Otherwose simply return the midset
        Returns:
            2D array int: multi-index set
        """

        if(self.midSet.shape == (1,1)):  # the case of trivial midset [0]
           return self.midSet
        if(np.linalg.norm(self.midSet[:,-1], ord=1) <1.e-4):
            return self.midSet[:, :-1:]
        return self.midSet

    def update(self, idx_margin):
        """ Add a multi-index in the margin to the multi-index set while keeping it downward-closed. 

        Args:
            idx_margin (int): Index in margin of the multi-index to be added
        """        

        midOriginal = self.margin[idx_margin,:]
        mid = self.margin[idx_margin,:]
        # check if mid is in reduced margin. If not, add first the margin element(s) that sits below it
        for n in range(self.dimMargin):
            en = unitVector(self.dimMargin, n)
            if(not(mid[n] == 0 or ((mid-en) in self.midSet))):  
                mid_tmp = mid - en  # in margin
                idx_mid_tmp = find_idx_in_margin(self.margin, mid_tmp)
                self.update(idx_mid_tmp)  # recursive call!
                if(mid.size < self.dimMargin):
                    mid = np.append(mid, 0) # NB dimension can have grown at most by 1
                    # TODO can it be that a mid in dimension dimMargin is NOT int the reduced margin? I guess no because any time the dimension is increased the *only* mid with highest dimension is the unit vector e_N
                idx_margin = find_idx_in_margin(self.margin, mid)
        
        #update midset
        self.midSet = np.row_stack((self.midSet, mid))
        self.midSet =self.midSet[np.lexsort(np.rot90(self.midSet))]
        self.numMids += 1
        
        #update margin within first dimMargin dimensions. If mid was added in buffer dimension, we add more mids later 
        self.margin = np.delete(self.margin, idx_margin,  0)  # remove added mid 
        self.margin = np.row_stack((self.margin, mid + np.identity(self.dimMargin, dtype=int)))
        self.margin = np.unique(self.margin, axis=0)  # mids just added may already have been in midset
        
        # increase N by 1 if mid is last coordinate unit vector 
        increaseN = (mid[-1] == 1 and np.all(mid[0:-1:] == np.zeros(self.dimMargin-1)))  # easier alternative (mid == unitVector(self.dimMargin, self.dimMargin))  #
        if increaseN:
            self.N+=1
            # if N < nMax, add (I,1) to margin. Otherwise midset has saturated all available dimensions
            if(self.N < self.maxN): # NBB remeber that self.N was just updated 2 lines above. Thus < not <=
                mid = np.append(mid, 0)  # needed in reduced margin computation
                marginNewDim = np.hstack((self.midSet, np.ones((self.midSet.shape[0], 1), dtype=int)))
                self.midSet = np.hstack((self.midSet, np.zeros((self.midSet.shape[0], 1), dtype=int)))
                self.margin = np.hstack((self.margin, np.zeros((self.margin.shape[0], 1), dtype=int)))
                self.margin = np.vstack((self.margin, marginNewDim))
                self.dimMargin+=1
        self.margin = self.margin[np.lexsort(np.rot90(self.margin))]
        
        # # update reduced margin 
        if(self.trackReducedMargin):
            # 1 delete element just added to margin. NB because of recursion, we can be sure it was in reduced margin
            idx_rm =np.where((self.reducedMargin==midOriginal).all(axis=1) )[0][0]
            self.reducedMargin = np.delete(self.reducedMargin, idx_rm,  0)

            # 2 update dimension reduced margin, if needed
            if (self.dimMargin > self.reducedMargin.shape[1]):
                dimDiff = self.dimMargin - self.reducedMargin.shape[1]
                self.reducedMargin = np.hstack( (self.reducedMargin, np.zeros((self.reducedMargin.shape[0], dimDiff), dtype=int)) )
                en = unitVector(self.dimMargin, self.dimMargin-1)
                self.reducedMargin = np.vstack((self.reducedMargin, en))  # if margin increased dimension, there is a new unit vector in it and it is also reduced
            
            # 3 add reduced margin elements among forward margin of mid
            for n in range(self.dimMargin):
                fwMid = mid + unitVector(self.dimMargin, n)
                if midIsInReducedMargin(fwMid, self.midSet):
                    self.reducedMargin = np.vstack((self.reducedMargin, np.reshape(fwMid, (1,-1))))
            self.reducedMargin = np.unique(self.reducedMargin, axis=0) 
            self.reducedMargin = self.reducedMargin[np.lexsort(np.rot90(self.reducedMargin))]

def computeMidSetFast(Profit, PMin):
    """Compute midset based on a profit function and a minimum profit threshold:
    \Lambda = \setc{\bnu\in\FF}{\PP_{\bnu} \geq PMin}
    NB Profit should be monotone (decreasing) wrt each dimension (to get downward-closed multi-index set!)
    NB This function just determines the max dimension possible, then call recursive function

    Args:
        Profit (function): give a multi-index, returns a profit (positive double)
        PMin (double): minimum profit threshold for multi-index to be included

    Returns:
        2D array int: multi-index set (each row is a multi-index)
    """    
    
    dMax = 1
    nu = unitVector(dMax+1,dMax)
    while(Profit(nu)>PMin):
        dMax+=1
        nu = unitVector(dMax+1,dMax)
    # at this point. dMax is the maximumdimension that allows the last coordinate unit vector to have profit > PMin
    return computeMidSetFastRecursive(Profit, PMin, dMax)

def computeMidSetFastRecursive(Profit, PMin, N):
    """ The 'workhorse' of computeMidSetFast.
    Computes recrursively over the N dimensions the multi-index set such that the profit is above the threshold PMin

    Args:
        Profit (function): give a multi-index, returns a profit (positive double)
        PMin (double): minimum profit threshold for multi-index to be included
        N (int): number of dimensions

    Returns:
        2D array int: multi-index set (each row is a multi-index of length N)
    """    
    
    assert(N>=0)
    if N == 0:
        return np.array([0], dtype=int)
    elif N == 1:  # this is the final case of the recursion
        nu = 0
        assert(Profit(nu) > 0)
        while(Profit(nu+1)>PMin):
            nu = nu+1
        # at this point, nu is the largest mid with P > Pmin
        return np.linspace(0, nu, nu+1, dtype=int).reshape((-1,1))
    else:  # N>1
        I = np.zeros((0, N), dtype=int)
        # check how large the LAST component can be
        nuNExtra = unitVector(N, N-1)  # has only a 1 in the last component
        while(Profit(nuNExtra)>PMin):
            nuNExtra[-1] += 1
        maxLastComp = nuNExtra[-1] - 1
        # recursively compute the other dimensions. Since I started with the LAST components, I can simply call again the function Profit that will look onyl at the first N-1 components
        for lastComponentNu in range(0, maxLastComp+1):
            nuCurr = np.zeros(N, dtype=int)
            nuCurr[-1] = lastComponentNu
            PRec = PMin / Profit(nuCurr)
            INewComps = computeMidSetFastRecursive(Profit, PRec, N-1)
            INewMids = np.hstack((INewComps, np.full((INewComps.shape[0], 1), lastComponentNu)))
            I = np.vstack((I, INewMids))
        return lexicSort(I)