import numpy as np
from numpy import inf
from sgmethods.utils import coordUnitVector, findMid, find_idx_in_margin, lexicSort, midIsInReducedMargin


class MidSet():
    """Multi-index set that can be grown by adding mids in the margin.
       Can be used for adaptivity, where one does not know a-priori the next 
       best multi-index set or for a priori profit-based approximation if the 
       definition of the profit is not monotonic.
       Start from {0}.

       Adding multi-indices to the margin using update() does not grow the 
       dimenisonality.
       Dimension-adaptivity is allowed with method increase_dim(), which adds
       to the multi-index set the first coordinate unit vector outside of the
       support of the multi-index set.
       If the maximum dimension maxN is reached, then the dimenison-adaptivity
       is not considered.
    """

    def __init__(self, maxN=inf, trackReducedMargin=False):
        """Initialize basic variables; then build default midset {0}  with its 
        margin, and reduced margin (if tracked).

        Args:
            maxN (int, optional): Maximum number of dimensions. Defaults to inf.
            trackReducedMargin (bool, optional): whether or not to keep track of
                reduced margin (costs more). Defaults to False.
        """
        self.maxN = maxN  # maximum number of allowed dimensions
        # self.N = 0  # dimensions of the multi-index set TODO delete?
        
        self.midSet = np.zeros((1, 1), dtype=int)
        self.margin = np.identity(1, dtype=int)
        self.trackReducedMargin = trackReducedMargin
        self.reducedMargin = np.identity(1, dtype=int)
        # self.numMids = self.midSet.shape[0]  # TODO remove and use getCardinality instead
        # NB always have midset.shape[1] = self.margin.shape[1] = min(N+1, maxN)
        # self.dimMargin = min(self.N+1, maxN)

    def get_dim(self):
        """Returns the dimensions fo the multi-index set.

        Returns:
            int: Dimension (or support size) of the multi index set
        """

        if(self.midSet.size == 1):
            return 0
        else:
            return self.midSet.shape[1]
        
    # TODO not used, remove?
    # def getMidSet(self):
    #     """Get the midset as an array without the buffer dimension, if there.
    #         Otherwose simply return the midset.
        
    #     Returns:
    #         2D array int: multi-index set
    #     """

    #     if (self.midSet.shape == (1, 1)):  # the case of trivial midset [0]
    #         return self.midSet
    #     if (np.linalg.norm(self.midSet[:, -1], ord=1) < 1.e-4):
    #         return self.midSet[:, :-1:]
    #     return self.midSet
    
    # TODO not used, remove?
    # def getN(self):
    #     """Get the number of dimensions in the multi-index set. The trivial 
    #     multi-index set {0} has N=0.

    #     Returns:
    #         int: number of dimensions
    #     """

    #     return self.N
    
    def get_cardinality_midSet(self):
        """Returns:
            int: Number of multi-indices in the mutli-index set
        """
        return self.midSet.shape[0]
    
    def get_cardinality_margin(self):
        """Returns:
            int: Number of multi-indices in the margin
        """
        return self.margin.shape[0]
    
    def get_cardinality_reduced_margin(self):
        """Returns:
            int: Number of multi-indices in the reducedmargin
        """
        assert self.trackReducedMargin, "Reduced margin not tracked"
        return self.reducedMargin.shape[0]

    def enlarge_from_margin(self, idx_margin):
        """ Add a multi-index in the margin to the multi-index set.
         A recursive function keeps the multi-index set downward-closed. 
         This function does not increase the dimensionality of the multi-index.
         See increase_dim() for that.

        Args:
            idx_margin (int): Index inside margin of the multi-index to be added
            to the multi-index set.
        """

        assert(idx_margin >= 0 and idx_margin < self.margin.shape[0])

        mid = self.margin[idx_margin, :]

        # Check if mid is in reduced margin. If not, recursively add mids from
        # backward margin. Check one dim at a time.
        for n in range(self.get_dim()):
            en = coordUnitVector(self.get_dim(), n)
            checkMember, _ = findMid(mid-en, self.midSet)
            if (not (mid[n] == 0 or checkMember)): # check admissibility 
                mid_tmp = mid - en
                idx_mid_tmp = find_idx_in_margin(self.margin, mid_tmp)
                self.enlarge_from_margin(idx_mid_tmp)  # Recursive call!

        # Because of the previous loop, we can assume the initial multi-index 
        # mid to be in the reduced margin of the multi-index set midSet.

        # Update midSet
        self.midSet = np.row_stack((self.midSet, mid))
        self.midSet = lexicSort(self.midSet)
        
        # Update margin (recall that it is also of dimensions N)
        self.margin = np.delete(self.margin, idx_margin,  0)  # remove added mid
        # Add to margin the forward margin of mid
        self.margin = np.row_stack(
            (self.margin, mid + np.identity(self.get_dim(), dtype=int))
            )
        # mids just added may already have been in midset
        self.margin = np.unique(self.margin, axis=0)  # keep unique elements
        self.margin = lexicSort(self.margin)

        # Update reduced margin
        if (self.trackReducedMargin):
            # Delete the multi-index mid we just added to the multi-index set.
            # Because of the recursion, we know it was in the reduced margin.
            checkMember, idx_RM = findMid(mid, self.reducedMargin)
            assert(checkMember, "mid not in reduced margin, error and exit")
            self.reducedMargin = np.delete(self.reducedMargin, idx_RM,  0)
            
            # add reduced margin elements among forward margin of mid
            for n in range(self.get_dim()):
                fwMid = mid + coordUnitVector(self.get_dim(), n)
                if midIsInReducedMargin(fwMid, self.midSet):  
                    self.reducedMargin = np.vstack(
                        (self.reducedMargin, np.reshape(fwMid, (1, -1)))
                        )
            self.reducedMargin = np.unique(self.reducedMargin, axis=0)
            self.reducedMargin = lexicSort(self.reducedMargin)

    def increase_dim(self):
        """Increase the dimensionality of the multi-index set by adding the 
        first coordinate unit vector outside of the support of the multi-index 
        set. If the maximum dimension maxN is reached, then the 
        dimenison-adaptivity is not considered.
        """
        # Stop if reached max dimension
        if (self.get_dim() >= self.maxN):  
            print("The multi-index set reached the  maximum dimension maxN="\
                + self.maxN + ", cannot increase it further")
            return
        
        # Update multi-index set:
        # First define the multi-index to add...
        e_newDim = coordUnitVector(self.get_dim()+1, self.get_dim())
        # ... then increase dimensionality existing midSet...
        self.midSet = np.column_stack(
            (self.midSet, \
             np.zeros((self.get_cardinality_midSet(), 1), dtype=int))
            )
        # ... and finally merge and sort
        self.midSet = np.row_stack((self.midSet, e_newDim))
        self.midSet = lexicSort(self.midSet)

        # Update margin:
        # First add new dimension to pre-existitng margin...
        self.margin = np.column_stack(
            (self.margin, \
             np.zeros((self.get_cardinality_margin(), 1), dtype=int))
            )  
        # ...then add margin in new dimension (copy midSet (recall we updated 
        # it above), remove the 0, and transalte up by 1)...
        margin_new_dim = np.copy(self.midSet)
        margin_new_dim = margin_new_dim[1:, :]
        margin_new_dim[:, -1] += 1
        # ... and finally merge and sort
        self.margin = np.row_stack((self.margin, margin_new_dim))  # merge
        self.margin = lexicSort(self.margin)   

        # Update reduced margin, if tracked
        if self.trackReducedMargin:
            # Increase dimensionality existing reduced margin...
            self.reducedMargin = np.column_stack(
                (self.reducedMargin, 
                np.zeros((self.get_cardinality_reduced_margin(), 1), dtype=int))
                )
            # ... define forward margin of e_{N+1} (NB updated midSet already)
            RM_new_dim = e_newDim + \
                np.identity(self.get_dim(), dtype=int)
            # ... and finally merge and sort
            self.reducedMargin = np.row_stack((self.reducedMargin, RM_new_dim))
            self.reducedMargin = lexicSort(self.reducedMargin)