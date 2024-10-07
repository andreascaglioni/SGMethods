""" Module with a class for adaptive multi-index sets."""

import numpy as np
from numpy import inf
from sgmethods.utils import coordUnitVector, findMid, find_idx_in_margin, \
    lexicSort, midIsInReducedMargin


class MidSet():
    """Multi-index set that can be grown by adding mids in the margin.
    Can be used for adaptivity, where one does not know a-priori the next best 
    multi-index set, or for a priori profit-based approximation when the 
    definition of the profit is not monotonic. 
    
    The multi-index set is initialized to :math:`\{0\}`.

    We can add multi-indices using either 
    :py:meth:`enlarge_from_margin`, which adds a multi-index from the margin and
    does not increasing the dimensionality, or :py:meth:`increase_dim`, 
    which adds the first coordinate unit vector outside of the support of the 
    multi-index set. If the maximum dimension :py:attr:`maxN` is reached, then 
    the dimenison-adaptivity is ignored.
    """

    def __init__(self, max_n=inf, track_reduced_margin=False):
        """Initialize basic variables; then build default midset 
        :math:`\{0\}`  with its margin, and reduced margin (if 
        tracked).

        Args:
            maxN (int, optional): Maximum number of dimensions. Defaults to inf.
            trackReducedMargin (bool, optional): whether or not to keep track of
                reduced margin (costs more). Defaults to False.
        
        Returns:
            None
        """

        self.max_n = max_n  # maximum number of allowed dimensions
        self.mid_set = np.zeros((1, 1), dtype=int)
        self.margin = np.identity(1, dtype=int)
        self.track_reduced_margin = track_reduced_margin
        self.reduced_margin = np.identity(1, dtype=int)

    def get_dim(self):
        """Returns the dimensions fo the multi-index set.

        Returns:
            int: Dimension (or support size) of the multi index set
        """
        if self.mid_set.size == 1:
            return 0
        return self.mid_set.shape[1]

    def get_cardinality_mid_set(self):
        """Returns the number of multi_indices in the multi-index set.
        
        Args:
            None

        Returns:
            int: Number of multi-indices in the mutli-index set
        """
        return self.mid_set.shape[0]

    def get_cardinality_margin(self):
        """Returns the number of multi_indices in the margin of the multi-index
        set.

        Args:
            None

        Returns:
            int: Number of multi-indices in the margin
        """
        return self.margin.shape[0]

    def get_cardinality_reduced_margin(self):
        """ Returns the number of multi_indices in the reduced margin of the 
        multi-index set.

        Args:
            None

        Returns:
            int: Number of multi-indices in the reduced margin
        """
        assert self.track_reduced_margin, "Reduced margin not tracked"
        return self.reduced_margin.shape[0]

    def enlarge_from_margin(self, idx_margin):
        """ Add the margin multi-index in position ``idx_margin`` to the 
        multi-index set. 
        A recursive function keeps the multi-index set downward-closed. 
        This function does not increase the dimensionality of the multi-index 
        (see :py:meth:`increase_dim` for that).

        Args:
            idx_margin (int): Index inside margin of the multi-index to be added
                to the multi-index set.
        
        Returns: 
            None
        """

        assert 0 <= idx_margin < self.margin.shape[0], "idx_margin out of bound"

        mid = self.margin[idx_margin, :]

        # Check if mid is in reduced margin. If not, recursively add mids from
        # backward margin. Check one dim at a time.
        for n in range(self.get_dim()):
            en = coordUnitVector(self.get_dim(), n)
            check_member, _ = findMid(mid-en, self.mid_set)
            if not (mid[n] == 0 or check_member): # check admissibility
                mid_tmp = mid - en
                idx_mid_tmp = find_idx_in_margin(self.margin, mid_tmp)
                self.enlarge_from_margin(idx_mid_tmp)  # Recursive call!

        # Because of the previous loop, we can assume the initial multi-index
        # mid to be in the reduced margin of the multi-index set midSet.

        # Update midSet
        self.mid_set = np.row_stack((self.mid_set, mid))
        self.mid_set = lexicSort(self.mid_set)

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
        if self.track_reduced_margin:
            # Delete the multi-index mid we just added to the multi-index set.
            # Because of the recursion, we know it was in the reduced margin.
            check_member, idx_rm = findMid(mid, self.reduced_margin)
            assert check_member, "mid not in reduced margin, error and exit"
            self.reduced_margin = np.delete(self.reduced_margin, idx_rm,  0)

            # add reduced margin elements among forward margin of mid
            for n in range(self.get_dim()):
                fw_mid = mid + coordUnitVector(self.get_dim(), n)
                if midIsInReducedMargin(fw_mid, self.mid_set):
                    self.reduced_margin = np.vstack(
                        (self.reduced_margin, np.reshape(fw_mid, (1, -1)))
                        )
            self.reduced_margin = np.unique(self.reduced_margin, axis=0)
            self.reduced_margin = lexicSort(self.reduced_margin)

    def increase_dim(self):
        """Increase the dimensionality of the multi-index set by adding the 
        first coordinate unit vector outside of the support of the multi-index 
        set. If the maximum dimension maxN is reached, then the 
        dimenison-adaptivity is not considered.

        Args:
            None

        Returns:
            None
        """
        # Stop if reached max dimension
        if self.get_dim() >= self.max_n:
            print("The multi-index set reached the maximum dimension maxN="\
                + self.max_n + ", cannot increase it further")
            return

        # Update multi-index set:
        # First define the multi-index to add...
        e_new_dim = coordUnitVector(self.get_dim()+1, self.get_dim())
        # ... then increase dimensionality existing midSet...
        self.mid_set = np.column_stack(
            (self.mid_set, \
             np.zeros((self.get_cardinality_mid_set(), 1), dtype=int))
            )
        # ... and finally merge and sort
        self.mid_set = np.row_stack((self.mid_set, e_new_dim))
        self.mid_set = lexicSort(self.mid_set)

        # Update margin:
        # First add new dimension to pre-existitng margin...
        self.margin = np.column_stack(
            (self.margin, \
             np.zeros((self.get_cardinality_margin(), 1), dtype=int))
            )
        # ...then add margin in new dimension (copy midSet (recall we updated
        # it above), remove the 0, and transalte up by 1)...
        margin_new_dim = np.copy(self.mid_set)
        margin_new_dim = margin_new_dim[1:, :]
        margin_new_dim[:, -1] += 1
        # ... and finally merge and sort
        self.margin = np.row_stack((self.margin, margin_new_dim))  # merge
        self.margin = lexicSort(self.margin)

        # Update reduced margin, if tracked
        if self.track_reduced_margin:
            # Increase dimensionality existing reduced margin...
            self.reduced_margin = np.column_stack(
                (self.reduced_margin,
                np.zeros((self.get_cardinality_reduced_margin(), 1), dtype=int))
                )
            # ... define forward margin of e_{N+1} (NB updated midSet already)
            rm_new_dim = e_new_dim + \
                np.identity(self.get_dim(), dtype=int)
            # ... and finally merge and sort
            self.reduced_margin = np.row_stack(
                (self.reduced_margin, rm_new_dim)
                )
            self.reduced_margin = lexicSort(self.reduced_margin)
