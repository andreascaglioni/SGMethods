import numpy as np
from scipy.interpolate import RegularGridInterpolator
from src.nodes_tp import TPKnots
from src.tp_interpolant_wrapper import TPInterpolatorWrapper
from multiprocessing import Pool


class SGInterpolant:
    """Sparse grid interpolant class. It stores all relevant information to 
        define it lke multi-index set, 1D notes etc. 
        It automatically computes the sparse grid and inclusio-exclusion 
        coefficients upon initialization.
        It allows to interpolate high dimensinal functions on the sparse grid.
    """

    def __init__(self, midSet, knots, lev2knots, \
                 TPInterpolant=RegularGridInterpolator, NParallel=1, \
                    verbose=True):
        """Initialize data, compute inclusion-exclusion coeff.s, sparse grid

        Args:
            midSet (2D array int): Multi-index set 
                NB downward closed!! 
            knots (function): given nin\mathbb{N}, computes n nodes
            lev2knots (function): given level nuin\mathbb{N}_0, computes 
                corresponding number of 1D nodes
            TPInterpolant (class, optional): Class to be used as tensor product
                interpolant. Defaults to piecewie linear. 
                The class must have a method __call__ that takes as input a np 
                array (parameters).
                Examples are (Piecewise linear (default), quadratic, cubic and 
                Lagrange (spectral polynomial) interpolation respectively):
                    RegularGridInterpolator(activeNodesTuple, self.fOnNodes, 
                        method='linear', bounds_error=False, fill_value=None)
                    TPPwQuadraticInterpolator(activeNodesTuple, self.fOnNodes)
                    TPPwCubicInterpolator(activeNodesTuple, self.fOnNodes)
                    TPLagrangeInterpolator(activeNodesTuple, self.fOnNodes)

            NParallel (int, optional): Number of parallel computations. Defaults 
                to 1.
            verbose (bool, optional): Verbose output. Defaults to True.
        """
        self.verbose = verbose
        self.midSet = midSet  # np array of shape (#mids, N)
        self.cardMidSet = midSet.shape[0]
        self.N = midSet.shape[1]
        # NBB need knots[1] = 0 to increase number of dimensions
        self.knots = knots
        # NBB need lev2knots(0)=1 to increase number of dimensions
        self.lev2knots = lev2knots
        self.TPInterpolant = TPInterpolant

        self.combinationCoeffs = []  # list of int
        self.activeMIds = []  # list of np arrays
        self.activeTPNodesList = []  # list of tuples
        # which dimensions of currcent TP interpolant (in inclusion-exclusion
        # formula) are active (more than 1 node)
        self.activeTPDims = []
        self.mapTPtoSG = []  # list of np arrays of shape ()
        self.setupInterpolant()

        self.SG = []  # np array of shape (#colloc. pts, N)
        self.numNodes = 0
        self.setupSG()

        self.NParallel = NParallel

    def setupInterpolant(self):
        """ Computes and saves in class attributes some important quantities:
        combinCoeff, activeMIds, activeTPDims, activeTPNodesList, mapTPtoSG 
        based on: midSet, knots, lev2knots
        """
        bookmarks = np.unique(self.midSet[:, 0], return_index=True)[1]
        bk = np.hstack((bookmarks[2:], np.array(
            [self.cardMidSet, self.cardMidSet])))
        for n in range(self.cardMidSet):
            currentMid = self.midSet[n, :]
            combinCoeff = 1
            # index of the 1st mid with 1st components = currentMid[0]+2 (dont
            # need to itertate over it or any following one)
            rangeIds = bk[currentMid[0]]
            # NB in np slice start:stop, stop is NOT included!!
            midsDifference = self.midSet[(n+1):(rangeIds), :] - currentMid
            isBinaryVector = np.all(np.logical_and(
                midsDifference >= 0, midsDifference <= 1), axis=1)
            binaryRows = midsDifference[isBinaryVector]
            compiELementary = np.power(-1, np.sum(binaryRows, axis=1))
            combinCoeff += np.sum(compiELementary)

            if combinCoeff != 0:
                self.combinationCoeffs.append(combinCoeff)
                self.activeMIds.append(currentMid)
                numNodesDir = self.lev2knots(currentMid).astype(int)
                activeTPDims = np.where(numNodesDir > 1)
                self.activeTPDims.append(activeTPDims[0])
                numNodesActiveDirs = numNodesDir[activeTPDims]
                # If no dimension is active, i.e. 1 cp, 1 need a length 1 array!
                if (numNodesActiveDirs.shape[0] == 0):
                    numNodesActiveDirs = np.array([1])
                self.activeTPNodesList.append(
                    TPKnots(self.knots, numNodesActiveDirs))
                shp = np.ndarray(tuple(numNodesActiveDirs), dtype=int)
                self.mapTPtoSG.append(shp)  # to be filled

    def setupSG(self):
        """Computes and saves in class attributes some important quantities:
        SG (sparse grid), numNodes, mapTPtoSG
        based on: self.activeTPNodesList
        """

        SG = np.array([]).reshape((0, self.N))
        for n, currActiveTPNodes in enumerate(self.activeTPNodesList):
            currActiveDims = self.activeTPDims[n]
            # NB "*" is "unpacking" operator (return comma-separated list)
            meshGrid = np.meshgrid(*currActiveTPNodes, indexing='ij')
            it = np.nditer(meshGrid[0], flags=['multi_index'])
            for x in it:
                currNodeActiveDims = [meshGrid[j][it.multi_index]
                                      for j in range(len(meshGrid))]
                # complete it with 0s in inactive dimensions
                currNode = np.zeros(self.N)
                currNode[currActiveDims] = currNodeActiveDims
                # check = np.where(~(SG-currNode).any(axis=1))[0]
                check = np.where(
                    np.sum(np.abs(SG-currNode), axis=1) < 1.e-10)[0]
                found = check.shape[0]
                # if currNode was in the SG more than once, something wrong
                assert (found <= 1)
                if found:  # if found, add the index to mapTPtoSG[n]
                    self.mapTPtoSG[n][it.multi_index] = check[0]
                else:  # if not found, add it to sparse grid, add to mapTPtoSG
                    SG = np.vstack((SG, np.array(currNode)))
                    self.mapTPtoSG[n][it.multi_index] = SG.shape[0]-1
        self.SG = SG
        self.numNodes = SG.shape[0]

    def sampleOnSG(self, Fun, dimF=None, oldXx=None, oldSamples=None):
        """Sample a given function on the sparse grid. Optionally recycle 
            previous samples. Also takes care automatically of the case in which
            in the meainwhile the sparse grid has increased dimension. First 
            check if there is anything to recycle, then sample new values
            NBB assume F takes as input an np array of parameters and the 1st 
            output are the relevant values

        Args:
            Fun (function): given array of parameters (double), return array in
                in codomain
            dimF (int, optional): dimension codomain F. Defaults to None
            oldXx (2D array double, optional): each row is a parameter vector. 
                Defaults to None.
            oldSamples (array double, optional): Each row is a sample value in 
                corresponding parameter vector in oldXx. 
                Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            2D array fo double: Values of Fun on the sparse grid 9each row is a 
            value correspondin to a paramter vector)
        """

        dimF = -1  # TODO remove dimF from signature; check all other scripts

        # Find dimF (dimension of output of F). First try oldSamples; if empty,
        # sample on first SG node and add to oldSamples and oldXx
        assert (self.SG.shape[0] > 0)
        if (not (oldSamples is None)):
            oldSamples = np.atleast_1d(oldSamples)
            dimF = oldSamples.shape[1]
            assert (dimF >= 1)
        else:  # Sample on first SG node
            node0 = self.SG[0]
            fOnSG0 = np.atleast_1d(Fun(node0))  # turn in array if scalar
            dimF = fOnSG0.size

        fOnSG = np.zeros((self.numNodes, dimF))  # the return array
        # TODO change, if previou else needed, one sample is forgotten

        # Sanity checks dimensions oldXx, oldSamples
        if ((oldXx is None) or (oldSamples is None)):
            oldXx = np.zeros((0, self.N))
            oldSamples = np.zeros((0, dimF))
        assert (oldXx.shape[0] == oldSamples.shape[0])
        assert (oldSamples.shape[1] == dimF)

        # Case 1: Cuurrent parameter space has larger dimension that oldXx.
        # Embed oldXx in space of self.SG by extending by 0
        if (oldXx.shape[1] < self.SG.shape[1]):
            filler = np.zeros((oldXx.shape[0], self.SG.shape[1]-oldXx.shape[1]))
            oldXx = np.hstack((oldXx, filler))

        # Case 2: Current parameter space is smaller than that of oldXx.
        # Remove from oldXx parameters out of subspace, also adapt oldSamples
        elif (oldXx.shape[1] > self.SG.shape[1]):
            currDim = self.N
            tailNorm = np.linalg.norm(
                oldXx[:, currDim::], ord=1, axis=1).astype(int)
            # last [0] so the result is np array
            validEntries = np.where(tailNorm == 0)[0]
            oldXx = oldXx[validEntries, 0:self.N]
            oldSamples = oldSamples[validEntries]

        # Find parametric points to sample now
        nRecycle = 0
        toCompute = []
        yyToCompute = []
        for n in range(self.numNodes):
            currNode = self.SG[n, :]
            check = np.where(np.linalg.norm(
                oldXx-currNode, 1, axis=1) < 1.e-10)[0]
            if (check.size > 1):
                print(check)
            assert (check.size <= 1)
            found = len(check)
            if found:
                fOnSG[n, :] = oldSamples[check[0], :]
                nRecycle += 1
            else:
                toCompute.append(n)
                yyToCompute.append(currNode)

        # Sample (possibily in parallel) on points found just above
        if (len(toCompute) > 0):
            if (self.NParallel == 1):
                for i in range(len(toCompute)):
                    fOnSG[toCompute[i], :] = Fun(yyToCompute[i])
            elif (self.NParallel > 1):
                pool = Pool(self.NParallel)
                tmp = np.array(pool.map(Fun, yyToCompute))
                if (len(tmp.shape) == 1):
                    tmp = tmp.reshape((-1, 1))
                fOnSG[toCompute, :] = tmp
            else:
                raise ValueError('self.NParallel not int >= 1"')
        if self.verbose:
            print("Recycled", nRecycle, \
                  "; Discarted", oldXx.shape[0]-nRecycle, \
                    "; Sampled", self.SG.shape[0]-nRecycle)
        return fOnSG

    def interpolate(self, xNew, fOnSG):
        """Interpolate the function values on new points

        Args:
            xNew (array double): THe new parametric points where to interpolate
            fOnSG (array double): Values of function on the sparse grid

        Returns:
            arrray double: Values of the interpolated function on xNew
        """
        out = np.zeros((xNew.shape[0], fOnSG.shape[1]))
        for n, MId in enumerate(self.activeMIds):
            currentActiveNodesTuple = self.activeTPNodesList[n]
            currentActiveDims = self.activeTPDims[n]
            mapCurrTPtoSG = self.mapTPtoSG[n]
            # output is a matrix of shape = shape(mapCurrTPtoSG) + (dimF,)
            fOnCurrentTPGrid = fOnSG[mapCurrTPtoSG, :]
            L = TPInterpolatorWrapper(currentActiveNodesTuple, \
                                      currentActiveDims, \
                                      fOnCurrentTPGrid, self.TPInterpolant)
            out = out + self.combinationCoeffs[n] * L(xNew)
        return np.squeeze(out)  # remove dimensions of length 1
