import numpy as np
from scipy.interpolate import RegularGridInterpolator
from SGMethods.multi_index_sets import TPMidSet
from SGMethods.nodes_tp import TPKnots
from SGMethods.tp_interpolant_wrapper import TPInterpolatorWrapper
from multiprocessing import Pool


class SGInterpolant:
    """Sparse grid interpolant class. It stores all relevant information to define it lke multi-index set, 1D notes etc. 
        It automatically computes the sparse grid and inclusio-exclusion coefficients upon initialization.
        It allows to interpolate high dimensinal functions on the sparse grid.
    """    
    def __init__(self, midSet, knots, lev2knots, TPInterpolant = RegularGridInterpolator, NParallel = 1, verbose=True):
        """Initializa important data and compute inclusion-exclusion coefficients and sparse grid

        Args:
            midSet (2D array int): Multi-index set NB downward closed!! 
            knots (function): given n\in\mathbb{N}, computes n nodes
            lev2knots (function): given level nu\in\mathbb{N}_0, computes corresponding number of 1D nodes
            # interpolationType (str, optional): Type of interpolant. For options, see class "TPInterpolatorWrapper". 
            #    Default is "linear". To give your favourite TP interpolant as an argument select "given"
            TPInterpolant (class, optional): Class to be used as tensor product interpolant. Defaults to piecewie linear. 
                Important is that the class has a method __call__ that takes as input a np array (parameters).
                Examples are (Piecewise linear (default), quadratic, cubic and Lagrange (spectral polynomial) interpolation respectively):
                    RegularGridInterpolator(activeNodesTuple, self.fOnNodes, method='linear', bounds_error=False, fill_value=None)
                    TPPwQuadraticInterpolator(activeNodesTuple, self.fOnNodes)
                    TPPwCubicInterpolator(activeNodesTuple, self.fOnNodes)
                    TPLagrangeInterpolator(activeNodesTuple, self.fOnNodes)

            NParallel (int, optional): Number of parallel ocmputations. Defaults to 1.
            verbose (bool, optional): Verbose output. Defaults to True.
        """        
        self.verbose=verbose
        self.midSet = midSet  # np array of shape (#mids, N)
        self.cardMidSet = midSet.shape[0]
        self.N = midSet.shape[1]
        self.knots = knots #  NBB need knots[1] = 0 to increase number of dimensions
        self.lev2knots = lev2knots #  NBB need lev2knots(0)=1 to increase number of dimensions
        self.TPInterpolant = TPInterpolant
        
        self.combinationCoeffs = [] #  list of int
        self.activeMIds = [] #  list of np arrays
        self.activeTPNodesList = [] #  list of tuples
        self.activeTPDims = [] #  which dimensions of currcent TP interpolant (in inclusion-exclusion formula) are active (more than 1 node)
        self.mapTPtoSG = []  #  list of np arrays of shape ()
        self.setupInterpolant()

        self.SG = [] #  np array of shape (#colloc. pts, N)
        self.numNodes = 0
        self.setupSG()

        self.NParallel = NParallel

    def setupInterpolant(self):
        """ Computes and saves in class attributes some important quantities:
        combinCoeff, activeMIds, activeTPDims, activeTPNodesList, mapTPtoSG 
        based on: midSet, knots, lev2knots
        """
        bookmarks = np.unique(self.midSet[:,0], return_index=True)[1]
        bk = np.hstack((bookmarks[2:], np.array([self.cardMidSet, self.cardMidSet])))
        for n in range(self.cardMidSet):
            currentMid = self.midSet[n, :]
            combinCoeff = 1
            rangeIds = bk[currentMid[0]]  # index of the 1st mid with 1st components = currentMid[0]+2 (dont need to itertate over it or any following one)
            midsDifference = self.midSet[(n+1):(rangeIds), :] - currentMid  # NB in np slice start:stop, stop is NOT included!!
            isBinaryVector = np.all(np.logical_and(midsDifference>=0, midsDifference<=1), axis=1)
            binaryRows = midsDifference[isBinaryVector]
            compiELementary = np.power(-1, np.sum(binaryRows, axis=1))
            combinCoeff += np.sum(compiELementary)

            if combinCoeff != 0:
                self.combinationCoeffs.append(combinCoeff)
                self.activeMIds.append(currentMid)
                numNodesDir = self.lev2knots(currentMid).astype(int)
                activeTPDims = np.where(numNodesDir>1)
                self.activeTPDims.append(activeTPDims[0])
                numNodesActiveDirs = numNodesDir[activeTPDims]
                if(numNodesActiveDirs.shape[0] == 0):  # in case no dimension is active, i.e. 1 cp, 1 need a length 1 array!
                    numNodesActiveDirs = np.array([1])
                self.activeTPNodesList.append(TPKnots(self.knots, numNodesActiveDirs))
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
            meshGrid = np.meshgrid(*currActiveTPNodes, indexing='ij')  # NBB in python * is "unpacking" operator (gives back comma-separated list)
            it = np.nditer(meshGrid[0], flags=['multi_index'])
            for x in it:
                currNodeActiveDims = [meshGrid[j][it.multi_index] for j in range(len(meshGrid))]
                # complete it with 0s in inactive dimensions
                currNode = np.zeros(self.N)
                currNode[currActiveDims] = currNodeActiveDims
                # check = np.where(~(SG-currNode).any(axis=1))[0]
                check = np.where( np.sum(np.abs(SG-currNode), axis=1) < 1.e-10)[0]
                # check = np.where(np.linalg.norm(SG-currNode, 1, axis=1) < 1.e-10)[0] #  check if current node is already in SG (NBB SG is always fully dimensional, also if some components are 0.)
                found = check.shape[0]
                assert(found <= 1)  # if currNode was in the SG more than once, something would be very wrong
                if found:  # if found, add the index to mapTPtoSG[n]
                    self.mapTPtoSG[n][it.multi_index] = check[0]
                else:  # if not found, add it to sparse grid and add new index to mapTPtoSG
                    SG = np.vstack((SG, np.array(currNode)))
                    self.mapTPtoSG[n][it.multi_index] = SG.shape[0]-1
        self.SG = SG
        self.numNodes = SG.shape[0]

    def sampleOnSG(self, Fun, dimF = None, oldXx = None , oldSamples = None):
        """Sample a given function on the sparse grid. Optionally recycle previous samples. Also takes care automatically of the case in which in the meainwhile the sparse grid has increased dimension
            First check if there is anything to recycle, then sample new values
            NBB assume F takes as input an np array of parameters and the 1st output are the relevant values
            
        Args:
            Fun (function): given array of parameters (double), give back array in codomain
            dimF (int, optional): dimension codomain F. Defaults to None.
            oldXx (2D array double, optional): each row is a parameter vector. Defaults to None.
            oldSamples (_type_, optional): Each row is a sample value in corresponding parameter vector in oldXx. Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            2D array fo double: Values of Fun on the sparse grid 9each row is a value correspondin to a paramter vector)
        """    

        dimF = -1  # TODO remove dimF from signature; check all other scripts



        # Find dimF (dimension of output of F). First try oldSamples; if empty, sample on the first SG node and add to oldSamples and oldXx
        assert(self.SG.shape[0] > 0)
        if(not(oldSamples is None)):
            oldSamples = np.atleast_1d(oldSamples)  # if have 1 sample and F returns a scalar, turn it into an array
            dimF = oldSamples.shape[1]
            assert(dimF >= 1)
        else: # compute the first element to assign dimF and use it to add a line to oldXx, oldSamples
            node0 = self.SG[0]
            fOnSG0 = Fun(node0)
            fOnSG0 = np.atleast_1d(fOnSG0)  # if F returns a scalar, turn it into an array 
            dimF = fOnSG0.size
            # oldXx = np.array([node0])
            # oldSamples = np.array([fOnSG0])
        
        fOnSG = np.zeros((self.numNodes, dimF))  # the return array

        # Sanity checks dimensions oldXx, oldSamples
        if((oldXx is None) or (oldSamples is None)):
            oldXx = np.zeros((0, self.N))
            oldSamples = np.zeros((0, dimF))
        assert(oldXx.shape[0] == oldSamples.shape[0])
        assert(oldSamples.shape[1] == dimF)
        
        # If current parameter space has larger dimension that oldXx, embed oldXx in space of self.SG by extending by 0
        if(oldXx.shape[1] < self.SG.shape[1]):
            filler = np.zeros((oldXx.shape[0], self.SG.shape[1]-oldXx.shape[1]))
            oldXx = np.hstack((oldXx, filler))

        # If current parameter space is smaller than that of oldXx, modify oldXx and oldSamples to keep only samples ending in 0s. THen purge extra components in oldXx
        if(oldXx.shape[1] > self.SG.shape[1]):
            currDim = self.N
            tailNorm = np.linalg.norm(oldXx[:, currDim::], ord=1, axis=1).astype(int)
            validEntries = np.where(tailNorm == 0)[0]  # last [0] so the result is np array
            oldXx = oldXx[validEntries, 0:self.N]
            oldSamples = oldSamples[validEntries]

        # go thorugh the SG nodes where we want to compute samples. Use oldSamples of mark samples to compute now    
        nRecycle = 0
        toCompute = []
        yyToCompute = []
        for n in range(self.numNodes):
            currNode = self.SG[n,:]
            check = np.where(np.linalg.norm(oldXx-currNode, 1, axis=1) < 1.e-10)[0]
            if(check.size>1):
                print(check)
            assert(check.size<=1)
            found = len(check)
            if found:
                fOnSG[n,:] = oldSamples[check[0], :]
                nRecycle += 1
            else:
                toCompute.append(n)
                yyToCompute.append(currNode)
        
        # compute (possibily in parallel) remaining nodes
        if(len(toCompute)>0):
            if(self.NParallel == 1):
                for i in range(len(toCompute)):
                    fOnSG[toCompute[i], :] = Fun(yyToCompute[i])
            elif(self.NParallel > 1):
                pool = Pool(self.NParallel)
                tmp = np.array(pool.map(Fun, yyToCompute))
                if(len(tmp.shape) == 1):
                    tmp = tmp.reshape((-1,1))
                fOnSG[toCompute, :] = tmp
            else:
                raise ValueError('self.NParallel not int >= 1"')
        if self.verbose:
            print("Recycled", nRecycle, "; Discarted", oldXx.shape[0]-nRecycle, "; Sampled", self.SG.shape[0]-nRecycle)
        return fOnSG

    def interpolate(self, xNew, fOnSG):
        out = np.zeros((xNew.shape[0], fOnSG.shape[1]))
        for n, MId in enumerate(self.activeMIds):
            currentActiveNodesTuple = self.activeTPNodesList[n]
            currentActiveDims = self.activeTPDims[n]
            mapCurrTPtoSG = self.mapTPtoSG[n]
            fOnCurrentTPGrid = fOnSG[mapCurrTPtoSG, :]  # output is a matrix of shape = shape(mapCurrTPtoSG) + (dimF,)
            L = TPInterpolatorWrapper(currentActiveNodesTuple, currentActiveDims, fOnCurrentTPGrid, self.TPInterpolant)
            out = out + self.combinationCoeffs[n] * L(xNew)
        return np.squeeze(out)  # remove dimensions of length 1