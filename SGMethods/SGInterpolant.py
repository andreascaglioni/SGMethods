import numpy as np
from SGMethods.MidSets import TPMidSet
from SGMethods.TPKnots import TPKnots
from SGMethods.TPInterpolatorWrapper import TPInterpolatorWrapper
from multiprocessing import Pool


class SGInterpolant:
    def __init__(self, midSet, knots, lev2knots, interpolationType="linear", NParallel = 1):
        self.midSet = midSet  # np array of shape (#mids, N)
        self.cardMidSet = midSet.shape[0]
        self.N = midSet.shape[1]
        self.knots = knots #  NBB need knots[1] = 0 to increase number of dimensions
        self.lev2knots = lev2knots #  NBB need lev2knots(0)=1 to increase number of dimensions
        self.interpolationType=interpolationType
        
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
        """assign combinCoeff, activeMIds, activeTPDims, activeTPNodesList, mapTPtoSG 
        based on midSet, knots, lev2knots"""
        bookmarks = np.unique(self.midSet[:,0], return_index=True)[1]
        bk = np.hstack((bookmarks[2:], np.array([self.cardMidSet, self.cardMidSet])))
        for n in range(self.cardMidSet):
            currentMid = self.midSet[n, :]
            combinCoeff = 1
            rangeIds = bk[currentMid[0]]  # index of the 1st mid with 1st components = currentMid[0]+2 (dont need to itertate over it or any following one)
            
            # for j in range(n+1, rangeIds):  #  in operator range the final bound is NOT included! we begin from n+1 and already considered the case n in definition combiCoeff
            #     d = self.midSet[j, :] - currentMid
            #     if(np.max(d)<=1 and np.min(d)>=0):
            #         combinCoeff += int(pow(-1, np.linalg.norm(d, 1)))
            
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
        """assign SG (sparse grid), numNodes, mapTPtoSG based on self.activeTPNodesList"""
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
                check = np.where(np.linalg.norm(SG-currNode, 1, axis=1) < 1.e-10)[0] #  check if current node is already in SG (NBB SG is always fully dimensional, also if some components are 0.)
                found = check.shape[0]
                assert(found <= 1)  # if currNode was in the SG more than once, somewthing woulod be very wrong
                if found:  # if found, add the index to mapTPtoSG[n]
                    self.mapTPtoSG[n][it.multi_index] = check[0]
                else:  # if not found, add it to sparse grid and add new index to mapTPtoSG
                    SG = np.vstack((SG, np.array(currNode)))
                    self.mapTPtoSG[n][it.multi_index] = SG.shape[0]-1
        self.SG = SG
        self.numNodes = SG.shape[0]

    def sampleOnSG(self, Fun, dimF = None, oldXx = None , oldSamples = None):
        """ First check if there is anything to recycle, then sample new values
        NBB assume F takes as input an np array of parameters and the 1st output are the relevant values"""

        dimF = -1  # TODO remove dimF from signature; check all other scripts

        # Find dimF (dimensoin of outoput of F). First try oldSamples; if empy, sample on the first SG node
        assert(self.SG.shape[0] > 0)
        if(not(oldSamples is None)):
            dimF = oldSamples.shape[1]
            assert(dimF >= 1)
        else: # compute the first element to assign dimF
            node0 = self.SG[0]
            fOnSG0 = Fun(node0)
            dimF = fOnSG0.size
        fOnSG = np.zeros((self.numNodes, dimF))  # the return array
        # reformat oldSamples and oldXx even if they are empty for smooth processing; Sanity checks
        if(oldSamples is None):
            oldXx = np.zeros((0, self.N))
            oldSamples = np.zeros((0, dimF))
        assert(oldXx.shape[0] == oldSamples.shape[0])
        assert(oldSamples.shape[1] == dimF)
        # assert(oldXx.shape[1] <= self.N)  # the parameter space may have gotten larger (smaller: not yet implemented)
        # If current parameter space has larger dimension that oldXx, embed oldXx in space of self.SG by extending by 0
        if(oldXx.shape[1] < self.SG.shape[1]):
            filler = np.zeros((oldXx.shape[0], self.SG.shape[1]-oldXx.shape[1]))
            oldXx = np.hstack((oldXx, filler))
        # If current parameter space is smaller than that of oldXx, modify oldXx and oldSamples to keep only samples ending in 0s. THen purge extra components in oldXx
        currDim = self.N
        tailNorm = np.linalg.norm(oldXx[:, currDim::], ord=1, axis=1).astype(int)
        validEntries = np.where(tailNorm == 0)[0]  # last [0] so theat result is np array
        oldXx = oldXx[validEntries, 0:self.N]
        oldSample = oldSamples[validEntries]

        # go thorugh the SG nodes where we want to compute samples. Use oldSamples of mark samples to compute now    
        nRecycle = 0
        toCompute = []
        yyToCompute = []
        for n in range(self.numNodes):
            currNode = self.SG[n,:]
            check = np.where(np.linalg.norm(oldXx-currNode, 1, axis=1) < 1.e-10)[0]
            assert(check.size<=1)
            found = len(check)
            if found:
                fOnSG[n,:] = oldSamples[check[0], :]
                nRecycle += 1
            else:
                toCompute.append(n)
                yyToCompute.append(currNode)
        
        # compute (possibily in parallel) remaining nodes
        if(not(len(toCompute)==0)):
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
        print("Recycled", nRecycle, "; Discarted", oldXx.shape[0]-nRecycle, "; Sampled", self.SG.shape[0]-nRecycle)
        return fOnSG

    def interpolate(self, xNew, fOnSG):
        out = np.zeros((xNew.shape[0], fOnSG.shape[1]))
        for n, MId in enumerate(self.activeMIds):
            currentActiveNodesTuple = self.activeTPNodesList[n]
            currentActiveDims = self.activeTPDims[n]
            mapCurrTPtoSG = self.mapTPtoSG[n]
            fOnCurrentTPGrid = fOnSG[mapCurrTPtoSG, :]  # output is a matrix of shape = shape(mapCurrTPtoSG) + (dimF,)
            L = TPInterpolatorWrapper(currentActiveNodesTuple, currentActiveDims, fOnCurrentTPGrid, self.interpolationType)
            out = out + self.combinationCoeffs[n] * L(xNew)
        return out