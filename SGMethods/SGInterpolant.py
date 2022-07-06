import numpy as np
from SGMethods.MidSets import TPMidSet
from SGMethods.TPKnots import TPKnots
from SGMethods.TPInterpolatorWrapper import TPInterpolatorWrapper


class SGInterpolant:
    def __init__(self, midSet, knots, lev2knots, pieceWise=False):
        self.midSet = midSet  # np array of shape (#mids, N)
        self.cardMidSet = midSet.shape[0]
        self.N = midSet.shape[1]
        self.knots = knots
        self.lev2knots = lev2knots
        self.pieceWise=pieceWise
        
        self.combinationCoeffs = []  # list of int
        self.activeMIds = []  # list of np arrays
        self.TPNodesList = []  # list of tuples
        self.TPGrids = []  # list of np arrays of shape (# TP nodes x N)
        self.mapTPtoSG = []  # list of np arrays of shape ()
        self.setupInterpolant()

        self.SG = []  # np array of shape (#colloc. pts, N)
        self.numNodes = 0
        self.setupSG()

    def setupInterpolant(self):
        jVec = TPMidSet(1, self.N)  # just a shortcut to list all increments in {0,1}^N as rows of a matrix
        for n in range(self.cardMidSet):
            currentMid = self.midSet[n, :]
            combinCoeff = 0
            for nj in range(jVec.shape[0]):
                j = jVec[nj, :]
                v = currentMid + j
                if v.tolist() in self.midSet.tolist():
                    combinCoeff += int(pow(-1, np.linalg.norm(j, 1)))
            if combinCoeff != 0:
                self.combinationCoeffs.append(combinCoeff)
                self.activeMIds.append(currentMid)
                numNodesDir = self.lev2knots(currentMid).astype(int)
                CurrentNodesList, currentTPGrid = TPKnots(self.knots, numNodesDir)
                self.TPNodesList.append(CurrentNodesList)
                self.TPGrids.append(currentTPGrid)
                shp = np.ndarray(tuple(numNodesDir), dtype=int)
                self.mapTPtoSG.append(shp)

    def setupSG(self):
        SG = np.array([]).reshape((0, self.N))
        for n, currTPNodesList in enumerate(self.TPNodesList):
            meshGrid = np.meshgrid(*currTPNodesList, indexing='ij')  # NBB in python * is "unpacking" operator (gives back comma separated list)
            it = np.nditer(meshGrid[0], flags=['multi_index'])
            for x in it:
                currNode = [meshGrid[j][it.multi_index] for j in range(self.N)]
                check = np.where(np.linalg.norm(SG-currNode, 1, axis=1) < 1.e-10)[0]  # check if current node is already in SG
                found = check.shape[0]
                assert(found <= 1)
                if found:  # if found, add the index to mapTPtoSG[n]
                    self.mapTPtoSG[n][it.multi_index] = check[0]
                else:  # if not found, add it to sparse grid and add new index to mapTPtoSG
                    SG = np.vstack((SG, np.array(currNode)))
                    self.mapTPtoSG[n][it.multi_index] = SG.shape[0]-1
        self.SG = SG
        self.numNodes = SG.shape[0]

    def sampleOnSG(self, Fun, dimF, oldXx = None , oldSamples = None):
        """NBB assume F 
        takes as input an np array of parameters 
        and the 1st output are the relevant values"""
        # sanity checks
        assert(dimF >= 1)
        if(oldSamples is None):
            oldXx = np.zeros((0, self.N))
            oldSamples = np.zeros((0, dimF))
        assert(oldXx.shape[1] == self.N)
        assert(oldSamples.shape[1] == dimF)
        assert(oldXx.shape[0] == oldSamples.shape[0])

        nRecycle = 0
        fOnSG = np.zeros((self.numNodes, dimF))
        for n in range(self.numNodes):
            currNode = self.SG[n,:]
            check = np.where(np.linalg.norm(oldXx-currNode, 1, axis=1) < 1.e-10)[0]
            assert(check.size<=1)
            found = len(check)
            if found:
                fOnSG[n,:] = oldSamples[check[0], :]
                nRecycle += 1
            else:
                A = Fun(currNode)
                fOnSG[n,:] = A
        print("Recycled", nRecycle, "; Discarted", oldXx.shape[0]-nRecycle, "; Sampled", self.SG.shape[0]-nRecycle)
        return fOnSG

    def interpolate(self, xNew, fOnSG):
        out = np.zeros((xNew.shape[0], fOnSG.shape[1]))
        for n, MId in enumerate(self.activeMIds):
            currentNodesTuple = self.TPNodesList[n]
            mapCurrTPtoSG = self.mapTPtoSG[n]
            fOnCurrentTPGrid = fOnSG[mapCurrTPtoSG, :]  # this will produce a matrix of shape = mapCurrTPtoSG + (dimF,)
            L = TPInterpolatorWrapper(currentNodesTuple, fOnCurrentTPGrid, pieceWise = self.pieceWise)
            out = out + self.combinationCoeffs[n] * L(xNew)
        return out