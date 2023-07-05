class MLInterpolant:
    def __init__(self, SLIL):
        """INPUT SLIS (Single Level Interpolants List): a list of interpolation methods that behaves like my class SGInterpolant"""
        self.nLevels = len(SLIL)
        self.SLIL = SLIL

    def sample(self, FApprox):
        """INPUT FApprox (Function Approximation): function w 2 inputs: a parameter (1D double array of ANY length); a level k (int >=0) that gives increasing precision ...
            OUTPUT MLSamplesF: list of length self.nLevels. Each list element is a 2D array; each row is a sample from FApprox for a parameter in the corresponding sparse grid; 
                        NB MLSamplesF is sorted in order of increasing physical space size (decreasing SG size)"""
        # TODO: FApprox is only allowed to return values that are 1D arrays of FIXED length, indepednet of the level
        # assert that FAppeox return onyly arrays of fixed length;
        MLSamplesF = []
        for k in range(self.nLevels):
            FApproxCurrLevel = lambda y : FApprox(y, k)
            samplesCurrentLevel = self.SLIL[self.nLevels-1-k].sampleOnSG(FApproxCurrLevel)
            MLSamplesF.append(samplesCurrentLevel)
        return MLSamplesF
    
    def interpolate(self, yy, MLSamplesF):
        """INPUT yy: 2D double array of parameters; each ROW is a parameter and can have any length; number of rows also arbitrary
           OUTPUT InterpolantOnYY: 2D array where each ROW is the approximation of the function F in a parameter yy[i,:]"""
        InterpolantOnYY = self.SLIL[0].interpolate(yy, MLSamplesF[self.nLevels-1])  # first must be handeld out of loop
        for k in range(1, self.nLevels):
            samplesCurr = MLSamplesF[self.nLevels-1-k]
            FMock = lambda y : 1/0  # TODO write something more decent
            samplesCurrReduced = self.SLIL[k-1].sampleOnSG(FMock, oldXx=self.SLIL[k].SG, oldSamples=samplesCurr)
            InterpolantOnYY += self.SLIL[k].interpolate(yy, samplesCurr) - self.SLIL[k-1].interpolate(yy, samplesCurrReduced)
        return InterpolantOnYY

    def getMLTerms(self, yy, MLSamplesF):
        """INPUT yy: double 2D array; each row is a parameter vector toe valuate
                 MLSamplesF : output of method sample
           OUTPUT MLTerms: list of length self.nLevels; k-th entry is 2D array with shape nY x kth physical space size representing (I_{K-k}}-I_{K-k-1}[u_k])"""
        MLTerms = []
        for k in range(self.nLevels):
            currSGLevel = self.nLevels-1-k
            currVals = self.SLIL[currSGLevel].interpolate(yy, MLSamplesF[k])
            if(k<self.nLevels-1):  # NBB the last FE level coresponds to only 1 interpolant since I_{-1}=0
                FMock = lambda x : 0.
                MLSamplesReduceds = self.SLIL[currSGLevel-1].sampleOnSG(FMock, oldXx = self.SLIL[currSGLevel].SG , oldSamples = MLSamplesF[k])
                currVals -= self.SLIL[currSGLevel-1].interpolate(yy, MLSamplesReduceds)
            MLTerms.append(currVals)
        return MLTerms

    def totalCost(self, costKK):
        """INPUT costKK array 1D of length self.nLevels. The k-th entry corresponds to the cost of the sampler at precision level k"""
        assert(costKK.size == self.nLevels)
        totalCostCurr = 0
        for k in range(self.nLevels):
            totalCostCurr += self.SLIL[k].numNodes * (costKK[self.nLevels-k-1])
        return totalCostCurr