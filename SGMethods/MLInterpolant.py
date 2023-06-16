class MLInterpolant:
    def __init__(self, SLIL, dimF):
        """INPUT SLIS (Single Level Interpolants List): a list of interpolation methods that behaves like my class SGInterpolant"""
        self.nLevels = len(SLIL)
        self.SLIL = SLIL
        self.dimF = dimF

    def sample(self, FApprox):
        """INPUT FApprox (Function Approximation): function w 2 inputs: a parameter (1D double array of ANY length); a level k (double >=0) that gives increasing precision ...
            OUTPUT MLSamplesF: list of length self.nLevels. Each list element is a 2D array; each row is a sample from FApprox for a parameter in the corresponding sparse grid"""
        # TODO: FApprox is only allowed to return values that are 1D arrays of FIXED length, indepednet of the level
        # assert that FAppeox return onyly arrays of fixed length;
        MLSamplesF = []
        for k in range(self.nLevels):
            FApproxCurrLevel = lambda y : FApprox(y, k)
            MLSamplesF.append(self.SLIL[self.nLevels-1-k].sampleOnSG(FApproxCurrLevel, self.dimF))
        return MLSamplesF
    
    def interpolate(self, yy, MLSamplesF):
        """INPUT yy: 2D double array of parameters; each ROW is a parameter and can have any length; number of rows also arbitrary
           OUTPUT InterpolantOnYY: 2D array where each ROW is the approximation of the function F in a parameter yy[i,:]"""
        InterpolantOnYY = self.SLIL[0].interpolate(yy, MLSamplesF[self.nLevels-1])  # first must be handeld out of loop
        for k in range(1, self.nLevels):
            samplesCurr = MLSamplesF[self.nLevels-1-k]
            FMock = lambda y : 1/0  # TODO write something more decent
            samplesCurrReduced = self.SLIL[k-1].sampleOnSG(FMock, self.dimF, self.SLIL[k].SG, samplesCurr)
            InterpolantOnYY += self.SLIL[k].interpolate(yy, samplesCurr) - self.SLIL[k-1].interpolate(yy, samplesCurrReduced)
        return InterpolantOnYY