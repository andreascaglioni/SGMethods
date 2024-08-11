

class MLInterpolant:
    """Class for multi level sparse grid interpolant like in 
    [Teckentrup, Jantsch, Webster, Gunzburger (2015)].
    """

    def __init__(self, SLIL):
        """Store Single level interpolants list (that behave li SGInterpolant 
        class)

        Args:
            SLIL (list of SGInterpolant): List of sparse grid interpolants with
            appropriate multi-index sets
        """

        self.nLevels = len(SLIL)
        self.SLIL = SLIL

    def sample(self, FApprox):
        """Sample the MLInterpolant

        Args:
            FApprox (function): Function with 2 inputs: a parameter (1D double 
            array of ANY length), and a level k (int >=0) that gives 
            approximation accuracy level

        Returns:
            list of 2D arrays double: List of length self.nLevels. Each list 
            element is a 2D array; each row is a sample from FApprox for a 
            parameter in the corresponding sparse grid; 
        """

        # TODO: FApprox is only allowed to return values that are 1D arrays of
        #       FIXED length, indepednet of the level
        # TODO: Assert that FAppeox return onyly arrays of fixed length;
        MLSamplesF = []
        for k in range(self.nLevels):
            def FApproxCurrLevel(y): return FApprox(y, k)
            samplesCurrentLevel = self.SLIL[self.nLevels -
                                            1-k].sampleOnSG(FApproxCurrLevel)
            MLSamplesF.append(samplesCurrentLevel)
        return MLSamplesF

    def interpolate(self, yy, MLSamplesF):
        """USe the interpolation oeprator to appeoximate the function F of which
         ML samples are given

        Args:
            yy (2D array double): each row is a parameter and can have any 
            length; number of rows also arbitrary
            MLSamplesF (_type_): _description_

        Returns:
            2D array double: each row is the approximation of the function F in 
            a parameter yy[i,:]
        """

        InterpolantOnYY = self.SLIL[0].interpolate(
            yy, MLSamplesF[self.nLevels-1])  # first must be handeld out of loop
        for k in range(1, self.nLevels):
            samplesCurr = MLSamplesF[self.nLevels-1-k]
            def FMock(y): return 1/0  # TODO write something more decent
            samplesCurrReduced = self.SLIL[k-1].sampleOnSG(
                FMock, oldXx=self.SLIL[k].SG, oldSamples=samplesCurr)
            InterpolantOnYY += self.SLIL[k].interpolate(yy, samplesCurr) - \
                self.SLIL[k-1].interpolate(yy, samplesCurrReduced)
        return InterpolantOnYY

    def getMLTerms(self, yy, MLSamplesF):
        """Get terms of multi-level expansion split based on FE spaces

        Args:
            yy (2D array double): each row is a parameter vector toe valuate
            MLSamplesF (list of 2D arrays double): k-th entry is 2D array with 
            shape nY x kth physical space size representing (I_{K-k}}-I_{K-k-1}[u_k])

        Returns:
            list of 2D arrays: k-th term is 2D array; each row corresponds to a
            parameter in yy; each colum gives the finite element corrdinates
        """

        MLTerms = []
        for k in range(self.nLevels):
            currSGLevel = self.nLevels-1-k
            currVals = self.SLIL[currSGLevel].interpolate(yy, MLSamplesF[k])
            # NBB the last FE level coresponds to 1 interpolant since I_{-1}=0
            if (k < self.nLevels-1):
                def FMock(x): return 0.
                MLSamplesReduceds = self.SLIL[currSGLevel-1].sampleOnSG(
                    FMock, oldXx=self.SLIL[currSGLevel].SG,
                    oldSamples=MLSamplesF[k])
                currVals -= self.SLIL[currSGLevel -
                                      1].interpolate(yy, MLSamplesReduceds)
            MLTerms.append(currVals)
        return MLTerms

    def totalCost(self, costKK):
        """Compute cost of computing this ML appeoximation

        Args:
            costKK (1D array of double): kth entry is the cost of computing 1 FE
            sample at level k

        Returns:
            double: Total cost based on number of SG nodes, level, FE samples
        """

        assert (costKK.size == self.nLevels)
        totalCostCurr = 0
        for k in range(self.nLevels):
            totalCostCurr += self.SLIL[k].numNodes * (costKK[self.nLevels-k-1])
        return totalCostCurr
