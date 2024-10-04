def TPKnots(ScalarKnots, numKnotsDir):
    """Generate tensor product nodes

    Args:
        ScalarKnots (function): ScalarKnots(n) generates set pf n 1D nodes 
        numKnotsDir (array int): Number of nodes in each direction

    Returns:
        tuple of array double: Nodes along each direction
    """

    kk = ()
    for i in range(len(numKnotsDir)):
        kk += (ScalarKnots(numKnotsDir[i]),)

    return kk
