"""
This module provides functions to generate tensor product nodes.

Functions:
    TPKnots: Generate tensor product nodes from scalar knots.
"""

def tp_knots(scalar_knots, num_knots_dir):
    """Generate tensor product nodes given 1D nodes family and number of nodes
    in each direction.

    Args:
        scalar_knots (Callable[[int], numpy.ndarray[float]]): scalar_knots(n) 
            generates set pf n 1D nodes 
        num_knots_dir (numpy.ndarray[int]): Number of nodes in each direction

    Returns:
        tuple[numpy.ndarray[float]]: Tensor product nodes along each direction.
    """

    kk = ()
    for i in range(len(num_knots_dir)):
        kk += (scalar_knots(num_knots_dir[i]),)

    # TODO consider changing to
    # kk = tuple(list(map(scalar_knots, num_knots_dir)))

    return kk