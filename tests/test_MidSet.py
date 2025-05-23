import numpy as np
from sgmethods.mid_set import MidSet


def test_MidSet():
    # Define and enlarge a multi-index set
    Lambda = MidSet(track_reduced_margin=True)
    Lambda.enlarge_from_margin(0)
    Lambda.enlarge_from_margin(0)
    Lambda.increase_dim()
    Lambda.enlarge_from_margin(1)
    Lambda.enlarge_from_margin(1)
    Lambda.increase_dim()
    assert Lambda.get_cardinality_mid_set() == 8
    assert Lambda.get_dim() == 3
    assert Lambda.get_cardinality_margin() == 12
    assert Lambda.get_cardinality_reduced_margin() == 6
    assert np.all(Lambda.mid_set == np.array([[0, 0, 0],
                                    [0, 0, 1],
                                    [0, 1, 0],
                                    [0, 2, 0],
                                    [1, 0, 0],
                                    [1, 1, 0],
                                    [1, 2, 0],
                                    [2, 0, 0]]
                                    ))
    assert np.all(Lambda.margin ==np.array([[0, 0, 2],
                                            [0, 1, 1],
                                            [0, 2, 1],
                                            [0, 3, 0],
                                            [1, 0, 1],
                                            [1, 1, 1],
                                            [1, 2, 1],
                                            [1, 3, 0],
                                            [2, 0, 1],
                                            [2, 1, 0],
                                            [2, 2, 0],
                                            [3, 0, 0]]
                                            ))
    assert np.all(Lambda.reduced_margin ==np.array([[0, 0, 2],
                                            [0, 1, 1],
                                            [0, 3, 0],
                                            [1, 0, 1],
                                            [2, 1, 0],
                                            [3, 0, 0]]
                                            ))
    