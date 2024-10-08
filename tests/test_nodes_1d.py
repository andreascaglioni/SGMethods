import numpy as np
from sgmethods.nodes_1d import equispaced_nodes, equispaced_interior_nodes, cc_nodes, hermite_nodes, optimal_gaussian_nodes, unbounded_nodes_nested
import pytest


@pytest.mark.parametrize("nNodes, nodes", [
    (1, 0.),
    (5, [-1., -0.5, 0., 0.5, 1.])
])
def test_equispacedNodes(nNodes, nodes):
    assert np.all(np.isclose(equispaced_nodes(nNodes), nodes, rtol=1.e-4))

@pytest.mark.parametrize("nNodesInt, nodesInt", [
    (1, 0.),
    (5, [-0.66666, -0.3333, 0., 0.3333, 0.6666])
])
def test_equispacedInteriorNodes(nNodesInt, nodesInt):
    assert np.all(np.isclose(equispaced_interior_nodes(nNodesInt), nodesInt, 
                             rtol=1.e-4))

@pytest.mark.parametrize("nNodesInt, nodesInt", [
    (1, 0.),
    (5, [-1., -0.7071, 0., 0.7071, 1.])
])
def test_CCNodes(nNodesInt, nodesInt):
    assert np.all(np.isclose(cc_nodes(nNodesInt), nodesInt, rtol=1.e-4))

@pytest.mark.parametrize("nNodesInt, nodesInt", [
    (1, 0.),
    (5, [-2.02018287, -9.58572465*1.e-1, 0., 9.58572465*1.e-1, 2.02018287])     
])
def test_HermiteNodes(nNodesInt, nodesInt):
    assert np.all(np.isclose(hermite_nodes(nNodesInt), nodesInt, rtol=1.e-4))

@pytest.mark.parametrize("nNodesInt, nodesInt", [
    (1, 0.),
    (5, [-2.16322038, -0.96313552, 0., 0.96313552,   2.16322038])     
])
def test_unboundedNodesOptimal(nNodesInt, nodesInt):
    assert np.all(np.isclose(optimal_gaussian_nodes(nNodesInt), nodesInt, 
                             rtol=1.e-4))

def test_unboundedNodesOptimal():
    with pytest.raises(AssertionError):
        optimal_gaussian_nodes(6)   