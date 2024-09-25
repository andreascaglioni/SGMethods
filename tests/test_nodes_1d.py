import numpy as np
from src.nodes_1d import equispacedNodes, equispacedInteriorNodes, CCNodes, HermiteNodes, unboundedNodesOptimal, unboundedKnotsNested
import pytest


@pytest.mark.parametrize("nNodes, nodes", [
    (1, 0.),
    (5, [-1., -0.5, 0., 0.5, 1.])
])
def test_equispacedNodes(nNodes, nodes):
    assert np.all(np.isclose(equispacedNodes(nNodes), nodes, rtol=1.e-4))

@pytest.mark.parametrize("nNodesInt, nodesInt", [
    (1, 0.),
    (5, [-0.66666, -0.3333, 0., 0.3333, 0.6666])
])
def test_equispacedInteriorNodes(nNodesInt, nodesInt):
    assert np.all(np.isclose(equispacedInteriorNodes(nNodesInt), nodesInt, rtol=1.e-4))

@pytest.mark.parametrize("nNodesInt, nodesInt", [
    (1, 0.),
    (5, [-1., -0.7071, 0., 0.7071, 1.])
])
def test_CCNodes(nNodesInt, nodesInt):
    assert np.all(np.isclose(CCNodes(nNodesInt), nodesInt, rtol=1.e-4))

@pytest.mark.parametrize("nNodesInt, nodesInt", [
    (1, 0.),
    (5, [-2.02018287, -9.58572465*1.e-1, 0., 9.58572465*1.e-1, 2.02018287])     
])
def test_HermiteNodes(nNodesInt, nodesInt):
    assert np.all(np.isclose(HermiteNodes(nNodesInt), nodesInt, rtol=1.e-4))

@pytest.mark.parametrize("nNodesInt, nodesInt", [
    (1, 0.),
    (5, [-2.16322038, -0.96313552, 0., 0.96313552,   2.16322038])     
])
def test_unboundedNodesOptimal(nNodesInt, nodesInt):
    assert np.all(np.isclose(unboundedNodesOptimal(nNodesInt), nodesInt, rtol=1.e-4))

def test_unboundedNodesOptimal():
    with pytest.raises(AssertionError):
        unboundedNodesOptimal(6)   