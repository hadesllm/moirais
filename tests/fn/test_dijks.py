"""Tests for dijks (Dijkstra shortest paths)."""
import numpy as np
from morie.fn.dijks import dijkstra


def test_dijkstra_basic():
    adj = np.array([
        [0, 1, 4, 0],
        [0, 0, 2, 6],
        [0, 0, 0, 3],
        [0, 0, 0, 0],
    ], dtype=float)
    r = dijkstra(adj, source=0)
    dists = r.extra["distances"]
    assert dists[0] == 0
    assert dists[1] == 1
    assert dists[2] == 3
    assert dists[3] == 6


def test_cheatsheet():
    from morie.fn.dijks import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
