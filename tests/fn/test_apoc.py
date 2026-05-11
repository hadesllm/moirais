"""Tests for morie.fn.apoc -- graph articulation points."""

import numpy as np
from morie.fn.apoc import articulation_points, apoc
from morie.fn._containers import DescriptiveResult


class TestApoc:
    def test_alias(self):
        assert apoc is articulation_points

    def test_bridge_graph(self):
        adj = np.array([
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
        ])
        result = articulation_points(adj)
        assert isinstance(result, DescriptiveResult)
        assert result.value >= 2

    def test_complete_graph(self):
        adj = np.ones((4, 4)) - np.eye(4)
        result = articulation_points(adj.astype(int))
        assert result.value == 0
