"""Tests for morie.fn.arrow -- Directed graph layout."""

import numpy as np
from morie.fn.arrow import directed_layout, arrow
from morie.fn._containers import DescriptiveResult


class TestArrow:
    def test_alias(self):
        assert arrow is directed_layout

    def test_circular(self):
        adj = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        result = directed_layout(adj, method="circular")
        assert isinstance(result, DescriptiveResult)
        assert len(result.value["positions"]) == 3
        assert len(result.value["edges"]) == 3

    def test_force_directed(self):
        adj = np.array([[0, 1, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]])
        result = directed_layout(adj, method="force", seed=42, n_iter=50)
        assert result.extra["n_nodes"] == 4
