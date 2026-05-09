"""Tests for moirais.fn.qween — Queen contiguity weights."""

import numpy as np
import pytest

from moirais.fn.qween import queen_weights


class TestQueenWeights:

    def test_four_cell_grid(self):
        """2x2 grid: corner cells have 1-3 neighbors under queen contiguity."""
        adj = [
            [1, 2, 3],
            [0, 2, 3],
            [0, 1, 3],
            [0, 1, 2],
        ]
        result = queen_weights(adj)
        W = result.value
        assert W.shape == (4, 4)
        for i in range(4):
            assert W[i].sum() == pytest.approx(3.0)

    def test_diagonal_is_zero(self):
        """Self-links are excluded even if listed in adjacency."""
        adj = [[1, 0], [0, 1]]
        result = queen_weights(adj)
        W = result.value
        assert W[0, 0] == 0.0
        assert W[1, 1] == 0.0

    def test_out_of_range_raises(self):
        """Out-of-range neighbor index raises ValueError."""
        adj = [[1], [5]]
        with pytest.raises(ValueError, match="out of range"):
            queen_weights(adj)
