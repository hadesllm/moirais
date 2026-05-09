"""Tests for moirais.fn.niobi -- A* pathfinding."""

import numpy as np
from moirais.fn.niobi import astar_path, niobi
from moirais.fn._containers import DescriptiveResult


class TestNiobi:
    def test_alias(self):
        assert niobi is astar_path

    def test_open_grid(self):
        grid = np.ones((5, 5))
        result = astar_path(grid, start=(0, 0), goal=(4, 4))
        assert isinstance(result, DescriptiveResult)
        assert result.value < float("inf")
        assert result.extra["path"][0] == [0, 0] or result.extra["path"][0] == (0, 0)

    def test_blocked(self):
        grid = np.ones((3, 3))
        grid[1, :] = 0
        result = astar_path(grid, start=(0, 0), goal=(2, 2), diagonal=False)
        assert result.value == float("inf")
