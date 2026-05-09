"""Tests for moirais.fn.zsnni -- Natural neighbor interpolation"""

import numpy as np
import pytest

from moirais.fn.zsnni import natural_neighbor


class TestNaturalNeighbor:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = natural_neighbor(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = natural_neighbor(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
