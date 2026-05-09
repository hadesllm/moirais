"""Tests for moirais.fn.zsgrz -- Zonal grid statistics"""

import numpy as np
import pytest

from moirais.fn.zsgrz import grid_zonal


class TestGridZonal:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = grid_zonal(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = grid_zonal(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
