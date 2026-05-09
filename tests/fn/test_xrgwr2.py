"""Tests for moirais.fn.xrgwr2 -- GWR local R-squared"""

import numpy as np
import pytest

from moirais.fn.xrgwr2 import gwr_rsquared


class TestGwrRsquared:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = gwr_rsquared(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = gwr_rsquared(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
