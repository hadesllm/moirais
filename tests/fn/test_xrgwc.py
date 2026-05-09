"""Tests for moirais.fn.xrgwc -- GWR local coefficients"""

import numpy as np
import pytest

from moirais.fn.xrgwc import gwr_coefficients


class TestGwrCoefficients:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = gwr_coefficients(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = gwr_coefficients(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
