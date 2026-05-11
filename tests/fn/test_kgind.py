"""Tests for morie.fn.kgind -- Indicator kriging"""

import numpy as np
import pytest

from morie.fn.kgind import indicator_kriging


class TestIndicatorKriging:
    def test_basic(self):
        vals = np.array([1.0, 2.0, 3.0, 2.5, 1.5])
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = indicator_kriging(vals, x)
        assert result.statistic is not None

    def test_output_type(self):
        result = indicator_kriging(np.array([1.,2.,3.]), np.array([0.,1.,2.]))
        assert hasattr(result, "statistic")
