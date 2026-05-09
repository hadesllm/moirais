"""Tests for moirais.fn.kgsmp -- Simple kriging prediction"""

import numpy as np
import pytest

from moirais.fn.kgsmp import simple_kriging


class TestSimpleKriging:
    def test_basic(self):
        vals = np.array([1.0, 2.0, 3.0, 2.5, 1.5])
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = simple_kriging(vals, x)
        assert result.statistic is not None

    def test_output_type(self):
        result = simple_kriging(np.array([1.,2.,3.]), np.array([0.,1.,2.]))
        assert hasattr(result, "statistic")
