"""Tests for morie.fn.kgfct -- Factorial kriging"""

import numpy as np
import pytest

from morie.fn.kgfct import factorial_kriging


class TestFactorialKriging:
    def test_basic(self):
        vals = np.array([1.0, 2.0, 3.0, 2.5, 1.5])
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = factorial_kriging(vals, x)
        assert result.statistic is not None

    def test_output_type(self):
        result = factorial_kriging(np.array([1.,2.,3.]), np.array([0.,1.,2.]))
        assert hasattr(result, "statistic")
