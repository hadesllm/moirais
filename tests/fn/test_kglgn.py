"""Tests for morie.fn.kglgn -- Lognormal kriging"""

import numpy as np
import pytest

from morie.fn.kglgn import lognormal_kriging


class TestLognormalKriging:
    def test_basic(self):
        vals = np.array([1.0, 2.0, 3.0, 2.5, 1.5])
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = lognormal_kriging(vals, x)
        assert result.statistic is not None

    def test_output_type(self):
        result = lognormal_kriging(np.array([1.,2.,3.]), np.array([0.,1.,2.]))
        assert hasattr(result, "statistic")
