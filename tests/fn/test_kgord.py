"""Tests for morie.fn.kgord -- Ordinary kriging prediction"""

import numpy as np
import pytest

from morie.fn.kgord import ordinary_kriging


class TestOrdinaryKriging:
    def test_basic(self):
        vals = np.array([1.0, 2.0, 3.0, 2.5, 1.5])
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = ordinary_kriging(vals, x)
        assert result.statistic is not None

    def test_output_type(self):
        result = ordinary_kriging(np.array([1.,2.,3.]), np.array([0.,1.,2.]))
        assert hasattr(result, "statistic")
