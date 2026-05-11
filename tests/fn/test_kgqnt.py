"""Tests for morie.fn.kgqnt -- Kriging quantile prediction"""

import numpy as np
import pytest

from morie.fn.kgqnt import kriging_quantile


class TestKrigingQuantile:
    def test_basic(self):
        vals = np.array([1.0, 2.0, 3.0, 2.5, 1.5])
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = kriging_quantile(vals, x)
        assert result.statistic is not None

    def test_output_type(self):
        result = kriging_quantile(np.array([1.,2.,3.]), np.array([0.,1.,2.]))
        assert hasattr(result, "statistic")
