"""Tests for morie.fn.vgrng -- Effective range estimation"""

import numpy as np
import pytest

from morie.fn.vgrng import range_est


class TestRangeEst:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = range_est(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = range_est(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
