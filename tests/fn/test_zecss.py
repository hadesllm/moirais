"""Tests for morie.fn.zecss -- Spatial CUSUM aberration detection"""

import numpy as np
import pytest

from morie.fn.zecss import cusum_spatial


class TestCusumSpatial:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = cusum_spatial(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = cusum_spatial(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
