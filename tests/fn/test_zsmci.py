"""Tests for moirais.fn.zsmci -- Monte Carlo spatial integration"""

import numpy as np
import pytest

from moirais.fn.zsmci import mc_spatial_int


class TestMcSpatialInt:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = mc_spatial_int(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = mc_spatial_int(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
