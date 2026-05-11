"""Tests for morie.fn.zescr -- Spatial cure rate model"""

import numpy as np
import pytest

from morie.fn.zescr import spatial_cure_rate


class TestSpatialCureRate:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = spatial_cure_rate(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = spatial_cure_rate(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
