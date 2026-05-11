"""Tests for morie.fn.zesfr -- Spatial frailty survival model"""

import numpy as np
import pytest

from morie.fn.zesfr import spatial_frailty


class TestSpatialFrailty:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = spatial_frailty(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = spatial_frailty(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
