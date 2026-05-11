"""Tests for morie.fn.xrspc -- Spatial Poisson count model"""

import numpy as np
import pytest

from morie.fn.xrspc import spatial_poisson


class TestSpatialPoisson:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = spatial_poisson(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = spatial_poisson(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
