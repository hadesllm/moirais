"""Tests for morie.fn.xrspz -- Spatial zero-inflated Poisson"""

import numpy as np
import pytest

from morie.fn.xrspz import spatial_zip


class TestSpatialZip:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = spatial_zip(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = spatial_zip(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
