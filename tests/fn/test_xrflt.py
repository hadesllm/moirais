"""Tests for morie.fn.xrflt -- Eigenvector spatial filtering"""

import numpy as np
import pytest

from morie.fn.xrflt import spatial_filter


class TestSpatialFilter:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = spatial_filter(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = spatial_filter(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
