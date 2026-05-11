"""Tests for morie.fn.zedbs -- Spatial DBSCAN cluster detection"""

import numpy as np
import pytest

from morie.fn.zedbs import spatial_dbscan


class TestSpatialDbscan:
    def test_basic(self):
        observed = np.array([5, 3, 8, 2, 10])
        result = spatial_dbscan(observed)
        assert result.statistic is not None

    def test_output_type(self):
        result = spatial_dbscan(np.array([1,2,3,4,5]))
        assert hasattr(result, "statistic")
