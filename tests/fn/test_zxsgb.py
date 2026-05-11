"""Tests for morie.fn.zxsgb -- Spatial gradient boosting"""

import numpy as np
import pytest

from morie.fn.zxsgb import spatial_gbm


class TestSpatialGbm:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = spatial_gbm(data)
        assert result.value is not None

    def test_output_type(self):
        result = spatial_gbm(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
