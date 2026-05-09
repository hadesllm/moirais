"""Tests for moirais.fn.zxsbg -- Spatial bagging"""

import numpy as np
import pytest

from moirais.fn.zxsbg import spatial_bagging


class TestSpatialBagging:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = spatial_bagging(data)
        assert result.value is not None

    def test_output_type(self):
        result = spatial_bagging(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
