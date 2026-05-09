"""Tests for moirais.fn.zxsen -- Spatial stacking ensemble"""

import numpy as np
import pytest

from moirais.fn.zxsen import spatial_stacking


class TestSpatialStacking:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = spatial_stacking(data)
        assert result.value is not None

    def test_output_type(self):
        result = spatial_stacking(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
