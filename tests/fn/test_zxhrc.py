"""Tests for morie.fn.zxhrc -- Hierarchical spatial (crossed)"""

import numpy as np
import pytest

from morie.fn.zxhrc import hier_spatial_cross


class TestHierSpatialCross:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = hier_spatial_cross(data)
        assert result.value is not None

    def test_output_type(self):
        result = hier_spatial_cross(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
