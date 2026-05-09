"""Tests for moirais.fn.zxhrs -- Hierarchical spatial (nested)"""

import numpy as np
import pytest

from moirais.fn.zxhrs import hier_spatial_fe


class TestHierSpatialFe:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = hier_spatial_fe(data)
        assert result.value is not None

    def test_output_type(self):
        result = hier_spatial_fe(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
