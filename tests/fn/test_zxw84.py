"""Tests for morie.fn.zxw84 -- WGS84 to local tangent plane"""

import numpy as np
import pytest

from morie.fn.zxw84 import wgs84_to_local


class TestWgs84ToLocal:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = wgs84_to_local(data)
        assert result.value is not None

    def test_output_type(self):
        result = wgs84_to_local(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
