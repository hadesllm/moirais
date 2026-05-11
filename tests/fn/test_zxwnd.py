"""Tests for morie.fn.zxwnd -- Wind rose directional stats"""

import numpy as np
import pytest

from morie.fn.zxwnd import wind_rose


class TestWindRose:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = wind_rose(data)
        assert result.value is not None

    def test_output_type(self):
        result = wind_rose(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
