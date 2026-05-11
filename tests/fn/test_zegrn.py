"""Tests for morie.fn.zegrn -- Spatial gradient estimation"""

import numpy as np
import pytest

from morie.fn.zegrn import gradient_spatial


class TestGradientSpatial:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = gradient_spatial(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = gradient_spatial(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
