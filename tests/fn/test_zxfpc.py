"""Tests for morie.fn.zxfpc -- Spatial functional PCA"""

import numpy as np
import pytest

from morie.fn.zxfpc import fpca_spatial


class TestFpcaSpatial:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = fpca_spatial(data)
        assert result.value is not None

    def test_output_type(self):
        result = fpca_spatial(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
