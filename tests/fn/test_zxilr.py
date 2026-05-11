"""Tests for morie.fn.zxilr -- Isometric log-ratio spatial"""

import numpy as np
import pytest

from morie.fn.zxilr import ilr_spatial


class TestIlrSpatial:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ilr_spatial(data)
        assert result.value is not None

    def test_output_type(self):
        result = ilr_spatial(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
