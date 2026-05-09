"""Tests for moirais.fn.zxwlt -- Spatial wavelet analysis"""

import numpy as np
import pytest

from moirais.fn.zxwlt import wavelet_spatial


class TestWaveletSpatial:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = wavelet_spatial(data)
        assert result.value is not None

    def test_output_type(self):
        result = wavelet_spatial(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
